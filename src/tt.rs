//! Transposition table: one entry per slot, replacement by age and depth, thread-safe without `unsafe`.
//!
//! A slot consists of two `AtomicU64`: `data` (packed entry) and `check = key ^ data`.
//! On reading, an entry is only valid if `check ^ data` yields the requested key. If two threads
//! write the same slot at the same time, a mixed pair can arise; it fails this check with very high
//! probability (probabilistic protection as in common Lazy SMP engines). Anything that still slips
//! through is harmless: the move is checked against the legal moves before use (`Searcher::find`,
//! move ordering), and the packing keeps values within fixed bounds.

use std::sync::atomic::{AtomicU64, Ordering};

/// Kind of the stored bound.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Bound {
    /// Empty slot.
    #[default]
    None,
    /// Exact score.
    Exact,
    /// Lower bound (beta cutoff).
    Lower,
    /// Upper bound (no move above alpha).
    Upper,
}

/// A table entry (unpacked).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Entry {
    /// Zobrist key of the position.
    pub key: u64,
    /// Best move (see `search::move_code`, 15 bits), 0 = none.
    pub mv: u16,
    /// Search score, mate scores normalized relative to the position (fits in `i16`).
    pub score: i32,
    /// Static evaluation (cp, ±20,000), if known.
    pub eval: Option<i32>,
    /// Remaining search depth with which `score` was computed (1–127).
    pub depth: i32,
    /// Kind of bound.
    pub bound: Bound,
    /// Search generation; set by [`Tt::store`].
    pub age: u8,
}

/// Layout of the fields in `data` (bit 0 = lowest): move 15, score 16, eval 16, depth 7,
/// bound 2, age 8 = 64 bits.
const MOVE_BITS: u32 = 15;
const SCORE_SHIFT: u32 = 15;
const EVAL_SHIFT: u32 = 31;
const DEPTH_SHIFT: u32 = 47;
const BOUND_SHIFT: u32 = 54;
const AGE_SHIFT: u32 = 56;
const DEPTH_MAX: i32 = 127;
/// `eval` without a value.
const NO_EVAL: i16 = i16::MIN;

impl Bound {
    fn code(self) -> u64 {
        match self {
            Bound::None => 0,
            Bound::Exact => 1,
            Bound::Lower => 2,
            Bound::Upper => 3,
        }
    }

    fn from_code(code: u64) -> Bound {
        match code & 3 {
            1 => Bound::Exact,
            2 => Bound::Lower,
            3 => Bound::Upper,
            _ => Bound::None,
        }
    }
}

/// `i16` as 16 bits without sign extension.
fn bits16(value: i16) -> u64 {
    u64::from(value.cast_unsigned())
}

/// Lower 16 bits starting at `shift` as `i16`.
fn field16(data: u64, shift: u32) -> i16 {
    let low = data >> shift & 0xFFFF;
    u16::try_from(low).unwrap_or(0).cast_signed()
}

impl Entry {
    /// Packs the entry without the key. Out-of-range values are reported in debug builds and
    /// clamped otherwise.
    fn pack(&self) -> u64 {
        debug_assert!(u32::from(self.mv) < 1 << MOVE_BITS, "move code {}", self.mv);
        debug_assert!((1..=DEPTH_MAX).contains(&self.depth), "depth {}", self.depth);
        debug_assert!(i16::try_from(self.score).is_ok(), "score {}", self.score);
        let score = i16::try_from(self.score).unwrap_or(if self.score > 0 { i16::MAX } else { i16::MIN + 1 });
        let eval = self.eval.map_or(NO_EVAL, |eval| {
            debug_assert!(
                eval > i32::from(NO_EVAL) && eval <= i32::from(i16::MAX),
                "eval {eval}"
            );
            i16::try_from(eval).unwrap_or(if eval > 0 { i16::MAX } else { NO_EVAL + 1 })
        });
        let depth = u64::try_from(self.depth.clamp(0, DEPTH_MAX)).unwrap_or(0);
        u64::from(self.mv) & ((1 << MOVE_BITS) - 1)
            | bits16(score) << SCORE_SHIFT
            | bits16(eval) << EVAL_SHIFT
            | depth << DEPTH_SHIFT
            | self.bound.code() << BOUND_SHIFT
            | u64::from(self.age) << AGE_SHIFT
    }

    /// Counterpart of [`Entry::pack`].
    fn unpack(key: u64, data: u64) -> Entry {
        let eval = field16(data, EVAL_SHIFT);
        Entry {
            key,
            mv: u16::try_from(data & ((1 << MOVE_BITS) - 1)).unwrap_or(0),
            score: i32::from(field16(data, SCORE_SHIFT)),
            eval: (eval != NO_EVAL).then_some(i32::from(eval)),
            depth: i32::try_from(data >> DEPTH_SHIFT & 0x7F).unwrap_or(0),
            bound: Bound::from_code(data >> BOUND_SHIFT),
            age: u8::try_from(data >> AGE_SHIFT).unwrap_or(0),
        }
    }
}

/// A slot: packed entry and `key ^ data`.
#[derive(Default)]
struct Slot {
    check: AtomicU64,
    data: AtomicU64,
}

/// Fixed-size transposition table (power of two), shareable by several search threads.
pub struct Tt {
    slots: Vec<Slot>,
}

impl Tt {
    /// Table of at most `mb` megabytes (16 bytes per slot).
    #[must_use]
    pub fn new(mb: usize) -> Tt {
        let n_slots = (mb.max(1) * 1024 * 1024 / size_of::<Slot>()).next_power_of_two() / 2;
        Tt {
            slots: (0..n_slots.max(1024)).map(|_| Slot::default()).collect(),
        }
    }

    /// Clears all entries (only while no search is running).
    pub fn clear(&self) {
        for slot in &self.slots {
            slot.check.store(0, Ordering::Relaxed);
            slot.data.store(0, Ordering::Relaxed);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "bits truncated on purpose (hash index)"
    )]
    fn idx(&self, key: u64) -> usize {
        (key as usize) & (self.slots.len() - 1)
    }

    /// Reads a slot; the key is recovered from `check ^ data`.
    fn read(&self, index: usize) -> Entry {
        let slot = &self.slots[index];
        let check = slot.check.load(Ordering::Relaxed);
        let data = slot.data.load(Ordering::Relaxed);
        Entry::unpack(check ^ data, data)
    }

    /// Entry for position `key`, if present.
    #[must_use]
    pub fn probe(&self, key: u64) -> Option<Entry> {
        let entry = self.read(self.idx(key));
        (entry.bound != Bound::None && entry.key == key).then_some(entry)
    }

    /// Stores an entry of generation `age`. An empty, identical, stale or shallower entry is
    /// replaced; without a new move, the old move of the same position is kept.
    pub fn store(&self, new: Entry, age: u8) {
        let index = self.idx(new.key);
        let entry = self.read(index);
        let same = entry.key == new.key;
        if entry.bound == Bound::None || same || entry.age != age || new.depth >= entry.depth {
            let mv = if new.mv == 0 && same { entry.mv } else { new.mv };
            let data = Entry { mv, age, ..new }.pack();
            let slot = &self.slots[index];
            slot.data.store(data, Ordering::Relaxed);
            slot.check.store(new.key ^ data, Ordering::Relaxed);
        }
    }

    /// Occupancy in permille (sample of the first 1,000 slots, generation `age` only).
    #[must_use]
    pub fn hashfull(&self, age: u8) -> usize {
        (0..1000.min(self.slots.len()))
            .map(|index| self.read(index))
            .filter(|entry| entry.bound != Bound::None && entry.age == age)
            .count()
    }
}

#[cfg(test)]
mod tests;
