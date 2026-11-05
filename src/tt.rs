//! Transpositionstabelle: ein Eintrag je Platz, Ersetzung nach Alter und Tiefe, threadsicher ohne `unsafe`.
//!
//! Ein Platz besteht aus zwei `AtomicU64`: `data` (gepackter Eintrag) und `check = key ^ data`.
//! Beim Lesen gilt ein Eintrag nur, wenn `check ^ data` den gesuchten Schlüssel ergibt. Schreiben zwei
//! Threads gleichzeitig denselben Platz, kann ein gemischtes Paar entstehen; es fällt bei dieser
//! Prüfung mit sehr hoher Wahrscheinlichkeit heraus (probabilistischer Schutz wie in üblichen
//! Lazy-SMP-Engines). Was dennoch durchrutscht, ist harmlos: Der Zug wird vor Gebrauch gegen die legalen
//! Züge geprüft (`Searcher::find`, Zugsortierung), Werte liegen durch die Packung in festen Grenzen.

use std::sync::atomic::{AtomicU64, Ordering};

/// Art der gespeicherten Schranke.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Bound {
    /// Leerer Platz.
    #[default]
    None,
    /// Genauer Wert.
    Exact,
    /// Untere Schranke (Beta-Schnitt).
    Lower,
    /// Obere Schranke (kein Zug über Alpha).
    Upper,
}

/// Ein Eintrag der Tabelle (entpackt).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Entry {
    /// Zobrist-Schlüssel der Stellung.
    pub key: u64,
    /// Bester Zug (siehe `search::move_code`, 15 Bit), 0 = keiner.
    pub mv: u16,
    /// Suchwert, Mattwerte relativ zur Stellung normalisiert (passt in `i16`).
    pub score: i32,
    /// Statische Bewertung (cp, ±20.000), falls bekannt.
    pub eval: Option<i32>,
    /// Restsuchtiefe, mit der `score` berechnet wurde (1–127).
    pub depth: i32,
    /// Art der Schranke.
    pub bound: Bound,
    /// Suchgeneration; setzt [`Tt::store`].
    pub age: u8,
}

/// Lage der Felder in `data` (Bit 0 = niedrigstes): Zug 15, Wert 16, Bewertung 16, Tiefe 7,
/// Schranke 2, Alter 8 = 64 Bit.
const MOVE_BITS: u32 = 15;
const SCORE_SHIFT: u32 = 15;
const EVAL_SHIFT: u32 = 31;
const DEPTH_SHIFT: u32 = 47;
const BOUND_SHIFT: u32 = 54;
const AGE_SHIFT: u32 = 56;
const DEPTH_MAX: i32 = 127;
/// `eval` ohne Wert.
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

/// `i16` als 16 Bit ohne Vorzeichenerweiterung.
fn bits16(value: i16) -> u64 {
    u64::from(value.cast_unsigned())
}

/// Untere 16 Bit ab `shift` als `i16`.
fn field16(data: u64, shift: u32) -> i16 {
    let low = data >> shift & 0xFFFF;
    u16::try_from(low).unwrap_or(0).cast_signed()
}

impl Entry {
    /// Packt den Eintrag ohne Schlüssel. Werte außerhalb der Bereiche werden in Debug-Builds gemeldet und
    /// sonst begrenzt.
    fn pack(&self) -> u64 {
        debug_assert!(u32::from(self.mv) < 1 << MOVE_BITS, "Zugkode {}", self.mv);
        debug_assert!((1..=DEPTH_MAX).contains(&self.depth), "Tiefe {}", self.depth);
        debug_assert!(i16::try_from(self.score).is_ok(), "Wert {}", self.score);
        let score = i16::try_from(self.score).unwrap_or(if self.score > 0 { i16::MAX } else { i16::MIN + 1 });
        let eval = self.eval.map_or(NO_EVAL, |eval| {
            debug_assert!(
                eval > i32::from(NO_EVAL) && eval <= i32::from(i16::MAX),
                "Bewertung {eval}"
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

    /// Gegenstück zu [`Entry::pack`].
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

/// Ein Platz: gepackter Eintrag und `key ^ data`.
#[derive(Default)]
struct Slot {
    check: AtomicU64,
    data: AtomicU64,
}

/// Transpositionstabelle fester Größe (Zweierpotenz), von mehreren Suchthreads gemeinsam nutzbar.
pub struct Tt {
    slots: Vec<Slot>,
}

impl Tt {
    /// Tabelle mit höchstens `mb` Megabyte (16 Byte je Platz).
    #[must_use]
    pub fn new(mb: usize) -> Tt {
        let n_slots = (mb.max(1) * 1024 * 1024 / size_of::<Slot>()).next_power_of_two() / 2;
        Tt {
            slots: (0..n_slots.max(1024)).map(|_| Slot::default()).collect(),
        }
    }

    /// Alle Einträge löschen (nur, während keine Suche läuft).
    pub fn clear(&self) {
        for slot in &self.slots {
            slot.check.store(0, Ordering::Relaxed);
            slot.data.store(0, Ordering::Relaxed);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Bits gezielt abgeschnitten (Hash-Index)"
    )]
    fn idx(&self, key: u64) -> usize {
        (key as usize) & (self.slots.len() - 1)
    }

    /// Liest einen Platz; der Schlüssel ergibt sich aus `check ^ data`.
    fn read(&self, index: usize) -> Entry {
        let slot = &self.slots[index];
        let check = slot.check.load(Ordering::Relaxed);
        let data = slot.data.load(Ordering::Relaxed);
        Entry::unpack(check ^ data, data)
    }

    /// Eintrag zur Stellung `key`, falls vorhanden.
    #[must_use]
    pub fn probe(&self, key: u64) -> Option<Entry> {
        let entry = self.read(self.idx(key));
        (entry.bound != Bound::None && entry.key == key).then_some(entry)
    }

    /// Speichert einen Eintrag der Generation `age`. Ersetzt wird ein leerer, gleicher, veralteter oder
    /// flacherer Eintrag; ohne neuen Zug bleibt der alte Zug derselben Stellung erhalten.
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

    /// Belegung in Promille (Stichprobe der ersten 1000 Plätze, nur Generation `age`).
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
