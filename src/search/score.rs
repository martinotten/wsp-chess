//! Scores: network output in centipawns, mate scores and their conversion for the transposition
//! table.

use super::{MATE, MATE_BOUND};
use crate::tt::Bound;

pub(super) const EVAL_CLAMP: f32 = 20_000.0;

pub(super) fn ply_i32(ply: usize) -> i32 {
    i32::try_from(ply).unwrap_or(i32::MAX)
}

/// Score for "to move and mated" at ply `ply`.
pub(super) fn mated_in(ply: usize) -> i32 {
    -(MATE - ply_i32(ply))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "value clamped to ±20000 beforehand"
)]
pub(super) fn to_cp(pawns: f32) -> i32 {
    (pawns * 100.0).round().clamp(-EVAL_CLAMP, EVAL_CLAMP) as i32
}

/// Normalise a mate score to the node's ply when storing ("mate in n from here").
pub(super) fn to_tt(score: i32, ply: usize) -> i32 {
    let height = ply_i32(ply);
    if score >= MATE_BOUND {
        score + height
    } else if score <= -MATE_BOUND {
        score - height
    } else {
        score
    }
}

/// Counterpart to [`to_tt`]: convert a stored mate score to the current ply.
pub(super) fn from_tt(score: i32, ply: usize) -> i32 {
    let height = ply_i32(ply);
    if score >= MATE_BOUND {
        score - height
    } else if score <= -MATE_BOUND {
        score + height
    } else {
        score
    }
}

/// Kind of score for the transposition table, measured against the window on entering the node.
pub(super) fn bound_for(best: i32, alpha_at_entry: i32, beta: i32) -> Bound {
    if best >= beta {
        Bound::Lower
    } else if best > alpha_at_entry {
        Bound::Exact
    } else {
        Bound::Upper
    }
}
