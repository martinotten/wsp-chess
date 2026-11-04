//! Werte: Netzausgabe in Centipawn, Mattwerte und ihre Umrechnung für die Transpositionstabelle.

use super::{MATE, MATE_BOUND};
use crate::tt::Bound;

pub(super) const EVAL_CLAMP: f32 = 20_000.0;

pub(super) fn ply_i32(ply: usize) -> i32 {
    i32::try_from(ply).unwrap_or(i32::MAX)
}

/// Wert für „am Zug und matt“ auf Höhe `ply`.
pub(super) fn mated_in(ply: usize) -> i32 {
    -(MATE - ply_i32(ply))
}

#[expect(clippy::cast_possible_truncation, reason = "Wert vorher auf ±20000 begrenzt")]
pub(super) fn to_cp(s: f32) -> i32 {
    (s * 100.0).round().clamp(-EVAL_CLAMP, EVAL_CLAMP) as i32
}

/// Mattwert beim Speichern auf die Knotenhöhe normalisieren („matt in n ab hier“).
pub(super) fn to_tt(s: i32, ply: usize) -> i32 {
    let p = ply_i32(ply);
    if s >= MATE_BOUND {
        s + p
    } else if s <= -MATE_BOUND {
        s - p
    } else {
        s
    }
}

/// Gegenstück zu [`to_tt`]: gespeicherten Mattwert auf die aktuelle Höhe umrechnen.
pub(super) fn from_tt(s: i32, ply: usize) -> i32 {
    let p = ply_i32(ply);
    if s >= MATE_BOUND {
        s - p
    } else if s <= -MATE_BOUND {
        s + p
    } else {
        s
    }
}

/// Art des Werts für die Transpositionstabelle, gemessen am Fenster beim Betreten des Knotens.
pub(super) fn bound_for(best: i32, alpha_at_entry: i32, beta: i32) -> Bound {
    if best >= beta {
        Bound::Lower
    } else if best > alpha_at_entry {
        Bound::Exact
    } else {
        Bound::Upper
    }
}
