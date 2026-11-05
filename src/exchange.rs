//! Materialwerte und statische Abtauschbewertung (SEE, *static exchange evaluation*).
//!
//! SEE schätzt, was ein Schlag auf einem Feld am Ende einbringt, wenn beide Seiten abwechselnd mit
//! ihrer jeweils billigsten Figur zurückschlagen und jede Seite aufhören darf, sobald Weiterschlagen
//! schlechter wäre. Fesselungen werden nicht beachtet.

use crate::board::{Color, Move, Piece, Position, Square};
use cozy_chess::BitBoard;

/// Längste betrachtete Schlagfolge auf einem Feld.
const MAX_EXCHANGE: usize = 32;

/// Materialwert in cp (für Sortierung, Abtausch und Delta-Pruning).
#[must_use]
pub fn value(role: Piece) -> i32 {
    match role {
        Piece::Pawn => 100,
        Piece::Knight | Piece::Bishop => 300,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20_000,
    }
}

/// Wert der geschlagenen Figur in cp (0 für Züge ohne Schlag).
pub(crate) fn victim_value(mv: Move) -> i32 {
    if mv.is_en_passant() {
        value(Piece::Pawn)
    } else {
        mv.capture().map_or(0, value)
    }
}

/// Abtauschbilanz eines Schlags in cp aus Sicht des Ziehenden; 0 für Züge ohne Schlag.
#[must_use]
pub fn see(pos: &Position, mv: Move) -> i32 {
    let (from, victim) = (mv.from(), victim_value(mv));
    if victim == 0 {
        return 0;
    }
    let mut gains = [0; MAX_EXCHANGE];
    let last = speculative_gains(pos, mv, from, &mut gains);
    best_outcome(&mut gains, last)
}

/// Ruhiger Zug: steht die Figur auf dem Zielfeld angegriffen und ungedeckt, gilt sie als verloren.
pub(crate) fn see_quiet(pos: &Position, mv: Move) -> i32 {
    let (us, them) = (pos.turn(), !pos.turn());
    let occ = pos.board().occupied() & !mv.from().bitboard();
    let attacked = !pos.attackers(mv.to(), them, occ).is_empty();
    let defended = !(pos.attackers(mv.to(), us, occ) & occ).is_empty();
    if attacked && !defended {
        -value(mv.role())
    } else {
        0
    }
}

/// Spielt die Schlagfolge auf dem Zielfeld durch. `gains[k]` ist die Bilanz der Seite, die den
/// k-ten Schlag ausführt, falls danach niemand mehr schlägt. Gibt den Index des letzten Schlags zurück.
fn speculative_gains(pos: &Position, mv: Move, from: Square, gains: &mut [i32; MAX_EXCHANGE]) -> usize {
    let target = mv.to();
    let mut occ = occupancy_after(pos, mv, from);
    let mut on_target = mv.role();
    let mut side = !pos.turn();
    let mut last: usize = 0;
    gains[0] = victim_value(mv);
    while let Some((role, sq)) = least_valuable_attacker(pos, target, side, occ) {
        if on_target == Piece::King {
            // Der König hat auf ein noch angegriffenes Feld geschlagen: unzulässig, Schlag zurücknehmen
            last = last.saturating_sub(1);
            break;
        }
        if last + 1 >= MAX_EXCHANGE {
            break;
        }
        last += 1;
        gains[last] = value(on_target) - gains[last - 1];
        on_target = role;
        occ &= !sq.bitboard();
        side = !side;
    }
    last
}

/// Belegung nach dem ersten Schlag ohne die schlagende Figur; en passant auch ohne den geschlagenen Bauern.
fn occupancy_after(pos: &Position, mv: Move, from: Square) -> BitBoard {
    let occ = pos.board().occupied() & !from.bitboard();
    if mv.is_en_passant() {
        occ & !Square::new(mv.to().file(), from.rank()).bitboard()
    } else {
        occ
    }
}

/// Billigste Figur von `side`, die `target` bei Belegung `occ` angreift (Röntgenangriffe eingeschlossen).
fn least_valuable_attacker(
    pos: &Position,
    target: Square,
    side: Color,
    occ: BitBoard,
) -> Option<(Piece, Square)> {
    let attackers = pos.attackers(target, side, occ) & occ;
    Piece::ALL.iter().find_map(|&role| {
        (attackers & pos.board().pieces(role))
            .next_square()
            .map(|sq| (role, sq))
    })
}

/// Wertet die Schlagfolge rückwärts aus: Jede Seite hört auf, wenn Weiterschlagen schlechter wäre.
fn best_outcome(gains: &mut [i32; MAX_EXCHANGE], last: usize) -> i32 {
    for step in (1..=last).rev() {
        gains[step - 1] = -(-gains[step - 1]).max(gains[step]);
    }
    gains[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(fen: &str, uci: &str) -> (Position, Move) {
        let pos = Position::from_fen(fen).unwrap();
        let mv = pos.parse_uci(uci).unwrap();
        (pos, mv)
    }

    #[test]
    fn undefended_piece_is_won() {
        let (position, mv) = play("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1", "e4d5");
        assert_eq!(see(&position, mv), 300);
    }

    #[test]
    fn defender_stops_when_recapture_loses() {
        // Turm schlägt Bauern, Dame könnte zurückschlagen, wird aber vom zweiten Turm wiedergeschlagen
        let (position, mv) = play("3qk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5");
        assert_eq!(see(&position, mv), 100);
    }

    #[test]
    fn xray_attacker_joins_the_exchange() {
        // Txd5 Txd5, dann lohnt Dxd5 nicht, weil der Turm d8 (hinter d7) zurückschlägt: Turm gegen Bauer
        let (position, mv) = play("3rk3/3r4/8/3p4/8/8/3R4/3QK3 w - - 0 1", "d2d5");
        assert_eq!(see(&position, mv), -400);
    }

    #[test]
    fn king_cannot_recapture_into_attack() {
        // Nur der König deckt d5, der zweite Turm steht dahinter: Kxd5 wäre unzulässig
        let (position, mv) = play("8/8/4k3/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5");
        assert_eq!(see(&position, mv), 100);
        // Ohne zweiten Turm darf der König zurückschlagen: Turm gegen Bauer
        let (other_position, other_mv) = play("8/8/4k3/3p4/8/8/3R4/4K3 w - - 0 1", "d2d5");
        assert_eq!(see(&other_position, other_mv), -400);
    }

    #[test]
    fn en_passant_counts_a_pawn() {
        let (position, mv) = play("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6");
        assert_eq!(see(&position, mv), 100);
    }

    #[test]
    fn quiet_move_into_attack() {
        let (safe_pos, safe) = play("4k3/8/3p4/8/8/8/8/2B1K3 w - - 0 1", "c1e3");
        assert_eq!(see_quiet(&safe_pos, safe), 0);
        let (hang_pos, hang) = play("4k3/8/5p2/8/8/8/8/2B1K3 w - - 0 1", "c1g5");
        assert_eq!(see_quiet(&hang_pos, hang), -300);
        let (guard_pos, guarded) = play("4k3/8/5p2/8/7P/8/8/2B1K3 w - - 0 1", "c1g5");
        assert_eq!(see_quiet(&guard_pos, guarded), 0);
    }

    #[test]
    fn quiet_moves_have_no_exchange() {
        let (position, mv) = play("4k3/8/8/8/8/8/3R4/4K3 w - - 0 1", "d2d5");
        assert_eq!(see(&position, mv), 0);
    }
}
