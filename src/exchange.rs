//! Materialwerte und statische Abtauschbewertung (SEE, *static exchange evaluation*).
//!
//! SEE schätzt, was ein Schlag auf einem Feld am Ende einbringt, wenn beide Seiten abwechselnd mit
//! ihrer jeweils billigsten Figur zurückschlagen und jede Seite aufhören darf, sobald Weiterschlagen
//! schlechter wäre. Fesselungen werden nicht beachtet.

use shakmaty::{Bitboard, Board, Chess, Color, Move, Position, Role, Square};

/// Längste betrachtete Schlagfolge auf einem Feld.
const MAX_EXCHANGE: usize = 32;

/// Materialwert in cp (für Sortierung, Abtausch und Delta-Pruning).
#[must_use]
pub fn value(role: Role) -> i32 {
    match role {
        Role::Pawn => 100,
        Role::Knight | Role::Bishop => 300,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 20_000,
    }
}

/// Wert der geschlagenen Figur in cp (0 für Züge ohne Schlag).
pub(crate) fn victim_value(mv: Move) -> i32 {
    if mv.is_en_passant() {
        value(Role::Pawn)
    } else {
        mv.capture().map_or(0, value)
    }
}

/// Abtauschbilanz eines Schlags in cp aus Sicht des Ziehenden; 0 für Züge ohne Schlag.
#[must_use]
pub fn see(pos: &Chess, mv: Move) -> i32 {
    let (Some(from), victim) = (mv.from(), victim_value(mv)) else {
        return 0;
    };
    if victim == 0 {
        return 0;
    }
    let mut gains = [0; MAX_EXCHANGE];
    let last = speculative_gains(pos, mv, from, &mut gains);
    best_outcome(&mut gains, last)
}

/// Ruhiger Zug: steht die Figur auf dem Zielfeld angegriffen und ungedeckt, gilt sie als verloren.
pub(crate) fn see_quiet(pos: &Chess, mv: Move) -> i32 {
    let Some(from) = mv.from() else { return 0 };
    let board = pos.board();
    let (us, them) = (pos.turn(), !pos.turn());
    let occ = board.occupied().without(from);
    let attacked = (board.attacks_to(mv.to(), them, occ) & board.by_color(them)).any();
    let defended = (board.attacks_to(mv.to(), us, occ) & board.by_color(us) & occ).any();
    if attacked && !defended {
        -value(mv.role())
    } else {
        0
    }
}

/// Spielt die Schlagfolge auf dem Zielfeld durch. `gains[k]` ist die Bilanz der Seite, die den
/// k-ten Schlag ausführt, falls danach niemand mehr schlägt. Gibt den Index des letzten Schlags zurück.
fn speculative_gains(pos: &Chess, mv: Move, from: Square, gains: &mut [i32; MAX_EXCHANGE]) -> usize {
    let board = pos.board();
    let target = mv.to();
    let mut occ = occupancy_after(board, mv, from);
    let mut on_target = mv.role();
    let mut side = !pos.turn();
    let mut last: usize = 0;
    gains[0] = victim_value(mv);
    while let Some((role, sq)) = least_valuable_attacker(board, target, side, occ) {
        if on_target == Role::King {
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
        occ = occ.without(sq);
        side = !side;
    }
    last
}

/// Belegung nach dem ersten Schlag ohne die schlagende Figur; en passant auch ohne den geschlagenen Bauern.
fn occupancy_after(board: &Board, mv: Move, from: Square) -> Bitboard {
    let occ = board.occupied().without(from);
    if mv.is_en_passant() {
        occ.without(Square::from_coords(mv.to().file(), from.rank()))
    } else {
        occ
    }
}

/// Billigste Figur von `side`, die `target` bei Belegung `occ` angreift (Röntgenangriffe eingeschlossen).
fn least_valuable_attacker(
    board: &Board,
    target: Square,
    side: Color,
    occ: Bitboard,
) -> Option<(Role, Square)> {
    let attackers = board.attacks_to(target, side, occ) & occ & board.by_color(side);
    Role::ALL
        .iter()
        .find_map(|&role| (attackers & board.by_role(role)).first().map(|sq| (role, sq)))
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
    use shakmaty::fen::Fen;
    use shakmaty::uci::UciMove;
    use shakmaty::CastlingMode;

    fn play(fen: &str, uci: &str) -> (Chess, Move) {
        let pos: Chess = fen
            .parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap();
        let mv = uci.parse::<UciMove>().unwrap().to_move(&pos).unwrap();
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
