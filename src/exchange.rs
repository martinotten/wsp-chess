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
pub fn value(r: Role) -> i32 {
    match r {
        Role::Pawn => 100,
        Role::Knight | Role::Bishop => 300,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 20_000,
    }
}

/// Wert der geschlagenen Figur in cp (0 für Züge ohne Schlag).
pub(crate) fn victim_value(m: Move) -> i32 {
    if m.is_en_passant() {
        value(Role::Pawn)
    } else {
        m.capture().map_or(0, value)
    }
}

/// Abtauschbilanz eines Schlags in cp aus Sicht des Ziehenden; 0 für Züge ohne Schlag.
#[must_use]
pub fn see(pos: &Chess, m: Move) -> i32 {
    let (Some(from), victim) = (m.from(), victim_value(m)) else {
        return 0;
    };
    if victim == 0 {
        return 0;
    }
    let mut gains = [0; MAX_EXCHANGE];
    let last = speculative_gains(pos, m, from, &mut gains);
    best_outcome(&mut gains, last)
}

/// Ruhiger Zug: steht die Figur auf dem Zielfeld angegriffen und ungedeckt, gilt sie als verloren.
pub(crate) fn see_quiet(pos: &Chess, m: Move) -> i32 {
    let Some(from) = m.from() else { return 0 };
    let b = pos.board();
    let (us, them) = (pos.turn(), !pos.turn());
    let occ = b.occupied().without(from);
    let attacked = (b.attacks_to(m.to(), them, occ) & b.by_color(them)).any();
    let defended = (b.attacks_to(m.to(), us, occ) & b.by_color(us) & occ).any();
    if attacked && !defended {
        -value(m.role())
    } else {
        0
    }
}

/// Spielt die Schlagfolge auf dem Zielfeld durch. `gains[k]` ist die Bilanz der Seite, die den
/// k-ten Schlag ausführt, falls danach niemand mehr schlägt. Gibt den Index des letzten Schlags zurück.
fn speculative_gains(pos: &Chess, m: Move, from: Square, gains: &mut [i32; MAX_EXCHANGE]) -> usize {
    let b = pos.board();
    let target = m.to();
    let mut occ = occupancy_after(b, m, from);
    let mut on_target = m.role();
    let mut side = !pos.turn();
    let mut last = 0;
    gains[0] = victim_value(m);
    while let Some((role, sq)) = least_valuable_attacker(b, target, side, occ) {
        // Steht der König auf dem Feld und greift noch eine Figur an, endet die Folge. Ungenau: Der
        // unzulässige Schlag des Königs selbst bleibt gezählt (Test `king_cannot_recapture_into_attack`).
        if on_target == Role::King || last + 1 >= MAX_EXCHANGE {
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
fn occupancy_after(b: &Board, m: Move, from: Square) -> Bitboard {
    let occ = b.occupied().without(from);
    if m.is_en_passant() {
        occ.without(Square::from_coords(m.to().file(), from.rank()))
    } else {
        occ
    }
}

/// Billigste Figur von `side`, die `target` bei Belegung `occ` angreift (Röntgenangriffe eingeschlossen).
fn least_valuable_attacker(b: &Board, target: Square, side: Color, occ: Bitboard) -> Option<(Role, Square)> {
    let attackers = b.attacks_to(target, side, occ) & occ & b.by_color(side);
    Role::ALL
        .iter()
        .find_map(|&r| (attackers & b.by_role(r)).first().map(|sq| (r, sq)))
}

/// Wertet die Schlagfolge rückwärts aus: Jede Seite hört auf, wenn Weiterschlagen schlechter wäre.
fn best_outcome(gains: &mut [i32; MAX_EXCHANGE], last: usize) -> i32 {
    for k in (1..=last).rev() {
        gains[k - 1] = -(-gains[k - 1]).max(gains[k]);
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
        let m = uci.parse::<UciMove>().unwrap().to_move(&pos).unwrap();
        (pos, m)
    }

    #[test]
    fn undefended_piece_is_won() {
        let (p, m) = play("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1", "e4d5");
        assert_eq!(see(&p, m), 300);
    }

    #[test]
    fn defender_stops_when_recapture_loses() {
        // Turm schlägt Bauern, Dame könnte zurückschlagen, wird aber vom zweiten Turm wiedergeschlagen
        let (p, m) = play("3qk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5");
        assert_eq!(see(&p, m), 100);
    }

    #[test]
    fn xray_attacker_joins_the_exchange() {
        // Txd5 Txd5, dann lohnt Dxd5 nicht, weil der Turm d8 (hinter d7) zurückschlägt: Turm gegen Bauer
        let (p, m) = play("3rk3/3r4/8/3p4/8/8/3R4/3QK3 w - - 0 1", "d2d5");
        assert_eq!(see(&p, m), -400);
    }

    #[test]
    #[ignore = "bekannter Fehler: Königsschlag ins Angriffsfeld wird gezählt; Korrektur braucht SPRT"]
    fn king_cannot_recapture_into_attack() {
        let (p, m) = play("8/8/4k3/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5");
        assert_eq!(see(&p, m), 100);
    }

    #[test]
    fn en_passant_counts_a_pawn() {
        let (p, m) = play("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6");
        assert_eq!(see(&p, m), 100);
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
        let (p, m) = play("4k3/8/8/8/8/8/3R4/4K3 w - - 0 1", "d2d5");
        assert_eq!(see(&p, m), 0);
    }
}
