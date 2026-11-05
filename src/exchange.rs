//! Material values and static exchange evaluation (SEE).
//!
//! SEE estimates the final outcome of a capture on a square when both sides alternately recapture
//! with their cheapest piece and each side may stop as soon as continuing to capture would be worse.
//! Pins are ignored.

use crate::board::{Color, Move, Piece, Position, Square};
use cozy_chess::BitBoard;

/// Longest capture sequence considered on one square.
const MAX_EXCHANGE: usize = 32;

/// Material value in cp (for move ordering, exchanges and delta pruning).
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

/// Value of the captured piece in cp (0 for non-captures).
pub(crate) fn victim_value(mv: Move) -> i32 {
    if mv.is_en_passant() {
        value(Piece::Pawn)
    } else {
        mv.capture().map_or(0, value)
    }
}

/// Exchange balance of a capture in cp from the mover's view; 0 for non-captures.
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

/// Quiet move: if the piece stands attacked and undefended on the target square, it counts as lost.
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

/// Plays out the capture sequence on the target square. `gains[k]` is the balance of the side making
/// the k-th capture if nobody captures afterwards. Returns the index of the last capture.
fn speculative_gains(pos: &Position, mv: Move, from: Square, gains: &mut [i32; MAX_EXCHANGE]) -> usize {
    let target = mv.to();
    let mut occ = occupancy_after(pos, mv, from);
    let mut on_target = mv.role();
    let mut side = !pos.turn();
    let mut last: usize = 0;
    gains[0] = victim_value(mv);
    while let Some((role, sq)) = least_valuable_attacker(pos, target, side, occ) {
        if on_target == Piece::King {
            // The king captured onto a square that is still attacked: illegal, take the capture back
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

/// Occupancy after the first capture without the capturer; for en passant also without the captured pawn.
fn occupancy_after(pos: &Position, mv: Move, from: Square) -> BitBoard {
    let occ = pos.board().occupied() & !from.bitboard();
    if mv.is_en_passant() {
        occ & !Square::new(mv.to().file(), from.rank()).bitboard()
    } else {
        occ
    }
}

/// Cheapest piece of `side` attacking `target` with occupancy `occ` (including X-ray attacks).
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

/// Evaluates the capture sequence backwards: each side stops when continuing to capture would be worse.
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
        // Rook takes pawn, queen could recapture but would be taken by the second rook
        let (position, mv) = play("3qk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5");
        assert_eq!(see(&position, mv), 100);
    }

    #[test]
    fn xray_attacker_joins_the_exchange() {
        // Rxd5 Rxd5, then Qxd5 does not pay off because the rook on d8 (behind d7) recaptures: rook for pawn
        let (position, mv) = play("3rk3/3r4/8/3p4/8/8/3R4/3QK3 w - - 0 1", "d2d5");
        assert_eq!(see(&position, mv), -400);
    }

    #[test]
    fn king_cannot_recapture_into_attack() {
        // Only the king defends d5, the second rook stands behind: Kxd5 would be illegal
        let (position, mv) = play("8/8/4k3/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5");
        assert_eq!(see(&position, mv), 100);
        // Without the second rook the king may recapture: rook for pawn
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
