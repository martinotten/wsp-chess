//! Ruhesuche: Schläge und Umwandlungen bis zur ruhigen Stellung, im Schach alle Abwehrzüge.

use super::node::{QNode, Window};
use super::score::mated_in;
use super::{is_quiet, Feature, Searcher, INF, MAX_PLY};
use crate::board::{Move, MoveList, Position};
use crate::exchange::{see, see_quiet, victim_value};

/// Delta-Pruning: Sicherheitsabstand in cp.
const DELTA_MARGIN: i32 = 200;

fn gives_check(pos: &Position, mv: Move) -> bool {
    let mut child = pos.clone();
    child.play_unchecked(mv);
    child.is_check()
}

impl Searcher {
    /// Ruhesuche: nur Schläge und Umwandlungen (im Schach alle Züge), bis die Stellung ruhig ist.
    pub(super) fn qsearch(&mut self, pos: &Position, mut win: Window, at: QNode) -> i32 {
        let ply = at.ply;
        if !self.enter(ply) {
            return 0;
        }
        if pos.is_insufficient_material() {
            return 0;
        }
        let in_check = pos.is_check();
        if ply >= MAX_PLY - 1 {
            return if in_check { 0 } else { self.eval(pos, ply) };
        }
        if in_check {
            return self.qsearch_evasions(pos, win, at);
        }
        let moves = self.quiescence_moves(pos, at.qdepth);
        // Stand-pat: Der Ziehende muss nicht schlagen, die statische Bewertung ist eine Untergrenze
        let stand = self.eval(pos, ply);
        if stand >= win.beta {
            return stand;
        }
        win.alpha = win.alpha.max(stand);
        self.qsearch_moves(pos, &moves, win, at, Some(stand))
    }

    /// Ruhesuche im Schach: alle Abwehrzüge, kein Stand-pat.
    fn qsearch_evasions(&mut self, pos: &Position, win: Window, at: QNode) -> i32 {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return mated_in(at.ply);
        }
        self.qsearch_moves(pos, &moves, win, at, None)
    }

    /// Schläge, ruhige Umwandlungen und (mit [`Feature::QChecks`], erste Ebene) sichere Schachgebote.
    fn quiescence_moves(&self, pos: &Position, qdepth: u32) -> MoveList {
        let mut moves = pos.noisy_moves();
        if qdepth == 0 && self.opts.on(Feature::QChecks) {
            moves.extend(
                pos.legal_moves()
                    .into_iter()
                    .filter(|&mv| is_quiet(mv) && gives_check(pos, mv) && see_quiet(pos, mv) >= 0),
            );
        }
        moves
    }

    /// Durchsucht die Züge der Ruhesuche. `stand` = Stand-pat-Wert, `None` im Schach.
    fn qsearch_moves(
        &mut self,
        pos: &Position,
        moves: &[Move],
        mut win: Window,
        at: QNode,
        stand: Option<i32>,
    ) -> i32 {
        let mut best = stand.unwrap_or(-INF);
        for mv in self.sorted(pos, moves, 0, at.ply) {
            if stand.is_some_and(|stand_pat| self.skip_capture(pos, mv, stand_pat, win.alpha)) {
                continue;
            }
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = -self.qsearch(&child, win.flip(), at.child());
            if self.aborted {
                return 0;
            }
            if score <= best {
                continue;
            }
            best = score;
            win.alpha = win.alpha.max(score);
            if win.alpha >= win.beta {
                break;
            }
        }
        best
    }

    /// Schlag, der in der Ruhesuche nicht lohnt: Selbst der Gewinn der Figur hebt den Wert nicht über
    /// alpha (Delta-Pruning), oder der Abtausch verliert Material (SEE).
    fn skip_capture(&self, pos: &Position, mv: Move, stand: i32, alpha: i32) -> bool {
        if mv.is_promotion() || !mv.is_capture() {
            return false;
        }
        let hopeless = self.opts.on(Feature::Delta) && stand + victim_value(mv) + DELTA_MARGIN < alpha;
        hopeless || (self.opts.on(Feature::SeePrune) && see(pos, mv) < 0)
    }
}
