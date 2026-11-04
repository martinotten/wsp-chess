//! Ruhesuche: Schläge und Umwandlungen bis zur ruhigen Stellung, im Schach alle Abwehrzüge.

use super::node::{QNode, Window};
use super::score::mated_in;
use super::{is_quiet, Feature, Searcher, INF, MAX_PLY};
use crate::exchange::{see, see_quiet, victim_value};
use shakmaty::{Chess, Move, MoveList, Position};

/// Delta-Pruning: Sicherheitsabstand in cp.
const DELTA_MARGIN: i32 = 200;

fn gives_check(pos: &Chess, m: Move) -> bool {
    let mut child = pos.clone();
    child.play_unchecked(m);
    child.is_check()
}

impl Searcher {
    /// Ruhesuche: nur Schläge und Umwandlungen (im Schach alle Züge), bis die Stellung ruhig ist.
    pub(super) fn qsearch(&mut self, pos: &Chess, mut w: Window, at: QNode) -> i32 {
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
            return self.qsearch_evasions(pos, w, at);
        }
        let moves = self.quiescence_moves(pos, at.qdepth);
        // Stand-pat: Der Ziehende muss nicht schlagen, die statische Bewertung ist eine Untergrenze
        let stand = self.eval(pos, ply);
        if stand >= w.beta {
            return stand;
        }
        w.alpha = w.alpha.max(stand);
        self.qsearch_moves(pos, &moves, w, at, Some(stand))
    }

    /// Ruhesuche im Schach: alle Abwehrzüge, kein Stand-pat.
    fn qsearch_evasions(&mut self, pos: &Chess, w: Window, at: QNode) -> i32 {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return mated_in(at.ply);
        }
        self.qsearch_moves(pos, &moves, w, at, None)
    }

    /// Schläge, ruhige Umwandlungen und (mit [`Feature::QChecks`], erste Ebene) sichere Schachgebote.
    fn quiescence_moves(&self, pos: &Chess, qdepth: u32) -> MoveList {
        let mut moves = pos.capture_moves();
        moves.extend(pos.promotion_moves().into_iter().filter(|m| !m.is_capture()));
        if qdepth == 0 && self.opts.on(Feature::QChecks) {
            moves.extend(
                pos.legal_moves()
                    .into_iter()
                    .filter(|&m| is_quiet(m) && gives_check(pos, m) && see_quiet(pos, m) >= 0),
            );
        }
        moves
    }

    /// Durchsucht die Züge der Ruhesuche. `stand` = Stand-pat-Wert, `None` im Schach.
    fn qsearch_moves(
        &mut self,
        pos: &Chess,
        moves: &[Move],
        mut w: Window,
        at: QNode,
        stand: Option<i32>,
    ) -> i32 {
        let mut best = stand.unwrap_or(-INF);
        for m in self.sorted(pos, moves, 0, at.ply) {
            if stand.is_some_and(|s| self.skip_capture(pos, m, s, w.alpha)) {
                continue;
            }
            let mut child = pos.clone();
            child.play_unchecked(m);
            let score = -self.qsearch(&child, w.flip(), at.child());
            if self.aborted {
                return 0;
            }
            if score <= best {
                continue;
            }
            best = score;
            w.alpha = w.alpha.max(score);
            if w.alpha >= w.beta {
                break;
            }
        }
        best
    }

    /// Schlag, der in der Ruhesuche nicht lohnt: Selbst der Gewinn der Figur hebt den Wert nicht über
    /// alpha (Delta-Pruning), oder der Abtausch verliert Material (SEE).
    fn skip_capture(&self, pos: &Chess, m: Move, stand: i32, alpha: i32) -> bool {
        if m.is_promotion() || !m.is_capture() {
            return false;
        }
        let hopeless = self.opts.on(Feature::Delta) && stand + victim_value(m) + DELTA_MARGIN < alpha;
        hopeless || (self.opts.on(Feature::SeePrune) && see(pos, m) < 0)
    }
}
