//! Hauptsuche: Negamax mit Alpha-Beta, Transpositionstabelle, Nullzug, Futility, PVS und LMR.

use super::node::{Best, Candidate, Node, Plan, QNode, TtProbe, Window};
use super::score::{bound_for, from_tt, mated_in, ply_i32, to_tt};
use super::{enc, hash, is_quiet, Feature, Searcher, INF, MATE_BOUND, MAX_PLY};
use crate::tt::{Bound, Entry};
use shakmaty::{Chess, Move, MoveList, Position};

/// Halbzüge ohne Schlag oder Bauernzug bis zum Remis (50-Züge-Regel).
const FIFTY_MOVE_PLIES: u32 = 100;

/// Nullzug: Mindesttiefe, Reduktion und Tiefe, ab der um einen Halbzug mehr reduziert wird.
const NULL_MIN_DEPTH: i32 = 3;
const NULL_REDUCTION: i32 = 2;
const NULL_DEEP_DEPTH: i32 = 7;

/// Futility: höchste Resttiefe und Sicherheitsabstand je Halbzug in cp.
pub(super) const FUTILITY_MAX_DEPTH: i32 = 2;
pub(super) const FUTILITY_MARGIN: i32 = 150;

/// LMR: ab dieser Tiefe und Zugnummer wird reduziert, ab den tiefen Werten um zwei Halbzüge.
const LMR_MIN_DEPTH: i32 = 3;
const LMR_MIN_INDEX: usize = 3;
const LMR_DEEP_DEPTH: i32 = 6;
const LMR_DEEP_INDEX: usize = 6;

/// Hat der Ziehende Figuren außer Bauern und König? Sonst droht Zugzwang, und der Nullzug täuscht.
fn non_pawn_material(pos: &Chess) -> bool {
    let b = pos.board();
    (b.by_color(pos.turn()) & !(b.pawns() | b.kings())).any()
}

impl Searcher {
    /// Negamax mit Alpha-Beta: Wert der Stellung aus Sicht des Ziehenden bei Resttiefe `depth`.
    pub(super) fn negamax(&mut self, pos: &Chess, depth: i32, win: Window, node: Node) -> i32 {
        if !self.enter(node.ply) {
            return 0;
        }
        if !node.is_root() && self.is_draw(pos, node.key) {
            return 0;
        }
        let moves = pos.legal_moves();
        if let Some(score) = self.leaf_score(pos, &moves, node.ply) {
            return score;
        }
        let in_check = pos.is_check();
        let depth = depth + self.extension(in_check, node.ply);
        if depth <= 0 {
            return self.qsearch(pos, win, QNode::entry(node.ply));
        }
        let (tt_move, tt_eval) = match self.probe_tt(node, depth, win) {
            TtProbe::Cutoff(score) => return score,
            TtProbe::Hint { mv, eval } => (mv, eval),
        };
        let static_eval = (!in_check).then(|| tt_eval.unwrap_or_else(|| self.eval(pos, node.ply)));
        if let Some(score) = self.try_null_move(pos, depth, win, node, static_eval) {
            return score;
        }
        if self.aborted {
            return 0;
        }
        let plan = Plan {
            node,
            depth,
            in_check,
            futile: static_eval.is_some_and(|se| self.is_futile(depth, win, se)),
            tt_move,
        };
        let mut moves_win = win;
        let Some(best) = self.search_moves(pos, &moves, &plan, &mut moves_win) else {
            return 0;
        };
        if best.score == -INF {
            // alle Züge durch Futility übersprungen
            return static_eval.unwrap_or(win.alpha).max(win.alpha);
        }
        let bound = bound_for(best.score, win.alpha, win.beta);
        let stored = to_tt(best.score, node.ply);
        self.tt.store(Entry {
            key: node.key,
            mv: best.code,
            score: stored,
            eval: static_eval,
            depth,
            bound,
            ..Entry::default()
        });
        best.score
    }

    /// Remis durch ungenügendes Material oder Wiederholung (nicht an der Wurzel prüfen).
    fn is_draw(&self, pos: &Chess, key: u64) -> bool {
        pos.is_insufficient_material() || self.is_repetition(key, pos.halfmoves())
    }

    /// Wert ohne Suche: Matt, Patt, 50-Züge-Regel oder größte Suchhöhe erreicht.
    pub(super) fn leaf_score(&mut self, pos: &Chess, moves: &MoveList, ply: usize) -> Option<i32> {
        if moves.is_empty() {
            return Some(if pos.is_check() { mated_in(ply) } else { 0 });
        }
        if ply > 0 && pos.halfmoves() >= FIFTY_MOVE_PLIES {
            return Some(0);
        }
        if ply >= MAX_PLY - 1 {
            return Some(self.eval(pos, ply));
        }
        None
    }

    /// Schachgebote um einen Halbzug verlängern, höchstens bis zur doppelten Iterationstiefe.
    fn extension(&self, in_check: bool, ply: usize) -> i32 {
        i32::from(in_check && self.opts.on(Feature::CheckExt) && ply_i32(ply) < 2 * self.root_depth)
    }

    /// Transpositionstabelle: Schnitt, wenn ein ausreichend tiefer Eintrag das Fenster entscheidet.
    /// Mit PVS werden Knoten der Hauptvariante nie abgeschnitten, damit die Variante vollständig bleibt.
    fn probe_tt(&self, node: Node, depth: i32, win: Window) -> TtProbe {
        let Some(entry) = self.tt.probe(node.key) else {
            return TtProbe::Hint { mv: 0, eval: None };
        };
        let protect_pv = win.is_pv() && self.opts.on(Feature::Pvs);
        if !node.is_root() && entry.depth >= depth && !protect_pv {
            let score = from_tt(entry.score, node.ply);
            let decides = match entry.bound {
                Bound::Exact => true,
                Bound::Lower => score >= win.beta,
                Bound::Upper => score <= win.alpha,
                Bound::None => false,
            };
            if decides {
                return TtProbe::Cutoff(score);
            }
        }
        TtProbe::Hint {
            mv: entry.mv,
            eval: entry.eval,
        }
    }

    /// Nullzug-Pruning, wo erlaubt; gibt den Schnittwert zurück.
    fn try_null_move(
        &mut self,
        pos: &Chess,
        depth: i32,
        win: Window,
        node: Node,
        static_eval: Option<i32>,
    ) -> Option<i32> {
        let allowed = self.opts.on(Feature::NullMove)
            && !win.is_pv()
            && node.can_null
            && depth >= NULL_MIN_DEPTH
            && !node.is_root()
            && static_eval.is_some_and(|se| se >= win.beta)
            && non_pawn_material(pos);
        if allowed {
            self.null_move(pos, depth, win.beta, node)
        } else {
            None
        }
    }

    /// Nullzug: Darf der Gegner zweimal ziehen und reicht es trotzdem für beta, gilt die Stellung als
    /// zu gut. Gibt den Schnittwert zurück oder `None`.
    fn null_move(&mut self, pos: &Chess, depth: i32, beta: i32, node: Node) -> Option<i32> {
        let child = pos.clone().swap_turn().ok()?;
        let r = NULL_REDUCTION + i32::from(depth >= NULL_DEEP_DEPTH);
        self.path.push(0);
        let key = hash(&child);
        let s = -self.negamax(
            &child,
            depth - 1 - r,
            Window::null(beta - 1).flip(),
            node.after_null_move(key),
        );
        self.path.pop();
        if self.aborted || s < beta {
            return None;
        }
        // kein ungeprüftes Matt aus einer Nullzug-Suche melden
        Some(if s >= MATE_BOUND { beta } else { s })
    }

    /// Futility: In geringer Resttiefe und weit unter alpha können ruhige Züge alpha nicht erreichen.
    pub(super) fn is_futile(&self, depth: i32, win: Window, static_eval: i32) -> bool {
        self.opts.on(Feature::Futility)
            && !win.is_pv()
            && depth <= FUTILITY_MAX_DEPTH
            && win.alpha.abs() < MATE_BOUND
            && static_eval + FUTILITY_MARGIN * depth <= win.alpha
    }

    /// Durchsucht alle Züge eines Knotens und hebt `win.alpha` an; `None` bei Abbruch.
    fn search_moves(&mut self, pos: &Chess, moves: &[Move], plan: &Plan, win: &mut Window) -> Option<Best> {
        let mut best = Best { score: -INF, code: 0 };
        let ordered = self.sorted(pos, moves, plan.tt_move, plan.node.ply);
        for (index, mv) in ordered.into_iter().enumerate() {
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let c = Candidate {
                mv,
                code: enc(mv),
                index,
                quiet: is_quiet(mv),
                gives_check: child.is_check(),
            };
            if plan.futile && c.index > 0 && c.quiet && !c.gives_check {
                continue;
            }
            let score = self.search_child(&child, &c, plan, *win);
            if self.aborted {
                return None;
            }
            if score <= best.score {
                continue;
            }
            best = Best { score, code: c.code };
            if plan.node.is_root() {
                self.root_best = Some((c.mv, score));
            }
            win.alpha = win.alpha.max(score);
            if win.alpha >= win.beta {
                if c.quiet {
                    self.reward_quiet(pos.turn(), c.code, plan.depth, plan.node.ply);
                }
                break;
            }
        }
        Some(best)
    }

    /// Wert eines Kindknotens aus Sicht des Ziehenden.
    fn search_child(&mut self, child: &Chess, c: &Candidate, plan: &Plan, win: Window) -> i32 {
        let key = hash(child);
        self.path.push(key);
        let node = plan.node.child(key);
        let score = if c.index == 0 {
            -self.negamax(child, plan.depth - 1, win.flip(), node)
        } else {
            self.search_late_move(child, c, plan, win, node)
        };
        self.path.pop();
        score
    }

    /// Späterer Zug: zuerst mit Null-Fenster (PVS) und eventuell reduziert (LMR) prüfen. Nur wenn er
    /// alpha übertrifft, ohne Reduktion und dann mit vollem Fenster nachsuchen.
    fn search_late_move(
        &mut self,
        child: &Chess,
        c: &Candidate,
        plan: &Plan,
        win: Window,
        node: Node,
    ) -> i32 {
        let depth = plan.depth - 1;
        let pvs = self.opts.on(Feature::Pvs);
        let probe = if pvs { Window::null(win.alpha) } else { win };
        let r = self.lmr_reduction(c, plan);
        let mut score = -self.negamax(child, depth - r, probe.flip(), node);
        if r > 0 && score > win.alpha && !self.aborted {
            score = -self.negamax(child, depth, probe.flip(), node);
        }
        if pvs && score > win.alpha && score < win.beta && !self.aborted {
            score = -self.negamax(child, depth, win.flip(), node);
        }
        score
    }

    /// Late Move Reduction: späte, ruhige Züge ohne Schach weniger tief durchsuchen.
    pub(super) fn lmr_reduction(&self, c: &Candidate, plan: &Plan) -> i32 {
        let reduce = self.opts.on(Feature::Lmr)
            && plan.depth >= LMR_MIN_DEPTH
            && c.index >= LMR_MIN_INDEX
            && c.quiet
            && !plan.in_check
            && !c.gives_check
            && !self.is_killer(plan.node.ply, c.code);
        if !reduce {
            return 0;
        }
        1 + i32::from(c.index >= LMR_DEEP_INDEX && plan.depth >= LMR_DEEP_DEPTH)
    }
}
