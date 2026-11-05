//! Root: iterative deepening with aspiration windows, fallback move and principal variation.

use super::alphabeta::FIFTY_MOVE_PLIES;
use super::node::{Node, Window};
use super::{hash, move_code, Feature, Info, Limits, Searcher, Team, INF, MATE, MATE_BOUND, MAX_PLY};
use crate::board::{Move, Position};

/// Maximum iteration depth.
const MAX_DEPTH: i32 = 100;

/// Aspiration: initial half window width, width from which the full window is used, minimum depth.
const ASPIRATION_DELTA: i32 = 40;
const ASPIRATION_MAX_DELTA: i32 = 1000;
const ASPIRATION_MIN_DEPTH: i32 = 4;

/// History: divisor at the start of a new search (older experience fades).
const HISTORY_DECAY: i32 = 8;

/// State of iterative deepening between two depths.
struct Deepening {
    /// Best move of the last completed depth (initially the fallback move).
    best: Move,
    /// Report of the last completed depth; its score is the centre of the next aspiration window.
    info: Info,
}

impl Searcher {
    /// Iterative deepening. `game` = keys of all earlier positions of the game (excluding the
    /// current one), for repetitions. `on_depth` is called after each completed iteration.
    pub fn search(
        &mut self,
        pos: &Position,
        game: &[u64],
        lim: Limits,
        on_depth: impl FnMut(&Info),
    ) -> (Option<Move>, Info) {
        let team = self.begin_team();
        self.search_main(pos, game, lim, &team, on_depth)
    }

    /// New search of this searcher as main thread: advance the generation, shared state from now on.
    pub(super) fn begin_team(&mut self) -> Team {
        self.age = self.age.wrapping_add(1);
        Team::new(self.age)
    }

    /// Iterative deepening of the main thread (returns move and report).
    pub(super) fn search_main(
        &mut self,
        pos: &Position,
        game: &[u64],
        lim: Limits,
        team: &Team,
        mut on_depth: impl FnMut(&Info),
    ) -> (Option<Move>, Info) {
        self.start_search(&lim, team, false);
        let root_key = self.begin_path(pos, game);
        let Some(mut state) = self.fallback(pos) else {
            return (None, Info::default());
        };
        for depth in 1..=Self::depth_limit(&lim) {
            if !self.complete_depth(pos, depth, root_key, &mut state) {
                break;
            }
            on_depth(&state.info);
            if self.deeper_search_pointless(&lim, &state.info) {
                break;
            }
        }
        self.finish(state)
    }

    /// Iterative deepening of a helper thread (number `index` ≥ 1): odd numbers start one depth
    /// deeper so that not all threads compute the same depth. Result only via the table.
    pub(super) fn search_helper(
        &mut self,
        pos: &Position,
        game: &[u64],
        lim: Limits,
        team: &Team,
        index: usize,
    ) {
        self.start_search(&lim, team, true);
        let root_key = self.begin_path(pos, game);
        let Some(mut state) = self.fallback(pos) else {
            return;
        };
        let first = 1 + i32::from(index % 2 == 1);
        for depth in first..=Self::depth_limit(&lim) {
            if !self.complete_depth(pos, depth, root_key, &mut state)
                || Self::mate_found(state.info.score, depth)
            {
                break;
            }
        }
    }

    fn start_search(&mut self, lim: &Limits, team: &Team, helper: bool) {
        self.team = team.clone();
        self.helper = helper;
        self.age = team.age;
        self.start = team.start;
        self.deadline = lim.movetime.map(|movetime| self.start + movetime);
        self.max_nodes = lim.nodes;
        self.nodes = 0;
        self.evals = 0;
        self.aborted = false;
        self.seldepth = 0;
        self.killers = [[0; 2]; MAX_PLY];
        for score in &mut self.history {
            *score /= HISTORY_DECAY;
        }
    }

    /// Search path for repetition detection: earlier positions of the game, then the root.
    fn begin_path(&mut self, pos: &Position, game: &[u64]) -> u64 {
        let root_key = hash(pos);
        self.path.clear();
        self.path.extend_from_slice(game);
        self.path.push(root_key);
        root_key
    }

    /// Initial state with the fallback move; `None` if there is no legal move.
    fn fallback(&mut self, pos: &Position) -> Option<Deepening> {
        let (best, score) = self.one_ply_best(pos)?;
        let info = Info {
            score,
            pv: vec![best],
            ..Info::default()
        };
        Some(Deepening { best, info })
    }

    /// Fallback in case not even depth 1 completes: best move according to the network (one ply).
    /// After the first move a legal move is guaranteed; from then on the stop is checked before
    /// each evaluation.
    fn one_ply_best(&mut self, pos: &Position) -> Option<(Move, i32)> {
        let mut best: Option<(Move, i32)> = None;
        for mv in pos.legal_moves() {
            if best.is_some() && self.should_stop(self.team_nodes()) {
                break;
            }
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = if child.is_checkmate() {
                MATE - 1
            } else {
                -self.eval(&child, 1)
            };
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((mv, score));
            }
        }
        best
    }

    fn depth_limit(lim: &Limits) -> i32 {
        lim.depth.unwrap_or(MAX_DEPTH).min(MAX_DEPTH)
    }

    /// Searches one depth and adopts its result; `false` if the search was aborted.
    fn complete_depth(&mut self, pos: &Position, depth: i32, root_key: u64, state: &mut Deepening) -> bool {
        self.root_depth = depth;
        self.root_best = None;
        let Some(score) = self.iterate(pos, depth, state.info.score, root_key) else {
            if depth == 1 {
                self.take_partial_first_depth(state);
            }
            return false;
        };
        if let Some((mv, _)) = self.root_best {
            state.best = mv;
        }
        state.info = self.iteration_info(pos, depth, score, state.best);
        true
    }

    /// One iteration with an aspiration window around `expected` (score of the previous depth);
    /// `None` on abort. If the score falls outside the window, the window is doubled on that side
    /// and fully opened from [`ASPIRATION_MAX_DELTA`] on.
    fn iterate(&mut self, pos: &Position, depth: i32, expected: i32, root_key: u64) -> Option<i32> {
        let mut delta = ASPIRATION_DELTA;
        let narrow =
            self.opts.on(Feature::Aspiration) && depth >= ASPIRATION_MIN_DEPTH && expected.abs() < MATE_BOUND;
        let mut win = if narrow {
            Window::around(expected, delta)
        } else {
            Window::FULL
        };
        loop {
            let score = self.negamax(pos, depth, win, Node::root(root_key));
            if self.aborted {
                return None;
            }
            if score <= win.alpha && win.alpha > -INF {
                win.alpha = (win.alpha - delta).max(-INF);
            } else if score >= win.beta && win.beta < INF {
                win.beta = (win.beta + delta).min(INF);
            } else {
                return Some(score);
            }
            delta *= 2;
            if delta > ASPIRATION_MAX_DELTA {
                win = Window::FULL;
            }
        }
    }

    /// Abort at depth 1: the move searched first is the network's best and replaces the fallback.
    fn take_partial_first_depth(&self, state: &mut Deepening) {
        if let Some((mv, score)) = self.root_best {
            state.best = mv;
            state.info.score = score;
            state.info.pv = vec![mv];
        }
    }

    fn iteration_info(&self, pos: &Position, depth: i32, score: i32, best: Move) -> Info {
        Info {
            depth,
            seldepth: self.seldepth,
            score,
            nodes: self.team_nodes(),
            evals: self.evals,
            time: self.start.elapsed(),
            pv: self.pv(pos, best, usize::try_from(depth).unwrap_or(0) + 4),
            hashfull: self.tt.hashfull(self.age),
        }
    }

    /// Principal variation from the transposition table, at most `max_len` moves. It ends before a
    /// position that already occurred in the game or the variation, and where the game would be
    /// drawn by the fifty-move rule or insufficient material.
    fn pv(&self, pos: &Position, first: Move, max_len: usize) -> Vec<Move> {
        let mut line = vec![first];
        let mut current = pos.clone();
        current.play_unchecked(first);
        // game history and root (self.path after the iteration): repetition against earlier moves too
        let mut seen = self.path.clone();
        seen.push(hash(pos));
        while line.len() < max_len {
            let key = hash(&current);
            if seen.contains(&key)
                || current.halfmoves() >= FIFTY_MOVE_PLIES
                || current.is_insufficient_material()
            {
                break;
            }
            seen.push(key);
            let Some(mv) = self
                .tt
                .probe(key)
                .and_then(|entry| Self::find(&current, entry.mv))
            else {
                break;
            };
            line.push(mv);
            current.play_unchecked(mv);
        }
        line
    }

    /// Legal move for the encoding `code` (see [`move_code`]); `None` for 0 or unknown moves.
    fn find(pos: &Position, code: u16) -> Option<Move> {
        if code == 0 {
            return None;
        }
        pos.legal_moves().into_iter().find(|&mv| move_code(mv) == code)
    }

    /// Another depth is pointless: mate found or not enough time left.
    fn deeper_search_pointless(&self, lim: &Limits, done: &Info) -> bool {
        Self::mate_found(done.score, done.depth) || self.time_for_next_depth_used(lim)
    }

    /// Mate found within the iteration depth: searching deeper changes nothing.
    fn mate_found(score: i32, depth: i32) -> bool {
        score.abs() >= MATE_BOUND && depth >= MATE - score.abs()
    }

    /// More than half of the thinking time is used up: the next depth would not complete.
    fn time_for_next_depth_used(&self, lim: &Limits) -> bool {
        lim.movetime
            .is_some_and(|movetime| !lim.infinite && self.start.elapsed() > movetime / 2)
    }

    /// Final report with the counters of the whole search.
    fn finish(&self, state: Deepening) -> (Option<Move>, Info) {
        let mut info = state.info;
        info.nodes = self.team_nodes();
        info.evals = self.evals;
        info.time = self.start.elapsed();
        (Some(state.best), info)
    }
}
