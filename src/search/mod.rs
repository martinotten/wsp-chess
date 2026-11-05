//! Alpha-beta search with the hand-written network as evaluation.
//!
//! Scores in centipawns (`100 · s`, `s` = network output in pawns), from the side to move's point
//! of view. Mate = ±(`MATE` − ply); normalised to the node's ply in the transposition table.
//! Every search technique can be switched off via [`Feature`] (UCI options) so it can be measured
//! on its own.
//!
//! Structure (each file is one part of [`Searcher`]):
//! - `root`: iterative deepening ([`Searcher::search`]) with aspiration windows, principal variation
//! - `alphabeta`: main search `negamax` – terminal positions, transposition table, null move,
//!   futility, then the moves with PVS and LMR
//! - `quiescence`: quiescence search at the leaves (captures, promotions, check evasions)
//! - `ordering`: move ordering (TT move, MVV-LVA, killers, history)
//! - `node`, `score`, `options`: search window and nodes, score conversion, switches and limits
//! - `parallel`: several search threads sharing one table (Lazy SMP), [`search_parallel`]

mod alphabeta;
mod node;
mod options;
mod ordering;
mod parallel;
mod quiescence;
mod root;
mod score;
#[cfg(test)]
mod tests;

pub use options::{Feature, Info, Limits, Opts};
pub use parallel::{search_parallel, Team};

use crate::board::{Color, Move, Position};
use crate::eval::Net;
use crate::tt::Tt;
use cozy_chess::Board;
use score::to_cp;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Score for "mate in 0 plies".
pub const MATE: i32 = 30_000;

/// Scores from here on (absolute value) are mate scores.
pub const MATE_BOUND: i32 = 29_000;

/// Greater than any score.
pub const INF: i32 = 32_000;

/// Maximum search ply.
pub const MAX_PLY: usize = 128;

/// Check interval for time, node limit and stop (power-of-two exponent).
const TICK_SHIFT: u32 = 10;

/// Zobrist key of the position.
#[must_use]
pub fn hash(pos: &Position) -> u64 {
    pos.hash()
}

/// Move as 16 bits: from | to << 6 | promotion << 12 (castling: the king's target square). 0 = no
/// move.
#[must_use]
pub fn move_code(mv: Move) -> u16 {
    let promo = mv.promotion().map_or(0, |piece| piece as u16);
    mv.from() as u16 | (mv.to() as u16) << 6 | promo << 12
}

/// Move in UCI notation (castling as a king move).
#[must_use]
pub fn uci(mv: Move) -> String {
    mv.uci()
}

fn is_quiet(mv: Move) -> bool {
    !mv.is_capture() && !mv.is_promotion()
}

/// Accumulator per search ply for incremental evaluation.
struct Slot {
    board: Option<Board>,
    persp: Color,
    acc: Vec<f32>,
}

/// Search state (tables are kept between moves).
pub struct Searcher {
    net: Arc<Net>,
    tt: Arc<Tt>,
    /// Search generation for replacement in the table; increases with each search, 0 after
    /// [`Searcher::new_game`].
    age: u8,
    /// Enabled techniques.
    pub opts: Opts,
    stop: Arc<AtomicBool>,
    /// Shared state of the running search (with a single thread, only this searcher).
    team: Team,
    /// Helper thread of a parallel search: additionally obeys the helper flag, reports no move.
    helper: bool,
    nodes: u64,
    evals: u64,
    start: Instant,
    deadline: Option<Instant>,
    max_nodes: Option<u64>,
    aborted: bool,
    seldepth: usize,
    root_depth: i32,
    killers: [[u16; 2]; MAX_PLY],
    history: Vec<i32>,
    path: Vec<u64>,
    root_best: Option<(Move, i32)>,
    slots: Vec<Slot>,
    work: Vec<f32>,
    diff: Vec<(usize, f32)>,
    /// Update the accumulator incrementally instead of recomputing it.
    pub incremental: bool,
    /// Test: compare every evaluation with a full recomputation and record the largest deviation
    /// (pawns).
    pub check_incremental: Option<f32>,
}

impl Searcher {
    /// New searcher with its own transposition table of `hash_mb` MB; `stop` aborts a search from
    /// outside.
    #[must_use]
    pub fn new(net: Arc<Net>, hash_mb: usize, stop: Arc<AtomicBool>) -> Searcher {
        Searcher::with_table(net, Arc::new(Tt::new(hash_mb)), stop)
    }

    /// New searcher that shares the table `tt` with other searchers.
    #[must_use]
    pub fn with_table(net: Arc<Net>, tt: Arc<Tt>, stop: Arc<AtomicBool>) -> Searcher {
        let n_targets = net.n_targets();
        let n_work = net.n_work();
        Searcher {
            net,
            tt,
            age: 0,
            opts: Opts::default(),
            stop,
            team: Team::new(0),
            helper: false,
            nodes: 0,
            evals: 0,
            start: Instant::now(),
            deadline: None,
            max_nodes: None,
            aborted: false,
            seldepth: 0,
            root_depth: 0,
            killers: [[0; 2]; MAX_PLY],
            history: vec![0; 2 * 4096],
            path: Vec::with_capacity(1024),
            root_best: None,
            slots: (0..MAX_PLY + 2)
                .map(|_| Slot {
                    board: None,
                    persp: Color::White,
                    acc: vec![0.0; n_targets],
                })
                .collect(),
            work: vec![0.0; n_work],
            diff: Vec::with_capacity(32),
            incremental: true,
            check_incremental: None,
        }
    }

    /// The evaluation network.
    #[must_use]
    pub fn net(&self) -> &Net {
        &self.net
    }

    /// New game: clear the tables.
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.age = 0;
        self.killers = [[0; 2]; MAX_PLY];
        self.history.fill(0);
    }

    /// Static evaluation in cp, from the side to move's point of view. The accumulator is updated
    /// from the board last evaluated at this ply (usually a sibling node) or from the one two plies
    /// higher up (same perspective), and only recomputed when the difference is large.
    pub fn eval(&mut self, pos: &Position, ply: usize) -> i32 {
        self.evals += 1;
        let persp = pos.turn();
        let ply = ply.min(MAX_PLY + 1);
        let net = &self.net;
        let (above, from_ply) = self.slots.split_at_mut(ply);
        let slot = &mut from_ply[0];
        let mut done = false;
        if self.incremental {
            if let (Some(board), true) = (&slot.board, slot.persp == persp) {
                done = net.update(board, pos.board(), persp, &mut slot.acc, &mut self.diff);
            }
            if !done && ply >= 2 {
                let grandparent = &above[ply - 2];
                if let (Some(board), true) = (&grandparent.board, grandparent.persp == persp) {
                    slot.acc.copy_from_slice(&grandparent.acc);
                    done = net.update(board, pos.board(), persp, &mut slot.acc, &mut self.diff);
                }
            }
        }
        if !done {
            net.refresh(pos.board(), persp, &mut slot.acc);
        }
        slot.board = Some(pos.board().clone());
        slot.persp = persp;
        let score = net.score_from(&slot.acc, &mut self.work);
        if let Some(worst) = self.check_incremental.as_mut() {
            *worst = worst.max((score - net.score(pos)).abs());
        }
        to_cp(score)
    }

    /// Counts a node and regularly checks stop, time and node limit. Every `2^TICK_SHIFT` nodes the
    /// thread reports them to the shared budget; with a single thread its count is then exactly
    /// `nodes`, with several threads the limit is exceeded by at most threads × 1,024 nodes.
    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes.trailing_zeros() >= TICK_SHIFT {
            let total = self.team.nodes.fetch_add(1 << TICK_SHIFT, Ordering::Relaxed) + (1 << TICK_SHIFT);
            if self.should_stop(total) {
                self.aborted = true;
            }
        }
    }

    /// External stop (UCI), helper flag (helper threads only), shared deadline or shared node budget.
    fn should_stop(&self, total_nodes: u64) -> bool {
        self.stop.load(Ordering::Relaxed)
            || (self.helper && self.team.halt.load(Ordering::Relaxed))
            || self.deadline.is_some_and(|deadline| Instant::now() >= deadline)
            || self.max_nodes.is_some_and(|max_nodes| total_nodes >= max_nodes)
    }

    /// Nodes of all threads of this search: reported budget plus this thread's not yet reported
    /// remainder (with a single thread exactly `nodes`).
    fn team_nodes(&self) -> u64 {
        self.team.nodes.load(Ordering::Relaxed) + (self.nodes & ((1 << TICK_SHIFT) - 1))
    }

    /// Enters a node at ply `ply`; `false` if the search has been aborted.
    fn enter(&mut self, ply: usize) -> bool {
        self.tick();
        self.seldepth = self.seldepth.max(ply);
        !self.aborted
    }

    /// Same position earlier in the path (same side to move), at most back to the last
    /// irreversible move; a 0 in the path marks a null move (boundary).
    fn is_repetition(&self, key: u64, halfmoves: u32) -> bool {
        let window = usize::try_from(halfmoves).unwrap_or(usize::MAX);
        self.path
            .iter()
            .rev()
            .skip(2)
            .step_by(2)
            .take(window / 2)
            .take_while(|&&earlier| earlier != 0)
            .any(|&earlier| earlier == key)
    }
}
