//! Several search threads (Lazy SMP): all search the same position and share only the
//! transposition table. Move and `info` come from the main thread (deliberate restriction, the
//! usual Lazy SMP variant); the helper threads fill the table and thereby speed up its search.
//!
//! Stop signals with separate responsibilities:
//! - the external flag (`stop` of each searcher) is set only by the UCI input thread (`stop`, `quit`, new `go`);
//! - the helper flag [`Team::halt`] is set only by [`search_parallel`] once the main thread is done.

use super::{Info, Limits, Searcher};
use crate::board::{Move, Position};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// What all threads of a search share: start time (one deadline for all), node budget, table
/// generation and the helper flag.
#[derive(Clone, Debug)]
pub struct Team {
    pub(super) start: Instant,
    pub(super) nodes: Arc<AtomicU64>,
    pub(super) halt: Arc<AtomicBool>,
    pub(super) age: u8,
}

impl Team {
    /// New shared state from now on, generation `age`.
    pub(super) fn new(age: u8) -> Team {
        Team {
            start: Instant::now(),
            nodes: Arc::new(AtomicU64::new(0)),
            halt: Arc::new(AtomicBool::new(false)),
            age,
        }
    }
}

/// Lazy SMP: `main` searches and reports via `on_depth`, the `helpers` search the same position
/// (every second one a depth ahead) and act only through the table. All must have the same table
/// (`Searcher::with_table`), the same network and the same external stop flag.
///
/// Once the main thread is done, the helper flag is set and all helper threads are awaited; only
/// then does the function return. `Info.nodes` and `Info.evals` of the result count all threads.
pub fn search_parallel(
    main: &mut Searcher,
    helpers: &mut [Searcher],
    pos: &Position,
    game: &[u64],
    lim: Limits,
    on_depth: impl FnMut(&Info),
) -> (Option<Move>, Info) {
    let team = main.begin_team();
    let (best, mut info) = thread::scope(|scope| {
        for (index, helper) in helpers.iter_mut().enumerate() {
            let team = &team;
            scope.spawn(move || helper.search_helper(pos, game, lim, team, index + 1));
        }
        let result = main.search_main(pos, game, lim, &team, on_depth);
        team.halt.store(true, Ordering::Relaxed);
        result
    });
    info.nodes = main.nodes + helpers.iter().map(|helper| helper.nodes).sum::<u64>();
    info.evals = main.evals + helpers.iter().map(|helper| helper.evals).sum::<u64>();
    (best, info)
}
