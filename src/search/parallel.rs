//! Mehrere Suchthreads (Lazy SMP): Alle durchsuchen dieselbe Stellung und teilen nur die
//! Transpositionstabelle. Zug und `info` kommen vom Hauptthread (bewusste Einschränkung, übliche
//! Lazy-SMP-Variante); die Helfer füllen die Tabelle und beschleunigen so dessen Suche.
//!
//! Stoppsignale mit getrennten Zuständigkeiten:
//! - das externe Flag (`stop` jedes Suchers) setzt nur der UCI-Eingabethread (`stop`, `quit`, neues `go`);
//! - das Helferflag [`Team::halt`] setzt nur [`search_parallel`], sobald der Hauptthread fertig ist.

use super::{Info, Limits, Searcher};
use crate::board::{Move, Position};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Was alle Threads einer Suche teilen: Startzeit (eine Deadline für alle), Knotenbudget, Generation
/// der Tabelle und das Helferflag.
#[derive(Clone, Debug)]
pub struct Team {
    pub(super) start: Instant,
    pub(super) nodes: Arc<AtomicU64>,
    pub(super) halt: Arc<AtomicBool>,
    pub(super) age: u8,
}

impl Team {
    /// Neuer gemeinsamer Zustand ab jetzt, Generation `age`.
    pub(super) fn new(age: u8) -> Team {
        Team {
            start: Instant::now(),
            nodes: Arc::new(AtomicU64::new(0)),
            halt: Arc::new(AtomicBool::new(false)),
            age,
        }
    }
}

/// Lazy SMP: `main` sucht und berichtet über `on_depth`, die `helpers` suchen dieselbe Stellung
/// (jeder zweite eine Tiefe voraus) und wirken nur über die Tabelle. Alle müssen dieselbe Tabelle
/// (`Searcher::with_table`), dasselbe Netz und dasselbe externe Stoppflag haben.
///
/// Ist der Hauptthread fertig, wird das Helferflag gesetzt und auf alle Helfer gewartet; erst dann
/// kehrt die Funktion zurück. `Info.nodes` und `Info.evals` des Ergebnisses zählen alle Threads.
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
