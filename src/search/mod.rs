//! Alpha-Beta-Suche mit dem Handnetz als Bewertung.
//!
//! Werte in Centipawn (`100 · s`, `s` = Netzausgabe in Bauern), Sicht des Spielers am Zug.
//! Matt = ±(`MATE` − ply); in der Transpositionstabelle auf die Knotenhöhe normalisiert.
//! Jede Suchtechnik ist über [`Feature`] abschaltbar (UCI-Optionen), damit sie einzeln gemessen
//! werden kann.
//!
//! Aufbau (je Datei ein Teil von [`Searcher`]):
//! - `root`: iterative Vertiefung ([`Searcher::search`]) mit Aspiration-Fenstern, Hauptvariante
//! - `alphabeta`: Hauptsuche `negamax` – Endstellungen, Transpositionstabelle, Nullzug, Futility,
//!   dann die Züge mit PVS und LMR
//! - `quiescence`: Ruhesuche an den Blättern (Schläge, Umwandlungen, Schachabwehr)
//! - `ordering`: Zugsortierung (TT-Zug, MVV-LVA, Killer, History)
//! - `node`, `score`, `options`: Suchfenster und Knoten, Werteumrechnung, Schalter und Grenzen
//! - `parallel`: mehrere Suchthreads mit gemeinsamer Tabelle (Lazy SMP), [`search_parallel`]

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

/// Wert für „matt in 0 Halbzügen“.
pub const MATE: i32 = 30_000;

/// Werte ab hier (Betrag) sind Mattwerte.
pub const MATE_BOUND: i32 = 29_000;

/// Größer als jeder Wert.
pub const INF: i32 = 32_000;

/// Größte Suchhöhe.
pub const MAX_PLY: usize = 128;

/// Prüfintervall für Zeit, Knotenlimit und Stopp (Zweierpotenz-Exponent).
const TICK_SHIFT: u32 = 10;

/// Zobrist-Schlüssel der Stellung.
#[must_use]
pub fn hash(pos: &Position) -> u64 {
    pos.hash()
}

/// Zug als 16 Bit: von | nach << 6 | Umwandlung << 12 (Rochade: Zielfeld des Königs). 0 = kein Zug.
#[must_use]
pub fn move_code(mv: Move) -> u16 {
    let promo = mv.promotion().map_or(0, |piece| piece as u16);
    mv.from() as u16 | (mv.to() as u16) << 6 | promo << 12
}

/// Zug in UCI-Schreibweise (Rochade als Königszug).
#[must_use]
pub fn uci(mv: Move) -> String {
    mv.uci()
}

fn is_quiet(mv: Move) -> bool {
    !mv.is_capture() && !mv.is_promotion()
}

/// Akkumulator je Suchhöhe für die inkrementelle Bewertung.
struct Slot {
    board: Option<Board>,
    persp: Color,
    acc: Vec<f32>,
}

/// Zustand der Suche (Tabellen bleiben zwischen Zügen erhalten).
pub struct Searcher {
    net: Arc<Net>,
    tt: Arc<Tt>,
    /// Suchgeneration für die Ersetzung in der Tabelle; steigt je Suche, 0 nach [`Searcher::new_game`].
    age: u8,
    /// Eingeschaltete Techniken.
    pub opts: Opts,
    stop: Arc<AtomicBool>,
    /// Gemeinsamer Zustand der laufenden Suche (bei einem Thread nur dieser Sucher).
    team: Team,
    /// Helfer einer parallelen Suche: hört zusätzlich auf das Helferflag, gibt keinen Zug aus.
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
    /// Akkumulator nachführen statt neu berechnen.
    pub incremental: bool,
    /// Test: jede Bewertung mit Neuberechnung vergleichen und die größte Abweichung (Bauern) merken.
    pub check_incremental: Option<f32>,
}

impl Searcher {
    /// Neuer Sucher mit eigener Transpositionstabelle von `hash_mb` MB; `stop` bricht eine Suche von
    /// außen ab.
    #[must_use]
    pub fn new(net: Arc<Net>, hash_mb: usize, stop: Arc<AtomicBool>) -> Searcher {
        Searcher::with_table(net, Arc::new(Tt::new(hash_mb)), stop)
    }

    /// Neuer Sucher, der die Tabelle `tt` mit anderen Suchern teilt.
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

    /// Das Netz der Bewertung.
    #[must_use]
    pub fn net(&self) -> &Net {
        &self.net
    }

    /// Neue Partie: Tabellen leeren.
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.age = 0;
        self.killers = [[0; 2]; MAX_PLY];
        self.history.fill(0);
    }

    /// Statische Bewertung in cp, Sicht des Spielers am Zug. Der Akkumulator wird vom zuletzt auf
    /// dieser Höhe bewerteten Brett (meist ein Geschwisterknoten) oder vom zwei Halbzüge höheren
    /// (gleiche Perspektive) nachgeführt und nur bei großem Unterschied neu berechnet.
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

    /// Zählt einen Knoten und prüft regelmäßig Stopp, Zeit und Knotenlimit. Alle `2^TICK_SHIFT` Knoten
    /// meldet der Thread sie an das gemeinsame Budget; bei einem Thread ist dessen Stand dann genau
    /// `nodes`, bei mehreren wird das Limit um höchstens Threads × 1.024 Knoten überschritten.
    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes.trailing_zeros() >= TICK_SHIFT {
            let total = self.team.nodes.fetch_add(1 << TICK_SHIFT, Ordering::Relaxed) + (1 << TICK_SHIFT);
            if self.should_stop(total) {
                self.aborted = true;
            }
        }
    }

    /// Externer Stopp (UCI), Helferflag (nur Helfer), gemeinsame Deadline oder gemeinsames Knotenbudget.
    fn should_stop(&self, total_nodes: u64) -> bool {
        self.stop.load(Ordering::Relaxed)
            || (self.helper && self.team.halt.load(Ordering::Relaxed))
            || self.deadline.is_some_and(|deadline| Instant::now() >= deadline)
            || self.max_nodes.is_some_and(|max_nodes| total_nodes >= max_nodes)
    }

    /// Knoten aller Threads dieser Suche: gemeldetes Budget plus der noch nicht gemeldete Rest dieses
    /// Threads (bei einem Thread genau `nodes`).
    fn team_nodes(&self) -> u64 {
        self.team.nodes.load(Ordering::Relaxed) + (self.nodes & ((1 << TICK_SHIFT) - 1))
    }

    /// Betritt einen Knoten auf Höhe `ply`; `false`, wenn die Suche abgebrochen ist.
    fn enter(&mut self, ply: usize) -> bool {
        self.tick();
        self.seldepth = self.seldepth.max(ply);
        !self.aborted
    }

    /// Gleiche Stellung früher im Pfad (gleiche Seite am Zug), höchstens bis zum letzten
    /// irreversiblen Zug; 0 im Pfad markiert einen Nullzug (Grenze).
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
