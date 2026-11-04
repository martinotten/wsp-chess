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

mod alphabeta;
mod node;
mod options;
mod ordering;
mod quiescence;
mod root;
mod score;
#[cfg(test)]
mod tests;

pub use options::{Feature, Info, Limits, Opts};

use crate::eval::Net;
use crate::tt::Tt;
use score::to_cp;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{Board, CastlingMode, Chess, Color, EnPassantMode, File, Move, Position, Square};
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
pub fn hash(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// Zug als 16 Bit: von | nach << 6 | Umwandlung << 12 (Rochade: Zielfeld des Königs). 0 = kein Zug.
#[must_use]
pub fn enc(m: Move) -> u16 {
    let (from, to) = match m {
        Move::Castle { king, rook } => {
            let f = if rook.file() > king.file() {
                File::G
            } else {
                File::C
            };
            (king, Square::from_coords(f, king.rank()))
        }
        _ => (m.from().unwrap_or(m.to()), m.to()),
    };
    let promo = m.promotion().map_or(0, u16::from);
    u16::from(from) | u16::from(to) << 6 | promo << 12
}

/// Zug in UCI-Schreibweise (Rochade als Königszug).
#[must_use]
pub fn uci(m: Move) -> String {
    UciMove::from_move(m, CastlingMode::Standard).to_string()
}

fn is_quiet(m: Move) -> bool {
    !m.is_capture() && !m.is_promotion()
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
    tt: Tt,
    /// Eingeschaltete Techniken.
    pub opts: Opts,
    stop: Arc<AtomicBool>,
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
    /// Neuer Sucher mit Transpositionstabelle von `hash_mb` MB; `stop` bricht eine Suche von außen ab.
    #[must_use]
    pub fn new(net: Arc<Net>, hash_mb: usize, stop: Arc<AtomicBool>) -> Searcher {
        let nt = net.n_targets();
        let nw = net.n_work();
        Searcher {
            net,
            tt: Tt::new(hash_mb),
            opts: Opts::default(),
            stop,
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
                    acc: vec![0.0; nt],
                })
                .collect(),
            work: vec![0.0; nw],
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
        self.killers = [[0; 2]; MAX_PLY];
        self.history.fill(0);
    }

    /// Statische Bewertung in cp, Sicht des Spielers am Zug. Der Akkumulator wird vom zuletzt auf
    /// dieser Höhe bewerteten Brett (meist ein Geschwisterknoten) oder vom zwei Halbzüge höheren
    /// (gleiche Perspektive) nachgeführt und nur bei großem Unterschied neu berechnet.
    pub fn eval(&mut self, pos: &Chess, ply: usize) -> i32 {
        self.evals += 1;
        let c = pos.turn();
        let ply = ply.min(MAX_PLY + 1);
        let net = &self.net;
        let (lo, hi) = self.slots.split_at_mut(ply);
        let slot = &mut hi[0];
        let mut done = false;
        if self.incremental {
            if let (Some(b), true) = (&slot.board, slot.persp == c) {
                done = net.update(b, pos.board(), c, &mut slot.acc, &mut self.diff);
            }
            if !done && ply >= 2 {
                let g = &lo[ply - 2];
                if let (Some(b), true) = (&g.board, g.persp == c) {
                    slot.acc.copy_from_slice(&g.acc);
                    done = net.update(b, pos.board(), c, &mut slot.acc, &mut self.diff);
                }
            }
        }
        if !done {
            net.refresh(pos.board(), c, &mut slot.acc);
        }
        slot.board = Some(pos.board().clone());
        slot.persp = c;
        let v = net.score_from(&slot.acc, &mut self.work);
        if let Some(worst) = self.check_incremental.as_mut() {
            *worst = worst.max((v - net.score(pos)).abs());
        }
        to_cp(v)
    }

    /// Zählt einen Knoten und prüft regelmäßig Stopp, Zeit und Knotenlimit.
    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes.trailing_zeros() >= TICK_SHIFT && self.should_stop() {
            self.aborted = true;
        }
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
            || self.max_nodes.is_some_and(|n| self.nodes >= n)
    }

    /// Betritt einen Knoten auf Höhe `ply`; `false`, wenn die Suche abgebrochen ist.
    fn enter(&mut self, ply: usize) -> bool {
        self.tick();
        self.seldepth = self.seldepth.max(ply);
        !self.aborted
    }

    /// Gleiche Stellung früher im Pfad (gleiche Seite am Zug), höchstens bis zum letzten
    /// irreversiblen Zug; 0 im Pfad markiert einen Nullzug (Grenze).
    fn is_repetition(&self, h: u64, halfmoves: u32) -> bool {
        let window = usize::try_from(halfmoves).unwrap_or(usize::MAX);
        self.path
            .iter()
            .rev()
            .skip(2)
            .step_by(2)
            .take(window / 2)
            .take_while(|&&x| x != 0)
            .any(|&x| x == h)
    }
}
