//! UCI-Protokoll. Die Suche läuft in einem eigenen Thread, damit `stop` und `isready` jederzeit
//! beantwortet werden. Mit der Option `Threads` > 1 startet dieser Thread zusätzliche Helfer
//! (Lazy SMP, siehe `search::search_parallel`) und gibt `bestmove` erst aus, wenn alle beendet sind.
//!
//! Zusätzliche Befehle (nicht UCI): `eval` (statische Bewertung in Bauern), `perft <n>`
//! (Zuggenerator-Test), `bench [tiefe]` (feste Tiefe auf festen Stellungen: Knotenzahl und Tempo).

use crate::eval::Net;
use crate::search::{self, search_parallel, Feature, Limits, Opts, Searcher};
use crate::tt::Tt;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Position};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const NAME: &str = "WSP Chess Net (Rust core)";

/// Höchstzahl Suchthreads (12 Performance-Kerne des Entwicklungsrechners).
pub const MAX_THREADS: usize = 12;

/// Stellungen für `bench`.
pub const BENCH: [&str; 8] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 10",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2r3k1/5ppp/p3p3/1p1nP3/3P4/P4N2/1P3PPP/2R3K1 b - - 0 25",
    "8/2k5/3p4/p2P1p2/P2P1P2/8/3K4/8 w - - 0 1",
    "8/8/4k3/8/2R5/8/4K3/3r4 w - - 0 1",
    "6k1/5pp1/7p/8/8/1Q5P/5PP1/6K1 w - - 0 1",
];

/// Gemeinsame Ausgabe (Hauptthread und Suchthread).
pub type Out = Arc<Mutex<Box<dyn Write + Send>>>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn send(out: &Out, line: &str) {
    let mut writer = lock(out);
    // Ausgabefehler (GUI hat die Pipe geschlossen) sind nicht behebbar und werden ignoriert
    let _ = writeln!(writer, "{line}");
    let _ = writer.flush();
}

/// Gewichtsdatei im Projekt: `target/release/wsp-chess` -> `weights/weights.bin`.
fn default_weights() -> String {
    let mut dir: PathBuf = std::env::current_exe().unwrap_or_default();
    for _ in 0..4 {
        dir.pop();
        let candidate = dir.join("weights").join("weights.bin");
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "weights/weights.bin".into()
}

/// Liest eine Stellung aus FEN und optionaler Zugfolge.
fn parse_fen(fen: &str) -> Option<Chess> {
    fen.parse::<Fen>()
        .ok()?
        .into_position(CastlingMode::Standard)
        .ok()
}

/// `position startpos|fen ... [moves ...]` -> Stellung und Schlüssel der früheren Stellungen.
#[must_use]
pub fn parse_position(args: &[&str]) -> Option<(Chess, Vec<u64>)> {
    let (mut pos, rest) = match *args.first()? {
        "startpos" => (Chess::default(), &args[1..]),
        "fen" => {
            let moves_at = args.iter().position(|arg| *arg == "moves").unwrap_or(args.len());
            (parse_fen(&args[1..moves_at].join(" "))?, &args[moves_at..])
        }
        _ => return None,
    };
    let mut hist = vec![];
    if rest.first() == Some(&"moves") {
        for text in &rest[1..] {
            let mv = text.parse::<UciMove>().ok()?.to_move(&pos).ok()?;
            hist.push(search::hash(&pos));
            pos.play_unchecked(mv);
        }
    }
    Some((pos, hist))
}

fn ms(millis: i64) -> Duration {
    Duration::from_millis(u64::try_from(millis.max(1)).unwrap_or(1))
}

/// `go ...` -> Suchgrenzen. Zeitverwaltung: `movetime` − 5 ms; bei Uhrzeit
/// Rest/(movestogo oder 30) + 0,8·Inkrement, höchstens Rest/3 − 20 ms; ohne Angabe 1 s.
#[must_use]
pub fn parse_go(args: &[&str], turn: Color) -> Limits {
    let mut lim = Limits::default();
    let mut num: HashMap<&str, i64> = HashMap::new();
    let mut index = 0;
    while index < args.len() {
        if matches!(args[index], "infinite" | "ponder") {
            lim.infinite = true;
            index += 1;
        } else if let Some(value) = args.get(index + 1).and_then(|value| value.parse::<i64>().ok()) {
            num.insert(args[index], value);
            index += 2;
        } else {
            index += 1;
        }
    }
    lim.depth = num.get("depth").and_then(|&depth| i32::try_from(depth).ok());
    lim.nodes = num.get("nodes").and_then(|&nodes| u64::try_from(nodes).ok());
    if lim.infinite || lim.depth.is_some() || lim.nodes.is_some() {
        return lim;
    }
    let (time, inc) = if turn == Color::White {
        ("wtime", "winc")
    } else {
        ("btime", "binc")
    };
    lim.movetime = Some(if let Some(&movetime) = num.get("movetime") {
        ms(movetime - 5)
    } else if let Some(&left) = num.get(time) {
        let inc = num.get(inc).copied().unwrap_or(0);
        let togo = num.get("movestogo").copied().unwrap_or(30).max(1);
        ms((left / togo + inc * 4 / 5).min(left / 3) - 20)
    } else {
        ms(1000)
    });
    lim
}

/// UCI-Score: `cp x` oder `mate n`.
#[must_use]
pub fn score_str(score: i32) -> String {
    if score.abs() >= search::MATE_BOUND {
        let moves = (search::MATE - score.abs() + 1) / 2;
        format!("mate {}", if score > 0 { moves } else { -moves })
    } else {
        format!("cp {score}")
    }
}

/// Zahl der Blätter in Tiefe `d` (Zuggenerator-Test).
#[must_use]
pub fn perft(pos: &Chess, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = pos.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .into_iter()
        .map(|mv| {
            let mut child = pos.clone();
            child.play_unchecked(mv);
            perft(&child, depth - 1)
        })
        .sum()
}

fn per_second(count: u64, elapsed: Duration) -> u64 {
    let ms = elapsed.as_millis().max(1);
    u64::try_from(u128::from(count) * 1000 / ms).unwrap_or(u64::MAX)
}

/// Zustand der Engine zwischen den Befehlen.
pub struct Engine {
    out: Out,
    /// Ein Sucher je Thread (Index 0 = Hauptthread), gemeinsame Tabelle; leer = Netz nicht geladen.
    searchers: Arc<Mutex<Vec<Searcher>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    weights: String,
    hash_mb: usize,
    threads: usize,
    opts: Opts,
    pos: Chess,
    hist: Vec<u64>,
    infinite: bool,
}

impl Engine {
    /// Neue Engine, die nach `out` schreibt.
    #[must_use]
    pub fn new(out: Out) -> Engine {
        Engine {
            out,
            searchers: Arc::new(Mutex::new(Vec::new())),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            weights: default_weights(),
            hash_mb: 64,
            threads: 1,
            opts: Opts::default(),
            pos: Chess::default(),
            hist: vec![],
            infinite: false,
        }
    }

    fn send(&self, line: &str) {
        send(&self.out, line);
    }

    /// Lädt das Netz und legt je Thread einen Sucher an, falls nötig. Meldet Fehler als `info string`
    /// und gibt `false` zurück.
    fn ensure(&mut self) -> bool {
        let mut searchers = lock(&self.searchers);
        if searchers.is_empty() {
            match Net::load(&self.weights) {
                Ok(net) => {
                    let (net, tt) = (Arc::new(net), Arc::new(Tt::new(self.hash_mb)));
                    searchers.extend((0..self.threads).map(|_| {
                        Searcher::with_table(Arc::clone(&net), Arc::clone(&tt), Arc::clone(&self.stop))
                    }));
                }
                Err(err) => {
                    drop(searchers);
                    self.send(&format!("info string Fehler: {err}"));
                    return false;
                }
            }
        }
        for searcher in searchers.iter_mut() {
            searcher.opts = self.opts;
        }
        true
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            // Ein abgestürzter Suchthread hat nichts mehr auszugeben
            let _ = handle.join();
        }
    }

    /// Eingabeende: begrenzte Suche zu Ende laufen lassen, unendliche abbrechen.
    pub fn finish(&mut self) {
        if self.infinite {
            self.stop_search();
        } else if let Some(handle) = self.thread.take() {
            // Ein abgestürzter Suchthread hat nichts mehr auszugeben
            let _ = handle.join();
        }
    }

    fn uci(&self) {
        self.send(&format!("id name {NAME}"));
        self.send("id author hand-written weights (weight-space programming)");
        self.send(&format!(
            "option name Weights type string default {}",
            self.weights
        ));
        self.send("option name Hash type spin default 64 min 1 max 4096");
        self.send(&format!(
            "option name Threads type spin default 1 min 1 max {MAX_THREADS}"
        ));
        for (_, name) in Feature::ALL {
            self.send(&format!("option name {name} type check default true"));
        }
        self.send("uciok");
    }

    fn setoption(&mut self, args: &[&str]) {
        let start = args.iter().position(|arg| *arg == "name").map_or(0, |at| at + 1);
        let value_at = args.iter().position(|arg| *arg == "value").unwrap_or(args.len());
        let name = args[start.min(value_at)..value_at].join(" ");
        let value = args
            .get(value_at + 1..)
            .map(|words| words.join(" "))
            .unwrap_or_default();
        self.stop_search();
        if name.eq_ignore_ascii_case("weights") {
            self.weights = value;
            lock(&self.searchers).clear();
        } else if name.eq_ignore_ascii_case("hash") {
            self.hash_mb = value.parse().unwrap_or(64);
            lock(&self.searchers).clear();
        } else if name.eq_ignore_ascii_case("threads") {
            self.threads = value.parse().unwrap_or(1).clamp(1, MAX_THREADS);
            lock(&self.searchers).clear();
        } else if let Some(feature) = Feature::by_name(&name) {
            self.opts.set(feature, value.eq_ignore_ascii_case("true"));
        } else {
            self.send(&format!("info string unbekannte Option {name}"));
        }
    }

    fn go(&mut self, args: &[&str]) {
        self.stop_search();
        if !self.ensure() {
            return;
        }
        self.stop.store(false, Ordering::SeqCst);
        let lim = parse_go(args, self.pos.turn());
        self.infinite = lim.infinite;
        let (pos, hist) = (self.pos.clone(), self.hist.clone());
        let (shared, stop, out) = (
            Arc::clone(&self.searchers),
            Arc::clone(&self.stop),
            Arc::clone(&self.out),
        );
        self.thread = Some(thread::spawn(move || {
            let best = {
                let mut searchers = lock(&shared);
                let Some((main, helpers)) = searchers.split_first_mut() else {
                    return;
                };
                let (best, info) = search_parallel(main, helpers, &pos, &hist, lim, |progress| {
                    let pv: Vec<String> = progress.pv.iter().map(|&mv| search::uci(mv)).collect();
                    send(
                        &out,
                        &format!(
                            "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
                            progress.depth,
                            progress.seldepth,
                            score_str(progress.score),
                            progress.nodes,
                            per_second(progress.nodes, progress.time),
                            progress.hashfull,
                            progress.time.as_millis().max(1),
                            pv.join(" ")
                        ),
                    );
                });
                send(
                    &out,
                    &format!(
                        "info string evals {} nodes {} time {}",
                        info.evals,
                        info.nodes,
                        info.time.as_millis()
                    ),
                );
                best
            };
            while lim.infinite && !stop.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(2));
            }
            let best_move = best.map_or_else(|| "0000".to_string(), search::uci);
            send(&out, &format!("bestmove {best_move}"));
        }));
    }

    fn bench(&mut self, args: &[&str]) {
        let depth = args.first().and_then(|arg| arg.parse().ok()).unwrap_or(7);
        if !self.ensure() {
            return;
        }
        let (mut nodes, mut evals, started) = (0u64, 0u64, Instant::now());
        {
            let mut searchers = lock(&self.searchers);
            let Some(searcher) = searchers.first_mut() else {
                return;
            };
            for fen in BENCH {
                let Some(position) = parse_fen(fen) else { continue };
                searcher.new_game();
                self.stop.store(false, Ordering::SeqCst);
                let lim = Limits {
                    depth: Some(depth),
                    ..Limits::default()
                };
                let (_, info) = searcher.search(&position, &[], lim, |_| {});
                nodes += info.nodes;
                evals += info.evals;
            }
        }
        let elapsed = started.elapsed();
        self.send(&format!(
            "info string bench depth {depth} nodes {nodes} evals {evals} time {} nps {} eps {}",
            elapsed.as_millis(),
            per_second(nodes, elapsed),
            per_second(evals, elapsed)
        ));
    }

    /// Verarbeitet eine Eingabezeile. Gibt `false` bei `quit` zurück.
    pub fn command(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((&cmd, args)) = tokens.split_first() else {
            return true;
        };
        match cmd {
            "uci" => self.uci(),
            "setoption" => self.setoption(args),
            "isready" => {
                if self.thread.is_none() {
                    self.ensure();
                }
                self.send("readyok");
            }
            "ucinewgame" => {
                self.stop_search();
                if self.ensure() {
                    for searcher in lock(&self.searchers).iter_mut() {
                        searcher.new_game();
                    }
                }
                self.pos = Chess::default();
                self.hist.clear();
            }
            "position" => {
                self.stop_search();
                if let Some((position, history)) = parse_position(args) {
                    self.pos = position;
                    self.hist = history;
                } else {
                    self.send("info string ungültige Stellung");
                }
            }
            "go" => self.go(args),
            "stop" | "ponderhit" => self.stop_search(),
            "quit" => {
                self.stop_search();
                return false;
            }
            "eval" => self.static_eval(),
            "perft" => self.perft(args),
            "bench" => self.bench(args),
            _ => {}
        }
        true
    }

    /// `eval`: statische Bewertung der aktuellen Stellung in Bauern.
    fn static_eval(&mut self) {
        if !self.ensure() {
            return;
        }
        let score = lock(&self.searchers)
            .first()
            .map(|searcher| searcher.net().score(&self.pos));
        if let Some(score) = score {
            self.send(&format!("info string eval {score:.6}"));
        }
    }

    /// `perft [tiefe]`: Zahl der Blätter für den Zuggenerator-Test.
    fn perft(&self, args: &[&str]) {
        let depth = args.first().and_then(|arg| arg.parse().ok()).unwrap_or(1);
        let started = Instant::now();
        let leaves = perft(&self.pos, depth);
        self.send(&format!(
            "info string perft {depth} {leaves} {} ms",
            started.elapsed().as_millis()
        ));
    }
}

/// Liest Befehle bis `quit` oder Eingabeende.
pub fn run(input: impl BufRead, out: Out) {
    let mut engine = Engine::new(out);
    for line in input.lines() {
        let Ok(line) = line else { break };
        if !engine.command(&line) {
            return;
        }
    }
    engine.finish();
}
