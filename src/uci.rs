//! UCI-Protokoll. Die Suche läuft in einem eigenen Thread, damit `stop` und `isready` jederzeit
//! beantwortet werden.
//!
//! Zusätzliche Befehle (nicht UCI): `eval` (statische Bewertung in Bauern), `perft <n>`
//! (Zuggenerator-Test), `bench [tiefe]` (feste Tiefe auf festen Stellungen: Knotenzahl und Tempo).

use crate::eval::Net;
use crate::search::{self, Feature, Limits, Opts, Searcher};
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

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn send(out: &Out, line: &str) {
    let mut o = lock(out);
    // Ausgabefehler (GUI hat die Pipe geschlossen) sind nicht behebbar und werden ignoriert
    let _ = writeln!(o, "{line}");
    let _ = o.flush();
}

/// Gewichtsdatei neben dem Projekt: `engine-rs/target/release/wsp-chess` -> `chess/weights.bin`.
fn default_weights() -> String {
    let mut p: PathBuf = std::env::current_exe().unwrap_or_default();
    for _ in 0..4 {
        p.pop();
        let c = p.join("weights.bin");
        if c.exists() {
            return c.to_string_lossy().into_owned();
        }
    }
    "weights.bin".into()
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
            let i = args.iter().position(|a| *a == "moves").unwrap_or(args.len());
            (parse_fen(&args[1..i].join(" "))?, &args[i..])
        }
        _ => return None,
    };
    let mut hist = vec![];
    if rest.first() == Some(&"moves") {
        for m in &rest[1..] {
            let mv = m.parse::<UciMove>().ok()?.to_move(&pos).ok()?;
            hist.push(search::hash(&pos));
            pos.play_unchecked(mv);
        }
    }
    Some((pos, hist))
}

fn ms(v: i64) -> Duration {
    Duration::from_millis(u64::try_from(v.max(1)).unwrap_or(1))
}

/// `go ...` -> Suchgrenzen. Zeitverwaltung: `movetime` − 5 ms; bei Uhrzeit
/// Rest/(movestogo oder 30) + 0,8·Inkrement, höchstens Rest/3 − 20 ms; ohne Angabe 1 s.
#[must_use]
pub fn parse_go(args: &[&str], turn: Color) -> Limits {
    let mut lim = Limits::default();
    let mut num: HashMap<&str, i64> = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if matches!(args[i], "infinite" | "ponder") {
            lim.infinite = true;
            i += 1;
        } else if let Some(v) = args.get(i + 1).and_then(|v| v.parse::<i64>().ok()) {
            num.insert(args[i], v);
            i += 2;
        } else {
            i += 1;
        }
    }
    lim.depth = num.get("depth").and_then(|&d| i32::try_from(d).ok());
    lim.nodes = num.get("nodes").and_then(|&n| u64::try_from(n).ok());
    if lim.infinite || lim.depth.is_some() || lim.nodes.is_some() {
        return lim;
    }
    let (time, inc) = if turn == Color::White {
        ("wtime", "winc")
    } else {
        ("btime", "binc")
    };
    lim.movetime = Some(if let Some(&mt) = num.get("movetime") {
        ms(mt - 5)
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
pub fn score_str(s: i32) -> String {
    if s.abs() >= search::MATE_BOUND {
        let n = (search::MATE - s.abs() + 1) / 2;
        format!("mate {}", if s > 0 { n } else { -n })
    } else {
        format!("cp {s}")
    }
}

/// Zahl der Blätter in Tiefe `d` (Zuggenerator-Test).
#[must_use]
pub fn perft(pos: &Chess, d: u32) -> u64 {
    if d == 0 {
        return 1;
    }
    let moves = pos.legal_moves();
    if d == 1 {
        return moves.len() as u64;
    }
    moves
        .into_iter()
        .map(|m| {
            let mut c = pos.clone();
            c.play_unchecked(m);
            perft(&c, d - 1)
        })
        .sum()
}

fn per_second(n: u64, t: Duration) -> u64 {
    let ms = t.as_millis().max(1);
    u64::try_from(u128::from(n) * 1000 / ms).unwrap_or(u64::MAX)
}

/// Zustand der Engine zwischen den Befehlen.
pub struct Engine {
    out: Out,
    searcher: Arc<Mutex<Option<Searcher>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    weights: String,
    hash_mb: usize,
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
            searcher: Arc::new(Mutex::new(None)),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            weights: default_weights(),
            hash_mb: 64,
            opts: Opts::default(),
            pos: Chess::default(),
            hist: vec![],
            infinite: false,
        }
    }

    fn send(&self, line: &str) {
        send(&self.out, line);
    }

    /// Lädt das Netz, falls nötig. Meldet Fehler als `info string` und gibt `false` zurück.
    fn ensure(&mut self) -> bool {
        let mut g = lock(&self.searcher);
        if g.is_none() {
            match Net::load(&self.weights) {
                Ok(net) => *g = Some(Searcher::new(Arc::new(net), self.hash_mb, Arc::clone(&self.stop))),
                Err(e) => {
                    drop(g);
                    self.send(&format!("info string Fehler: {e}"));
                    return false;
                }
            }
        }
        if let Some(s) = g.as_mut() {
            s.opts = self.opts;
        }
        true
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            // Ein abgestürzter Suchthread hat nichts mehr auszugeben
            let _ = t.join();
        }
    }

    /// Eingabeende: begrenzte Suche zu Ende laufen lassen, unendliche abbrechen.
    fn finish(&mut self) {
        if self.infinite {
            self.stop_search();
        } else if let Some(t) = self.thread.take() {
            // Ein abgestürzter Suchthread hat nichts mehr auszugeben
            let _ = t.join();
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
        for (_, n) in Feature::ALL {
            self.send(&format!("option name {n} type check default true"));
        }
        self.send("uciok");
    }

    fn setoption(&mut self, args: &[&str]) {
        let i = args.iter().position(|a| *a == "name").map_or(0, |i| i + 1);
        let j = args.iter().position(|a| *a == "value").unwrap_or(args.len());
        let name = args[i.min(j)..j].join(" ");
        let value = args.get(j + 1..).map(|v| v.join(" ")).unwrap_or_default();
        self.stop_search();
        if name.eq_ignore_ascii_case("weights") {
            self.weights = value;
            *lock(&self.searcher) = None;
        } else if name.eq_ignore_ascii_case("hash") {
            self.hash_mb = value.parse().unwrap_or(64);
            *lock(&self.searcher) = None;
        } else if let Some(f) = Feature::by_name(&name) {
            self.opts.set(f, value.eq_ignore_ascii_case("true"));
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
        let (sm, stop, out) = (
            Arc::clone(&self.searcher),
            Arc::clone(&self.stop),
            Arc::clone(&self.out),
        );
        self.thread = Some(thread::spawn(move || {
            let best = {
                let mut g = lock(&sm);
                let Some(s) = g.as_mut() else { return };
                let (best, info) = s.search(&pos, &hist, lim, |i| {
                    let pv: Vec<String> = i.pv.iter().map(|&m| search::uci(m)).collect();
                    send(
                        &out,
                        &format!(
                            "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
                            i.depth,
                            i.seldepth,
                            score_str(i.score),
                            i.nodes,
                            per_second(i.nodes, i.time),
                            i.hashfull,
                            i.time.as_millis().max(1),
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
            let bm = best.map_or_else(|| "0000".to_string(), search::uci);
            send(&out, &format!("bestmove {bm}"));
        }));
    }

    fn bench(&mut self, args: &[&str]) {
        let depth = args.first().and_then(|d| d.parse().ok()).unwrap_or(7);
        if !self.ensure() {
            return;
        }
        let (mut nodes, mut evals, started) = (0u64, 0u64, Instant::now());
        {
            let mut g = lock(&self.searcher);
            let Some(searcher) = g.as_mut() else {
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
        let el = started.elapsed();
        self.send(&format!(
            "info string bench depth {depth} nodes {nodes} evals {evals} time {} nps {} eps {}",
            el.as_millis(),
            per_second(nodes, el),
            per_second(evals, el)
        ));
    }

    /// Verarbeitet eine Eingabezeile. Gibt `false` bei `quit` zurück.
    pub fn command(&mut self, line: &str) -> bool {
        let tok: Vec<&str> = line.split_whitespace().collect();
        let Some((&cmd, args)) = tok.split_first() else {
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
                    if let Some(s) = lock(&self.searcher).as_mut() {
                        s.new_game();
                    }
                }
                self.pos = Chess::default();
                self.hist.clear();
            }
            "position" => {
                self.stop_search();
                if let Some((p, h)) = parse_position(args) {
                    self.pos = p;
                    self.hist = h;
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
            "eval" => {
                if self.ensure() {
                    let s = lock(&self.searcher).as_ref().map(|s| s.net().score(&self.pos));
                    if let Some(s) = s {
                        self.send(&format!("info string eval {s:.6}"));
                    }
                }
            }
            "perft" => {
                let d = args.first().and_then(|d| d.parse().ok()).unwrap_or(1);
                let t = Instant::now();
                let n = perft(&self.pos, d);
                self.send(&format!(
                    "info string perft {d} {n} {} ms",
                    t.elapsed().as_millis()
                ));
            }
            "bench" => self.bench(args),
            _ => {}
        }
        true
    }
}

/// Liest Befehle bis `quit` oder Eingabeende.
pub fn run(input: impl BufRead, out: Out) {
    let mut eng = Engine::new(out);
    for line in input.lines() {
        let Ok(line) = line else { break };
        if !eng.command(&line) {
            return;
        }
    }
    eng.finish();
}
