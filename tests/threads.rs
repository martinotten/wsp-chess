//! Multiple search threads and stop signals via UCI and directly via `search_parallel`.
#![expect(clippy::unwrap_used, reason = "test: a failure should abort the test")]
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use wsp_chess::board::Position;
use wsp_chess::eval::Net;
use wsp_chess::search::{search_parallel, Limits, Searcher};
use wsp_chess::tt::Tt;
use wsp_chess::uci::{parse_position, Engine};

#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl Write for Buf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Buf {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }

    fn bestmoves(&self) -> Vec<String> {
        self.text()
            .lines()
            .filter_map(|line| line.strip_prefix("bestmove ").map(str::to_string))
            .collect()
    }
}

/// Engine with `threads` search threads that writes to `buf`.
fn engine(threads: usize) -> (Engine, Buf) {
    let buf = Buf::default();
    let out: Box<dyn Write + Send> = Box::new(buf.clone());
    let mut engine = Engine::new(Arc::new(Mutex::new(out)));
    engine.command("setoption name Weights value weights/weights.bin");
    engine.command(&format!("setoption name Threads value {threads}"));
    (engine, buf)
}

fn is_legal(position_args: &str, text: &str) -> bool {
    let args: Vec<&str> = position_args.split_whitespace().collect();
    let (pos, _) = parse_position(&args).unwrap();
    pos.parse_uci(text).is_some()
}

const POSITION: &str = "startpos moves e2e4 e7e5 g1f3";

#[test]
fn threads_give_legal_moves_with_movetime() {
    for (threads, movetime) in [(4, 50), (3, 10), (1, 10)] {
        let (mut engine, buf) = engine(threads);
        engine.command(&format!("position {POSITION}"));
        engine.command(&format!("go movetime {movetime}"));
        engine.finish();
        let moves = buf.bestmoves();
        assert_eq!(moves.len(), 1, "{}", buf.text());
        assert!(is_legal(POSITION, &moves[0]), "{}", buf.text());
    }
}

#[test]
fn infinite_waits_for_stop_and_stop_is_fast() {
    let (mut engine, buf) = engine(4);
    engine.command(&format!("position {POSITION}"));
    engine.command("go infinite");
    thread::sleep(Duration::from_millis(300));
    assert!(buf.bestmoves().is_empty(), "bestmove before stop: {}", buf.text());
    let started = Instant::now();
    engine.command("stop");
    let latency = started.elapsed();
    println!("abort after stop: {latency:?}");
    let moves = buf.bestmoves();
    assert_eq!(moves.len(), 1);
    assert!(is_legal(POSITION, &moves[0]));
    // Target 5 ms (release, idle machine: 1-3 ms); generous limit so the test does not flake on a loaded
    // machine. Under coverage instrumentation (cargo llvm-cov) the timing is not meaningful.
    if !cfg!(coverage) {
        assert!(latency < Duration::from_millis(50), "{latency:?}");
    }
}

#[test]
fn immediate_stop_still_gives_a_legal_move() {
    let (mut engine, buf) = engine(3);
    engine.command(&format!("position {POSITION}"));
    engine.command("go infinite");
    engine.command("stop");
    let moves = buf.bestmoves();
    assert_eq!(moves.len(), 1);
    assert_ne!(moves[0], "0000");
    assert!(is_legal(POSITION, &moves[0]));
}

#[test]
fn new_go_quit_and_newgame_during_search() {
    let (mut engine, buf) = engine(4);
    engine.command(&format!("position {POSITION}"));
    engine.command("go infinite");
    engine.command("go depth 3");
    engine.command("ucinewgame");
    engine.command("position startpos");
    engine.command("go infinite");
    assert!(!engine.command("quit"));
    let moves = buf.bestmoves();
    assert_eq!(moves.len(), 3, "{}", buf.text());
    assert!(is_legal(POSITION, &moves[0]) && is_legal(POSITION, &moves[1]));
    assert!(is_legal("startpos", &moves[2]));
}

#[test]
fn node_budget_is_shared() {
    let (mut engine, buf) = engine(4);
    engine.command(&format!("position {POSITION}"));
    engine.command("go nodes 50000");
    engine.finish();
    let text = buf.text();
    let total: u64 = text
        .lines()
        .find_map(|line| line.strip_prefix("info string evals "))
        .and_then(|rest| rest.split_whitespace().nth(2))
        .unwrap()
        .parse()
        .unwrap();
    // Overshoot at most threads × 1,024 nodes (check interval) plus the remainder per thread
    assert!(total <= 50_000 + 4 * 2 * 1024, "{total} nodes: {text}");
    assert_eq!(buf.bestmoves().len(), 1);
}

/// External stop flag already set when the search starts: still a legal move (fallback), not `None`.
#[test]
fn stop_flag_set_before_the_search() {
    let net = Arc::new(Net::load("weights/weights.bin").unwrap());
    let tt = Arc::new(Tt::new(4));
    let stop = Arc::new(AtomicBool::new(true));
    let mut searchers: Vec<Searcher> = (0..3)
        .map(|_| Searcher::with_table(Arc::clone(&net), Arc::clone(&tt), Arc::clone(&stop)))
        .collect();
    let (main, helpers) = searchers.split_first_mut().unwrap();
    let pos = Position::default();
    let (best, _) = search_parallel(main, helpers, &pos, &[], Limits::default(), |_| {});
    let best = best.unwrap();
    assert!(pos.legal_moves().contains(&best));
}
