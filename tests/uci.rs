//! UCI-Ablauf und Suchergebnisse.
#![expect(clippy::unwrap_used, reason = "Test: Fehlschlag soll den Test abbrechen")]
use std::io::{Cursor, Write};
use std::sync::{Arc, Mutex};
use wsp_chess::search::MATE;
use wsp_chess::uci::{parse_go, parse_position, run, score_str};

#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn session(input: &str) -> String {
    let buf = Buf::default();
    let out: Box<dyn Write + Send> = Box::new(buf.clone());
    let opts = "setoption name Weights value ../weights/weights.bin\n";
    run(Cursor::new(format!("{opts}{input}")), Arc::new(Mutex::new(out)));
    let bytes = buf.0.lock().unwrap().clone();
    String::from_utf8(bytes).unwrap()
}

fn bestmove(out: &str) -> &str {
    out.lines()
        .rev()
        .find_map(|l| l.strip_prefix("bestmove "))
        .unwrap()
}

#[test]
fn handshake_and_search() {
    let out = session("uci\nisready\nposition startpos moves e2e4\ngo depth 4\nisready\n");
    assert!(out.contains("uciok") && out.contains("readyok"));
    assert!(out.contains("info depth 4"), "{out}");
    let bm = bestmove(&out);
    assert_eq!(bm.len(), 4, "{out}");
}

#[test]
fn finds_mate_in_one_and_two() {
    let out = session("position fen 6k1/5ppp/8/8/8/8/1Q3PPP/1R4K1 w - - 0 1\ngo depth 3\nisready\n");
    assert_eq!(bestmove(&out), "b2b8", "{out}");
    assert!(out.contains("score mate 1"), "{out}");
    // Grundreihenmatt in 2: 1. Te8+ Txe8 2. Txe8#
    let out2 = session("position fen r5k1/5ppp/8/8/8/8/4RPPP/4R1K1 w - - 0 1\ngo depth 6\nisready\n");
    assert!(out2.contains("score mate 2"), "{out2}");
}

#[test]
fn stop_ends_infinite_search() {
    let out = session("position startpos\ngo infinite\nstop\n");
    assert_eq!(bestmove(&out).len(), 4, "{out}");
}

#[test]
fn time_management() {
    use shakmaty::Color;
    let l = parse_go(&["movetime", "100"], Color::White);
    assert_eq!(l.movetime.unwrap().as_millis(), 95);
    let clock = parse_go(&["wtime", "60000", "btime", "1000", "winc", "0"], Color::Black);
    assert!(clock.movetime.unwrap().as_millis() <= 1000 / 3);
    assert!(parse_go(&["infinite"], Color::White).infinite);
    assert_eq!(parse_go(&["depth", "6"], Color::White).depth, Some(6));
}

#[test]
fn position_history_and_scores() {
    let (_, hist) = parse_position(&["startpos", "moves", "g1f3", "g8f6", "f3g1", "f6g8"]).unwrap();
    assert_eq!(hist.len(), 4);
    assert_eq!(hist[0], wsp_chess::search::hash(&shakmaty::Chess::default()));
    assert!(parse_position(&["startpos", "moves", "e2e5"]).is_none());
    assert_eq!(score_str(MATE - 1), "mate 1");
    assert_eq!(score_str(-(MATE - 2)), "mate -1");
    assert_eq!(score_str(37), "cp 37");
}

#[test]
fn qchecks_is_opt_in() {
    use wsp_chess::search::{Feature, Opts};
    assert!(!Opts::default().on(Feature::QChecks));
    assert_eq!(Feature::by_name("qchecks"), Some(Feature::QChecks));
    let out = session("setoption name QChecks value true\nposition startpos\ngo depth 5\nisready\n");
    assert_eq!(bestmove(&out).len(), 4, "{out}");
}
