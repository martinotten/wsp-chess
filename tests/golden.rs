//! Behavioural equivalence of the search: node count, best move and score for the bench positions at fixed
//! depth, once with default settings and once per toggled technique.
//!
//! The search is deterministic at fixed depth. Any change that shifts even a single search decision
//! changes the node count. A pure refactoring must therefore pass this test unchanged.
//! After an intended behaviour change, rewrite it: `GOLDEN_UPDATE=1 cargo test --release --test golden`.
#![expect(clippy::unwrap_used, reason = "test: a failure should abort the test")]
use std::fmt::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use wsp_chess::board::Position;
use wsp_chess::eval::Net;
use wsp_chess::search::{self, Feature, Limits, Opts, Searcher};
use wsp_chess::uci::BENCH;

const GOLDEN: &str = "tests/golden.txt";
const DEPTH: i32 = 5;

/// Default, then each technique toggled (on -> off, `QChecks` off -> on).
fn configs() -> Vec<(String, Opts)> {
    let mut out = vec![("default".to_string(), Opts::default())];
    for &(feature, name) in &Feature::ALL {
        let mut opts = Opts::default();
        opts.set(feature, !opts.on(feature));
        out.push((format!("{name}={}", opts.on(feature)), opts));
    }
    out
}

fn fingerprint() -> String {
    let net = Arc::new(Net::load("weights/weights.bin").unwrap());
    let mut engine = Searcher::new(net, 16, Arc::new(AtomicBool::new(false)));
    let mut text = String::new();
    for (name, opts) in configs() {
        engine.opts = opts;
        for fen in BENCH {
            let pos = Position::from_fen(fen).unwrap();
            engine.new_game();
            let lim = Limits {
                depth: Some(DEPTH),
                ..Limits::default()
            };
            let (best, info) = engine.search(&pos, &[], lim, |_| {});
            let mv = best.map_or_else(|| "0000".to_string(), search::uci);
            writeln!(text, "{name} {fen} | {} {mv} {}", info.nodes, info.score).unwrap();
        }
    }
    text
}

#[test]
fn search_is_unchanged() {
    let now = fingerprint();
    if std::env::var_os("GOLDEN_UPDATE").is_some() {
        std::fs::write(GOLDEN, &now).unwrap();
        return;
    }
    let want = std::fs::read_to_string(GOLDEN).unwrap();
    for (expected, actual) in want.lines().zip(now.lines()) {
        assert_eq!(expected, actual, "search behaves differently than in {GOLDEN}");
    }
    assert_eq!(want.lines().count(), now.lines().count());
}
