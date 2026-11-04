//! Verhaltensgleichheit der Suche: Knotenzahl, bester Zug und Wert für die Bench-Stellungen bei fester
//! Tiefe, einmal mit Standardeinstellungen und einmal je umgeschalteter Technik.
//!
//! Die Suche ist bei fester Tiefe deterministisch. Jede Änderung, die auch nur eine Suchentscheidung
//! verschiebt, ändert die Knotenzahl. Ein reiner Umbau muss diesen Test also unverändert bestehen.
//! Bei gewollter Verhaltensänderung neu schreiben: `GOLDEN_UPDATE=1 cargo test --release --test golden`.
#![expect(clippy::unwrap_used, reason = "Test: Fehlschlag soll den Test abbrechen")]
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};
use std::fmt::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use wsp_chess::eval::Net;
use wsp_chess::search::{self, Feature, Limits, Opts, Searcher};
use wsp_chess::uci::BENCH;

const GOLDEN: &str = "tests/golden.txt";
const DEPTH: i32 = 5;

/// Standard, jede Technik umgeschaltet (an -> aus, `QChecks` aus -> an).
fn configs() -> Vec<(String, Opts)> {
    let mut out = vec![("default".to_string(), Opts::default())];
    for &(f, name) in &Feature::ALL {
        let mut o = Opts::default();
        o.set(f, !o.on(f));
        out.push((format!("{name}={}", o.on(f)), o));
    }
    out
}

fn fingerprint() -> String {
    let net = Arc::new(Net::load("../weights/weights.bin").unwrap());
    let mut s = Searcher::new(net, 16, Arc::new(AtomicBool::new(false)));
    let mut text = String::new();
    for (name, opts) in configs() {
        s.opts = opts;
        for fen in BENCH {
            let pos: Chess = fen
                .parse::<Fen>()
                .unwrap()
                .into_position(CastlingMode::Standard)
                .unwrap();
            s.new_game();
            let lim = Limits {
                depth: Some(DEPTH),
                ..Limits::default()
            };
            let (best, info) = s.search(&pos, &[], lim, |_| {});
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
    for (a, b) in want.lines().zip(now.lines()) {
        assert_eq!(a, b, "Suche verhält sich anders als in {GOLDEN}");
    }
    assert_eq!(want.lines().count(), now.lines().count());
}
