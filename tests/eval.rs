//! Rust-Bewertung gegen die Python-Sollwerte aus `export_rs.py` (`weights_check.json`).
#![expect(clippy::unwrap_used, reason = "Test: Fehlschlag soll den Test abbrechen")]
use serde_json::Value;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};
use wsp_chess::eval::Net;
use wsp_chess::exchange::see;
use wsp_chess::search;
use wsp_chess::uci::perft;

fn net() -> Net {
    Net::load("../weights/weights.bin").unwrap()
}

fn pos(fen: &str) -> Chess {
    fen.parse::<Fen>()
        .unwrap()
        .into_position(CastlingMode::Standard)
        .unwrap()
}

#[test]
fn eval_matches_python() {
    let net = net();
    let check: Value =
        serde_json::from_str(&std::fs::read_to_string("../weights/weights_check.json").unwrap()).unwrap();
    let mut worst: f64 = 0.0;
    for p in check["positions"].as_array().unwrap() {
        let s = f64::from(net.score(&pos(p["fen"].as_str().unwrap())));
        worst = worst.max((s - p["s"].as_f64().unwrap()).abs());
    }
    // Zugfolgen mit Sonderzügen: Stellungen durch Ziehen erzeugen (nicht aus FEN)
    for sq in check["sequences"].as_array().unwrap() {
        let mut p = pos(sq["fen"].as_str().unwrap());
        let soll: Vec<f64> = sq["s"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        worst = worst.max((f64::from(net.score(&p)) - soll[0]).abs());
        for (k, m) in sq["moves"].as_array().unwrap().iter().enumerate() {
            let u: UciMove = m.as_str().unwrap().parse().unwrap();
            let mv = u.to_move(&p).unwrap();
            p.play_unchecked(mv);
            let d = (f64::from(net.score(&p)) - soll[k + 1]).abs();
            assert!(d < 1e-3, "Folge {} Zug {k}: |Δs| = {d}", sq["fen"]);
            worst = worst.max(d);
        }
    }
    println!("max |Δs| = {worst:e}");
    assert!(worst < 1e-3, "max |Δs| = {worst}");
}

#[test]
fn see_values() {
    // Bauer schlägt gedeckten Springer: +300 - 100 = +200 (Springer wird geschlagen, Bauer zurück)
    let p = pos("4k3/8/2p5/3n4/4P3/8/8/4K3 w - - 0 1");
    let m = p
        .legal_moves()
        .into_iter()
        .find(|mv| search::uci(*mv) == "e4d5")
        .unwrap();
    assert_eq!(see(&p, m), 200);
    // Dame schlägt von Bauer gedeckten Bauern: 100 - 900 = -800
    let q = pos("4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1");
    let qm = q
        .legal_moves()
        .into_iter()
        .find(|mv| search::uci(*mv) == "d2d5")
        .unwrap();
    assert_eq!(see(&q, qm), -800);
}

#[test]
fn perft_reference() {
    assert_eq!(perft(&Chess::default(), 5), 4_865_609);
    let kiwipete = pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    assert_eq!(perft(&kiwipete, 4), 4_085_603);
}

#[test]
fn rejects_corrupt_weights() {
    let mut b = std::fs::read("../weights/weights.bin").unwrap();
    assert!(Net::from_bytes(&b[..b.len() - 1]).is_err());
    b[0] = b'X';
    assert!(Net::from_bytes(&b).is_err());
}
