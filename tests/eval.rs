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
    for position in check["positions"].as_array().unwrap() {
        let score = f64::from(net.score(&pos(position["fen"].as_str().unwrap())));
        worst = worst.max((score - position["s"].as_f64().unwrap()).abs());
    }
    // Zugfolgen mit Sonderzügen: Stellungen durch Ziehen erzeugen (nicht aus FEN)
    for sequence in check["sequences"].as_array().unwrap() {
        let mut position = pos(sequence["fen"].as_str().unwrap());
        let soll: Vec<f64> = sequence["s"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        worst = worst.max((f64::from(net.score(&position)) - soll[0]).abs());
        for (ply, uci_text) in sequence["moves"].as_array().unwrap().iter().enumerate() {
            let uci_move: UciMove = uci_text.as_str().unwrap().parse().unwrap();
            let mv = uci_move.to_move(&position).unwrap();
            position.play_unchecked(mv);
            let deviation = (f64::from(net.score(&position)) - soll[ply + 1]).abs();
            assert!(
                deviation < 1e-3,
                "Folge {} Zug {ply}: |Δs| = {deviation}",
                sequence["fen"]
            );
            worst = worst.max(deviation);
        }
    }
    println!("max |Δs| = {worst:e}");
    assert!(worst < 1e-3, "max |Δs| = {worst}");
}

#[test]
fn see_values() {
    // Bauer schlägt gedeckten Springer: +300 - 100 = +200 (Springer wird geschlagen, Bauer zurück)
    let position = pos("4k3/8/2p5/3n4/4P3/8/8/4K3 w - - 0 1");
    let pawn_capture = position
        .legal_moves()
        .into_iter()
        .find(|candidate| search::uci(*candidate) == "e4d5")
        .unwrap();
    assert_eq!(see(&position, pawn_capture), 200);
    // Dame schlägt von Bauer gedeckten Bauern: 100 - 900 = -800
    let other_position = pos("4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1");
    let queen_capture = other_position
        .legal_moves()
        .into_iter()
        .find(|candidate| search::uci(*candidate) == "d2d5")
        .unwrap();
    assert_eq!(see(&other_position, queen_capture), -800);
}

#[test]
fn perft_reference() {
    assert_eq!(perft(&Chess::default(), 5), 4_865_609);
    let kiwipete = pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    assert_eq!(perft(&kiwipete, 4), 4_085_603);
}

#[test]
fn rejects_corrupt_weights() {
    let mut bytes = std::fs::read("../weights/weights.bin").unwrap();
    assert!(Net::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    bytes[0] = b'X';
    assert!(Net::from_bytes(&bytes).is_err());
}
