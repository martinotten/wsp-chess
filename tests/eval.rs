//! Rust evaluation against the Python reference values from `generator/export.py` (`weights/weights_check.json`,
//! frozen test data from the original project, no longer regenerated).
#![expect(clippy::unwrap_used, reason = "test: a failure should abort the test")]
use serde_json::Value;
use wsp_chess::board::Position;
use wsp_chess::eval::Net;
use wsp_chess::exchange::see;
use wsp_chess::search;
use wsp_chess::uci::perft;

fn net() -> Net {
    Net::load("weights/weights.bin").unwrap()
}

fn pos(fen: &str) -> Position {
    Position::from_fen(fen).unwrap()
}

#[test]
fn eval_matches_python() {
    let net = net();
    let check: Value =
        serde_json::from_str(&std::fs::read_to_string("weights/weights_check.json").unwrap()).unwrap();
    let mut worst: f64 = 0.0;
    for position in check["positions"].as_array().unwrap() {
        let score = f64::from(net.score(&pos(position["fen"].as_str().unwrap())));
        worst = worst.max((score - position["s"].as_f64().unwrap()).abs());
    }
    // Move sequences with special moves: create positions by playing moves (not from FEN)
    for sequence in check["sequences"].as_array().unwrap() {
        let mut position = pos(sequence["fen"].as_str().unwrap());
        let expected: Vec<f64> = sequence["s"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        worst = worst.max((f64::from(net.score(&position)) - expected[0]).abs());
        for (ply, uci_text) in sequence["moves"].as_array().unwrap().iter().enumerate() {
            let mv = position.parse_uci(uci_text.as_str().unwrap()).unwrap();
            position.play_unchecked(mv);
            let deviation = (f64::from(net.score(&position)) - expected[ply + 1]).abs();
            assert!(
                deviation < 1e-3,
                "sequence {} move {ply}: |Δs| = {deviation}",
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
    // Pawn captures a defended knight: +300 - 100 = +200 (knight is captured, pawn is recaptured)
    let position = pos("4k3/8/2p5/3n4/4P3/8/8/4K3 w - - 0 1");
    let pawn_capture = position
        .legal_moves()
        .into_iter()
        .find(|candidate| search::uci(*candidate) == "e4d5")
        .unwrap();
    assert_eq!(see(&position, pawn_capture), 200);
    // Queen captures a pawn defended by a pawn: 100 - 900 = -800
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
    assert_eq!(perft(&Position::default(), 5), 4_865_609);
    let kiwipete = pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    assert_eq!(perft(&kiwipete, 4), 4_085_603);
    // Positions 3 to 6 from the Chess Programming Wiki (Perft Results)
    for (fen, depth, nodes) in [
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674_624),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            4,
            422_333,
        ),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            4,
            2_103_487,
        ),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            4,
            3_894_594,
        ),
    ] {
        assert_eq!(perft(&pos(fen), depth), nodes, "{fen}");
    }
}

#[test]
fn rejects_corrupt_weights() {
    let mut bytes = std::fs::read("weights/weights.bin").unwrap();
    assert!(Net::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    bytes[0] = b'X';
    assert!(Net::from_bytes(&bytes).is_err());
}
