//! Incremental evaluation during a real search equals full recomputation.
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use wsp_chess::board::Position;
use wsp_chess::eval::Net;
use wsp_chess::search::{Limits, Searcher};

#[test]
fn incremental_equals_refresh() {
    let net = Arc::new(Net::load("weights/weights.bin").unwrap());
    let mut engine = Searcher::new(net, 16, Arc::new(AtomicBool::new(false)));
    engine.check_incremental = Some(0.0);
    for fen in [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "3qk3/2P5/8/8/8/8/7p/R3K3 w Q - 0 1",
        "8/2k5/3p4/p2P1p2/P2P1P2/8/3K4/8 w - - 0 1",
    ] {
        let position = Position::from_fen(fen).unwrap();
        engine.new_game();
        engine.search(
            &position,
            &[],
            Limits {
                depth: Some(5),
                ..Default::default()
            },
            |_| {},
        );
    }
    let worst = engine.check_incremental.unwrap();
    println!("max |Δs| = {worst:e}");
    assert!(worst < 1e-3, "max |Δs| = {worst}");
}
