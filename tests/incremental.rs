//! Inkrementelle Bewertung während einer echten Suche = Neuberechnung.
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use wsp_chess::eval::Net;
use wsp_chess::search::{Limits, Searcher};

#[test]
fn incremental_equals_refresh() {
    let net = Arc::new(Net::load("../weights.bin").unwrap());
    let mut s = Searcher::new(net, 16, Arc::new(AtomicBool::new(false)));
    s.check_incremental = Some(0.0);
    for fen in [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "3qk3/2P5/8/8/8/8/7p/R3K3 w Q - 0 1",
        "8/2k5/3p4/p2P1p2/P2P1P2/8/3K4/8 w - - 0 1",
    ] {
        let p: Chess = fen
            .parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap();
        s.new_game();
        s.search(
            &p,
            &[],
            Limits {
                depth: Some(5),
                ..Default::default()
            },
            |_| {},
        );
    }
    let worst = s.check_incremental.unwrap();
    println!("max |Δs| = {worst:e}");
    assert!(worst < 1e-3, "max |Δs| = {worst}");
}
