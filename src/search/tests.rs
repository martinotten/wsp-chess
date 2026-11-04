//! Tests der Suchbausteine.
use super::alphabeta::{FUTILITY_MARGIN, FUTILITY_MAX_DEPTH};
use super::node::{Candidate, Node, Plan, Window};
use super::score::{bound_for, from_tt, mated_in, to_tt};
use super::*;
use crate::tt::Bound;
use shakmaty::fen::Fen;

fn pos(fen: &str) -> Chess {
    fen.parse::<Fen>()
        .unwrap()
        .into_position(CastlingMode::Standard)
        .unwrap()
}

fn searcher() -> Searcher {
    let net = Arc::new(Net::load("../weights/weights.bin").unwrap());
    Searcher::new(net, 1, Arc::new(AtomicBool::new(false)))
}

fn search_depth(s: &mut Searcher, p: &Chess, game: &[u64], depth: i32) -> (Option<Move>, Info) {
    let lim = Limits {
        depth: Some(depth),
        ..Limits::default()
    };
    s.search(p, game, lim, |_| {})
}

#[test]
fn mate_scores_survive_the_tt_at_any_height() {
    // Matt in 5 Halbzügen von der Wurzel, gefunden auf Höhe 3: im Knoten „matt in 2“
    let at_root = MATE - 5;
    assert_eq!(to_tt(at_root, 3), MATE - 2);
    assert_eq!(from_tt(to_tt(at_root, 3), 3), at_root);
    // derselbe Eintrag auf Höhe 7 erreicht: Matt in 9 von der Wurzel
    assert_eq!(from_tt(to_tt(at_root, 3), 7), MATE - 9);
    assert_eq!(from_tt(to_tt(-at_root, 3), 7), -(MATE - 9));
    assert_eq!(to_tt(123, 9), 123);
    assert_eq!(from_tt(-123, 9), -123);
}

#[test]
fn bound_follows_the_entry_window() {
    assert_eq!(bound_for(50, 0, 50), Bound::Lower);
    assert_eq!(bound_for(10, 0, 50), Bound::Exact);
    assert_eq!(bound_for(0, 0, 50), Bound::Upper);
}

#[test]
fn windows() {
    let w = Window::around(10, 5);
    assert_eq!(w, Window { alpha: 5, beta: 15 });
    assert_eq!(w.flip(), Window { alpha: -15, beta: -5 });
    assert!(w.is_pv());
    assert!(!Window::null(3).is_pv());
    assert_eq!(Window::FULL.flip(), Window::FULL);
}

#[test]
fn move_encoding() {
    let p = pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let castle = p.legal_moves().into_iter().find(|m| m.is_castle()).unwrap();
    // Rochade wird als Königszug kodiert (e1g1 bzw. e1c1)
    let (from, to) = (enc(castle) & 0x3f, (enc(castle) >> 6) & 0x3f);
    assert_eq!(from, u16::from(Square::E1));
    assert!(to == u16::from(Square::G1) || to == u16::from(Square::C1));
    assert!(uci(castle) == "e1g1" || uci(castle) == "e1c1");
    let p_promo = pos("8/4P3/8/8/8/8/k7/4K3 w - - 0 1");
    let promo = p_promo
        .legal_moves()
        .into_iter()
        .find(|m| m.promotion().is_some())
        .unwrap();
    assert_ne!(enc(promo) >> 12, 0);
}

#[test]
fn repetition_is_found_only_within_reversible_moves() {
    let mut s = searcher();
    s.path = vec![7, 1, 2, 3, 4];
    // gleiche Seite am Zug: Schlüssel 2 Halbzüge, 4 Halbzüge zurück
    assert!(s.is_repetition(2, 10));
    assert!(!s.is_repetition(3, 10), "andere Seite am Zug");
    assert!(!s.is_repetition(2, 1), "irreversibler Zug dazwischen");
    s.path = vec![2, 0, 9, 9];
    assert!(!s.is_repetition(2, 10), "Nullzug ist eine Grenze");
}

#[test]
fn fifty_move_rule_makes_the_search_a_draw() {
    let mut s = searcher();
    let p = pos("8/8/8/4k3/8/8/R7/4K3 w - - 99 80");
    let (_, info) = search_depth(&mut s, &p, &[], 3);
    assert_eq!(info.score, 0);
    let fresh = pos("8/8/8/4k3/8/8/R7/4K3 w - - 0 80");
    let (_, fresh_info) = search_depth(&mut s, &fresh, &[], 3);
    assert!(fresh_info.score > 300, "Turm mehr: {}", fresh_info.score);
}

#[test]
fn repetition_against_game_history_is_a_draw() {
    // Schwarz steht klar schlechter, kann aber durch Rückkehr in eine frühere Stellung remisieren
    let mut s = searcher();
    let p = pos("8/8/8/4k3/8/8/R7/4K3 w - - 60 80");
    let earlier: Vec<u64> = p
        .legal_moves()
        .into_iter()
        .map(|m| {
            let mut c = p.clone();
            c.play_unchecked(m);
            hash(&c)
        })
        .collect();
    // Jede Folgestellung kam schon vor: jeder Zug wiederholt, die Suche sieht Remis
    // (Abstand 2 Halbzüge zur Folgestellung: Füllwert 1 steht für die Stellungen mit Schwarz am Zug)
    let game: Vec<u64> = earlier.iter().flat_map(|&h| [1, h]).collect();
    let (_, info) = search_depth(&mut s, &p, &game, 3);
    assert_eq!(info.score, 0);
}

#[test]
fn stalemate_and_mate_are_leaves() {
    let mut s = searcher();
    let stalemate = pos("k7/8/1Q6/8/8/8/8/4K3 b - - 0 1");
    assert_eq!(s.leaf_score(&stalemate, &stalemate.legal_moves(), 3), Some(0));
    let mated = pos("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1");
    assert_eq!(s.leaf_score(&mated, &mated.legal_moves(), 3), Some(mated_in(3)));
    let normal = pos("k7/8/8/8/8/8/8/K7 w - - 0 1");
    assert_eq!(s.leaf_score(&normal, &normal.legal_moves(), 3), None);
}

#[test]
fn lmr_reduces_only_late_quiet_moves() {
    let s = searcher();
    let plan = Plan {
        node: Node::root(1),
        depth: 8,
        in_check: false,
        futile: false,
        tt_move: 0,
    };
    let p = pos("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let mv = p.legal_moves()[0];
    let cand = |index, quiet, gives_check| Candidate {
        mv,
        code: 1,
        index,
        quiet,
        gives_check,
    };
    assert_eq!(s.lmr_reduction(&cand(2, true, false), &plan), 0, "früher Zug");
    assert_eq!(s.lmr_reduction(&cand(3, true, false), &plan), 1);
    assert_eq!(
        s.lmr_reduction(&cand(6, true, false), &plan),
        2,
        "später Zug, große Tiefe"
    );
    assert_eq!(s.lmr_reduction(&cand(6, false, false), &plan), 0, "Schlag");
    assert_eq!(s.lmr_reduction(&cand(6, true, true), &plan), 0, "Schachgebot");
}

#[test]
fn futility_needs_low_depth_and_a_large_gap() {
    let s = searcher();
    let w = Window::null(0);
    assert!(s.is_futile(1, w, -FUTILITY_MARGIN));
    assert!(!s.is_futile(1, w, -FUTILITY_MARGIN + 1));
    assert!(!s.is_futile(FUTILITY_MAX_DEPTH + 1, w, -10_000));
    assert!(
        !s.is_futile(1, Window::around(0, 50), -10_000),
        "nicht in der Hauptvariante"
    );
}

#[test]
fn node_limit_stops_the_search() {
    let mut s = searcher();
    let lim = Limits {
        nodes: Some(5_000),
        ..Limits::default()
    };
    let (best, info) = s.search(&Chess::default(), &[], lim, |_| {});
    assert!(best.is_some());
    assert!(info.nodes < 5_000 + (1 << TICK_SHIFT), "{}", info.nodes);
}
