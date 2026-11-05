//! Tests der Suchbausteine.
use super::alphabeta::{FUTILITY_MARGIN, FUTILITY_MAX_DEPTH};
use super::node::{Candidate, Node, Plan, Window};
use super::score::{bound_for, from_tt, mated_in, to_tt};
use super::*;
use crate::board::Square;
use crate::tt::Bound;

fn pos(fen: &str) -> Position {
    Position::from_fen(fen).unwrap()
}

fn searcher() -> Searcher {
    let net = Arc::new(Net::load("weights/weights.bin").unwrap());
    Searcher::new(net, 1, Arc::new(AtomicBool::new(false)))
}

fn search_depth(
    engine: &mut Searcher,
    position: &Position,
    game: &[u64],
    depth: i32,
) -> (Option<Move>, Info) {
    let lim = Limits {
        depth: Some(depth),
        ..Limits::default()
    };
    engine.search(position, game, lim, |_| {})
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
    let win = Window::around(10, 5);
    assert_eq!(win, Window { alpha: 5, beta: 15 });
    assert_eq!(win.flip(), Window { alpha: -15, beta: -5 });
    assert!(win.is_pv());
    assert!(!Window::null(3).is_pv());
    assert_eq!(Window::FULL.flip(), Window::FULL);
}

#[test]
fn move_encoding() {
    let position = pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let castle = position
        .legal_moves()
        .into_iter()
        .find(|mv| mv.is_castle())
        .unwrap();
    // Rochade wird als Königszug kodiert (e1g1 bzw. e1c1)
    let (from, to) = (move_code(castle) & 0x3f, (move_code(castle) >> 6) & 0x3f);
    assert_eq!(from, Square::E1 as u16);
    assert!(to == Square::G1 as u16 || to == Square::C1 as u16);
    assert!(uci(castle) == "e1g1" || uci(castle) == "e1c1");
    let p_promo = pos("8/4P3/8/8/8/8/k7/4K3 w - - 0 1");
    let promo = p_promo
        .legal_moves()
        .into_iter()
        .find(|mv| mv.promotion().is_some())
        .unwrap();
    assert_ne!(move_code(promo) >> 12, 0);
}

#[test]
fn repetition_is_found_only_within_reversible_moves() {
    let mut engine = searcher();
    engine.path = vec![7, 1, 2, 3, 4];
    // gleiche Seite am Zug: Schlüssel 2 Halbzüge, 4 Halbzüge zurück
    assert!(engine.is_repetition(2, 10));
    assert!(!engine.is_repetition(3, 10), "andere Seite am Zug");
    assert!(!engine.is_repetition(2, 1), "irreversibler Zug dazwischen");
    engine.path = vec![2, 0, 9, 9];
    assert!(!engine.is_repetition(2, 10), "Nullzug ist eine Grenze");
}

#[test]
fn fifty_move_rule_makes_the_search_a_draw() {
    let mut engine = searcher();
    let position = pos("8/8/8/4k3/8/8/R7/4K3 w - - 99 80");
    let (_, info) = search_depth(&mut engine, &position, &[], 3);
    assert_eq!(info.score, 0);
    assert_eq!(info.pv.len(), 1, "Hauptvariante endet an der 50-Züge-Grenze");
    let fresh = pos("8/8/8/4k3/8/8/R7/4K3 w - - 0 80");
    let (_, fresh_info) = search_depth(&mut engine, &fresh, &[], 3);
    assert!(fresh_info.score > 300, "Turm mehr: {}", fresh_info.score);
}

#[test]
fn insufficient_material_is_a_draw_and_mate_beats_the_fifty_move_rule() {
    let mut engine = searcher();
    // König und Läufer gegen König: Remis, egal wie tief
    let (_, info) = search_depth(&mut engine, &pos("8/8/8/4k3/8/8/2B5/4K3 w - - 0 1"), &[], 4);
    assert_eq!(info.score, 0);
    // Ta8 setzt matt und erreicht zugleich die 50-Züge-Grenze: Matt hat Vorrang
    let (best, mate_info) = search_depth(&mut engine, &pos("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80"), &[], 3);
    assert_eq!(best.map(uci).as_deref(), Some("a1a8"));
    assert!(mate_info.score > MATE_BOUND, "{}", mate_info.score);
}

#[test]
fn repetition_against_game_history_is_a_draw() {
    // Schwarz steht klar schlechter, kann aber durch Rückkehr in eine frühere Stellung remisieren
    let mut engine = searcher();
    let position = pos("8/8/8/4k3/8/8/R7/4K3 w - - 60 80");
    let earlier: Vec<u64> = position
        .legal_moves()
        .into_iter()
        .map(|mv| {
            let mut child = position.clone();
            child.play_unchecked(mv);
            hash(&child)
        })
        .collect();
    // Jede Folgestellung kam schon vor: jeder Zug wiederholt, die Suche sieht Remis
    // (Abstand 2 Halbzüge zur Folgestellung: Füllwert 1 steht für die Stellungen mit Schwarz am Zug)
    let game: Vec<u64> = earlier.iter().flat_map(|&key| [1, key]).collect();
    let (_, info) = search_depth(&mut engine, &position, &game, 3);
    assert_eq!(info.score, 0);
    assert_eq!(info.pv.len(), 1, "Hauptvariante endet vor der Wiederholung");
}

#[test]
fn stalemate_and_mate_are_leaves() {
    let mut engine = searcher();
    let stalemate = pos("k7/8/1Q6/8/8/8/8/4K3 b - - 0 1");
    assert_eq!(
        engine.leaf_score(&stalemate, &stalemate.legal_moves(), 3),
        Some(0)
    );
    let mated = pos("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1");
    assert_eq!(
        engine.leaf_score(&mated, &mated.legal_moves(), 3),
        Some(mated_in(3))
    );
    let normal = pos("k7/8/8/8/8/8/8/K7 w - - 0 1");
    assert_eq!(engine.leaf_score(&normal, &normal.legal_moves(), 3), None);
}

#[test]
fn lmr_reduces_only_late_quiet_moves() {
    let engine = searcher();
    let plan = Plan {
        node: Node::root(1),
        depth: 8,
        in_check: false,
        futile: false,
        tt_move: 0,
    };
    let position = pos("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let mv = position.legal_moves()[0];
    let cand = |index, quiet, gives_check| Candidate {
        mv,
        code: 1,
        index,
        quiet,
        gives_check,
    };
    assert_eq!(
        engine.lmr_reduction(&cand(2, true, false), &plan),
        0,
        "früher Zug"
    );
    assert_eq!(engine.lmr_reduction(&cand(3, true, false), &plan), 1);
    assert_eq!(
        engine.lmr_reduction(&cand(6, true, false), &plan),
        2,
        "später Zug, große Tiefe"
    );
    assert_eq!(engine.lmr_reduction(&cand(6, false, false), &plan), 0, "Schlag");
    assert_eq!(
        engine.lmr_reduction(&cand(6, true, true), &plan),
        0,
        "Schachgebot"
    );
}

#[test]
fn futility_needs_low_depth_and_a_large_gap() {
    let engine = searcher();
    let win = Window::null(0);
    assert!(engine.is_futile(1, win, -FUTILITY_MARGIN));
    assert!(!engine.is_futile(1, win, -FUTILITY_MARGIN + 1));
    assert!(!engine.is_futile(FUTILITY_MAX_DEPTH + 1, win, -10_000));
    assert!(
        !engine.is_futile(1, Window::around(0, 50), -10_000),
        "nicht in der Hauptvariante"
    );
}

#[test]
fn node_limit_stops_the_search() {
    let mut engine = searcher();
    let lim = Limits {
        nodes: Some(5_000),
        ..Limits::default()
    };
    let (best, info) = engine.search(&Position::default(), &[], lim, |_| {});
    assert!(best.is_some());
    assert!(info.nodes < 5_000 + (1 << TICK_SHIFT), "{}", info.nodes);
}
