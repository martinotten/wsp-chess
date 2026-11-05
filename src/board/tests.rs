//! Brettschicht: Zugarten, UCI, Schlüssel nach FIDE-Gleichheit, Remis durch Material.

use super::{Move, Piece, Position, Square};

fn pos(fen: &str) -> Position {
    Position::from_fen(fen).expect("gültige FEN")
}

fn play(position: &mut Position, moves: &str) {
    for text in moves.split_whitespace() {
        let mv = position.parse_uci(text).expect("legaler Zug");
        position.play_unchecked(mv);
    }
}

#[test]
fn castling_is_written_as_king_move_and_parsed_back() {
    let position = pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let castles: Vec<Move> = position
        .legal_moves()
        .into_iter()
        .filter(|mv| mv.is_castle())
        .collect();
    let mut texts: Vec<String> = castles.iter().map(|mv| mv.uci()).collect();
    texts.sort();
    assert_eq!(texts, ["e1c1", "e1g1"]);
    for mv in castles {
        assert_eq!(position.parse_uci(&mv.uci()), Some(mv));
        assert_eq!(
            (mv.from(), mv.role(), mv.capture()),
            (Square::E1, Piece::King, None)
        );
    }
}

#[test]
fn move_kinds_and_pieces() {
    let position = pos("4k3/1P6/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let ep = position.parse_uci("e5d6").expect("en passant");
    assert!(ep.is_en_passant() && ep.is_capture());
    assert_eq!(
        (ep.role(), ep.capture(), ep.to()),
        (Piece::Pawn, Some(Piece::Pawn), Square::D6)
    );
    let promotion = position.parse_uci("b7b8n").expect("Umwandlung");
    assert_eq!(
        (promotion.promotion(), promotion.is_capture()),
        (Some(Piece::Knight), false)
    );
    assert_eq!(promotion.uci(), "b7b8n");
    assert!(position.parse_uci("e5e7").is_none());
    assert!(position.parse_uci("xx").is_none());
}

#[test]
fn noisy_moves_are_the_captures_and_promotions() {
    for fen in [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "4k3/1P6/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    ] {
        let position = pos(fen);
        let legal = position.legal_moves();
        let noisy: Vec<String> = position.noisy_moves().iter().map(|mv| mv.uci()).collect();
        let expected: Vec<String> = legal
            .iter()
            .filter(|mv| mv.is_capture() || mv.is_promotion())
            .map(|mv| mv.uci())
            .collect();
        assert_eq!(sorted(noisy), sorted(expected), "{fen}");
    }
}

fn sorted(mut texts: Vec<String>) -> Vec<String> {
    texts.sort();
    texts
}

/// Nach einem Doppelschritt zählt en passant für den Schlüssel nur, wenn der Schlag legal ist.
#[test]
fn hash_ignores_en_passant_that_cannot_be_played() {
    // kein Bauer kann schlagen: gleich wie dieselbe Stellung ohne en-passant-Feld
    let mut pushed = pos("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    play(&mut pushed, "e2e4");
    assert_eq!(pushed.hash(), pos("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1").hash());
    // gefesselter Bauer (Turm auf a4 fesselt durch den eigenen König auf h4) darf nicht schlagen
    let mut pinned = pos("8/8/8/8/R4p1k/8/4P3/4K3 w - - 0 1");
    play(&mut pinned, "e2e4");
    assert!(pinned.parse_uci("f4e3").is_none());
    assert_eq!(pinned.hash(), pos("8/8/8/8/R3Pp1k/8/8/4K3 b - - 0 1").hash());
    // ausführbarer Schlag: anderer Schlüssel
    let mut open = pos("4k3/8/8/8/5p2/8/4P3/4K3 w - - 0 1");
    play(&mut open, "e2e4");
    assert!(open.parse_uci("f4e3").is_some());
    assert_ne!(open.hash(), pos("4k3/8/8/8/4Pp2/8/8/4K3 b - - 0 1").hash());
}

#[test]
fn hash_is_the_same_after_moving_back_and_forth() {
    let mut position = Position::default();
    let start = position.hash();
    play(&mut position, "g1f3 g8f6 f3g1 f6g8");
    assert_eq!(position.hash(), start);
    play(&mut position, "g1f3");
    assert_ne!(position.hash(), start);
}

#[test]
fn insufficient_material() {
    for (fen, expected) in [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", true),
        ("4k3/8/8/8/8/8/8/4KN2 w - - 0 1", true),
        ("4k3/8/8/8/8/8/8/4KB2 w - - 0 1", true),
        ("4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1", true), // Läufer auf gleichfarbigen Feldern
        ("4k1b1/8/8/8/8/8/8/2B1K3 w - - 0 1", false), // verschiedenfarbig
        ("4kn2/8/8/8/8/8/8/4KN2 w - - 0 1", false),
        ("4k3/8/8/8/8/8/8/3NKN2 w - - 0 1", false),
        ("4kn2/8/8/8/8/8/8/4KB2 w - - 0 1", false),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", false),
        ("4k3/8/8/8/8/8/8/4KR2 w - - 0 1", false),
    ] {
        assert_eq!(pos(fen).is_insufficient_material(), expected, "{fen}");
    }
}

#[test]
fn check_mate_null_move_and_attackers() {
    let mated = pos("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1");
    assert!(mated.is_check() && mated.is_checkmate());
    assert!(mated.null_move().is_none());
    let quiet = Position::default();
    assert!(!quiet.is_check() && !quiet.is_checkmate());
    let swapped = quiet.null_move().expect("Nullzug");
    assert_ne!(swapped.turn(), quiet.turn());
    let board = quiet.board();
    let attackers = quiet.attackers(Square::F3, super::Color::White, board.occupied());
    assert_eq!(attackers.len(), 3); // Bauern e2, g2 und Springer g1
    assert!(Position::from_fen("not a fen").is_none());
}
