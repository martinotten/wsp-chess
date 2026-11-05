//! Brett, Züge und Regeln auf Grundlage von `cozy-chess` (MIT).
//!
//! Die Suche braucht von einem Zug mehr als `cozy-chess` speichert (bewegte und geschlagene Figur,
//! Rochade, en passant); [`Move`] hält das fest. Rochaden kodiert `cozy-chess` als „König schlägt
//! eigenen Turm“; nach außen (UCI, [`Move::to`]) gilt das Zielfeld des Königs.

use arrayvec::ArrayVec;
use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, BitBoard, Board,
    File, Rank,
};
pub use cozy_chess::{Color, Piece, Square};

/// Höchstzahl legaler Züge in einer Stellung (bekanntes Maximum 218).
const MAX_MOVES: usize = 256;

/// Art eines Zugs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Normal,
    EnPassant,
    Castle,
}

/// Ein legaler Zug mit bewegter und geschlagener Figur.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    raw: cozy_chess::Move,
    role: Piece,
    capture: Option<Piece>,
    kind: Kind,
}

impl Move {
    /// Ausgangsfeld.
    #[must_use]
    pub fn from(self) -> Square {
        self.raw.from
    }

    /// Zielfeld; bei der Rochade das Zielfeld des Königs.
    #[must_use]
    pub fn to(self) -> Square {
        if self.kind == Kind::Castle {
            let file = if self.raw.to.file() > self.raw.from.file() {
                File::G
            } else {
                File::C
            };
            Square::new(file, self.raw.from.rank())
        } else {
            self.raw.to
        }
    }

    /// Bewegte Figur.
    #[must_use]
    pub fn role(self) -> Piece {
        self.role
    }

    /// Geschlagene Figur (en passant: Bauer).
    #[must_use]
    pub fn capture(self) -> Option<Piece> {
        self.capture
    }

    /// Umwandlungsfigur.
    #[must_use]
    pub fn promotion(self) -> Option<Piece> {
        self.raw.promotion
    }

    /// Schlagzug (auch en passant).
    #[must_use]
    pub fn is_capture(self) -> bool {
        self.capture.is_some()
    }

    /// Umwandlung.
    #[must_use]
    pub fn is_promotion(self) -> bool {
        self.raw.promotion.is_some()
    }

    /// En-passant-Schlag.
    #[must_use]
    pub fn is_en_passant(self) -> bool {
        self.kind == Kind::EnPassant
    }

    /// Rochade.
    #[must_use]
    pub fn is_castle(self) -> bool {
        self.kind == Kind::Castle
    }

    /// UCI-Schreibweise, Rochade als Königszug (`e1g1`).
    #[must_use]
    pub fn uci(self) -> String {
        let mut text = format!("{}{}", self.from(), self.to());
        if let Some(piece) = self.promotion() {
            text.push(char::from(piece));
        }
        text
    }
}

/// Zugliste fester Größe (ohne Speicheranforderung und ohne Vorbelegung je Knoten).
pub type MoveList = ArrayVec<Move, MAX_MOVES>;

/// Zufällige, feste Schlüssel je en-passant-Linie für [`Position::hash`] (splitmix64 von 1..=8).
const EP_KEYS: [u64; 8] = {
    let mut keys = [0; 8];
    let mut index = 0;
    while index < 8 {
        let mut mixed = (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        keys[index] = mixed ^ (mixed >> 31);
        index += 1;
    }
    keys
};

/// Eine Stellung (Standardschach).
#[derive(Clone, Debug, Default)]
pub struct Position {
    board: Board,
}

impl Position {
    /// Stellung aus FEN; `None` bei ungültiger FEN oder illegaler Stellung.
    #[must_use]
    pub fn from_fen(fen: &str) -> Option<Position> {
        Board::from_fen(fen.trim(), false)
            .ok()
            .map(|board| Position { board })
    }

    /// Das Brett (Figurenstellung und Zustand).
    #[must_use]
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Seite am Zug.
    #[must_use]
    pub fn turn(&self) -> Color {
        self.board.side_to_move()
    }

    /// Halbzüge seit dem letzten Schlag oder Bauernzug.
    #[must_use]
    pub fn halfmoves(&self) -> u32 {
        u32::from(self.board.halfmove_clock())
    }

    /// Steht der König der Seite am Zug im Schach?
    #[must_use]
    pub fn is_check(&self) -> bool {
        !self.board.checkers().is_empty()
    }

    /// Matt: im Schach und ohne legalen Zug.
    #[must_use]
    pub fn is_checkmate(&self) -> bool {
        self.is_check() && !self.board.generate_moves(|_| true)
    }

    /// Alle legalen Züge.
    #[must_use]
    pub fn legal_moves(&self) -> MoveList {
        self.collect(|_| BitBoard::FULL, |_| true)
    }

    /// Alle legalen Schläge (auch en passant) und Umwandlungen, in einem Durchgang.
    #[must_use]
    pub fn noisy_moves(&self) -> MoveList {
        let them = self.board.colors(!self.turn());
        let ep = self.en_passant_square().map_or(BitBoard::EMPTY, Square::bitboard);
        let last_rank = Rank::Eighth.relative_to(self.turn()).bitboard();
        self.collect(
            |piece| {
                if piece == Piece::Pawn {
                    them | ep | last_rank
                } else {
                    them
                }
            },
            |mv| mv.is_capture() || mv.is_promotion(),
        )
    }

    /// Spielt einen legalen Zug (aus [`Position::legal_moves`] dieser Stellung).
    pub fn play_unchecked(&mut self, mv: Move) {
        self.board.play_unchecked(mv.raw);
    }

    /// Nullzug (Seite am Zug wechselt); `None` im Schach.
    #[must_use]
    pub fn null_move(&self) -> Option<Position> {
        self.board.null_move().map(|board| Position { board })
    }

    /// Kein Matt mehr möglich: weder Bauern noch Türme noch Damen, und höchstens eine Leichtfigur
    /// oder nur Läufer auf Feldern einer Farbe.
    #[must_use]
    pub fn is_insufficient_material(&self) -> bool {
        let board = &self.board;
        let heavy = board.pieces(Piece::Pawn) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
        if !heavy.is_empty() {
            return false;
        }
        let (knights, bishops) = (board.pieces(Piece::Knight), board.pieces(Piece::Bishop));
        let one_square_color =
            bishops.is_subset(BitBoard::DARK_SQUARES) || bishops.is_subset(BitBoard::LIGHT_SQUARES);
        (knights | bishops).len() <= 1 || (knights.is_empty() && one_square_color)
    }

    /// Zobrist-Schlüssel nach FIDE-Gleichheit: en passant zählt nur, wenn der Schlag legal möglich ist.
    #[must_use]
    pub fn hash(&self) -> u64 {
        let base = self.board.hash_without_ep();
        match self.legal_en_passant_file() {
            Some(file) => base ^ EP_KEYS[file as usize],
            None => base,
        }
    }

    /// Angreifer der Farbe `color` auf `target` bei Belegung `occupied` (für die Abtauschbewertung).
    #[must_use]
    pub fn attackers(&self, target: Square, color: Color, occupied: BitBoard) -> BitBoard {
        let board = &self.board;
        let diagonal = board.pieces(Piece::Bishop) | board.pieces(Piece::Queen);
        let straight = board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
        let attackers = (get_pawn_attacks(target, !color) & board.pieces(Piece::Pawn))
            | (get_knight_moves(target) & board.pieces(Piece::Knight))
            | (get_bishop_moves(target, occupied) & diagonal)
            | (get_rook_moves(target, occupied) & straight)
            | (get_king_moves(target) & board.pieces(Piece::King));
        attackers & board.colors(color)
    }

    /// Zug aus UCI-Text (`e2e4`, `e7e8q`, Rochade `e1g1`); `None`, wenn er nicht legal ist.
    #[must_use]
    pub fn parse_uci(&self, text: &str) -> Option<Move> {
        let raw: cozy_chess::Move = text.parse().ok()?;
        self.legal_moves()
            .into_iter()
            .find(|mv| mv.from() == raw.from && mv.to() == raw.to && mv.promotion() == raw.promotion)
    }

    /// Feld hinter einem gerade doppelt gezogenen Bauern.
    fn en_passant_square(&self) -> Option<Square> {
        let file = self.board.en_passant()?;
        Some(Square::new(file, Rank::Sixth.relative_to(self.turn())))
    }

    /// Linie eines legal ausführbaren en-passant-Schlags.
    fn legal_en_passant_file(&self) -> Option<File> {
        let target = self.en_passant_square()?;
        let us = self.turn();
        let pawns = get_pawn_attacks(target, !us) & self.board.colored_pieces(us, Piece::Pawn);
        pawns
            .into_iter()
            .any(|from| {
                self.board.is_legal(cozy_chess::Move {
                    from,
                    to: target,
                    promotion: None,
                })
            })
            .then(|| target.file())
    }

    /// Erzeugt legale Züge mit Zielfeldern aus `targets(Figur)` und behält die, die `keep` erfüllen.
    fn collect(&self, targets: impl Fn(Piece) -> BitBoard, keep: impl Fn(Move) -> bool) -> MoveList {
        let mut list = MoveList::new();
        self.board.generate_moves(|mut piece_moves| {
            piece_moves.to &= targets(piece_moves.piece);
            let role = piece_moves.piece;
            for raw in piece_moves {
                let mv = self.describe(raw, role);
                if keep(mv) {
                    list.push(mv);
                }
            }
            false
        });
        list
    }

    /// Ergänzt einen `cozy-chess`-Zug um Figur, Schlag und Art.
    fn describe(&self, raw: cozy_chess::Move, role: Piece) -> Move {
        let us = self.turn();
        let (kind, capture) = if self.board.colors(us).has(raw.to) {
            (Kind::Castle, None)
        } else if self.board.colors(!us).has(raw.to) {
            (Kind::Normal, self.board.piece_on(raw.to))
        } else if role == Piece::Pawn && raw.from.file() != raw.to.file() {
            (Kind::EnPassant, Some(Piece::Pawn))
        } else {
            (Kind::Normal, None)
        };
        Move {
            raw,
            role,
            capture,
            kind,
        }
    }
}

#[cfg(test)]
mod tests;
