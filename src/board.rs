//! Board, moves and rules based on `cozy-chess` (MIT).
//!
//! The search needs more information about a move than `cozy-chess` stores (moved and captured
//! piece, castling, en passant); [`Move`] records it. `cozy-chess` encodes castling as "king captures
//! own rook"; externally (UCI, [`Move::to`]) the king's target square is used.

use arrayvec::ArrayVec;
use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, BitBoard, Board,
    File, Rank,
};
pub use cozy_chess::{Color, Piece, Square};

/// Maximum number of legal moves in a position (known maximum 218).
const MAX_MOVES: usize = 256;

/// Kind of a move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Normal,
    EnPassant,
    Castle,
}

/// A legal move with moved and captured piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    raw: cozy_chess::Move,
    role: Piece,
    capture: Option<Piece>,
    kind: Kind,
}

impl Move {
    /// Origin square.
    #[must_use]
    pub fn from(self) -> Square {
        self.raw.from
    }

    /// Target square; for castling, the king's target square.
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

    /// Moved piece.
    #[must_use]
    pub fn role(self) -> Piece {
        self.role
    }

    /// Captured piece (en passant: pawn).
    #[must_use]
    pub fn capture(self) -> Option<Piece> {
        self.capture
    }

    /// Promotion piece.
    #[must_use]
    pub fn promotion(self) -> Option<Piece> {
        self.raw.promotion
    }

    /// Capture (including en passant).
    #[must_use]
    pub fn is_capture(self) -> bool {
        self.capture.is_some()
    }

    /// Promotion.
    #[must_use]
    pub fn is_promotion(self) -> bool {
        self.raw.promotion.is_some()
    }

    /// En passant capture.
    #[must_use]
    pub fn is_en_passant(self) -> bool {
        self.kind == Kind::EnPassant
    }

    /// Castling.
    #[must_use]
    pub fn is_castle(self) -> bool {
        self.kind == Kind::Castle
    }

    /// UCI notation, castling as a king move (`e1g1`).
    #[must_use]
    pub fn uci(self) -> String {
        let mut text = format!("{}{}", self.from(), self.to());
        if let Some(piece) = self.promotion() {
            text.push(char::from(piece));
        }
        text
    }
}

/// Fixed-size move list (no heap allocation and no initialization per node).
pub type MoveList = ArrayVec<Move, MAX_MOVES>;

/// Random, fixed keys per en passant file for [`Position::hash`] (splitmix64 of 1..=8).
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

/// A position (standard chess).
#[derive(Clone, Debug, Default)]
pub struct Position {
    board: Board,
}

impl Position {
    /// Position from FEN; `None` for an invalid FEN or an illegal position.
    #[must_use]
    pub fn from_fen(fen: &str) -> Option<Position> {
        Board::from_fen(fen.trim(), false)
            .ok()
            .map(|board| Position { board })
    }

    /// The board (piece placement and state).
    #[must_use]
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Side to move.
    #[must_use]
    pub fn turn(&self) -> Color {
        self.board.side_to_move()
    }

    /// Plies since the last capture or pawn move.
    #[must_use]
    pub fn halfmoves(&self) -> u32 {
        u32::from(self.board.halfmove_clock())
    }

    /// Is the king of the side to move in check?
    #[must_use]
    pub fn is_check(&self) -> bool {
        !self.board.checkers().is_empty()
    }

    /// Checkmate: in check and without a legal move.
    #[must_use]
    pub fn is_checkmate(&self) -> bool {
        self.is_check() && !self.board.generate_moves(|_| true)
    }

    /// All legal moves.
    #[must_use]
    pub fn legal_moves(&self) -> MoveList {
        self.collect(|_| BitBoard::FULL, |_| true)
    }

    /// All legal captures (including en passant) and promotions, in a single pass.
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

    /// Plays a legal move (from [`Position::legal_moves`] of this position).
    pub fn play_unchecked(&mut self, mv: Move) {
        self.board.play_unchecked(mv.raw);
    }

    /// Null move (side to move switches); `None` when in check.
    #[must_use]
    pub fn null_move(&self) -> Option<Position> {
        self.board.null_move().map(|board| Position { board })
    }

    /// Checkmate is no longer possible: no pawns, rooks or queens, and at most one minor piece or
    /// only bishops on squares of one color.
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

    /// Zobrist key by FIDE position equality: en passant only counts if the capture is legal.
    #[must_use]
    pub fn hash(&self) -> u64 {
        let base = self.board.hash_without_ep();
        match self.legal_en_passant_file() {
            Some(file) => base ^ EP_KEYS[file as usize],
            None => base,
        }
    }

    /// Attackers of color `color` on `target` with occupancy `occupied` (for static exchange evaluation).
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

    /// Move from UCI text (`e2e4`, `e7e8q`, castling `e1g1`); `None` if it is not legal.
    #[must_use]
    pub fn parse_uci(&self, text: &str) -> Option<Move> {
        let raw: cozy_chess::Move = text.parse().ok()?;
        self.legal_moves()
            .into_iter()
            .find(|mv| mv.from() == raw.from && mv.to() == raw.to && mv.promotion() == raw.promotion)
    }

    /// Square behind a pawn that has just made a double step.
    fn en_passant_square(&self) -> Option<Square> {
        let file = self.board.en_passant()?;
        Some(Square::new(file, Rank::Sixth.relative_to(self.turn())))
    }

    /// File of an en passant capture that can legally be played.
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

    /// Generates legal moves with target squares from `targets(piece)` and keeps those satisfying `keep`.
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

    /// Extends a `cozy-chess` move with piece, capture and kind.
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
