"""Eingabekodierung: 768 Bits (12 Figurenarten x 64 Felder), Sicht des Spielers am Zug.

Ebenen 0..5 = eigene P,N,B,R,Q,K, Ebenen 6..11 = gegnerische. Ist Schwarz am Zug,
wird das Brett vertikal gespiegelt, damit "eigene" Bauern immer nach oben laufen.
Feldindex s = 8*Reihe + Linie (a1 = 0, wie python-chess).
"""
import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
P, N, B, R, Q, K = range(6)
OWN, OPP = 0, 1
NAMES = "PNBRQK"


def xi(side, t, s):
    """Index des Eingabebits: Seite (OWN/OPP), Figurentyp t (0..5), Feld s."""
    return (side * 6 + t) * 64 + s


def indices(board):
    """Liste der gesetzten Eingabebits."""
    stm = board.turn
    flip = stm == chess.BLACK
    out = []
    for side, color in ((OWN, stm), (OPP, not stm)):
        for t, pt in enumerate(PIECE_TYPES):
            bb = board.pieces_mask(pt, color)
            if flip:
                bb = chess.flip_vertical(bb)
            base = (side * 6 + t) * 64
            out.extend(base + s for s in chess.scan_forward(bb))
    return out


def encode(board):
    x = np.zeros(768, np.float32)
    x[indices(board)] = 1.0
    return x


def encode_many(boards):
    X = np.zeros((len(boards), 768), np.float32)
    for i, b in enumerate(boards):
        X[i, indices(b)] = 1.0
    return X
