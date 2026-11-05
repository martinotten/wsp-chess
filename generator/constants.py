"""Input encoding constants: 768 bits (12 piece kinds x 64 squares), seen from the side to move.

Planes 0..5 = own P, N, B, R, Q, K, planes 6..11 = opponent's. If Black is to move, the board is
mirrored vertically so that "own" pawns always move up. Square index s = 8 * rank + file (a1 = 0).
The engine (src/eval.rs) computes the same encoding.
"""
P, N, B, R, Q, K = range(6)
OWN, OPP = 0, 1
NAMES = "PNBRQK"


def xi(side, t, s):
    """Index of the input bit: side (OWN/OPP), piece type t (0..5), square s."""
    return (side * 6 + t) * 64 + s
