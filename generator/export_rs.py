"""Export des Handnetzes für den Rust-Kern (engine-rs).

    python export_rs.py [--disable gruppe ...] [--out weights.bin]

Grundlage ist build(alpha=0, n_free=0): reines ReLU, ohne freie Neuronen. Wegen der exakten Leck-Kompensation
ist das dieselbe Funktion wie das Leaky-Netz in weights.npz (Abweichung ~1e-4 Bauern); freie Neuronen speisen nur
andere freie Neuronen und haben keinen Pfad zum Ausgang (geprüft).

Format (Little-Endian):
  b"WSPN", u32 Version=1, u32 n1, n2, n3
  f32[n1+n2+n3+1]  Biases der Ziele  T = [z1 | z2 | z3 | out]
  u32[S+1]         Zeilenanfänge je Quelle, S = 768 + n1 + n2 + n3 (Eingaben, h1, h2, h3)
  u32[nnz]         Zielindex in T
  f32[nnz]         Gewicht
Eine Quelle wirkt nur auf spätere Schichten (dichte Skip-Verbindungen).

Außerdem weights_check.json: Sollwerte s (Bauern, vor tanh) für Teststellungen und Zugfolgen mit Sonderzügen.
"""
import argparse
import json
import struct

import chess
import numpy as np
import torch

from build_weights import build
from features import encode_many
from net import ChessNet, load


def to_bin(p, path):
    n1, n2, n3 = len(p["b1"]), len(p["b2"]), len(p["b3"])
    S = 768 + n1 + n2 + n3
    bias = np.concatenate([p["b1"], p["b2"], p["b3"], p["bo"]]).astype(np.float32)
    # Matrix Ziele x Quellen, Ziele = z1, z2, z3, out
    off = [0, n1, n1 + n2, n1 + n2 + n3]
    starts, tgts, wts = [0], [], []
    W = [(p["W1"], 0, 768), (p["W2"], off[1], 768 + n1), (p["W3"], off[2], 768 + n1 + n2),
         (p["wo"][None, :], off[3], S)]
    cols = {}
    for M, t0, ncol in W:
        r, c = np.nonzero(M)
        for rr, cc in zip(r, c):
            cols.setdefault(int(cc), []).append((t0 + int(rr), float(M[rr, cc])))
    for s in range(S):
        for t, w in sorted(cols.get(s, [])):
            tgts.append(t)
            wts.append(w)
        starts.append(len(tgts))
    with open(path, "wb") as fh:
        fh.write(b"WSPN" + struct.pack("<4I", 1, n1, n2, n3))
        fh.write(bias.tobytes())
        fh.write(np.asarray(starts, np.uint32).tobytes())
        fh.write(np.asarray(tgts, np.uint32).tobytes())
        fh.write(np.asarray(wts, np.float32).tobytes())
    return n1, n2, n3, len(tgts)


def game_positions(n, seed):
    """Stellungen aus Zufallspartien mit Schlagvorliebe (erreicht auch Endspiele)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        stop = int(rng.integers(2, 200))
        for _ in range(stop):
            moves = list(b.legal_moves)
            if not moves:
                break
            caps = [m for m in moves if b.is_capture(m)]
            b.push(caps[rng.integers(len(caps))] if caps and rng.random() < 0.6 else moves[rng.integers(len(moves))])
        if not b.is_game_over():
            out.append(b.fen())
    return out


SEQUENCES = [  # (Start-FEN, Züge): Rochaden, en passant, Schlagumwandlung, Unterverwandlung
    (chess.STARTING_FEN, "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1 f8c5 d2d3 d7d6 c1g5 c8g4 b1c3 d8d7 d1d2 e8c8".split()),
    ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "e5f6 g7f6 d2d4 e7e5 d4e5 f6e5".split()),
    ("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2", "e4d3 c2d3".split()),
    ("4k3/P7/8/8/8/8/6p1/R3K3 w Q - 0 1", "e1c1 g2g1n a7a8b e8f7".split()),
    ("3qk3/2P5/8/8/8/8/7p/R3K3 w Q - 0 1", "c7d8r e8d8 e1c1 d8e7 c1b2 h2h1b".split()),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8 e1g1 c8b8 a1a8 b8a8".split()),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disable", nargs="*", default=[])
    ap.add_argument("--out", default="weights.bin")
    ap.add_argument("--scale", nargs="*", default=[], help="Gruppe=Faktor, z. B. king_safety=3")
    ap.add_argument("--extra", nargs="*", default=[], help="Zusatzbausteine, z. B. king_danger")
    a = ap.parse_args()
    scale = {k: float(v) for k, v in (x.split("=") for x in a.scale)}
    p, _ = build(disable=a.disable, alpha=0.0, n_free=0, scale=scale, extra=a.extra)
    n1, n2, n3, nnz = to_bin(p, a.out)
    print(f"{a.out}: n1={n1} n2={n2} n3={n3}, {nnz} Gewichte")

    ref = ChessNet.from_params(p, leaky=0.0).eval()           # explizit ReLU
    fens = game_positions(300, 7)
    seqs = []
    for fen, moves in SEQUENCES:
        b = chess.Board(fen)
        steps = [b.fen()]
        for m in moves:
            b.push_uci(m)                                        # wirft bei illegalem Zug
            steps.append(b.fen())
        seqs.append({"fen": fen, "moves": moves, "fens": steps})
    allf = fens + [f for s in seqs for f in s["fens"]]
    with torch.no_grad():
        x = torch.from_numpy(encode_many([chess.Board(f) for f in allf]))
        s0 = ref.score(x).numpy()
        if not a.disable:                                        # Dreiwegeprüfung: Leaky-Artefakt
            s1 = load("weights.npz").eval().score(x).numpy()
            print(f"ReLU-Export vs. weights.npz (leaky): max |Δs| = {np.abs(s0 - s1).max():.2e}")
    vals = dict(zip(allf, map(float, s0)))
    check = {"positions": [{"fen": f, "s": vals[f]} for f in fens],
             "sequences": [{**s, "s": [vals[f] for f in s["fens"]]} for s in seqs]}
    name = a.out.replace(".bin", "_check.json")
    json.dump(check, open(name, "w"), indent=0)
    print(f"{name}: {len(fens)} Stellungen, {len(seqs)} Zugfolgen")


if __name__ == "__main__":
    main()
