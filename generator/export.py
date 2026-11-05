"""Builds the hand-written network and writes it in the engine's binary format.

    python generator/export.py [--out weights/weights.bin] [--disable group ...]
                               [--scale group=factor ...] [--extra unit ...]

The network is exported with ReLU (alpha = 0) and without free (random) units; only non-zero weights
are stored, per input or neuron as a list of targets. Next to the binary file, units.json lists the
meaning of every hidden unit (group, description) in the order of the exported layers.

Format (little endian): b"WSPN", version, n1, n2, n3 (u32); biases of z1, z2, z3, out (f32);
for each source (768 inputs, then h1, h2, h3) the start offset into the target list (u32, one more
entry than sources); targets (u32); weights (f32).
"""
import argparse
import json
import os
import struct

import numpy as np

from build_weights import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_bin(p, path):
    n1, n2, n3 = len(p["b1"]), len(p["b2"]), len(p["b3"])
    sources = 768 + n1 + n2 + n3
    bias = np.concatenate([p["b1"], p["b2"], p["b3"], p["bo"]]).astype(np.float32)
    # matrix targets x sources, targets = z1, z2, z3, out
    off = [0, n1, n1 + n2, n1 + n2 + n3]
    starts, tgts, wts = [0], [], []
    layers = [(p["W1"], 0), (p["W2"], off[1]), (p["W3"], off[2]), (p["wo"][None, :], off[3])]
    cols = {}
    for matrix, t0 in layers:
        rows, columns = np.nonzero(matrix)
        for row, col in zip(rows, columns):
            cols.setdefault(int(col), []).append((t0 + int(row), float(matrix[row, col])))
    for s in range(sources):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "weights", "weights.bin"))
    ap.add_argument("--disable", nargs="*", default=[], help="unit groups to leave out")
    ap.add_argument("--scale", nargs="*", default=[], help="group=factor, e.g. king_safety=3")
    ap.add_argument("--extra", nargs="*", default=[], help="additional units, e.g. king_danger")
    a = ap.parse_args()
    scale = {k: float(v) for k, v in (x.split("=") for x in a.scale)}
    p, meta = build(disable=a.disable, alpha=0.0, n_free=0, scale=scale, extra=a.extra)
    n1, n2, n3, nnz = to_bin(p, a.out)
    print(f"{a.out}: n1={n1} n2={n2} n3={n3}, {nnz} weights")
    units = os.path.join(os.path.dirname(a.out), "units.json")
    with open(units, "w") as fh:
        json.dump(meta, fh)
    print(f"{units}: {', '.join(f'{k} {len(v)}' for k, v in meta.items())}")


if __name__ == "__main__":
    main()
