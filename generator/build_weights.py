"""Chess evaluation network with hand-written weights.

Architecture (every layer sees the input and all earlier layers):

    x  (768 bits, from the side to move's perspective)
    h1 = lrelu(W1 x + b1)                  rays, pawn structure, king phase
    h2 = lrelu(W2 [x,h1] + b2)             thresholds "at least 2 attackers"
    h3 = lrelu(W3 [x,h1,h2] + b3)          hanging/threatened pieces, king zone
    s  = wo [x,h1,h2,h3] + bo              evaluation in pawn units
    v  = tanh(s / 4)                        normalized output in (-1, 1)

Building blocks (in construction order):
 1. Material and piece-square values: linear from x to s.
 2. Pawn structure (h1): doubled relu(sum file - 1), isolated relu(x - sum adjacent files),
    passed relu(x - sum opposing pawns ahead).
 3. Attacks (h1): bishop/rook/queen on a attacks b if the squares in between are empty:
        relu(x[t,a] - sum_{s between a and b} occupied(s))       (exactly 0 or 1)
    Knights, kings, pawns and adjacent squares need no unit; their attacks are
    linear in x. The sum of the ray units gives the mobility.
    Attack count cnt(side, class, square) is linear in [x, h1]. Classes are cumulative by value:
    <=pawn, <=minor piece, <=rook, all.
    h2: relu(cnt - 1). The indicator "at least one attacker" is then cnt - relu(cnt - 1).
 4. h3, from indicators A (opponent attacks) and D (own side defends):
        hangs             relu(x[t,b] + A_all(b) - D(b) - 1)                output -0.2 value(t) own, +0.8 opponent's
        cheaper-threatened relu(x[t,b] + A_<=l(b) + D(b) - 2)               loss telescopes
    so that the total loss is value(t) - value(cheapest attacker) when defended (factor as for hangs).
    A and D are 0/1 indicators ("at least one attacker"), not counts.
    King zone: relu(10 x[K,k] + sum_{z adjacent to k} A(z) - 10) = number of attacked squares.

Trainability: normalized output, leaky ReLU, soft king phase, free units in every layer
with small random input weights and output weight 0 (they do not change the
evaluation initially, but they receive gradients).
"""
import numpy as np

from constants import xi, P, N, B, R, Q, K, OWN, OPP, NAMES

LEAKY = 0.01
SCALE = 4.0                       # v = tanh(s / SCALE)
VALUE = [1.0, 3.2, 3.3, 5.0, 9.0, 0.0]
LEVELS = [1.0, 3.0, 5.0]           # attacker classes <=pawn, <=minor piece, <=rook
CLS_ALL = 3
N_FREE = 64
GROUPS = ["material", "pst", "pawns", "mobility", "threat_opp", "threat_own", "king_safety", "king_eg",
          "king_danger", "free"]
# King danger: weight per attack of a piece type on a king-zone square; thresholds and additional slope
KD_WEIGHT = {N: 2.0, B: 2.0, R: 3.0, Q: 4.0}      # (pawn and king attacks do not count)
KD_STEPS = [(6, 0.05), (12, 0.10), (18, 0.15)]    # value > threshold: + slope (pawns per point)
KD_GATE = 200.0                                   # king on k enables the unit

DIRS = {B: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
        R: [(1, 0), (-1, 0), (0, 1), (0, -1)]}
DIRS[Q] = DIRS[B] + DIRS[R]
KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def sq(f, r):
    return 8 * r + f if 0 <= f < 8 and 0 <= r < 8 else None


def fr(s):
    return s % 8, s // 8


def ring(s):
    f, r = fr(s)
    return int(max(abs(f - 3.5), abs(r - 3.5)) - 0.5)


def mirror(s):
    return s ^ 56


# ---------------------------------------------------------------- piece-square values (own perspective)
MOB = {B: 0.04, R: 0.03, Q: 0.015}


def pst_table():
    T = np.zeros((6, 64))
    for s in range(64):
        f, r = fr(s)
        T[P, s] = [0, 0, 0.05, 0.1, 0.2, 0.35, 0.6, 0][r]
        if r in (3, 4):
            T[P, s] += {3: 0.1, 4: 0.1, 2: 0.05, 5: 0.05}.get(f, 0)
        if r == 1 and f in (3, 4):
            T[P, s] -= 0.05
        T[N, s] = [0.2, 0.1, 0.0, -0.2][ring(s)]
        T[B, s] = [0.1, 0.05, 0.0, -0.1][ring(s)]
        T[R, s] = 0.2 if r == 6 else 0.0
        T[Q, s] = [0.05, 0.03, 0.0, -0.05][ring(s)]
        T[K, s] = king_mg(s)
        # attacks on adjacent squares are linear in x: they belong to bishop/rook/queen mobility
    return T


def king_mg(s):
    f, r = fr(s)
    if r == 0:
        return [0.15, 0.25, 0.2, 0.0, 0.0, 0.05, 0.3, 0.2][f]
    if r == 1:
        return -0.15
    return -0.4 - 0.1 * (r - 2)


def king_eg(s):
    return [0.3, 0.15, 0.0, -0.15][ring(s)]


# ---------------------------------------------------------------- construction helper
class Builder:
    """Collects units as linear expressions over earlier layers.

    Leak compensation: lrelu(z) = (1-a) relu(z) + a z, and z is linear in earlier layers.
    A consumer of unit u therefore gets the expression
        E(u) = (h_u - a * z_u) / (1 - a)
    which equals relu(z_u) exactly, for any a. The hand-written evaluation thus does not depend on a.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.units = {"h1": [], "h2": [], "h3": []}
        self.exact = {}
        self.out = {}             # key -> list of (weight, group)
        self.info = {}            # key -> kind and location (side to move's perspective), see unit()

    def unit(self, layer, expr, bias, group, name, info):
        """info: kind and location of the pattern for explanations, squares in net coordinates (own side at the bottom):
        {"kind": ..., "side": OWN/OPP, "piece": t?, "cls": c?, "place": ("ray", a, b) | ("square", s) |
        ("file", f) | ("zone", k)}. Does not change any weights."""
        expr = dict(expr)
        bias += expr.pop(("const", 0), 0.0)
        self.units[layer].append((expr, bias, group, name))
        key = (layer, len(self.units[layer]) - 1)
        self.info[key] = info
        a = self.alpha
        e = {k: -a / (1 - a) * v for k, v in expr.items()}
        e[key] = e.get(key, 0.0) + 1 / (1 - a)
        e[("const", 0)] = e.get(("const", 0), 0.0) - a / (1 - a) * bias
        self.exact[key] = e
        return key

    def ref(self, key):
        return self.exact.get(key, {key: 1.0})

    def to_out(self, expr, w, group):
        for k, v in expr.items():
            self.out.setdefault(k, []).append((w * v, group))


def X(side, t, s, c=1.0):
    return {("x", xi(side, t, s)): c}


def add(*exprs, coefs=None):
    out = {}
    for i, e in enumerate(exprs):
        c = 1.0 if coefs is None else coefs[i]
        for k, v in e.items():
            out[k] = out.get(k, 0.0) + c * v
    return out


def occ(s):
    return {("x", xi(side, t, s)): 1.0 for side in (OWN, OPP) for t in range(6)}


def build(disable=(), seed=0, n_free=N_FREE, alpha=LEAKY, scale=None, extra=()):
    return assemble(construct(disable, seed, n_free, alpha, scale, extra))


def construct(disable=(), seed=0, n_free=N_FREE, alpha=LEAKY, scale=None, extra=()):
    """Builds the units; the builder keeps weight and group per output contribution (for decompositions).

    scale: {group: factor} multiplies the output weights of these groups (experiments).
    extra: additional building blocks that are disabled by default, e.g. ("king_danger",).
    """
    disable = set(disable)
    scale = scale or {}
    bld = Builder(alpha)
    rng = np.random.default_rng(seed)
    sign = {OWN: 1.0, OPP: -1.0}

    def out(key, w, group):
        if group not in disable:
            bld.to_out(bld.ref(key), w * scale.get(group, 1.0), group)

    # -------- 1. Material and piece-square values
    T = pst_table()
    for side in (OWN, OPP):
        for t in range(6):
            for s in range(64):
                own_s = s if side == OWN else mirror(s)
                out(("x", xi(side, t, s)), sign[side] * VALUE[t], "material")
                out(("x", xi(side, t, s)), sign[side] * T[t, own_s], "pst")

    # -------- 3a. Ray units (h1) and attack counts
    # att[side][t][b] = expression (linear in x, h1) = number of pieces of type t attacking b
    att = {side: {t: {b: {} for b in range(64)} for t in range(6)} for side in (OWN, OPP)}
    for side in (OWN, OPP):
        fwd = 1 if side == OWN else -1
        for a in range(64):
            f, r = fr(a)
            for df in (-1, 1):
                b = sq(f + df, r + fwd)
                if b is not None:
                    att[side][P][b] = add(att[side][P][b], X(side, P, a))
            for t, deltas in ((N, KNIGHT), (K, KING)):
                for df, dr in deltas:
                    b = sq(f + df, r + dr)
                    if b is not None:
                        att[side][t][b] = add(att[side][t][b], X(side, t, a))
            for t in (B, R, Q):
                for df, dr in DIRS[t]:
                    between = []
                    k = 1
                    while (b := sq(f + k * df, r + k * dr)) is not None:
                        if not between:
                            att[side][t][b] = add(att[side][t][b], X(side, t, a))
                            out(("x", xi(side, t, a)), sign[side] * MOB[t], "mobility")
                        else:
                            e = X(side, t, a)
                            for s in between:
                                e = add(e, occ(s), coefs=[1, -1])
                            u = bld.unit("h1", e, 0.0, "mobility",
                                         f"{'own' if side == OWN else 'opp'} {NAMES[t]} {a}->{b}",
                                         {"kind": "ray", "side": side, "piece": t, "place": ("ray", a, b)})
                            att[side][t][b] = add(att[side][t][b], bld.ref(u))
                            out(u, sign[side] * MOB[t], "mobility")
                        between.append(b)
                        k += 1

    # -------- 2. Pawn structure (h1)
    PASSED = [0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 0]
    for side in (OWN, OPP):
        other = OPP if side == OWN else OWN
        sname = "own" if side == OWN else "opp"
        for f in range(8):
            e = {}
            for r in range(8):
                e = add(e, X(side, P, sq(f, r)))
            u = bld.unit("h1", e, -1.0, "pawns", f"{sname} doubled file {f}",
                         {"kind": "doubled", "side": side, "piece": P, "place": ("file", f)})
            out(u, -0.2 * sign[side], "pawns")
        for s in range(8, 56):
            f, r = fr(s)
            e = X(side, P, s)
            for ff in (f - 1, f + 1):
                for rr in range(8):
                    if sq(ff, rr) is not None:
                        e = add(e, X(side, P, sq(ff, rr), -1.0))
            u = bld.unit("h1", e, 0.0, "pawns", f"{sname} isolated {s}",
                         {"kind": "isolated", "side": side, "piece": P, "place": ("square", s)})
            out(u, -0.15 * sign[side], "pawns")
            e = X(side, P, s)
            ahead = range(r + 1, 8) if side == OWN else range(0, r)
            for ff in (f - 1, f, f + 1):
                for rr in ahead:
                    if sq(ff, rr) is not None:
                        e = add(e, X(other, P, sq(ff, rr), -1.0))
            u = bld.unit("h1", e, 0.0, "pawns", f"{sname} passed {s}",
                         {"kind": "passed", "side": side, "piece": P, "place": ("square", s)})
            rel = r if side == OWN else 7 - r
            out(u, PASSED[rel] * sign[side], "pawns")

    # -------- King phase (h1): soft threshold on the opponent's piece material
    PHASE_W = {N: 0.15, B: 0.15, R: 0.25, Q: 0.5}
    for side in (OWN, OPP):
        other = OPP if side == OWN else OWN
        for k in range(64):
            e = X(side, K, k)
            for t, w in PHASE_W.items():
                for s in range(64):
                    e = add(e, X(other, t, s, -w))
            own_k = k if side == OWN else mirror(k)
            u = bld.unit("h1", e, 0.0, "king_eg", f"{'own' if side == OWN else 'opp'} K endgame {k}",
                         {"kind": "king_phase", "side": side, "piece": K, "place": ("square", k)})
            out(u, sign[side] * (king_eg(own_k) - king_mg(own_k)), "king_eg")

    # -------- Pawn shield (h1)
    for side in (OWN, OPP):
        back = 0 if side == OWN else 7
        fwd = 1 if side == OWN else -1
        for f in range(8):
            k = sq(f, back)
            e = X(side, K, k, 10.0)
            for ff in (f - 1, f, f + 1):
                for d in (1, 2):
                    s = sq(ff, back + d * fwd)
                    if s is not None:
                        e = add(e, X(side, P, s))
            u = bld.unit("h1", e, -10.0, "king_safety", f"{'own' if side == OWN else 'opp'} shield {k}",
                         {"kind": "shield", "side": side, "piece": K, "place": ("square", k)})
            out(u, 0.1 * sign[side], "king_safety")

    free_units(bld, "h1", 768, 0.1, n_free, rng)

    # -------- 3b. Attack counts per class, threshold "at least 2" (h2)
    order = [[P], [P, N, B], [P, N, B, R], [P, N, B, R, Q, K]]
    cnt = {side: {c: {} for c in range(4)} for side in (OWN, OPP)}
    ind = {side: {c: {} for c in range(4)} for side in (OWN, OPP)}
    for side in (OWN, OPP):
        for c, types in enumerate(order):
            for b in range(64):
                e = {}
                for t in types:
                    e = add(e, att[side][t][b])
                cnt[side][c][b] = e
                u = bld.unit("h2", e, -1.0, "attack",
                             f"{'own' if side == OWN else 'opp'} attacks<= {'PMRA'[c]} {b} >=2",
                             {"kind": "count2", "side": side, "cls": c, "place": ("square", b)})
                ind[side][c][b] = add(e, bld.ref(u), coefs=[1, -1])

    n1 = len(bld.units["h1"])
    free_units(bld, "h2", 768 + n1, 0.05, n_free, rng)

    # -------- 4. Hanging pieces and pieces threatened by cheaper attackers (h3)
    for side in (OWN, OPP):
        other = OPP if side == OWN else OWN
        group = "threat_own" if side == OWN else "threat_opp"
        gamma = -0.2 if side == OWN else 0.8      # side to move: opponent's losses are gains
        sname = "own" if side == OWN else "opp"
        for b in range(64):
            D = ind[side][CLS_ALL][b]
            for t in (P, N, B, R, Q):
                e = add(X(side, t, b), ind[other][CLS_ALL][b], D, coefs=[1, 1, -1])
                u = bld.unit("h3", e, -1.0, group, f"{sname} {NAMES[t]} {b} hangs",
                             {"kind": "hangs", "side": side, "piece": t, "place": ("square", b)})
                out(u, gamma * VALUE[t], group)
                lv = [l for l in LEVELS if l < VALUE[t] - 0.5]
                for i, l in enumerate(lv):
                    nxt = lv[i + 1] if i + 1 < len(lv) else VALUE[t]
                    e = add(X(side, t, b), ind[other][i][b], D)
                    u = bld.unit("h3", e, -2.0, group,
                                 f"{sname} {NAMES[t]} {b} attacked by <= {'PMR'[i]}, defended",
                                 {"kind": "cheaper", "side": side, "piece": t, "cls": i, "place": ("square", b)})
                    out(u, gamma * (nxt - l), group)

    # -------- King zone (h3)
    for side in (OWN, OPP):
        other = OPP if side == OWN else OWN
        for k in range(64):
            f, r = fr(k)
            e = X(side, K, k, 10.0)
            for df, dr in KING:
                z = sq(f + df, r + dr)
                if z is not None:
                    e = add(e, ind[other][CLS_ALL][z])
            u = bld.unit("h3", e, -10.0, "king_safety",
                         f"{'own' if side == OWN else 'opp'} K {k} zone attacked",
                         {"kind": "zone", "side": side, "piece": K, "place": ("zone", k)})
            out(u, -0.1 if side == OWN else 0.05, "king_safety")

    # -------- King danger (h3, only with extra=("king_danger",)): weighted attacks on the king zone,
    # piecewise linear with increasing slope (approximation of Stockfish's quadratic king danger).
    if "king_danger" in extra:
        for side in (OWN, OPP):
            other = OPP if side == OWN else OWN
            for k in range(64):
                f, r = fr(k)
                e = X(side, K, k, KD_GATE)
                for df, dr in KING:
                    z = sq(f + df, r + dr)
                    if z is not None:
                        for t, w in KD_WEIGHT.items():
                            e = add(e, att[other][t][z], coefs=[1, w])
                for thr, slope in KD_STEPS:
                    u = bld.unit("h3", e, -KD_GATE - thr, "king_danger",
                                 f"{'own' if side == OWN else 'opp'} K {k} danger >{thr}",
                                 {"kind": "danger", "side": side, "piece": K, "place": ("zone", k)})
                    out(u, -slope if side == OWN else slope, "king_danger")

    n2 = len(bld.units["h2"])
    free_units(bld, "h3", 768 + n1 + n2, 0.05, n_free, rng)
    return bld


def free_units(bld, layer, fan_in, sigma, n, rng):
    """Free units: random inputs, output 0. Their input weights are filled in by assemble."""
    for i in range(n):
        bld.units[layer].append(("FREE", (fan_in, sigma, rng.integers(1 << 31)), "free", f"{layer} free {i}"))


def assemble(bld):
    n = {l: len(bld.units[l]) for l in ("h1", "h2", "h3")}
    off = {"x": 0, "h1": 768, "h2": 768 + n["h1"], "h3": 768 + n["h1"] + n["h2"]}
    params = {}
    for l in ("h1", "h2", "h3"):
        W = np.zeros((n[l], off[l]), np.float32)
        bvec = np.zeros(n[l], np.float32)
        for i, (expr, bias, group, name) in enumerate(bld.units[l]):
            if expr == "FREE":
                fan_in, sigma, s = bias
                W[i] = np.random.default_rng(s).normal(0, sigma, fan_in)
                continue
            for (layer, j), v in expr.items():
                W[i, off[layer] + j] += v
            bvec[i] = bias
        params["W" + l[1]] = W
        params["b" + l[1]] = bvec
    wo = np.zeros(off["h3"] + n["h3"], np.float32)
    bo = 0.0
    for (layer, j), lst in bld.out.items():
        if layer == "const":
            bo += sum(w for w, _ in lst)
        else:
            wo[off[layer] + j] += sum(w for w, _ in lst)
    params["wo"] = wo
    params["bo"] = np.array([bo], np.float32)
    meta = {l: [(g, nm) for (_, _, g, nm) in bld.units[l]] for l in ("h1", "h2", "h3")}
    return params, meta


if __name__ == "__main__":
    from collections import Counter
    params, meta = build(alpha=0.0, n_free=0)
    for layer in ("h1", "h2", "h3"):
        print(layer, len(meta[layer]), dict(Counter(g for g, _ in meta[layer])))
    total = sum(v.size for v in params.values())
    nz = sum(int((v != 0).sum()) for v in params.values())
    print(f"parameters: {total:,}, non-zero: {nz:,}")
