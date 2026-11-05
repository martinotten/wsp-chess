"""Schach-Bewertungsnetz mit von Hand geschriebenen Gewichten.

Architektur (jede Schicht sieht die Eingabe und alle früheren Schichten):

    x  (768 Bits, Sicht des Spielers am Zug)
    h1 = lrelu(W1 x + b1)                  Strahlen, Bauernstruktur, Königsphase
    h2 = lrelu(W2 [x,h1] + b2)             Schwellen "mind. 2 Angreifer"
    h3 = lrelu(W3 [x,h1,h2] + b3)          hängende/bedrohte Figuren, Königszone
    s  = wo [x,h1,h2,h3] + bo              Bewertung in Bauerneinheiten
    v  = tanh(s / 4)                        normierte Ausgabe in (-1, 1)

Bausteine (Reihenfolge des Bauplans):
 1. Material und Feldwerte: linear von x nach s.
 2. Bauernstruktur (h1): Doppelbauer relu(sum Linie - 1), isoliert relu(x - sum Nachbarlinien),
    Freibauer relu(x - sum gegnerische Bauern davor).
 3. Angriffe (h1): Läufer/Turm/Dame auf a greift b an, wenn die Felder dazwischen leer sind:
        relu(x[t,a] - sum_{s zwischen a und b} belegt(s))       (genau 0 oder 1)
    Springer, König, Bauern und Nachbarfelder brauchen keine Einheit, ihre Angriffe sind
    linear in x. Die Summe der Strahleinheiten ergibt die Mobilität.
    Angriffszahl cnt(Seite, Klasse, Feld) ist linear in [x, h1]. Klassen kumulativ nach Wert:
    <=Bauer, <=Leichtfigur, <=Turm, alle.
    h2: relu(cnt - 1). Damit ist der Indikator "mind. ein Angreifer" = cnt - relu(cnt - 1).
 4. h3, aus Indikatoren A (Gegner greift an) und D (eigene Seite deckt):
        hängt         relu(x[t,b] + A_alle(b) - D(b) - 1)                Ausgang -0,2 Wert(t) eigene, +0,8 gegnerische
        billiger bedroht relu(x[t,b] + A_<=l(b) + D(b) - 2)                 Verlust teleskopiert
    so dass der Verlust insgesamt Wert(t) - Wert(billigster Angreifer) ist, wenn gedeckt (Faktor wie bei hängt).
    A und D sind Indikatoren 0/1 ("mind. ein Angreifer"), keine Zahlen.
    Königszone: relu(10 x[K,k] + sum_{z Nachbar von k} A(z) - 10) = Zahl angegriffener Felder.

Lernfähigkeit: normierte Ausgabe, Leaky-ReLU, weiche Königsphase, je Schicht freie
Einheiten mit kleinen zufälligen Eingangsgewichten und Ausgangsgewicht 0 (ändern die
Bewertung anfangs nicht, bekommen aber Gradienten).
"""
import numpy as np

import paths
from features import xi, P, N, B, R, Q, K, OWN, OPP, NAMES

LEAKY = 0.01
SCALE = 4.0                       # v = tanh(s / SCALE)
VALUE = [1.0, 3.2, 3.3, 5.0, 9.0, 0.0]
LEVELS = [1.0, 3.0, 5.0]           # Angreiferklassen <=Bauer, <=Leichtfigur, <=Turm
CLS_ALL = 3
N_FREE = 64
GROUPS = ["material", "pst", "pawns", "mobility", "threat_opp", "threat_own", "king_safety", "king_eg",
          "king_danger", "free"]
# Königsgefahr: Gewicht je Angriff eines Figurentyps auf ein Feld der Königszone; Schwellen und Zusatzsteigung
KD_WEIGHT = {N: 2.0, B: 2.0, R: 3.0, Q: 4.0}      # (Bauern- und Königsangriffe zählen nicht)
KD_STEPS = [(6, 0.05), (12, 0.10), (18, 0.15)]    # Wert > Schwelle: + Steigung (Bauern je Punkt)
KD_GATE = 200.0                                   # König auf k schaltet die Einheit frei

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


# ---------------------------------------------------------------- Feldwerte (eigene Sicht)
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
        # Angriffe auf Nachbarfelder sind linear in x: gehören zur Mobilität der Läufer/Türme/Damen
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


# ---------------------------------------------------------------- Bauhelfer
class Builder:
    """Sammelt Einheiten als lineare Ausdrücke über frühere Schichten.

    Leck-Kompensation: lrelu(z) = (1-a) relu(z) + a z, und z ist linear in früheren Schichten.
    Wer eine Einheit u benutzt, bekommt daher den Ausdruck
        E(u) = (h_u - a * z_u) / (1 - a)
    der exakt relu(z_u) ist, für jedes a. Die handgeschriebene Bewertung hängt so nicht von a ab.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.units = {"h1": [], "h2": [], "h3": []}
        self.exact = {}
        self.out = {}             # Schlüssel -> Liste (Gewicht, Gruppe)
        self.info = {}            # Schlüssel -> Art und Ort (Sicht des Ziehenden), siehe unit()

    def unit(self, layer, expr, bias, group, name, info):
        """info: Art und Ort des Musters für Erklärungen, Felder in Netzkoordinaten (eigene Seite unten):
        {"kind": ..., "side": OWN/OPP, "piece": t?, "cls": c?, "place": ("ray", a, b) | ("square", s) |
        ("file", f) | ("zone", k)}. Ändert keine Gewichte."""
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
    """Baut die Einheiten auf; der Builder hält je Ausgangsbeitrag Gewicht und Gruppe (für Zerlegungen).

    scale: {Gruppe: Faktor} multipliziert die Ausgangsgewichte dieser Gruppen (Experimente).
    extra: zusätzliche, standardmäßig abgeschaltete Bausteine, z. B. ("king_danger",).
    """
    disable = set(disable)
    scale = scale or {}
    bld = Builder(alpha)
    rng = np.random.default_rng(seed)
    sign = {OWN: 1.0, OPP: -1.0}

    def out(key, w, group):
        if group not in disable:
            bld.to_out(bld.ref(key), w * scale.get(group, 1.0), group)

    # -------- 1. Material und Feldwerte
    T = pst_table()
    for side in (OWN, OPP):
        for t in range(6):
            for s in range(64):
                own_s = s if side == OWN else mirror(s)
                out(("x", xi(side, t, s)), sign[side] * VALUE[t], "material")
                out(("x", xi(side, t, s)), sign[side] * T[t, own_s], "pst")

    # -------- 3a. Strahleinheiten (h1) und Angriffszahlen
    # attackers[side][t][b] = Ausdruck (linear in x, h1) = Zahl der Figuren vom Typ t, die b angreifen
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

    # -------- 2. Bauernstruktur (h1)
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

    # -------- Königsphase (h1): weiche Schwelle über gegnerisches Figurenmaterial
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

    # -------- Bauernschild (h1)
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

    # -------- 3b. Angriffszahlen je Klasse, Schwelle "mind. 2" (h2)
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

    # -------- 4. Hängende und billiger bedrohte Figuren (h3)
    for side in (OWN, OPP):
        other = OPP if side == OWN else OWN
        group = "threat_own" if side == OWN else "threat_opp"
        gamma = -0.2 if side == OWN else 0.8      # am Zug: gegnerische Verluste sind Gewinne
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

    # -------- Königszone (h3)
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

    # -------- Königsgefahr (h3, nur mit extra=("king_danger",)): gewichtete Angriffe auf die Königszone,
    # stückweise linear mit wachsender Steigung (Näherung an Stockfishs quadratische king danger).
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
    """Freie Einheiten: zufällige Eingänge, Ausgang 0. Ihr Eingang wird in assemble gefüllt."""
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


SEARCH_DISABLE = ("threat_opp", "threat_own")   # Variante für die Suche: nur bei fester Tiefe stärker (README, "Mit Suche")

if __name__ == "__main__":
    import sys, time, json
    t0 = time.time()
    if sys.argv[1:] == ["search"]:          # Gewichte für die Engine mit Suche
        params, meta = build(disable=SEARCH_DISABLE)
        np.savez_compressed(paths.weights("weights_search.npz"), **params)
        print(f"weights_search.npz ohne {SEARCH_DISABLE}, {time.time() - t0:.1f} s")
        sys.exit()
    params, meta = build()
    np.savez_compressed(paths.weights("weights.npz"), **params)
    with open(paths.weights("units.json"), "w") as fh:
        json.dump(meta, fh)
    total = sum(v.size for v in params.values())
    nz = sum(int((v != 0).sum()) for k, v in params.items())
    print(f"{time.time() - t0:.1f} s")
    for l in ("h1", "h2", "h3"):
        from collections import Counter
        print(l, len(meta[l]), dict(Counter(g for g, _ in meta[l])))
    print(f"Parameter: {total:,}, davon != 0: {nz:,}")
