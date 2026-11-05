# wsp-chess

A UCI chess engine whose evaluation is a neural network with **hand-written weights**: no
training, no data. Every neuron was written down on purpose (a ray between two squares, a pawn
structure rule, "this piece hangs", "the king zone is attacked"), and the network still evaluates
positions well enough to play at club level and beyond. The idea follows
[weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/): treat the
weights of a network as a program you can write by hand.

The search around it is a conventional alpha-beta engine written in Rust.

## What is in here

| Path | Content |
|---|---|
| `src/` | the engine (Rust): board on top of [cozy-chess](https://github.com/analog-hors/cozy-chess), sparse network evaluation, search, UCI |
| `tests/` | integration tests: evaluation check values, perft, golden search fingerprint, threads, UCI |
| `generator/` | the hand-written network (`build_weights.py`) and the exporter to the engine format (`export.py`); numpy only |
| `weights/` | `weights.bin` (the exported network), `units.json` (meaning of every hidden unit), `weights_check.json` (frozen check values) |
| `match/` | scripts for SPRTs and Stockfish ladders with [fastchess](https://github.com/Disservin/fastchess), and the openings used for all measurements |
| `chess-net-rs` | launcher for GUIs: builds the engine if needed and starts it |

## The network

```
x  = 768 input bits (12 piece kinds x 64 squares, seen from the side to move)
h1 = relu(W1 x + b1)                 4,496 units: rays, pawn structure, king phase
h2 = relu(W2 [x, h1] + b2)             512 units: "at least two attackers"
h3 = relu(W3 [x, h1, h2] + b3)       1,664 units: hanging and threatened pieces, king zone
s  = wo [x, h1, h2, h3] + bo         evaluation in pawns, from the side to move's view
```

- Material and piece-square values go straight from `x` to `s`.
- A ray unit for a bishop, rook or queen on square a towards square b is
  `relu(x[piece, a] - sum of occupied squares between a and b)`: exactly 1 if the line is open.
  The sum of all ray units is the mobility; counts of attackers per square are linear in `[x, h1]`.
- `h2` turns attacker counts into "at least two" indicators; `h3` combines attacks and defences
  into "this piece hangs" or "this piece is attacked by something cheaper", and counts attacked
  squares around the king.

All units and their groups are listed in `weights/units.json`; the construction with all formulas
is documented in `generator/build_weights.py`. Only 0.37 million of 15.8 million weights are
non-zero, and for a typical position only about 20 of the 4,496 `h1` units are active. The engine
exploits that: it keeps the input part of every layer as an accumulator, updates it incrementally
when a move changes a few input bits, and only adds the rows of the active neurons.

## Build and run

```sh
cargo build --release
./chess-net-rs            # UCI engine on stdin/stdout, builds on first start
```

Add `chess-net-rs` as a UCI engine in a GUI (e.g. En Croissant, Cute Chess, Arena).

UCI options:

| Option | Default | Meaning |
|---|---|---|
| `Weights` | `weights/weights.bin` next to the executable | network file |
| `Hash` | 64 MB | transposition table size |
| `Threads` | 1 | search threads (Lazy SMP), 1 to 12 |
| `PVS`, `Aspiration`, `NullMove`, `LMR`, `Futility`, `SEEPrune`, `Delta`, `CheckExt`, `Killers`, `History` | on | switch single search techniques off, to measure what each one is worth |
| `QChecks` | off | quiet checks in the first quiescence ply (measured to be harmful) |

Extra commands outside UCI: `eval` (static evaluation of the current position), `perft <depth>`,
`bench [depth]`.

### Rebuild the weights

```sh
uv run --with numpy python generator/export.py      # or: python3 generator/export.py with numpy installed
```

This writes `weights/weights.bin` and `weights/units.json` and reproduces both bit for bit. Groups
of units can be left out (`--disable threat_opp threat_own`) or scaled (`--scale king_safety=3`)
to try variants.

### Checks

`./check.sh` runs formatting, Clippy (pedantic, warnings are errors, limits for complexity, length
and names), all tests and line coverage (at least 95 %, needs `cargo-llvm-cov`). The golden test
pins node count, move and score of a fixed-depth search on 8 positions with 12 switch settings;
pure refactorings must not change it.

## Search

Negamax with alpha-beta and iterative deepening, transposition table (lock-free, 16 bytes per
entry), quiescence search (all evasions when in check, otherwise captures and promotions), move
ordering (hash move, MVV-LVA, killers, history), static exchange evaluation, delta pruning,
principal variation search, aspiration windows, null move, late move reductions, futility
pruning, check extension, and Lazy SMP for several threads.

## Measurements

All numbers below are **measured** unless marked as estimated. Games use 0.1 s per move for both
sides, openings from `match/openings.pgn` (four random plies) played twice with colours swapped,
and a pentanomial evaluation over game pairs.

### Measured in the original project

The engine was developed inside a larger research project; these results were measured there
(revision `c61d62a` of that project, with the board library shakmaty and Python match tools that
used the same openings and rules). They were **not** re-measured on this repository's final state.

Value of each search technique (SPRT, the engine against itself with one technique switched off):

| Technique | Elo lost without it [95 %] |
|---|---|
| LMR | 127 [74, 187] |
| PVS | 108 [57, 162] |
| History | 104 [56, 157] |
| Futility | 89 [43, 138] |
| Aspiration | 69 [30, 109] |
| SEE pruning | 52 [22, 82] |
| Delta pruning | 26 [7, 45] |
| Check extension | 20 [3, 37] (no decision) |
| Null move | 17 [1, 34] (no decision) |
| Killers | 13 [-2, 29] (no decision) |

More threads at the same time per move (SPRT): 4 threads against 1: +130 Elo [+87, +177];
8 threads against 4: +53 Elo [+23, +85].

Against Stockfish 19 with `UCI_LimitStrength` (Elo difference [95 %], 200 games each; the
one-thread row was measured before a later SEE fix worth +27 Elo [-0, +55]):

| Engine | sfelo 1500 | sfelo 1800 | sfelo 2000 | sfelo 2200 | sfelo 2500 | sfelo 2800 |
|---|---|---|---|---|---|---|
| 1 thread | +323 [+253, +393] | +111 [+61, +162] | -30 [-78, +19] | -111 [-162, -61] | -198 [-254, -142] | |
| 8 threads | | | | +48 [+10, +88] | -36 [-78, +5] | -299 [-352, -255] |

With 8 threads the comparison is **same time, more cores**: Stockfish plays with one thread.

**Estimated** (not measured): at 0.1 s per move about 1,950-2,000 on Stockfish's UCI_Elo scale with
one thread and 2,250-2,500 with eight. The scale is calibrated for longer time controls and is not
even in this range, so these numbers are a rough guide only.

### Measured on this repository

Replacing shakmaty with cozy-chess (commit "Replace shakmaty with cozy-chess (MIT)") was checked
against the original engine: same weights (SHA-256 `f767cdb5...`), one thread, 64 MB hash, 0.1 s per
move, fastchess 1.8.2 with `match/sprt.sh`, SPRT with bounds [-10, 0] Elo.

| Comparison | Pairs | Pentanomial [0, 0.5, 1, 1.5, 2] | Elo [95 %] | Result |
|---|---|---|---|---|
| this repository (`25eedb7`) against the original project (`c61d62a`) | 3,000 | [404, 536, 1176, 507, 377] | -4.8 [-12.1, +2.5] | no decision |

The test ended at its maximum length without a decision: a loss of more than about 12 Elo is
unlikely, but "not worse than 10 Elo" is not proven either. Static evaluations are bit-identical,
and on 120 test positions the best move at depth 6 is the same for both versions. Nodes per second
differ by -0.3 % (one process) and -1.1 % (eleven processes at once). Later commits only change the
reported principal variation, the option defaults shown to the GUI and the translation; the golden
search fingerprint is the same.

### Possible improvement

The remaining gap, if it is real, most likely comes from move order. cozy-chess generates moves in a
different order than shakmaty, and the search sorts by score, so moves with equal scores (quiet
moves without history, captures with the same MVV-LVA value) are searched in a different order. On the
bench positions at depth 8 the new version searches 13 % fewer nodes (1,196,376 instead of
1,373,427), which shows that the trees differ. Two things to try, each measured with `match/sprt.sh`:

1. A deterministic tie-break in move ordering (for example by from and to square), so that the order
   no longer depends on the board library.
2. Cheaper move descriptions: `Position::describe` looks up the captured piece for every generated
   capture; generating captures per victim type would avoid that and might recover the 1 % of speed
   lost under load.

## Measuring with fastchess

`match/sprt.sh` and `match/ladder.sh` call [fastchess](https://github.com/Disservin/fastchess)
(MIT, not included; build it from source and set `FASTCHESS`). Settings: fixed time per move,
game pairs with colours swapped, openings in fixed order, the game ends by the rules (mate,
stalemate, threefold repetition, fifty moves, insufficient material) or as a draw after 148 moves
following the opening, no resign or draw adjudication, logistic Elo, alpha = beta = 0.05.

```sh
FASTCHESS=/path/to/fastchess match/sprt.sh target/release/wsp-chess /path/to/reference/wsp-chess 0 10
THREADS=8 FASTCHESS=/path/to/fastchess match/ladder.sh target/release/wsp-chess 2200 2500
```

The original project used its own SPRT (pentanomial likelihood, decisions in blocks of 20 pairs,
at least 50 pairs). fastchess decides after every pair without a minimum. On simulated tests at
the exact hypothesis bounds (24 cases: three hypothesis pairs, four outcome distributions, 2,000
tests each) the fastchess SPRT code kept the error rate at or below 6.0 % (target 5 %), the same
range as the original rule, and both reached the same decisions on test sequences.

## License

MIT, see `LICENSE`. The engine depends on cozy-chess (MIT) and arrayvec (MIT or Apache-2.0); the
tests also use serde_json (MIT or Apache-2.0). Stockfish and fastchess are not part of this
repository. Earlier commits in the history used the board library shakmaty (GPL-3.0-or-later);
it was replaced in commit "Replace shakmaty with cozy-chess (MIT)", and the license statement
applies to the current state.
