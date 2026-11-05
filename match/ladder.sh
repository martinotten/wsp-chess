#!/bin/sh
# Fixed-length matches of one engine against Stockfish at several UCI_Elo levels (fastchess).
#
#   match/ladder.sh ENGINE ELO [ELO ...]
#
# Stockfish plays with UCI_LimitStrength, one thread and 64 MB hash; the engine with THREADS (1).
# Same time per move for both (ST, 0.1 s). PAIRS opening pairs per level (100). Results are
# Elo differences on Stockfish's UCI_Elo scale, which is calibrated for longer time controls.
# Environment: FASTCHESS, STOCKFISH (stockfish), ST, THREADS, HASH, WEIGHTS, CONCURRENCY, PAIRS, OUT.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 2 ] || { sed -n '2,10p' "$0"; exit 2; }
ENGINE=$1; shift
FASTCHESS=${FASTCHESS:-fastchess} STOCKFISH=${STOCKFISH:-stockfish}
ST=${ST:-0.1} THREADS=${THREADS:-1} HASH=${HASH:-64} PAIRS=${PAIRS:-100}
WEIGHTS=${WEIGHTS:-$ROOT/weights/weights.bin}
CONCURRENCY=${CONCURRENCY:-$(( 11 / THREADS > 0 ? 11 / THREADS : 1 ))}
OUT=${OUT:-$ROOT/logs}
mkdir -p "$OUT"
for ELO in "$@"; do
  NAME="ladder-$ELO-$(date +%Y%m%d-%H%M%S)"
  "$FASTCHESS" \
    -engine cmd="$ENGINE" name=wsp-chess option.Threads="$THREADS" option.Hash="$HASH" option.Weights="$WEIGHTS" \
    -engine cmd="$STOCKFISH" name="stockfish-$ELO" option.UCI_LimitStrength=true option.UCI_Elo="$ELO" \
      option.Threads=1 option.Hash=64 \
    -each proto=uci st="$ST" \
    -openings file="$ROOT/match/openings.pgn" format=pgn order=sequential \
    -rounds "$PAIRS" -repeat -maxmoves 148 \
    -concurrency "$CONCURRENCY" -report penta=true -autosaveinterval 0 -ratinginterval 50 \
    -pgnout file="$OUT/$NAME.pgn" notation=uci \
    2>&1 | tee "$OUT/$NAME.log"
done
