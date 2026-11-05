#!/bin/sh
# SPRT between two UCI engines with fastchess (https://github.com/Disservin/fastchess, MIT).
#
#   match/sprt.sh ENGINE_A ENGINE_B [ELO0 ELO1]
#
# A is the candidate, B the reference. H0: A is ELO0 Elo stronger, H1: ELO1 (logistic Elo,
# alpha = beta = 0.05, pentanomial). Both play the same opening twice with colours swapped.
# Environment (defaults): FASTCHESS (fastchess), ST (0.1 s per move), THREADS_A / THREADS_B (1),
# HASH (64 MB), WEIGHTS (weights/weights.bin of this repository), CONCURRENCY (11 cores divided by
# the larger thread count), ROUNDS (3000 pairs at most), SEED (1, first opening), OUT (logs/).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 2 ] || { sed -n '2,12p' "$0"; exit 2; }
A=$1 B=$2 ELO0=${3:-0} ELO1=${4:-10}
FASTCHESS=${FASTCHESS:-fastchess}
ST=${ST:-0.1} HASH=${HASH:-64} ROUNDS=${ROUNDS:-3000} SEED=${SEED:-1}
THREADS_A=${THREADS_A:-1} THREADS_B=${THREADS_B:-1}
WEIGHTS=${WEIGHTS:-$ROOT/weights/weights.bin}
MAX_THREADS=$(( THREADS_A > THREADS_B ? THREADS_A : THREADS_B ))
CONCURRENCY=${CONCURRENCY:-$(( 11 / MAX_THREADS > 0 ? 11 / MAX_THREADS : 1 ))}
OUT=${OUT:-$ROOT/logs}
mkdir -p "$OUT"
NAME="sprt-$(date +%Y%m%d-%H%M%S)"
"$FASTCHESS" \
  -engine cmd="$A" name=A option.Threads="$THREADS_A" \
  -engine cmd="$B" name=B option.Threads="$THREADS_B" \
  -each proto=uci st="$ST" option.Hash="$HASH" option.Weights="$WEIGHTS" \
  -openings file="$ROOT/match/openings.pgn" format=pgn order=sequential start="$SEED" \
  -rounds "$ROUNDS" -repeat -maxmoves 148 \
  -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 model=logistic \
  -concurrency "$CONCURRENCY" -report penta=true -autosaveinterval 0 -ratinginterval 20 \
  -pgnout file="$OUT/$NAME.pgn" notation=uci \
  2>&1 | tee "$OUT/$NAME.log"
