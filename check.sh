#!/bin/sh
# Prüft den Rust-Kern: Formatierung, Clippy (Warnungen = Fehler, Regeln in Cargo.toml und clippy.toml),
# Tests und Testabdeckung (mindestens MIN_LINES % der Zeilen, ohne den Einstiegspunkt main.rs).
set -e
cd "$(dirname "$0")"
MIN_LINES=95
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release -q

# Abdeckung: cargo-llvm-cov braucht llvm-cov/llvm-profdata in der LLVM-Version des Compilers.
# Homebrew-Rust passt zu Homebrews LLVM, nicht zu rustups llvm-tools-preview.
if ! cargo llvm-cov --version >/dev/null 2>&1; then
  echo "check.sh: cargo-llvm-cov fehlt (cargo install cargo-llvm-cov)" >&2
  exit 1
fi
LLVM_BIN="$(brew --prefix llvm 2>/dev/null)/bin"
if [ -x "$LLVM_BIN/llvm-cov" ]; then
  export LLVM_COV="$LLVM_BIN/llvm-cov" LLVM_PROFDATA="$LLVM_BIN/llvm-profdata"
fi
# Ausgabe erst in eine Datei, damit ein Fehlschlag (Test oder Abdeckung) nicht in einer Pipe verloren geht.
COV_OUT="$(mktemp)"
trap 'rm -f "$COV_OUT"' EXIT
if ! CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}/coverage" cargo llvm-cov --quiet --summary-only \
  --ignore-filename-regex 'main\.rs' --fail-under-lines "$MIN_LINES" >"$COV_OUT" 2>&1; then
  cat "$COV_OUT" >&2
  echo "check.sh: Abdeckungslauf fehlgeschlagen" >&2
  exit 1
fi
tail -1 "$COV_OUT"
echo "check.sh: alles in Ordnung"
