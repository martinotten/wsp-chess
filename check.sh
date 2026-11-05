#!/bin/sh
# Checks the Rust core: formatting, Clippy (warnings = errors, rules in Cargo.toml and clippy.toml),
# tests and test coverage (at least MIN_LINES % of lines, excluding the entry point main.rs).
set -e
cd "$(dirname "$0")"
MIN_LINES=95
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release -q

# Coverage: cargo-llvm-cov needs llvm-cov/llvm-profdata matching the compiler's LLVM version.
# Homebrew Rust matches Homebrew's LLVM, not rustup's llvm-tools-preview.
if ! cargo llvm-cov --version >/dev/null 2>&1; then
  echo "check.sh: cargo-llvm-cov is missing (cargo install cargo-llvm-cov)" >&2
  exit 1
fi
LLVM_BIN="$(brew --prefix llvm 2>/dev/null)/bin"
if [ -x "$LLVM_BIN/llvm-cov" ]; then
  export LLVM_COV="$LLVM_BIN/llvm-cov" LLVM_PROFDATA="$LLVM_BIN/llvm-profdata"
fi
# Write output to a file first so that a failure (test or coverage) is not lost in a pipe.
COV_OUT="$(mktemp)"
trap 'rm -f "$COV_OUT"' EXIT
if ! CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}/coverage" cargo llvm-cov --quiet --summary-only \
  --ignore-filename-regex 'main\.rs' --fail-under-lines "$MIN_LINES" >"$COV_OUT" 2>&1; then
  cat "$COV_OUT" >&2
  echo "check.sh: coverage run failed" >&2
  exit 1
fi
tail -1 "$COV_OUT"
echo "check.sh: all checks passed"
