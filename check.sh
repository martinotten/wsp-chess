#!/bin/sh
# Prüft den Rust-Kern: Formatierung, Clippy (Warnungen = Fehler, Regeln in Cargo.toml), Tests.
set -e
cd "$(dirname "$0")"
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release -q
echo "check.sh: alles in Ordnung"
