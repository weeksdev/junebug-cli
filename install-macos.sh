#!/bin/sh
# Build and install (or replace) the latest Febo CLI on macOS.
# Installs to ~/.local/bin/febo — the same convention Claude Code uses —
# so no sudo is needed. Re-run any time to upgrade to the current checkout.
set -eu

cd "$(dirname "$0")"

if ! command -v cargo > /dev/null 2>&1; then
    echo "error: cargo (Rust) is required. Install from https://rustup.rs" >&2
    exit 1
fi

echo "building febo (release)…"
cargo build --release --quiet

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
install -m 755 target/release/febo "$BIN_DIR/febo"

VERSION=$("$BIN_DIR/febo" --version)
echo "installed $VERSION to $BIN_DIR/febo"

case ":$PATH:" in
    *":$BIN_DIR:"*)
        echo "run: febo"
        ;;
    *)
        echo ""
        echo "$BIN_DIR is not on your PATH. Add it with:"
        echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
        ;;
esac
