#!/bin/sh
# Build and install (or replace) the latest Junebug CLI on macOS.
# Installs to ~/.local/bin/junebug — the same convention Claude Code uses —
# so no sudo is needed. Re-run any time to upgrade to the current checkout.
set -eu

cd "$(dirname "$0")"

if ! command -v cargo > /dev/null 2>&1; then
    echo "error: cargo (Rust) is required. Install from https://rustup.rs" >&2
    exit 1
fi

# The search tool shells out to ripgrep on a sanitized PATH
# (/opt/homebrew/bin, /usr/local/bin, and the system directories).
if [ ! -x /opt/homebrew/bin/rg ] && [ ! -x /usr/local/bin/rg ] && [ ! -x /usr/bin/rg ]; then
    if command -v brew > /dev/null 2>&1; then
        echo "installing ripgrep (required by the search tool)…"
        brew install ripgrep
    else
        echo "warning: ripgrep (rg) is not installed and Homebrew was not found." >&2
        echo "         The search tool will not work until you install it:" >&2
        echo "         https://github.com/BurntSushi/ripgrep#installation" >&2
    fi
fi

echo "building junebug (release)…"
cargo build --release --quiet

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
install -m 755 target/release/junebug "$BIN_DIR/junebug"

VERSION=$("$BIN_DIR/junebug" --version)
echo "installed $VERSION to $BIN_DIR/junebug"

case ":$PATH:" in
    *":$BIN_DIR:"*)
        echo "run: junebug"
        ;;
    *)
        echo ""
        echo "$BIN_DIR is not on your PATH. Add it with:"
        echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
        ;;
esac
