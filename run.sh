#!/bin/sh
# Build and run Febo CLI with this project root as the workspace.
# Usage: ./run.sh [febo flags] "prompt"
# Examples:
#   ./run.sh "What files are in this directory?"
#   ./run.sh --provider deepseek --permission ask "Create hello.txt containing hi"
#   ./run.sh exec --json --permission workspace-write "prompt"
set -eu
cd "$(dirname "$0")"
cargo build --quiet
exec target/debug/febo "$@"
