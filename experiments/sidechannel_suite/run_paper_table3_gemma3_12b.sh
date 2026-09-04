#!/usr/bin/env bash
# Thin wrapper for the frozen paper Table 3 gemma3_12b configuration.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_paper_table3.sh" gemma3_12b "$@"
