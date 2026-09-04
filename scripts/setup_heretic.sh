#!/usr/bin/env bash
# Prepare the patched Heretic dependency without vendoring it in this repository.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERETIC_DIR="${HERETIC_DIR:-$ROOT_DIR/external/heretic}"
HERETIC_UPSTREAM_REPO="${HERETIC_UPSTREAM_REPO:-https://github.com/p-e-w/heretic.git}"
HERETIC_UPSTREAM_REF="${HERETIC_UPSTREAM_REF:-v1.2.0}"
PATCH_FILE="$ROOT_DIR/patches/heretic/bait-and-recover-heretic-v1.2.0.patch"
PYTHON="${PYTHON:-python}"

if [[ -e "$HERETIC_DIR" && "${HERETIC_OVERWRITE:-0}" != "1" ]]; then
  echo "[ERROR] $HERETIC_DIR already exists."
  echo "        Set HERETIC_OVERWRITE=1 to replace it, or set HERETIC_DIR to another path."
  exit 2
fi

if [[ -e "$HERETIC_DIR" ]]; then
  mv "$HERETIC_DIR" "${HERETIC_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$(dirname "$HERETIC_DIR")"

git clone --branch "$HERETIC_UPSTREAM_REF" --depth 1 "$HERETIC_UPSTREAM_REPO" "$HERETIC_DIR"
(
  cd "$HERETIC_DIR"
  git apply --3way "$PATCH_FILE"
)
"$PYTHON" -m pip install -e "$HERETIC_DIR"

echo "Patched Heretic is ready at: $HERETIC_DIR"
