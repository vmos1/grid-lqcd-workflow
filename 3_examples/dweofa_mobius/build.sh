#!/usr/bin/env bash
# Compile the DW EOFA Mobius HMC driver against the installed Grid library.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config.sh"

GRID_CONFIG="$GRID_INSTALL/bin/grid-config"
SRC="$SCRIPT_DIR/src/dweofa_mobius_HSDM_v3.cc"
BIN="$SCRIPT_DIR/bin/dweofa_mobius_HSDM_v3"

mkdir -p "$SCRIPT_DIR/bin"

echo "==> Compiling dweofa_mobius_HSDM_v3..."
"$MPICXX" \
    $("$GRID_CONFIG" --cxxflags) \
    "$SRC" -o "$BIN" \
    $("$GRID_CONFIG" --ldflags) \
    $("$GRID_CONFIG" --libs)

echo "✓ Built: $BIN"
