#!/usr/bin/env bash
# Compile executables against the installed Grid library.
#
# No argument:        build all programs in Grid-*/production/
# With argument:      compile a single .cc file from 2_compile/src/
#
# Usage:
#   ./2_compile/build_exec.sh
#   ./2_compile/build_exec.sh myprogram.cc
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

mkdir -p "$SCRIPT_DIR/src" "$SCRIPT_DIR/bin"

if [ -z "$1" ]; then
    # ── No argument: build all production/ programs ───────────────────────────
    echo "==> No source file specified — building production/ programs..."

    PROD_DIR="$GRID_SRC/production"
    if [ ! -d "$PROD_DIR" ]; then
        echo "Error: production/ directory not found at $PROD_DIR"
        echo "  (mainline Grid does not have a production/ directory — pass a .cc file instead)"
        exit 1
    fi

    GRID_CONFIG_PATH="$GRID_INSTALL/bin/grid-config"
    sed -i.bak "s|^GRID_CONFIG.*|GRID_CONFIG = $GRID_CONFIG_PATH|" "$PROD_DIR/Makefile"

    make -C "$PROD_DIR" -j"$(sysctl -n hw.logicalcpu)" 2>&1 | tee "$SCRIPT_DIR/compile.log"

    echo ""
    echo "✓ Production executables built in $PROD_DIR"
else
    # ── Argument given: compile a single .cc from src/ ────────────────────────
    SRC_FILE="$1"
    SRC_PATH="$SCRIPT_DIR/src/$SRC_FILE"

    if [ ! -f "$SRC_PATH" ]; then
        echo "Error: $SRC_PATH not found"
        echo "Place your .cc file in $SCRIPT_DIR/src/ and re-run."
        exit 1
    fi

    EXEC_NAME="$(basename "$SRC_FILE" .cc)"
    echo "==> Compiling $SRC_FILE -> bin/$EXEC_NAME ..."

    export GRID_INSTALL MPICXX OMPI_CXX
    make -C "$SCRIPT_DIR" "bin/$EXEC_NAME" 2>&1 | tee "$SCRIPT_DIR/compile.log"

    echo ""
    echo "✓ Executable: $SCRIPT_DIR/bin/$EXEC_NAME"
fi
