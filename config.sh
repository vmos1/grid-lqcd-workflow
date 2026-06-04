#!/usr/bin/env bash
# Central configuration — edit GRID_PROFILE to switch between Grid forks.
# Sourced by all build/compile/run scripts.
#
# Machine env is loaded from machines/<machine>.sh.
# Auto-detected via $NERSC_HOST, /lustre2/nplqcd, or uname; or force with:
#   MACHINE=perlmutter source config.sh

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$WORKFLOW_DIR")"
export WORKFLOW_DIR BASE_DIR

# ── Grid profile ──────────────────────────────────────────────────────────────
# "txqcd"    → mlwagman/Grid-TXQCD (TXQCD physics extensions)
# "mainline" → paboyle/Grid        (upstream Grid)
export GRID_PROFILE="txqcd"

case "$GRID_PROFILE" in
    txqcd)
        export GRID_REPO="https://github.com/mlwagman/Grid-TXQCD.git"
        export GRID_SRC="$BASE_DIR/Grid-TXQCD"
        export CONFIGURE_EXTRA="--disable-fermion-reps --disable-gparity"
        ;;
    mainline)
        export GRID_REPO="https://github.com/paboyle/Grid.git"
        export GRID_SRC="$BASE_DIR/Grid-mainline"
        export CONFIGURE_EXTRA=""
        ;;
    *)
        echo "Unknown GRID_PROFILE: '$GRID_PROFILE'. Use 'txqcd' or 'mainline'." >&2
        return 1
        ;;
esac

# ── Paths (derived from profile; machine file may override GRID_BUILD) ────────
export GRID_BUILD="$BASE_DIR/build-$GRID_PROFILE"
export GRID_INSTALL="$BASE_DIR/install-$GRID_PROFILE"
export DEPS_DIR="$BASE_DIR/deps"
export RUNS_DIR="$BASE_DIR/runs"

# ── Machine auto-detection ────────────────────────────────────────────────────
if [ -z "${MACHINE:-}" ]; then
    if [ "${NERSC_HOST:-}" = "perlmutter" ]; then
        MACHINE=perlmutter
    elif [ -d /lustre2/nplqcd ]; then
        MACHINE=lq
    elif [ "$(uname)" = Darwin ]; then
        MACHINE=macos
    fi
fi

# ── Source machine-specific environment ───────────────────────────────────────
if [ -n "${MACHINE:-}" ]; then
    _mf="$WORKFLOW_DIR/machines/${MACHINE}.sh"
    if [ -f "$_mf" ]; then
        source "$_mf"
    else
        echo "Warning: MACHINE='$MACHINE' set but machines/${MACHINE}.sh not found." >&2
    fi
    unset _mf
fi
