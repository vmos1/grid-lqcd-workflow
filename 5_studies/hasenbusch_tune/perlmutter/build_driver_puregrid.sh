#!/bin/bash
# Build the hasenbusch_tune driver on Perlmutter — PURE-GRID (no QUDA).
#
# Compiles against an IN-TREE Grid-TXQCD build (build/Grid/libGrid.a; no `make install`).
# grid-config alone is NOT sufficient in-tree: it omits the Grid source/build include
# paths and the libGrid.a -L path, so we add them here (mirrors production/Makefile's
# `-I.. -I build/Grid` and `-L build/Grid`). No -DGRID_HAVE_QUDA -> the source's
# #ifdef GRID_HAVE_QUDA guards drop the QUDA strange-force path.
#
# Prereq: Grid-TXQCD built WITH LIME (see ../../1_build_grid/PERLMUTTER_BUILD_NOTES.md).
# Usage:  bash build_driver_puregrid.sh (sources workflow config.sh → machines/perlmutter.sh)

# Resolve paths relative to THIS script (works from anywhere).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
HB=$(cd "$HERE/.." && pwd)

WORKFLOW=$(cd "$HERE/../../.." && pwd)
source "$WORKFLOW/config.sh"              # modules + MPFR lib path (via machines/perlmutter.sh)
set -e

GRID_TXQCD=${GRID_TXQCD:-$GRID_SRC}      # GRID_SRC set by config.sh
GRID_CONFIG=${GRID_CONFIG:-$GRID_TXQCD/build/grid-config}
TXQCD_PROD=${TXQCD_PROD:-$GRID_TXQCD/production}     # for params.h
# libGrid.a + generated Config.h live next to the grid-config actually in use.
# Derive from GRID_CONFIG (NOT the inherited GRID_BUILD, which config.sh points
# at an out-of-tree build-$PROFILE dir that this in-tree builder does not use).
GRID_BUILD=$(cd "$(dirname "$GRID_CONFIG")" && pwd)

if [ ! -x "$GRID_CONFIG" ]; then
  echo "ERROR: grid-config not found/executable at: $GRID_CONFIG" >&2
  echo "       Build Grid-TXQCD first (1_build_grid/PERLMUTTER_BUILD_NOTES.md)." >&2
  exit 1
fi

# Default builds the compact-clover driver.  SRC/BIN are overridable to build other
# variants, e.g. SRC=$HB/src/gen_qcd_hasenbusch_tune_compact_schur.cc BIN=$HB/bin/..._compact_schur
SRC=${SRC:-$HB/src/gen_qcd_hasenbusch_tune_compact.cc}
BIN=${BIN:-$HB/bin/gen_qcd_hasenbusch_tune_compact}

CXX=$($GRID_CONFIG --cxx)
CXXFLAGS="$($GRID_CONFIG --cxxflags) -I$TXQCD_PROD -I$GRID_TXQCD -I$GRID_BUILD/Grid"
LDFLAGS="$($GRID_CONFIG --ldflags) -L$GRID_BUILD/Grid"
LIBS=$($GRID_CONFIG --libs)

echo "Compiler: $CXX"
echo "Source:   $SRC"
echo "Binary:   $BIN"
echo "Building PURE-GRID (no -DGRID_HAVE_QUDA) ..."
$CXX $CXXFLAGS -o "$BIN" "$SRC" $LDFLAGS $LIBS

echo "Done: $BIN"
echo "--- sanity: should show libGrid statically in, mpfr/cuda/mpi dynamic, NO libquda ---"
ldd "$BIN" 2>/dev/null | grep -iE 'quda' && echo "WARNING: QUDA linked unexpectedly" \
  || echo "OK: no QUDA linked (pure-Grid)."
