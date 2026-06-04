#!/bin/bash
# Build gen_qcd_hasenbusch_tune on Perlmutter — PURE-GRID (no QUDA).
#
# Compiles against an IN-TREE Grid-TXQCD build (build/Grid/libGrid.a; no `make install`).
# grid-config alone is NOT sufficient in-tree: it omits the Grid source/build include
# paths and the libGrid.a -L path, so we add them here (mirrors production/Makefile's
# `-I.. -I build/Grid` and `-L build/Grid`). No -DGRID_HAVE_QUDA -> the source's
# #ifdef GRID_HAVE_QUDA guards drop the QUDA strange-force path.
#
# Prereq: Grid-TXQCD built WITH LIME (see ../../1_build_grid/PERLMUTTER_BUILD_NOTES.md).
# Usage:  bash build_puregrid.sh        (sources sourceme.sh itself)

# Resolve paths relative to THIS script (works from anywhere).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
HB=$(cd "$HERE/.." && pwd)

GRID_TXQCD=${GRID_TXQCD:-/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/Grid-TXQCD}
source "$GRID_TXQCD/sourceme.sh"          # modules (cudatoolkit/12.9 ...) + MPFR lib path
set -e

GRID_CONFIG=${GRID_CONFIG:-$GRID_TXQCD/build/grid-config}
TXQCD_PROD=${TXQCD_PROD:-$GRID_TXQCD/production}     # for params.h
GRID_BUILD=${GRID_BUILD:-$GRID_TXQCD/build}          # for libGrid.a + generated Config.h

if [ ! -x "$GRID_CONFIG" ]; then
  echo "ERROR: grid-config not found/executable at: $GRID_CONFIG" >&2
  echo "       Build Grid-TXQCD first (1_build_grid/PERLMUTTER_BUILD_NOTES.md)." >&2
  exit 1
fi

SRC=$HB/src/gen_qcd_hasenbusch_tune.cc
BIN=$HB/bin/gen_qcd_hasenbusch_tune

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
