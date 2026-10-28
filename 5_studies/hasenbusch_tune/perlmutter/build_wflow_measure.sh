#!/bin/bash
# Build the offline Wilson-flow measurement (src/wilson_flow/wflow_measure.cc, 2026-10-26) on
# Perlmutter against the STOCK (patched) Grid tree -- pure Grid, no QUDA. Modelled on
# build_test_fused_clover.sh: same environment (config.sh -> machines/perlmutter.sh), same stock
# tree resolution and ldd checks. The program needs only Grid (WilsonFlow, WilsonLoops, the
# NERSC/ILDG readers), so none of this study's carried headers are on the include path.
#
# One single nvcc compile, niced (login-node safe per machines/perlmutter.md); no make, no -j.
#
# Env overrides:
#   GRID_CONFIG   grid-config of the stock in-tree build
#                 (default ${PSCRATCH}/grid_pure_hmc/stock-grid/<commit>/Grid/build/grid-config)
#   GRID_SOURCE   Grid source root (default: the directory above GRID_CONFIG's)
#   SRC, BIN      source and output (default src/wilson_flow/wflow_measure.cc -> bin/wflow_measure)
#   NICE          nice increment for the compile (default 10)
#
# Usage:  bash build_wflow_measure.sh > <log> 2>&1

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
HB=$(cd "$HERE/.." && pwd)
WORKFLOW=$(cd "$HERE/../../.." && pwd)

# Capture caller overrides BEFORE config.sh, so nothing it exports can leak in.
_GRID_CONFIG_IN=${GRID_CONFIG:-}
_GRID_SOURCE_IN=${GRID_SOURCE:-}

source "$WORKFLOW/config.sh"              # modules + MPFR lib path (via machines/perlmutter.sh)
set -e

STOCK_GRID_COMMIT=3d3eff86f3664f4068af9814ae92150406bb812d

if [ -n "$_GRID_CONFIG_IN" ]; then
  GRID_CONFIG=$_GRID_CONFIG_IN
elif [ -n "${PSCRATCH:-}" ]; then
  GRID_CONFIG=${PSCRATCH}/grid_pure_hmc/stock-grid/${STOCK_GRID_COMMIT}/Grid/build/grid-config
else
  echo "ERROR: PSCRATCH is not set and GRID_CONFIG was not given." >&2
  exit 1
fi
if [ ! -x "$GRID_CONFIG" ]; then
  echo "ERROR: stock Grid grid-config not found/executable at: $GRID_CONFIG" >&2
  exit 1
fi

GRID_BUILD=$(cd "$(dirname "$GRID_CONFIG")" && pwd)
if [ -n "$_GRID_SOURCE_IN" ]; then
  GRID_SOURCE=$_GRID_SOURCE_IN
else
  GRID_SOURCE=$(cd "$GRID_BUILD/.." && pwd)
fi
if [ ! -f "$GRID_BUILD/Grid/libGrid.a" ]; then
  echo "ERROR: libGrid.a not found at: $GRID_BUILD/Grid/libGrid.a" >&2
  exit 1
fi
if [ ! -f "$GRID_SOURCE/Grid/qcd/smearing/WilsonFlow.h" ]; then
  echo "ERROR: GRID_SOURCE has no Grid/qcd/smearing/WilsonFlow.h: $GRID_SOURCE" >&2
  exit 1
fi

SRC=${SRC:-$HB/src/wilson_flow/wflow_measure.cc}
BIN=${BIN:-$HB/bin/wflow_measure}
NICE=${NICE:-10}
mkdir -p "$(dirname "$BIN")"

CXX=$("$GRID_CONFIG" --cxx)
CXXFLAGS="-I$GRID_SOURCE -I$GRID_BUILD/Grid $("$GRID_CONFIG" --cxxflags)"
LDFLAGS="$("$GRID_CONFIG" --ldflags) -L$GRID_BUILD/Grid"
LIBS=$("$GRID_CONFIG" --libs)

echo "Date:        $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)"
echo "Compiler:    $CXX"
echo "GRID_CONFIG: $GRID_CONFIG"
echo "GRID_GIT:    $("$GRID_CONFIG" --git 2>/dev/null || echo unknown)"
echo "GRID_SOURCE: $GRID_SOURCE"
echo "GRID_BUILD:  $GRID_BUILD"
echo "Source:      $SRC"
echo "Output:      $BIN"
echo "CXXFLAGS:    $CXXFLAGS"

echo "Building PURE-GRID wflow_measure against STOCK Grid (no -DGRID_HAVE_QUDA) ..."
nice -n "$NICE" $CXX $CXXFLAGS -o "$BIN" "$SRC" $LDFLAGS $LIBS
echo "Done: $BIN"

echo "--- sanity: libGrid statically in, mpfr/cuda/mpi dynamic, NO libquda, 0 'not found' ---"
LDD_OUT=$(ldd "$BIN" 2>&1) || { echo "ERROR: ldd failed on $BIN:" >&2; echo "$LDD_OUT" >&2; exit 1; }
echo "$LDD_OUT" | grep -iE 'quda' && echo "WARNING: QUDA linked unexpectedly" \
  || echo "OK: no QUDA linked (pure-Grid)."
NOTFOUND=$(echo "$LDD_OUT" | grep -c 'not found' || true)
if [ "$NOTFOUND" -ne 0 ]; then
  echo "ERROR: ldd reports $NOTFOUND unresolved shared librar(y/ies):" >&2
  echo "$LDD_OUT" | grep 'not found' >&2
  exit 1
fi
echo "OK: ldd reports 0 'not found'."
