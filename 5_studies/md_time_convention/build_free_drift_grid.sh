#!/bin/bash
# Build free_drift_grid.cc (or PROG=pure_gauge_hmc_grid) against the stock in-tree Grid build
# (3d3eff86), pure Grid, no QUDA.
# Same recipe as 5_studies/hasenbusch_tune/perlmutter/build_test_fused_clover.sh: environment from
# config.sh -> machines/perlmutter.sh, the staged Grid source tree and its build dir before
# grid-config's flags, one niced nvcc compile (login-node safe), ldd check for the PE-bump trap.
# The binary goes to $PSCRATCH (build product, outside both git trees).
#
# Env overrides: PROG (default free_drift_grid), GRID_CONFIG, BIN, NICE (default 10).
# Usage: bash build_free_drift_grid.sh > <log> 2>&1

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORKFLOW=$(cd "$HERE/../.." && pwd)

_GRID_CONFIG_IN=${GRID_CONFIG:-}
source "$WORKFLOW/config.sh"
set -e

STOCK_GRID_COMMIT=3d3eff86f3664f4068af9814ae92150406bb812d
GRID_CONFIG=${_GRID_CONFIG_IN:-${PSCRATCH}/grid_pure_hmc/stock-grid/${STOCK_GRID_COMMIT}/Grid/build/grid-config}
[ -x "$GRID_CONFIG" ] || { echo "ERROR: grid-config not found at $GRID_CONFIG" >&2; exit 1; }
GRID_BUILD=$(cd "$(dirname "$GRID_CONFIG")" && pwd)
GRID_SOURCE=$(cd "$GRID_BUILD/.." && pwd)
[ -f "$GRID_BUILD/Grid/libGrid.a" ] || { echo "ERROR: no libGrid.a in $GRID_BUILD/Grid" >&2; exit 1; }

PROG=${PROG:-free_drift_grid}               # or pure_gauge_hmc_grid
SRC=$HERE/$PROG.cc
BIN=${BIN:-${PSCRATCH}/grid_pure_hmc/md_time/bin/$PROG}
NICE=${NICE:-10}
mkdir -p "$(dirname "$BIN")"

CXX=$("$GRID_CONFIG" --cxx)
CXXFLAGS="-I$GRID_SOURCE -I$GRID_BUILD/Grid $("$GRID_CONFIG" --cxxflags)"
LDFLAGS="$("$GRID_CONFIG" --ldflags) -L$GRID_BUILD/Grid"
LIBS=$("$GRID_CONFIG" --libs)

echo "Date:        $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)"
echo "GRID_CONFIG: $GRID_CONFIG"
echo "GRID_GIT:    $("$GRID_CONFIG" --git 2>/dev/null || echo unknown)"
echo "Source:      $SRC"
echo "Output:      $BIN"

nice -n "$NICE" $CXX $CXXFLAGS -o "$BIN" "$SRC" $LDFLAGS $LIBS
echo "Done: $BIN"

NOTFOUND=$(ldd "$BIN" 2>&1 | grep -c 'not found' || true)
if [ "$NOTFOUND" -ne 0 ]; then
  echo "ERROR: ldd reports $NOTFOUND unresolved libraries (PE-bump cudart trap?):" >&2
  ldd "$BIN" | grep 'not found' >&2
  exit 1
fi
echo "OK: ldd reports 0 'not found'."
