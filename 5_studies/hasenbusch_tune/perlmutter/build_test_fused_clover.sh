#!/bin/bash
# Build the standalone fused-clover-force test (src/clover_force/test_fused_clover_force.cc) on
# Perlmutter against the STOCK (patched) Grid tree -- pure Grid, no QUDA. Modelled on
# build_test_grid_mg.sh: same environment (config.sh -> machines/perlmutter.sh), same stock tree
# resolution, include order, shadowing guard and ldd checks. Spec: M5 fix 2 of
# __docs/2026_09_30_pure_grid_force_cost_analysis.md.
#
# Include order: -I$HB/include (carried headers + params.h) FIRST, then -I$HB/src (so
# "clover_force/..." resolves), then the staged Grid SOURCE tree and its build dir, and only then
# grid-config's flags (whose -I<prefix>/include is a pre-patch-04 `make install`; see
# build_test_grid_mg.sh).
#
# Extra check: the staged source tree must carry patch 01 (the MooDeriv/MeeDeriv bodies this test
# compares against). The bodies are compiled into libGrid.a, so this is a proxy; an unpatched
# library aborts at the first MooDeriv (GRID_ASSERT(0)).
#
# One single nvcc compile, niced (login-node safe); no make, no -j.
#
# Env overrides:
#   GRID_CONFIG   grid-config of the stock in-tree build
#                 (default ${PSCRATCH}/grid_pure_hmc/stock-grid/<commit>/Grid/build/grid-config)
#   GRID_SOURCE   Grid source root (default: the directory above GRID_CONFIG's)
#   SRC, BIN      source and output (default src/clover_force/test_fused_clover_force.cc ->
#                 bin/test_fused_clover_force)
#   NICE          nice increment for the compile (default 10)
#
# Usage:  bash build_test_fused_clover.sh > <log> 2>&1

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
  echo "ERROR: stock Grid grid-config not found/executable at:" >&2
  echo "       $GRID_CONFIG" >&2
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
if [ ! -f "$GRID_SOURCE/Grid/Grid.h" ]; then
  echo "ERROR: GRID_SOURCE does not look like a Grid source root (no Grid/Grid.h):" >&2
  echo "       $GRID_SOURCE" >&2
  exit 1
fi
if [ ! -f "$HB/include/params.h" ]; then
  echo "ERROR: $HB/include/params.h missing; see $HB/include/PROVENANCE.md" >&2
  exit 1
fi
if [ ! -f "$HB/src/clover_force/fused_clover_force.h" ]; then
  echo "ERROR: $HB/src/clover_force/fused_clover_force.h missing" >&2
  exit 1
fi

# Same shadowing guard as build_driver_stock.sh.
for h in $(cd "$HB/include" && find Grid -name '*.h'); do
  if [ -e "$GRID_SOURCE/$h" ]; then
    echo "ERROR: $GRID_SOURCE/$h exists; -I$HB/include would shadow it." >&2
    echo "       This does not look like a stock Grid tree. Refusing to build." >&2
    exit 1
  fi
done

# Patch 01 must be in the tree (see header): both compact bodies carry this comment.
CWCF=$GRID_SOURCE/Grid/qcd/action/fermion/implementation/CompactWilsonCloverFermionImplementation.h
N01=$(grep -c 'Clover-diagonal force on the' "$CWCF" || true)
if [ "$N01" -lt 2 ]; then
  echo "ERROR: $CWCF does not carry patch 01 (found $N01 of 2 body comments)." >&2
  echo "       Apply 1_build_grid/patches/stock-grid-3d3eff86/01-*.patch to the staged tree." >&2
  exit 1
fi
echo "OK: patch 01 present in $CWCF ($N01 body comments)."

SRC=${SRC:-$HB/src/clover_force/test_fused_clover_force.cc}
BIN=${BIN:-$HB/bin/test_fused_clover_force}
NICE=${NICE:-10}
mkdir -p "$(dirname "$BIN")"

CXX=$("$GRID_CONFIG" --cxx)
CXXFLAGS="-I$HB/include -I$HB/src -I$GRID_SOURCE -I$GRID_BUILD/Grid $("$GRID_CONFIG" --cxxflags)"
LDFLAGS="$("$GRID_CONFIG" --ldflags) -L$GRID_BUILD/Grid"
LIBS=$("$GRID_CONFIG" --libs)

echo "Date:        $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)"
echo "Compiler:    $CXX"
echo "GRID_CONFIG: $GRID_CONFIG"
echo "GRID_GIT:    $("$GRID_CONFIG" --git 2>/dev/null || echo unknown)"
echo "GRID_SOURCE: $GRID_SOURCE"
echo "GRID_BUILD:  $GRID_BUILD"
echo "Include:     $HB/include (first), $HB/src (second)"
echo "Source:      $SRC"
echo "Output:      $BIN"
echo "CXXFLAGS:    $CXXFLAGS"

DEPS=$BIN.d
echo "Building PURE-GRID test against STOCK Grid (no -DGRID_HAVE_QUDA) ..."
nice -n "$NICE" $CXX $CXXFLAGS -MD -MF "$DEPS" -o "$BIN" "$SRC" $LDFLAGS $LIBS

echo "Done: $BIN"
# The dependency file proves the fused header compiled is this tree's copy.
FCF_USED=$(tr ' \\' '\n\n' < "$DEPS" | grep 'clover_force/fused_clover_force.h$' | head -1)
if [ -z "$FCF_USED" ] || [ "$(readlink -f "$FCF_USED")" != "$(readlink -f "$HB/src/clover_force/fused_clover_force.h")" ]; then
  echo "ERROR: compiled fused_clover_force.h = '${FCF_USED:-none}', expected $HB/src/clover_force/fused_clover_force.h" >&2
  exit 1
fi
echo "OK: compiled fused_clover_force.h: $FCF_USED"
echo "--- sanity: libGrid statically in, mpfr/cuda/mpi dynamic, NO libquda, 0 'not found' ---"
LDD_OUT=$(ldd "$BIN" 2>&1) || { echo "ERROR: ldd failed on $BIN:" >&2; echo "$LDD_OUT" >&2; exit 1; }

echo "$LDD_OUT" | grep -iE 'quda' && echo "WARNING: QUDA linked unexpectedly" \
  || echo "OK: no QUDA linked (pure-Grid)."

NOTFOUND=$(echo "$LDD_OUT" | grep -c 'not found' || true)
if [ "$NOTFOUND" -ne 0 ]; then
  echo "ERROR: ldd reports $NOTFOUND unresolved shared librar(y/ies):" >&2
  echo "$LDD_OUT" | grep 'not found' >&2
  echo "       Check LD_LIBRARY_PATH from machines/perlmutter.sh (PE-bump cudart trap)." >&2
  exit 1
fi
echo "OK: ldd reports 0 'not found'."
