#!/bin/bash
# Build the standalone Grid-multigrid rung-solver test (src/grid_mg/test_grid_mg_odd.cc) on
# Perlmutter against the STOCK (patched) Grid tree -- pure Grid, no QUDA. Modelled on
# build_driver_stock.sh: same environment (config.sh -> machines/perlmutter.sh), same stock
# tree resolution, same shadowing and ldd checks. Spec:
# __docs/2026_09_29_pure_grid_m2_mg_solver_design.md section 2 item 8, section 4 item 1.
#
# Include order: -I$HB/include (the carried pseudofermion headers + params.h) FIRST, then
# -I$HB/src (so "grid_mg/..." resolves), then the staged Grid SOURCE tree and its build dir,
# and only then grid-config's flags. ⛔ This differs from build_driver_stock.sh, which puts
# grid-config's flags first: they contain -I<prefix>/include, a `make install` made before
# patch 04, so its GeneralCoarsenedMatrix.h is unpatched and would shadow the source tree.
#
# Extra checks over build_driver_stock.sh: the source tree must carry patch 04
# (04-coarsen-operator-honour-subspace-checkerboard.patch), without which CoarsenOperator
# aborts on the Odd subspace this test exists to exercise, and the compiler's dependency list
# must show that the patched copy was the one compiled. The patch is header-only, so the
# 15:27 libGrid.a is still the right library.
#
# One single nvcc compile, niced (login-node safe); no make, no -j.
#
# Env overrides:
#   GRID_CONFIG   grid-config of the stock in-tree build
#                 (default ${PSCRATCH}/grid_pure_hmc/stock-grid/<commit>/Grid/build/grid-config)
#   GRID_SOURCE   Grid source root (default: the directory above GRID_CONFIG's)
#   SRC, BIN      source and output (default src/grid_mg/test_grid_mg_odd.cc -> bin/test_grid_mg_odd)
#   COMPILE_ONLY  1 = compile SRC to the object file BIN with -c (no link, no ldd); used for
#                 the header self-containment check of src/grid_mg/*.h
#   NICE          nice increment for the compile (default 10)
#
# Usage:  bash build_test_grid_mg.sh > <log> 2>&1

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
if [ ! -d "$HB/src/grid_mg" ]; then
  echo "ERROR: $HB/src/grid_mg missing" >&2
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

# Patch 04 must be in the tree (see header).
GCM=$GRID_SOURCE/Grid/algorithms/multigrid/GeneralCoarsenedMatrix.h
N04=$(grep -c 'Honour the checkerboard of the subspace' "$GCM" || true)
if [ "$N04" -lt 2 ]; then
  echo "ERROR: $GCM does not carry patch 04 (found $N04 of 2 hunks)." >&2
  echo "       Apply 1_build_grid/patches/stock-grid-3d3eff86/04-*.patch to the staged tree." >&2
  exit 1
fi
echo "OK: patch 04 present in $GCM ($N04 hunks)."

SRC=${SRC:-$HB/src/grid_mg/test_grid_mg_odd.cc}
BIN=${BIN:-$HB/bin/test_grid_mg_odd}
COMPILE_ONLY=${COMPILE_ONLY:-0}
NICE=${NICE:-10}
mkdir -p "$(dirname "$BIN")"

CXX=$("$GRID_CONFIG" --cxx)
# ⛔ INCLUDE ORDER: the staged SOURCE tree must come BEFORE grid-config's flags. grid-config
# --cxxflags carries -I<prefix>/include, and the staged tree has a `make install` from 15:27,
# i.e. BEFORE patch 04 was applied (16:07): its GeneralCoarsenedMatrix.h is the unpatched one
# (the only header that differs, checked with diff -rq on 2026-09-29). build_driver_stock.sh
# appends -I$GRID_SOURCE AFTER grid-config's flags, so it compiles the INSTALLED headers.
# Config.h is found through -I$GRID_BUILD/Grid (GridStd.h does #include "Config.h").
CXXFLAGS="-I$HB/include -I$HB/src -I$GRID_SOURCE -I$GRID_BUILD/Grid $("$GRID_CONFIG" --cxxflags)"
for PFX in $("$GRID_CONFIG" --cxxflags); do
  case "$PFX" in
    -I*) d=${PFX#-I}
         if [ -f "$d/Grid/algorithms/multigrid/GeneralCoarsenedMatrix.h" ] && \
            ! cmp -s "$d/Grid/algorithms/multigrid/GeneralCoarsenedMatrix.h" "$GCM"; then
           echo "NOTE: $d/Grid carries a GeneralCoarsenedMatrix.h WITHOUT patch 04; it is shadowed by -I$GRID_SOURCE (placed first)."
         fi ;;
  esac
done
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
# The dependency file (written during the compile, no extra pass) proves which
# GeneralCoarsenedMatrix.h was compiled: it must be the patched source-tree copy.
check_gcm_used() {
  local used
  used=$(tr ' \\' '\n\n' < "$DEPS" | grep 'multigrid/GeneralCoarsenedMatrix.h$' | head -1)
  if [ -z "$used" ] || [ "$(readlink -f "$used")" != "$(readlink -f "$GCM")" ]; then
    echo "ERROR: compiled GeneralCoarsenedMatrix.h = '${used:-none}', expected $GCM (patch 04)." >&2
    exit 1
  fi
  echo "OK: compiled GeneralCoarsenedMatrix.h is the patched one: $used"
}

if [ "$COMPILE_ONLY" = "1" ]; then
  echo "Compiling (-c, no link) against STOCK Grid ..."
  nice -n "$NICE" $CXX $CXXFLAGS -MD -MF "$DEPS" -c -o "$BIN" "$SRC"
  echo "Done (object): $BIN"
  check_gcm_used
  exit 0
fi

echo "Building PURE-GRID test against STOCK Grid (no -DGRID_HAVE_QUDA) ..."
nice -n "$NICE" $CXX $CXXFLAGS -MD -MF "$DEPS" -o "$BIN" "$SRC" $LDFLAGS $LIBS

echo "Done: $BIN"
check_gcm_used
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
