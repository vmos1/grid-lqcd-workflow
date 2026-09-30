#!/bin/bash
# Build the hasenbusch_tune compact-Schur driver on Perlmutter against a STOCK
# (patched) Grid tree -- PURE-GRID (no QUDA), no include path into Grid-TXQCD.
#
# The seven project-authored files the driver needs and stock Grid lacks (six
# Grid/qcd/action/pseudofermion/*.h headers + params.h) are carried in
# ../include; see ../include/PROVENANCE.md. -I$HB/include goes FIRST so the
# driver's <Grid/qcd/action/pseudofermion/...> and "params.h" includes resolve
# there with no source edits. It must shadow nothing, so this script refuses a
# Grid tree that already contains any of those headers (e.g. the fork).
#
# Compiles against an IN-TREE stock Grid build (<Grid>/build/Grid/libGrid.a; no
# `make install`). grid-config alone is NOT sufficient in-tree: it omits the Grid
# source/build include paths and the libGrid.a -L path, so we add them here
# (same as build_driver_puregrid.sh). No -DGRID_HAVE_QUDA -> the driver's
# #ifdef GRID_HAVE_QUDA blocks drop out.
#
# Env overrides:
#   GRID_CONFIG  grid-config of the stock in-tree build
#                (default ${PSCRATCH}/grid_pure_hmc/stock-grid/<commit>/Grid/build/grid-config)
#   GRID_SOURCE  Grid source root (default: two levels above GRID_CONFIG, i.e. .../Grid)
#   SRC, BIN     driver source and output binary
# GRID_BUILD is always the directory containing GRID_CONFIG. config.sh exports
# its own GRID_BUILD/GRID_SRC for the fork build; neither is used here.
#
# Usage:  bash build_driver_stock.sh (sources workflow config.sh -> machines/perlmutter.sh)

# Resolve paths relative to THIS script (works from anywhere).
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
  echo "       The stock Grid library has not been built there yet. Build it (in-tree," >&2
  echo "       WITH LIME) first, or point GRID_CONFIG at an existing stock build." >&2
  exit 1
fi

# libGrid.a + generated Config.h live next to the grid-config actually in use.
GRID_BUILD=$(cd "$(dirname "$GRID_CONFIG")" && pwd)
if [ -n "$_GRID_SOURCE_IN" ]; then
  GRID_SOURCE=$_GRID_SOURCE_IN
else
  GRID_SOURCE=$(cd "$GRID_BUILD/.." && pwd)
fi

if [ ! -f "$GRID_BUILD/Grid/libGrid.a" ]; then
  echo "ERROR: libGrid.a not found at: $GRID_BUILD/Grid/libGrid.a" >&2
  echo "       grid-config exists but the stock Grid library is not built yet" >&2
  echo "       (configure done, make not run or not finished)." >&2
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

# -I$HB/include is first, so a same-named header in the Grid tree would be
# silently shadowed. Stock Grid has none of ours; if one exists this is not a
# stock tree (e.g. the Grid-TXQCD fork -- use build_driver_puregrid.sh for that).
for h in $(cd "$HB/include" && find Grid -name '*.h'); do
  if [ -e "$GRID_SOURCE/$h" ]; then
    echo "ERROR: $GRID_SOURCE/$h exists; -I$HB/include would shadow it." >&2
    echo "       This does not look like a stock Grid tree. Refusing to build." >&2
    exit 1
  fi
done

# Patch 04 (parity-agnostic CoarsenOperator) must be in the tree: the driver's opt-in
# Grid-MG route (HASEN_GRID_MG_RUNGS, src/grid_mg/) coarsens on the Odd checkerboard and
# aborts without it. Same check as build_test_grid_mg.sh.
GCM=$GRID_SOURCE/Grid/algorithms/multigrid/GeneralCoarsenedMatrix.h
N04=$(grep -c 'Honour the checkerboard of the subspace' "$GCM" || true)
if [ "$N04" -lt 2 ]; then
  echo "ERROR: $GCM does not carry patch 04 (found $N04 of 2 hunks)." >&2
  echo "       Apply 1_build_grid/patches/stock-grid-3d3eff86/04-*.patch to the staged tree." >&2
  exit 1
fi
echo "OK: patch 04 present in $GCM ($N04 hunks)."

SRC=${SRC:-$HB/src/gen_qcd_hasenbusch_tune_compact_schur.cc}
BIN=${BIN:-$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
mkdir -p "$(dirname "$BIN")"

CXX=$("$GRID_CONFIG" --cxx)
# INCLUDE ORDER: $HB/include, $HB/src ("grid_mg/..."), the staged Grid SOURCE tree and its
# build dir (Config.h) all come BEFORE grid-config's flags. grid-config --cxxflags carries
# -I<prefix>/include, a `make install` made before patch 04 was applied to the source tree,
# so its GeneralCoarsenedMatrix.h is the unpatched one (the only header that differs); with
# grid-config first it would shadow the patched copy. Same order as build_test_grid_mg.sh.
CXXFLAGS="-I$HB/include -I$HB/src -I$GRID_SOURCE -I$GRID_BUILD/Grid $("$GRID_CONFIG" --cxxflags)"
LDFLAGS="$("$GRID_CONFIG" --ldflags) -L$GRID_BUILD/Grid"
LIBS=$("$GRID_CONFIG" --libs)

echo "Compiler:    $CXX"
echo "GRID_CONFIG: $GRID_CONFIG"
echo "GRID_SOURCE: $GRID_SOURCE"
echo "GRID_BUILD:  $GRID_BUILD"
echo "Include:     $HB/include (first), $HB/src, $GRID_SOURCE, $GRID_BUILD/Grid, then grid-config"
echo "Source:      $SRC"
echo "Binary:      $BIN"

# The dependency file (written during the compile, no extra pass) proves which
# GeneralCoarsenedMatrix.h was compiled: it must be the patched source-tree copy.
DEPS=$BIN.d
check_gcm_used() {
  local used
  used=$(tr ' \\' '\n\n' < "$DEPS" | grep 'multigrid/GeneralCoarsenedMatrix.h$' | head -1)
  if [ -z "$used" ] || [ "$(readlink -f "$used")" != "$(readlink -f "$GCM")" ]; then
    echo "ERROR: compiled GeneralCoarsenedMatrix.h = '${used:-none}', expected $GCM (patch 04)." >&2
    exit 1
  fi
  echo "OK: compiled GeneralCoarsenedMatrix.h is the patched one: $used"
}

echo "Building PURE-GRID against STOCK Grid (no -DGRID_HAVE_QUDA) ..."
$CXX $CXXFLAGS -MD -MF "$DEPS" -o "$BIN" "$SRC" $LDFLAGS $LIBS

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
