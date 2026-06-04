#!/bin/bash
# Build gen_qcd_hasenbusch_tune on the lq cluster (AlmaLinux 8, A100).
#
# Compiles against the Grid-TXQCD build at $GRID_BUILD (set by machines/lq.sh).
# Grid-TXQCD must already be built — see 1_build_grid/build_grid_lq.sh.
#
# QUDA (Option B — not yet built):
#   When Grid-TXQCD is rebuilt with QUDA support, uncomment the QUDA_* lines
#   below. The source already has #ifdef GRID_HAVE_QUDA guards.
#
# Usage:  bash build.sh        (sources workflow config.sh → machines/lq.sh)

set -e

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
HB=$(cd "$HERE/.." && pwd)

WORKFLOW=$(cd "$HERE/../../.." && pwd)
export MACHINE=lq
source "$WORKFLOW/config.sh"              # modules + paths (via machines/lq.sh)

GRID_CONFIG=${GRID_CONFIG:-$GRID_BUILD/grid-config}
TXQCD_PROD=${TXQCD_PROD:-$GRID_SRC/production}   # for params.h

SRC=$HB/src/gen_qcd_hasenbusch_tune.cc
BIN=$HB/bin/gen_qcd_hasenbusch_tune

if [ ! -x "$GRID_CONFIG" ]; then
  echo "ERROR: grid-config not found/executable at: $GRID_CONFIG" >&2
  echo "       Build Grid-TXQCD first (1_build_grid/build_grid_lq.sh)." >&2
  exit 1
fi

CXX=$($GRID_CONFIG --cxx)
CXXFLAGS="$($GRID_CONFIG --cxxflags) -I${TXQCD_PROD}"
LDFLAGS=$($GRID_CONFIG --ldflags)
LIBS=$($GRID_CONFIG --libs)

# Option B (QUDA): uncomment when Grid-TXQCD is rebuilt with QUDA support.
# QUDA_INSTALL=${BASE_DIR}/Grid-TXQCD/external/quda-install
# QMP_INSTALL=/lustre2/nplqcd/install/quda
# CXXFLAGS+=" -DGRID_HAVE_QUDA -I${QUDA_INSTALL}/include -I${QMP_INSTALL}/include"
# LDFLAGS+=" -L${QUDA_INSTALL}/lib -L${QMP_INSTALL}/lib"
# LDFLAGS+=" -Xlinker -rpath -Xlinker ${QUDA_INSTALL}/lib"
# LIBS="-lquda -lqmp ${LIBS}"

echo "Compiler: ${CXX}"
echo "Source:   ${SRC}"
echo "Binary:   ${BIN}"

$CXX $CXXFLAGS -o "${BIN}" "${SRC}" $LDFLAGS $LIBS

echo "Done: ${BIN}"
