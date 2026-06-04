#!/bin/bash
# Build gen_qcd_hasenbusch_tune against the TXQCD Grid install.
#
# Requirements:
#   - TXQCD Grid install with EO clover actions (QCDLogDetCloverEOAction,
#     TwoFlavourSchurCloverAction, OneFlavourSchurCloverRationalActionMP).
#     Mainline Grid does not have these.
#   - params.h from the TXQCD production directory (for lattice_size(),
#     mass_light, csw, etc. with env-var overrides).
#
# QUDA (Option B — not yet built):
#   When Grid-TXQCD is rebuilt with QUDA support, uncomment the QUDA_* lines
#   below.  The source already has #ifdef GRID_HAVE_QUDA guards.
#
# Usage:
#   source /lustre2/nplqcd/vayyar/grid_qcd/env.sh
#   bash build.sh

set -e

BASE=/lustre2/nplqcd/vayyar/grid_qcd
GRID_CONFIG=${BASE}/build-txqcd-gpu/grid-config
TXQCD_PROD=${BASE}/Grid-TXQCD/production   # for params.h

SRC=$(dirname "$0")/src/gen_qcd_hasenbusch_tune.cc
BIN=$(dirname "$0")/bin/gen_qcd_hasenbusch_tune

CXX=$($GRID_CONFIG --cxx)
CXXFLAGS="$($GRID_CONFIG --cxxflags) -I${TXQCD_PROD}"
LDFLAGS=$($GRID_CONFIG --ldflags)
LIBS=$($GRID_CONFIG --libs)

# Option B (QUDA): uncomment when Grid-TXQCD is rebuilt with QUDA support.
# QUDA_INSTALL=${BASE}/Grid-TXQCD/external/quda-install
# QMP_INSTALL=/lustre2/nplqcd/install/quda
# CXXFLAGS+=" -DGRID_HAVE_QUDA -I${QUDA_INSTALL}/include -I${QMP_INSTALL}/include"
# LDFLAGS+=" -L${QUDA_INSTALL}/lib -L${QMP_INSTALL}/lib"
# LDFLAGS+=" -Xlinker -rpath -Xlinker ${QUDA_INSTALL}/lib"
# LIBS="-lquda -lqmp ${LIBS}"

echo "Compiler: ${CXX}"
echo "Source:   ${SRC}"
echo "Binary:   ${BIN}"

$CXX $CXXFLAGS -o ${BIN} ${SRC} $LDFLAGS $LIBS

echo "Done: ${BIN}"
