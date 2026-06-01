#!/usr/bin/env bash
# Compile 2+1 flavor Wilson Clover HMC binaries.
#
#   wclover_2p1_rhmc  — non-EO (full-lattice CG), builds against install-grid-gpu
#   wclover_2p1_eo    — EO-preconditioned (Schur + LogDet), builds against install-txqcd-gpu
#                       (QCDLogDetCloverEOAction and TwoFlavourSchurCloverAction are
#                       TXQCD-fork additions not yet in mainline Grid)
#
# Run after: source /lustre2/nplqcd/vayyar/grid_qcd/env.sh

set -euo pipefail

BASE=/lustre2/nplqcd/vayyar/grid_qcd
GRID_CONFIG_MAIN=${GRID_CONFIG_MAIN:-${BASE}/install-grid-gpu/bin/grid-config}
GRID_CONFIG_TXQCD=${GRID_CONFIG_TXQCD:-${BASE}/install-txqcd-gpu/bin/grid-config}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC=${SCRIPT_DIR}/src
BIN=${SCRIPT_DIR}/bin
mkdir -p ${BIN}

compile() {
    local gc=$1 src=$2 exe=$3
    if [ ! -x "${gc}" ]; then
        echo "ERROR: grid-config not found at ${gc}"
        exit 1
    fi
    echo "==> Compiling ${exe} ($(${gc} --prefix))..."
    $(${gc} --cxx) \
        $(${gc} --cxxflags) \
        ${SRC}/${src} -o ${BIN}/${exe} \
        $(${gc} --ldflags) \
        $(${gc} --libs)
    echo "-- Done: ${BIN}/${exe}"
}

compile ${GRID_CONFIG_MAIN}  wclover_2p1_rhmc.cc wclover_2p1_rhmc
compile ${GRID_CONFIG_TXQCD} wclover_2p1_eo.cc   wclover_2p1_eo
compile ${GRID_CONFIG_MAIN}  wclover_hasenbusch_tune.cc wclover_hasenbusch_tune
