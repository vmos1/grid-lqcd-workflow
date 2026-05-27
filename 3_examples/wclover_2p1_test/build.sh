#!/usr/bin/env bash
# Compile 2+1 flavor Wilson Clover HMC against the lq mainline Grid install.
# Run after: source /lustre2/nplqcd/vayyar/grid_qcd/env.sh

set -euo pipefail

GRID_CONFIG=${GRID_CONFIG:-/lustre2/nplqcd/vayyar/grid_qcd/install-grid-gpu/bin/grid-config}

if [ ! -x "${GRID_CONFIG}" ]; then
    echo "ERROR: grid-config not found at ${GRID_CONFIG}"
    echo "Set GRID_CONFIG or source env.sh first."
    exit 1
fi

echo "-- Using: ${GRID_CONFIG}"
echo "-- Prefix: $(${GRID_CONFIG} --prefix)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC=${SCRIPT_DIR}/src
BIN=${SCRIPT_DIR}/bin
mkdir -p ${BIN}

compile() {
    local src=$1 exe=$2
    echo "==> Compiling ${exe}..."
    $(${GRID_CONFIG} --cxx) \
        $(${GRID_CONFIG} --cxxflags) \
        ${SRC}/${src} -o ${BIN}/${exe} \
        $(${GRID_CONFIG} --ldflags) \
        $(${GRID_CONFIG} --libs)
    echo "-- Done: ${BIN}/${exe}"
}

compile wclover_2p1_rhmc.cc wclover_2p1_rhmc
