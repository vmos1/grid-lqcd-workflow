#!/usr/bin/env bash
# Compile EOFA and RHMC executables against the lq Grid-TXQCD install.
# Run after: source /lustre2/nplqcd/vayyar/grid_qcd/grid-lqcd-workflow/config.sh

set -euo pipefail

GRID_CONFIG=${GRID_CONFIG:-/lustre2/nplqcd/vayyar/grid_qcd/install-txqcd-gpu/bin/grid-config}

if [ ! -x "${GRID_CONFIG}" ]; then
    echo "ERROR: grid-config not found at ${GRID_CONFIG}"
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

compile dweofa_mobius.cc dweofa_mobius
compile dwrhmc_mobius.cc dwrhmc_mobius
