#!/usr/bin/env bash
# Compile 2-flavor Möbius DWF comparison codes against the lq Grid install.
# Run after: source /lustre2/nplqcd/vayyar/grid_qcd/env.sh
#
# Both codes represent [det(M_phys)/det(M_PV)]^2 via different algorithms:
#   dw2f_cg_mobius   -- exact 2-flavor CG (TwoFlavourEvenOddRatioPseudoFermionAction)
#   dw2f_eofa_mobius -- 2x1-flavor EOFA   (ExactOneFlavourRatioPseudoFermionAction x2)

set -euo pipefail

# Build against mainline Grid — TwoFlavourEvenOddRatioPseudoFermionAction
# is in both mainline and TXQCD installs; use mainline as the reference.
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

compile dw2f_cg_mobius.cc   dw2f_cg_mobius
compile dw2f_eofa_mobius.cc dw2f_eofa_mobius
