#!/usr/bin/env bash
# Smoke test: run 3 trajectories (1 thermalisation + 2 production with Metropolis)
# on a 4^4 lattice with Ls=4. Completes in ~1 minute.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config.sh"

BIN="$SCRIPT_DIR/bin/dweofa_mobius_HSDM_v3"
XML="$SCRIPT_DIR/inputs/ip_hmc_mobius_test.xml"

if [ ! -f "$BIN" ]; then
    echo "Binary not found — run ./build.sh first"
    exit 1
fi

LOG=/tmp/dweofa_test.log

echo "==> DW EOFA Mobius smoke test (4^4, Ls=4, 3 trajectories)..."
mpirun -np 1 "$BIN" --grid 4.4.4.4 --mpi 1.1.1.1 --ParameterFile "$XML" 2>&1 | \
    tee "$LOG" | \
    grep --line-buffered -E "Trajectory =|dH|plaquette|ACCEPT|REJECT|Skipping|Finalize|ASSERT"

if grep -q "ASSERT\|MPI_ABORT" "$LOG"; then
    echo ""
    echo "FAIL  (see $LOG)"
    exit 1
fi

echo ""
echo "✓ DW EOFA Mobius smoke test passed"
