#!/usr/bin/env bash
# Run Grid correctness tests.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

BIN="$GRID_INSTALL/bin"

mpirun -np 1 "$BIN/Benchmark_dwf"    --grid 4.4.4.4 --mpi 1.1.1.1
mpirun -np 1 "$BIN/Benchmark_wilson" --grid 4.4.4.4 --mpi 1.1.1.1
