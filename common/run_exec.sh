#!/usr/bin/env bash
# Run a Grid executable with MPI. Creates an auto-numbered, timestamped output directory.
#
# Usage:
#   ./3_run/run_exec.sh <executable> [param_file] [grid] [mpi] [nproc]
#
# Arguments:
#   executable   binary name (required)
#   param_file   XML parameter file from inputs/ (optional)
#   grid         lattice geometry,  e.g. 4.4.4.8   (default: 4.4.4.8)
#   mpi          MPI decomposition, e.g. 1.1.1.2   (default: 1.1.1.1)
#   nproc        number of MPI ranks               (default: 1)
#
# Executable search order:
#   1. 2_compile/bin/       your custom compiled programs
#   2. install-*/bin/       programs installed with Grid
#   3. Grid-*/production/   in-source production programs
#   4. Grid-*/HMC/          in-source HMC programs
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

usage() {
    echo "Usage: $0 <executable> [param_file] [grid] [mpi] [nproc]"
    echo ""
    echo "  executable  name of binary"
    echo "  param_file  file from inputs/ (optional, pass '' to skip)"
    echo "  grid        lattice geometry,  e.g. 4.4.4.8   (default: 4.4.4.8)"
    echo "  mpi         MPI decomposition, e.g. 1.1.1.2   (default: 1.1.1.1)"
    echo "  nproc       number of MPI ranks               (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0 TXQCD_Wilson_small '' 4.4.4.4 1.1.1.1 1"
    echo "  $0 TXQCD_Wilson_small ip_hmc_mobius.xml 4.4.4.4 1.1.1.1 1"
    exit 1
}

[ $# -lt 1 ] && usage

EXEC_NAME="$1"
PARAM_FILE="${2:-}"
GRID="${3:-4.4.4.8}"
MPI="${4:-1.1.1.1}"
NPROC="${5:-1}"

# ── Find executable ───────────────────────────────────────────────────────────
REPO_DIR="$(dirname "$SCRIPT_DIR")"
EXEC_PATH=""

# Search 3_examples/*/bin/ first, then installed and in-source locations
for examples_bin in "$REPO_DIR"/3_examples/*/bin/"$EXEC_NAME"; do
    [ -f "$examples_bin" ] && EXEC_PATH="$examples_bin" && break
done

if [ -z "$EXEC_PATH" ]; then
    for candidate in \
        "$GRID_INSTALL/bin/$EXEC_NAME" \
        "$GRID_SRC/production/$EXEC_NAME" \
        "$GRID_SRC/HMC/$EXEC_NAME"; do
        if [ -f "$candidate" ]; then
            EXEC_PATH="$candidate"
            break
        fi
    done
fi

if [ -z "$EXEC_PATH" ]; then
    echo "Error: '$EXEC_NAME' not found in any of:"
    echo "  3_examples/*/bin/  install-${GRID_PROFILE}/bin/  production/  HMC/"
    exit 1
fi

# ── Find parameter file (optional) ───────────────────────────────────────────
PARAM_ARGS=()
if [ -n "$PARAM_FILE" ]; then
    if [ -f "$SCRIPT_DIR/inputs/$PARAM_FILE" ]; then
        PARAM_ARGS=("$SCRIPT_DIR/inputs/$PARAM_FILE")
    elif [ -f "$PARAM_FILE" ]; then
        PARAM_ARGS=("$PARAM_FILE")
    else
        echo "Error: '$PARAM_FILE' not found in inputs/ or current directory"
        exit 1
    fi
fi

# ── Create auto-numbered run directory ────────────────────────────────────────
mkdir -p "$RUNS_DIR"
RUN_NUM=$(printf "%03d" $(( $(ls -d "$RUNS_DIR"/run_* 2>/dev/null | wc -l) + 1 )))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RUNS_DIR/run_${RUN_NUM}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

cp "$EXEC_PATH" "$RUN_DIR/"
[ ${#PARAM_ARGS[@]} -gt 0 ] && cp "${PARAM_ARGS[0]}" "$RUN_DIR/"

echo "==> Profile:       $GRID_PROFILE"
echo "==> Run directory: $RUN_DIR"
echo "==> mpirun -np $NPROC $EXEC_NAME --grid $GRID --mpi $MPI"
echo ""

cd "$RUN_DIR"
mpirun -np "$NPROC" "./$EXEC_NAME" \
    --grid "$GRID" --mpi "$MPI" \
    "${PARAM_ARGS[@]}" \
    2>&1 | tee run.log

echo ""
echo "✓ Done. Output: $RUN_DIR/run.log"
