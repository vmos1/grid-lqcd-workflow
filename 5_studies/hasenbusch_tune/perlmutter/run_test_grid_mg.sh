#!/bin/bash
# Run the standalone Grid-MG Odd/Even test (bin/test_grid_mg_odd) inside an EXISTING
# interactive allocation on one GPU node, 4 ranks x 1 GPU, the same launch model as
# smoke16_puregrid.sh (select_gpu 1:1 rank->GPU+NUMA, --gpu-bind=none, gated NUMA check).
#
# Usage:  SLURM_JOB_ID=<id> RUN=<name> bash run_test_grid_mg.sh [extra test args]
#   e.g.  SLURM_JOB_ID=123 RUN=mgtest16 bash run_test_grid_mg.sh --cb both
# Env:    BIN (default bin/test_grid_mg_odd), MPI_GEOM (default 1.1.2.2), LATT (default 16.16.16.48),
#         GRID_MG_* (hierarchy knobs, see src/grid_mg/grid_mg_params.h), SRUN_EXTRA (e.g. --overlap).
# Output: runs/YYYY_M_D_<RUN>/test.log (unpadded date, runs/ convention); never overwrites.

set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
HB=$(cd "$HERE/.." && pwd)
WORKFLOW=$(cd "$HERE/../../.." && pwd)
BASE=$(cd "$WORKFLOW/.." && pwd)
die() { echo "run_test_grid_mg: $*" >&2; exit 1; }

: "${SLURM_JOB_ID:?set SLURM_JOB_ID to an existing allocation (alloc_gpu.sh)}"
: "${RUN:?set RUN=<name> for the run directory}"
BIN=${BIN:-$HB/bin/test_grid_mg_odd}
[ -x "$BIN" ] || die "binary not executable: $BIN"
MPI_GEOM=${MPI_GEOM:-1.1.2.2}
LATT=${LATT:-16.16.16.48}
NTASKS=4

source "$WORKFLOW/machines/perlmutter.sh"
[ "$(ldd "$BIN" | grep -c 'not found')" -eq 0 ] || die "ldd reports missing libraries (source machines/perlmutter.sh?)"
ldd "$BIN" | grep -qi quda && die "binary links QUDA; this is the pure-Grid test"

# Multi-rank GPU runs need these (port plan s.7); NIC policy from halo-nic.
export MPICH_GPU_IPC_ENABLED=0
export MPICH_RDMA_ENABLED_CUDA=0
export MPICH_OFI_NIC_POLICY=GPU

DATE_TAG=$(date +%Y_%-m_%-d)
RUN_DIR=$BASE/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
LOG=$RUN_DIR/test.log

SELECT_GPU=$RUN_DIR/select_gpu
printf '%s\n' \
  '#!/bin/bash' \
  '# 1:1 local rank -> GPU + NUMA binding (paboyle model, as run_probe_grid_mg.sh)' \
  'export GPU=$SLURM_LOCALID' \
  'export NUMA=$SLURM_LOCALID' \
  'export CUDA_VISIBLE_DEVICES=$GPU' \
  'exec numactl -m $NUMA -N $NUMA "$@"' > "$SELECT_GPU"
chmod +x "$SELECT_GPU"

SRUN=(srun --jobid="$SLURM_JOB_ID" -N 1 -n "$NTASKS" --ntasks-per-node=4
      --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=cores --gpu-bind=none
      --export=ALL --chdir="$RUN_DIR" ${SRUN_EXTRA:+$SRUN_EXTRA})

# Hard NUMA gate (machines/perlmutter.md "Node binding").
"${SRUN[@]}" "$SELECT_GPU" true || die "NUMA gate failed; nothing was run (rundir $RUN_DIR)"

GRID_ARGS=(--grid "$LATT" --mpi "$MPI_GEOM" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap "$@")
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
printf 'ENV TEST_GRID_MG RUN=%s JOBID=%s BIN=%s BIN_SHA256=%s LATT=%s MPI=%s GRID_MG_ENV="%s" ARGS="%s"\n' \
  "$RUN" "$SLURM_JOB_ID" "$(basename "$BIN")" "$BIN_SHA256" "$LATT" "$MPI_GEOM" \
  "$(compgen -e | grep '^GRID_MG_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
  "$*" > "$LOG"

echo "running $(basename "$BIN") -> $LOG"
timeout -k 60s 45m "${SRUN[@]}" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"
echo "==== $RUN: exit=$rc  log: $LOG"
grep -a -E '^ENV |GALERKIN|SOLVE cb=|TEST RESULT|Assert|GRID_ASSERT|abort|error' "$LOG" | head -40
exit "$rc"
