#!/bin/bash
#SBATCH --account=m4599_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/runs/slurm_%x_%j.out
# =============================================================================
# wflow48_batch.sh -- offline Wilson flow (bin/wflow_measure, src/wilson_flow/wflow_measure.cc) on
# 48^3x96 configurations, one configuration list per node, all nodes in parallel (2026-10-27).
# Each node runs the launch model of run_test_grid_mg.sh (4 ranks x 1 GPU, MPI 1.1.2.2, the
# select_gpu 1:1 rank -> GPU + NUMA binding, the gated NUMA check); this script only fans the lists
# out over the job's nodes and lifts that launcher's 45-minute cap (a configuration takes ~14 min).
#
# Usage (one line; --nodes = number of lists, --time mandatory):
#   env RUN=<name> WF_LISTDIR=<dir with list_0.txt ... list_<N-1>.txt> sbatch --nodes=<N>
#     --time=<hh:mm:ss> -J <name> --export=ALL wflow48_batch.sh
# or in an existing interactive allocation of N nodes, from the login node:
#   env SLURM_JOB_ID=<id> RUN=<name> WF_LISTDIR=<dir> bash wflow48_batch.sh
# Env: WF_EPS (0.02), WF_TMAX (4.2), WF_MEAS (2): the settings validated on cfg_2000 on 10-26
#   (identical t0, w0 to step 0.01); BIN (default bin/wflow_measure).
# Output: runs/<YYYY_M_D>_<RUN>/flow_<i>.log per node (+ list_<i>.txt copied), never overwrites;
#   analysis: runs/2026_10_26_wflow_driver/wflow_scales.py <logs>.
# =============================================================================
set -u
HB=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/grid-lqcd-workflow/5_studies/hasenbusch_tune
WORKFLOW=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/grid-lqcd-workflow
BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
die() { echo "wflow48_batch: $*" >&2; exit 1; }

[ -n "${SLURM_JOB_ID:-}" ] || die "not inside a Slurm job; submit with sbatch"
: "${RUN:?set RUN=<name>}"
: "${WF_LISTDIR:?set WF_LISTDIR=<dir with list_0.txt ...>}"
BIN=${BIN:-$HB/bin/wflow_measure}
[ -x "$BIN" ] || die "binary not executable: $BIN"
export WF_EPS=${WF_EPS:-0.02} WF_TMAX=${WF_TMAX:-4.2} WF_MEAS=${WF_MEAS:-2}

source "$WORKFLOW/machines/perlmutter.sh"
[ "$(ldd "$BIN" | grep -c 'not found')" -eq 0 ] || die "ldd reports missing libraries"
# As traj48_puregrid_batch.sh: the header's gpus-per-node would otherwise reach srun beside
# --gpus-per-task.
unset SLURM_GPUS_PER_NODE
export MPICH_GPU_IPC_ENABLED=0 MPICH_RDMA_ENABLED_CUDA=0 MPICH_OFI_NIC_POLICY=GPU

# Inside sbatch Slurm sets the node list; driving an interactive allocation from the login node
# (SLURM_JOB_ID=<id> bash wflow48_batch.sh) it does not, so ask squeue for it.
NODELIST=${SLURM_JOB_NODELIST:-$(squeue -h -j "$SLURM_JOB_ID" -o %N)}
[ -n "$NODELIST" ] || die "cannot find the node list of job $SLURM_JOB_ID"
mapfile -t NODES < <(scontrol show hostnames "$NODELIST")
NLIST=$(ls "$WF_LISTDIR"/list_*.txt 2>/dev/null | wc -l)
[ "$NLIST" -ge 1 ] || die "no list_*.txt in $WF_LISTDIR"
[ "$NLIST" -le "${#NODES[@]}" ] || die "$NLIST lists but only ${#NODES[@]} nodes"

DATE_TAG=$(date +%Y_%-m_%-d)
RUN_DIR=$BASE/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
SELECT_GPU=$RUN_DIR/select_gpu
printf '%s\n' \
  '#!/bin/bash' \
  '# 1:1 local rank -> GPU + NUMA binding (as run_test_grid_mg.sh)' \
  'export GPU=$SLURM_LOCALID' \
  'export NUMA=$SLURM_LOCALID' \
  'export CUDA_VISIBLE_DEVICES=$GPU' \
  'exec numactl -m $NUMA -N $NUMA "$@"' > "$SELECT_GPU"
chmod +x "$SELECT_GPU"

GRID_ARGS=(--grid 48.48.48.96 --mpi 1.1.2.2 --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap)
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
for i in $(seq 0 $((NLIST - 1))); do
  LIST=$WF_LISTDIR/list_$i.txt
  [ -f "$LIST" ] || die "missing $LIST"
  cp "$LIST" "$RUN_DIR/"
  while read -r cfg; do
    case "$cfg" in ''|'#'*) continue ;; esac
    [ -f "$cfg" ] || die "configuration not found ($LIST): $cfg"
  done < "$LIST"
  SRUN=(srun --jobid="$SLURM_JOB_ID" -N 1 -n 4 --nodelist="${NODES[$i]}" --ntasks-per-node=4
        --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=cores --gpu-bind=none
        --export=ALL,WF_CONFIGS="$RUN_DIR/list_$i.txt" --chdir="$RUN_DIR")
  # Hard NUMA gate per node (machines/perlmutter.md "Node binding").
  "${SRUN[@]}" "$SELECT_GPU" true || die "NUMA gate failed on ${NODES[$i]}"
  printf 'ENV WFLOW48 RUN=%s JOBID=%s NODE=%s LIST=%s BIN=%s BIN_SHA256=%s WF_EPS=%s WF_TMAX=%s WF_MEAS=%s\n' \
    "$RUN" "$SLURM_JOB_ID" "${NODES[$i]}" "list_$i.txt" "$(basename "$BIN")" "$BIN_SHA256" \
    "$WF_EPS" "$WF_TMAX" "$WF_MEAS" > "$RUN_DIR/flow_$i.log"
  ( "${SRUN[@]}" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}" >> "$RUN_DIR/flow_$i.log" 2>&1
    printf 'exit=%s\n' "$?" >> "$RUN_DIR/flow_$i.log" ) &
done
wait
echo "==== $RUN done: $RUN_DIR"
grep -a -h -E "WFLOW .* (plaq0|seconds)|^exit=" "$RUN_DIR"/flow_*.log
