#!/bin/bash
# Pure-gauge ForceGradient step scan on the production configuration (cfg_2000, 48^3x96, TRAJL 1.0):
# how the Luscher-Weisz gauge monomial alone responds to the MD step count. Decides whether the
# gauge level can take fewer than the 48 steps per trajectory it runs in base+G.
#
# For each step count in STEPS_LIST and each seed in SEEDS: one run of pure_gauge_fg_grid from
# cfg_2000 with NTRAJ trajectories (Metropolis on; accepted trajectories move the field). The same
# seed at every step count draws the same momenta for the first trajectory, so the step counts are
# compared on common momentum draws. Each run writes OUT/<PREFIX><steps>_s<seed>/hmc.log whose
# first line is the ENV provenance line.
#
# One node, 4 ranks, one GPU + NUMA domain per rank (select_gpu, as traj48_puregrid.sh), MPI 1.1.2.2,
# gated NUMA binding. Inside an EXISTING allocation; never allocates, never submits.
#
# Env overrides: OUT, BIN, CFG, STEPS_LIST ("48 32 24 16 12"), SEEDS ("0"), NTRAJ (4), TRAJL (1.0),
#   INTEGRATOR (forcegradient | leapfrog), PREFIX (run-dir prefix, e.g. smoke_), TASK_MIN (per-run
#   timeout in minutes, 40), DEVICE_MEM_MB (24000: Grid's device lattice cache; local volume
#   48.48.24.48 is 1.5 GB per double gauge field, and the default 128 MB cap would thrash).
# Usage: SLURM_JOB_ID=<id> bash run_pure_gauge_fg.sh > <log> 2>&1
#   smoke: SLURM_JOB_ID=<id> STEPS_LIST=48 NTRAJ=1 PREFIX=smoke_ bash run_pure_gauge_fg.sh > <log> 2>&1

set -uo pipefail
: "${SLURM_JOB_ID:?attach to an allocation: SLURM_JOB_ID=<id> bash run_pure_gauge_fg.sh}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORKFLOW=$(cd "$HERE/../.." && pwd)
BASE=$(cd "$WORKFLOW/.." && pwd)

source "$WORKFLOW/config.sh" > /dev/null 2>&1    # modules + LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH

OUT=${OUT:-$BASE/runs/2026_10_9_pure_gauge_fg48}
BIN=${BIN:-${PSCRATCH}/grid_pure_hmc/md_time/bin/pure_gauge_fg_grid}
CFG=${CFG:-$BASE/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime}
STEPS_LIST=${STEPS_LIST:-"48 32 24 16 12"}
SEEDS=${SEEDS:-0}
NTRAJ=${NTRAJ:-4}
TRAJL=${TRAJL:-1.0}
INTEGRATOR=${INTEGRATOR:-forcegradient}
PREFIX=${PREFIX:-}
TASK_MIN=${TASK_MIN:-40}
DEVICE_MEM_MB=${DEVICE_MEM_MB:-24000}
BETA=6.3
U0=0.84570646270714
LATT=48.48.48.96
MPI_GEOM=1.1.2.2
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "ERROR: binary missing: $BIN" >&2; exit 1; }
[ -f "$CFG" ] || { echo "ERROR: configuration missing: $CFG" >&2; exit 1; }
NOTFOUND=$(ldd "$BIN" 2>&1 | grep -c 'not found')
[ "$NOTFOUND" -eq 0 ] || { echo "ERROR: ldd reports $NOTFOUND unresolved libraries for $BIN" >&2; exit 1; }

# MPI / threads: the 48^3 launcher's set (traj48_puregrid.sh section 9).
export SLURM_CPU_BIND=cores
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=${MPICH_GPU_IPC_ENABLED:-1} MPICH_RDMA_ENABLED_CUDA=${MPICH_RDMA_ENABLED_CUDA:-1}
export MPICH_GPU_EAGER_REGISTER_HOST_MEM=${MPICH_GPU_EAGER_REGISTER_HOST_MEM:-0} MPICH_GPU_NO_ASYNC_MEMCPY=${MPICH_GPU_NO_ASYNC_MEMCPY:-0}
export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS=8

GRID_ARGS=(--grid "$LATT" --mpi "$MPI_GEOM" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --device-mem "$DEVICE_MEM_MB")

# 1:1 local rank -> GPU + NUMA domain (traj48_puregrid.sh section 13), --gpu-bind=none.
SELECT_GPU=$OUT/select_gpu
printf '%s\n' \
  '#!/bin/bash' \
  '# 1:1 local rank -> GPU + NUMA binding; copied from traj48_puregrid.sh' \
  'export GPU=$SLURM_LOCALID' \
  'export NUMA=$SLURM_LOCALID' \
  'export CUDA_VISIBLE_DEVICES=$GPU' \
  'exec numactl -m $NUMA -N $NUMA "$@"' > "$SELECT_GPU"
chmod +x "$SELECT_GPU"

SRUN=(srun --jobid="$SLURM_JOB_ID" -N 1 -n 4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32
      --cpu-bind=cores --gpu-bind=none --export=ALL)

echo "=== pure gauge FG scan $(date '+%Y-%m-%d %H:%M:%S %Z') job $SLURM_JOB_ID"
echo "STEPS_LIST='$STEPS_LIST' SEEDS='$SEEDS' NTRAJ=$NTRAJ TRAJL=$TRAJL INTEGRATOR=$INTEGRATOR PREFIX='$PREFIX' OUT=$OUT"

# Hard NUMA gate: run exactly what select_gpu runs on all 4 ranks.
if ! "${SRUN[@]}" "$SELECT_GPU" true; then
  echo "ERROR: NUMA gate failed (or cannot attach to $SLURM_JOB_ID); nothing was run" >&2
  exit 1
fi
"${SRUN[@]}" --label "$SELECT_GPU" numactl --show 2>&1 | grep -E 'nodebind' > "$OUT/numa_binding.txt"
NODE=$("${SRUN[@]:0:2}" -N 1 -n 1 hostname 2> /dev/null)

BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
GRID_SHA=${GRID_HASH_LINE:0:40}

for seed in $SEEDS; do
  for n in $STEPS_LIST; do
    d=$OUT/${PREFIX}${n}_s${seed}
    log=$d/hmc.log
    mkdir -p "$d"
    printf 'ENV PURE_GAUGE_FG STEPS=%s SEED=%s NTRAJ=%s TRAJL=%s INTEGRATOR=%s ACTION=lw BETA=%s U0=%s METROP=on RANDOM_SHIFT=no BIN=%s BIN_SHA256=%s GRID_SHA=%s CFG=%s LATT=%s MPI=%s NODES=1 NTASKS=4 NODE=%s JOBID=%s OMP=%s DEVICE_MEM_MB=%s GRID_FLAGS="%s"\n' \
      "$n" "$seed" "$NTRAJ" "$TRAJL" "$INTEGRATOR" "$BETA" "$U0" "$BIN" "$BIN_SHA256" "${GRID_SHA:-unknown}" \
      "$CFG" "$LATT" "$MPI_GEOM" "${NODE:-unknown}" "$SLURM_JOB_ID" "$OMP_NUM_THREADS" "$DEVICE_MEM_MB" \
      "${GRID_ARGS[*]}" > "$log"
    sed 's/^/NUMA /' "$OUT/numa_binding.txt" >> "$log"
    echo "$(date '+%H:%M:%S') start steps=$n seed=$seed -> $log"
    PG_IMPORT_CFG=$CFG PG_INTEGRATOR=$INTEGRATOR PG_MDSTEPS=$n PG_TRAJL=$TRAJL PG_NTRAJ=$NTRAJ \
    PG_SEED=$seed PG_BETA=$BETA PG_U0=$U0 \
      timeout -k 60s "${TASK_MIN}m" "${SRUN[@]}" --chdir="$d" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}" >> "$log" 2>&1
    rc=$?
    printf 'exit=%s\n' "$rc" >> "$log"
    echo "$(date '+%H:%M:%S') done  steps=$n seed=$seed exit=$rc"
    grep -a -E 'Initial plaquette|dH = |Metropolis_test|Total time for trajectory' "$log" | sed 's/^/    /'
  done
done
echo "=== done $(date '+%Y-%m-%d %H:%M:%S %Z')"
