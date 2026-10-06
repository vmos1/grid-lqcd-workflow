#!/bin/bash
# Hopping-term comparison: point-source C(t) on cfg_2000, production strange clover action, Grid
# (pion_corr_grid) then Chroma (`chroma` + pion_corr_chroma.ini.xml), 2 nodes / 8 GPUs each
# (__docs/2026_10_05_grid_chroma_md_time_handoff.md §8). Grid: --mpi 1.2.2.2, one GPU per rank via
# CUDA_VISIBLE_DEVICES=$SLURM_LOCALID (no NUMA binding). Chroma: -geom 1 2 2 2, the production
# launch shape (submit_scripts/chroma_validation/chroma_baseg_mds.sh) at 8 ranks.
#
# Env: OUT (default runs/2026_10_5_pion_corr), GRID_BIN, CFG, RUN_GRID (1), RUN_CHROMA (1),
#      STEP_MIN (per-code timeout, default 60).
# Usage: SLURM_JOB_ID=<id of a >=2-node allocation> bash run_pion_corr.sh > <log> 2>&1

set -uo pipefail
: "${SLURM_JOB_ID:?attach to an allocation: SLURM_JOB_ID=<id> bash run_pion_corr.sh}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORKFLOW=$(cd "$HERE/../.." && pwd)
BASE=$(cd "$WORKFLOW/.." && pwd)

OUT=${OUT:-$BASE/runs/2026_10_5_pion_corr}
GRID_BIN=${GRID_BIN:-${PSCRATCH}/grid_pure_hmc/md_time/bin/pion_corr_grid}
CFG=${CFG:-$BASE/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime}
RUN_GRID=${RUN_GRID:-1}
RUN_CHROMA=${RUN_CHROMA:-1}
STEP_MIN=${STEP_MIN:-60}
mkdir -p "$OUT/grid" "$OUT/chroma"

SRUN=(srun --jobid="$SLURM_JOB_ID" -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 --cpu-bind=cores)

echo "=== pion corr $(date '+%Y-%m-%d %H:%M:%S %Z') job $SLURM_JOB_ID"
echo "ENV PION_CORR OUT=$OUT GRID_BIN=$GRID_BIN CFG=$CFG mass=-0.2050 csw=1.20536588031793 stout=0.125x1 bc=1,1,1,-1 tol=1e-11"
srun --jobid="$SLURM_JOB_ID" -N 2 -n 2 --ntasks-per-node=1 hostname || { echo "ERROR: cannot attach" >&2; exit 1; }

if [ "$RUN_GRID" = 1 ]; then
  (
    source "$WORKFLOW/config.sh" > /dev/null 2>&1
    export OMP_NUM_THREADS=8 MPICH_GPU_SUPPORT_ENABLED=1
    echo "--- grid $(date '+%H:%M:%S')"
    timeout -k 60s "${STEP_MIN}m" "${SRUN[@]}" --chdir="$OUT/grid" --export=ALL \
      bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec "$0" "$@"' "$GRID_BIN" \
      --grid 48.48.48.96 --mpi 1.2.2.2 --accelerator-threads 8 --shm 2048 --shm-mpi 0 \
      --device-mem 8000 --config "$CFG" > "$OUT/grid/stdout.log" 2>&1
    rc=$?
    echo "grid exit=$rc $(date '+%H:%M:%S')"
  )
fi

if [ "$RUN_CHROMA" = 1 ]; then
  (
    source "$BASE/chroma/env_chroma_pm.sh" > /dev/null 2>&1
    export QUDA_RESOURCE_PATH=$OUT/chroma/quda_resource
    mkdir -p "$QUDA_RESOURCE_PATH"
    echo "--- chroma $(date '+%H:%M:%S')"
    timeout -k 60s "${STEP_MIN}m" "${SRUN[@]}" --chdir="$OUT/chroma" --export=ALL \
      "$CHROMA_BIN" -i "$HERE/pion_corr_chroma.ini.xml" -o "$OUT/chroma/out.xml" \
      -l "$OUT/chroma/log.xml" -geom 1 2 2 2 > "$OUT/chroma/stdout.log" 2>&1
    rc=$?
    echo "chroma exit=$rc $(date '+%H:%M:%S')"
  )
fi
echo "=== done $(date '+%Y-%m-%d %H:%M:%S %Z')"
