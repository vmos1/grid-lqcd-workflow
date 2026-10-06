#!/bin/bash
# Free-drift test: Grid vs Chroma MD time unit (__docs/2026_10_05_grid_chroma_md_time_handoff.md §5).
#
# Both codes: 8^4 cold start, Wilson gauge action at beta 0 (zero force), one leapfrog trajectory,
# no accept/reject. Every link ends at exp(t P), so the final plaquette is a ruler for how far each
# code moves the links in its own MD time t. Chroma runs at CHROMA_TAUS, Grid scans GRID_TRAJLS; the
# Grid TRAJL whose plaquette matches Chroma's at tau0 gives the conversion factor tau0 / TRAJL.
# Predictions (NumPy model of both refreshes): Chroma tau0 0.354 -> 0.712; Grid TRAJL 0.25 (sqrt 2)
# -> 0.713, Grid TRAJL 0.0885 (factor 4) -> 0.959.
#
# Allocation-neutral: needs SLURM_JOB_ID of an existing allocation (alloc_gpu.sh), one node used.
# Each run is one rank owning the whole node (1-GPU recipe in machines/perlmutter.md); nothing here
# binds NUMA, so the 1-task binding trap does not apply.
#
# Env: OUT (default runs/2026_10_5_free_drift), GRID_BIN, CHROMA_TAUS, GRID_TRAJLS, NSTEPS,
#      SEEDS (default "0 1": two momentum draws per setting, to show the noise).
# Usage: SLURM_JOB_ID=<id> bash run_free_drift.sh > <log> 2>&1

set -uo pipefail
: "${SLURM_JOB_ID:?attach to an allocation: SLURM_JOB_ID=<id> bash run_free_drift.sh}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORKFLOW=$(cd "$HERE/../.." && pwd)
BASE=$(cd "$WORKFLOW/.." && pwd)

OUT=${OUT:-$BASE/runs/2026_10_5_free_drift}
GRID_BIN=${GRID_BIN:-${PSCRATCH}/grid_pure_hmc/md_time/bin/free_drift_grid}
CHROMA_TAUS=${CHROMA_TAUS:-"0.1 0.354"}
GRID_TRAJLS=${GRID_TRAJLS:-"0.025 0.0707 0.0885 0.125 0.177 0.25 0.354"}
NSTEPS=${NSTEPS:-10}
SEEDS=${SEEDS:-"0 1"}
TEMPLATE=$HERE/free_drift_chroma.ini.xml.in
mkdir -p "$OUT"

SRUN=(srun --jobid="$SLURM_JOB_ID" -N 1 -n 1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128)

echo "=== free drift $(date '+%Y-%m-%d %H:%M:%S %Z') job $SLURM_JOB_ID"
echo "ENV FREE_DRIFT OUT=$OUT GRID_BIN=$GRID_BIN CHROMA_TAUS='$CHROMA_TAUS' GRID_TRAJLS='$GRID_TRAJLS' NSTEPS=$NSTEPS SEEDS='$SEEDS' BETA=0"
"${SRUN[@]}" hostname || { echo "ERROR: cannot attach to allocation $SLURM_JOB_ID" >&2; exit 1; }
[ -x "$GRID_BIN" ] || { echo "ERROR: Grid binary missing: $GRID_BIN (run build_free_drift_grid.sh)" >&2; exit 1; }

# ---- Chroma ----
(
  source "$BASE/chroma/env_chroma_pm.sh"
  [ -x "${CHROMA_HMC:-}" ] || { echo "ERROR: Chroma hmc not found" >&2; exit 1; }
  for seed in $SEEDS; do
    for tau in $CHROMA_TAUS; do
      d=$OUT/chroma_tau${tau}_seed${seed}
      mkdir -p "$d"
      sed -e "s/@TAU@/$tau/" -e "s/@NSTEPS@/$NSTEPS/" -e "s/@BETA@/0.0/" -e "s/@SEED@/$((11 + seed))/" \
        "$TEMPLATE" > "$d/in.xml"
      echo "--- chroma tau0=$tau seed=$seed"
      timeout -k 30s 10m "${SRUN[@]}" --chdir="$d" "$CHROMA_HMC" -i "$d/in.xml" -o "$d/out.xml" \
        -l "$d/log.xml" -geom 1 1 1 1 > "$d/stdout.log" 2>&1
      echo "exit=$?"
    done
  done
)

# ---- Grid ----
(
  source "$WORKFLOW/config.sh"
  export OMP_NUM_THREADS=8
  for seed in $SEEDS; do
    for t in $GRID_TRAJLS; do
      d=$OUT/grid_trajl${t}_seed${seed}
      mkdir -p "$d"
      echo "--- grid TRAJL=$t seed=$seed"
      FD_TRAJL=$t FD_MDSTEPS=$NSTEPS FD_BETA=0.0 FD_SEED=$seed \
        timeout -k 30s 10m "${SRUN[@]}" --chdir="$d" --export=ALL "$GRID_BIN" \
        --grid 8.8.8.8 --mpi 1.1.1.1 --accelerator-threads 8 --shm 1024 > "$d/stdout.log" 2>&1
      echo "exit=$?"
    done
  done
)

# ---- Summary: raw lines; the comparison is done by hand against the predictions above ----
echo "=== summary"
for d in "$OUT"/chroma_tau*; do
  echo "$(basename "$d"): $(grep -o '<w_plaq>[^<]*' "$d/out.xml" 2>/dev/null | tail -1) $(grep -o '<KE_old>[^<]*' "$d/log.xml" 2>/dev/null | tail -1)"
done
for d in "$OUT"/grid_trajl*; do
  echo "$(basename "$d"): $(grep -o 'Plaquette: \[ 1 \] [0-9.eE+-]*' "$d/stdout.log" | tail -1) | $(grep -o 'Total H before trajectory = [0-9.eE+-]*' "$d/stdout.log" | tail -1)"
done
echo "=== done $(date '+%Y-%m-%d %H:%M:%S %Z')"
