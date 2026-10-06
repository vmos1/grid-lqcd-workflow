#!/bin/bash
#SBATCH --account=m4599_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/runs/slurm_%x_%j.out
# =============================================================================
# scan_trajl1_batch.sh -- base+G at TRAJL 1.0 (Chroma tau0 1.414 under the sqrt 2 map,
# __docs/2026_10_05_grid_chroma_md_time_handoff.md §5) for ONE MD step count, several seeds.
# Each seed is a FRESH single trajectory from cfg_2000 (not a chain), so every step count sees
# the same momenta and pseudofermions per seed and dH vs step size is compared trajectory by
# trajectory. Seed 300 is the seed of the reference, T1-T4 (agent1 review §3).
# Physics, routes, gates, checkpoints and the ENV line are traj48_puregrid.sh's (production
# base+G, strange interval [7e-4, 35]); this only loops it over SEEDS with
# DIAG_TRAJL=1.0 DIAG_MDSTEPS=$MDS and a per-trajectory timeout, so one stalled trajectory (a
# blow-up can stall the MG solves) cannot eat the others.
#
# Usage (one line; --time is mandatory, machines/perlmutter.md):
#   sbatch --time=04:45:00 -J trajl1_md12 --export=ALL,MDS=12 scan_trajl1_batch.sh
# MDS          MD steps (required)
# SEEDS        default "300 301 302" (HMC_SEED_OFFSET of each fresh trajectory)
# PER_TRAJ_MIN default 20 + 7*MDS (expected ~12 + 4.75*MDS min on 40 GB nodes + ~5 min startup)
# BIN          default bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m7b (as traj48_puregrid_batch.sh)
# RUN_TAG      optional suffix of the run names (a rerun on the same date; the launcher refuses to
#              overwrite a run directory). 2026-10-06: job 59387301 (MDS 8) lost all three seeds to
#              a host OOM of task 15 on nid001536 at ~270 s; rerun as job 59426576 with RUN_TAG=b,
#              no node excluded (if it recurs on another node, suspect the code, not the node).
# Output: runs/<date>_trajl1_md<MDS>_s<seed>/ per trajectory, runs/slurm_<job name>_<id>.out.
# =============================================================================

HB=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/grid-lqcd-workflow/5_studies/hasenbusch_tune

[ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: not inside a Slurm job; submit with sbatch" >&2; exit 1; }
[ -n "${MDS:-}" ] || { echo "ERROR: MDS is not set (pass it with --export=ALL,MDS=<steps>)" >&2; exit 1; }
SEEDS=${SEEDS:-"300 301 302"}
PER_TRAJ_MIN=${PER_TRAJ_MIN:-$(( 20 + 7 * MDS ))}
export BIN=${BIN:-$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m7b}
# Same reason as traj48_puregrid_batch.sh: the header's SLURM_GPUS_PER_NODE would reach srun.
unset SLURM_GPUS_PER_NODE

export LADDER_PROFILE=baseG DIAG_TRAJL=1.0 DIAG_MDSTEPS=$MDS N_TRAJ=1

left_min() {
  local LEFT
  LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L | head -1)
  if [[ "$LEFT" =~ ^(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)$ ]]; then
    echo $(( 10#${BASH_REMATCH[2]:-0} * 1440 + 10#${BASH_REMATCH[4]:-0} * 60 + 10#${BASH_REMATCH[5]} ))
  fi
}

echo "scan_trajl1_batch: job $SLURM_JOB_ID on $(squeue -h -j "$SLURM_JOB_ID" -o '%N %f'), MDS=$MDS SEEDS='$SEEDS' PER_TRAJ_MIN=$PER_TRAJ_MIN TRAJL=1.0 BIN=$BIN"
for seed in $SEEDS; do
  LEFT=$(left_min)
  [ -n "$LEFT" ] || { echo "ERROR: cannot read the job's remaining time; stopping" >&2; exit 1; }
  if [ "$LEFT" -lt 30 ]; then
    echo "scan_trajl1_batch: only $LEFT min left, skipping seed $seed and later"
    break
  fi
  T=$PER_TRAJ_MIN
  [ "$T" -le $(( LEFT - 10 )) ] || T=$(( LEFT - 10 ))
  export HMC_SEED_OFFSET=$seed TIMEOUT_MIN=$T RUN=trajl1_md${MDS}_s${seed}${RUN_TAG:+_$RUN_TAG}
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') seed $seed RUN=$RUN TIMEOUT_MIN=$T"
  bash "$HB/perlmutter/traj48_puregrid.sh"
  rc=$?
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') seed $seed launcher exit=$rc"
done
echo "scan_trajl1_batch: done"
