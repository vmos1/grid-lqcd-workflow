#!/bin/bash
#SBATCH --account=m4599_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/runs/slurm_%x_%j.out
# =============================================================================
# traj48_puregrid_batch.sh -- batch entry point for traj48_puregrid.sh: a whole
# N_TRAJ chain in ONE regular-QOS job on four 40 GB nodes. The batch job is the
# allocation, so this only fills in BIN and TIMEOUT_MIN and calls the launcher,
# which attaches its srun steps with --jobid=$SLURM_JOB_ID exactly as it does
# inside an interactive allocation. Everything else (physics, ladder profiles,
# solver routes, gates, checkpoints, ENV line, summary) is the launcher's.
#
# Usage (one line; --time is mandatory, a missing one has defaulted to 10 min,
# machines/perlmutter.md):
#   sbatch --time=08:00:00 -J c3_m7_10 --export=ALL,LADDER_PROFILE=c3,N_TRAJ=10,HMC_SEED_OFFSET=300,RUN=c3_m7_seed300_10 traj48_puregrid_batch.sh
# C1: the wrapper sets the three-rung routes of the C1 screening itself (below).
#
# BIN          default bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m8 (m7b + strange bounds check, 2026-10-06)
# TIMEOUT_MIN  default: the job's remaining time minus 10 min, so the launcher's
#              summary and the checkpoint listing still run before Slurm ends the job.
# Card size: --constraint=gpu&hbm40g pins the 40 GB nodes, so timing rows are never
# mixed with 80 GB ones (L119). Output: runs/<date>_<RUN>/ (launcher) and
# runs/slurm_<job name>_<job id>.out (this script and the launcher's summary).
# =============================================================================
HB=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/grid-lqcd-workflow/5_studies/hasenbusch_tune

[ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: not inside a Slurm job; submit with sbatch" >&2; exit 1; }
[ -n "${RUN:-}" ] || { echo "ERROR: RUN is not set (pass it with --export=ALL,RUN=<name>,...)" >&2; exit 1; }
export BIN=${BIN:-$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m8}
# The batch environment carries SLURM_GPUS_PER_NODE=4 from the header, which srun reads as
# --gpus-per-node beside the launcher's --gpus-per-task=1. An interactive allocation's srun
# never sees it; drop it so the step shape is the one the launcher was validated with.
unset SLURM_GPUS_PER_NODE
# C1 has three ratio rungs (0..2), the launcher's default routes assume four: use the routes
# of the C1 force-only screening (runs/2026_10_2_fo_screen_c1: MG 0,1,2, MG heatbath 0,1, no
# mixed-CG rung, mixed-CG heatbath 2) unless the caller set them. Set here, not through
# --export, because an empty value is what switches the mixed-CG rung route off.
if [ "${LADDER_PROFILE:-}" = c1 ]; then
  [ -n "${HASEN_GRID_MIXED_CG_RUNGS+x}" ] || export HASEN_GRID_MIXED_CG_RUNGS=
  [ -n "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS+x}" ] || export HASEN_GRID_MIXED_CG_HEATBATH_RUNGS=2
fi

if [ -z "${TIMEOUT_MIN:-}" ]; then
  LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L | head -1)
  if [[ "$LEFT" =~ ^(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)$ ]]; then
    LEFT_MIN=$(( 10#${BASH_REMATCH[2]:-0} * 1440 + 10#${BASH_REMATCH[4]:-0} * 60 + 10#${BASH_REMATCH[5]} ))
  else
    echo "ERROR: cannot read the job's remaining time from squeue ('$LEFT'); pass TIMEOUT_MIN" >&2
    exit 1
  fi
  [ "$LEFT_MIN" -gt 30 ] || { echo "ERROR: only $LEFT_MIN min left in the job" >&2; exit 1; }
  export TIMEOUT_MIN=$(( LEFT_MIN - 10 ))
fi

echo "traj48_puregrid_batch: job $SLURM_JOB_ID on $(squeue -h -j "$SLURM_JOB_ID" -o '%N %f'), RUN=$RUN LADDER_PROFILE=${LADDER_PROFILE:-baseG} N_TRAJ=${N_TRAJ:-1} HMC_SEED_OFFSET=${HMC_SEED_OFFSET:-0} TIMEOUT_MIN=$TIMEOUT_MIN BIN=$BIN"
date
bash "$HB/perlmutter/traj48_puregrid.sh"
rc=$?
date
echo "traj48_puregrid_batch: launcher exit=$rc"
exit "$rc"
