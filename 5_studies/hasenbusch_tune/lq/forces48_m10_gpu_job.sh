#!/bin/bash
#SBATCH --job-name=forces48-m10-gpu
#SBATCH --partition=lq2_gpu
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/forces48_m10_gpu_%j.log
#
# One 48^3 FORCES_ONLY run of release m10_gpu (forces48_lq.sh) on 4 lq2 nodes (16 A100, one rank
# each, --mpi 1.2.2.4, Perlmutter's split): every monomial's heatbath and one force evaluation per
# sample, from cfg_2000, with the trajectory's solver routes. Same start, seed and routes as the lq3
# baseline (forces48_m10_cpu_job.sh, job 1364292), so the FORCES_ONLY lines compare across machines.
# Optional: BIN, RUN (default forces48_gpu), FORCES_SAMPLES (default 2), GRID_EXTRA_ARGS,
# FO_ROUTES (traj|production), HMC_SEED_OFFSET, DEVICE_MEM_MB (default 32000), TIMEOUT_MIN
# (default 40).
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
export BIN=${BIN:-$ROOT/installs/prod/m10_gpu/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
export RUN=${RUN:-forces48_gpu}
export TIMEOUT_MIN=${TIMEOUT_MIN:-40}
export LQ_TARGET=gpu LQ_MPI_STACK=gompi
echo "=== job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, $(date), RUN=${RUN} MPI_GEOM=${MPI_GEOM:-1.2.2.4} samples=${FORCES_SAMPLES:-2} routes=${FO_ROUTES:-traj} cap=${DEVICE_MEM_MB:-32000} extra='${GRID_EXTRA_ARGS:-}'"
bash "$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune/lq/forces48_lq.sh"
echo "=== done $(date)"
