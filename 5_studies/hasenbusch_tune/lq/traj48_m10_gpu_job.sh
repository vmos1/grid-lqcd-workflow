#!/bin/bash
#SBATCH --job-name=traj48-m10-gpu
#SBATCH --partition=lq2_gpu
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/traj48_m10_gpu_%j.log
#
# 48^3 baseline trajectories of release m10_gpu on 4 lq2 nodes (16 A100, one rank each, --mpi
# 1.2.2.4): Perlmutter's layout, for the match against its 48^3 reference and the like-for-like
# timing. One traj48_lq.sh run; Perlmutter's base+G trajectory is ~3,300-3,500 s of MD, so the
# default 2 trajectories fit the 4 h with the 6.1 GB read and the checkpoints.
# Optional: BIN, RUN (default traj48_base_gpu), N_TRAJ (default 2), HMC_SEED_OFFSET (default 0),
# RESUME_FROM, TIMEOUT_MIN (default 225).
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
export BIN=${BIN:-$ROOT/installs/prod/m10_gpu/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
export RUN=${RUN:-traj48_base_gpu}
export N_TRAJ=${N_TRAJ:-2}
export TIMEOUT_MIN=${TIMEOUT_MIN:-225}
export LQ_TARGET=gpu LQ_MPI_STACK=gompi
echo "=== job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, $(date), RUN=${RUN} N_TRAJ=${N_TRAJ} seed=${HMC_SEED_OFFSET:-0} resume=${RESUME_FROM:-none}"
bash "$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune/lq/traj48_lq.sh"
echo "=== done $(date)"
