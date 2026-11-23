#!/bin/bash
#SBATCH --job-name=forces48-m10-cpu
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=5
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/forces48_m10_cpu_%j.log
#
# One 48^3 FORCES_ONLY run of release m10_cpu (forces48_lq.sh): every monomial's heatbath and one
# force evaluation per sample, from cfg_2000, with the trajectory's solver routes. Default layout =
# the lq3 trajectory chain's 4-node one: 12 ranks x 5 cores per node, --mpi 3.2.2.4 (48 ranks; the
# multigrid allows 16, 24 or 48). Other layouts: sbatch --nodes / --ntasks-per-node /
# --cpus-per-task with MPI_GEOM. A/B tests: GRID_EXTRA_ARGS (e.g. --dslash-asm), HASEN_GRID_* lists.
# Optional: BIN, RUN (default forces48_cpu), FORCES_SAMPLES (default 2), MPI_GEOM (default
# 3.2.2.4), GRID_EXTRA_ARGS, FO_ROUTES (traj|production), HMC_SEED_OFFSET, TIMEOUT_MIN (default 80).
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
export BIN=${BIN:-$ROOT/installs/prod/m10_cpu/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
export RUN=${RUN:-forces48_cpu}
export MPI_GEOM=${MPI_GEOM:-3.2.2.4}
export TIMEOUT_MIN=${TIMEOUT_MIN:-80}
export LQ_TARGET=cpu LQ_MPI_STACK=lq3-ompi418
echo "=== job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, $(date), RUN=${RUN} MPI_GEOM=${MPI_GEOM} samples=${FORCES_SAMPLES:-2} routes=${FO_ROUTES:-traj} extra='${GRID_EXTRA_ARGS:-}'"
bash "$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune/lq/forces48_lq.sh"
echo "=== done $(date)"
