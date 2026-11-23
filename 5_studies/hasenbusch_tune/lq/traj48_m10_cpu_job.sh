#!/bin/bash
#SBATCH --job-name=traj48-m10-cpu
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/traj48_m10_cpu_%j.log
#
# 48^3 baseline trajectories of release m10_cpu on lq3: 16 ranks x 4 cores per node (Grid-CPU's
# best layout), --mpi 1.2.2.4 on one node = Perlmutter's split. The cost of one 48^3 trajectory on
# one lq3 node is not known yet (rough guess from the 16^3 gate: 4-12 h); every finished
# trajectory is checkpointed and RESUME_FROM continues the chain. More nodes: sbatch --nodes=N
# with MPI_GEOM for 16N ranks (e.g. 4 nodes: MPI_GEOM=1.2.4.8).
# Optional: BIN, RUN (default traj48_base_cpu), N_TRAJ (default 2), HMC_SEED_OFFSET (default 0),
# RESUME_FROM, MPI_GEOM (default 1.2.2.4), TIMEOUT_MIN (default 1420).
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
export BIN=${BIN:-$ROOT/installs/prod/m10_cpu/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
export RUN=${RUN:-traj48_base_cpu}
export N_TRAJ=${N_TRAJ:-2}
export TIMEOUT_MIN=${TIMEOUT_MIN:-1420}
export LQ_TARGET=cpu LQ_MPI_STACK=lq3-ompi418
echo "=== job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, $(date), RUN=${RUN} N_TRAJ=${N_TRAJ} MPI_GEOM=${MPI_GEOM:-1.2.2.4} seed=${HMC_SEED_OFFSET:-0} resume=${RESUME_FROM:-none}"
lscpu | grep -E "^(Model name|Socket|Core|NUMA node)"
bash "$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune/lq/traj48_lq.sh"
echo "=== done $(date)"
