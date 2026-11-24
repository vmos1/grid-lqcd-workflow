#!/bin/bash
#SBATCH --job-name=traj48-m10-cpu
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=6
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/traj48_m10_cpu_%j.log
#
# 48^3 baseline trajectories of release m10_cpu on lq3, in the production layout (user, 11-23):
# 6 nodes x 8 ranks x 8 cores, --mpi 3.2.2.4 (48 ranks; with the default multigrid blocks only
# 16, 24 or 48 ranks work, handoff 7e). Every finished trajectory is checkpointed and RESUME_FROM
# continues the chain.
# lq3 speed defaults (forces-only A/B job 1364668, confirmed on trajectories 2011-2012 by job
# 1364701: MD 1.77x -> ~1.57x Perlmutter's, forces identical): Grid's AVX-512 assembly kernel
# (GRID_EXTRA_ARGS=--dslash-asm) and rung PF3 on the shared multigrid instead of mixed-precision CG
# (HASEN_GRID_MG_RUNGS=0,1,2,3, HASEN_GRID_MIXED_CG_RUNGS empty). A caller's own value wins, set
# EMPTY to switch one off (e.g. GRID_EXTRA_ARGS= for the plain kernel).
# Other layouts: sbatch --nodes / --ntasks-per-node / --cpus-per-task with MPI_GEOM (e.g. 1 node:
# --nodes=1 --ntasks-per-node=16 --cpus-per-task=4 MPI_GEOM=1.2.2.4).
# Optional: BIN, RUN (default traj48_base_cpu), N_TRAJ (default 2), HMC_SEED_OFFSET (default 0),
# RESUME_FROM, MPI_GEOM (default 3.2.2.4), TIMEOUT_MIN (default 1420).
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
export BIN=${BIN:-$ROOT/installs/prod/m10_cpu/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
export RUN=${RUN:-traj48_base_cpu}
export N_TRAJ=${N_TRAJ:-2}
export TIMEOUT_MIN=${TIMEOUT_MIN:-1420}
export MPI_GEOM=${MPI_GEOM:-3.2.2.4}
export GRID_EXTRA_ARGS=${GRID_EXTRA_ARGS---dslash-asm}
if [ -z "${HASEN_GRID_MG_RUNGS+x}" ]; then export HASEN_GRID_MG_RUNGS=0,1,2,3; fi
if [ -z "${HASEN_GRID_MIXED_CG_RUNGS+x}" ]; then export HASEN_GRID_MIXED_CG_RUNGS=; fi
export LQ_TARGET=cpu LQ_MPI_STACK=lq3-ompi418
echo "=== job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, $(date), RUN=${RUN} N_TRAJ=${N_TRAJ} MPI_GEOM=${MPI_GEOM} seed=${HMC_SEED_OFFSET:-0} resume=${RESUME_FROM:-none} extra='${GRID_EXTRA_ARGS}' mg_rungs=${HASEN_GRID_MG_RUNGS} mixed_cg=${HASEN_GRID_MIXED_CG_RUNGS:-none}"
lscpu | grep -E "^(Model name|Socket|Core|NUMA node)"
bash "$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune/lq/traj48_lq.sh"
echo "=== done $(date)"
