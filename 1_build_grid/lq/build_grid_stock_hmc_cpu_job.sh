#!/bin/bash
#SBATCH --job-name=grid-stock-p08-cpu-build
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_grid_stock_hmc_cpu_%j.log
#
# Configure, compile and install the lq3 CPU libGrid (stock 3d3eff86 + patches 01-08 + lq fixes,
# TARGET=cpu of build_grid_stock_hmc.sh) on an lq3 node. prepare and patch run first on the
# login node:
#   TARGET=cpu bash 1_build_grid/lq/build_grid_stock_hmc.sh all
# lq3 runs EL9 and the toolchain (spack gcc 15.3, Grid-CPU's OpenMPI 4.1.8) is EL9-only, so
# configure runs here too, unlike the GPU build. 64 cores, 1.5 TB: make -j64.
# Before the build, one MPI locality check with this OpenMPI (16 ranks, the lq3 reference
# layout): gompi on lq2 gave MPI_COMM_TYPE_SHARED groups of 1 (job 1363335); "SHM_SIZE 16 16 16"
# here means Grid's on-node shared-memory path will be used. Informational: it never stops
# the build.
# The driver and the release (installs/dev/m10_cpu) are the next job:
#   5_studies/hasenbusch_tune/lq/build_driver_stock_cpu_job.sh (submit with afterok on this one).
set -euo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
WF=${ROOT}/grid-lqcd-workflow
SCRIPT=${WF}/1_build_grid/lq/build_grid_stock_hmc.sh
PROBE_DIR=${ROOT}/runs/2026_11_19_mpi_locality_probe

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date)"
LQ_TARGET=cpu source "${WF}/machines/lq.sh"
echo "mpicxx: $(command -v mpicxx) ($(mpicxx --version | head -1))"
echo "mpirun: $(mpirun --version 2>&1 | head -1)"
echo "GMP=${GMP} MPFR=${MPFR}"

echo
echo "---- MPI locality check (16 ranks x 4 cores)"
set +e
mpicc -O1 -o "${PROBE_DIR}/shmprobe_lq3" "${PROBE_DIR}/shmprobe.c"
OMP_NUM_THREADS=1 timeout 120 mpirun -np 16 --map-by ppr:16:node:PE=4 --bind-to core \
  "${PROBE_DIR}/shmprobe_lq3" 2>&1 | grep -E "SHM_SIZE|rank 0/|rror"
echo "locality check exit=${PIPESTATUS[0]}"
set -e

echo
echo "---- Grid configure, build, install ($(date +%T))"
TARGET=cpu bash "${SCRIPT}" configure
TARGET=cpu JOBS=64 bash "${SCRIPT}" build
TARGET=cpu bash "${SCRIPT}" install
echo "=== done $(date)"
