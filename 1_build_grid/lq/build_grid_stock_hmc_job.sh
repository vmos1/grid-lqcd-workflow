#!/bin/bash
#SBATCH --job-name=grid-stock-p08-gpu-build
#SBATCH --partition=lq1_cpu
#SBATCH --account=nplqcd.lq1_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=intel6248
#SBATCH --time=02:30:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_grid_stock_hmc_gpu_%j.log
#
# Compile and install the lq GPU libGrid (stock 3d3eff86 + patches 01-08 + lq fixes) on an
# lq1 CPU node. prepare, patch and configure run first on the login node:
#   bash 1_build_grid/lq/build_grid_stock_hmc.sh all
# Why lq1 and not lq2: nvcc needs no GPU, and lq1 runs EL8 like the lq2 GPU nodes. Not lq3:
# it runs EL9, so a library built there would carry the wrong glibc for lq2.
# 40 cores, 190 GB per node: JOBS=20 keeps nvcc (2-5 GB per process) inside memory.
set -euo pipefail
SCRIPT=/lustre2/nplqcd/vayyar/grid_qcd/grid-lqcd-workflow/1_build_grid/lq/build_grid_stock_hmc.sh

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date)"
# Guard: the node must give the same toolchain configure saw on the login node (modules can
# differ on compute nodes; /lustre2/nplqcd/Grid-CPU/AGENT_REFERENCE.md reports unusable
# MODULEPATH on lq1 when jobs are submitted from the EL9 lq3 login node). Stop in seconds if not.
source /lustre2/nplqcd/vayyar/grid_qcd/grid-lqcd-workflow/machines/lq.sh
echo "nvcc:   $(command -v nvcc)"
echo "mpicxx: $(command -v mpicxx)"
[[ "$(command -v nvcc)" == /srv/software/el8/x86_64/hpc/cuda/12.2.1/bin/nvcc ]]
[[ "$(command -v mpicxx)" == /srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/bin/mpicxx ]]

JOBS=${JOBS:-20} bash "${SCRIPT}" build
bash "${SCRIPT}" install
echo "=== done $(date)"
