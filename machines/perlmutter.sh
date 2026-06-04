#!/usr/bin/env bash
# Perlmutter (NERSC) — A100 GPU stack, 2026 toolchain.
#
# Sourced by config.sh (auto-detected via $NERSC_HOST=perlmutter) or directly.
# At build time AND by sbatch scripts on compute nodes.
#
# GMP: complete in /usr (auto-detected by Grid configure, no flag needed).
# MPFR: Grid's RHMC Remez requires it. No dev header in /usr on Perlmutter;
#       complete copy (header + libmpfr.so.4) lives in the Cray PE gcc tree.

export CRAY_ACCEL_TARGET=nvidia80

module load PrgEnv-gnu
module load cudatoolkit/12.9
module load craype-accel-nvidia80

export GRID_MPFR_PREFIX=/opt/cray/pe/gcc/mpfr/3.1.4
export LD_LIBRARY_PATH="${GRID_MPFR_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Cluster-specific dependency paths
export CLIME_ROOT=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/lime-install
export CUDA_ARCH=sm_80
