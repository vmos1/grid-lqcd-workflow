#!/usr/bin/env bash
# lq cluster (AlmaLinux 8, NVIDIA A100) environment.
#
# Sourced by config.sh when MACHINE=lq (set explicitly by build_grid_lq.sh,
# or auto-detected via /lustre2/nplqcd).

# Modules loaded explicitly in step_setup_env (build_grid_lq.sh)
export LQ_MODULES="cmake gompi/2023a ucx_cuda/1.14.1_cuda_12.2.1 ucc_cuda/1.2.0_cuda_12.2.1 gcc/12.3.0"

# NVHPC 23.7 / CUDA 12.2
export NVHPC_ROOT=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/23.7
export CUDA_HOME=${NVHPC_ROOT}/cuda/12.2
export NVIDIALIB=${NVHPC_ROOT}/math_libs/lib64
export NVIDIAINCLUDE=${NVHPC_ROOT}/math_libs/include

# Shared group dependencies (pre-built, read-only)
export CLIME_ROOT=/lustre2/nplqcd/dwf/c-lime/install
export HDF5_ROOT=/srv/software/el8/x86_64/eb/HDF5/1.14.2-gompi-2023a

export CUDA_ARCH=sm_80

# lq builds go in a -gpu suffixed dir to distinguish from CPU builds
# BASE_DIR and GRID_PROFILE are set by config.sh before this file is sourced
export GRID_BUILD="${BASE_DIR}/build-${GRID_PROFILE}-gpu"
