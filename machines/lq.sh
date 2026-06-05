#!/usr/bin/env bash
# lq cluster (AlmaLinux 8, NVIDIA A100) environment.
#
# The single source of truth for the lq toolchain + runtime environment, for
# BOTH building and running. Sourced by config.sh when MACHINE=lq (auto-detected
# via /lustre2/nplqcd, or forced with MACHINE=lq). config.sh sets BASE_DIR and
# GRID_PROFILE before this file is sourced.

# ── Modules ───────────────────────────────────────────────────────────────────
# cmake + MPI (gompi) + CUDA-aware comms (ucx/ucc) + gcc + HDF5.
# CUDA/nvcc itself comes from NVHPC via the paths below (no cuda module).
export LQ_MODULES="cmake gompi/2023a ucx_cuda/1.14.1_cuda_12.2.1 ucc_cuda/1.2.0_cuda_12.2.1 gcc/12.3.0 hdf5/1.14.2_gompi_2023a"
if command -v module >/dev/null 2>&1; then
    module load ${LQ_MODULES}
fi

# ── NVHPC 23.7 / CUDA 12.2 ────────────────────────────────────────────────────
export NVHPC_ROOT=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/23.7
export CUDA_HOME=${NVHPC_ROOT}/cuda/12.2
export NVIDIALIB=${NVHPC_ROOT}/math_libs/lib64
export NVIDIAINCLUDE=${NVHPC_ROOT}/math_libs/include
export PATH=${NVHPC_ROOT}/cuda/bin:${PATH}
export LD_LIBRARY_PATH=${NVHPC_ROOT}/cuda/lib64:${NVHPC_ROOT}/math_libs/lib64:${NVHPC_ROOT}/compilers/lib:${LD_LIBRARY_PATH}
export CUDA_CACHE_PATH=${BASE_DIR}/.cuda_cache
export CUDA_ARCH=sm_80

# ── Shared group dependencies (pre-built, read-only) ──────────────────────────
export CLIME_ROOT=/lustre2/nplqcd/dwf/c-lime/install
export HDF5_ROOT=/srv/software/el8/x86_64/eb/HDF5/1.14.2-gompi-2023a

# ── Build tree (lq GPU builds use a -gpu suffix to distinguish from CPU) ───────
export GRID_BUILD="${BASE_DIR}/build-${GRID_PROFILE}-gpu"
