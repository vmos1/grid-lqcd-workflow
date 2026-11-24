#!/bin/bash
# env_chroma_lq.sh -- run-time environment of the group's Chroma GPU build on lq2 (Fermilab), the
# lq twin of ../perlmutter/env_chroma_pm.sh. Sourced by chroma_baseg_mds.sh beside it, at the top
# of the job; source it in a login shell to inspect the binary (ldd "$CHROMA_HMC").
#
# The binary. /lustre2/nplqcd/chroma/install_oct_2025 (group space, read only), reached in our
# layout through the link installs/external/chroma. Built 2025-10-07 by the group with
# /lustre2/nplqcd/chroma/build_scripts/lq2.sh: chromaform --mg --cuda --zen3 --std=c++20
# --blas=mkl, flavour chroma-quda-qdp-jit-double-nd4-cmake (QDP-JIT with LLVM 17, double, Nd 4;
# QUDA sm_80 with multigrid and QMP comms). Its CMake caches name the toolchain: gcc/g++ of
# GCCcore 12.3.0, nvcc of NVHPC 23.11 (CUDA 12.3), MPI = the Open MPI 3.1.5 bundled with NVHPC
# 23.11 (module nvhpc-openmpi3/23.11). The modules below are the group's own run-time set
# (/lustre2/nplqcd/env_lq2_chroma.sh, sourced by their lq2 production up to 2025-12) without
# cmake and their Python, and with every cache in this workspace instead of group space.
#
# What each piece is for (ldd of bin/hmc on the login node, 2026-11-23):
#   gcc/12.3.0             libstdc++ (GLIBCXX_3.4.29; EL8's /lib64 copy is too old), libgomp
#   nvhpc-openmpi3/23.11   libcublas.so.12, libcufft.so.11 (math_libs, CUDA 12.3, the build's own).
#                          libmpi.so.40 is Open MPI 3.1.5 from the binary's RPATH in any case.
#   cuda/12.2.1            CUDA_HOME / CUDA_PATH only: QDP-JIT links $CUDA_PATH/nvvm/libdevice/
#                          libdevice.10.bc at run time. No library of the binary comes from it.
#   $CHROMA_INSTALL/llvm/lib              libLTO.so.17, libRemarks.so.17 (QDP-JIT's LLVM)
#   GCCcore 12.3.0 lib64 first, again     NVHPC's compilers/lib (prepended by nvhpc-openmpi3)
#                          carries its own libgomp.so.1; the binary is built by GCC and must load
#                          GNU's (the group's env does the same).
#
# Open MPI 3.1.5 on lq2 (the launcher checks both before the trajectory):
#  - No UCX: pml ob1, btl openib / vader / smcuda / tcp. Its bundled libibverbs (rdma-core 46)
#    looks for providers under /usr/local, which lq does not have, so openib finds no HCA and
#    inter-node messages fall back to TCP (the group's logs: "OpenFabrics (openib) ... unable to
#    find any relevant network interfaces"). CHROMA_IBV=system (default) sets RDMAV_DRIVERS to
#    the system mlx5 provider (rdma-core 48, same private ABI 34): on the login node the bundled
#    library then sees 1 device instead of 0 (2026-11-23). CHROMA_IBV=bundled = the group's setup.
#  - Its internal PMIx is 2.2.3 (security modules native and none, no munge). srun --mpi=pmix
#    (plugin pmix_v5) started it in the group's runs up to 2025-12; HPC-X's PMIx 3 could not be
#    started under today's Slurm 25.11.8 for want of munge (logs/mpi_locality_hpcx_1363336.log).
# =============================================================================================

case $- in *u*) _cl_nounset=1 ;; *) _cl_nounset=0 ;; esac
set +u
_cl_err=""

CHROMA_LQ_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # workspace root
export CHROMA_LQ_ROOT

# Lmod is not initialised in non-login shells (sbatch, scripts): as machines/lq.sh.
if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/00-modulepath.sh
    source /etc/profile.d/modules.sh
    source /etc/profile.d/zz_lmod.sh
fi
module purge
module load gcc/12.3.0 cuda/12.2.1 nvhpc-openmpi3/23.11 || _cl_err="module load gcc/12.3.0 cuda/12.2.1 nvhpc-openmpi3/23.11 failed"

export CHROMA_INSTALL=${CHROMA_INSTALL:-$CHROMA_LQ_ROOT/installs/external/chroma}
export CHROMA_FLAVOUR=chroma-quda-qdp-jit-double-nd4-cmake
export CHROMA_BINDIR=$CHROMA_INSTALL/$CHROMA_FLAVOUR/bin
export CHROMA_HMC=$CHROMA_BINDIR/hmc       # gauge generation (our <MCControl>/<HMCTrj> inputs)
export CHROMA_BIN=$CHROMA_BINDIR/chroma    # measurement driver, other input schema
export CHROMA_INPUTS=$CHROMA_LQ_ROOT/grid-lqcd-workflow/5_studies/chroma_inputs/candidates
if [ ! -x "$CHROMA_HMC" ]; then
    _cl_err="no executable hmc at $CHROMA_HMC; the layout link is made with: ln -s /lustre2/nplqcd/chroma/install_oct_2025 $CHROMA_LQ_ROOT/installs/external/chroma"
fi
export LD_LIBRARY_PATH=$CHROMA_INSTALL/llvm/lib:/srv/software/el8/x86_64/eb/GCCcore/12.3.0/lib64:$LD_LIBRARY_PATH
export PATH=$CHROMA_BINDIR:$PATH

# Run-time knobs as env_chroma_pm.sh (they came from the lq2 Grid-TXQCD environment).
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_MANAGED_MEMORY=1
# CUDA's JIT cache for QDP-JIT's PTX: the workspace's (machines/lq.sh uses the same directory),
# never the group's /lustre2/nplqcd/cache or $HOME. QUDA_RESOURCE_PATH is set per run by the
# launcher (runs/<run>/quda_resource).
export CUDA_CACHE_PATH=$CHROMA_LQ_ROOT/builds/.cuda_cache

# Inter-node transport (see the header).
export CHROMA_IBV=${CHROMA_IBV:-system}
case $CHROMA_IBV in
    system)  export RDMAV_DRIVERS=/usr/lib64/libibverbs/libmlx5 ;;
    bundled) unset RDMAV_DRIVERS ;;
    *)       _cl_err="CHROMA_IBV=$CHROMA_IBV: system or bundled" ;;
esac

if [ "${_cl_nounset}" = 1 ]; then set -u; fi
unset _cl_nounset
if [ -n "${_cl_err}" ]; then
    echo "env_chroma_lq.sh: ${_cl_err}" >&2
    unset _cl_err
    return 1
fi
unset _cl_err

if [ -t 1 ]; then
    module list 2>&1
    echo "CHROMA_HMC=$CHROMA_HMC  CHROMA_IBV=$CHROMA_IBV  CUDA_PATH=$CUDA_PATH"
fi
