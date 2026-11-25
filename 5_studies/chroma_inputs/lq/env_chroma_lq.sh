#!/bin/bash
# env_chroma_lq.sh -- run-time environment of the Chroma GPU builds on lq2 (Fermilab), the lq twin
# of ../perlmutter/env_chroma_pm.sh. Sourced by chroma_baseg_mds.sh beside it, at the top of the
# job; source it in a login shell to inspect the binary (ldd "$CHROMA_HMC").
#
# CHROMA_STACK selects the build and the MPI stack it runs on:
#   group (default)  the group's install_oct_2025, described below: NVHPC 23.11's Open MPI 3.1.5,
#                    whose inter-node messages go over TCP on lq (2.9 GB/s, job 1365190).
#   gpu-lq2          ours, installs/external/chroma-gpu-lq2 (1_build_chroma/lq/build_chroma_gpu.sh;
#                    its PROVENANCE.txt): Perlmutter's source commits, built with gompi's mpicc/mpicxx
#                    and cuda/12.2.1. Run-time environment = machines/lq.sh's gpu stack, the one our
#                    Grid runs on lq2 (LQ_TARGET=gpu LQ_MPI_STACK=gompi: gompi/2023a = Open MPI 4.1.5
#                    with external PMIx 4.2.6 + UCX 1.14.1, ucx_cuda/1.14.1_cuda_12.2.1 which loads
#                    cuda/12.2.1, ucc_cuda) plus the openblas module the build linked (also reached
#                    through the binary's RUNPATH). libmpi.so.40 must be gompi's (RUNPATH set by
#                    mpicxx). Inter-node messages go through UCX on the mlx5 HCAs; there is no openib,
#                    so CHROMA_IBV does not apply (na). QDP-JIT carries CUDA 12.2's libdevice inside
#                    the binary (chromaform packs it at build time); $CHROMA_INSTALL/llvm/lib comes
#                    first on LD_LIBRARY_PATH as for the group build.
#   Keep group the default until gpu-lq2 has passed the 48^3 validation (same deltaH as Perlmutter's
#   2001 to solver tolerance, seconds_for_trajectory within 5% of 1,652 s).
#
# The group binary. /lustre2/nplqcd/chroma/install_oct_2025 (group space, read only), reached in our
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

# The stack (see the header). CHROMA_LIBMPI_WANT is the libmpi.so.40 the launcher requires in ldd.
export CHROMA_STACK=${CHROMA_STACK:-group}
case $CHROMA_STACK in
group)
    module purge
    module load gcc/12.3.0 cuda/12.2.1 nvhpc-openmpi3/23.11 || _cl_err="module load gcc/12.3.0 cuda/12.2.1 nvhpc-openmpi3/23.11 failed"
    export CHROMA_INSTALL=${CHROMA_INSTALL:-$CHROMA_LQ_ROOT/installs/external/chroma}
    export CHROMA_LIBMPI_WANT=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/23.11/comm_libs/openmpi/openmpi-3.1.5/lib/libmpi.so.40
    _cl_ld=$CHROMA_INSTALL/llvm/lib:/srv/software/el8/x86_64/eb/GCCcore/12.3.0/lib64
    _cl_missing="the layout link is made with: ln -s /lustre2/nplqcd/chroma/install_oct_2025 $CHROMA_LQ_ROOT/installs/external/chroma"
    ;;
gpu-lq2)
    # machines/lq.sh purges and loads gompi/2023a ucx_cuda/1.14.1_cuda_12.2.1 ucc_cuda/1.2.0_cuda_12.2.1.
    LQ_TARGET=gpu LQ_MPI_STACK=gompi source "$CHROMA_LQ_ROOT/grid-lqcd-workflow/machines/lq.sh" ||
        _cl_err="machines/lq.sh (LQ_TARGET=gpu LQ_MPI_STACK=gompi) failed"
    set +u
    module load openblas/0.3.24_gcc_12.3.0 || _cl_err="module load openblas/0.3.24_gcc_12.3.0 failed"
    export CHROMA_INSTALL=${CHROMA_INSTALL:-$CHROMA_LQ_ROOT/installs/external/chroma-gpu-lq2}
    export CHROMA_LIBMPI_WANT=/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/lib/libmpi.so.40
    _cl_ld=$CHROMA_INSTALL/llvm/lib
    _cl_missing="build it with grid-lqcd-workflow/1_build_chroma/lq/build_chroma_gpu_job.sh"
    ;;
*)
    _cl_err="CHROMA_STACK=$CHROMA_STACK: group or gpu-lq2"
    ;;
esac

export CHROMA_FLAVOUR=chroma-quda-qdp-jit-double-nd4-cmake
export CHROMA_BINDIR=$CHROMA_INSTALL/$CHROMA_FLAVOUR/bin
export CHROMA_HMC=$CHROMA_BINDIR/hmc       # gauge generation (our <MCControl>/<HMCTrj> inputs)
export CHROMA_BIN=$CHROMA_BINDIR/chroma    # measurement driver, other input schema
export CHROMA_INPUTS=$CHROMA_LQ_ROOT/grid-lqcd-workflow/5_studies/chroma_inputs/candidates
if [ -z "${_cl_err}" ] && [ ! -x "$CHROMA_HMC" ]; then
    _cl_err="no executable hmc at $CHROMA_HMC (CHROMA_STACK=$CHROMA_STACK); ${_cl_missing}"
fi
export LD_LIBRARY_PATH=${_cl_ld}:$LD_LIBRARY_PATH
unset _cl_ld _cl_missing
export PATH=$CHROMA_BINDIR:$PATH

# Run-time knobs as env_chroma_pm.sh (they came from the lq2 Grid-TXQCD environment).
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_MANAGED_MEMORY=1
# CUDA's JIT cache for QDP-JIT's PTX: the workspace's (machines/lq.sh uses the same directory),
# never the group's /lustre2/nplqcd/cache or $HOME. QUDA_RESOURCE_PATH is set per run by the
# launcher (runs/<run>/quda_resource).
export CUDA_CACHE_PATH=$CHROMA_LQ_ROOT/builds/.cuda_cache

# Inter-node transport (see the header). gpu-lq2: UCX, no openib, nothing to set.
if [ "$CHROMA_STACK" = gpu-lq2 ]; then
    export CHROMA_IBV=na
    unset RDMAV_DRIVERS
else
    export CHROMA_IBV=${CHROMA_IBV:-system}
    case $CHROMA_IBV in
        system)  export RDMAV_DRIVERS=/usr/lib64/libibverbs/libmlx5 ;;
        bundled) unset RDMAV_DRIVERS ;;
        *)       _cl_err="CHROMA_IBV=$CHROMA_IBV: system or bundled" ;;
    esac
fi

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
    echo "CHROMA_STACK=$CHROMA_STACK  CHROMA_HMC=$CHROMA_HMC  CHROMA_IBV=$CHROMA_IBV  CUDA_PATH=$CUDA_PATH"
fi
