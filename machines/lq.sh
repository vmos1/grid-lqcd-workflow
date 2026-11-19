#!/usr/bin/env bash
# lq (Fermilab) environment for grid_qcd, at build time and inside sbatch jobs.
# Sourced directly by the lq build and launch scripts (and by config.sh, MACHINE=lq).
#
# GPU target: lq2_gpu, 18 nodes x 4 A100 (sm_80). CPU target (lq3_cpu): added later.
# One CUDA only: the cuda/12.2.1 module that ucx_cuda loads. The 2026-05 build also put
#   NVHPC 23.7's CUDA first on PATH, mixing two CUDA installs in one build; dropped.
# MPFR and GMP: complete in /usr (EL8 packages); Grid's configure finds them with no flag.
#   Never pass --with-mpfr=/usr: it adds -I/usr/include, which breaks libstdc++'s
#   #include_next, and -L/usr/lib, the 32-bit directory on EL8.
# OpenMPI 4.1.5 (gompi/2023a) is built --with-cuda=internal; UCX carries GPU buffers.

# Lmod is not initialised in non-login shells (sbatch, scripts): run the three profile.d
# files a login shell runs, in its order. They and Lmod's `module` read unset variables,
# so a caller's `set -u` is paused around them and restored after (2026-11-18: with it
# on, zz_lmod.sh aborts with "MODULEPATH: unbound variable").
case $- in *u*) _lq_nounset=1 ;; *) _lq_nounset=0 ;; esac
set +u
if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/00-modulepath.sh
    source /etc/profile.d/modules.sh
    source /etc/profile.d/zz_lmod.sh
fi
module purge
module load gompi/2023a ucx_cuda/1.14.1_cuda_12.2.1 ucc_cuda/1.2.0_cuda_12.2.1

# MPI at run time: LQ_MPI_STACK=gompi (default, the build's) or hpcx.
#   gompi's OpenMPI 4.1.5 (external PMIx 4.2.6) sees ranks on one node as non-local:
#   MPI_COMM_TYPE_SHARED gives groups of 1 under every mpirun and srun form, so Grid prints
#   "Node communicator of size 1" and sends every halo through MPI (probe job 1363335).
#   HPC-X 2.15 from the NVHPC 23.7 tree (OpenMPI 4.1.5rc2, internal PMIx 3, CUDA-aware, UCX 1.15)
#   groups them (probe job 1363336). Same libmpi.so.40 ABI, and the binaries carry RUNPATH, which
#   LD_LIBRARY_PATH beats: a gompi-built binary runs on HPC-X unchanged. Its module file is
#   broken (loads a missing "hpcx" module); its own hpcx-init.sh is used instead. hcoll and UCC
#   collectives are switched off so the collectives stay Open MPI's tuned ones, as with gompi.
#   HPC-X cannot be launched by srun (its PMIx has no munge plugin): use its mpirun.
LQ_HPCX_INIT=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/2023/comm_libs/12.2/hpcx/hpcx-2.15/hpcx-init.sh
LQ_MPI_STACK=${LQ_MPI_STACK:-gompi}
_lq_stack_ok=1
case ${LQ_MPI_STACK} in
    gompi) ;;
    hpcx)
        source "${LQ_HPCX_INIT}" && hpcx_load || _lq_stack_ok=0
        export OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_coll_ucc_enable=0
        ;;
    *) _lq_stack_ok=0 ;;
esac
if [ "${_lq_nounset}" = 1 ]; then set -u; fi
unset _lq_nounset
if [ "${_lq_stack_ok}" != 1 ]; then
    unset _lq_stack_ok
    echo "lq.sh: LQ_MPI_STACK=${LQ_MPI_STACK} could not be set up (gompi or hpcx)" >&2
    return 1
fi
unset _lq_stack_ok

LQ_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # workspace root

export CUDA_ARCH=sm_80
export CLIME_ROOT=${LQ_ROOT}/installs/external/lime  # link -> /lustre2/nplqcd/dwf/c-lime/install
export CUDA_CACHE_PATH=${LQ_ROOT}/builds/.cuda_cache

# Python: the repo's uv environment (uv in ~/.local/bin, Python 3.12).
export PATH="$HOME/.local/bin:$PATH"
