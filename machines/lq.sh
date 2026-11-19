#!/usr/bin/env bash
# lq (Fermilab) environment for grid_qcd, at build time and inside sbatch jobs.
# Sourced directly by the lq build and launch scripts (and by config.sh, MACHINE=lq).
#
# Two targets, chosen with LQ_TARGET:
#   gpu (default)  lq2_gpu, 18 nodes x 4 A100 (sm_80), EL8, Lmod modules (EasyBuild tree).
#                  Built on lq1 (EL8), run on lq2.
#   cpu            lq3_cpu, 23 nodes x 2 x 32-core EPYC 9355 (Zen 5), 1.5 TB, NDR InfiniBand
#                  (mlx5_0), EL9. Built and run on lq3 only: the EL9 toolchain does not run on
#                  the EL8 login node or on lq1/lq2.
#
# GPU notes.
# One CUDA only: the cuda/12.2.1 module that ucx_cuda loads. The 2026-05 build also put
#   NVHPC 23.7's CUDA first on PATH, mixing two CUDA installs in one build; dropped.
# MPFR and GMP: complete in /usr (EL8 packages); Grid's configure finds them with no flag.
#   Never pass --with-mpfr=/usr: it adds -I/usr/include, which breaks libstdc++'s
#   #include_next, and -L/usr/lib, the 32-bit directory on EL8.
# OpenMPI 4.1.5 (gompi/2023a) is built --with-cuda=internal; UCX carries GPU buffers.
#
# CPU notes. The toolchain of the lq3 reference Grid build (/lustre2/nplqcd/Grid-CPU,
#   AGENT_REFERENCE.md and systems/LQ3/sourceme.sh): gcc 15.3 from the EL9 spack tree
#   (environment gcc-15-toolset), GMP and MPFR from the same spack (--with-gmp/--with-mpfr
#   point there, never at /usr), and OpenMPI 4.1.8 built by the Grid-CPU project against the
#   system UCX 1.21, the only UCX on lq3 with the mlx5 IB transports (validated 1-16 nodes).
#   pml/osc ucx and the IB device are set here; forward them with -x when mpirun spans nodes.

case $- in *u*) _lq_nounset=1 ;; *) _lq_nounset=0 ;; esac
set +u
LQ_TARGET=${LQ_TARGET:-gpu}
_lq_err=""

case ${LQ_TARGET} in
gpu)
    # Lmod is not initialised in non-login shells (sbatch, scripts): run the three profile.d
    # files a login shell runs, in its order. They and Lmod's `module` read unset variables,
    # so a caller's `set -u` is paused around them and restored after (2026-11-18: with it
    # on, zz_lmod.sh aborts with "MODULEPATH: unbound variable").
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
    #   HPC-X 2.15 from the NVHPC 23.7 tree (OpenMPI 4.1.5rc2, internal PMIx 3, CUDA-aware,
    #   UCX 1.15) groups them (probe job 1363336). Same libmpi.so.40 ABI, and the binaries carry
    #   RUNPATH, which LD_LIBRARY_PATH beats: a gompi-built binary runs on HPC-X unchanged. Its
    #   module file is broken (loads a missing "hpcx" module); its own hpcx-init.sh is used
    #   instead. hcoll and UCC collectives are switched off so the collectives stay Open MPI's
    #   tuned ones, as with gompi. HPC-X cannot be launched by srun (its PMIx has no munge
    #   plugin): use its mpirun.
    LQ_HPCX_INIT=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/2023/comm_libs/12.2/hpcx/hpcx-2.15/hpcx-init.sh
    LQ_MPI_STACK=${LQ_MPI_STACK:-gompi}
    case ${LQ_MPI_STACK} in
        gompi) ;;
        hpcx)
            source "${LQ_HPCX_INIT}" && hpcx_load || _lq_err="HPC-X setup failed (${LQ_HPCX_INIT})"
            export OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_coll_ucc_enable=0
            ;;
        *) _lq_err="LQ_MPI_STACK=${LQ_MPI_STACK}: gompi or hpcx" ;;
    esac
    ;;
cpu)
    LQ_MPI_STACK=${LQ_MPI_STACK:-lq3-ompi418}
    if [ "${LQ_MPI_STACK}" != lq3-ompi418 ]; then
        _lq_err="LQ_TARGET=cpu has one MPI (lq3-ompi418), got LQ_MPI_STACK=${LQ_MPI_STACK}"
    elif ! grep -q '^VERSION_ID="9' /etc/os-release 2>/dev/null; then
        _lq_err="LQ_TARGET=cpu needs an EL9 lq3 node; $(hostname) runs $(. /etc/os-release && echo "${PRETTY_NAME}")"
    else
        # FFTW is required, not optional: stock Grid's CPU FFT.h does not compile without it
        # (GPU builds use cuFFT instead; job 1363360 failed in FFT.h without --with-fftw).
        source /srv/software/el9/spack/v1.2.2/setup-env.sh &&
            spack env activate gcc-15-toolset &&
            spack load gcc@15 &&
            spack load fftw || _lq_err="spack gcc-15-toolset setup failed"
        if [ -z "${_lq_err}" ]; then
            export GMP=$(spack find --paths gmp | awk '/^gmp/ {print $2}')
            export MPFR=$(spack find --paths mpfr | awk '/^mpfr/ {print $2}')
            export FFTW=$(spack find --paths fftw | awk '/^fftw/ {print $2}')
            { [ -d "${GMP}/lib" ] && [ -d "${MPFR}/lib" ] && [ -d "${FFTW}/lib" ]; } ||
                _lq_err="spack gmp/mpfr/fftw not found (GMP='${GMP}' MPFR='${MPFR}' FFTW='${FFTW}')"
            export LQ_CPU_MPI_ROOT=/lustre2/nplqcd/install/openmpi-4.1.8-ucx-gcc15
            export PATH=${LQ_CPU_MPI_ROOT}/bin:${PATH}
            export LD_LIBRARY_PATH=${LQ_CPU_MPI_ROOT}/lib:${GMP}/lib:${MPFR}/lib:${FFTW}/lib:${LD_LIBRARY_PATH}
            export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx UCX_NET_DEVICES=mlx5_0:1
        fi
    fi
    ;;
*)
    _lq_err="LQ_TARGET=${LQ_TARGET}: gpu or cpu"
    ;;
esac

if [ "${_lq_nounset}" = 1 ]; then set -u; fi
unset _lq_nounset
if [ -n "${_lq_err}" ]; then
    echo "lq.sh: ${_lq_err}" >&2
    unset _lq_err
    return 1
fi
unset _lq_err

LQ_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # workspace root

export CLIME_ROOT=${LQ_ROOT}/installs/external/lime  # link -> /lustre2/nplqcd/dwf/c-lime/install
if [ "${LQ_TARGET}" = gpu ]; then
    export CUDA_ARCH=sm_80
    export CUDA_CACHE_PATH=${LQ_ROOT}/builds/.cuda_cache
fi

# Python: the repo's uv environment (uv in ~/.local/bin, Python 3.12).
export PATH="$HOME/.local/bin:$PATH"
