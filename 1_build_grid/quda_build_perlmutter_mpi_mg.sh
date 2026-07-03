#!/bin/bash
# Build QUDA from source on Perlmutter (NERSC) for Grid-TXQCD's QUDA wrappers,
# using **native MPI comms** (QUDA_MPI=ON) AND **multigrid** (QUDA_MULTIGRID=ON).
#
# This is the MG variant of quda_build_perlmutter_mpi.sh. It is identical except:
#   * -DQUDA_MULTIGRID=ON      (light-sector MG-GCR preconditioner)
#   * SEPARATE build dir (build-mpi-mg) + install prefix (quda-install-mpi-mg)
#     so the validated strange MPI build (quda-install-mpi) stays pristine.
#
# WHY MPI comms (unchanged from the non-MG build): under QMP, QUDA's rank layout
# cannot reproduce Grid's NUMA-remapped MPI rank order -> halo mismatch for >=2
# partitioned dirs. QUDA_MPI=ON lets us hand QUDA Grid's own communicator + rank
# map (setMPICommHandleQuda + initCommsGridQuda rank_from_coords).
#
# MG note: QUDA_MULTIGRID_NVEC_LIST (default 6,24,32) and QUDA_MULTIGRID_MRHS_LIST
# (default 16) already cover the n_vec=24 we use (QudaMultigridConfig.h), so no
# extra MG sub-flags are needed. MG roughly doubles the build time vs MG-off.
#
# Target: A100 (sm_80), CUDA 12.9, GCC 14.3 via Cray CC wrapper (nvcc -ccbin CC).
#
# Usage:
#   bash quda_build_perlmutter_mpi_mg.sh configure   # cmake (needs login-node network)
#   bash quda_build_perlmutter_mpi_mg.sh build       # make -j (no network needed)
#   bash quda_build_perlmutter_mpi_mg.sh install
#   bash quda_build_perlmutter_mpi_mg.sh all
set -euo pipefail

ROOT=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
QUDA_SRC=${ROOT}/quda
BUILD=${QUDA_SRC}/build-mpi-mg
INSTALL=${ROOT}/quda-install-mpi-mg
JOBS=${JOBS:-16}

source ${ROOT}/grid-lqcd-workflow/machines/perlmutter.sh >/dev/null 2>&1 || \
  source ${ROOT}/grid-lqcd-workflow/machines/perlmutter.sh

configure() {
  # nvcc -ccbin needs an absolute path to the Cray CC wrapper.
  local CC_ABS CXX_ABS
  CC_ABS=$(command -v cc)
  CXX_ABS=$(command -v CC)
  mkdir -p "${BUILD}"
  cd "${BUILD}"
  cmake "${QUDA_SRC}" \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
    -DCMAKE_C_COMPILER="${CC_ABS}" \
    -DCMAKE_CXX_COMPILER="${CXX_ABS}" \
    -DCMAKE_CUDA_HOST_COMPILER="${CXX_ABS}" \
    -DQUDA_GPU_ARCH=sm_80 \
    -DQUDA_DIRAC_WILSON=ON \
    -DQUDA_DIRAC_CLOVER=ON \
    -DQUDA_DIRAC_CLOVER_HASENBUSCH=ON \
    -DQUDA_DIRAC_TWISTED_MASS=OFF \
    -DQUDA_DIRAC_TWISTED_CLOVER=OFF \
    -DQUDA_DIRAC_DOMAIN_WALL=OFF \
    -DQUDA_DIRAC_STAGGERED=OFF \
    -DQUDA_DIRAC_LAPLACE=OFF \
    -DQUDA_DIRAC_COVDEV=OFF \
    -DQUDA_MULTIGRID=ON \
    -DQUDA_QMP=OFF \
    -DQUDA_QIO=OFF \
    -DQUDA_MPI=ON \
    -DQUDA_CLOVER_DYNAMIC=ON \
    -DQUDA_CLOVER_RECONSTRUCT=ON \
    -DQUDA_BUILD_SHAREDLIB=ON \
    -DQUDA_BUILD_ALL_TESTS=OFF
}

build()   { cd "${BUILD}"; make -j"${JOBS}"; }
install() { cd "${BUILD}"; make -j"${JOBS}" install; }

case "${1:-all}" in
  configure) configure ;;
  build)     build ;;
  install)   install ;;
  all)       configure; build; install ;;
  *) echo "usage: $0 {configure|build|install|all}" >&2; exit 1 ;;
esac
