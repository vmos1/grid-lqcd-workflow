#!/bin/bash
# Build QUDA from source on Perlmutter (NERSC) for Grid-TXQCD's QUDA wrappers.
#
# Target: A100 (sm_80), CUDA 12.9, GCC 14.3 via Cray CC wrapper (nvcc -ccbin CC,
# the same host-compiler combo Grid-TXQCD/systems/Perlmutter/config-command uses).
#
# Features enabled for our hasenbusch tuning (QUDA_FORCE=1 strange RHMC force):
#   * Wilson + clover Dirac (clover fermion force = computeCloverForceQuda)
#   * QMP multi-GPU comms (Grid links -lqmp), built via CPM (QUDA_DOWNLOAD_USQCD)
#   * Multigrid OFF — only the QUDA_SOLVER path needs it; the tuning force path
#     does not. Drop -DQUDA_MULTIGRID=ON in if you later want the QUDA solver too.
#
# Usage:
#   bash quda_build_perlmutter.sh configure   # cmake (needs login-node network)
#   bash quda_build_perlmutter.sh build       # make -j (no network needed)
#   bash quda_build_perlmutter.sh install
#   bash quda_build_perlmutter.sh all
set -euo pipefail

ROOT=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
QUDA_SRC=${ROOT}/quda
BUILD=${QUDA_SRC}/build
INSTALL=${ROOT}/quda-install
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
    -DQUDA_MULTIGRID=OFF \
    -DQUDA_QMP=ON \
    -DQUDA_QIO=OFF \
    -DQUDA_MPI=OFF \
    -DQUDA_DOWNLOAD_USQCD=ON \
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
