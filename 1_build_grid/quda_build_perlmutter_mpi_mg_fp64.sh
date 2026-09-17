#!/bin/bash
# Build QUDA with DOUBLE-PRECISION MULTIGRID enabled, for the MG precision study.
#
# This is quda_build_perlmutter_mpi_mg.sh with exactly ONE difference:
#   -DCMAKE_CUDA_FLAGS=-DGPU_MULTIGRID_DOUBLE
# plus a separate build dir and install prefix so quda-install-mpi-mg -- which the
# whole MG campaign links against -- stays pristine.
#
# WHY THIS BUILD EXISTS
# ---------------------
# QUDA_PRECISION=14 (the default, and what quda-install-mpi-mg has) compiles in
# double+single+half, so precision is normally a pure RUNTIME knob. Double
# precision MULTIGRID is the exception: upstream gates the MG kernels behind a
# macro that is commented out, quda/include/multigrid.h:5-13 --
#
#     // at the moment double-precision multigrid is only enabled when debugging
#     #ifdef HOST_DEBUG
#     //#define GPU_MULTIGRID_DOUBLE
#     #endif
#
# so the double MG template instantiations are simply absent from libquda.so.
# Asking for them at runtime does NOT fail validation; it aborts partway through
# MG setup at `is_enabled_multigrid_double()`:
#   "Double precision multigrid has not been enabled"
#   build-mpi-mg/lib/block_orthogonalize_24_32.cu:290
# Measured 2026-09-15, run c3_qudamg_fp64, all 16 ranks.
#
# Because the guard is a bare #ifdef (NOT #cmakedefine -- there is no
# QUDA_MULTIGRID_DOUBLE cmake option anywhere in the tree), defining the macro on
# the compile line is sufficient and NO SOURCE EDIT IS NEEDED. The upstream clone
# stays untouched, per the standing fork-don't-edit-shared-code preference.
#
# WHAT IT IS FOR: Grid's general-coarsened MG path is fp64 throughout, while
# QUDA's production MG runs single preconditioner + half null vectors + half
# coarse halos. A Grid-vs-QUDA MG ratio therefore confounds "how good is the
# machinery" with "how much arithmetic is being skipped". This build supplies the
# fp64-vs-fp64 cell that separates them.
#
# ⛔ THE fp64 QUDA ROW IS A DIAGNOSTIC, NOT A DELIVERABLE. It is slower than
# QUDA's best and therefore FLATTERS Grid. Never quote it as the headline
# Grid-vs-QUDA number; the production-relevant comparison is against mixed.
#
# ⛔ BUILD ON $PSCRATCH, NOT CFS: m4599 is at ~99% of its CFS FILE-COUNT quota at
# only ~56% space, and a QUDA build tree is tens of thousands of objects. This is
# invisible to df/du. See memory cfs-inode-quota-wall.
#
# ⛔ TOOLCHAIN MUST MATCH: quda-install-mpi-mg was built with cudatoolkit/12.9
# (see CMAKE_CRAYPE_LOADEDMODULES in its CMakeCache.txt). A CPU login/compute node
# may default to a different nvcc -- machines/perlmutter.sh pins the modules, so
# it is sourced below exactly as the MG build does.
#
# Usage (build on a CPU allocation to avoid burning GPU hours -- nvcc
# cross-compiles for sm_80 and needs no GPU present):
#   bash quda_build_perlmutter_mpi_mg_fp64.sh configure
#   JOBS=64 bash quda_build_perlmutter_mpi_mg_fp64.sh build
#   bash quda_build_perlmutter_mpi_mg_fp64.sh install
set -euo pipefail

ROOT=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
QUDA_SRC=${ROOT}/quda
: "${PSCRATCH:?PSCRATCH must be set on Perlmutter}"
BUILD=${BUILD:-${PSCRATCH}/quda-build-mpi-mg-fp64}
INSTALL=${INSTALL:-${PSCRATCH}/quda-install-mpi-mg-fp64}
JOBS=${JOBS:-16}

source ${ROOT}/grid-lqcd-workflow/machines/perlmutter.sh >/dev/null 2>&1 || \
  source ${ROOT}/grid-lqcd-workflow/machines/perlmutter.sh

configure() {
  local CC_ABS CXX_ABS
  CC_ABS=$(command -v cc)
  CXX_ABS=$(command -v CC)
  mkdir -p "${BUILD}"
  cd "${BUILD}"
  # Every flag below is copied verbatim from quda_build_perlmutter_mpi_mg.sh.
  # The ONLY addition is -DCMAKE_CUDA_FLAGS.
  cmake "${QUDA_SRC}" \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
    -DCMAKE_C_COMPILER="${CC_ABS}" \
    -DCMAKE_CXX_COMPILER="${CXX_ABS}" \
    -DCMAKE_CUDA_HOST_COMPILER="${CXX_ABS}" \
    -DCMAKE_CUDA_FLAGS="-DGPU_MULTIGRID_DOUBLE" \
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

# Confirms the macro actually reached the compiled library. Without this the
# failure mode is a successful build that still aborts at runtime with
# "Double precision multigrid has not been enabled" -- the exact thing this
# build exists to fix, and it is only detectable on a GPU.
verify() {
  local flags
  flags=$(grep -m1 "CMAKE_CUDA_FLAGS:STRING" "${BUILD}/CMakeCache.txt" || true)
  printf 'CMakeCache CUDA flags: %s\n' "${flags}"
  case "${flags}" in
    *GPU_MULTIGRID_DOUBLE*) printf 'OK: GPU_MULTIGRID_DOUBLE is in the build flags\n' ;;
    *) printf 'ERROR: GPU_MULTIGRID_DOUBLE NOT set -- the fp64 MG row will abort at runtime\n' >&2; exit 1 ;;
  esac
  [[ -f "${INSTALL}/lib/libquda.so" ]] && printf 'OK: %s\n' "${INSTALL}/lib/libquda.so"
}

case "${1:-all}" in
  configure) configure ;;
  build)     build ;;
  install)   install ;;
  verify)    verify ;;
  all)       configure; build; install; verify ;;
  *) echo "usage: $0 {configure|build|install|verify|all}" >&2; exit 1 ;;
esac
