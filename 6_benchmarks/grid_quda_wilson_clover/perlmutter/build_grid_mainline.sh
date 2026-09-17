#!/usr/bin/env bash
# Build an unmodified stock Grid checkout in a disposable PSCRATCH staging tree.
#
# This script deliberately does not run Grid/bootstrap.sh: that script deletes the
# downloaded Eigen archive.  The equivalent preparation below is non-destructive.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKFLOW=$(cd "${HERE}/../../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)

source "${WORKFLOW}/machines/perlmutter.sh"

ACTION=${1:-all}
GRID_SRC=${GRID_SRC:-${ROOT}/Grid}
GRID_SHA=${GRID_SHA:-$(git -C "${GRID_SRC}" rev-parse HEAD)}
JOBS=${JOBS:-4}

# Grid's compile-time tracing: none|nvtx|roctx|timer. Empty (default) leaves the
# flag off the configure line entirely, so the standard build is unchanged.
#
# ⛔ --enable-tracing=nvtx DOES NOT BUILD ON CUDA 12.9 as shipped. Two breaks:
#   1. Grid/perfmon/Tracing.h includes <nvToolsExt.h>, but 12.9 ships that header
#      only under nvtx3/.  Worked around below with an extra -I; the upstream fix
#      is to include <nvtx3/nvToolsExt.h>.
#   2. configure.ac appends -lnvToolsExt, but that library no longer exists in the
#      toolkit at all -- NVTX has been header-only since CUDA 10 and the
#      compatibility shim was removed in 12.9.  The `patch-tracing` action below
#      strips it from the generated configure.
# Both affect any CUDA >= 12.9, and are worth reporting upstream.
GRID_TRACING=${GRID_TRACING:-}

: "${PSCRATCH:?PSCRATCH must be set on Perlmutter}"
STAGE_ROOT=${STAGE_ROOT:-${PSCRATCH}/grid_quda_wilson_clover/stock-grid/${GRID_SHA}}
STAGE_SRC=${STAGE_SRC:-${STAGE_ROOT}/Grid}
BUILD_DIR=${BUILD_DIR:-${STAGE_SRC}/build}
INSTALL_PREFIX=${INSTALL_PREFIX:-${STAGE_ROOT}/install}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-${PSCRATCH}/grid_quda_wilson_clover/downloads}
EIGEN_ARCHIVE=${DOWNLOAD_DIR}/eigen-3.4.0.tar.bz2
EIGEN_DIR=${STAGE_ROOT}/eigen-3.4.0
EIGEN_URL=https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
EIGEN_SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626
PREPARED_MARKER=${STAGE_SRC}/.grid-benchmark-prepared

print_paths() {
  printf 'GRID_SRC=%s\n' "${GRID_SRC}"
  printf 'GRID_SHA=%s\n' "${GRID_SHA}"
  printf 'STAGE_SRC=%s\n' "${STAGE_SRC}"
  printf 'BUILD_DIR=%s\n' "${BUILD_DIR}"
  printf 'INSTALL_PREFIX=%s\n' "${INSTALL_PREFIX}"
  printf 'JOBS=%s\n' "${JOBS}"
}

prepare() {
  if [[ -f "${PREPARED_MARKER}" ]]; then
    local prepared_sha
    prepared_sha=$(<"${PREPARED_MARKER}")
    if [[ "${prepared_sha}" != "${GRID_SHA}" ]]; then
      printf 'ERROR: staging tree contains Grid %s, requested %s\n' \
        "${prepared_sha}" "${GRID_SHA}" >&2
      exit 1
    fi
    printf 'Prepared source already exists: %s\n' "${STAGE_SRC}"
    return
  fi

  if [[ -e "${STAGE_SRC}" ]]; then
    printf 'ERROR: incomplete staging source exists: %s\n' "${STAGE_SRC}" >&2
    printf 'Inspect it. To discard it yourself, run:\n  rm -rf %q\n' "${STAGE_ROOT}" >&2
    exit 1
  fi

  if [[ -n "$(git -C "${GRID_SRC}" status --porcelain --untracked-files=no)" ]]; then
    printf 'ERROR: stock Grid has tracked modifications: %s\n' "${GRID_SRC}" >&2
    exit 1
  fi

  mkdir -p "${STAGE_ROOT}" "${DOWNLOAD_DIR}"
  git clone --no-hardlinks --no-checkout "${GRID_SRC}" "${STAGE_SRC}"
  git -C "${STAGE_SRC}" checkout --detach "${GRID_SHA}"

  if [[ ! -f "${EIGEN_ARCHIVE}" ]]; then
    wget --no-check-certificate --output-document="${EIGEN_ARCHIVE}" "${EIGEN_URL}"
  fi
  printf '%s  %s\n' "${EIGEN_SHA256}" "${EIGEN_ARCHIVE}" | sha256sum --check

  tar -xjf "${EIGEN_ARCHIVE}" -C "${STAGE_ROOT}"
  ln -s "${EIGEN_DIR}/Eigen" "${STAGE_SRC}/Grid/Eigen"
  ln -s "${EIGEN_DIR}/unsupported/Eigen" "${STAGE_SRC}/Grid/Eigen/unsupported"

  # bootstrap.sh's non-destructive equivalent: scripts/update_eigen.sh also
  # generates Grid/Eigen.inc (the automake file list for the vendored Eigen
  # headers, followed through the symlinks above) before autoreconf runs.
  # Without it, automake fails with "cannot open < Grid/Eigen.inc".
  (
    cd "${STAGE_SRC}/Grid"
    { echo 'eigen_files =\'
      find -L Eigen -type f -print | sed 's/^/  /;$q;s/$/ \\/'
    } > Eigen.inc
  )

  (
    cd "${STAGE_SRC}"
    ./scripts/filelist
    autoreconf -fvi
  )
  printf '%s\n' "${GRID_SHA}" > "${PREPARED_MARKER}"
}

# Strip the dead -lnvToolsExt from the generated configure. Idempotent, and a
# no-op unless tracing is requested. See the GRID_TRACING comment above.
patch_tracing() {
  [[ -n "${GRID_TRACING}" ]] || { printf 'GRID_TRACING unset; nothing to patch\n'; return; }
  local cfg="${STAGE_SRC}/configure"
  [[ -f "${cfg}" ]] || { printf 'ERROR: %s not found; run prepare first\n' "${cfg}" >&2; exit 1; }
  if grep -q -- '-lnvToolsExt' "${cfg}"; then
    sed -i 's/ -lnvToolsExt//' "${cfg}"
    printf 'patched: removed -lnvToolsExt from %s\n' "${cfg}"
  else
    printf 'already patched: no -lnvToolsExt in %s\n' "${cfg}"
  fi
}

configure_grid() {
  [[ -x "${STAGE_SRC}/configure" ]] || {
    printf 'ERROR: prepared configure script not found; run %s prepare first\n' "$0" >&2
    exit 1
  }

  local tracing_args=()
  local tracing_cxxflags=
  if [[ -n "${GRID_TRACING}" ]]; then
    tracing_args+=("--enable-tracing=${GRID_TRACING}")
    if [[ "${GRID_TRACING}" == nvtx ]]; then
      # ⛔ THIRD CUDA >= 12.9 BREAK, and the least obvious one.
      # CUB emits its own NVTX ranges: cub/detail/nvtx.cuh includes
      # <nvtx3/nvtx3.hpp>, the C++ wrapper, unless CCCL_DISABLE_NVTX or
      # NVTX_DISABLE is defined. Grid's Tracing.h uses the NVTX *C* API. Having
      # both in one translation unit breaks the C++ wrapper, which dies with
      # ~40 errors inside nvtx3.hpp ("nvtxColorType_t is undefined",
      # "_domain is not a nonstatic data member", ...) that name neither Grid
      # nor CUB and so look like a toolkit bug.
      # Opting CUB out is the minimal fix and costs nothing: CUB's ranges are
      # not what we are profiling.
      # ⛔ Do NOT instead put ${CUDA_HOME}/include/nvtx3 on the include path to
      # satisfy Tracing.h -- that does not avoid the clash (tried; same errors)
      # and additionally exposes the whole nvtx3 directory. The include is
      # corrected at source by patch_grid_clover_tracing.py.
      # Overridable so the need for it can be retested: `${VAR-default}`, not
      # `${VAR:-default}`, so an explicitly EMPTY value is honoured.
      # NOTE: with Tracing.h's include hoisted out of namespace Grid
      # (patch_grid_clover_tracing.py), this may no longer be required --
      # the namespace bug was the root cause of the CUB clash.
      tracing_cxxflags=${TRACING_EXTRA_CXXFLAGS-" -DCCCL_DISABLE_NVTX"}
    fi
  fi

  mkdir -p "${BUILD_DIR}" "${INSTALL_PREFIX}"
  (
    cd "${BUILD_DIR}"
    "${STAGE_SRC}/configure" \
      --prefix="${INSTALL_PREFIX}" \
      ${tracing_args[@]+"${tracing_args[@]}"} \
      --enable-comms=mpi \
      --enable-simd=GPU \
      --enable-shm=nvlink \
      --enable-gen-simd-width=64 \
      --enable-accelerator=cuda \
      --enable-setdevice \
      --disable-fermion-reps \
      --disable-unified \
      --disable-gparity \
      --with-mpfr="${GRID_MPFR_PREFIX}" \
      --with-lime="${CLIME_ROOT}" \
      CXX=nvcc \
      LDFLAGS="-cudart shared" \
      CXXFLAGS="-ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 -cudart shared${tracing_cxxflags}"
  )

  "${BUILD_DIR}/grid-config" --summary
}

build_grid() {
  [[ -x "${BUILD_DIR}/grid-config" ]] || {
    printf 'ERROR: Grid is not configured; run %s configure first\n' "$0" >&2
    exit 1
  }
  make -C "${BUILD_DIR}/Grid" -j"${JOBS}"
}

install_grid() {
  [[ -f "${BUILD_DIR}/Grid/libGrid.a" ]] || {
    printf 'ERROR: libGrid.a is absent; run %s build first\n' "$0" >&2
    exit 1
  }
  make -C "${BUILD_DIR}/Grid" install
  make -C "${BUILD_DIR}" install-binSCRIPTS

  test -x "${INSTALL_PREFIX}/bin/grid-config"
  test -f "${INSTALL_PREFIX}/lib/libGrid.a"
  "${INSTALL_PREFIX}/bin/grid-config" --summary
}

print_paths
printf 'GRID_TRACING=%s\n' "${GRID_TRACING:-off}"
case "${ACTION}" in
  prepare) prepare ;;
  patch-tracing) patch_tracing ;;
  configure) configure_grid ;;
  build) build_grid ;;
  install) install_grid ;;
  all)
    prepare
    patch_tracing
    configure_grid
    build_grid
    install_grid
    ;;
  info) ;;
  *)
    printf 'usage: %s {prepare|patch-tracing|configure|build|install|all|info}\n' "$0" >&2
    exit 2
    ;;
esac
