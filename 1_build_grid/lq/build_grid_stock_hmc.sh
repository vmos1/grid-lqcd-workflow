#!/usr/bin/env bash
# Stage, patch, configure, build and install STOCK upstream Grid for the pure-Grid HMC
# on lq (lq2_gpu, A100). The lq copy of ../build_grid_stock_hmc.sh (Perlmutter), kept
# separate so the Perlmutter script, which running campaigns use, stays untouched.
# Same actions, same patch set, same Eigen pin; differences from Perlmutter:
#   - environment: machines/lq.sh (one CUDA 12.2.1, OpenMPI 4.1.5 + UCX)
#   - nvcc host compiler: -ccbin mpicxx (Perlmutter: the Cray wrapper CC)
#   - MPFR: system /usr, found with no flag (lq.sh says why --with-mpfr=/usr is wrong)
#   - lq-only build fixes in lq/patches/, applied after the shared set, no physics change:
#       lq-01-perfcount-rdtsc-stubs-cuda.patch: stock PerfCount.h has the GRID_CUDA stubs for
#       __rdtsc/__rdpmc commented out, and nvcc -ccbin mpicxx then fails on every source file
#       ("identifier __rdtsc is undefined", seen 2026-05 on f8b2eacf and 2026-11-18 on
#       3d3eff86). The patch uncomments the two stubs (cyclecount() returns 0 on GPU builds).
#       Perlmutter's Cray wrapper does not hit this. The build folder keeps the name p08
#       (the shared physics patches); PROVENANCE.txt lists the lq fix with its sha256.
#   - paths: the workspace layout (__docs/2026_11_18_handoff_lq_port.md section 2)
#       source  builds/external/Grid            (stock clone, never edited)
#       stage   builds/grid-stock-<sha8>-p08_gpu/{Grid,Grid/build,install,eigen-3.4.0}
#       Eigen   builds/downloads/eigen-3.4.0.tar.bz2
#
# Actions:
#   prepare    clone GRID_SRC@GRID_SHA into the stage, vendor Eigen 3.4.0, filelist + autoreconf
#   patch      apply patches/stock-grid-3d3eff86/*.patch (sorted, idempotent)
#   configure  configure for lq A100
#   build      make libGrid     -- COMPUTE NODE ONLY (sbatch), JOBS default 32
#   install    make install     -- COMPUTE NODE ONLY
#   all        prepare + patch + configure   (login node; never builds)
#   info       print paths only
#
# Like the Perlmutter script this does not run Grid/bootstrap.sh (it deletes the
# downloaded Eigen archive); prepare() is its non-destructive equivalent.
# Every applied patch and the configure command are recorded in ${STAGE_ROOT}/PROVENANCE.txt.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKFLOW=$(cd "${HERE}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)

source "${WORKFLOW}/machines/lq.sh"

ACTION=${1:-all}
GRID_SRC=${GRID_SRC:-${ROOT}/builds/external/Grid}
# Pinned: builds/external/Grid follows upstream develop, so HEAD is not the build's commit.
GRID_SHA=${GRID_SHA:-3d3eff86f3664f4068af9814ae92150406bb812d}
JOBS=${JOBS:-32}
PATCH_DIR=${PATCH_DIR:-${WORKFLOW}/1_build_grid/patches/stock-grid-3d3eff86}
LQ_PATCH_DIR=${LQ_PATCH_DIR:-${HERE}/patches}

STAGE_ROOT=${STAGE_ROOT:-${ROOT}/builds/grid-stock-${GRID_SHA:0:8}-p08_gpu}
STAGE_SRC=${STAGE_SRC:-${STAGE_ROOT}/Grid}
BUILD_DIR=${BUILD_DIR:-${STAGE_SRC}/build}
INSTALL_PREFIX=${INSTALL_PREFIX:-${STAGE_ROOT}/install}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-${ROOT}/builds/downloads}
EIGEN_ARCHIVE=${DOWNLOAD_DIR}/eigen-3.4.0.tar.bz2
EIGEN_DIR=${STAGE_ROOT}/eigen-3.4.0
EIGEN_URL=https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
EIGEN_SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626
PREPARED_MARKER=${STAGE_SRC}/.grid-benchmark-prepared
PROVENANCE=${STAGE_ROOT}/PROVENANCE.txt

now() { date '+%Y-%m-%d %H:%M:%S %Z'; }

print_paths() {
  printf 'GRID_SRC=%s\n' "${GRID_SRC}"
  printf 'GRID_SHA=%s\n' "${GRID_SHA}"
  printf 'PATCH_DIR=%s\n' "${PATCH_DIR}"
  printf 'LQ_PATCH_DIR=%s\n' "${LQ_PATCH_DIR}"
  printf 'STAGE_SRC=%s\n' "${STAGE_SRC}"
  printf 'BUILD_DIR=%s\n' "${BUILD_DIR}"
  printf 'INSTALL_PREFIX=%s\n' "${INSTALL_PREFIX}"
  printf 'JOBS=%s\n' "${JOBS}"
}

# Write the PROVENANCE.txt header once; later actions append to it.
provenance_header() {
  [[ -f "${PROVENANCE}" ]] && return
  mkdir -p "${STAGE_ROOT}"
  {
    printf '# Provenance of a patched stock Grid tree (1_build_grid/lq/build_grid_stock_hmc.sh)\n'
    printf 'created:    %s on %s\n' "$(now)" "$(hostname)"
    printf 'workflow:   %s (%s%s)\n' "$(git -C "${WORKFLOW}" rev-parse --short HEAD)" \
      "$(git -C "${WORKFLOW}" rev-parse --abbrev-ref HEAD)" \
      "$([[ -n "$(git -C "${WORKFLOW}" status --porcelain --untracked-files=no)" ]] && echo ', dirty')"
    printf 'GRID_SHA:   %s\n' "${GRID_SHA}"
    printf 'GRID_SRC:   %s\n' "${GRID_SRC}"
    printf 'STAGE_SRC:  %s\n' "${STAGE_SRC}"
    printf 'PATCH_DIR:  %s\n' "${PATCH_DIR}"
  } > "${PROVENANCE}"
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
  provenance_header
  git clone --no-hardlinks --no-checkout "${GRID_SRC}" "${STAGE_SRC}"
  git -C "${STAGE_SRC}" checkout --detach "${GRID_SHA}"

  if [[ ! -f "${EIGEN_ARCHIVE}" ]]; then
    wget --no-check-certificate --output-document="${EIGEN_ARCHIVE}" "${EIGEN_URL}"
  fi
  printf '%s  %s\n' "${EIGEN_SHA256}" "${EIGEN_ARCHIVE}" | sha256sum --check

  tar -xjf "${EIGEN_ARCHIVE}" -C "${STAGE_ROOT}"
  ln -s "${EIGEN_DIR}/Eigen" "${STAGE_SRC}/Grid/Eigen"
  ln -s "${EIGEN_DIR}/unsupported/Eigen" "${STAGE_SRC}/Grid/Eigen/unsupported"

  # bootstrap.sh's non-destructive equivalent: generate Grid/Eigen.inc (the
  # automake list of vendored Eigen headers, followed through the symlinks)
  # before autoreconf, else automake fails with "cannot open < Grid/Eigen.inc".
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
  printf 'prepared:   %s (Eigen 3.4.0 sha256 %s)\n' "$(now)" "${EIGEN_SHA256}" >> "${PROVENANCE}"
}

# Apply each patch once, in sorted order.  A patch recorded in PROVENANCE.txt
# (name + sha256) is already in the tree and is skipped; any other patch that
# does not apply aborts the script (git apply --check prints why).
# The patches only edit existing headers, so filelist/autoreconf need no rerun.
apply_patches() {
  [[ -d "${STAGE_SRC}/.git" ]] || {
    printf 'ERROR: %s is not a staged git tree; run %s prepare first\n' "${STAGE_SRC}" "$0" >&2
    exit 1
  }
  local head
  head=$(git -C "${STAGE_SRC}" rev-parse HEAD)
  [[ "${head}" == "${GRID_SHA}" ]] || {
    printf 'ERROR: staged HEAD %s != GRID_SHA %s\n' "${head}" "${GRID_SHA}" >&2
    exit 1
  }

  local patches=()
  mapfile -t patches < <(find "${PATCH_DIR}" -maxdepth 1 -type f -name '*.patch' | LC_ALL=C sort)
  (( ${#patches[@]} > 0 )) || { printf 'ERROR: no *.patch in %s\n' "${PATCH_DIR}" >&2; exit 1; }
  # lq-only build fixes after the shared set (header comment).
  mapfile -t -O "${#patches[@]}" patches < <(find "${LQ_PATCH_DIR}" -maxdepth 1 -type f -name '*.patch' | LC_ALL=C sort)

  provenance_header
  local p name sum
  for p in "${patches[@]}"; do
    name=$(basename "${p}")
    sum=$(sha256sum "${p}" | cut -d' ' -f1)
    # "Already applied" = recorded in PROVENANCE.txt with this sha256. The Perlmutter script's
    # reverse-apply test fails on a re-run: 06 and 07 edit files 03 edits, so 03 no longer
    # reverses cleanly on the patched tree and the run aborts (seen 2026-11-18).
    if grep -qF "patch:      ${name}  sha256=${sum}" "${PROVENANCE}"; then
      printf 'already applied, skipping: %s\n' "${name}"
      continue
    fi
    git -C "${STAGE_SRC}" apply --check "${p}"
    git -C "${STAGE_SRC}" apply "${p}"
    printf 'patch:      %s  sha256=%s  applied %s\n' "${name}" "${sum}" "$(now)" >> "${PROVENANCE}"
    printf 'applied: %s\n' "${name}"
  done
  git -C "${STAGE_SRC}" status --short --untracked-files=no
}

configure_grid() {
  [[ -x "${STAGE_SRC}/configure" ]] || {
    printf 'ERROR: prepared configure script not found; run %s prepare first\n' "$0" >&2
    exit 1
  }

  # Perlmutter's flags (../build_grid_stock_hmc.sh) with the lq substitutions above.
  local cmd=(
    "${STAGE_SRC}/configure"
    --prefix="${INSTALL_PREFIX}"
    --enable-comms=mpi
    --enable-simd=GPU
    --enable-shm=nvlink
    --enable-gen-simd-width=64
    --enable-accelerator=cuda
    --enable-setdevice
    --disable-fermion-reps
    --disable-unified
    --disable-gparity
    --with-lime="${CLIME_ROOT}"
    CXX=nvcc
    LDFLAGS="-cudart shared"
    CXXFLAGS="-ccbin mpicxx -gencode arch=compute_80,code=${CUDA_ARCH} -std=c++17 -cudart shared"
  )

  provenance_header
  {
    printf 'configure:  %s  (in %s)\n  %s\n' "$(now)" "${BUILD_DIR}" "${cmd[*]@Q}"
    printf 'modules:    %s\n' "$(module -t list 2>&1 | tr '\n' ' ')"
    printf 'nvcc:       %s\n' "$(nvcc --version | tail -1)"
    printf 'mpicxx:     %s (%s)\n' "$(command -v mpicxx)" "$(mpicxx --version | head -1)"
  } >> "${PROVENANCE}"
  printf 'configure command:\n  %s\n' "${cmd[*]@Q}"

  mkdir -p "${BUILD_DIR}" "${INSTALL_PREFIX}"
  ( cd "${BUILD_DIR}" && "${cmd[@]}" )

  "${BUILD_DIR}/grid-config" --summary
}

# build/install compile with nvcc for a long time: never on the shared login node.
require_allocation() {
  if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_NO_ALLOC:-0}" != 1 ]]; then
    printf 'ERROR: "%s" must run inside a Slurm job (compute node); SLURM_JOB_ID is unset.\n' "${ACTION}" >&2
    printf '       Set ALLOW_NO_ALLOC=1 to override.\n' >&2
    exit 1
  fi
}

build_grid() {
  require_allocation
  [[ -x "${BUILD_DIR}/grid-config" ]] || {
    printf 'ERROR: Grid is not configured; run %s configure first\n' "$0" >&2
    exit 1
  }
  printf 'build:      %s on %s, make -j%s\n' "$(now)" "$(hostname)" "${JOBS}" >> "${PROVENANCE}"
  make -C "${BUILD_DIR}/Grid" -j"${JOBS}"
  printf 'built:      %s\n' "$(now)" >> "${PROVENANCE}"
}

install_grid() {
  require_allocation
  [[ -f "${BUILD_DIR}/Grid/libGrid.a" ]] || {
    printf 'ERROR: libGrid.a is absent; run %s build first\n' "$0" >&2
    exit 1
  }
  make -C "${BUILD_DIR}/Grid" install
  make -C "${BUILD_DIR}" install-binSCRIPTS

  test -x "${INSTALL_PREFIX}/bin/grid-config"
  test -f "${INSTALL_PREFIX}/lib/libGrid.a"
  printf 'installed:  %s  libGrid.a sha256=%s\n' "$(now)" \
    "$(sha256sum "${INSTALL_PREFIX}/lib/libGrid.a" | cut -d' ' -f1)" >> "${PROVENANCE}"
  "${INSTALL_PREFIX}/bin/grid-config" --summary
}

print_paths
case "${ACTION}" in
  prepare)   prepare ;;
  patch)     apply_patches ;;
  configure) configure_grid ;;
  build)     build_grid ;;
  install)   install_grid ;;
  all)
    prepare
    apply_patches
    configure_grid
    ;;
  info) ;;
  *)
    printf 'usage: %s {prepare|patch|configure|build|install|all|info}\n' "$0" >&2
    exit 2
    ;;
esac
