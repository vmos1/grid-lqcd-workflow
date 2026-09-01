#!/usr/bin/env bash
# Compile the shared Grid/QUDA benchmark outside the git repository.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(cd "${HERE}/.." && pwd)
WORKFLOW=$(cd "${BENCH_DIR}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)

source "${WORKFLOW}/machines/perlmutter.sh"

: "${PSCRATCH:?PSCRATCH must be set on Perlmutter}"
GRID_SHA=${GRID_SHA:-$(git -C "${ROOT}/Grid" rev-parse HEAD)}
STAGE_SRC=${STAGE_SRC:-${PSCRATCH}/grid_quda_wilson_clover/stock-grid/${GRID_SHA}/Grid}
GRID_PREFIX=${GRID_PREFIX:-${PSCRATCH}/grid_quda_wilson_clover/stock-grid/${GRID_SHA}/install}
if [[ -x "${GRID_PREFIX}/bin/grid-config" ]]; then
  GRID_CONFIG=${GRID_CONFIG:-${GRID_PREFIX}/bin/grid-config}
else
  GRID_CONFIG=${GRID_CONFIG:-${STAGE_SRC}/build/grid-config}
  GRID_SOURCE=${GRID_SOURCE:-${STAGE_SRC}}
fi
QUDA_PREFIX=${QUDA_PREFIX:-${ROOT}/quda-install-mpi}
QUDA_LIBDIR=${QUDA_LIBDIR:-lib}
BUILD_ROOT=${BUILD_ROOT:-${PSCRATCH}/grid_quda_wilson_clover/benchmark/${GRID_SHA}}
BIN=${BIN:-${BUILD_ROOT}/bin/benchmark_grid_quda_wilson_clover}
SRC=${SRC:-${BENCH_DIR}/src/benchmark_grid_quda_wilson_clover.cc}
LOG=${LOG:-${BUILD_ROOT}/build.log}

[[ -x "${GRID_CONFIG}" ]] || {
  printf 'ERROR: stock Grid grid-config not found: %s\n' "${GRID_CONFIG}" >&2
  printf 'Build it first with: %s build\n' "${HERE}/build_grid_mainline.sh" >&2
  exit 1
}
[[ -f "${QUDA_PREFIX}/include/quda.h" ]] || {
  printf 'ERROR: QUDA headers not found under %s\n' "${QUDA_PREFIX}" >&2
  exit 1
}
[[ -f "${QUDA_PREFIX}/${QUDA_LIBDIR}/libquda.so" ]] || {
  printf 'ERROR: libquda.so not found under %s/%s\n' "${QUDA_PREFIX}" "${QUDA_LIBDIR}" >&2
  exit 1
}

mkdir -p "$(dirname "${BIN}")" "${BUILD_ROOT}"

CXX=$(${GRID_CONFIG} --cxx)
GRID_EXTRA_CXXFLAGS=
GRID_EXTRA_LDFLAGS=
if [[ ! -f "${GRID_PREFIX}/include/Grid/Grid.h" ]]; then
  GRID_SOURCE=${GRID_SOURCE:-${ROOT}/Grid}
  GRID_BUILD=$(cd "$(dirname "${GRID_CONFIG}")" && pwd)
  [[ -f "${GRID_SOURCE}/Grid/Grid.h" && -f "${GRID_BUILD}/Grid/Config.h" ]] || {
    printf 'ERROR: Grid headers are neither installed nor available in an in-tree build\n' >&2
    exit 1
  }
  GRID_EXTRA_CXXFLAGS="-I${GRID_SOURCE} -I${GRID_BUILD}/Grid"
  GRID_EXTRA_LDFLAGS="-L${GRID_BUILD}/Grid"
fi
CXXFLAGS="$(${GRID_CONFIG} --cxxflags) ${GRID_EXTRA_CXXFLAGS} -DGRID_HAVE_QUDA -I${QUDA_PREFIX}/include -I${BENCH_DIR}/src"
LDFLAGS="$(${GRID_CONFIG} --ldflags) ${GRID_EXTRA_LDFLAGS} -L${QUDA_PREFIX}/${QUDA_LIBDIR} -Xlinker -rpath -Xlinker ${QUDA_PREFIX}/${QUDA_LIBDIR}"
LIBS="-lquda $(${GRID_CONFIG} --libs)"

{
  printf 'DATE=%s\n' "$(date --iso-8601=seconds)"
  printf 'GRID_SHA=%s\n' "$(${GRID_CONFIG} --git)"
  printf 'QUDA_SHA=%s\n' "$(git -C "${ROOT}/quda" rev-parse HEAD)"
  printf 'GRID_CONFIG=%s\n' "${GRID_CONFIG}"
  printf 'QUDA_PREFIX=%s\n' "${QUDA_PREFIX}"
  printf 'CXX=%s\n' "${CXX}"
  printf 'CXXFLAGS=%s\n' "${CXXFLAGS}"
  printf 'LDFLAGS=%s\n' "${LDFLAGS}"
  printf 'LIBS=%s\n' "${LIBS}"
  printf 'SRC=%s\n' "${SRC}"
  printf 'BIN=%s\n' "${BIN}"
} | tee "${LOG}"

# shellcheck disable=SC2086
${CXX} ${CXXFLAGS} -o "${BIN}" "${SRC}" ${LDFLAGS} ${LIBS} 2>&1 | tee -a "${LOG}"

{
  printf '\n--- ldd ---\n'
  ldd "${BIN}"
} | tee -a "${LOG}"

printf 'Built %s\n' "${BIN}"
