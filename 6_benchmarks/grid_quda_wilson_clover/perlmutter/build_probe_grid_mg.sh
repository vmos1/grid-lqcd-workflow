#!/usr/bin/env bash
# Compile the Phase 0 Grid-MG probe outside the git repository.
#
# Same stock-Grid resolution and $PSCRATCH build root as build_benchmark.sh, but
# with NO QUDA: the probe is deliberately Grid-only so a failure cannot be
# confused with QUDA session state. Everything QUDA-related in build_benchmark.sh
# (headers, -lquda, rpath, QUDA_SHA in the log) is therefore absent here.

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
BUILD_ROOT=${BUILD_ROOT:-${PSCRATCH}/grid_quda_wilson_clover/benchmark/${GRID_SHA}}
BIN=${BIN:-${BUILD_ROOT}/bin/probe_grid_mg_schur_clover}
SRC=${SRC:-${BENCH_DIR}/src/probe_grid_mg_schur_clover.cc}
LOG=${LOG:-${BUILD_ROOT}/build_probe_grid_mg.log}

[[ -x "${GRID_CONFIG}" ]] || {
  printf 'ERROR: stock Grid grid-config not found: %s\n' "${GRID_CONFIG}" >&2
  printf 'Build it first with: %s build\n' "${HERE}/build_grid_mainline.sh" >&2
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
CXXFLAGS="$(${GRID_CONFIG} --cxxflags) ${GRID_EXTRA_CXXFLAGS} -I${BENCH_DIR}/src"
LDFLAGS="$(${GRID_CONFIG} --ldflags) ${GRID_EXTRA_LDFLAGS}"
LIBS="$(${GRID_CONFIG} --libs)"

{
  printf 'DATE=%s\n' "$(date --iso-8601=seconds)"
  printf 'GRID_SHA=%s\n' "$(${GRID_CONFIG} --git)"
  printf 'GRID_CONFIG=%s\n' "${GRID_CONFIG}"
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
