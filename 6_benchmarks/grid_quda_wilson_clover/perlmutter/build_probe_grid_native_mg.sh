#!/usr/bin/env bash
# Build probe_grid_native_mg.cc -- our fork of Grid's OWN tests/solver/Test_wilson_mg.cc.
#
# The only structural difference from build_probe_grid_mg.sh is the extra include
# path to Grid's tests/solver, where Test_multigrid_common.h lives: Grid's
# purpose-built Wilson/clover multigrid framework is a TEST HEADER, not part of
# the installed API, so it cannot be picked up from the install prefix.

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
fi
BUILD_ROOT=${BUILD_ROOT:-${PSCRATCH}/grid_quda_wilson_clover/benchmark/${GRID_SHA}}
BIN=${BIN:-${BUILD_ROOT}/bin/probe_grid_native_mg}
SRC=${SRC:-${BENCH_DIR}/src/probe_grid_native_mg.cc}
LOG=${LOG:-${BUILD_ROOT}/build_probe_grid_native_mg.log}

# Test_multigrid_common.h is only in the SOURCE tree.
MG_COMMON_DIR=${MG_COMMON_DIR:-${STAGE_SRC}/tests/solver}
[[ -f "${MG_COMMON_DIR}/Test_multigrid_common.h" ]] || {
  MG_COMMON_DIR=${ROOT}/Grid/tests/solver
}
[[ -f "${MG_COMMON_DIR}/Test_multigrid_common.h" ]] || {
  printf 'ERROR: Test_multigrid_common.h not found (looked in %s)\n' "${MG_COMMON_DIR}" >&2
  exit 1
}

[[ -x "${GRID_CONFIG}" ]] || {
  printf 'ERROR: stock Grid grid-config not found: %s\n' "${GRID_CONFIG}" >&2
  exit 1
}

mkdir -p "$(dirname "${BIN}")" "${BUILD_ROOT}"

CXX=$(${GRID_CONFIG} --cxx)
GRID_EXTRA_CXXFLAGS=
GRID_EXTRA_LDFLAGS=
if [[ ! -f "${GRID_PREFIX}/include/Grid/Grid.h" ]]; then
  # ⛔ Must be the STAGED tree, not ${ROOT}/Grid: only the staged copy has been
  # bootstrapped, so the CFS clone lacks Grid/Eigen/Dense and the build dies with
  # "Grid/Eigen/Dense: No such file or directory".
  GRID_SOURCE=${GRID_SOURCE:-${STAGE_SRC}}
  GRID_BUILD=$(cd "$(dirname "${GRID_CONFIG}")" && pwd)
  GRID_EXTRA_CXXFLAGS="-I${GRID_SOURCE} -I${GRID_BUILD}/Grid"
  GRID_EXTRA_LDFLAGS="-L${GRID_BUILD}/Grid"
fi
CXXFLAGS="$(${GRID_CONFIG} --cxxflags) ${GRID_EXTRA_CXXFLAGS} -I${BENCH_DIR}/src -I${MG_COMMON_DIR}"
LDFLAGS="$(${GRID_CONFIG} --ldflags) ${GRID_EXTRA_LDFLAGS}"
LIBS="$(${GRID_CONFIG} --libs)"

{
  printf 'DATE=%s\n' "$(date --iso-8601=seconds)"
  printf 'GRID_SHA=%s\n' "$(${GRID_CONFIG} --git)"
  printf 'MG_COMMON_DIR=%s\n' "${MG_COMMON_DIR}"
  printf 'CXXFLAGS=%s\n' "${CXXFLAGS}"
  printf 'SRC=%s\n' "${SRC}"
  printf 'BIN=%s\n' "${BIN}"
} | tee "${LOG}"

# shellcheck disable=SC2086
${CXX} ${CXXFLAGS} -o "${BIN}" "${SRC}" ${LDFLAGS} ${LIBS} 2>&1 | tee -a "${LOG}"

printf 'Built %s\n' "${BIN}"
