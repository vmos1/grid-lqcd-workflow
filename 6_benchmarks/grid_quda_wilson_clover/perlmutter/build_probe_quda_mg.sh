#!/usr/bin/env bash
# Compile the Phase 0 QUDA-MG probe outside the git repository.
#
# ⛔ PINNED TO THE MG-ENABLED QUDA. The campaign's usual QUDA_PREFIX,
# quda-install-mpi, has `/* #undef QUDA_MULTIGRID */` in quda_define.h -- every MG
# call against it fails. quda-install-mpi-mg is the same build with multigrid on;
# the two quda_define.h files differ in NOTHING but the MG flags (verified by diff
# 2026-09-11), so the CG path is unaffected and the in-run CG gate confirms it.
#
# The default is overridden here rather than left to the caller precisely because
# linking the wrong one produces a runtime failure deep inside MG setup rather than
# a link error.

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
QUDA_PREFIX=${QUDA_PREFIX:-${ROOT}/quda-install-mpi-mg}
QUDA_LIBDIR=${QUDA_LIBDIR:-lib}
BUILD_ROOT=${BUILD_ROOT:-${PSCRATCH}/grid_quda_wilson_clover/benchmark/${GRID_SHA}}
BIN=${BIN:-${BUILD_ROOT}/bin/probe_quda_mg_clover}
SRC=${SRC:-${BENCH_DIR}/src/probe_quda_mg_clover.cc}
LOG=${LOG:-${BUILD_ROOT}/build_probe_quda_mg.log}

[[ -x "${GRID_CONFIG}" ]] || {
  printf 'ERROR: stock Grid grid-config not found: %s\n' "${GRID_CONFIG}" >&2
  exit 1
}
[[ -f "${QUDA_PREFIX}/include/quda.h" ]] || {
  printf 'ERROR: QUDA headers not found under %s\n' "${QUDA_PREFIX}" >&2
  exit 1
}
# Fail loudly here rather than at MG setup time on a 4-node allocation.
if ! grep -q '^#define QUDA_MULTIGRID$' "${QUDA_PREFIX}/include/quda_define.h"; then
  printf 'ERROR: QUDA at %s was built WITHOUT multigrid (QUDA_MULTIGRID undefined).\n' "${QUDA_PREFIX}" >&2
  printf '       Use quda-install-mpi-mg.\n' >&2
  exit 1
fi

mkdir -p "$(dirname "${BIN}")" "${BUILD_ROOT}"

CXX=$(${GRID_CONFIG} --cxx)
GRID_EXTRA_CXXFLAGS=
GRID_EXTRA_LDFLAGS=
if [[ ! -f "${GRID_PREFIX}/include/Grid/Grid.h" ]]; then
  GRID_SOURCE=${GRID_SOURCE:-${ROOT}/Grid}
  GRID_BUILD=$(cd "$(dirname "${GRID_CONFIG}")" && pwd)
  GRID_EXTRA_CXXFLAGS="-I${GRID_SOURCE} -I${GRID_BUILD}/Grid"
  GRID_EXTRA_LDFLAGS="-L${GRID_BUILD}/Grid"
fi
CXXFLAGS="$(${GRID_CONFIG} --cxxflags) ${GRID_EXTRA_CXXFLAGS} -DGRID_HAVE_QUDA -I${QUDA_PREFIX}/include -I${BENCH_DIR}/src"
LDFLAGS="$(${GRID_CONFIG} --ldflags) ${GRID_EXTRA_LDFLAGS} -L${QUDA_PREFIX}/${QUDA_LIBDIR} -Xlinker -rpath -Xlinker ${QUDA_PREFIX}/${QUDA_LIBDIR}"
LIBS="-lquda $(${GRID_CONFIG} --libs)"

{
  printf 'DATE=%s\n' "$(date --iso-8601=seconds)"
  printf 'GRID_SHA=%s\n' "$(${GRID_CONFIG} --git)"
  printf 'QUDA_PREFIX=%s (MULTIGRID enabled)\n' "${QUDA_PREFIX}"
  printf 'CXX=%s\n' "${CXX}"
  printf 'CXXFLAGS=%s\n' "${CXXFLAGS}"
  printf 'SRC=%s\n' "${SRC}"
  printf 'BIN=%s\n' "${BIN}"
} | tee "${LOG}"

# shellcheck disable=SC2086
${CXX} ${CXXFLAGS} -o "${BIN}" "${SRC}" ${LDFLAGS} ${LIBS} 2>&1 | tee -a "${LOG}"

{
  printf '\n--- ldd ---\n'
  ldd "${BIN}" | grep -E "libquda|libGrid" || true
} | tee -a "${LOG}"

printf 'Built %s\n' "${BIN}"
