#!/usr/bin/env bash
# Compile stock QUDA's dslash_test source without rebuilding libquda.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(cd "${HERE}/.." && pwd)
WORKFLOW=$(cd "${BENCH_DIR}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)

source "${WORKFLOW}/machines/perlmutter.sh"

: "${PSCRATCH:?PSCRATCH must be set on Perlmutter}"
QUDA_SOURCE=${QUDA_SOURCE:-${ROOT}/quda}
QUDA_BUILD=${QUDA_BUILD:-${QUDA_SOURCE}/build-mpi}
QUDA_PREFIX=${QUDA_PREFIX:-${ROOT}/quda-install-mpi}
QUDA_SHA=${QUDA_SHA:-$(git -C "${QUDA_SOURCE}" rev-parse HEAD)}
BUILD_ROOT=${BUILD_ROOT:-${PSCRATCH}/grid_quda_wilson_clover/quda-dslash/${QUDA_SHA}}
OBJ=${OBJ:-${BUILD_ROOT}/dslash_test.cpp.o}
BIN=${BIN:-${BUILD_ROOT}/bin/dslash_test}
LOG=${LOG:-${BUILD_ROOT}/build.log}

FLAGS_FILE=${QUDA_BUILD}/tests/CMakeFiles/dslash_test.dir/flags.make
TEST_LIB=${QUDA_BUILD}/tests/libquda_test.so
QUDA_LIB=${QUDA_PREFIX}/lib/libquda.so
SRC=${QUDA_SOURCE}/tests/dslash_test.cpp
CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
MATH_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9

for required in "${FLAGS_FILE}" "${TEST_LIB}" "${QUDA_LIB}" "${SRC}"; do
  [[ -f "${required}" ]] || {
    printf 'ERROR: required completed QUDA artifact not found: %s\n' "${required}" >&2
    exit 1
  }
done

# Keep this direct compile synchronized with the configured dslash_test target.
expected_defines='CXX_DEFINES = -DBUILD_MILC_INTERFACE -DBUILD_QDP_INTERFACE -DMPI_COMMS -DMULTI_GPU -DQUDA_PRECISION=14 -DQUDA_RECONSTRUCT=7'
grep -Fxq "${expected_defines}" "${FLAGS_FILE}" || {
  printf 'ERROR: %s no longer has the expected native-MPI precision/reconstruction definitions\n' "${FLAGS_FILE}" >&2
  exit 1
}
CXX=$(grep -m1 '^# compile CXX with ' "${FLAGS_FILE}" | cut -d ' ' -f 5-)
[[ -x "${CXX}" ]] || {
  printf 'ERROR: configured QUDA C++ compiler not executable: %s\n' "${CXX}" >&2
  exit 1
}

mkdir -p "$(dirname "${BIN}")" "${BUILD_ROOT}"

{
  printf 'DATE=%s\n' "$(date --iso-8601=seconds)"
  printf 'QUDA_SHA=%s\n' "${QUDA_SHA}"
  printf 'QUDA_SOURCE=%s\n' "${QUDA_SOURCE}"
  printf 'QUDA_BUILD=%s\n' "${QUDA_BUILD}"
  printf 'QUDA_PREFIX=%s\n' "${QUDA_PREFIX}"
  printf 'CXX=%s\n' "${CXX}"
  printf 'TEST_LIB=%s\n' "${TEST_LIB}"
  printf 'TEST_LIB_SHA256=%s\n' "$(sha256sum "${TEST_LIB}" | cut -d ' ' -f 1)"
  printf 'QUDA_LIB=%s\n' "${QUDA_LIB}"
  printf 'QUDA_LIB_SHA256=%s\n' "$(sha256sum "${QUDA_LIB}" | cut -d ' ' -f 1)"
  printf 'SRC=%s\n' "${SRC}"
  printf 'SRC_SHA256=%s\n' "$(sha256sum "${SRC}" | cut -d ' ' -f 1)"
  printf 'BIN=%s\n' "${BIN}"
} | tee "${LOG}"

"${CXX}" \
  -O3 -mtune=native -std=c++20 -w -Wno-unknown-pragmas \
  -DBUILD_MILC_INTERFACE -DBUILD_QDP_INTERFACE -DMPI_COMMS -DMULTI_GPU \
  -DQUDA_PRECISION=14 -DQUDA_RECONSTRUCT=7 \
  -I"${QUDA_BUILD}/tests" \
  -I"${QUDA_SOURCE}/tests" \
  -I"${QUDA_SOURCE}/tests/utils" \
  -I"${QUDA_SOURCE}/tests/host_reference" \
  -I"${QUDA_BUILD}/include/targets/cuda" \
  -I"${QUDA_SOURCE}/include" \
  -I"${QUDA_BUILD}/include" \
  -isystem "${QUDA_SOURCE}/tests/googletest/include" \
  -isystem "${QUDA_SOURCE}/tests/googletest" \
  -isystem "${QUDA_SOURCE}/include/externals" \
  -isystem "${QUDA_BUILD}/_deps/eigen-src" \
  -isystem "${CUDA_ROOT}/targets/x86_64-linux/include" \
  -c "${SRC}" -o "${OBJ}" 2>&1 | tee -a "${LOG}"

"${CXX}" -O3 -mtune=native "${OBJ}" -o "${BIN}" \
  -Wl,-rpath,"${QUDA_BUILD}/tests:${QUDA_PREFIX}/lib" \
  "${TEST_LIB}" "${QUDA_LIB}" \
  "${CUDA_ROOT}/targets/x86_64-linux/lib/stubs/libcuda.so" \
  "${CUDA_ROOT}/targets/x86_64-linux/lib/stubs/libnvidia-ml.so" \
  "${CUDA_ROOT}/targets/x86_64-linux/lib/libcudart_static.a" \
  -ldl /usr/lib64/librt.a \
  "${MATH_ROOT}/lib64/libcublas.so" \
  "${MATH_ROOT}/lib64/libcufft.so" 2>&1 | tee -a "${LOG}"

{
  printf '\n--- ldd ---\n'
  ldd "${BIN}"
  printf '\n--- CLI smoke ---\n'
  "${BIN}" --help >/dev/null
  printf 'dslash_test --help: PASS\n'
} | tee -a "${LOG}"

printf 'Built %s\n' "${BIN}"
