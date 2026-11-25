#!/usr/bin/env bash
# mg_proto's own operator tests on lq3 (COMPUTE NODE ONLY; run by chroma_cpu_debug_job.sh).
#
# Question: is mg_proto's QPhiX clover operator the same operator as mg_proto's QDP++ reference?
# test/qphix/test_qphix_interface.cpp, cases TestQPhiXUnprecOp{Iso,Aniso}Constructor (double) and
# ...AnisoConstructorF (float): full op, M_ee, M_oo, M_ee_inv, M_eo, M_oe, op and dagger, on an 8^4
# random SU(3) field, against MG::QDPWilsonCloverLinearOperator. Context: in job 1365191 every mg_proto
# QPhiX solve converged by mg_proto's own measure to an operator that is not chroma's (rel resid ~1.5),
# while chroma's own QPhiX solver (uncompressed links) passed.
#
# mg_proto (the staged source, builds/chroma-cpu-lq3/src/mgproto) is built twice with the base build's
# flags (as in builds/chroma-cpu-lq3/build/mgproto-*/CMakeCache.txt), BUILD_TESTING=ON, against the
# base install's QDP++, QPhiX and PRIMME (installs/external/chroma-cpu-lq3, read only):
#   comp12    MG_QPHIX_COMPRESS12=TRUE   chromaform's setting, as installed
#   nocomp12  MG_QPHIX_COMPRESS12=FALSE  mg_proto's own default
# gtest is vendored (external/gtest): no download. The test builds link OpenBLAS dynamically (with
# BUILD_TESTING mg_proto also needs LAPACK, and FindLAPACK cannot link the static, gfortran-built
# libopenblas.a in its check); the operators under test do not use BLAS. Arguments as mg_proto's own
# add_test lines (one rank, one thread). Build trees: builds/mgproto-gtest-lq3/<flavour>-<jobid>/
# (never installed). Results: runs/<date>_mgproto_unit_<jobid>/<flavour>.log and SUMMARY.txt.
# Exit status: 0 if both flavours built and ran (whatever the verdicts), 1 otherwise.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKFLOW=$(cd "${HERE}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)
JOBS=${JOBS:-64}
SRC=${ROOT}/builds/chroma-cpu-lq3/src
BASE_PREFIX=${ROOT}/installs/external/chroma-cpu-lq3
BUILDTOOLS=${ROOT}/installs/external/buildtools-lq3
FILTER='QPhiXIntegration.TestQPhiXUnprecOp*'
TEST_ARGS=(-geom 1 1 1 1 -by 4 -bz 4 -pxy 0 -pxyz 0 -c 1 -sy 1 -sz 1 -minct 1)

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "runs only inside a Slurm job on lq3"

LQ_TARGET=cpu source "${WORKFLOW}/machines/lq.sh"
export PATH=${BUILDTOOLS}/bin:${PATH}
OPENBLAS=$(spack find --paths openblas | awk '/^openblas/ {print $2}')
[[ -f "${OPENBLAS}/lib/libopenblas.so" ]] || die "no spack openblas (OPENBLAS='${OPENBLAS}')"
export CC=${LQ_CPU_MPI_ROOT}/bin/mpicc CXX=${LQ_CPU_MPI_ROOT}/bin/mpicxx
xml_cflags=$(xml2-config --cflags)
export CFLAGS="-O3 -march=znver5 -fopenmp -std=gnu17 ${xml_cflags}"
export CXXFLAGS="-O3 -march=znver5 -fopenmp -std=c++14 ${xml_cflags} -I${BASE_PREFIX}/primme-cpu/include"
export LDFLAGS="-O3 -fopenmp -L${OPENBLAS}/lib -Wl,-rpath,${OPENBLAS}/lib"
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9
unset no_proxy NO_PROXY

RUN_DIR=${ROOT}/runs/$(date +%Y_%m_%d)_mgproto_unit_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"
SUMMARY=${RUN_DIR}/SUMMARY.txt
{
  printf '# mg_proto operator tests (%s), job %s on %s, %s\n' "${FILTER}" "${SLURM_JOB_ID}" "$(hostname)" "$(date)"
  printf 'mg_proto source: %s (%s)\n' "${SRC}/mgproto" "$(git -C "${SRC}/mgproto" rev-parse --short HEAD)"
  printf 'against: %s/{qdpxx-double-nd4,qphix-qdpxx-double-nd4-avx512,primme-cpu}\n' "${BASE_PREFIX}"
  printf 'CXX=%s (%s)\nCXXFLAGS=%s\n' "${CXX}" "$("${CXX}" --version | head -1)" "${CXXFLAGS}"
} > "${SUMMARY}"

status=0
for flavour in comp12 nocomp12; do
  case ${flavour} in comp12) comp=TRUE ;; nocomp12) comp=FALSE ;; esac
  bld=${ROOT}/builds/mgproto-gtest-lq3/${flavour}-${SLURM_JOB_ID}
  log=${RUN_DIR}/${flavour}.log
  printf -- '---- %s: MG_QPHIX_COMPRESS12=%s, build %s (%s)\n' "${flavour}" "${comp}" "${bld}" "$(date +%T)"
  mkdir -p "${bld}"
  set +e
  cmake -S "${SRC}/mgproto" -B "${bld}" \
      -DCMAKE_INSTALL_PREFIX="${bld}/never-installed" -DBUILD_TESTING=ON \
      -DMG_USE_KOKKOS=FALSE -DMG_USE_QPHIX=TRUE -DQPHIX_DIR="${BASE_PREFIX}/qphix-qdpxx-double-nd4-avx512" \
      -DMG_QPHIX_COMPRESS12="${comp}" -DMG_QPHIX_SOALEN=4 -DMG_USE_AVX512=FALSE \
      -DEigen3_DIR="${SRC}/eigen" -DKOKKOS_ENABLE_CUDA=FALSE -DQDPXX_DIR="${BASE_PREFIX}/qdpxx-double-nd4" \
      -DMG_DEFAULT_LOGLEVEL=DEBUG -DMG_KOKKOS_USE_NEIGHBOR_TABLE=FALSE \
      -DMG_USE_PRIMME=ON -DPRIMME_HOME="${BASE_PREFIX}/primme-cpu" \
      -DBLA_VENDOR=OpenBLAS -DBLA_STATIC=OFF \
      -DBLAS_openblas_LIBRARY="${OPENBLAS}/lib/libopenblas.so" \
      -DLAPACK_openblas_LIBRARY="${OPENBLAS}/lib/libopenblas.so" > "${log}" 2>&1 &&
    make -C "${bld}" -j "${JOBS}" test_qphix_interface >> "${log}" 2>&1
  rc=$?
  set -e
  exe=${bld}/test/qphix/test_qphix_interface
  if (( rc != 0 )) || [[ ! -x "${exe}" ]]; then
    printf '%s: BUILD FAILED (rc=%s), see %s\n' "${flavour}" "${rc}" "${log}" | tee -a "${SUMMARY}"
    status=1
    continue
  fi
  grep -q "^#define MG_QPHIX_COMPRESS12" "${bld}/MG_config.h" && got=on || got=off
  printf -- '---- %s: run (%s), compress12 in MG_config.h: %s\n' "${flavour}" "$(date +%T)" "${got}" | tee -a "${log}"
  set +e
  ( cd "$(dirname "${exe}")" && OMP_NUM_THREADS=1 timeout 900 srun --mpi=pmix --ntasks=1 --cpus-per-task=1 \
      "${exe}" "${TEST_ARGS[@]}" --gtest_filter="${FILTER}" ) >> "${log}" 2>&1
  rc=$?
  set -e
  {
    printf '%s (compress12 %s): test exit %s\n' "${flavour}" "${got}" "${rc}"
    grep -E '^\[ +(OK|FAILED|PASSED) +\]' "${log}" | sed 's/^/  /'
  } | tee -a "${SUMMARY}"
done
printf -- '---- mg_proto unit tests done (%s): %s\n' "$(date +%T)" "${SUMMARY}"
exit "${status}"
