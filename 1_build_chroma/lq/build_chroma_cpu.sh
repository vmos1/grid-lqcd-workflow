#!/usr/bin/env bash
# CPU Chroma for lq3 (2 x 32-core EPYC 9355, Zen 5, AVX-512, EL9): QMP + QDP++ + QPhiX + mg_proto
# + PRIMME + Chroma, with Chroma's CPU multigrid (mg_proto on QPhiX), built offline by JLab's
# chromaform.
#
# Recipe: the group's chromaform (/lustre2/nplqcd/chroma/build_scripts/chromaform, commit
# 2f9428dc, 2025-06-09) run as in the group's EL8 CPU recipe build_scripts/lq2_cpu.sh
# (--cpu --mg), with these changes:
#   toolchain  lq3's, from machines/lq.sh LQ_TARGET=cpu (spack gcc 15.3, Grid-CPU's OpenMPI 4.1.8),
#              the one our Grid CPU build uses. The EL8 recipe used gcc 12.3 + gompi OpenMPI 4.1.5.
#   ISA        --avx512 (QPhiX isa avx512) and -march=znver5 by hand (--autoflags=no). chromaform's
#              autodetection maps an EPYC 9355 to zen2 (-march=znver2, AVX2); the EL8 recipe's --zen3
#              means QPhiX avx2; --avx512 with autoflags would add -march=skylake-avx512.
#   flags      the flags autoflags would add, spelled out: -O3, -fopenmp, C++ -std=c++14 (chromaform's
#              CPU default; the group's 2024 qdpxx build used it; CXXSTD overrides). C gets -std=gnu17,
#              not chromaform's "-std=c11 -D_XOPEN_SOURCE=500": that hides POSIX-2001 declarations
#              (posix_memalign, ...) and GCC 14+ makes an implicit declaration an error. gnu17 is the
#              default of GCC 8-14; GCC 15 defaults to gnu23, which old C code fails on (bool, ()).
#   BLAS       spack's OpenBLAS 0.3.34 (x86_64_v4): the static library for mg_proto's FindBLAS
#              (chromaform forces BLA_STATIC=ON), the shared one with an rpath for chroma. Without it
#              chromaform would download and build OpenBLAS 0.3.28.
#   offline    "prepare" stages every source in src/ by copying (no download); chromaform skips the
#              download of any package whose src/<pkg> exists. In the build the network is fenced off
#              (dead proxy, git file:// only, pip with no index): a slip fails, it never downloads.
#   jinja2     QPhiX's code generator imports jinja2; chromaform makes a venv and runs
#              "pip install jinja2" from PyPI. Here pip has no index, and jinja2 + MarkupSafe, copied
#              from the user's ~/.venvs/jupyter (uv CPython 3.12, also the PYTHON used), come in
#              through PYTHONPATH.
#   make       MAKEFLAGS=-k. chromaform retries a failed make three times at -j1 (meant for small
#              login nodes); with -k the first pass builds all it can and lists every failing file, so
#              the retries only redo the failed targets instead of serially compiling the rest.
#   tools      lq3 has no cmake, autoconf/automake or libxml2 headers (preflight, job 1364670), so they
#              come from installs/external/buildtools-lq3 (build_buildtools.sh: cmake 3.31.8, autoconf
#              2.72, automake 1.16.5, libtool 2.4.7, static libxml2 2.11.9), first on PATH. libxml2's
#              flags go into CFLAGS/CXXFLAGS/LIBS from its xml2-config, which also stops chromaform's
#              fallback guess (-I/usr/include/libxml2 -lxml2) for a "system" libxml2.
#
# Sources (group trees are read-only: copied, never built in place, since chromaform runs
# autoreconf inside the source tree):
#   qmp qdpxx qphix mgproto primme  /lustre2/nplqcd/chroma/build_scripts/src: chromaform's devel
#                                   clones of 2024-03, from the group's EL8 CPU attempt, which
#                                   installed mg_proto but no chroma (its chroma folders are empty).
#   chroma                          /lustre2/nplqcd/chroma/source/chroma, devel a761e7e (2025-04-24):
#                                   the commit of the group's GPU production build install_oct_2025
#                                   (whose own clone, source_jul_2025/chroma, has a broken .git). Its
#                                   mg_proto/ and qphix/ inverters and configure.ac are unchanged since
#                                   the 2024-02 chroma that came with these mg_proto and QPhiX clones.
#   eigen                           builds/downloads/eigen-3.4.0.tar.bz2 (sha256 checked; chromaform's URL)
#
# Layout (__docs/2026_11_18_handoff_lq_port.md section 2):
#   builds/chroma-cpu-lq3/            src/ build/ pylib/ chromaform PROVENANCE.txt   (recreatable)
#   installs/external/chroma-cpu-lq3/ one folder per package (chromaform's names), bin -> the chroma
#                                     package's bin/, PROVENANCE.txt
# chromaform itself runs "rm -rf" on a package's build and install folder before building it, and on
# a failed package's install folder when it exits; both lie under the two folders above.
#
# Actions:
#   prepare  login node: copy sources, chromaform, jinja2; record commits (no toolchain, no network)
#   check    lq3 job: toolchain, cmake, autotools, xml2-config, python + jinja2, OpenBLAS, sources
#   build    lq3 job: chromaform (COMPUTE NODE ONLY); never overwrites an installed chroma
#   test     lq3 job: ldd, then two multi-rank weak-field clover propagators, QPhiX fine level + mg_proto
#            coarse levels, inputs in tests/ (COMPUTE NODE ONLY)
#   info     print paths
# The job wrapper is build_chroma_cpu_job.sh beside this file.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKFLOW=$(cd "${HERE}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)
SELF=${HERE}/$(basename "${BASH_SOURCE[0]}")

ACTION=${1:-info}
JOBS=${JOBS:-64}
CXXSTD=${CXXSTD:-c++14}
# Variants (2026-11-24, job 1365191: the base build's mg_proto QPhiX solves converge to an operator that
# is not chroma's, rel resid ~1.5, while chroma's own uncompressed QPhiX solver passes). VARIANT unset is
# the base build; a variant has its own build stage and prefix, never touching the base, and takes the
# sources, the chromaform copy and jinja2 from the base stage (BASE_STAGE):
#   nocomp12  mg_proto without MG_QPHIX_COMPRESS12 (mg_proto's own default; chromaform forces it on, and
#             chroma's own QPhiX solver, which passed, is built uncompressed). Rebuilds mg_proto + chroma.
#   o2sky     mg_proto + chroma at -O2 -march=skylake-avx512 (chromaform's own --avx512 flags), compression
#             on. Rebuilds mg_proto + chroma.
#   gcc11     the whole stack with EL9's system gcc 11.5 (/usr/bin/gcc, through the OpenMPI wrappers via
#             OMPI_CC/OMPI_CXX) at -march=skylake-avx512 (gcc 11 has no znver5), compression on.
# REUSE lists base packages a variant links instead of rebuilding (chromaform skips an installed package).
VARIANT=${VARIANT:-}
BASE_STAGE=${ROOT}/builds/chroma-cpu-lq3
BASE_PREFIX=${ROOT}/installs/external/chroma-cpu-lq3
OPT=-O3
ARCH=-march=znver5
EXTRA_CMAKE=""
REUSE=()
COMPILER=gcc15
case ${VARIANT} in
  "") ;;
  nocomp12) EXTRA_CMAKE="-DMG_QPHIX_COMPRESS12=FALSE"
            REUSE=(qmp qdpxx-double-nd3 qdpxx-double-nd4 qphix-qdpxx-double-nd4-avx512 primme-cpu) ;;
  o2sky)    OPT=-O2; ARCH=-march=skylake-avx512
            REUSE=(qmp qdpxx-double-nd3 qdpxx-double-nd4 qphix-qdpxx-double-nd4-avx512 primme-cpu) ;;
  gcc11)    ARCH=-march=skylake-avx512; COMPILER=gcc11 ;;
  *) printf 'ERROR: unknown VARIANT=%s (nocomp12, o2sky, gcc11)\n' "${VARIANT}" >&2; exit 2 ;;
esac
STAGE=${STAGE:-${BASE_STAGE}${VARIANT:+-${VARIANT}}}
SRC=${BASE_STAGE}/src
PYLIB=${BASE_STAGE}/pylib
CHROMAFORM=${BASE_STAGE}/chromaform
BUILD=${STAGE}/build
PROVENANCE=${STAGE}/PROVENANCE.txt
PREFIX=${PREFIX:-${BASE_PREFIX}${VARIANT:+-${VARIANT}}}
BUILDTOOLS=${BUILDTOOLS:-${ROOT}/installs/external/buildtools-lq3}
# Test cases to run (default all; see test_chroma), e.g. CASES="mgproto_ref_4r".
CASES=${CASES:-all}
# chromaform's install name for --cpu --mg --avx512 (double precision, Nd=4)
CHROMA_NAME=chroma-mgproto-qphix-qdpxx-double-nd4-avx512
BIN=${PREFIX}/${CHROMA_NAME}/bin/chroma

GROUP=/lustre2/nplqcd/chroma
GROUP_CHROMAFORM=${GROUP}/build_scripts/chromaform/chromaform
GROUP_CHROMAFORM_SHA=2f9428dc736077bdd40df6314b53392e94c38bcb
JINJA_SITE=${JINJA_SITE:-${HOME}/.venvs/jupyter/lib/python3.12/site-packages}
PYTHON=${PYTHON:-${HOME}/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}
EIGEN_ARCHIVE=${ROOT}/builds/downloads/eigen-3.4.0.tar.bz2
EIGEN_SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626

# One edit to the staged chromaform copy (never the group's), approved by the user 2026-11-23: upstream
# chromaform f8997ce (2025-08-25, "compile primme with c++ for cpu, that avoids issues with tgmath.h";
# in Perlmutter's 5f55055, not in our 2f9428d) builds PRIMME on CPU with the C++ compiler and flags.
# Upstream's line also switches the make target to install_static, which our PRIMME (eloy/primme-3.3
# f693af4) does not have, so the target stays "install" (chromaform keeps only libprimme.a afterwards).
PRIMME_CPU_OLD='run $MYMAKE -C ${bdir} lib install PREFIX=${idir} -j $MAKE_JN CFLAGS="$CFLAGS -fPIC" ;;'
PRIMME_CPU_NEW='run $MYMAKE -C ${bdir} lib install PREFIX=${idir} -j $MAKE_JN CC="$CXX" CFLAGS="$CXXFLAGS -fPIC" ;;'

PKGS=(qmp qdpxx qphix mgproto primme chroma)
declare -A FROM=(
  [qmp]=${GROUP}/build_scripts/src/qmp
  [qdpxx]=${GROUP}/build_scripts/src/qdpxx
  [qphix]=${GROUP}/build_scripts/src/qphix
  [mgproto]=${GROUP}/build_scripts/src/mgproto
  [primme]=${GROUP}/build_scripts/src/primme
  [chroma]=${GROUP}/source/chroma
)
declare -A PIN=(
  [qmp]=c7444a2990802138e50d6d7fcc0cadefcb85d9cc       # devel, tag qmp2-5-3, 2021-06-10
  [qdpxx]=7a4bd2c2f021b0c7e3a15d48b917d75cceaff3eb     # devel, 2023-03-03
  [qphix]=04f5ee7d7a3d79396a12b4b2d8a6bed9cda1dba0     # devel, 2024-03-20
  [mgproto]=6ec4ae79b0186238c54c8493c775bc28864449c6   # devel, 2023-12-24
  [primme]=f693af42f458805359e7bf1a7064d51b81033687    # eloy/primme-3.3, 2023-05-12
  [chroma]=a761e7e03e50bbec74d43a8a3c096e0d04fcb8e5    # devel, 2025-04-24
)

now() { date '+%Y-%m-%d %H:%M:%S %Z'; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
# The group trees belong to another user: git refuses them without a per-call safe.directory.
gitg() { git -c safe.directory="$1" -C "$1" "${@:2}"; }

print_paths() {
  printf 'VARIANT=%s\nSTAGE=%s\nPREFIX=%s\nBIN=%s\nBUILDTOOLS=%s\nCHROMAFORM=%s\nPYTHON=%s\nJINJA_SITE=%s\n' \
    "${VARIANT:-base}" "${STAGE}" "${PREFIX}" "${BIN}" "${BUILDTOOLS}" "${CHROMAFORM}" "${PYTHON}" "${JINJA_SITE}"
  printf 'JOBS=%s CXXSTD=%s OPT=%s ARCH=%s COMPILER=%s EXTRA_CMAKE=%s REUSE=%s CASES=%s\n' "${JOBS}" "${CXXSTD}" \
    "${OPT}" "${ARCH}" "${COMPILER}" "${EXTRA_CMAKE:-none}" "${REUSE[*]:-none}" "${CASES}"
}

provenance_header() {
  [[ -f "${PROVENANCE}" ]] && return
  mkdir -p "${STAGE}"
  {
    printf '# Provenance of the lq3 CPU Chroma build (1_build_chroma/lq/build_chroma_cpu.sh)\n'
    printf 'created:    %s on %s\n' "$(now)" "$(hostname)"
    printf 'workflow:   %s (%s%s)\n' "$(git -C "${WORKFLOW}" rev-parse --short HEAD)" \
      "$(git -C "${WORKFLOW}" rev-parse --abbrev-ref HEAD)" \
      "$([[ -n "$(git -C "${WORKFLOW}" status --porcelain --untracked-files=no)" ]] && echo ', dirty')"
    printf 'script:     %s  sha256=%s\n' "${SELF}" "$(sha256sum "${SELF}" | cut -d' ' -f1)"
    printf 'STAGE:      %s\nPREFIX:     %s\n' "${STAGE}" "${PREFIX}"
    if [[ -n "${VARIANT}" ]]; then
      printf 'variant:    %s (OPT=%s ARCH=%s COMPILER=%s EXTRA_CMAKE=%s REUSE=%s)\n' "${VARIANT}" "${OPT}" "${ARCH}" \
        "${COMPILER}" "${EXTRA_CMAKE:-none}" "${REUSE[*]:-none}"
      printf 'sources:    as staged in %s (source commits, chromaform edit and jinja2 recorded in its PROVENANCE.txt)\n' \
        "${BASE_STAGE}"
    fi
  } > "${PROVENANCE}"
}

# The EL9 toolchain (lq.sh refuses on anything but an EL9 lq3 node), spack's OpenBLAS, the MPI wrappers,
# and our build tools first on PATH (cmake, autotools, xml2-config). Our aclocal searches its own
# prefix's share/aclocal, where libtool's macros are, so ACLOCAL_PATH needs nothing.
load_env() {
  LQ_TARGET=cpu source "${WORKFLOW}/machines/lq.sh"
  export PATH=${BUILDTOOLS}/bin:${PATH}
  export PKG_CONFIG_PATH=${BUILDTOOLS}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}
  OPENBLAS=$(spack find --paths openblas | awk '/^openblas/ {print $2}')
  MPICC=${LQ_CPU_MPI_ROOT}/bin/mpicc
  MPICXX=${LQ_CPU_MPI_ROOT}/bin/mpicxx
  # gcc11: the same OpenMPI wrappers, driving EL9's system compiler (Open MPI's OMPI_CC/OMPI_CXX).
  if [[ "${COMPILER}" == gcc11 ]]; then
    export OMPI_CC=/usr/bin/gcc OMPI_CXX=/usr/bin/g++
  fi
}

require_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "\"${ACTION}\" runs only inside a Slurm job on lq3 (SLURM_JOB_ID is unset)"
}

# Copy one group git tree into src/<pkg>, at its pinned commit. A marker file says the copy finished.
stage_repo() {
  local pkg=$1 from=${FROM[$1]} head mods
  if [[ -f "${SRC}/.staged-${pkg}" ]]; then
    head=$(git -C "${SRC}/${pkg}" rev-parse HEAD)
    [[ "${head}" == "${PIN[$pkg]}" ]] || die "${SRC}/${pkg} is at ${head}, pinned ${PIN[$pkg]}"
    printf 'already staged: %-8s %s\n' "${pkg}" "${head:0:8}"
    return
  fi
  if [[ -e "${SRC}/${pkg}" ]]; then
    die "incomplete copy ${SRC}/${pkg} (no .staged-${pkg}). Inspect it; to discard it yourself: rm -rf ${SRC}/${pkg}"
  fi
  head=$(gitg "${from}" rev-parse HEAD)
  [[ "${head}" == "${PIN[$pkg]}" ]] || die "${from} is at ${head}, pinned ${PIN[$pkg]}"
  printf 'copying %s (%s) ...\n' "${from}" "${head:0:8}"
  cp -R --preserve=mode,timestamps "${from}" "${SRC}/${pkg}"
  head=$(git -C "${SRC}/${pkg}" rev-parse HEAD)
  [[ "${head}" == "${PIN[$pkg]}" ]] || die "the copy ${SRC}/${pkg} reads ${head}"
  # Slow on Lustre (the copy's index is stale, so git re-reads every file), once per tree.
  mods=$(git -C "${SRC}/${pkg}" status --porcelain --untracked-files=no | wc -l)
  {
    printf 'source:     %-8s %s  %s  from %s\n' "${pkg}" "${head}" \
      "$(git -C "${SRC}/${pkg}" log -1 --format='%ad%d' --date=short)" "${from}"
    git -C "${SRC}/${pkg}" submodule status --recursive | sed 's/^/  submodule/'
    if (( mods > 0 )); then
      printf '  WARNING: %s tracked modifications in the group tree, kept:\n' "${mods}"
      git -C "${SRC}/${pkg}" diff --stat | sed 's/^/    /'
    fi
  } >> "${PROVENANCE}"
  (( mods == 0 )) || printf 'WARNING: %s has %s tracked modifications (kept; see PROVENANCE.txt)\n' "${pkg}" "${mods}"
  : > "${SRC}/.staged-${pkg}"
}

# Apply the PRIMME edit (header) to the staged copy once; record the diff against the group's file.
patch_chromaform() {
  if grep -qF -- "${PRIMME_CPU_NEW}" "${CHROMAFORM}"; then
    printf 'already patched: chromaform (PRIMME built as C++, f8997ce)\n'
    return
  fi
  [[ "$(grep -cF -- "${PRIMME_CPU_OLD}" "${CHROMAFORM}")" == 1 ]] ||
    die "${CHROMAFORM}: the PRIMME cpu line is not there exactly once; not patching"
  local content
  content=$(< "${CHROMAFORM}")
  printf '%s\n' "${content/"${PRIMME_CPU_OLD}"/"${PRIMME_CPU_NEW}"}" > "${CHROMAFORM}"
  grep -qF -- "${PRIMME_CPU_NEW}" "${CHROMAFORM}" || die "PRIMME edit did not apply"
  {
    printf 'patch:      %s  PRIMME cpu build with the C++ compiler and flags, as upstream chromaform f8997ce\n' "$(now)"
    printf '            (2025-08-25, "compile primme with c++ for cpu, that avoids issues with tgmath.h"),\n'
    printf '            target kept "install" (our PRIMME f693af4 has no install_static). User-approved 2026-11-23.\n'
    printf '            chromaform sha256 now %s; diff against %s:\n' \
      "$(sha256sum "${CHROMAFORM}" | cut -d' ' -f1)" "${GROUP_CHROMAFORM}"
    diff "${GROUP_CHROMAFORM}" "${CHROMAFORM}" | sed 's/^/              /' || true
  } >> "${PROVENANCE}"
  printf 'patched: chromaform (PRIMME built as C++, f8997ce)\n'
}

prepare() {
  [[ -z "${VARIANT}" ]] || die "prepare stages the base sources only; variants use ${BASE_STAGE}"
  mkdir -p "${SRC}" "${PYLIB}"
  provenance_header
  local pkg head d
  for pkg in "${PKGS[@]}"; do stage_repo "${pkg}"; done

  # Eigen: chromaform's src/eigen is the top folder of eigen-3.4.0.tar.bz2.
  if [[ -f "${SRC}/.staged-eigen" ]]; then
    printf 'already staged: eigen\n'
  else
    [[ -e "${SRC}/eigen" ]] && die "incomplete ${SRC}/eigen. To discard it yourself: rm -rf ${SRC}/eigen"
    printf '%s  %s\n' "${EIGEN_SHA256}" "${EIGEN_ARCHIVE}" | sha256sum --check
    mkdir -p "${SRC}/eigen"
    tar -xjf "${EIGEN_ARCHIVE}" -C "${SRC}/eigen" --strip-components=1
    printf 'source:     eigen    3.4.0  %s  sha256=%s\n' "${EIGEN_ARCHIVE}" "${EIGEN_SHA256}" >> "${PROVENANCE}"
    : > "${SRC}/.staged-eigen"
  fi

  # chromaform: a copy, so the build does not follow later edits of the group's checkout.
  if [[ -x "${CHROMAFORM}" ]]; then
    printf 'already staged: chromaform\n'
  else
    head=$(gitg "$(dirname "${GROUP_CHROMAFORM}")" rev-parse HEAD)
    [[ "${head}" == "${GROUP_CHROMAFORM_SHA}" ]] || die "group chromaform is at ${head}, pinned ${GROUP_CHROMAFORM_SHA}"
    gitg "$(dirname "${GROUP_CHROMAFORM}")" diff --quiet HEAD -- chromaform ||
      die "the group's chromaform differs from its commit ${head}"
    cp --preserve=mode,timestamps "${GROUP_CHROMAFORM}" "${CHROMAFORM}"
    printf 'chromaform: %s  commit %s  sha256=%s\n' "${GROUP_CHROMAFORM}" "${head}" \
      "$(sha256sum "${CHROMAFORM}" | cut -d' ' -f1)" >> "${PROVENANCE}"
  fi
  patch_chromaform

  # jinja2 + MarkupSafe for QPhiX's code generator, with their dist-info so pip sees them installed.
  if [[ -f "${PYLIB}/.staged" ]]; then
    printf 'already staged: pylib\n'
  else
    [[ -x "${PYTHON}" ]] || die "PYTHON=${PYTHON} is not an executable"
    for d in jinja2 markupsafe; do
      [[ -d "${JINJA_SITE}/${d}" ]] || die "no ${d} in ${JINJA_SITE}"
      cp -R --preserve=mode,timestamps "${JINJA_SITE}/${d}" "${PYLIB}/"
    done
    cp -R --preserve=mode,timestamps "${JINJA_SITE}"/jinja2-*.dist-info "${JINJA_SITE}"/markupsafe-*.dist-info "${PYLIB}/"
    PYTHONPATH="${PYLIB}" "${PYTHON}" -c 'import jinja2, markupsafe; print("jinja2", jinja2.__version__, jinja2.__file__)'
    printf 'pylib:      %s from %s, for %s (%s)\n' "$(cd "${PYLIB}" && ls -d *.dist-info | tr '\n' ' ')" \
      "${JINJA_SITE}" "${PYTHON}" "$("${PYTHON}" --version 2>&1)" >> "${PROVENANCE}"
    : > "${PYLIB}/.staged"
  fi
  printf 'prepared:   %s\n' "$(now)" >> "${PROVENANCE}"
  printf 'Staged in %s. Next: the lq3 job build_chroma_cpu_job.sh (check, build, test).\n' "${STAGE}"
}

# Everything the build needs on the node, checked before chromaform runs (it would otherwise try to
# download what is missing). Exits non-zero, having built nothing, if anything is absent.
check() {
  load_env
  local bad=0 v t pkg
  ok()   { printf '  ok    %-12s %s\n' "$1" "$2"; }
  fail() { printf '  FAIL  %-12s %s\n' "$1" "$2"; bad=1; }
  printf -- '---- preflight on %s (%s), %s\n' "$(hostname)" "$(. /etc/os-release && echo "${PRETTY_NAME}")" "$(now)"
  printf '  cpu   %s, %s logical CPUs, avx512f %s\n' "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)" \
    "$(nproc)" "$(grep -q -m1 avx512f /proc/cpuinfo && echo yes || echo no)"
  [[ -x "${MPICC}" && -x "${MPICXX}" ]] && ok mpicxx "${MPICXX} ($("${MPICXX}" --version | head -1))" || fail mpicxx "${MPICXX} missing"
  if [[ "${COMPILER}" == gcc11 ]]; then
    v=$(/usr/bin/g++ -dumpversion 2>/dev/null || true)
    [[ "${v}" == 11* ]] && ok "system g++" "/usr/bin/g++ ${v}; mpicxx --showme: $("${MPICXX}" --showme 2>&1 | head -1)" ||
      fail "system g++" "/usr/bin/g++ is '${v:-missing}', variant gcc11 needs 11.x"
  fi
  local p
  for p in "${REUSE[@]}"; do
    [[ -d "${BASE_PREFIX}/${p}" ]] && ok "reuse" "${BASE_PREFIX}/${p}" || fail "reuse" "${BASE_PREFIX}/${p} missing"
  done
  [[ -f "${BUILDTOOLS}/PROVENANCE.txt" ]] && ok buildtools "${BUILDTOOLS}" ||
    fail buildtools "${BUILDTOOLS} not built/tested: run build_buildtools_job.sh first"
  if command -v cmake >/dev/null 2>&1; then
    v=$(cmake --version | awk 'NR==1 {print $3}')
    case ${v} in
      3.1[89].*|3.[2-9][0-9].*) ok cmake "${v} ($(command -v cmake))" ;;
      *) fail cmake "${v} ($(command -v cmake)): needs 3.18 <= cmake < 4 (chromaform's test; cmake 4 rejects QPhiX's and mg_proto's cmake_minimum_required 3.1)" ;;
    esac
  else
    fail cmake "none on PATH (lq3 has none; expected ${BUILDTOOLS}/bin/cmake)"
  fi
  for t in autoreconf autoconf automake aclocal m4; do
    if command -v "${t}" >/dev/null 2>&1; then
      ok "${t}" "$("${t}" --version 2>&1 | head -1)"
    else
      fail "${t}" "none on PATH (chromaform runs autoreconf for qmp, qdpxx and chroma; expected in ${BUILDTOOLS}/bin)"
    fi
  done
  if command -v xml2-config >/dev/null 2>&1; then
    ok libxml2 "$(xml2-config --version) ($(command -v xml2-config)), $(xml2-config --cflags)"
  else
    fail libxml2 "no xml2-config (QDP++'s configure needs it; expected ${BUILDTOOLS}/bin/xml2-config)"
  fi
  if PYTHONPATH="${PYLIB}" "${PYTHON}" -c 'import jinja2, markupsafe, venv, ensurepip' 2>/dev/null; then
    ok python "$("${PYTHON}" --version 2>&1), jinja2 $(PYTHONPATH="${PYLIB}" "${PYTHON}" -c 'import jinja2; print(jinja2.__version__)')"
  else
    fail python "${PYTHON} cannot import jinja2/markupsafe/venv/ensurepip with PYTHONPATH=${PYLIB}"
  fi
  [[ -f "${OPENBLAS}/lib/libopenblas.a" && -f "${OPENBLAS}/lib/libopenblas.so" ]] &&
    ok openblas "${OPENBLAS}" || fail openblas "spack openblas not found (OPENBLAS='${OPENBLAS}')"
  for pkg in "${PKGS[@]}" eigen; do
    [[ -f "${SRC}/.staged-${pkg}" ]] && ok "src/${pkg}" "staged" || fail "src/${pkg}" "not staged: run prepare on the login node"
  done
  [[ -x "${CHROMAFORM}" && -f "${PYLIB}/.staged" ]] && ok chromaform "${CHROMAFORM}" || fail chromaform "chromaform or pylib not staged"
  grep -qF -- "${PRIMME_CPU_NEW}" "${CHROMAFORM}" && ok "primme edit" "PRIMME built as C++ (f8997ce)" ||
    fail "primme edit" "staged chromaform lacks the PRIMME edit: run prepare"
  (( bad == 0 )) || die "preflight failed; nothing was built"
  printf -- '---- preflight passed\n'
}

build() {
  require_allocation
  [[ -e "${BIN}" ]] && die "${BIN} exists: an install is never overwritten (choose another PREFIX)"
  load_env
  provenance_header
  local pkg
  for pkg in "${PKGS[@]}" eigen; do
    [[ -f "${SRC}/.staged-${pkg}" ]] || die "${SRC}/${pkg} is not staged: run prepare"
  done
  [[ -x "${CHROMAFORM}" && -f "${PYLIB}/.staged" ]] || die "chromaform or pylib not staged: run prepare"

  # --autoflags=no: chromaform adds no flags, so -O3, -fopenmp and the language standards are all here.
  # libxml2 (static, ours): its xml2-config flags; "xml2" in LIBS makes chromaform add nothing more.
  local xml_cflags xml_libs
  xml_cflags=$(xml2-config --cflags)
  xml_libs=$(xml2-config --libs)
  export CC=${MPICC} CXX=${MPICXX}
  export CFLAGS="${OPT} ${ARCH} -fopenmp -std=gnu17 ${xml_cflags}"
  export CXXFLAGS="${OPT} ${ARCH} -fopenmp -std=${CXXSTD} ${xml_cflags}"
  export LDFLAGS="${OPT} -fopenmp -L${OPENBLAS}/lib -Wl,-rpath,${OPENBLAS}/lib"
  export LIBS="-lopenblas ${xml_libs}"
  # EXTRA_CMAKE comes after chromaform's own -D options on its cmake lines, so it wins (last -D wins).
  export CMAKE_EXTRA_FLAGS="-DBLAS_openblas_LIBRARY=${OPENBLAS}/lib/libopenblas.a${EXTRA_CMAKE:+ ${EXTRA_CMAKE}}"

  # A variant links the base packages it does not rebuild; chromaform skips any installed package.
  mkdir -p "${PREFIX}"
  for pkg in "${REUSE[@]}"; do
    [[ -d "${BASE_PREFIX}/${pkg}" ]] || die "reuse: ${BASE_PREFIX}/${pkg} missing"
    [[ -e "${PREFIX}/${pkg}" ]] || ln -s "${BASE_PREFIX}/${pkg}" "${PREFIX}/${pkg}"
  done
  export MAKE_JN=${JOBS} MAKEFLAGS=-k
  export PYTHON PYTHONPATH=${PYLIB} PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
  # No network: any download chromaform should not attempt fails at once.
  export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 ftp_proxy=http://127.0.0.1:9
  export HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy} GIT_ALLOW_PROTOCOL=file
  unset no_proxy NO_PROXY

  local cmd=(
    "${CHROMAFORM}" --cpu --mg --avx512 --autoflags=no
    --cmake=system --blas=openblas-system --libxml2=system --gmp=build
    --source-dir="${SRC}" --build-dir="${BUILD}" --install-dir="${PREFIX}"
    chroma
  )
  {
    printf 'build:      %s on %s, job %s\n  %s\n' "$(now)" "$(hostname)" "${SLURM_JOB_ID}" "${cmd[*]@Q}"
    printf '  script:   %s  sha256=%s\n' "${SELF}" "$(sha256sum "${SELF}" | cut -d' ' -f1)"
    printf '  os:       %s\n' "$(. /etc/os-release && echo "${PRETTY_NAME}")"
    printf '  CC=%s\n  CXX=%s (%s)\n' "${CC}" "${CXX}" "$("${CXX}" --version | head -1)"
    printf '  mpi:      %s (%s)\n' "${LQ_CPU_MPI_ROOT}" "$(mpirun --version 2>&1 | head -1)"
    printf '  CFLAGS=%s\n  CXXFLAGS=%s\n  LDFLAGS=%s\n  LIBS=%s\n  CMAKE_EXTRA_FLAGS=%s\n' \
      "${CFLAGS}" "${CXXFLAGS}" "${LDFLAGS}" "${LIBS}" "${CMAKE_EXTRA_FLAGS}"
    printf '  MAKE_JN=%s MAKEFLAGS=%s\n  PYTHON=%s PYTHONPATH=%s\n' "${MAKE_JN}" "${MAKEFLAGS}" "${PYTHON}" "${PYTHONPATH}"
    printf '  cmake:    %s (%s)\n  libxml2:  %s (%s)\n' "$(cmake --version | head -1)" "$(command -v cmake)" \
      "$(xml2-config --version)" "$(command -v xml2-config)"
    printf '  tools:    %s (autoconf %s, automake %s)\n' "${BUILDTOOLS}" \
      "$(autoconf --version | head -1 | awk '{print $NF}')" "$(automake --version | head -1 | awk '{print $NF}')"
  } >> "${PROVENANCE}"

  mkdir -p "${BUILD}"
  ( cd "${BUILD}" && "${cmd[@]}" )

  [[ -x "${BIN}" ]] || die "chromaform finished but ${BIN} is missing"
  # Which compression the installed mg_proto really has (the nocomp12 override must have won).
  local mgcfg=${PREFIX}/mgproto-qphix-qdpxx-double-nd4-avx512/include/MG_config.h comp=off
  grep -q '^#define MG_QPHIX_COMPRESS12' "${mgcfg}" && comp=on
  [[ "${VARIANT}" != nocomp12 || "${comp}" == off ]] || die "nocomp12: ${mgcfg} still defines MG_QPHIX_COMPRESS12"
  printf 'built:      %s  chroma sha256=%s  mg_proto compress12=%s\n' "$(now)" \
    "$(sha256sum "${BIN}" | cut -d' ' -f1)" "${comp}" >> "${PROVENANCE}"
  [[ -e "${PREFIX}/bin" ]] || ln -s "${CHROMA_NAME}/bin" "${PREFIX}/bin"
}

# One chroma run under srun (lq launch rule: srun --mpi=pmix inside sbatch). Appends to RESULTS.
# A QPhiX-enabled chroma aborts at start unless every QPhiX option is given (job 1364686):
# block sizes -by/-bz (divide the local Y and Z extents, 16 here), padding -pxy/-pxyz, -minct, and
# the threading -c (cores per rank, = OMP threads) with -sy/-sz (SMT threads per core, 1 here).
QPHIX_ARGS=(-by 4 -bz 4 -pxy 1 -pxyz 0 -minct 1 -sy 1 -sz 1)
want() { [[ "${CASES}" == all || " ${CASES} " == *" $1 "* ]]; }
run_case() {
  local name=$1 ranks=$2 threads=$3 input=$4 geom=$5 rc
  want "${name}" || return 0
  set +e
  ( cd "${RUN_DIR}" && OMP_NUM_THREADS=${threads} OMP_PLACES=cores OMP_PROC_BIND=close \
      timeout 300 srun --mpi=pmix --ntasks="${ranks}" --cpus-per-task="${threads}" --cpu-bind=cores \
      "${BIN}" -i "${input}" -o "${name}.out.xml" -geom ${geom} "${QPHIX_ARGS[@]}" -c "${threads}" \
      > "${name}.log" 2>&1 )
  rc=$?
  set -e
  if (( rc == 0 )) && grep -q 'CHROMA: ran successfully' "${RUN_DIR}/${name}.log"; then
    RESULTS+=("${name}: PASS  (${ranks} ranks x ${threads} threads, -geom ${geom})")
  else
    RESULTS+=("${name}: FAIL  rc=${rc}, see ${RUN_DIR}/${name}.log")
    FAILED=1
  fi
  # The solvers' own residual lines against chroma's operator (last two of each run).
  local line
  while IFS= read -r line; do RESULTS+=("    ${line}"); done < <(
    grep -E 'Relative Rsd|\|\| r \|\| / \|\| b \|\||rel resid|SOLVE FAILED|Exception' "${RUN_DIR}/${name}.log" | tail -n 2)
}

# ldd, then two weak-field clover propagators (12 solves each) shaped like a CPU production solve:
# QPhiX AVX-512 on the fine level, mg_proto on two coarse levels (blocking 4^4 then 2^4, NullVecs 24 32),
# several ranks, local 16^4. Inputs in tests/ beside this script, derived from chroma's
# tests/chroma/hadron/propagator/unprec_clover_mg_proto.ini.xml (their annotations say what changed).
# Not testable, by design of mg_proto 6ec4ae7 (also current devel), not a build fault:
#   - one rank: with every direction on-node, CommunicateHaloSyncInOMPParallel (mg_proto halo.h) calls
#     QMP_start on the never-declared combined handle -> "assert failed mh!=NULL" (job 1364700). The
#     serial CommunicateHaloSync guards with NumNonLocalDirs() > 0; production always splits >= 1 dir.
#   - NullVecs not in {6, 8, 16, ..., 64}: "Unhandled dispatch size" in the QPhiX/coarse transfers, then
#     an FGMRES "dim > 0" assertion (the shipped test's 12; job 1364700).
# Boundary rule for the mg_proto QPhiX solvers: <AntiPeriodicT> must match the action's t boundary
# (chroma's helpers flip the links' sign off and pass t_bc to QPhiX). It was NOT the cause of the wrong
# solutions: job 1365191 failed the same way with it fixed and with a periodic control (rel resid ~1.5
# in all mg_proto QPhiX cases, mg_proto's own FGMRES converged), while chroma's own QPhiX solver passed.
# Cases (CASES selects; default all):
#   mgproto_qphix_eo_4r           the EO mg_proto QPhiX solve (aniso, as the shipped test)
#   mgproto_qphix_eo_periodic_4r  control: periodic t, AntiPeriodicT false (links unmodified)
#   mgproto_qphix_eo_iso_4r       isotropic, production-like action (our ensemble's c_sw and stout)
#   mgproto_qphix_2r              the unprec mg_proto QPhiX solve
#   qphix_vs_native_2r            reference without mg_proto: native BiCGStab, then chroma's own QPhiX
#                                 BiCGStab (AVX-512 kernels, uncompressed, checked against chroma's operator)
#   mgproto_ref_4r                mg_proto WITHOUT QPhiX (MG_PROTO_CLOVER_INVERTER: mg_proto's QDP++
#                                 reference fine operator), the isotropic input: does mg_proto's generic
#                                 path agree with chroma's operator?
# PREFIX/PROVENANCE.txt is rewritten only when all cases ran; every run writes RUN_DIR/RESULTS.txt.
test_chroma() {
  require_allocation
  load_env
  [[ -x "${BIN}" ]] || die "no ${BIN}: run build first"
  RUN_DIR=${ROOT}/runs/$(date +%Y_%m_%d)_chroma_cpu_lq3${VARIANT:+_${VARIANT}}_smoke_${SLURM_JOB_ID}
  mkdir -p "${RUN_DIR}"
  RESULTS=()
  FAILED=0

  ldd "${BIN}" > "${RUN_DIR}/ldd.txt"
  local unresolved
  unresolved=$(grep -c 'not found' "${RUN_DIR}/ldd.txt" || true)
  if (( unresolved == 0 )); then
    RESULTS+=("ldd: 0 unresolved")
  else
    RESULTS+=("ldd: ${unresolved} unresolved, see ${RUN_DIR}/ldd.txt")
    FAILED=1
  fi

  local eo=mgproto_qphix_eo_16x16x32x32.ini.xml per=mgproto_qphix_eo_periodic_16x16x32x32.ini.xml
  cp "${HERE}/tests/${eo}" "${HERE}/tests/mgproto_qphix_eo_iso_16x16x32x32.ini.xml" \
     "${HERE}/tests/mgproto_qphix_16x32.ini.xml" "${HERE}/tests/qphix_vs_native_16x32.ini.xml" "${RUN_DIR}/"
  # The periodic control: the EO input with a periodic t boundary and AntiPeriodicT false.
  sed -e 's|<boundary>1 1 1 -1</boundary>|<boundary>1 1 1 1</boundary>|' \
      -e 's|<AntiPeriodicT>true</AntiPeriodicT>|<AntiPeriodicT>false</AntiPeriodicT>|' \
      "${HERE}/tests/${eo}" > "${RUN_DIR}/${per}"
  grep -q '<boundary>1 1 1 1</boundary>' "${RUN_DIR}/${per}" &&
    grep -q '<AntiPeriodicT>false</AntiPeriodicT>' "${RUN_DIR}/${per}" || die "periodic control edit did not apply"
  # The mg_proto reference (no QPhiX): the isotropic input with the non-QPhiX mg_proto solver.
  local ref=mgproto_ref_iso_16x16x32x32.ini.xml
  sed 's|<invType>MG_PROTO_QPHIX_EO_CLOVER_INVERTER</invType>|<invType>MG_PROTO_CLOVER_INVERTER</invType>|' \
    "${HERE}/tests/mgproto_qphix_eo_iso_16x16x32x32.ini.xml" > "${RUN_DIR}/${ref}"
  grep -q '<invType>MG_PROTO_CLOVER_INVERTER</invType>' "${RUN_DIR}/${ref}" || die "reference invType edit did not apply"

  # 2 ranks x 32 threads and 4 ranks x 16 threads: both fill the 64-core node. At most 5 min each.
  run_case mgproto_qphix_eo_4r          4 16 "${eo}"                                    "1 1 2 2"
  run_case mgproto_qphix_eo_periodic_4r 4 16 "${per}"                                   "1 1 2 2"
  run_case mgproto_qphix_eo_iso_4r      4 16 mgproto_qphix_eo_iso_16x16x32x32.ini.xml   "1 1 2 2"
  run_case mgproto_qphix_2r             2 32 mgproto_qphix_16x32.ini.xml                "1 1 1 2"
  run_case qphix_vs_native_2r           2 32 qphix_vs_native_16x32.ini.xml              "1 1 1 2"
  run_case mgproto_ref_4r               4 16 "${ref}"                                   "1 1 2 2"

  printf -- '---- test results, variant %s, cases %s (%s)\n' "${VARIANT:-base}" "${CASES}" "${RUN_DIR}"
  printf '  %s\n' "${RESULTS[@]}"
  printf '%s\n' "${RESULTS[@]}" > "${RUN_DIR}/RESULTS.txt"
  if [[ "${CASES}" != all ]]; then
    (( FAILED == 0 )) || die "a test failed (results in ${RUN_DIR}/RESULTS.txt)"
    return 0
  fi

  local dirty=""
  [[ -n "$(git -C "${WORKFLOW}" status --porcelain --untracked-files=no)" ]] && dirty=", dirty"
  {
    printf '# chroma-cpu-lq3%s: CPU Chroma (QMP + QDP++ + QPhiX avx512 + mg_proto + PRIMME), lq3 EPYC 9355\n' \
      "${VARIANT:+-${VARIANT}}"
    printf 'tested:     %s on %s, job %s\n' "$(now)" "$(hostname)" "${SLURM_JOB_ID}"
    printf 'workflow:   %s (%s%s)\n' "$(git -C "${WORKFLOW}" rev-parse HEAD)" "$(git -C "${WORKFLOW}" rev-parse --abbrev-ref HEAD)" "${dirty}"
    printf 'exe:        %s  sha256=%s  bytes=%s\n' "${BIN}" "$(sha256sum "${BIN}" | cut -d' ' -f1)" "$(stat -c %s "${BIN}")"
    printf 'runtime:    LQ_TARGET=cpu source grid-lqcd-workflow/machines/lq.sh; srun --mpi=pmix\n'
    printf 'test:       %s\n' "${RESULTS[@]}"
    printf -- '--- build: %s ---\n' "${PROVENANCE}"
    cat "${PROVENANCE}"
  } > "${PREFIX}/PROVENANCE.txt"
  (( FAILED == 0 )) || die "a test failed (install kept; results in ${PREFIX}/PROVENANCE.txt)"
}

print_paths
case "${ACTION}" in
  prepare) prepare ;;
  check)   check ;;
  build)   build ;;
  test)    test_chroma ;;
  info)    ;;
  *) printf 'usage: %s {prepare|check|build|test|info}\n' "$0" >&2; exit 2 ;;
esac
