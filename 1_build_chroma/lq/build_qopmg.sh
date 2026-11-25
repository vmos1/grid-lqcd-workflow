#!/usr/bin/env bash
# QOP-MG CPU Chroma for lq3: USQCD's QLA + QDP/C + QOPQDP (Wilson-clover multigrid) and a Chroma built
# with --enable-qop-mg. Its QOP_CLOVER_MULTIGRID_INVERTER is registered as an MdagM solver, so it is
# usable inside HMC monomials; chroma's mg_proto solvers are LinOp (propagator) solvers only.
#
# Sources (user's approval 2026-11-24: these three downloads exactly), cloned on the login node and
# checked out at pinned commits; a commit that does not resolve stops the fetch. No publisher checksums
# exist; git's content hash is the check, and PROVENANCE records the full SHA and the sha256 of
# `git archive` of that commit:
#   QLA     https://github.com/usqcd-software/qla.git     eaffb9220793ec5048150d6abba940a96be3c006 (qla1-9-0)
#   QDP/C   https://github.com/usqcd-software/qdp.git     tag qdp1-11-1 -> f6b0a533bd3b9c4d7131efe090ef085cb3093fa3
#   QOPQDP  https://github.com/usqcd-software/qopqdp.git  4aef533030cb04c1ff96317badc26d816c646cfe (master
#           2019-01, after 0.21.1, with a08da00 "merged MG leak fixes")
# The code is untrusted: fetch clones and checks out only; nothing from it runs on the login node
# (autoreconf, configure and make run in the lq3 job).
#
# Reused from the base build (installs/external/chroma-cpu-lq3, read only): QMP, QDP++ (and its QIO 3.0
# + LIME, which QDP/C and QOPQDP are built against, so the binary has one QIO), QPhiX, PRIMME, mg_proto.
#
# Build (lq3 job; spack gcc 15.3 through Grid-CPU's OpenMPI 4.1.8 wrappers; our autotools 2.72/1.16.5):
#   QLA     --enable-openmp (OpenMP in its site loops, which is where QDP/C's per-site work runs); all
#           colours and precisions (the default; chroma's QOP-MG link line names them all, incl. quad)
#   QDP/C   --with-qmp (base) --with-qla --with-qio (QDP++'s); all colours (default)
#   QOPQDP  --with-qdp --with-qla --with-qio --with-qmp; all colours: its multigrid (USE_MG) is compiled
#           only with the Nc=N libraries (configure.ac)
#   chroma  a761e7e from a copy of the base's staged source, configured as the base build plus
#           --enable-qop-mg --with-qla --with-qdpc --with-qopqdp --with-qio. Include paths: chroma's C++
#           must see only QDP++'s qdp.h; its QOP-MG files reach QLA alone through wilsonmg-interface.h
#           ("without exposing parts of the QDP software stack", it says), and QOP_MG_CXXFLAGS holds
#           -I<qla>/include (its -I${QOP_DIR}/include is empty and unused). The QOPQDP/QDP-C/QLA include
#           dirs go into CFLAGS only, ahead of the bridge's own list, which puts QDP++'s dir (holding QIO
#           and a C++ qdp.h) before QDP/C's. Job 1365306 had them in CXXFLAGS too: QDP/C's qdp.h shadowed
#           QDP++'s and every chroma C++ file failed. Link: QDP/C's qdp-config library order (all
#           colours, with QIO/LIME/QMP) and QMP/QIO/LIME again are appended to LIBS, because chroma's
#           QOP_MG_LIBS lists qdp_common/qdp_int and qla_c99/qla_cmath before the libraries using them.
#   C flags -O3 -march=znver5 -fopenmp -std=gnu17. Each C library is tried strict first; if that build
#           fails it is rebuilt in a fresh tree with -fpermissive (GCC <= 13 C semantics for this 2014-2019
#           code); PROVENANCE says which. Network fenced off as in build_chroma_cpu.sh.
#   threads QDP/C has no OpenMP configure switch. Its per-site arithmetic runs in QLA routines, threaded
#           only when QLA is configured --enable-openmp (done here); QDP/C itself has a few _OPENMP regions
#           (QDP_T_eq_func(i), QDP_extract_T, layout_shiftopt, com_common, qdp_common) that -fopenmp turns
#           on. QOPQDP's Wilson dslash and multigrid carry their own OpenMP pragmas (-fopenmp).
#
# Patch (patches/ beside this file; fetch applies it once to our chroma copy only, PROVENANCE records it):
#   chroma-wilsonmg-free-gauge-copies.patch  other_libs/wilsonmg/lib/wilsonmg_p.c, MGP(initialize): free
#           the four QDP/C gauge fields and their array after QOP_create_G_from_qdp (which copies them) and
#           the QOP gauge field once both Wilson operators exist (QOP_wilson_create_L_from_G and
#           QOP_FD_wilson_create_L_from_L copy the links). Unpatched, every QOP-MG solver object (one per
#           HMC solve) leaks 8 gauge fields per rank. QOP_wilsonMgNew is declared (void) in QOPQDP and the
#           bridge calls it so: not patched.
#   chroma-qopmg-release-static-gauge.patch  lib/actions/ferm/invert/qop_mg/syssolver_linop_qop_mg_w.cc:
#           u.resize(0) after MGP(initialize). The file-scope static gauge copy u is read only there (the
#           peekpoke callbacks); kept, it is destroyed at exit after QDP++'s allocator and every QOP-MG
#           chroma/hmc run crashed at exit with correct results printed (job 1365333).
#
# Layout:
#   builds/chroma-cpu-lq3-qopmg/src/{qla,qdp,qopqdp,chroma}   sources (fetch)
#   builds/chroma-cpu-lq3-qopmg/build/<pkg>-<jobid>-<mode>    build trees (fresh per attempt)
#   installs/external/chroma-cpu-lq3-qopmg/{qla,qdp,qopqdp,<chroma>}, links to the reused base packages,
#   bin -> <chroma>/bin, PROVENANCE.txt
#
# Actions:
#   fetch    login node: clone + pinned checkout + copy of the chroma source + patch; records provenance
#   check    lq3 job: toolchain, base packages, fetched and patched sources, validation inputs
#   build    lq3 job: QLA, QDP/C, QOPQDP, chroma (COMPUTE NODE ONLY); each installed once (marker)
#   test     lq3 job (COMPUTE NODE ONLY):
#              prop_2r, prop_4r  16^3x32 weak-field clover propagator with QOP_CLOVER_MULTIGRID_INVERTER
#                       on 2 ranks (-geom 1 1 1 2) and 4 ranks (-geom 1 1 2 2): passes when chroma ends
#                       normally (the solver aborts if || r || / || b || against chroma's own operator
#                       exceeds Residual x RsdToleranceFactor) AND the plaquette QDP/C sees ("QDP: Average
#                       plaquette", printed with %g) equals chroma's of the same links ("Plaquette from
#                       State: w_plaq") to 5e-6: the QDP++ -> QDP/C transfer and layouts agree.
#              hmc_check        one short HMC trajectory (16^3x32 weak field, LW gauge + one even-odd clover
#                       2-flavour monomial, leapfrog 4 steps, tau 0.2, 4 ranks), run twice with the same
#                       seeds: QOP-MG MdagM vs QPhiX CG (validated), both to 1e-12. Passes when both end
#                       normally, AcceptP matches, every force record (F_sq, F_avg, F_max) agrees to 1e-6 and
#                       |deltaH difference| < 1e-3 (the fermion action is ~1.6e6 = eta^dag eta on this
#                       volume, so a 1e-12 residual leaves ~1e-6..1e-4 in H); the inputs must differ only
#                       in <InvertParam> (checked).
#   validate lq3 job (COMPUTE NODE ONLY), the input agent's checks on the real 16^3x48 configuration
#            cl3_16_48_b6p1_m0p2450 cfg_11100, inputs read from 5_studies/chroma_inputs/lq/ (copied, never
#            edited), 4 ranks x 16 threads, -geom 1 1 1 4:
#              prop16   qopmg_validate_prop16.ini.xml (chroma): point-source propagator at -0.2450, QPhiX
#                       BiCGStab reference vs QOP-MG on the unpreconditioned clover action. Passes when chroma
#                       ends normally, all >= 12 QOPMG_LINOP_SOLVER relative residuals (chroma's operator)
#                       are < 5e-10, the QDP/C plaquette equals chroma's to 5e-6, and the two prop_corr arrays
#                       agree to 1e-6 relative on every timeslice (expected about 1e-9; the value is printed).
#              hmc16    qopmg_validate_hmc16_qopmg.ini.xml vs ..._qphixcg.ini.xml (hmc, each in its own
#                       directory): 2 trajectories, same seeds; MD::ratio's force solve has RefreshThreshold
#                       10, so the subspace erase/rebuild path runs. The pair differs in solver blocks,
#                       predictors and the QOP side's ERASE_MG_SUBSPACE measurements (the input agent's
#                       generators prove the rest identical). Passes when both end normally with 2 deltaH
#                       records each, AcceptP matches, |deltaH difference| < 1e-2, every force record agrees to
#                       1e-5 relative and every w_plaq to 1e-8 relative (measured values printed).
#   Every srun step is named after its case; RUN_DIR/maxrss.txt gets, per case, sacct's MaxRSS (largest
#   task) and a 10 s ps sample of the ranks' RSS (RUN_DIR/<case>.rss.txt), with the largest rank's peak
#   in the first and second half of the run: a per-solve leak shows as a larger second half.
#   info     print paths
# Job wrapper: build_qopmg_job.sh beside this file.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKFLOW=$(cd "${HERE}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)
SELF=${HERE}/$(basename "${BASH_SOURCE[0]}")

ACTION=${1:-info}
JOBS=${JOBS:-64}
STAGE=${ROOT}/builds/chroma-cpu-lq3-qopmg
SRC=${STAGE}/src
BUILD=${STAGE}/build
PROVENANCE=${STAGE}/PROVENANCE.txt
PREFIX=${ROOT}/installs/external/chroma-cpu-lq3-qopmg
BASE_STAGE=${ROOT}/builds/chroma-cpu-lq3
BASE_PREFIX=${ROOT}/installs/external/chroma-cpu-lq3
BUILDTOOLS=${ROOT}/installs/external/buildtools-lq3
QIO_DIR=${BASE_PREFIX}/qdpxx-double-nd4          # QDP++'s install carries QIO 3.0 + LIME + qio-config
QMP_DIR=${BASE_PREFIX}/qmp
CHROMA_NAME=chroma-qopmg-mgproto-qphix-qdpxx-double-nd4-avx512
BIN_DIR=${PREFIX}/${CHROMA_NAME}/bin
REUSE=(qmp qdpxx-double-nd3 qdpxx-double-nd4 qphix-qdpxx-double-nd4-avx512 primme-cpu mgproto-qphix-qdpxx-double-nd4-avx512)
PATCH_DIR=${HERE}/patches
CHROMA_PATCHES=(chroma-wilsonmg-free-gauge-copies chroma-qopmg-release-static-gauge)  # applied in this order to SRC/chroma (patch -p1)
declare -A PATCH_WHY=(
  [chroma-wilsonmg-free-gauge-copies]="MGP(initialize) leaked 4 QDP/C gauge fields + array and 1 QOP gauge field per solver object (one per HMC solve); QOPQDP copies the links (common_p.c create_G_from_qdp, wilson_dslash_p.c create_L_from_G / FD create_L_from_L), so free them after use"
  [chroma-qopmg-release-static-gauge]="syssolver_linop_qop_mg_w.cc's file-scope static gauge field (read only inside MGP(initialize)) was destroyed at exit after QDP++'s allocator: every QOP-MG chroma/hmc run crashed at exit (job 1365333); u.resize(0) after MGP(initialize)"
)
INPUTS=${WORKFLOW}/5_studies/chroma_inputs/lq       # the input agent's validation inputs (read only)
VALIDATE_INPUTS=(qopmg_validate_prop16.ini.xml qopmg_validate_hmc16_qopmg.ini.xml qopmg_validate_hmc16_qphixcg.ini.xml)
CFG16=${ROOT}/inputs/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime

PKGS=(qla qdp qopqdp)
declare -A URL=(
  [qla]=https://github.com/usqcd-software/qla.git
  [qdp]=https://github.com/usqcd-software/qdp.git
  [qopqdp]=https://github.com/usqcd-software/qopqdp.git
)
declare -A REF=(                                   # what is checked out
  [qla]=eaffb9220793ec5048150d6abba940a96be3c006
  [qdp]=refs/tags/qdp1-11-1
  [qopqdp]=4aef533030cb04c1ff96317badc26d816c646cfe
)
declare -A WANT=(                                  # the commit it must resolve to
  [qla]=eaffb9220793ec5048150d6abba940a96be3c006
  [qdp]=f6b0a533bd3b9c4d7131efe090ef085cb3093fa3
  [qopqdp]=4aef533030cb04c1ff96317badc26d816c646cfe
)

now() { date '+%Y-%m-%d %H:%M:%S %Z'; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

print_paths() {
  printf 'STAGE=%s\nPREFIX=%s\nBASE_PREFIX=%s\nQIO_DIR=%s\nBIN_DIR=%s\nJOBS=%s\n' \
    "${STAGE}" "${PREFIX}" "${BASE_PREFIX}" "${QIO_DIR}" "${BIN_DIR}" "${JOBS}"
}

provenance_header() {
  [[ -f "${PROVENANCE}" ]] && return
  mkdir -p "${STAGE}"
  {
    printf '# Provenance of the lq3 QOP-MG CPU Chroma (1_build_chroma/lq/build_qopmg.sh)\n'
    printf 'created:    %s on %s\n' "$(now)" "$(hostname)"
    printf 'workflow:   %s (%s)\n' "$(git -C "${WORKFLOW}" rev-parse --short HEAD)" "$(git -C "${WORKFLOW}" rev-parse --abbrev-ref HEAD)"
    printf 'script:     %s  sha256=%s\n' "${SELF}" "$(sha256sum "${SELF}" | cut -d' ' -f1)"
    printf 'PREFIX:     %s\nreuses:     %s (qmp, qdpxx-double-nd3/nd4 with QIO+LIME, qphix, primme-cpu, mgproto)\n' \
      "${PREFIX}" "${BASE_PREFIX}"
  } > "${PROVENANCE}"
}

# ---- fetch (login node) ---------------------------------------------------------------------------
fetch_repo() {
  local pkg=$1 dir=${SRC}/$1 sha tarsha
  if [[ -f "${SRC}/.fetched-${pkg}" ]]; then
    sha=$(git -C "${dir}" rev-parse HEAD)
    [[ "${sha}" == "${WANT[$pkg]}" ]] || die "${dir} is at ${sha}, want ${WANT[$pkg]}"
    printf 'already fetched: %-7s %s\n' "${pkg}" "${sha}"
    return
  fi
  [[ -e "${dir}" ]] && die "incomplete ${dir} (no .fetched-${pkg}). Inspect it; to discard it yourself: rm -rf ${dir}"
  printf 'cloning %s ...\n' "${URL[$pkg]}"
  GIT_TERMINAL_PROMPT=0 git clone --quiet --no-checkout --no-recurse-submodules "${URL[$pkg]}" "${dir}"
  sha=$(git -C "${dir}" rev-parse --verify --quiet "${REF[$pkg]}^{commit}") ||
    die "${pkg}: ${REF[$pkg]} does not resolve to a commit in ${URL[$pkg]}"
  [[ "${sha}" == "${WANT[$pkg]}" ]] || die "${pkg}: ${REF[$pkg]} resolves to ${sha}, want ${WANT[$pkg]}"
  git -C "${dir}" checkout --quiet --detach "${sha}"
  [[ "$(git -C "${dir}" rev-parse HEAD)" == "${sha}" ]] || die "${pkg}: checkout did not land on ${sha}"
  git -C "${dir}" fsck --no-dangling --no-progress >/dev/null 2>&1 || die "${pkg}: git fsck failed in ${dir}"
  [[ ! -f "${dir}/.gitmodules" ]] || die "${pkg}: has submodules (${dir}/.gitmodules); not expected, stopping"
  tarsha=$(git -C "${dir}" archive --format=tar "${sha}" | sha256sum | cut -d' ' -f1)
  printf 'source:     %-7s %s  %s  ref %s  commit %s (%s)  git-archive-tar sha256=%s  fsck ok\n' "${pkg}" \
    "${URL[$pkg]}" "$(git -C "${dir}" log -1 --format='%ad %s' --date=short)" "${REF[$pkg]}" "${sha}" \
    "$(git -C "${dir}" describe --tags --always "${sha}" 2>/dev/null)" "${tarsha}" >> "${PROVENANCE}"
  : > "${SRC}/.fetched-${pkg}"
  printf 'fetched: %-7s %s\n' "${pkg}" "${sha}"
}

fetch() {
  [[ -z "${SLURM_JOB_ID:-}" ]] || die "fetch runs on the login node"
  mkdir -p "${SRC}"
  provenance_header
  local pkg
  for pkg in "${PKGS[@]}"; do fetch_repo "${pkg}"; done
  # chroma: a copy of the base's staged source (a761e7e + submodules, incl. other_libs/wilsonmg), so
  # --enable-qop-mg's autoreconf/configure never touches the base source.
  if [[ -f "${SRC}/.copied-chroma" ]]; then
    printf 'already copied: chroma\n'
  else
    [[ -e "${SRC}/chroma" ]] && die "incomplete ${SRC}/chroma. To discard it yourself: rm -rf ${SRC}/chroma"
    cp -R --preserve=mode,timestamps "${BASE_STAGE}/src/chroma" "${SRC}/chroma"
    printf 'source:     chroma  copy of %s  commit %s\n' "${BASE_STAGE}/src/chroma" \
      "$(git -C "${SRC}/chroma" rev-parse HEAD)" >> "${PROVENANCE}"
    : > "${SRC}/.copied-chroma"
    printf 'copied: chroma\n'
  fi
  patch_chroma
  printf 'fetched:    %s on %s\n' "$(now)" "$(hostname)" >> "${PROVENANCE}"
}

# Apply patches/<name>.patch to our chroma copy once; the marker holds the patch's sha256. A patch file
# that changed after it was applied stops here (the source would no longer match it).
patch_chroma() {
  local name file sha marker
  [[ -f "${PREFIX}/.installed-chroma" ]] && die "chroma is already installed in ${PREFIX}; a patch now would not reach the binary"
  for name in "${CHROMA_PATCHES[@]}"; do
    file=${PATCH_DIR}/${name}.patch
    marker=${SRC}/.patched-${name}
    sha=$(sha256sum "${file}" | cut -d' ' -f1)
    if [[ -f "${marker}" ]]; then
      [[ "$(cat "${marker}")" == "${sha}" ]] ||
        die "${name}: applied with sha256 $(cat "${marker}"), the file now has ${sha}; restore ${SRC}/chroma by hand first"
      printf 'already patched: %s\n' "${name}"
      continue
    fi
    patch --dry-run --forward --batch --silent -p1 -d "${SRC}/chroma" -i "${file}" ||
      die "${name}: does not apply cleanly to ${SRC}/chroma (nothing changed)"
    patch --forward --batch --no-backup-if-mismatch -p1 -d "${SRC}/chroma" -i "${file}" ||
      die "${name}: patch failed after a clean dry run; inspect ${SRC}/chroma"
    printf '%s\n' "${sha}" > "${marker}"
    {
      printf 'patch:      chroma  %s  sha256=%s\n            why: %s\n' "${file}" "${sha}" "${PATCH_WHY[$name]}"
      printf 'not patched: QOP_wilsonMgNew is declared (void) (qopqdp include/qop_mg.h, qop_d3.h) and the bridge calls it with no arguments (wilsonmg_p.c:43)\n'
      printf 'threading:  QDP/C has no OpenMP switch; its site loops are QLA routines, threaded by QLA --enable-openmp; QDP/C has a few own _OPENMP regions (eq_func, extract, shift layout, comms) enabled by -fopenmp; QOPQDP dslash/MG use OpenMP\n'
    } >> "${PROVENANCE}"
    printf 'patched: %s\n' "${name}"
  done
}

# Exit 0 when every chroma patch is recorded (marker = the patch file's sha256) and each line it adds is
# in its target file. Uses no `patch`, which compute nodes may lack (each patch touches one file).
chroma_patched() {
  local name file target line
  for name in "${CHROMA_PATCHES[@]}"; do
    file=${PATCH_DIR}/${name}.patch
    [[ -f "${SRC}/.patched-${name}" ]] || return 1
    [[ "$(cat "${SRC}/.patched-${name}")" == "$(sha256sum "${file}" | cut -d' ' -f1)" ]] || return 1
    target=${SRC}/chroma/$(awk '/^\+\+\+ b\// { print substr($2, 3); exit }' "${file}")
    [[ -f "${target}" ]] || return 1
    while IFS= read -r line; do
      grep -qxF -- "${line}" "${target}" || return 1
    done < <(awk '/^\+\+\+ / { next } /^\+/ { print substr($0, 2) }' "${file}")
  done
}

# ---- job-side environment -------------------------------------------------------------------------
require_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "\"${ACTION}\" runs only inside a Slurm job on lq3 (SLURM_JOB_ID is unset)"
}

load_env() {
  LQ_TARGET=cpu source "${WORKFLOW}/machines/lq.sh"
  export PATH=${BUILDTOOLS}/bin:${PATH}
  export PKG_CONFIG_PATH=${BUILDTOOLS}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}
  OPENBLAS=$(spack find --paths openblas | awk '/^openblas/ {print $2}')
  MPICC=${LQ_CPU_MPI_ROOT}/bin/mpicc
  MPICXX=${LQ_CPU_MPI_ROOT}/bin/mpicxx
}

fence_network() {
  export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 ftp_proxy=http://127.0.0.1:9
  export HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy} GIT_ALLOW_PROTOCOL=file
  unset no_proxy NO_PROXY
}

check() {
  require_allocation
  load_env
  local bad=0 t p
  ok()   { printf '  ok    %-22s %s\n' "$1" "$2"; }
  fail() { printf '  FAIL  %-22s %s\n' "$1" "$2"; bad=1; }
  printf -- '---- preflight on %s (%s), %s\n' "$(hostname)" "$(. /etc/os-release && echo "${PRETTY_NAME}")" "$(now)"
  [[ -x "${MPICC}" && -x "${MPICXX}" ]] && ok mpicc "${MPICC} ($("${MPICC}" --version | head -1))" || fail mpicc "missing"
  for t in autoreconf autoconf automake aclocal libtoolize m4 perl make cmake xml2-config; do
    command -v "${t}" >/dev/null 2>&1 && ok "${t}" "$(command -v "${t}")" || fail "${t}" "none on PATH"
  done
  for p in "${REUSE[@]}"; do
    [[ -d "${BASE_PREFIX}/${p}" ]] && ok "base ${p}" "" || fail "base ${p}" "missing in ${BASE_PREFIX}"
  done
  [[ -f "${QIO_DIR}/lib/libqio.a" && -f "${QIO_DIR}/lib/liblime.a" && -f "${QIO_DIR}/include/qio.h" ]] &&
    ok "QIO + LIME" "${QIO_DIR}" || fail "QIO + LIME" "not in ${QIO_DIR}"
  [[ -x "${QMP_DIR}/bin/qmp-config" ]] && ok qmp-config "$("${QMP_DIR}/bin/qmp-config" --version 2>&1 | head -1)" ||
    fail qmp-config "no ${QMP_DIR}/bin/qmp-config"
  for p in "${PKGS[@]}"; do
    if [[ -f "${SRC}/.fetched-${p}" && "$(git -C "${SRC}/${p}" rev-parse HEAD)" == "${WANT[$p]}" ]]; then
      ok "src ${p}" "${WANT[$p]}"
    else
      fail "src ${p}" "not fetched at ${WANT[$p]} (run fetch on the login node)"
    fi
  done
  [[ -f "${SRC}/.copied-chroma" ]] && ok "src chroma" "$(git -C "${SRC}/chroma" rev-parse --short HEAD)" || fail "src chroma" "not copied"
  chroma_patched && ok "chroma patches" "${CHROMA_PATCHES[*]}" || fail "chroma patches" "not applied (run fetch on the login node)"
  for p in "${VALIDATE_INPUTS[@]}"; do
    [[ -f "${INPUTS}/${p}" ]] && ok "input ${p%%.ini.xml}" "" || fail "input ${p}" "missing in ${INPUTS}"
  done
  [[ -f "${CFG16}" ]] && ok "cfg 16^3x48" "${CFG16}" || fail "cfg 16^3x48" "missing ${CFG16}"
  [[ -f "${OPENBLAS}/lib/libopenblas.so" ]] && ok openblas "${OPENBLAS}" || fail openblas "missing"
  (( bad == 0 )) || die "preflight failed; nothing was built"
  printf -- '---- preflight passed\n'
}

# ---- build ----------------------------------------------------------------------------------------
# autoreconf a source tree once (marker in the tree's parent).
autoreconf_once() {
  local pkg=$1 dir=${SRC}/$1
  [[ -f "${SRC}/.autoreconf-${pkg}" ]] && return
  printf -- '---- autoreconf %s (%s)\n' "${pkg}" "$(now)"
  ( cd "${dir}" && autoreconf -fi ) || die "${pkg}: autoreconf failed"
  : > "${SRC}/.autoreconf-${pkg}"
}

# Configure + make + install one C library into PREFIX/<pkg>; strict first, then -fpermissive.
build_c_pkg() {
  local pkg=$1; shift
  local marker=${PREFIX}/.installed-${pkg} mode bld cflags rc
  if [[ -f "${marker}" ]]; then printf 'already installed: %s\n' "${pkg}"; return; fi
  autoreconf_once "${pkg}"
  for mode in strict permissive; do
    cflags="-O3 -march=znver5 -fopenmp -std=gnu17"
    [[ "${mode}" == permissive ]] && cflags+=" -fpermissive"
    bld=${BUILD}/${pkg}-${SLURM_JOB_ID}-${mode}
    mkdir -p "${bld}"
    printf -- '---- %s (%s): build in %s, CFLAGS=%s\n' "${pkg}" "$(now)" "${bld}" "${cflags}"
    printf 'build:      %s %s on %s, job %s, CFLAGS=%s, configure %s\n' "${pkg}" "${mode}" "$(hostname)" \
      "${SLURM_JOB_ID}" "${cflags}" "$*" >> "${PROVENANCE}"
    set +e
    ( cd "${bld}" && "${SRC}/${pkg}/configure" --prefix="${PREFIX}/${pkg}" CC="${MPICC}" CFLAGS="${cflags}" "$@" &&
      make -j "${JOBS}" && make install )
    rc=$?
    set -e
    if (( rc == 0 )); then
      printf 'installed:  %s (%s) %s\n' "${pkg}" "${mode}" "$(now)" >> "${PROVENANCE}"
      : > "${marker}"
      return
    fi
    printf 'failed:     %s (%s), rc %s\n' "${pkg}" "${mode}" "${rc}" >> "${PROVENANCE}"
    printf -- '---- %s: %s build failed (rc %s)\n' "${pkg}" "${mode}" "${rc}"
  done
  die "${pkg}: both the strict and the -fpermissive build failed (see the log above)"
}

build_chroma() {
  local marker=${PREFIX}/.installed-chroma bld xml_cflags xml_libs p
  if [[ -f "${marker}" ]]; then printf 'already installed: chroma\n'; return; fi
  chroma_patched || die "chroma: patches (${CHROMA_PATCHES[*]}) not applied to ${SRC}/chroma; run fetch on the login node"
  autoreconf_once chroma
  xml_cflags=$(xml2-config --cflags)
  xml_libs=$(xml2-config --libs)
  # C only: the bridge's own QOP_MG_CFLAGS lists QDP++'s include dir (it holds QIO) before QDP/C's, and
  # both have a qdp.h, so QDP/C's dirs must come first in CFLAGS. Never in CXXFLAGS: there QDP/C's qdp.h
  # shadows QDP++'s and every chroma file fails (job 1365306); chroma's C++ sees only qla.h through
  # wilsonmg-interface.h, and QOP_MG_CXXFLAGS already carries -I<qla>/include.
  local inc="-I${PREFIX}/qopqdp/include -I${PREFIX}/qdp/include -I${PREFIX}/qla/include"
  # Static link: chroma's QOP_MG_LIBS names qdp_common/qdp_int and qla_c99/qla_cmath before the libraries
  # that use them; QDP/C's own order (qdp-config, all colours) is appended through LIBS for a second pass.
  local qdpc_libs
  qdpc_libs="$("${PREFIX}/qdp/bin/qdp-config" --ldflags) $("${PREFIX}/qdp/bin/qdp-config" --libs)" ||
    die "chroma: qdp-config failed"
  local cmd=(
    "${SRC}/chroma/configure" --prefix="${PREFIX}/${CHROMA_NAME}"
    --with-qdp="${BASE_PREFIX}/qdpxx-double-nd4" --with-qmp="${QMP_DIR}"
    --enable-static-packed-gauge --enable-fused-clover-deriv-loops
    --with-qphix-solver="${BASE_PREFIX}/qphix-qdpxx-double-nd4-avx512" --enable-qphix-solver-arch=avx512
    --enable-qphix-solver-soalen=4 --disable-qphix-solver-compress12 --enable-qphix-solver-inner-type=f
    --enable-qphix-solver-inner-soalen=8
    --with-mg-proto="${BASE_PREFIX}/mgproto-qphix-qdpxx-double-nd4-avx512"
    --with-primme="${BASE_PREFIX}/primme-cpu"
    --enable-qop-mg --with-qla="${PREFIX}/qla" --with-qdpc="${PREFIX}/qdp" --with-qopqdp="${PREFIX}/qopqdp"
    --with-qio="${QIO_DIR}"
    CC="${MPICC}" CXX="${MPICXX}"
    CFLAGS="-O3 -march=znver5 -fopenmp -std=gnu17 ${xml_cflags} -I${BASE_PREFIX}/primme-cpu/include ${inc}"
    CXXFLAGS="-O3 -march=znver5 -fopenmp -std=c++14 ${xml_cflags} -I${BASE_PREFIX}/primme-cpu/include"
    LDFLAGS="-O3 -fopenmp -L${OPENBLAS}/lib -Wl,-rpath,${OPENBLAS}/lib"
    LIBS="-L${BASE_PREFIX}/primme-cpu/lib -lprimme -lopenblas ${xml_libs} ${qdpc_libs} -L${QMP_DIR}/lib -lqmp -L${QIO_DIR}/lib -lqio -llime -lm"
  )
  bld=${BUILD}/chroma-${SLURM_JOB_ID}
  mkdir -p "${bld}"
  {
    printf 'build:      chroma on %s, job %s, in %s\n  %s\n' "$(hostname)" "${SLURM_JOB_ID}" "${bld}" "${cmd[*]@Q}"
    printf '  CXX: %s\n' "$("${MPICXX}" --version | head -1)"
    printf '  source patched with: %s (checked present)\n' "${CHROMA_PATCHES[*]}"
  } >> "${PROVENANCE}"
  printf -- '---- chroma (%s): configure in %s\n' "$(now)" "${bld}"
  ( cd "${bld}" && "${cmd[@]}" ) || die "chroma: configure failed (${bld}/config.log)"
  grep -q 'define BUILD_QOP_MG' "${bld}/lib/chroma_config_internal.h" 2>/dev/null ||
    grep -rq 'BUILD_QOP_MG' "${bld}/lib/"*config*.h 2>/dev/null || die "chroma: configured without BUILD_QOP_MG"
  printf -- '---- chroma (%s): make -k -j%s\n' "$(now)" "${JOBS}"
  ( cd "${bld}" && make -k -j "${JOBS}" && make install ) || die "chroma: build failed (log above; -k lists every failing file)"
  [[ -x "${BIN_DIR}/hmc" && -x "${BIN_DIR}/chroma" ]] || die "chroma: no hmc/chroma in ${BIN_DIR}"
  # grep without -q reads all of nm's output: -q exits at the first match, nm then dies of SIGPIPE and pipefail fails the check (job 1365331)
  nm "${BIN_DIR}/hmc" | grep 'QOP_D3_wilsonMgSolve' > /dev/null || die "chroma: hmc does not contain QOPQDP's multigrid"
  for p in "${REUSE[@]}"; do [[ -e "${PREFIX}/${p}" ]] || ln -s "${BASE_PREFIX}/${p}" "${PREFIX}/${p}"; done
  [[ -e "${PREFIX}/bin" ]] || ln -s "${CHROMA_NAME}/bin" "${PREFIX}/bin"
  printf 'installed:  chroma %s  hmc sha256=%s  chroma sha256=%s\n' "$(now)" \
    "$(sha256sum "${BIN_DIR}/hmc" | cut -d' ' -f1)" "$(sha256sum "${BIN_DIR}/chroma" | cut -d' ' -f1)" >> "${PROVENANCE}"
  : > "${marker}"
}

build() {
  require_allocation
  load_env
  fence_network
  provenance_header
  printf 'build job:  %s on %s, job %s, script sha256=%s\n' "$(now)" "$(hostname)" "${SLURM_JOB_ID}" \
    "$(sha256sum "${SELF}" | cut -d' ' -f1)" >> "${PROVENANCE}"
  mkdir -p "${PREFIX}" "${BUILD}"
  build_c_pkg qla --enable-openmp
  build_c_pkg qdp --with-qmp="${QMP_DIR}" --with-qla="${PREFIX}/qla" --with-qio="${QIO_DIR}"
  build_c_pkg qopqdp --with-qdp="${PREFIX}/qdp" --with-qla="${PREFIX}/qla" --with-qio="${QIO_DIR}" --with-qmp="${QMP_DIR}"
  build_chroma
}

# ---- test / validate ------------------------------------------------------------------------------
QPHIX_ARGS=(-by 4 -bz 4 -pxy 1 -pxyz 0 -minct 1 -sy 1 -sz 1)

# start_results: fresh RUN_DIR, RESULTS, FAILED, RESULTS.txt and maxrss.txt.
start_results() {
  mkdir -p "${RUN_DIR}"
  RESULTS=()
  FAILED=0
  : > "${RUN_DIR}/RESULTS.txt"
  printf '# per srun step: exit code; sacct MaxRSS = largest task peak; sampled = ps RSS of the ranks every 10 s\n' \
    > "${RUN_DIR}/maxrss.txt"
}

# add_result LINE: kept for the summary and appended to RUN_DIR/RESULTS.txt at once, so a job that hits
# its time limit still leaves what finished.
add_result() {
  RESULTS+=("$1")
  printf '%s\n' "$1" >> "${RUN_DIR}/RESULTS.txt"
}

# rss_sampler COMM FILE: every 10 s, the number, summed RSS and largest RSS of the processes named COMM
# (the job has the node to itself, so these are the ranks of the running step).
rss_sampler() {
  local comm=$1 out=$2 t0=${SECONDS}
  printf '# elapsed_s ranks sum_rss_MiB max_rank_rss_MiB  (ps -C %s, every 10 s)\n' "${comm}" > "${out}"
  while sleep 10; do
    ps -C "${comm}" -o rss= 2>/dev/null | awk -v t=$(( SECONDS - t0 )) \
      '{ s += $1; if ($1 > m) m = $1; n++ } END { if (n) printf "%d %d %.0f %.0f\n", t, n, s / 1024, m / 1024 }' \
      >> "${out}" || true
  done
}

# rss_summary NAME DIR: sacct's record of step NAME (MaxRSS is the largest task's peak) and the sampled
# largest-rank RSS peak in the first and in the second half of the run (a per-solve leak makes the
# second half larger).
rss_summary() {
  local name=$1 file=$2/$1.rss.txt acct="" samp="" i
  for i in 1 2 3 4 5 6; do
    acct=$(sacct -j "${SLURM_JOB_ID}" -P --noheader --format=JobName,MaxRSS,MaxRSSTask,AveRSS,Elapsed,State 2>/dev/null |
      awk -F'|' -v n="${name}" '$1 == n && $2 != "" { r = $2 " (task " $3 "), AveRSS " $4 ", " $5 ", " $6 } END { print r }') || true
    [[ -n "${acct}" ]] && break
    sleep 5
  done
  samp=$(awk '!/^#/ { k++; t[k] = $1; m[k] = $4 }
    END { if (k == 0) exit
          h = t[k] / 2
          for (i = 1; i <= k; i++) { if (t[i] <= h) { if (m[i] > a) a = m[i] } else { if (m[i] > b) b = m[i] } }
          printf "largest rank %d MiB in the 1st half, %d MiB in the 2nd half (%d samples, %d s)", a, b, k, t[k] }' \
    "${file}" 2>/dev/null) || true
  printf 'sacct MaxRSS %s; sampled %s' "${acct:-n/a}" "${samp:-n/a}"
}

# run_exe NAME EXE RANKS THREADS INPUT GEOM [LIMIT_S] [DIR] -> DIR/NAME.{log,out.xml,xmldat,rss.txt}; the
# srun step is named NAME. Sets RSS_NOTE (also appended to RUN_DIR/maxrss.txt); returns the exit code.
run_exe() {
  local name=$1 exe=$2 ranks=$3 threads=$4 input=$5 geom=$6 limit=${7:-900} dir=${8:-${RUN_DIR}} rc sampler
  rss_sampler "$(basename "${exe}")" "${dir}/${name}.rss.txt" &
  sampler=$!
  set +e
  ( cd "${dir}" && OMP_NUM_THREADS=${threads} OMP_PLACES=cores OMP_PROC_BIND=close \
      timeout "${limit}" srun --mpi=pmix --job-name="${name}" --ntasks="${ranks}" --cpus-per-task="${threads}" \
      --cpu-bind=cores "${exe}" -i "${input}" -o "${name}.out.xml" -l "${name}.xmldat" -geom ${geom} \
      "${QPHIX_ARGS[@]}" -c "${threads}" > "${name}.log" 2>&1 )
  rc=$?
  kill "${sampler}" 2>/dev/null
  wait "${sampler}" 2>/dev/null
  set -e
  (( rc != 124 )) || printf '\n*** killed by timeout after %s s\n' "${limit}" >> "${dir}/${name}.log"
  RSS_NOTE=$(rss_summary "${name}" "${dir}")
  printf '%-16s rc=%-3s %s\n' "${name}" "${rc}" "${RSS_NOTE}" >> "${RUN_DIR}/maxrss.txt"
  return "${rc}"
}

# plaq_check LOG: sets PQ (QDP/C's "Average plaquette", printed with %g), PC (chroma's w_plaq of the same
# links) and PLAQ_OK (ok, MISMATCH or missing). Agreement to 5e-6 (6 significant digits); a layout mismatch
# between QDP++ and QDP/C scrambles the field and changes the plaquette grossly.
plaq_check() {
  PQ=$(grep -a -m1 'QDP:   Average plaquette' "$1" | awk '{print $NF}') || true
  PC=$(grep -a -m1 -A1 'Plaquette from State' "$1" | awk '/w_plaq/ {print $NF}') || true
  PLAQ_OK=$(awk -v a="${PQ:-x}" -v b="${PC:-y}" 'BEGIN { if (a == "x" || b == "y") print "missing"; else { d = a - b; if (d < 0) d = -d; print (d < 5e-6 ? "ok" : "MISMATCH") } }')
}

# cmp_records FILE1 FILE2 TAGS ABS_TAGS ABS_TOL REL_TOL: pairs the <TAG>value</TAG> records (TAGS is an
# alternation) of two runs in file order. Tags matching ABS_TAGS are compared by |difference|, the others
# by relative difference; a non-numeric value (AcceptP) must match exactly. Prints a summary that ends in
# ": AGREE" or ": DISAGREE".
cmp_records() {
  awk -v tags="$3" -v abstags="$4" -v atol="$5" -v rtol="$6" '
    FNR == 1 { f++ }
    { line = $0
      while (match(line, "<(" tags ")>[^<]*")) {
        tok = substr(line, RSTART + 1, RLENGTH - 1)
        line = substr(line, RSTART + RLENGTH)
        p = index(tok, ">")
        n[f]++
        key[f, n[f]] = substr(tok, 1, p - 1)
        val[f, n[f]] = substr(tok, p + 1)
      }
    }
    END {
      if (n[1] == 0 || n[1] != n[2]) { printf "%d vs %d records: DISAGREE", n[1], n[2]; exit }
      bad = 0; wa = 0; wr = 0; na = 0; nr = 0; ne = 0; dh = ""; note = ""
      for (i = 1; i <= n[1]; i++) {
        k = key[1, i]; a = val[1, i]; b = val[2, i]
        if (k != key[2, i]) { printf "record %d is %s vs %s: DISAGREE", i, k, key[2, i]; exit }
        if (a !~ /^[ \t]*[-+]?[0-9.]/) {
          ne++
          if (a != b) { bad = 1; note = note sprintf("; %s #%d %s vs %s", k, i, a, b) }
          continue
        }
        d = a - b; if (d < 0) d = -d
        if (k == "deltaH") dh = dh sprintf(" %.10g/%.10g", a, b)
        if (k ~ ("^(" abstags ")$")) { na++; if (d > wa) wa = d; if (d > atol + 0) bad = 1 }
        else { s = (b < 0) ? -b : b; r = (s > 0) ? d / s : d; nr++; if (r > wr) wr = r; if (r > rtol + 0) bad = 1 }
      }
      printf "%d records: %d absolute (max |diff| %.3g, tol %s), %d relative (max %.3g, tol %s), %d exact%s%s: %s",
        n[1], na, wa, atol, nr, wr, rtol, ne, (dh != "" ? "; deltaH" dh : ""), note, (bad ? "DISAGREE" : "AGREE")
    }' "$1" "$2" 2>&1
}

# Propagator: chroma ends normally (the QOP-MG LinOp solver aborts on a bad residual) and the plaquettes agree.
test_prop() {
  local name=$1 ranks=$2 threads=$3 geom=$4 rc=0 line
  run_exe "${name}" "${BIN_DIR}/chroma" "${ranks}" "${threads}" qopmg_prop_16x32.ini.xml "${geom}" || rc=$?
  plaq_check "${RUN_DIR}/${name}.log"
  if (( rc == 0 )) && grep -q 'CHROMA: ran successfully' "${RUN_DIR}/${name}.log" && [[ "${PLAQ_OK}" == ok ]]; then
    add_result "${name}: PASS  (${ranks} ranks x ${threads} threads, -geom ${geom}; plaquette QDP/C ${PQ} = chroma ${PC})"
  else
    add_result "${name}: FAIL  rc=${rc}, plaquette QDP/C '${PQ}' vs chroma '${PC}' (${PLAQ_OK}), see ${RUN_DIR}/${name}.log"
    FAILED=1
  fi
  while IFS= read -r line; do add_result "    ${line}"; done < <(
    grep -a -E 'QOPMG_LINOP_SOLVER:|QDP: Inversion took|Exiting|abort' "${RUN_DIR}/${name}.log" | tail -n 3 || true)
  add_result "    rss: ${RSS_NOTE}"
}

# HMC check: QOP-MG MdagM vs QPhiX CG, same seeds; deltaH, AcceptP and every force record must agree.
test_hmc() {
  local ranks=4 threads=16 geom="1 1 2 2" rc1=0 rc2=0 rss1 rss2 cmp
  local a=qopmg_hmc_check_16x32.ini.xml b=qphixcg_hmc_check_16x32.ini.xml
  # The two inputs may differ only inside <InvertParam> blocks.
  if ! diff <(awk '/<InvertParam>/{s=1} !s{print} /<\/InvertParam>/{s=0}' "${RUN_DIR}/${a}") \
            <(awk '/<InvertParam>/{s=1} !s{print} /<\/InvertParam>/{s=0}' "${RUN_DIR}/${b}") > /dev/null; then
    add_result "hmc_check: FAIL  the two inputs differ outside <InvertParam>"
    FAILED=1
    return
  fi
  run_exe hmc_qopmg "${BIN_DIR}/hmc" "${ranks}" "${threads}" "${a}" "${geom}" || rc1=$?
  rss1=${RSS_NOTE}
  run_exe hmc_qphixcg "${BIN_DIR}/hmc" "${ranks}" "${threads}" "${b}" "${geom}" || rc2=$?
  rss2=${RSS_NOTE}
  cmp=$(cmp_records "${RUN_DIR}/hmc_qopmg.xmldat" "${RUN_DIR}/hmc_qphixcg.xmldat" \
        'deltaH|F_sq|F_avg|F_max|AcceptP' deltaH 1e-3 1e-6) || true
  if (( rc1 == 0 && rc2 == 0 )) && [[ "${cmp}" == *": AGREE" ]]; then
    add_result "hmc_check: PASS  (QOP-MG vs QPhiX CG, ${ranks} ranks, -geom ${geom}): ${cmp}"
  else
    add_result "hmc_check: FAIL  rc ${rc1}/${rc2}: ${cmp}; see ${RUN_DIR}/hmc_qopmg.log, hmc_qphixcg.log"
    FAILED=1
  fi
  add_result "    rss hmc_qopmg:   ${rss1}"
  add_result "    rss hmc_qphixcg: ${rss2}"
}

test_qopmg() {
  require_allocation
  load_env
  [[ -x "${BIN_DIR}/hmc" && -x "${BIN_DIR}/chroma" ]] || die "no hmc/chroma in ${BIN_DIR}: run build first"
  RUN_DIR=${ROOT}/runs/$(date +%Y_%m_%d)_chroma_cpu_lq3_qopmg_smoke_${SLURM_JOB_ID}
  start_results
  local exe unresolved
  for exe in hmc chroma; do
    ldd "${BIN_DIR}/${exe}" > "${RUN_DIR}/ldd_${exe}.txt"
    unresolved=$(grep -c 'not found' "${RUN_DIR}/ldd_${exe}.txt" || true)
    if (( unresolved == 0 )); then
      add_result "ldd ${exe}: 0 unresolved"
    else
      add_result "ldd ${exe}: ${unresolved} unresolved"
      FAILED=1
    fi
  done
  cp "${HERE}/tests/qopmg_prop_16x32.ini.xml" "${HERE}/tests/qopmg_hmc_check_16x32.ini.xml" \
     "${HERE}/tests/qphixcg_hmc_check_16x32.ini.xml" "${RUN_DIR}/"

  test_prop prop_2r 2 32 "1 1 1 2"
  test_prop prop_4r 4 16 "1 1 2 2"
  test_hmc

  printf -- '---- QOP-MG test results (%s)\n' "${RUN_DIR}"
  printf '  %s\n' "${RESULTS[@]}"
  {
    printf '# chroma-cpu-lq3-qopmg: CPU Chroma with QOPQDP Wilson-clover multigrid (QOP_CLOVER_MULTIGRID_INVERTER)\n'
    printf 'tested:     %s on %s, job %s\n' "$(now)" "$(hostname)" "${SLURM_JOB_ID}"
    printf 'exe:        %s/hmc  sha256=%s\n' "${BIN_DIR}" "$(sha256sum "${BIN_DIR}/hmc" | cut -d' ' -f1)"
    printf 'runtime:    LQ_TARGET=cpu source grid-lqcd-workflow/machines/lq.sh; srun --mpi=pmix; QPhiX options needed\n'
    printf 'test:       %s\n' "${RESULTS[@]}"
    printf -- '--- build: %s ---\n' "${PROVENANCE}"
    cat "${PROVENANCE}"
  } > "${PREFIX}/PROVENANCE.txt"
  (( FAILED == 0 )) || die "a test failed (results in ${RUN_DIR}/RESULTS.txt)"
}

# prop16: the input agent's convention check on the real 16^3x48 configuration (see the header).
validate_prop16() {
  local name=prop16 rc=0 log rsd corr
  cp "${INPUTS}/qopmg_validate_prop16.ini.xml" "${RUN_DIR}/"
  run_exe "${name}" "${BIN_DIR}/chroma" 4 16 qopmg_validate_prop16.ini.xml "1 1 1 4" 2400 || rc=$?
  log=${RUN_DIR}/${name}.log
  plaq_check "${log}"
  # QOP-MG's relative residuals, computed with chroma's operator: 12 spin-colour solves, each < 5 x 1e-10.
  rsd=$(grep -a 'QOPMG_LINOP_SOLVER:' "${log}" | sed -n 's/.*Relative Rsd: *\([^ ]*\).*/\1/p' |
    awk '{ n++; if ($1 + 0 > m) m = $1 + 0 }
      END { printf "%d QOP-MG solves, max relative residual %.3g: %s", n, m, (n >= 12 && m < 5e-10) ? "ok" : "BAD" }') || true
  # prop_corr: the QPhiX reference first, QOP-MG second; relative difference on every timeslice.
  corr=$(tr '\n' ' ' < "${RUN_DIR}/${name}.out.xml" | grep -o '<prop_corr>[^<]*</prop_corr>' | sed 's/<[^>]*>//g' |
    awk '{ n++; nf[n] = NF; for (i = 1; i <= NF; i++) v[n, i] = $i }
      END { if (n != 2 || nf[1] != nf[2] || nf[1] == 0) { printf "%d prop_corr arrays (%d/%d entries): BAD", n, nf[1], nf[2]; exit }
            for (i = 1; i <= nf[1]; i++) {
              d = v[1, i] - v[2, i]; if (d < 0) d = -d
              s = v[1, i]; if (s < 0) s = -s
              r = (s > 0) ? d / s : d; if (r > m) m = r
            }
            printf "prop_corr QPhiX vs QOP-MG on %d timeslices, max relative difference %.3g: %s", nf[1], m, (m < 1e-6) ? "ok" : "BAD" }') || true
  if (( rc == 0 )) && grep -q 'CHROMA: ran successfully' "${log}" &&
     [[ "${PLAQ_OK}" == ok && "${rsd}" == *": ok" && "${corr}" == *": ok" ]]; then
    add_result "prop16: PASS  (QOP-MG vs QPhiX BiCGStab, 16^3x48 cfg_11100 at -0.2450, 4 ranks x 16 threads, -geom 1 1 1 4)"
  else
    add_result "prop16: FAIL  rc=${rc}; see ${log}"
    FAILED=1
  fi
  add_result "    ${rsd}"
  add_result "    plaquette QDP/C ${PQ:-?} vs chroma ${PC:-?}: ${PLAQ_OK}"
  add_result "    ${corr}"
  add_result "    rss: ${RSS_NOTE}"
}

# hmc16: the input agent's paired HMC check, each run in its own directory (both save to the same prefix).
validate_hmc16() {
  local q=hmc16_qopmg c=hmc16_qphixcg rc1=0 rc2=0 rss1 rss2 n1 n2 er cmp_md cmp_pl
  mkdir -p "${RUN_DIR}/${q}" "${RUN_DIR}/${c}"
  cp "${INPUTS}/qopmg_validate_hmc16_qopmg.ini.xml" "${RUN_DIR}/${q}/"
  cp "${INPUTS}/qopmg_validate_hmc16_qphixcg.ini.xml" "${RUN_DIR}/${c}/"
  run_exe "${q}" "${BIN_DIR}/hmc" 4 16 qopmg_validate_hmc16_qopmg.ini.xml "1 1 1 4" 5400 "${RUN_DIR}/${q}" || rc1=$?
  rss1=${RSS_NOTE}
  run_exe "${c}" "${BIN_DIR}/hmc" 4 16 qopmg_validate_hmc16_qphixcg.ini.xml "1 1 1 4" 2700 "${RUN_DIR}/${c}" || rc2=$?
  rss2=${RSS_NOTE}
  n1=$(grep -a -o '<deltaH>' "${RUN_DIR}/${q}/${q}.xmldat" 2>/dev/null | wc -l) || true
  n2=$(grep -a -o '<deltaH>' "${RUN_DIR}/${c}/${c}.xmldat" 2>/dev/null | wc -l) || true
  er=$(grep -a -c 'Erasing subspace' "${RUN_DIR}/${q}/${q}.log" 2>/dev/null) || true
  cmp_md=$(cmp_records "${RUN_DIR}/${q}/${q}.xmldat" "${RUN_DIR}/${c}/${c}.xmldat" \
           'deltaH|F_sq|F_avg|F_max|AcceptP' deltaH 1e-2 1e-5) || true
  cmp_pl=$(cmp_records "${RUN_DIR}/${q}/${q}.out.xml" "${RUN_DIR}/${c}/${c}.out.xml" w_plaq none 0 1e-8) || true
  if (( rc1 == 0 && rc2 == 0 && ${n1:-0} == 2 && ${n2:-0} == 2 )) &&
     [[ "${cmp_md}" == *": AGREE" && "${cmp_pl}" == *": AGREE" ]]; then
    add_result "hmc16: PASS  (QOP-MG vs QPhiX CG, 2 trajectories each, 4 ranks x 16 threads, -geom 1 1 1 4)"
  else
    add_result "hmc16: FAIL  rc ${rc1}/${rc2}, deltaH records ${n1:-0}/${n2:-0}; see ${RUN_DIR}/${q}/${q}.log, ${RUN_DIR}/${c}/${c}.log"
    FAILED=1
  fi
  add_result "    MD records: ${cmp_md}"
  add_result "    plaquettes: ${cmp_pl}"
  add_result "    QOP-MG subspace erasures (RefreshThreshold reached): ${er:-0}"
  add_result "    rss ${q}:   ${rss1}"
  add_result "    rss ${c}: ${rss2}"
}

validate_qopmg() {
  require_allocation
  load_env
  [[ -x "${BIN_DIR}/hmc" && -x "${BIN_DIR}/chroma" ]] || die "no hmc/chroma in ${BIN_DIR}: run build first"
  local f
  for f in "${VALIDATE_INPUTS[@]}"; do [[ -f "${INPUTS}/${f}" ]] || die "missing ${INPUTS}/${f}"; done
  [[ -f "${CFG16}" ]] || die "missing ${CFG16}"
  RUN_DIR=${ROOT}/runs/$(date +%Y_%m_%d)_chroma_cpu_lq3_qopmg_validate_${SLURM_JOB_ID}
  start_results
  for f in "${VALIDATE_INPUTS[@]}"; do
    add_result "input ${INPUTS}/${f}  sha256=$(sha256sum "${INPUTS}/${f}" | cut -d' ' -f1)"
  done

  validate_prop16
  validate_hmc16

  printf -- '---- QOP-MG validation results (%s)\n' "${RUN_DIR}"
  printf '  %s\n' "${RESULTS[@]}"
  {
    printf 'validated:  %s on %s, job %s, %s\n' "$(now)" "$(hostname)" "${SLURM_JOB_ID}" "${RUN_DIR}"
    printf 'validate:   %s\n' "${RESULTS[@]}"
  } >> "${PREFIX}/PROVENANCE.txt"
  (( FAILED == 0 )) || die "a validation check failed (results in ${RUN_DIR}/RESULTS.txt)"
}

print_paths
case "${ACTION}" in
  fetch)    fetch ;;
  check)    check ;;
  build)    build ;;
  test)     test_qopmg ;;
  validate) validate_qopmg ;;
  info)     ;;
  *) printf 'usage: %s {fetch|check|build|test|validate|info}\n' "$0" >&2; exit 2 ;;
esac
