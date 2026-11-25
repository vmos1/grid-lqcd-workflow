#!/usr/bin/env bash
# Build tools that lq3's compute nodes lack and the CPU Chroma build needs (build_chroma_cpu.sh):
# autoconf, automake, libtool, cmake and libxml2 (headers + xml2-config), into one prefix,
# installs/external/buildtools-lq3. Preflight of job 1364670 (2026-11-23): lq3 has m4 and perl but no
# cmake, autoreconf/autoconf/automake/aclocal or xml2-config, and the EL9 spack tree has none of them.
#
# Versions, from the publishers' sites only (user's approval, 2026-11-23: download once, login node):
#   autoconf 2.72   (2023-12) GNU ftp, .sig checked with gpgv against GNU's keyring (gnu-keyring.gpg)
#   automake 1.16.5 (2021-10) GNU ftp, same
#   libtool  2.4.7  (2022-03) GNU ftp, same
#   cmake    3.31.8 (2025-06) Kitware's GitHub release, sha256 against the release's SHA-256.txt.
#            Below 4: cmake 4 drops cmake_minimum_required < 3.5, and QPhiX/mg_proto ask for 3.1.
#   libxml2  2.11.9 (2024-07) download.gnome.org, sha256 against its .sha256sum. The 2.11 line, as the
#            group's EL8 qdpxx build (EasyBuild 2.11.4); QDP++ uses only long-stable libxml2 calls.
# Downloads are untrusted data: "fetch" only downloads and verifies (no extraction, nothing run) and
# records URL, version, sha256 and the check in builds/buildtools-lq3/PROVENANCE.txt. The job
# re-checks each tarball against that recorded sha256 before extracting it.
#
# Build (lq3 job, lq.sh LQ_TARGET=cpu: spack gcc 15.3, plain gcc/g++, no MPI), in this order:
#   autoconf -> automake (needs autoconf on PATH) -> libtool -> cmake (bootstrap, bundled third-party
#   libraries, no OpenSSL, -static-libstdc++ so it runs without gcc 15's runtime on the library path)
#   -> libxml2 (static only, -fPIC, no python/http/html/lzma/zlib: chromaform's own libxml2 flags).
# C code gets -std=gnu17 (GCC 15 defaults to gnu23). Network fenced off as in build_chroma_cpu.sh.
# Each tool is installed once (marker PREFIX/.installed-<tool>-<version>); each attempt builds in a
# fresh folder build/<tool>-<version>-<jobid>, so a failed attempt never has to be deleted first.
#
# Actions:
#   fetch      login node: download + verify into builds/downloads/, record URL, version, sha256
#   preflight  lq3 job: compiler, make, perl (+ modules autoconf/automake use), m4, xz, tarballs
#   build      lq3 job: extract and build the five tools (COMPUTE NODE ONLY)
#   test       lq3 job: every tool runs (--version), cmake's ldd, xml2-config; writes PREFIX/PROVENANCE.txt
#   info       print paths
# Job wrapper: build_buildtools_job.sh beside this file.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKFLOW=$(cd "${HERE}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)
SELF=${HERE}/$(basename "${BASH_SOURCE[0]}")

ACTION=${1:-info}
JOBS=${JOBS:-64}
DL=${ROOT}/builds/downloads
STAGE=${STAGE:-${ROOT}/builds/buildtools-lq3}
PREFIX=${PREFIX:-${ROOT}/installs/external/buildtools-lq3}
PROVENANCE=${STAGE}/PROVENANCE.txt
GNU_KEYRING_URL=https://ftp.gnu.org/gnu/gnu-keyring.gpg
GNU_KEYRING=${DL}/gnu-keyring.gpg

TOOLS=(autoconf automake libtool cmake libxml2)
declare -A VER=([autoconf]=2.72 [automake]=1.16.5 [libtool]=2.4.7 [cmake]=3.31.8 [libxml2]=2.11.9)

now() { date '+%Y-%m-%d %H:%M:%S %Z'; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
first_line() { local out; out=$("$@" 2>&1) || return 1; printf '%s\n' "${out%%$'\n'*}"; }

url_of() {
  local t=$1 v=${VER[$1]}
  case ${t} in
    autoconf|automake|libtool) printf 'https://ftp.gnu.org/gnu/%s/%s-%s.tar.xz\n' "${t}" "${t}" "${v}" ;;
    cmake)   printf 'https://github.com/Kitware/CMake/releases/download/v%s/cmake-%s.tar.gz\n' "${v}" "${v}" ;;
    libxml2) printf 'https://download.gnome.org/sources/libxml2/%s/libxml2-%s.tar.xz\n' "${v%.*}" "${v}" ;;
  esac
}
# The publisher's verification file for each tarball.
check_url_of() {
  local t=$1 v=${VER[$1]}
  case ${t} in
    autoconf|automake|libtool) printf '%s.sig\n' "$(url_of "${t}")" ;;
    cmake)   printf 'https://github.com/Kitware/CMake/releases/download/v%s/cmake-%s-SHA-256.txt\n' "${v}" "${v}" ;;
    libxml2) printf 'https://download.gnome.org/sources/libxml2/%s/libxml2-%s.sha256sum\n' "${v%.*}" "${v}" ;;
  esac
}
tarball_of() { printf '%s/%s\n' "${DL}" "$(basename "$(url_of "$1")")"; }

print_paths() {
  printf 'DL=%s\nSTAGE=%s\nPREFIX=%s\nJOBS=%s\n' "${DL}" "${STAGE}" "${PREFIX}" "${JOBS}"
  local t
  for t in "${TOOLS[@]}"; do printf '  %-9s %-7s %s\n' "${t}" "${VER[$t]}" "$(url_of "${t}")"; done
}

provenance_header() {
  [[ -f "${PROVENANCE}" ]] && return
  mkdir -p "${STAGE}"
  {
    printf '# Provenance of the lq3 build tools (1_build_chroma/lq/build_buildtools.sh)\n'
    printf 'created:    %s on %s\n' "$(now)" "$(hostname)"
    printf 'workflow:   %s (%s)\n' "$(git -C "${WORKFLOW}" rev-parse --short HEAD)" "$(git -C "${WORKFLOW}" rev-parse --abbrev-ref HEAD)"
    printf 'script:     %s  sha256=%s\n' "${SELF}" "$(sha256sum "${SELF}" | cut -d' ' -f1)"
    printf 'PREFIX:     %s\n' "${PREFIX}"
  } > "${PROVENANCE}"
}

# The publisher's check (login node, at fetch). Prints the evidence; reads the files, runs nothing
# from them.
verify_publisher() {
  local t=$1 tarball chk out
  tarball=$(tarball_of "${t}")
  chk=${DL}/$(basename "$(check_url_of "${t}")")
  [[ -f "${tarball}" && -f "${chk}" ]] || die "${t}: ${tarball} or ${chk} missing"
  case ${t} in
    autoconf|automake|libtool)
      out=$(gpgv --keyring "${GNU_KEYRING}" "${chk}" "${tarball}" 2>&1) || die "${t}: signature check failed: ${out}"
      printf '%s\n' "${out}" | grep -E 'using|Good signature' | sed 's/^gpgv: //' | paste -sd ';' -
      ;;
    cmake|libxml2)
      local name expect got
      name=$(basename "${tarball}")
      expect=$(awk -v n="${name}" '$2 == n || $2 == "*"n {print $1}' "${chk}")
      [[ -n "${expect}" ]] || die "${t}: ${name} not listed in ${chk}"
      got=$(sha256sum "${tarball}" | cut -d' ' -f1)
      [[ "${got}" == "${expect}" ]] || die "${t}: sha256 ${got} != published ${expect}"
      printf 'sha256 equals the one published in %s\n' "$(basename "${chk}")"
      ;;
  esac
}

# In the job: the tarball must have the sha256 recorded at fetch, after the publisher's check.
verify_recorded() {
  local t=$1 want got
  want=$(awk -v t="${t}" -v v="${VER[$t]}" '$1 == "tarball:" && $2 == t && $3 == v {sub("sha256=", "", $5); print $5}' \
    "${PROVENANCE}" 2>/dev/null)
  [[ -n "${want}" ]] || { printf 'no fetch record for %s %s in %s\n' "${t}" "${VER[$t]}" "${PROVENANCE}"; return 1; }
  got=$(sha256sum "$(tarball_of "${t}")" 2>/dev/null | cut -d' ' -f1)
  [[ "${got}" == "${want}" ]] || { printf 'sha256 %s, recorded %s\n' "${got:-missing}" "${want}"; return 1; }
  printf 'sha256 as recorded at fetch\n'
}

fetch() {
  [[ -z "${SLURM_JOB_ID:-}" ]] || die "fetch runs on the login node"
  mkdir -p "${DL}"
  provenance_header
  local t url f evidence
  if [[ ! -f "${GNU_KEYRING}" ]]; then
    printf 'downloading %s\n' "${GNU_KEYRING_URL}"
    curl -fsSL --proto '=https' -o "${GNU_KEYRING}" "${GNU_KEYRING_URL}"
    printf 'keyring:    %s  sha256=%s  fetched %s\n' "${GNU_KEYRING_URL}" \
      "$(sha256sum "${GNU_KEYRING}" | cut -d' ' -f1)" "$(now)" >> "${PROVENANCE}"
  fi
  for t in "${TOOLS[@]}"; do
    for url in "$(url_of "${t}")" "$(check_url_of "${t}")"; do
      f=${DL}/$(basename "${url}")
      if [[ -f "${f}" ]]; then
        printf 'present: %s\n' "${f}"
      else
        printf 'downloading %s\n' "${url}"
        curl -fsSL --proto '=https' -o "${f}" "${url}"
      fi
    done
    evidence=$(verify_publisher "${t}") ||
      die "${t}: verification failed; inspect $(tarball_of "${t}") (remove it yourself to fetch again)"
    printf 'verified %-9s %s\n' "${t}" "${evidence}"
    grep -qF "tarball:    ${t} ${VER[$t]} " "${PROVENANCE}" ||
      printf 'tarball:    %s %s  %s  sha256=%s\n  check:    %s: %s\n' "${t}" "${VER[$t]}" "$(url_of "${t}")" \
        "$(sha256sum "$(tarball_of "${t}")" | cut -d' ' -f1)" "$(check_url_of "${t}")" "${evidence}" >> "${PROVENANCE}"
  done
  printf 'fetched:    %s on %s\n' "$(now)" "$(hostname)" >> "${PROVENANCE}"
}

require_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "\"${ACTION}\" runs only inside a Slurm job on lq3 (SLURM_JOB_ID is unset)"
}

load_env() {
  LQ_TARGET=cpu source "${WORKFLOW}/machines/lq.sh"
  CC_BIN=$(command -v gcc)
  CXX_BIN=$(command -v g++)
}

preflight() {
  require_allocation
  load_env
  local bad=0 t m v
  ok()   { printf '  ok    %-16s %s\n' "$1" "$2"; }
  fail() { printf '  FAIL  %-16s %s\n' "$1" "$2"; bad=1; }
  printf -- '---- preflight on %s (%s), %s\n' "$(hostname)" "$(. /etc/os-release && echo "${PRETTY_NAME}")" "$(now)"
  ok gcc "${CC_BIN} ($(first_line "${CC_BIN}" --version))"
  ok g++ "${CXX_BIN}"
  for t in make perl m4 xz gzip tar; do
    command -v "${t}" >/dev/null 2>&1 && ok "${t}" "$(command -v "${t}")" || fail "${t}" "none on PATH"
  done
  # Modules autom4te, automake and libtoolize load (Data::Dumper is a separate RPM on EL9).
  for m in Data::Dumper Getopt::Long File::Basename File::Spec File::Temp File::Path File::Copy File::Compare \
           File::stat Cwd IO::File Text::ParseWords POSIX Carp Exporter Time::HiRes; do
    perl -M"${m}" -e 1 2>/dev/null && ok "perl ${m}" "" || fail "perl ${m}" "missing"
  done
  perl -Mthreads -MThread::Queue -e 1 2>/dev/null && ok "perl threads" "(automake -j)" ||
    printf '  note  %-16s %s\n' "perl threads" "absent: automake then runs serially, fine"
  for t in "${TOOLS[@]}"; do
    if v=$(verify_recorded "${t}"); then ok "${t} ${VER[$t]}" "${v}"; else fail "${t} ${VER[$t]}" "${v}"; fi
  done
  (( bad == 0 )) || die "preflight failed; nothing was built"
  printf -- '---- preflight passed\n'
}

build_tool() {
  local t=$1 v=${VER[$1]} src bld why
  local marker=${PREFIX}/.installed-${t}-${v}
  if [[ -f "${marker}" ]]; then printf 'already installed: %s %s\n' "${t}" "${v}"; return; fi
  why=$(verify_recorded "${t}") || die "${t}: ${why}"
  src=${STAGE}/src/${t}-${v}
  bld=${STAGE}/build/${t}-${v}-${SLURM_JOB_ID}
  if [[ ! -d "${src}" ]]; then
    mkdir -p "${STAGE}/src"
    tar -xf "$(tarball_of "${t}")" -C "${STAGE}/src"
    [[ -d "${src}" ]] || die "${t}: the tarball did not unpack to ${src}"
  fi
  mkdir -p "${bld}"
  printf -- '---- %s %s (%s)\n' "${t}" "${v}" "$(now)"
  printf 'build:      %s %s on %s, job %s, in %s\n' "${t}" "${v}" "$(hostname)" "${SLURM_JOB_ID}" "${bld}" >> "${PROVENANCE}"
  case ${t} in
    autoconf|automake|libtool)
      ( cd "${bld}" && "${src}/configure" --prefix="${PREFIX}" && make -j "${JOBS}" && make install )
      ;;
    cmake)
      ( cd "${bld}" &&
        LDFLAGS="-static-libstdc++ -static-libgcc" "${src}/bootstrap" --prefix="${PREFIX}" --parallel="${JOBS}" -- \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_USE_OPENSSL=OFF -DBUILD_TESTING=OFF \
          -DBUILD_CursesDialog=OFF -DBUILD_QtDialog=OFF &&
        make -j "${JOBS}" && make install )
      ;;
    libxml2)
      ( cd "${bld}" && "${src}/configure" --prefix="${PREFIX}" --disable-shared --enable-static --with-pic \
          --without-python --without-http --without-html --without-lzma --without-zlib &&
        make -j "${JOBS}" && make install )
      ;;
  esac
  : > "${marker}"
  printf 'installed:  %s %s  %s\n' "${t}" "${v}" "$(now)" >> "${PROVENANCE}"
}

build() {
  require_allocation
  load_env
  [[ -f "${PROVENANCE}" ]] || die "no ${PROVENANCE}: run fetch on the login node first"
  export CC=${CC_BIN} CXX=${CXX_BIN}
  export CFLAGS="-O2 -std=gnu17" CXXFLAGS="-O2"
  export PATH=${PREFIX}/bin:${PATH}          # automake's configure needs the new autoconf
  # No network: nothing here should download.
  export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 ftp_proxy=http://127.0.0.1:9
  export HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy} GIT_ALLOW_PROTOCOL=file
  unset no_proxy NO_PROXY
  printf 'toolchain:  CC=%s (%s) CXX=%s CFLAGS=%s CXXFLAGS=%s, make -j%s\n' "${CC}" \
    "$(first_line "${CC}" --version)" "${CXX}" "${CFLAGS}" "${CXXFLAGS}" "${JOBS}" >> "${PROVENANCE}"
  local t
  for t in "${TOOLS[@]}"; do build_tool "${t}"; done
}

test_tools() {
  require_allocation
  load_env
  local bad=0 t line unresolved
  local -a lines=()
  for t in autoconf autoreconf autoheader automake aclocal libtoolize cmake ctest xml2-config; do
    if line=$(first_line "${PREFIX}/bin/${t}" --version); then
      lines+=("${t}: ${line}")
    else
      lines+=("${t}: FAIL"); bad=1
    fi
  done
  unresolved=$(ldd "${PREFIX}/bin/cmake" | grep -c 'not found' || true)
  lines+=("cmake ldd: ${unresolved} unresolved")
  (( unresolved == 0 )) || bad=1
  lines+=("xml2-config --cflags: $("${PREFIX}/bin/xml2-config" --cflags)")
  lines+=("xml2-config --libs: $("${PREFIX}/bin/xml2-config" --libs)")
  if [[ -f "${PREFIX}/lib/libxml2.a" && -f "${PREFIX}/include/libxml2/libxml/xpath.h" ]]; then
    lines+=("libxml2: libxml2.a and headers present")
  else
    lines+=("libxml2: libxml2.a or headers MISSING"); bad=1
  fi
  printf -- '---- tools in %s\n' "${PREFIX}/bin"
  printf '  %s\n' "${lines[@]}"
  {
    printf '# buildtools-lq3: autoconf, automake, libtool, cmake, libxml2 for the lq3 CPU Chroma build\n'
    printf 'tested:     %s on %s, job %s\n' "$(now)" "$(hostname)" "${SLURM_JOB_ID}"
    printf 'use:        PATH=%s/bin:$PATH (build_chroma_cpu.sh does this)\n' "${PREFIX}"
    printf 'check:      %s\n' "${lines[@]}"
    printf -- '--- build: %s ---\n' "${PROVENANCE}"
    cat "${PROVENANCE}"
  } > "${PREFIX}/PROVENANCE.txt"
  (( bad == 0 )) || die "a tool failed its check (see above and ${PREFIX}/PROVENANCE.txt)"
}

print_paths
case "${ACTION}" in
  fetch)     fetch ;;
  preflight) preflight ;;
  build)     build ;;
  test)      test_tools ;;
  info)      ;;
  *) printf 'usage: %s {fetch|preflight|build|test|info}\n' "$0" >&2; exit 2 ;;
esac
