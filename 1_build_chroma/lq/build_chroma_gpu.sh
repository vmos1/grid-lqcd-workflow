#!/usr/bin/env bash
# GPU Chroma for lq2 (4 x A100-SXM4-80GB per node, sm_80, EL8): QMP + QDP-JIT (LLVM 18.1.8) + QUDA with
# multigrid + Chroma, built by chromaform from the source commits of Perlmutter's GPU Chroma, against the
# MPI stack our Grid uses on lq2 (gompi/2023a: Open MPI 4.1.5 + UCX 1.14.1 with the UCX-CUDA plugin).
#
# Why: the group's install (/lustre2/nplqcd/chroma/install_oct_2025) links NVHPC 23.11's Open MPI 3.1.5,
# which has no UCX and whose openib finds no HCA on lq, so its inter-node MPI runs over TCP (2.9 GB/s,
# 23 us; logs/chroma_baseg_lq_1365190.log NET line) and the 48^3 baseline takes 3,486 s per trajectory
# against Perlmutter's 1,652 s on the same 16 A100s. Grid on the same nodes with gompi matches Perlmutter.
#
# Recipe: Perlmutter's (inputs/perlmutter_ref/perlmutter_ref_48cube_chroma/chroma/build_chroma_pm.sh):
#   chromaform --mg --cuda --zen3 --std=c++20 --cmake=system --thrust=system --blas=<b> --libxml2=system
#              --source-dir=... --build-dir=... --install-dir=... llvm chroma CC=... CXX=... SM=sm_80
# with these lq changes, each deliberate:
#   toolchain  machines/lq.sh's gpu set (gompi/2023a = GCC 12.3.0 + Open MPI 4.1.5, cuda/12.2.1 through
#              ucx_cuda/1.14.1_cuda_12.2.1, ucc_cuda/1.2.0_cuda_12.2.1), plus cmake/3.27.2 (>= 3.24 for
#              CMAKE_PROJECT_TOP_LEVEL_INCLUDES; compute nodes have no system cmake) and openblas. One CUDA only.
#              Perlmutter: GNU 14.3, cray-mpich 9.0.1, CUDA 12.9, CMake 3.28.3.
#   MPI        CC/CXX = gompi's mpicc/mpicxx (GCC 12.3 underneath). The wrappers carry MPI into every compile
#              and link, as Perlmutter's Cray cc/CC did, and they add -rpath <OpenMPI 4.1.5 lib> with
#              --enable-new-dtags, so libmpi.so.40 resolves to gompi even without modules. QUDA's nvcc host
#              compiler is CMAKE_CXX_COMPILER = mpicxx. LLVM is built by chromaform with plain gcc/g++ (GCC 12.3).
#              The group's lq2.sh used CC=gcc CXX=g++ with an MPI module, which left the MPI choice to
#              find_package and to whatever libmpi came first: NVHPC's 3.1.5, through its RPATH.
#   BLAS       --blas=openblas-system: EasyBuild's OpenBLAS 0.3.24 (GCC 12.3.0, same toolchain as gompi),
#              linked by LIBS=-lopenblas with -rpath. Perlmutter used OpenBLAS (0.3.28, built by chromaform,
#              a further download); the group used MKL (imkl/2023.1.0 exists on lq el8, but it would put
#              Intel's runtime beside GNU OpenMP for no gain). BLAS is off the HMC path: QUDA and Chroma do not
#              use the BLA_* settings (both report them unused; Chroma builds DUMMY qdp-lapack bindings), the
#              library is only on Chroma's link line. This OpenBLAS is OpenMP-threaded and ignores
#              OPENBLAS_NUM_THREADS: every run must set OMP_NUM_THREADS (the lq2 launcher does).
#   CCCL       QUDA b6998853 pulls CCCL (Thrust/CUB/libcudacxx) 3.1.4 with CPM at configure time (Perlmutter's
#              build log: "CPM: Adding package CCCL@3.1.4"). Here it is fetched beforehand into src/cccl and
#              handed to CPM with -DCPM_CCCL_SOURCE (no download in the job). CUDA 12.2's own Thrust is 2.1.1.
#   phases     chromaform is run three times with the same flags, each run skipping what is installed:
#                1. llvm         (make -j JOBS_LLVM: LLVM compiles are light, the node has 40 cores)
#                2. llvm quda    (QMP, QDP-JIT nd3/nd4, QUDA; make -j JOBS: nvcc takes 2-5 GB per process)
#                3. llvm chroma  (Chroma; CMAKE_EXTRA_FLAGS also carries the inject file below)
#              Perlmutter's build also went in two steps: QUDA etc. without the inject file (job 57185716),
#              then Chroma alone with it (job 57217371). Keeping the inject file out of QUDA's configure keeps
#              its find_package(CCCL) from defining CCCL::CCCL before QUDA's own CPMAddPackage(CCCL).
#   inject     STAGE/inject_missing_cmake_targets.cmake (written by build; sha256 in PROVENANCE): QUDA's
#              exported QUDA::quda lists CCCL::CCCL and QMP's QMP::qmp lists MPI::MPI_C, and neither installed
#              config calls find_package for them, so Chroma's configure fails without it (Perlmutter BUILD
#              GUIDANCE 1). CCCL_DIR points at the CCCL 3.1.4 QUDA was built with (Perlmutter pointed at CUDA
#              12.9's CCCL); with Open MPI, find_package(MPI) gives the real MPI targets.
#   arch       --zen3 (-march=znver3), as Perlmutter and the group: lq2's CPUs. The build node is an Intel 6248
#              (lq1); nothing built with these flags runs during the build (LLVM is built without them; CUDA
#              is cross-compiled for SM=sm_80, no GPU needed).
#   offline    fetch stages every source in src/ (chromaform skips the download of any src/<pkg> that exists);
#              in the job the network is fenced off (dead proxy, git file:// only): a slip fails, never downloads.
#   make       MAKEFLAGS=-k, as the lq3 CPU build: chromaform retries a failed make three times at -j1; with -k
#              the first pass builds all it can and lists every failing file.
#
# Sources (each in its own new directory under src/; fetch records URL, commit or sha256 in PROVENANCE.txt).
# Code from them is untrusted: fetch clones, checks out and unpacks only; nothing in them runs on the login
# node. In the job chromaform runs the build tools the build needs (cmake, make, qdp-jit's pack_libdevice.sh).
#   chromaform  https://github.com/orginos/chromaform.git            5f55055743cde91b0c003056a4eb41bbabbce0b4
#   qmp         https://github.com/usqcd-software/qmp.git            c7444a2990802138e50d6d7fcc0cadefcb85d9cc
#   qdp-jit     https://github.com/JeffersonLab/qdp-jit.git          5279034fa66c23a24a00e50113af9b1b5ad7abe9 (+ submodules)
#   quda        https://github.com/lattice/quda.git                  b6998853f6b605e22d67ea2ddfa3cab0d752679a
#   chroma      https://github.com/JeffersonLab/chroma.git           f571838545a67202141e7a6e4d39d2d233ab5676 (+ submodules)
#   llvm        llvm-project-18.1.8.src.tar.xz (GitHub release llvmorg-18.1.8, chromaform's URL)
#   cpm         CPM.cmake v0.40.2 tag archive (chromaform's URL)
#   cccl        https://github.com/NVIDIA/cccl.git tag v3.1.4 (the version QUDA b6998853 asks CPM for)
#   eigen       builds/downloads/eigen-3.4.0.tar.bz2 (local copy of chromaform's URL, sha256 checked)
# The submodule commits must equal those Perlmutter's build checked out (its build log).
#
# Layout:
#   builds/chroma-gpu-lq2/src/<pkg>        sources (fetch); src/archives/<pkg>/ the downloaded archives
#   builds/chroma-gpu-lq2/build/           chromaform's build trees; logs/ the chromaform output per phase
#   builds/chroma-gpu-lq2/PROVENANCE.txt   fetch and build records
#   installs/external/chroma-gpu-lq2/      chromaform's install folders, bin -> the chroma package's bin/,
#                                          PROVENANCE.txt (commits, modules, flags, sha256 of bin/hmc, ldd)
# chromaform itself runs "rm -rf" on a package's build and install folder before building it, and on a failed
# package's install folder when it exits; both lie under the two folders above.
#
# Actions:
#   fetch   login node: clone + pinned checkout (+ submodules), download + unpack the archives; provenance
#   check   job (or login node, read only): EL8, modules and their paths, cmake >= 3.24, no llvm-config on
#           PATH, gmp/libxml2/openblas, every source at its pinned commit. Builds nothing.
#   build   job (COMPUTE NODE ONLY): the three chromaform phases; never overwrites an installed hmc
#   accept  job or login node: ldd/readelf acceptance of bin/hmc in the run-time environment of the lq2
#           launcher (env_chroma_lq.sh with CHROMA_STACK=gpu-lq2); appends the result to the install's
#           PROVENANCE.txt, ldd and readelf output in PREFIX/accept/<time>-<job>/. libmpi.so.40 must be
#           gompi's Open MPI 4.1.5, nothing from NVHPC's comm_libs or Open MPI 3.1.5, libcuda.so.1 the
#           driver (never a toolkit stub), nothing unresolved. On a node without the NVIDIA driver
#           libcuda.so.1 and libnvidia-ml.so.1 are left to the lq2 launcher, which checks them on the GPU node.
#   info    print paths
# Job wrapper: build_chroma_gpu_job.sh beside this file.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKFLOW=$(cd "${HERE}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)
SELF=${HERE}/$(basename "${BASH_SOURCE[0]}")

ACTION=${1:-info}
JOBS=${JOBS:-24}             # phases 2-3 (nvcc 2-5 GB per process; lq1 nodes have 40 cores, 190 GB)
JOBS_LLVM=${JOBS_LLVM:-40}   # phase 1
STAGE=${ROOT}/builds/chroma-gpu-lq2
SRC=${STAGE}/src
ARCHIVES=${SRC}/archives
BUILD=${STAGE}/build
LOGS=${STAGE}/logs
PROVENANCE=${STAGE}/PROVENANCE.txt
PREFIX=${PREFIX:-${ROOT}/installs/external/chroma-gpu-lq2}
CHROMA_NAME=chroma-quda-qdp-jit-double-nd4-cmake      # chromaform's name for --mg --cuda (double, Nd 4)
BIN_DIR=${PREFIX}/${CHROMA_NAME}/bin
INJECT=${STAGE}/inject_missing_cmake_targets.cmake
ENV_CHROMA=${WORKFLOW}/5_studies/chroma_inputs/lq/env_chroma_lq.sh

# The build environment: machines/lq.sh's gpu modules, then these.
EXTRA_MODULES=(cmake/3.27.2 openblas/0.3.24_gcc_12.3.0)
WANT_NVCC=/srv/software/el8/x86_64/hpc/cuda/12.2.1/bin/nvcc
WANT_MPI=/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0
WANT_LIBMPI=${WANT_MPI}/lib/libmpi.so.40

GIT_PKGS=(chromaform qmp qdp-jit quda chroma)
declare -A URL=(
  [chromaform]=https://github.com/orginos/chromaform.git
  [qmp]=https://github.com/usqcd-software/qmp.git
  [qdp-jit]=https://github.com/JeffersonLab/qdp-jit.git
  [quda]=https://github.com/lattice/quda.git
  [chroma]=https://github.com/JeffersonLab/chroma.git
)
declare -A PIN=(
  [chromaform]=5f55055743cde91b0c003056a4eb41bbabbce0b4   # 2025-11-24 "Fix compilation with superb"
  [qmp]=c7444a2990802138e50d6d7fcc0cadefcb85d9cc          # devel, tag qmp2-5-3, 2021-06-10
  [qdp-jit]=5279034fa66c23a24a00e50113af9b1b5ad7abe9      # devel, 2026-07-05
  [quda]=b6998853f6b605e22d67ea2ddfa3cab0d752679a         # develop, 2026-08-17 (QUDA 1.1.0)
  [chroma]=f571838545a67202141e7a6e4d39d2d233ab5676       # devel, 2026-07-03
)
# Submodule commits Perlmutter's build checked out (build-57185716.log), path -> commit.
declare -A SUBMOD=(
  [qdp-jit:other_libs/filedb]=8430f70e2c1caebd275c34b76e8ac4e9b81eb416
  [qdp-jit:other_libs/qio]=114007142ea8d0c68fb89ef39ad92391b46fb595
  [qdp-jit:other_libs/qio/other_libs/c-lime]=924aa0fce27a6cd73d94c9adc4b995f00e13232b
  [qdp-jit:other_libs/xpath_reader]=05cbc2d6e03cfe32e88433d3a3df77c6fd69f244
  [chroma:other_libs/cpp_wilson_dslash]=5a14dd16ba57aeed0166a22158116be5e499a5aa
  [chroma:other_libs/qdp-lapack]=89976ee41f54689ae8aec218864f0234109ea7a0
  [chroma:other_libs/sse_wilson_dslash]=0f9e5ed76589ce5f4bdd314990999b818a7a6c22
  [chroma:other_libs/wilsonmg]=cad7edd92e5b6a6871856e9dbff9cf269e32440c
)
# Submodule URLs may only point at these GitHub organisations.
SUBMOD_URL_OK='^https://github\.com/(usqcd-software|JeffersonLab)/[A-Za-z0-9_.-]+(\.git)?$'

LLVM_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/llvm-project-18.1.8.src.tar.xz
LLVM_SIZE=132067260     # GitHub release API, asset size (the release carries no digest)
CPM_URL=https://github.com/cpm-cmake/CPM.cmake/archive/refs/tags/v0.40.2.tar.gz
CPM_GROUP_COPY=/lustre2/nplqcd/chroma/source_jul_2025/cpm/cmake/CPM.cmake   # the group's chromaform fetched the same URL
CCCL_URL=https://github.com/NVIDIA/cccl.git
CCCL_TAG=v3.1.4
EIGEN_ARCHIVE=${ROOT}/builds/downloads/eigen-3.4.0.tar.bz2
EIGEN_SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626

now() { date '+%Y-%m-%d %H:%M:%S %Z'; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

print_paths() {
  printf 'STAGE=%s\nSRC=%s\nBUILD=%s\nPREFIX=%s\nBIN_DIR=%s\nJOBS=%s JOBS_LLVM=%s\n' \
    "${STAGE}" "${SRC}" "${BUILD}" "${PREFIX}" "${BIN_DIR}" "${JOBS}" "${JOBS_LLVM}"
}

provenance_header() {
  [[ -f "${PROVENANCE}" ]] && return
  mkdir -p "${STAGE}"
  {
    printf '# Provenance of the lq2 GPU Chroma (1_build_chroma/lq/build_chroma_gpu.sh)\n'
    printf 'created:    %s on %s\n' "$(now)" "$(hostname)"
    printf 'workflow:   %s (%s%s)\n' "$(git -C "${WORKFLOW}" rev-parse --short HEAD)" \
      "$(git -C "${WORKFLOW}" rev-parse --abbrev-ref HEAD)" \
      "$([[ -n "$(git -C "${WORKFLOW}" status --porcelain --untracked-files=no)" ]] && echo ', dirty')"
    printf 'script:     %s  sha256=%s\n' "${SELF}" "$(sha256sum "${SELF}" | cut -d' ' -f1)"
    printf 'STAGE:      %s\nPREFIX:     %s\n' "${STAGE}" "${PREFIX}"
  } > "${PROVENANCE}"
}

# ---- fetch (login node) ---------------------------------------------------------------------------
fetch_git() {
  local pkg=$1 dir=${SRC}/$1 sha tarsha key path want got url
  if [[ -f "${SRC}/.fetched-${pkg}" ]]; then
    sha=$(git -C "${dir}" rev-parse HEAD)
    [[ "${sha}" == "${PIN[$pkg]}" ]] || die "${dir} is at ${sha}, pinned ${PIN[$pkg]}"
    printf 'already fetched: %-10s %s\n' "${pkg}" "${sha}"
    return
  fi
  if [[ -e "${dir}" ]]; then
    # An earlier fetch stopped after the clone: resume only in a clone of the same URL (every step below
    # is idempotent, and fsck checks the object store).
    [[ "$(git -C "${dir}" config --get remote.origin.url 2>/dev/null)" == "${URL[$pkg]}" ]] ||
      die "incomplete ${dir} is not a clone of ${URL[$pkg]}. Inspect it; to discard it yourself: rm -rf ${dir}"
    printf 'resuming %s in %s\n' "${pkg}" "${dir}"
  else
    printf 'cloning %s ...\n' "${URL[$pkg]}"
    GIT_TERMINAL_PROMPT=0 git clone --quiet --no-checkout --no-recurse-submodules "${URL[$pkg]}" "${dir}"
  fi
  sha=$(git -C "${dir}" rev-parse --verify --quiet "${PIN[$pkg]}^{commit}") ||
    die "${pkg}: ${PIN[$pkg]} is not a commit of ${URL[$pkg]}"
  git -C "${dir}" -c advice.detachedHead=false checkout --quiet --detach "${sha}"
  [[ "$(git -C "${dir}" rev-parse HEAD)" == "${PIN[$pkg]}" ]] || die "${pkg}: checkout did not land on ${PIN[$pkg]}"
  git -C "${dir}" fsck --no-dangling --no-progress >/dev/null 2>&1 || die "${pkg}: git fsck failed in ${dir}"
  if [[ -f "${dir}/.gitmodules" ]]; then
    # Only the organisations of the approved sources (or a relative URL, which resolves inside the
    # superproject's organisation); checked before anything is cloned.
    while read -r key url; do
      [[ "${url}" =~ ${SUBMOD_URL_OK} || "${url}" =~ ^\.\./[A-Za-z0-9_.-]+$ ]] ||
        die "${pkg}: submodule ${key} points at ${url}, outside the approved organisations"
    done < <(git -C "${dir}" config -f .gitmodules --get-regexp '\.url$')
    GIT_TERMINAL_PROMPT=0 git -C "${dir}" submodule --quiet update --init --recursive
    # Every submodule clone, nested ones included (qio's c-lime has the relative URL ../c-lime.git):
    # the URL it was really cloned from.
    while read -r url; do
      [[ "${url}" =~ ${SUBMOD_URL_OK} ]] || die "${pkg}: a submodule was cloned from ${url}, outside the approved organisations"
    done < <(git -C "${dir}" submodule --quiet foreach --recursive 'git config --get remote.origin.url')
  fi
  # Every expected submodule at Perlmutter's commit, and no others.
  local n_want=0 n_got
  for key in "${!SUBMOD[@]}"; do
    [[ "${key%%:*}" == "${pkg}" ]] || continue
    n_want=$(( n_want + 1 ))
    path=${key#*:}
    want=${SUBMOD[$key]}
    got=$(git -C "${dir}/${path}" rev-parse HEAD 2>/dev/null) || die "${pkg}: submodule ${path} missing"
    [[ "${got}" == "${want}" ]] || die "${pkg}: submodule ${path} at ${got}, Perlmutter's build had ${want}"
  done
  n_got=$(git -C "${dir}" submodule status --recursive | wc -l)
  [[ "${n_got}" == "${n_want}" ]] || die "${pkg}: ${n_got} submodules, expected ${n_want}"
  tarsha=$(git -C "${dir}" archive --format=tar "${sha}" | sha256sum | cut -d' ' -f1)
  {
    printf 'source:     %-10s %s  commit %s (%s; %s)  git-archive-tar sha256=%s  fsck ok\n' "${pkg}" "${URL[$pkg]}" \
      "${sha}" "$(git -C "${dir}" describe --tags --always "${sha}" 2>/dev/null)" \
      "$(git -C "${dir}" log -1 --format='%ad %s' --date=short)" "${tarsha}"
    git -C "${dir}" submodule status --recursive | sed 's/^/            submodule/'
  } >> "${PROVENANCE}"
  : > "${SRC}/.fetched-${pkg}"
  printf 'fetched: %-10s %s\n' "${pkg}" "${sha}"
}

# download URL into ARCHIVES/<pkg>/ (a new directory), record sha256; prints the file path
download() {
  local pkg=$1 url=$2 file=${ARCHIVES}/$1/${2##*/}
  [[ -e "${ARCHIVES}/${pkg}" ]] && die "${ARCHIVES}/${pkg} exists without a finished fetch of ${pkg}. To discard it yourself: rm -rf ${ARCHIVES}/${pkg}"
  mkdir -p "${ARCHIVES}/${pkg}"
  printf 'downloading %s ...\n' "${url}" >&2
  curl --fail --silent --show-error --location --proto '=https' --output "${file}" "${url}" ||
    die "${pkg}: download of ${url} failed"
  printf '%s\n' "${file}"
}

fetch_llvm() {
  local file sig sha size
  if [[ -f "${SRC}/.fetched-llvm" ]]; then printf 'already fetched: llvm\n'; return; fi
  [[ -e "${SRC}/llvm" ]] && die "incomplete ${SRC}/llvm. To discard it yourself: rm -rf ${SRC}/llvm ${ARCHIVES}/llvm"
  file=$(download llvm "${LLVM_URL}")
  sig=${file}.sig
  curl --fail --silent --show-error --location --proto '=https' --output "${sig}" "${LLVM_URL}.sig" ||
    die "llvm: download of ${LLVM_URL}.sig failed"
  size=$(stat -c %s "${file}")
  [[ "${size}" == "${LLVM_SIZE}" ]] || die "llvm: ${file} has ${size} bytes, the release lists ${LLVM_SIZE}"
  sha=$(sha256sum "${file}" | cut -d' ' -f1)
  # chromaform's own download: the archive's single top folder becomes src/llvm.
  mkdir -p "${SRC}/llvm"
  tar -xJf "${file}" -C "${SRC}/llvm" --strip-components=1 --no-same-owner
  [[ -f "${SRC}/llvm/llvm/CMakeLists.txt" && -d "${SRC}/llvm/lld" ]] || die "llvm: unpacked tree lacks llvm/ or lld/"
  # (grep -q on the files, never on a pipe: with pipefail an early grep exit fails the pipe)
  grep -q -s -E 'set\(LLVM_VERSION_MAJOR 18\)' "${SRC}/llvm/cmake/Modules/LLVMVersion.cmake" \
    "${SRC}/llvm/llvm/CMakeLists.txt" || die "llvm: unpacked tree is not LLVM 18"
  printf 'source:     llvm       %s  size %s (= release API)  sha256=%s  .sig sha256=%s (signature not verified: no LLVM release key here)\n' \
    "${LLVM_URL}" "${size}" "${sha}" "$(sha256sum "${sig}" | cut -d' ' -f1)" >> "${PROVENANCE}"
  : > "${SRC}/.fetched-llvm"
  printf 'fetched: llvm 18.1.8 (sha256 %s)\n' "${sha}"
}

fetch_cpm() {
  local file sha csha gsha
  if [[ -f "${SRC}/.fetched-cpm" ]]; then printf 'already fetched: cpm\n'; return; fi
  [[ -e "${SRC}/cpm" ]] && die "incomplete ${SRC}/cpm. To discard it yourself: rm -rf ${SRC}/cpm ${ARCHIVES}/cpm"
  file=$(download cpm "${CPM_URL}")
  sha=$(sha256sum "${file}" | cut -d' ' -f1)
  mkdir -p "${SRC}/cpm"
  tar -xzf "${file}" -C "${SRC}/cpm" --strip-components=1 --no-same-owner
  [[ -f "${SRC}/cpm/cmake/CPM.cmake" ]] || die "cpm: no cmake/CPM.cmake in the archive"
  csha=$(sha256sum "${SRC}/cpm/cmake/CPM.cmake" | cut -d' ' -f1)
  gsha=$(sha256sum "${CPM_GROUP_COPY}" 2>/dev/null | cut -d' ' -f1) || gsha=unreadable
  printf 'source:     cpm        %s  sha256=%s  cmake/CPM.cmake sha256=%s (%s the group copy %s)\n' "${CPM_URL}" \
    "${sha}" "${csha}" "$([[ "${csha}" == "${gsha}" ]] && echo equals || echo DIFFERS FROM)" "${CPM_GROUP_COPY}" >> "${PROVENANCE}"
  : > "${SRC}/.fetched-cpm"
  printf 'fetched: cpm v0.40.2 (sha256 %s; CPM.cmake %s the group copy)\n' "${sha}" \
    "$([[ "${csha}" == "${gsha}" ]] && echo equals || echo DIFFERS FROM)"
}

fetch_cccl() {
  local dir=${SRC}/cccl sha
  if [[ -f "${SRC}/.fetched-cccl" ]]; then printf 'already fetched: cccl %s\n' "$(git -C "${dir}" rev-parse HEAD)"; return; fi
  [[ -e "${dir}" ]] && die "incomplete ${dir}. To discard it yourself: rm -rf ${dir}"
  printf 'cloning %s at %s (shallow) ...\n' "${CCCL_URL}" "${CCCL_TAG}"
  GIT_TERMINAL_PROMPT=0 git -c advice.detachedHead=false clone --quiet --depth 1 --branch "${CCCL_TAG}" \
    --no-recurse-submodules "${CCCL_URL}" "${dir}"
  sha=$(git -C "${dir}" rev-parse HEAD)
  [[ "$(git -C "${dir}" rev-parse "refs/tags/${CCCL_TAG}^{commit}" 2>/dev/null)" == "${sha}" ]] ||
    die "cccl: HEAD ${sha} is not tag ${CCCL_TAG}"
  [[ -f "${dir}/lib/cmake/cccl/cccl-config.cmake" ]] || die "cccl: no lib/cmake/cccl/cccl-config.cmake"
  printf 'source:     cccl       %s  tag %s -> commit %s  git-archive-tar sha256=%s (shallow clone)\n' "${CCCL_URL}" \
    "${CCCL_TAG}" "${sha}" "$(git -C "${dir}" archive --format=tar "${sha}" | sha256sum | cut -d' ' -f1)" >> "${PROVENANCE}"
  : > "${SRC}/.fetched-cccl"
  printf 'fetched: cccl %s (%s)\n' "${CCCL_TAG}" "${sha}"
}

fetch_eigen() {
  if [[ -f "${SRC}/.fetched-eigen" ]]; then printf 'already fetched: eigen\n'; return; fi
  [[ -e "${SRC}/eigen" ]] && die "incomplete ${SRC}/eigen. To discard it yourself: rm -rf ${SRC}/eigen"
  printf '%s  %s\n' "${EIGEN_SHA256}" "${EIGEN_ARCHIVE}" | sha256sum --check --quiet || die "eigen: sha256 mismatch"
  mkdir -p "${SRC}/eigen"
  tar -xjf "${EIGEN_ARCHIVE}" -C "${SRC}/eigen" --strip-components=1 --no-same-owner
  [[ -f "${SRC}/eigen/Eigen/Core" ]] || die "eigen: no Eigen/Core"
  printf 'source:     eigen      3.4.0 from %s  sha256=%s (chromaform URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2)\n' \
    "${EIGEN_ARCHIVE}" "${EIGEN_SHA256}" >> "${PROVENANCE}"
  : > "${SRC}/.fetched-eigen"
  printf 'fetched: eigen 3.4.0\n'
}

fetch() {
  [[ -z "${SLURM_JOB_ID:-}" ]] || die "fetch runs on the login node"
  mkdir -p "${SRC}" "${ARCHIVES}"
  provenance_header
  local pkg
  for pkg in "${GIT_PKGS[@]}"; do fetch_git "${pkg}"; done
  fetch_llvm
  fetch_cpm
  fetch_cccl
  fetch_eigen
  printf 'fetched:    %s on %s, script sha256=%s\n' "$(now)" "$(hostname)" "$(sha256sum "${SELF}" | cut -d' ' -f1)" >> "${PROVENANCE}"
  printf 'All sources in %s. Next: the lq1 job build_chroma_gpu_job.sh (check, build, accept).\n' "${SRC}"
}

# ---- environment ----------------------------------------------------------------------------------
require_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "\"${ACTION}\" runs only inside a Slurm job (SLURM_JOB_ID is unset)"
}

load_env() {
  LQ_TARGET=gpu LQ_MPI_STACK=gompi source "${WORKFLOW}/machines/lq.sh" || die "machines/lq.sh (gpu, gompi) failed"
  set +u
  module load "${EXTRA_MODULES[@]}" || { set -u; die "module load ${EXTRA_MODULES[*]} failed"; }
  set -u
  MPICC=${EBROOTOPENMPI:-}/bin/mpicc
  MPICXX=${EBROOTOPENMPI:-}/bin/mpicxx
  OPENBLAS=${EBROOTOPENBLAS:-}
  export CUDADIR=${CUDA_HOME:-}
}

fence_network() {
  export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 ftp_proxy=http://127.0.0.1:9
  export HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy} GIT_ALLOW_PROTOCOL=file
  unset no_proxy NO_PROXY
}

check() {
  local bad=0 v t pkg os
  ok()   { printf '  ok    %-16s %s\n' "$1" "$2"; }
  fail() { printf '  FAIL  %-16s %s\n' "$1" "$2"; bad=1; }
  os=$(. /etc/os-release && echo "${PRETTY_NAME} (VERSION_ID ${VERSION_ID})")
  printf -- '---- preflight on %s, %s, %s\n' "$(hostname)" "${os}" "$(now)"
  printf '  cpu   %s, %s CPUs, %s GB\n' "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)" "$(nproc)" \
    "$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)"
  # EL8 first: the EL9 lq3 nodes cannot run this toolchain, and their libraries would not run on lq2.
  if grep -q '^VERSION_ID="8' /etc/os-release && [[ -d /srv/software/el8/x86_64 ]]; then
    ok os "${os}, /srv/software/el8 present"
  else
    die "not an EL8 node with /srv/software/el8 (${os}); lq2 runs EL8: build on lq1 (EL8) or lq2"
  fi
  load_env
  printf '  modules: %s\n' "$(module -t list 2>&1 | tr '\n' ' ')"
  [[ "$(command -v nvcc)" == "${WANT_NVCC}" ]] && ok nvcc "$(nvcc --version | tail -1)" || fail nvcc "$(command -v nvcc || echo none), want ${WANT_NVCC}"
  [[ "${MPICC}" == "${WANT_MPI}/bin/mpicc" && -x "${MPICC}" && -x "${MPICXX}" ]] &&
    ok mpicc/mpicxx "${MPICXX} -> $("${MPICXX}" --showme:command 2>/dev/null) ($("${MPICXX}" --version | head -1))" ||
    fail mpicc/mpicxx "'${MPICC}', want ${WANT_MPI}/bin/mpicc"
  [[ "$(mpirun --version 2>&1 | head -1)" == *"4.1.5"* ]] && ok mpirun "$(mpirun --version 2>&1 | head -1)" || fail mpirun "not Open MPI 4.1.5"
  v=$(gcc -dumpfullversion 2>/dev/null || true)
  [[ "${v}" == 12.3.0 ]] && ok gcc "$(command -v gcc) ${v}" || fail gcc "'${v}', want 12.3.0 (GCCcore)"
  if command -v cmake >/dev/null 2>&1; then
    v=$(cmake --version | awk 'NR==1 {print $3}')
    case ${v} in
      3.2[4-9].*|3.3[0-9].*) ok cmake "${v} ($(command -v cmake))" ;;
      *) fail cmake "${v}: needs >= 3.24 (CMAKE_PROJECT_TOP_LEVEL_INCLUDES) and < 4" ;;
    esac
  else
    fail cmake "none on PATH (module cmake/3.27.2)"
  fi
  # chromaform asks llvm-config for its version when choosing the QDP-JIT flavour: an LLVM 14-16 on PATH
  # would select the old qdp-jit (qdp-jit-prev-llvm17) and a download.
  if command -v llvm-config >/dev/null 2>&1; then
    fail llvm-config "$(command -v llvm-config) $(llvm-config --version) on PATH: chromaform would pick it up"
  else
    ok llvm-config "none on PATH (chromaform builds LLVM 18.1.8)"
  fi
  [[ -f "${CUDADIR}/nvvm/libdevice/libdevice.10.bc" ]] && ok libdevice "${CUDADIR}/nvvm/libdevice/libdevice.10.bc" ||
    fail libdevice "none under CUDADIR=${CUDADIR}"
  [[ -f /usr/include/gmpxx.h && -e /usr/lib64/libgmpxx.so ]] && ok gmp "/usr (gmpxx.h, libgmpxx.so)" || fail gmp "no gmpxx in /usr"
  pkg-config --exists libxml-2.0 && ok libxml2 "$(pkg-config --modversion libxml-2.0) ($(pkg-config --variable=prefix libxml-2.0))" ||
    fail libxml2 "pkg-config finds no libxml-2.0"
  [[ -f "${OPENBLAS}/lib/libopenblas.so" ]] && ok openblas "${OPENBLAS}" || fail openblas "no libopenblas.so under '${OPENBLAS}'"
  # xxd: qdp-jit's pack_libdevice.sh (run by chromaform) embeds CUDA's libdevice.10.bc with it.
  for t in make git tar xxd; do
    command -v "${t}" >/dev/null 2>&1 && ok "${t}" "$(command -v "${t}")" || fail "${t}" "none on PATH"
  done
  # python3 >= 3.6: LLVM 18's configure (LLVM_MINIMUM_PYTHON_VERSION).
  if python3 -c 'import sys; sys.exit(sys.version_info < (3, 6))' 2>/dev/null; then
    ok python3 "$(command -v python3) $(python3 --version 2>&1)"
  else
    fail python3 "no python3 >= 3.6 on PATH (LLVM's configure needs one)"
  fi
  for pkg in "${GIT_PKGS[@]}"; do
    if [[ -f "${SRC}/.fetched-${pkg}" && "$(git -C "${SRC}/${pkg}" rev-parse HEAD 2>/dev/null)" == "${PIN[$pkg]}" ]]; then
      ok "src ${pkg}" "${PIN[$pkg]}"
    else
      fail "src ${pkg}" "not fetched at ${PIN[$pkg]} (run fetch on the login node)"
    fi
  done
  for pkg in llvm cpm cccl eigen; do
    [[ -f "${SRC}/.fetched-${pkg}" ]] && ok "src ${pkg}" "fetched" || fail "src ${pkg}" "not fetched (run fetch on the login node)"
  done
  [[ -f "${SRC}/cccl/lib/cmake/cccl/cccl-config.cmake" ]] && ok cccl-config "${SRC}/cccl/lib/cmake/cccl" || fail cccl-config "missing"
  if [[ -e "${BIN_DIR}/hmc" ]]; then
    printf '  note  %-16s %s\n' install "${BIN_DIR}/hmc exists (build would refuse; accept can run)"
  else
    ok install "${PREFIX} (no hmc yet)"
  fi
  (( bad == 0 )) || die "preflight failed; nothing was built"
  printf -- '---- preflight passed\n'
}

# ---- build ----------------------------------------------------------------------------------------
write_inject() {
  cat > "${INJECT}" << 'EOF'
# Injected into chromaform's cmake configure of Chroma (phase 3 of build_chroma_gpu.sh) through
#   -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=<this file>     (CMake >= 3.24)
# Written by grid-lqcd-workflow/1_build_chroma/lq/build_chroma_gpu.sh; adapted from Perlmutter's
# inject_missing_cmake_targets.cmake (inputs/perlmutter_ref/perlmutter_ref_48cube_chroma/chroma/).
#
# QUDA b6998853's QUDATargets.cmake puts CCCL::CCCL in QUDA::quda's link interface (QUDA fetched CCCL 3.1.4
# with CPM), but its installed config only does find_dependency(CUDAToolkit). QMP's QMPTargets.cmake puts
# MPI::MPI_C in QMP::qmp's link interface, but QMPConfig.cmake never does find_dependency(MPI). Both configure
# inside their own build trees and break every downstream consumer: Chroma's configure fails with
# "The link interface of target QUDA::quda contains: CCCL::CCCL but the target was not found".
#
# CCCL_DIR is set on the command line to the CCCL 3.1.4 source tree QUDA was built with. QUDA links CCCL
# PRIVATE, so Chroma sees it as $<LINK_ONLY:CCCL::CCCL>: no include directories reach Chroma, and CCCL is
# header-only, so the interface stub below (Perlmutter's fallback) loses nothing either. The compilers are
# Open MPI's mpicc/mpicxx (gompi), so find_package(MPI) gives the real MPI targets; the MPI stubs are a
# fallback that loses nothing, since the wrappers carry MPI into every compile and link.
find_package(CCCL QUIET)
if(NOT TARGET CCCL::CCCL)
  message(STATUS "inject: find_package(CCCL) found nothing (CCCL_DIR=${CCCL_DIR}); interface stub")
  add_library(CCCL::CCCL INTERFACE IMPORTED GLOBAL)
else()
  message(STATUS "inject: CCCL::CCCL provided by find_package(CCCL) from ${CCCL_DIR} (version ${CCCL_VERSION})")
endif()

if(NOT TARGET MPI::MPI_C)
  find_package(MPI QUIET COMPONENTS C CXX)
endif()
if(NOT TARGET MPI::MPI_C)
  message(STATUS "inject: MPI::MPI_C not found; interface stub (the mpicc/mpicxx wrappers carry MPI)")
  add_library(MPI::MPI_C INTERFACE IMPORTED GLOBAL)
else()
  message(STATUS "inject: MPI::MPI_C available (MPI_C_LIBRARIES=${MPI_C_LIBRARIES})")
endif()
if(NOT TARGET MPI::MPI_CXX)
  add_library(MPI::MPI_CXX INTERFACE IMPORTED GLOBAL)
endif()
EOF
}

# run_phase NAME JN EXTRA_CMAKE PKG...: one chromaform run, output to the job log and LOGS/NAME-<job>.log.
run_phase() {
  local name=$1 jn=$2 extra=$3 log rc
  shift 3
  log=${LOGS}/${name}-${SLURM_JOB_ID}.log
  local cmd=(
    "${SRC}/chromaform/chromaform" --mg --cuda --zen3 --std=c++20
    --cmake=system --thrust=system --blas=openblas-system --libxml2=system
    --source-dir="${SRC}" --build-dir="${BUILD}" --install-dir="${PREFIX}"
    "$@" CC="${MPICC}" CXX="${MPICXX}" SM=sm_80
  )
  {
    printf 'phase:      %s  %s on %s, job %s, MAKE_JN=%s\n' "${name}" "$(now)" "$(hostname)" "${SLURM_JOB_ID}" "${jn}"
    printf '  %s\n' "${cmd[*]@Q}"
    printf '  CMAKE_EXTRA_FLAGS=%s\n  LDFLAGS=%s\n  LIBS=%s\n  log: %s\n' "${extra}" "${LDFLAGS}" "${LIBS}" "${log}"
  } >> "${PROVENANCE}"
  printf -- '---- phase %s (%s): %s\n' "${name}" "$(now)" "${cmd[*]}"
  set +e
  ( cd "${BUILD}" && MAKE_JN=${jn} CMAKE_EXTRA_FLAGS=${extra} "${cmd[@]}" ) 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  # chromaform's resolved settings (compilers, flags, BLAS, CUDADIR), as it printed them.
  # Plus the inject file's verdicts, QUDA's version line and CPM's CCCL line, where this phase printed them.
  grep -a -E '^#> Using (CUDADIR|cmake|llvm|thrust|BLAS|GMP|libxml2|CC|CFLAGS|CXX|CXXFLAGS|LDFLAGS|LIBS)[:=]|^-- inject:|^-- QUDA 1\.|^-- CPM: Adding package|^-- Found CCCL' \
    "${log}" | sed 's/^/  /' >> "${PROVENANCE}" || true
  printf '  phase %s: exit %s at %s\n' "${name}" "${rc}" "$(now)" >> "${PROVENANCE}"
  (( rc == 0 )) || die "chromaform phase ${name} failed (exit ${rc}); log ${log}. A re-run resumes: installed packages are skipped"
}

build() {
  require_allocation
  [[ -e "${BIN_DIR}/hmc" ]] && die "${BIN_DIR}/hmc exists: an install is never overwritten (choose another PREFIX)"
  load_env
  fence_network
  provenance_header
  local pkg
  for pkg in "${GIT_PKGS[@]}" llvm cpm cccl eigen; do
    [[ -f "${SRC}/.fetched-${pkg}" ]] || die "${SRC}/${pkg} is not fetched: run fetch on the login node"
  done
  mkdir -p "${BUILD}" "${LOGS}" "${PREFIX}"
  write_inject
  {
    printf 'build job:  %s on %s (%s; %s), job %s\n' "$(now)" "$(hostname)" "$(. /etc/os-release && echo "${PRETTY_NAME}")" \
      "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)" "${SLURM_JOB_ID}"
    printf '  script:   %s  sha256=%s\n' "${SELF}" "$(sha256sum "${SELF}" | cut -d' ' -f1)"
    printf '  modules:  %s\n' "$(module -t list 2>&1 | tr '\n' ' ')"
    printf '  CC=%s  CXX=%s (%s; --showme: %s)\n' "${MPICC}" "${MPICXX}" "$("${MPICXX}" --version | head -1)" "$("${MPICXX}" --showme)"
    printf '  nvcc:     %s (%s)\n  CUDADIR=%s\n' "$(command -v nvcc)" "$(nvcc --version | tail -1)" "${CUDADIR}"
    printf '  mpi:      %s (%s)\n' "${WANT_MPI}" "$(mpirun --version 2>&1 | head -1)"
    printf '  cmake:    %s (%s)\n  openblas: %s\n' "$(cmake --version | head -1)" "$(command -v cmake)" "${OPENBLAS}"
    printf '  inject:   %s  sha256=%s\n' "${INJECT}" "$(sha256sum "${INJECT}" | cut -d' ' -f1)"
  } >> "${PROVENANCE}"

  # chromaform appends its own flags to these (-march=znver3 -fopenmp -O3, -std=..., libxml2, gmp).
  unset CFLAGS CXXFLAGS
  export LDFLAGS="-L${OPENBLAS}/lib -Wl,-rpath,${OPENBLAS}/lib"
  export LIBS="-lopenblas"
  export MAKEFLAGS=-k
  local base_extra="-DCPM_CCCL_SOURCE=${SRC}/cccl"
  local chroma_extra="${base_extra} -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${INJECT} -DCCCL_DIR=${SRC}/cccl/lib/cmake/cccl"

  run_phase 1-llvm "${JOBS_LLVM}" "${base_extra}" llvm
  run_phase 2-quda "${JOBS}" "${base_extra}" llvm quda
  run_phase 3-chroma "${JOBS}" "${chroma_extra}" llvm chroma

  [[ -x "${BIN_DIR}/hmc" && -x "${BIN_DIR}/chroma" ]] || die "chromaform finished but ${BIN_DIR}/hmc or chroma is missing"
  [[ -e "${PREFIX}/bin" ]] || ln -s "${CHROMA_NAME}/bin" "${PREFIX}/bin"
  printf 'built:      %s  hmc sha256=%s  chroma sha256=%s\n' "$(now)" \
    "$(sha256sum "${BIN_DIR}/hmc" | cut -d' ' -f1)" "$(sha256sum "${BIN_DIR}/chroma" | cut -d' ' -f1)" >> "${PROVENANCE}"
  install_provenance
}

# The install's PROVENANCE.txt: the stage record (sources, build phases) plus what was installed.
install_provenance() {
  local q
  {
    printf '# lq2 GPU Chroma, installed by 1_build_chroma/lq/build_chroma_gpu.sh. Run-time environment:\n'
    printf '#   CHROMA_STACK=gpu-lq2 source 5_studies/chroma_inputs/lq/env_chroma_lq.sh\n'
    printf '# Launcher: 5_studies/chroma_inputs/lq/chroma_baseg_mds.sh with CHROMA_STACK=gpu-lq2.\n\n'
    cat "${PROVENANCE}"
    printf '\ninstalled packages:\n'
    ls -1 "${PREFIX}" | sed 's/^/  /'
    q=$(grep -a -m1 -o 'QUDA 1\.1\.0 ([^)]*)' "${LOGS}/2-quda-${SLURM_JOB_ID}.log" 2>/dev/null || true)
    printf 'quda configure: %s\n' "${q:-not in this job (installed earlier)}"
    printf 'bin/hmc:    %s  sha256=%s\n' "${BIN_DIR}/hmc" "$(sha256sum "${BIN_DIR}/hmc" | cut -d' ' -f1)"
    printf 'bin/chroma: %s  sha256=%s\n' "${BIN_DIR}/chroma" "$(sha256sum "${BIN_DIR}/chroma" | cut -d' ' -f1)"
  } > "${PREFIX}/PROVENANCE.txt"
}

# ---- accept ---------------------------------------------------------------------------------------
# ldd/readelf of bin/hmc in the launcher's run-time environment. On a node without the NVIDIA driver
# (no /proc/driver/nvidia) libcuda.so.1 cannot resolve; the lq2 launcher repeats the check there.
accept() {
  [[ -x "${BIN_DIR}/hmc" ]] || die "no ${BIN_DIR}/hmc: run build first"
  local out=${PREFIX}/accept/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-login}
  mkdir -p "${out}"
  # (exported, not prefixed to source: bash drops a prefix assignment, and what the file sets in it, on return)
  export CHROMA_STACK=gpu-lq2 CHROMA_INSTALL=${PREFIX}
  source "${ENV_CHROMA}" || die "sourcing ${ENV_CHROMA} (gpu-lq2) failed"
  set -u
  local ldd_txt=${out}/ldd_hmc.txt res=() bad=0 libmpi libcuda unresolved
  ldd "${BIN_DIR}/hmc" > "${ldd_txt}" 2>&1 || die "ldd failed on ${BIN_DIR}/hmc"
  readelf -d "${BIN_DIR}/hmc" > "${out}/readelf_d_hmc.txt"
  libmpi=$(awk '$1 == "libmpi.so.40" {print $3}' "${ldd_txt}")
  if [[ "${libmpi}" == "${WANT_LIBMPI}" ]]; then res+=("libmpi.so.40 -> ${libmpi}: ok")
  else res+=("libmpi.so.40 -> '${libmpi}': FAIL (want ${WANT_LIBMPI})"); bad=1; fi
  if grep -q -E 'nvhpc|comm_libs|openmpi-3\.1\.5|libmpi_cxx' "${ldd_txt}"; then
    res+=("NVHPC / Open MPI 3.1.5 / libmpi_cxx: FAIL: $(grep -E 'nvhpc|comm_libs|openmpi-3\.1\.5|libmpi_cxx' "${ldd_txt}" | xargs)"); bad=1
  else res+=("nothing from NVHPC comm_libs or Open MPI 3.1.5, no libmpi_cxx: ok"); fi
  libcuda=$(awk '$1 == "libcuda.so.1" {print $3}' "${ldd_txt}")
  if [[ -z "${libcuda}" ]]; then
    res+=("libcuda.so.1: not loaded by hmc ($(grep -c 'NEEDED.*libcuda' "${out}/readelf_d_hmc.txt" || true) NEEDED entries)")
  elif [[ "${libcuda}" == "not" ]]; then
    if [[ -d /proc/driver/nvidia ]]; then res+=("libcuda.so.1 not found on a node with the NVIDIA driver: FAIL"); bad=1
    else res+=("libcuda.so.1 not found: no NVIDIA driver on $(hostname); the lq2 launcher checks it on the GPU node"); fi
  elif [[ "${libcuda}" == */stubs/* || "${libcuda}" == "${CUDA_HOME}"/* ]]; then
    res+=("libcuda.so.1 -> ${libcuda}: FAIL (toolkit stub)"); bad=1
  else res+=("libcuda.so.1 -> $(readlink -f "${libcuda}"): ok (driver)"); fi
  # Without the driver on this node, the driver's own libraries (libcuda, NVML) are checked on lq2.
  if [[ -d /proc/driver/nvidia ]]; then
    unresolved=$(grep 'not found' "${ldd_txt}" || true)
  else
    unresolved=$(grep 'not found' "${ldd_txt}" | grep -v -E '^[[:space:]]*lib(cuda|nvidia-ml)\.so\.1 ' || true)
  fi
  if [[ -z "${unresolved}" ]]; then res+=("unresolved libraries: none")
  else res+=("unresolved libraries: FAIL: $(xargs <<< "${unresolved}")"); bad=1; fi
  if grep -E 'R(UN)?PATH' "${out}/readelf_d_hmc.txt" | grep -E 'stubs|nvhpc' > /dev/null; then
    res+=("RPATH/RUNPATH: FAIL, names a stubs or nvhpc directory"); bad=1
  else res+=("RPATH/RUNPATH: $(grep -E 'R(UN)?PATH' "${out}/readelf_d_hmc.txt" | sed 's/.*\[\(.*\)\]/\1/' | xargs || true)"); fi
  {
    printf '\naccept:     %s on %s (job %s), env %s (CHROMA_STACK=gpu-lq2), files %s\n' "$(now)" "$(hostname)" \
      "${SLURM_JOB_ID:-none}" "${ENV_CHROMA}" "${out}"
    printf '  modules:  %s\n' "$(module -t list 2>&1 | tr '\n' ' ')"
    printf '  %s\n' "${res[@]}"
    printf '  NEEDED:   %s\n' "$(grep NEEDED "${out}/readelf_d_hmc.txt" | sed 's/.*\[\(.*\)\]/\1/' | xargs)"
    printf '  result:   %s\n' "$( (( bad == 0 )) && echo PASS || echo FAIL)"
  } | tee -a "${PREFIX}/PROVENANCE.txt"
  (( bad == 0 )) || die "acceptance failed (${out})"
}

print_paths
case "${ACTION}" in
  fetch)  fetch ;;
  check)  check ;;
  build)  build ;;
  accept) accept ;;
  info)   ;;
  *) printf 'usage: %s {fetch|check|build|accept|info}\n' "$0" >&2; exit 2 ;;
esac
