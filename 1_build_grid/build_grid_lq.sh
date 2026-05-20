#!/usr/bin/env bash
# Build Grid on the lq HPC cluster (AlmaLinux 8, NVIDIA A100 GPUs).
#
# Edit GRID_PROFILE in config.sh to choose which fork to build:
#   "txqcd"    → mlwagman/Grid-TXQCD
#   "mainline" → paboyle/Grid
# Then run this script once per profile.
#
# ── Normal usage (builds everything) ─────────────────────────────────────────
#   ./1_build_grid/build_grid_lq.sh
#
# ── Step-by-step (for debugging or resuming a failed build) ──────────────────
#   source ./1_build_grid/build_grid_lq.sh   # loads all step functions, runs nothing
#   step_setup_env    # 1. load modules + set CUDA paths
#   step_clone        # 2. git clone the Grid repo
#   step_bootstrap    # 3. generate configure script
#   step_configure    # 4. run configure with GPU/CUDA flags
#   step_build        # 5. make -j16
#
# Each step is idempotent — it skips if its output already exists,
# so you can re-run safely after fixing a failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MACHINE=lq
source "${SCRIPT_DIR}/../config.sh"

echo "========================================"
echo " Grid build — lq cluster"
echo " Profile:  ${GRID_PROFILE}"
echo " Repo:     ${GRID_REPO}"
echo " Source:   ${GRID_SRC}"
echo " Build:    ${GRID_BUILD}"
echo "========================================"


# ── Step 1: Load modules and set CUDA environment ────────────────────────────
step_setup_env() {
    echo ""
    echo "==> [1/5] Setting up environment..."
    module load ${LQ_MODULES}
    export PATH=${NVHPC_ROOT}/cuda/bin:${PATH}
    export LD_LIBRARY_PATH=${NVHPC_ROOT}/cuda/lib64:${NVIDIALIB}:${NVHPC_ROOT}/compilers/lib:${LD_LIBRARY_PATH:-}
    export CUDA_CACHE_PATH=${BASE_DIR}/.cuda_cache
    echo "-- nvcc: $(which nvcc)"
    echo "-- mpicxx: $(which mpicxx)"
}


# ── Step 2: Clone ─────────────────────────────────────────────────────────────
step_clone() {
    echo ""
    echo "==> [2/5] Cloning ${GRID_REPO}..."
    if [ -d "${GRID_SRC}/.git" ]; then
        echo "-- Already cloned at ${GRID_SRC}, skipping."
    else
        git clone "${GRID_REPO}" "${GRID_SRC}"
    fi
}


# ── Step 3: Bootstrap ─────────────────────────────────────────────────────────
step_bootstrap() {
    echo ""
    echo "==> [3/5] Bootstrapping..."
    if [ -f "${GRID_SRC}/configure" ]; then
        echo "-- configure already exists, skipping bootstrap."
    else
        cd "${GRID_SRC}"
        ./bootstrap.sh
    fi
}


# ── Step 4: Configure ─────────────────────────────────────────────────────────
step_configure() {
    echo ""
    echo "==> [4/5] Configuring (CUDA, ${CUDA_ARCH})..."
    mkdir -p "${GRID_BUILD}"

    # Mainline Grid has __rdtsc/__rdpmc stubs accidentally commented out in the
    # GRID_CUDA branch of PerfCount.h — uncomment them so CUDA builds compile.
    if [ "${GRID_PROFILE}" = "mainline" ]; then
        local perfcount="${GRID_SRC}/Grid/perfmon/PerfCount.h"
        if grep -q "^//accelerator_inline uint64_t __rdtsc" "${perfcount}" 2>/dev/null; then
            echo "-- Patching PerfCount.h (__rdtsc/__rdpmc stubs commented out upstream)..."
            sed -i 's|^//accelerator_inline uint64_t __rdtsc|accelerator_inline uint64_t __rdtsc|g' "${perfcount}"
            sed -i 's|^//accelerator_inline uint64_t __rdpmc|accelerator_inline uint64_t __rdpmc|g' "${perfcount}"
        fi
    fi

    local COMPUTE_CAP="${CUDA_ARCH#sm_}"   # strips "sm_" → e.g. 80

    cd "${GRID_BUILD}"
    "${GRID_SRC}/configure" \
        --enable-comms=mpi3 \
        --enable-simd=GPU \
        --enable-gen-simd-width=64 \
        --enable-accelerator=cuda \
        --enable-shm=nvlink \
        --enable-unified=no \
        --with-lime="${CLIME_ROOT}" \
        --with-hdf5="${HDF5_ROOT}" \
        ${CONFIGURE_EXTRA} \
        CXX=nvcc \
        MPICXX=mpicxx \
        LDFLAGS="-cudart shared -L/usr/lib64 -L${CLIME_ROOT}/lib -L${NVIDIALIB}" \
        CXXFLAGS="-ccbin mpicxx -gencode arch=compute_${COMPUTE_CAP},code=${CUDA_ARCH} -std=c++17 -cudart shared -I${NVIDIAINCLUDE} -Xcompiler -fno-strict-aliasing --expt-extended-lambda --expt-relaxed-constexpr" \
        2>&1 | tee configure.log

    echo "-- Summary: ${GRID_BUILD}/grid.configure.summary"
}


# ── Step 5: Build ─────────────────────────────────────────────────────────────
step_build() {
    echo ""
    echo "==> [5/5] Building (~30-60 min on login node)..."
    cd "${GRID_BUILD}"
    if [ -f "${GRID_BUILD}/Grid/libGrid.a" ]; then
        echo "-- libGrid.a already exists, skipping build."
    else
        make -j16 2>&1 | tee build.log
    fi
    echo ""
    echo "========================================"
    echo " Build complete."
    echo " Source:      ${GRID_SRC}"
    echo " Build:       ${GRID_BUILD}"
    echo " grid-config: ${GRID_BUILD}/grid-config"
    echo "========================================"
}


# ── Run all steps ─────────────────────────────────────────────────────────────
# To resume after a failure: comment out the steps that already succeeded.
_run_all=true
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # Script is being sourced — just load the functions, don't run anything.
    echo "-- Functions loaded. Call step_setup_env / step_clone / step_bootstrap / step_configure / step_build individually."
    _run_all=false
fi

if [ "${_run_all}" = "true" ]; then
    step_setup_env
    step_clone
    step_bootstrap
    step_configure
    step_build
fi
