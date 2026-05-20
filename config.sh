#!/usr/bin/env bash
# Central configuration — edit GRID_PROFILE to switch between Grid forks.
# Sourced automatically by all build/compile/run scripts.

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$WORKFLOW_DIR")"

# ── Profile ───────────────────────────────────────────────────────────────────
# "txqcd"    → mlwagman/Grid-TXQCD (TXQCD physics extensions)
# "mainline" → paboyle/Grid        (upstream Grid)
export GRID_PROFILE="txqcd"

case "$GRID_PROFILE" in
    txqcd)
        export GRID_REPO="https://github.com/mlwagman/Grid-TXQCD.git"
        export GRID_SRC="$BASE_DIR/Grid-TXQCD"
        export CONFIGURE_EXTRA="--disable-fermion-reps --disable-gparity"
        ;;
    mainline)
        export GRID_REPO="https://github.com/paboyle/Grid.git"
        export GRID_SRC="$BASE_DIR/Grid-mainline"
        export CONFIGURE_EXTRA=""
        ;;
    *)
        echo "Unknown GRID_PROFILE: '$GRID_PROFILE'. Use 'txqcd' or 'mainline'." >&2
        return 1
        ;;
esac

# ── Paths (all derived from profile — change GRID_PROFILE, everything follows) ─
export GRID_BUILD="$BASE_DIR/build-$GRID_PROFILE"
export GRID_INSTALL="$BASE_DIR/install-$GRID_PROFILE"
export DEPS_DIR="$BASE_DIR/deps"        # from-scratch dep builds (shared between profiles)
export RUNS_DIR="$BASE_DIR/runs"

# ── Compilers ─────────────────────────────────────────────────────────────────
export MPICXX=/opt/homebrew/bin/mpicxx
_gcc_ver=$(ls /opt/homebrew/bin/g++-* 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tail -1)
export OMPI_CXX="/opt/homebrew/bin/g++-${_gcc_ver}"

# ── Homebrew dependency prefixes (used by build_grid_homebrew.sh only) ────────
export GMP=/opt/homebrew/opt/gmp
export MPFR=/opt/homebrew/opt/mpfr
export OPENSSL=/opt/homebrew/opt/openssl@3
export FFTW=/opt/homebrew/opt/fftw

# ── Machine: lq (AlmaLinux 8, NVIDIA A100 GPUs, module system) ────────────────
# Set MACHINE=lq before sourcing this file (build_grid_lq.sh does this automatically).
if [ "${MACHINE:-}" = "lq" ]; then
    # Modules to load
    export LQ_MODULES="cmake gompi/2023a ucx_cuda/1.14.1_cuda_12.2.1 ucc_cuda/1.2.0_cuda_12.2.1 gcc/12.3.0"

    # NVHPC 23.7 / CUDA 12.2
    export NVHPC_ROOT=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/23.7
    export CUDA_HOME=${NVHPC_ROOT}/cuda/12.2
    export NVIDIALIB=${NVHPC_ROOT}/math_libs/lib64
    export NVIDIAINCLUDE=${NVHPC_ROOT}/math_libs/include

    # Shared group dependencies (read-only — pre-built for this cluster)
    export CLIME_ROOT=/lustre2/nplqcd/dwf/c-lime/install
    export HDF5_ROOT=/srv/software/el8/x86_64/eb/HDF5/1.14.2-gompi-2023a

    # GPU target: A100 = sm_80. Change for other GPU generations.
    export CUDA_ARCH=sm_80

    # Build dir gets a -gpu suffix to distinguish from any future CPU builds
    export GRID_BUILD="${BASE_DIR}/build-${GRID_PROFILE}-gpu"
fi
