#!/usr/bin/env bash
# Build Grid using Homebrew-installed dependencies.
# Fastest setup for macOS Apple Silicon — recommended for local development.
#
# Usage: ./1_build_grid/build_grid_homebrew.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2"

echo "==> Profile: $GRID_PROFILE"
echo "==> Repo:    $GRID_REPO"

# ── 1. Homebrew dependencies ──────────────────────────────────────────────────
echo "==> Installing brew dependencies..."
brew install open-mpi gmp mpfr fftw openssl@3 autoconf automake libtool

# ── 2. Clone source ───────────────────────────────────────────────────────────
if [ ! -d "$GRID_SRC/.git" ]; then
    echo "==> Cloning $GRID_REPO..."
    git clone "$GRID_REPO" "$GRID_SRC"
else
    echo "==> Source already exists at $GRID_SRC, skipping clone."
fi

# ── 3. Bootstrap ──────────────────────────────────────────────────────────────
cd "$GRID_SRC"

if [ ! -f "$GRID_SRC/configure" ]; then
    echo "==> Bootstrapping ($GRID_PROFILE)..."
    if [ "$GRID_PROFILE" = "txqcd" ]; then
        ARC=$(basename "$EIGEN_URL")
        echo "-- deploying Eigen source..."
        curl -L "$EIGEN_URL" -o "$ARC"
        ./scripts/update_eigen.sh "$ARC"
        rm "$ARC"
        echo "-- generating Make.inc files..."
        ./scripts/filelist
        echo "-- generating configure script..."
        PATH="/opt/homebrew/opt/libtool/libexec/gnubin:/opt/homebrew/bin:$PATH" \
            autoreconf -fvi
    else
        # mainline Grid ships bootstrap.sh which handles everything
        ./bootstrap.sh
    fi
else
    echo "==> Configure script already exists, skipping bootstrap."
    echo "    (Remove $GRID_SRC/configure to force re-bootstrap)"
fi

# ── 4. Configure ──────────────────────────────────────────────────────────────
mkdir -p "$GRID_BUILD"
cd "$GRID_BUILD"

echo "==> Configuring ($GRID_PROFILE)..."
"$GRID_SRC/configure" \
    CXX="$MPICXX" \
    CXXFLAGS="-O2" \
    --enable-simd=GEN \
    --enable-comms=mpi-auto \
    --with-gmp="$GMP" \
    --with-mpfr="$MPFR" \
    --with-openssl="$OPENSSL" \
    --with-fftw="$FFTW" \
    --prefix="$GRID_INSTALL" \
    $CONFIGURE_EXTRA \
    2>&1 | tee configure.log

# ── 5. Build ──────────────────────────────────────────────────────────────────
echo "==> Building libGrid (this takes several minutes)..."
make -j"$(sysctl -n hw.logicalcpu)" 2>&1 | tee build.log

# ── 6. Install ────────────────────────────────────────────────────────────────
echo "==> Installing..."
make install 2>&1 | tee -a build.log

echo ""
echo "✓ Grid ($GRID_PROFILE) built and installed."
echo "  Source:      $GRID_SRC"
echo "  Build:       $GRID_BUILD"
echo "  Install:     $GRID_INSTALL"
echo "  grid-config: $GRID_INSTALL/bin/grid-config"
