#!/usr/bin/env bash
# Build Grid by compiling all dependencies from source.
# Use this when you need precise control over dependency versions,
# or on systems where Homebrew is unavailable (e.g. Linux HPC clusters).
#
# Dependencies built: make, GMP, MPFR, HDF5, OpenSSL, FFTW, LIME
# All installed under: $DEPS_DIR/local/
#
# Usage: ./1_build_grid/build_grid_fromscratch.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

add_patch_ventura=true

EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2"

dir_deps="$DEPS_DIR/"
dir_local="$DEPS_DIR/local/"

mkdir -p "$dir_deps" "$dir_local" "$GRID_BUILD" "$GRID_INSTALL"

echo "==> Profile:  $GRID_PROFILE"
echo "==> Repo:     $GRID_REPO"
echo "==> Deps dir: $DEPS_DIR"

export CC=/opt/homebrew/bin/gcc-15
export CXX=/opt/homebrew/bin/g++-15
export CFLAGS="-std=c11"
export CXXFLAGS="-std=c++17"

# ── System packages via Homebrew (minimal — compiler + MPI only) ──────────────
echo "==> Installing system packages via Homebrew..."
brew install gcc coreutils wget automake open-mpi


# ── Make ──────────────────────────────────────────────────────────────────────
echo "==> Building make..."
dir_make="${dir_deps}make-4.4/"
if [ ! -d "$dir_make" ]; then
    cd "$dir_deps"
    curl https://ftp.gnu.org/gnu/make/make-4.4.tar.gz | tar -xz
    cd "$dir_make"
    ./configure --prefix="$dir_local" 2>&1 | tee log_configure
    make -j4 2>&1 | tee log_make
    make check -j4 2>&1 | tee log_check
    make install -j4 2>&1 | tee log_install
    make installcheck -j4 2>&1 | tee log_installcheck
    echo "-- make installed --"
else
    echo "-- make already built, skipping --"
fi


# ── GMP ───────────────────────────────────────────────────────────────────────
echo "==> Building GMP..."
dir_gmp="${dir_deps}gmp-6.2.1/"
if [ ! -d "$dir_gmp" ]; then
    cd "$dir_deps"
    curl https://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.2.1.tar.bz2 | tar -xj
    cd "$dir_gmp"
    if [ "$add_patch_ventura" = true ]; then
        curl https://gmplib.org/repo/gmp/raw-rev/5f32dbc41afc -o gmp.diff
        patch -p1 < gmp.diff
    fi
    ./configure --prefix="$dir_local" 2>&1 | tee log_configure
    make -j4 2>&1 | tee log_make
    make check -j4 2>&1 | tee log_check
    make install -j4 2>&1 | tee log_install
    echo "-- GMP installed --"
else
    echo "-- GMP already built, skipping --"
fi


# ── MPFR ──────────────────────────────────────────────────────────────────────
echo "==> Building MPFR..."
dir_mpfr="${dir_deps}mpfr-4.1.0/"
if [ ! -d "$dir_mpfr" ]; then
    cd "$dir_deps"
    curl https://gcc.gnu.org/pub/gcc/infrastructure/mpfr-4.1.0.tar.bz2 | tar -xj
    cd "$dir_mpfr"
    ./configure --prefix="$dir_local" \
        --with-gmp-include="${dir_local}include" \
        --with-gmp-lib="${dir_local}lib" \
        2>&1 | tee log_configure
    make -j4 2>&1 | tee log_make
    make check -j4 2>&1 | tee log_check
    make install -j4 2>&1 | tee log_install
    echo "-- MPFR installed --"
else
    echo "-- MPFR already built, skipping --"
fi


# ── HDF5 ──────────────────────────────────────────────────────────────────────
echo "==> Building HDF5..."
dir_hdf5="${dir_deps}hdf5-1.14.3/"
if [ ! -d "$dir_hdf5" ]; then
    cd "$dir_deps"
    curl -L https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.3/src/hdf5-1.14.3.tar.gz | tar -xz
    cd "$dir_hdf5"
    ./configure --prefix="$dir_local" --enable-cxx 2>&1 | tee log_configure
    make -j4 2>&1 | tee log_make
    make check -j4 2>&1 | tee log_check
    make install -j4 2>&1 | tee log_install
    echo "-- HDF5 installed --"
else
    echo "-- HDF5 already built, skipping --"
fi


# ── OpenSSL ───────────────────────────────────────────────────────────────────
echo "==> Building OpenSSL..."
dir_openssl="${dir_deps}openssl-3.3.0/"
if [ ! -d "$dir_openssl" ]; then
    cd "$dir_deps"
    curl -L https://www.openssl.org/source/openssl-3.3.0.tar.gz | tar -xz
    cd "$dir_openssl"
    ./config --prefix="$dir_local" --openssldir="${dir_local}ssl" 2>&1 | tee log_config
    make -j4 2>&1 | tee log_make
    make test -j4 2>&1 | tee log_test
    make install 2>&1 | tee log_install
    echo "-- OpenSSL installed --"
else
    echo "-- OpenSSL already built, skipping --"
fi


# ── FFTW ──────────────────────────────────────────────────────────────────────
echo "==> Building FFTW..."
dir_fftw="${dir_deps}fftw-3.3.10/"
if [ ! -d "$dir_fftw" ]; then
    cd "$dir_deps"
    curl https://www.fftw.org/fftw-3.3.10.tar.gz | tar -xz
    cd "$dir_fftw"
    # double precision
    ./configure --prefix="$dir_local" --enable-threads --with-openmp --enable-mpi \
        2>&1 | tee log_config
    make -j4 2>&1 | tee log_make
    make check -j4 2>&1 | tee log_check
    make install -j4 2>&1 | tee log_install
    # single precision
    ./configure --prefix="$dir_local" --enable-float --enable-threads --with-openmp --enable-mpi \
        2>&1 | tee log_config2
    make -j4 2>&1 | tee log_make2
    make check -j4 2>&1 | tee log_check2
    make install -j4 2>&1 | tee log_install2
    echo "-- FFTW installed --"
else
    echo "-- FFTW already built, skipping --"
fi


# ── LIME ──────────────────────────────────────────────────────────────────────
echo "==> Building LIME..."
dir_lime="${dir_deps}lime-1.3.2/"
if [ ! -d "$dir_lime" ]; then
    cd "$dir_deps"
    curl http://usqcd-software.github.io/downloads/c-lime/lime-1.3.2.tar.gz | tar -xz
    cd "$dir_lime"
    ./configure --prefix="$dir_local"
    make -j4 2>&1 | tee log_make
    make check -j4 2>&1 | tee log_check
    make install -j4 2>&1 | tee log_install
    echo "-- LIME installed --"
else
    echo "-- LIME already built, skipping --"
fi


# ── Grid ──────────────────────────────────────────────────────────────────────
export PATH="${dir_local}bin:/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
export CC=/opt/homebrew/bin/gcc-15
export CXX=/opt/homebrew/bin/g++-15
export MPICXX=$(brew --prefix open-mpi)/bin/mpic++
export CXXFLAGS="-std=c++17 -I${dir_local}include -I$(brew --prefix open-mpi)/include"
export LDFLAGS="-L${dir_local}lib -L$(brew --prefix open-mpi)/lib -lhdf5 -lhdf5_cpp -lfftw3 -lfftw3f"

# Clone
if [ ! -d "$GRID_SRC/.git" ]; then
    echo "==> Cloning $GRID_REPO..."
    git clone "$GRID_REPO" "$GRID_SRC"
else
    echo "==> Source already exists at $GRID_SRC, skipping clone."
fi

# Bootstrap (profile-aware)
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
        PATH="/opt/homebrew/opt/libtool/libexec/gnubin:$PATH" autoreconf -fvi
    else
        ./bootstrap.sh
    fi
else
    echo "==> Configure script already exists, skipping bootstrap."
fi

# Configure
mkdir -p "$GRID_BUILD"
cd "$GRID_BUILD"
echo "==> Configuring ($GRID_PROFILE)..."
"$GRID_SRC/configure" \
    --enable-simd=NEONv8 \
    --prefix="$GRID_INSTALL" \
    --with-lime="$dir_lime" \
    --enable-comms=mpi-auto \
    --with-fftw="$dir_local" \
    --with-hdf5="$dir_local" \
    --with-openssl="$dir_local" \
    $CONFIGURE_EXTRA \
    2>&1 | tee configure.log

# Build and install
echo "==> Building libGrid (this takes several minutes)..."
make -j4 2>&1 | tee build.log
make check -j4 2>&1 | tee -a build.log
make install 2>&1 | tee -a build.log

echo ""
echo "✓ Grid ($GRID_PROFILE) built and installed."
echo "  Source:      $GRID_SRC"
echo "  Build:       $GRID_BUILD"
echo "  Install:     $GRID_INSTALL"
echo "  Deps:        $DEPS_DIR"
echo "  grid-config: $GRID_INSTALL/bin/grid-config"
