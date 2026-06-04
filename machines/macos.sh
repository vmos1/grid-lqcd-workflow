#!/usr/bin/env bash
# macOS (Homebrew) — local development / CPU-only builds.
#
# Sourced by config.sh when MACHINE=macos (auto-detected via uname=Darwin).

_gcc_ver=$(ls /opt/homebrew/bin/g++-* 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tail -1)

export MPICXX=/opt/homebrew/bin/mpicxx
export OMPI_CXX="/opt/homebrew/bin/g++-${_gcc_ver}"

export GMP=/opt/homebrew/opt/gmp
export MPFR=/opt/homebrew/opt/mpfr
export OPENSSL=/opt/homebrew/opt/openssl@3
export FFTW=/opt/homebrew/opt/fftw

unset _gcc_ver
