#!/usr/bin/env bash
# Perlmutter (NERSC) — A100 GPU stack, 2026 toolchain.
#
# Sourced by config.sh (auto-detected via $NERSC_HOST=perlmutter) or directly.
# At build time AND by sbatch scripts on compute nodes.
#
# GMP: complete in /usr (auto-detected by Grid configure, no flag needed).
# MPFR: Grid's RHMC Remez requires it. No dev header in /usr on Perlmutter;
#       complete copy (header + libmpfr.so.4) lives in the Cray PE gcc tree.

export CRAY_ACCEL_TARGET=nvidia80

module load PrgEnv-gnu
# Pin cray-mpich. Perlmutter's PE default moved (PrgEnv-gnu -> 8.7.0,
# cpe -> 26.03) around 2026-08-19, which drags cray-mpich 9.0.1 -> 9.1.0.
# Our binaries link the 9.0.1 soname (libmpi_gnu_123.so.12), so we hold 9.0.1.
module swap cray-mpich cray-mpich/9.0.1
module load cudatoolkit/12.9
module load craype-accel-nvidia80

# ---------------------------------------------------------------------------
# CRITICAL (2026-08-20): make the RUNTIME loader honour the modules above.
#
# Cray modules only populate CRAY_LD_LIBRARY_PATH.  The dynamic loader reads
# LD_LIBRARY_PATH, so without this line the module choice above has NO effect
# at run time and the system symlink farm /opt/cray/pe/lib64 (which IS on the
# default linker path) wins instead.
#
# That farm is currently INCONSISTENT after the PE bump:
#   /opt/cray/pe/lib64/libmpi_gnu_123.so.12 -> cray-mpich 9.0.1   (CUDA 12)
#   /opt/cray/pe/lib64/libmpi_gtl_cuda.so.0 -> cray-mpich 9.1.0   (CUDA 13)
# The GTL name carries no version, so it silently picked up the new default.
# Result: our binary loads the 9.0.1 MPI lib but the 9.1.0 GPU transport, whose
# NEEDED libcudart.so.13 cannot resolve under cudatoolkit/12.9 ->
#   "error while loading shared libraries: libcudart.so.13" on EVERY rank,
#   exit 127, ~10 s after launch.
# This killed jobs 57269305 (z1 20-traj ext) and 57236621 (chroma timing) on
# 2026-08-20 01:0x.  Note the binary does NOT need libcudart.so.13 directly --
# it needs libcudart.so.12; the .13 demand is transitive through that symlink.
#
# Verify after any PE change with:
#   ldd $SOP_BIN | grep -E 'gtl|cudart|not found'         # want 0 "not found"
#
# Full postmortem, incl. why `module swap` ALONE is not enough and the smoke
# tests that validated this:
#   __docs/2026_8_20_perlmutter_pe_bump_cudart13_postmortem.md
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

export GRID_MPFR_PREFIX=/opt/cray/pe/gcc/mpfr/3.1.4
export LD_LIBRARY_PATH="${GRID_MPFR_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Cluster-specific dependency paths
export CLIME_ROOT=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/lime-install
export CUDA_ARCH=sm_80
