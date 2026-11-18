#!/bin/bash
# Unified environment for the Chroma (QDP-JIT + QUDA/MG) build and validation
# runs on Perlmutter.  Source once per shell, or at the top of a SLURM job:
#   source /global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/env_chroma_pm.sh
#
# This is the Perlmutter analogue of Grid-TXQCD/env_lq2_grid.sh (the lq2/FNAL
# environment).  Differences from that file, and why:
#   - No squid.fnal.gov proxy: NERSC login nodes reach GitHub directly.
#   - Cray PE (PrgEnv-gnu + cray-mpich) instead of gompi/ucx_cuda.  MPI is
#     carried by the `cc`/`CC` wrappers, NOT by mpicc/mpicxx.
#   - Build tree lives on $PSCRATCH (project CFS inode quota is ~99% full);
#     the install and everything else stays on CFS.  See README.md.

# Ensure `module` exists in non-interactive shells (SLURM job scripts).
if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
  elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
  fi
fi

# Perlmutter default GPU stack, loaded explicitly so batch jobs are
# reproducible rather than inheriting whatever the login shell had.
module load PrgEnv-gnu
# Pin cray-mpich. Perlmutter's PE default moved (PrgEnv-gnu -> 8.7.0,
# cpe -> 26.03) around 2026-08-19, which drags cray-mpich 9.0.1 -> 9.1.0.
# Our binaries link the 9.0.1 soname (libmpi_gnu_123.so.12), so we hold 9.0.1.
module swap cray-mpich cray-mpich/9.0.1
module load gcc-native/14
module load cudatoolkit/12.9
module load craype-accel-nvidia80
module load craype-x86-milan          # zen3 -- matches lq2.sh's --zen3

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
# This killed jobs 57236621 (chroma timing) and 57269305 (grid z1 20-traj ext)
# on 2026-08-20 01:0x.  Note neither binary needs libcudart.so.13 directly --
# the demand is purely transitive through that GTL symlink.
#
# Verify after any PE change with:
#   ldd $CHROMA_HMC | grep -E 'gtl|cudart|not found'      # want 0 "not found"
#
# Full postmortem, incl. why `module swap` ALONE is not enough and the smoke
# tests that validated this:
#   __docs/2026_8_20_perlmutter_pe_bump_cudart13_postmortem.md
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Canonical paths.  Set ONCE here so build and SLURM scripts never depend on
# cwd or SLURM_SUBMIT_DIR.
# ---------------------------------------------------------------------------
export CHROMA_ROOT=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma
export CHROMA_INSTALL=$CHROMA_ROOT/install          # CFS: the artifact we keep
export CHROMA_INPUTS=$CHROMA_ROOT/inputs
export CHROMAFORM=$CHROMA_ROOT/chromaform

# Scratch: regenerable compile tree only.  PURGED after ~8 weeks untouched --
# nothing here is precious, see README.md for how to regenerate.
export CHROMA_SCRATCH=${PSCRATCH:-/pscratch/sd/v/vpa}/chroma_build
export CHROMA_SRC=$CHROMA_SCRATCH/source
export CHROMA_BUILD=$CHROMA_SCRATCH/build

# chromaform names the install dir after the config it built, e.g.
#   install/chroma-quda-qdp-jit-double-nd4-cmake/
# NOT install/chroma/.  Discover it rather than hard-coding, so the path keeps
# working if the QUDA/QDP flavour changes.
# NOTE: must not fail when the glob matches nothing (i.e. before Chroma is
# built).  build_chroma_pm.sh runs under `set -euo pipefail`, and a bare
# `$(ls ...)` returning 2 aborts the whole build before it starts.
#
# Two binaries matter here:
#   hmc     - the gauge-generation program.  This is what our <MCControl> +
#             <HMCTrj> inputs drive, and what the validation runs use.
#   chroma  - the general measurement driver (different input schema).
CHROMA_BINDIR=""
for _d in "$CHROMA_INSTALL"/chroma*/bin; do
  if [ -x "$_d/hmc" ]; then
    CHROMA_BINDIR="$_d"
    break
  fi
done
unset _d
export CHROMA_BINDIR
export CHROMA_HMC="${CHROMA_BINDIR:+$CHROMA_BINDIR/hmc}"
export CHROMA_BIN="${CHROMA_BINDIR:+$CHROMA_BINDIR/chroma}"
if [ -n "$CHROMA_BINDIR" ]; then
  export PATH="$CHROMA_BINDIR:$PATH"
fi

# ---------------------------------------------------------------------------
# Runtime knobs
# ---------------------------------------------------------------------------
# GPU-aware cray-mpich (Perlmutter requires this to be explicit).
export MPICH_GPU_SUPPORT_ENABLED=1

# QUDA memory behaviour -- carried over verbatim from env_lq2_grid.sh so the
# two machines' QUDA runs stay comparable.
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_MANAGED_MEMORY=1

# Kernel autotune caches.  Worth 10-30 s per run, more for MG.  Deliberately on
# CFS, not scratch: these are cheap in inodes and annoying to lose.
export CUDA_CACHE_PATH=$CHROMA_ROOT/cache
export QUDA_RESOURCE_PATH=$CHROMA_ROOT/cache/quda_resource
mkdir -p "$QUDA_RESOURCE_PATH"

export CUDADIR=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}

# Emit a summary when sourced interactively; stay quiet inside jobs.
if [ -t 1 ]; then
  module list 2>&1
  echo "CHROMA_INSTALL=$CHROMA_INSTALL"
  echo "CHROMA_SRC=$CHROMA_SRC   (scratch, purgeable)"
  echo "CHROMA_BUILD=$CHROMA_BUILD (scratch, purgeable)"
fi
