#!/bin/bash
#SBATCH -A m4599_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J chroma_baseg
#
# Chroma base+G validation run -- the independent cross-check of the Grid
# base+G MDSTEPS scan.  Submit as:
#
#   MDS=12 sbatch --export=ALL,MDS=12 submit_scripts/chroma_validation/chroma_baseg_mds.sh
#
# GEOMETRY IS PINNED TO THE GRID REFERENCE and must not be changed casually:
# every Grid base+G run (jobs 55904828 / 55984845 / 55982818 / 56407103) ran on
# 4 nodes, 16 ranks, --mpi 1.2.2.4 (see submit_scripts/hmc48_compact/
# chroma_match_sop_48.sh).  A timing comparison at a different node count or
# decomposition is worthless, which is the whole point of this run.
#
# Grid reference wall-clock, 10 trajectories, 4 nodes:
#   MDSTEPS=12  job 55904828  09:43:16
#   MDSTEPS=10  job 55984845  08:14:31
#   MDSTEPS=8   (see runs/2026_7_16_baseG_mdsteps8_seed300_48)
#
# The input recipe is transcribed verbatim from cfg_2000.lime's LIME metadata
# (chroma/inputs/, regenerate with extract_baseline.sh + make_inputs.py), so the
# monomials, masses, ladder and solver settings are the production baseline's
# by construction, not by retyping.
#
# MG CADENCE -- the baseline is adaptive, and our input reproduces it verbatim.
# One shared <SubspaceID>mg_subspace</SubspaceID> with <ThresholdCount>70</>
# on the near-critical Mass=-0.2416 solver and 500 on the other 17.  That 70 is
# the same threshold the Grid setup was matched to, so this IS close to a
# like-for-like comparison.  (An earlier version of this comment claimed Chroma
# rebuilt MG every trajectory -- that was wrong.)
#
# STILL OPEN: Chroma logged 8 "Refreshing Subspace" events in ONE trajectory,
# while Grid logged 6 "MG full-rebuilt" across TEN.  Either Chroma's refresh is
# a cheaper operation than Grid's rebuild, or the effective cadences really do
# differ.  Settle this from the per-solve timings before quoting a speedup.
set -uo pipefail

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
source "$BASE/chroma/env_chroma_pm.sh"

MDS=${MDS:?set MDS=12, 10 or 8}
RUNDATE=${RUNDATE:-$(date +%Y_%-m_%-d)}
# SUFFIX selects an input variant, e.g. SUFFIX=_1traj for the stage-4
# single-trajectory physics cross-check, or _nomgerase for the
# integrator-only timing variant.  Empty = the standard 10-trajectory input.
SUFFIX=${SUFFIX:-}

INPUT=$CHROMA_INPUTS/hmc_baseg_mds${MDS}${SUFFIX}.ini.xml
[ -f "$INPUT" ] || { echo "missing input $INPUT" >&2; exit 1; }

# Discovered by env_chroma_pm.sh.  NOTE: the binary is `hmc`, NOT `chroma` --
# `chroma` is the measurement driver with a different input schema; `hmc` is the
# gauge-generation program our <MCControl>+<HMCTrj> input drives.
# chromaform also names the install dir after the build config
# (chroma-quda-qdp-jit-double-nd4-cmake), not "chroma".
[ -n "${CHROMA_HMC:-}" ] && [ -x "$CHROMA_HMC" ] || {
  echo "hmc binary not found under $CHROMA_INSTALL/chroma*/bin -- build first" >&2
  exit 1
}

OUT=$BASE/runs/${RUNDATE}_chroma_baseg_mds${MDS}${SUFFIX}
mkdir -p "$OUT/plaquette" "$OUT/eigen_mdagm" "$OUT/quda_resource"
# Chroma's InlineMeasurements write to relative paths (./plaquette/...) and will
# abort if the directory is absent -- hence the mkdir above.

export QUDA_RESOURCE_PATH=$OUT/quda_resource

# Seed the QUDA kernel-autotune cache from a previous run.  The stage-4 run
# paid ~24 min of cold-start autotuning; the Grid reference (job 55904828) never
# paid that, so charging it to Chroma would bias the comparison.  Same trick the
# Grid scripts use (accept_baseG_seed300_regular.sh:59-62).
SRC_TUNE=${SRC_TUNE:-$BASE/runs/2026_8_18_chroma_baseg_mds12_1traj/quda_resource}
if [ -d "$SRC_TUNE" ] && [ -z "$(ls -A "$QUDA_RESOURCE_PATH" 2>/dev/null)" ]; then
  cp "$SRC_TUNE"/tunecache.tsv "$QUDA_RESOURCE_PATH/" 2>/dev/null || true
  echo "seeded tune cache from $SRC_TUNE"
fi

# Match the Grid reference decomposition exactly.
# The flag is -geom (QDP's own option, qdp_parscalar_init.cc:541): it sets
# logical_geom, from which QDP declares the QMP logical topology.
# Do NOT use -qmp-geom.  QMP declares the topology on the *allocated* comm only,
# so QMP_get_logical_dimensions() (which reads the *default* comm) returns NULL
# and Chroma::initialize dereferences it -> SIGSEGV before any input is read.
# Verified 2026-08-18 on the smoke test.
GEOM="1 2 2 4"
NTASKS=16

# Extra arguments for hmc, appended after -geom; empty by default (= the old command line).
# Added 2026-10-07 for QDP-JIT's -poolsize: by default QDP-JIT reserves 92% of free GPU
# memory, leaving QUDA's managed memory ~3 GB on a 40 GB card (qdp_jit_config.cc).
CHROMA_EXTRA_ARGS=${CHROMA_EXTRA_ARGS:-}

# Words inserted before the hmc binary, split on spaces; empty by default (= the old command
# line). Added 2026-11-10 for profiling: grid-lqcd-workflow/common/nsys_rank.sh runs Nsight
# Systems on rank 0 only, its NSYS_* settings exported by the caller.
LAUNCH_PREFIX=${LAUNCH_PREFIX:-}
read -r -a LAUNCH_PREFIX_WORDS <<< "$LAUNCH_PREFIX"

LOG=$OUT/chroma_baseg_mds${MDS}${SUFFIX}.log
{
  echo "ENV MDSTEPS=$MDS QMP_GEOM='$GEOM' NTASKS=$NTASKS NODES=4 CHROMA_EXTRA_ARGS='$CHROMA_EXTRA_ARGS'"
  [ -n "$LAUNCH_PREFIX" ] && echo "ENV LAUNCH_PREFIX='$LAUNCH_PREFIX' NSYS='ranks=${NSYS_RANKS:-0} trace=${NSYS_TRACE:-cuda,nvtx} delay=${NSYS_DELAY:-0} duration=${NSYS_DURATION:-all} out=${NSYS_OUT:-unset}' QUDA_RESOURCE_PATH=$QUDA_RESOURCE_PATH SRC_TUNE=$SRC_TUNE"
  echo "INPUT=$INPUT"
  echo "BIN=$CHROMA_HMC"
  echo "outer n_steps: $(grep -m1 -o '<n_steps>[0-9]*</n_steps>' "$INPUT")"
  echo "inner n_steps: $(grep -o '<n_steps>[0-9]*</n_steps>' "$INPUT" | sed -n 2p)"
} > "$LOG"

# 700m guard inside the 12h limit, mirroring the Grid scripts: reaps a hung
# teardown after all HMC data is written (the QUDA teardown abort is cosmetic --
# CRASH at finalize does NOT mean a failed run; see memory note).
# Works either as an sbatch script or against an existing allocation:
#   JOBID=<id> bash chroma_baseg_mds.sh
SRUN=(srun)
[ -n "${JOBID:-}" ] && SRUN=(srun --jobid="$JOBID")

# CHROMA_TIMEOUT_MIN (2026-11-12): shorter guard for smoke tests; default the old 700.
timeout -k 60s "${CHROMA_TIMEOUT_MIN:-700}m" \
  "${SRUN[@]}" --chdir="$OUT" --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 --cpu-bind=cores \
  -n"$NTASKS" "${LAUNCH_PREFIX_WORDS[@]}" "$CHROMA_HMC" \
    -i "$INPUT" \
    -o "$OUT/chroma_baseg_mds${MDS}${SUFFIX}.out.xml" \
    -l "$OUT/chroma_baseg_mds${MDS}${SUFFIX}.xmldat" \
    -geom $GEOM $CHROMA_EXTRA_ARGS \
  >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"

echo "==== chroma base+G MDSTEPS=$MDS$SUFFIX summary ===="
grep -aE "DeltaH|AcceptP|ACCEPT|REJECT|w_plaq|Total time|HMC trajectory|SEC|error|Error" \
  "$LOG" | head -60
echo
echo "---- per-monomial forces (MonitorForces) ----"
grep -aE "MonomialForce|<Name>|ForceAve|ForceMax" \
  "$OUT/chroma_baseg_mds${MDS}${SUFFIX}.xmldat" 2>/dev/null | head -40
