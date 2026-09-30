#!/bin/bash
# =============================================================================
# smoke16_puregrid.sh -- ONE 16^3x48 HMC trajectory of the pure-Grid
# compact-Schur driver (src/gen_qcd_hasenbusch_tune_compact_schur.cc), run
# inside an EXISTING allocation. This script never allocates and never submits.
#
# Purpose: pure-grid-hmc milestone M1 (__docs/2026_09_29_pure_grid_stock_build_plan.md
# section 4): run the fork QUDA-off reference binary, and later the stock-Grid
# binary, with IDENTICAL settings, then compare per-monomial forces, dH and the
# plaquette. Every physics, solver, seed and launch knob below is HARD-CODED on
# purpose. The only inputs are which binary runs and what the run is called.
#
# Usage (one line each):
#   SLURM_JOB_ID=<id> BIN=<binary> RUN=<name> bash smoke16_puregrid.sh
# e.g.
#   SLURM_JOB_ID=<id> BIN=<hb>/bin/gen_qcd_hasenbusch_tune_compact_schur_puregrid_fork RUN=smoke16_fork bash smoke16_puregrid.sh
#   SLURM_JOB_ID=<id> BIN=<hb>/bin/gen_qcd_hasenbusch_tune_compact_schur_stock RUN=smoke16_stock bash smoke16_puregrid.sh
# The allocation comes from `bash ~/.claude/bin/alloc_gpu.sh 1 <hours>` and is
# released with `release_alloc.sh <id>` (machines/perlmutter.md, "Allocations
# go through wrappers only").
#
# Output: runs/<YYYY_M_D>_<RUN>/hmc.log (unpadded date, machines/perlmutter.md
# "Filesystems"). Line 1 is the ENV provenance line; per the conventions entry
# of __docs/_SUM.md, that line, not this script, is the ground truth for what ran.
#
# Starting configuration. The driver has NO hot or tepid start. Without
# IMPORT_CFG (and without CKPT_RESUME_TRAJ) it prints "No IMPORT_CFG -- cold
# start." and calls SU<Nc>::ColdConfiguration (unit links; driver source, gauge
# field block). A cold start with the default STOUT_NSMEAR=1 gives NaN smeared
# links and -nan solves (README.md, "Cold-start caveat (NaN)"). So this smoke
# imports the thermalized 16^3 ensemble config, as every 16^3 recipe does
# (e.g. submit_scripts/hmc48_compact/three_level_recheck_tol11_16.sh). The M1
# gate requires that anyway ("Both complete and read the config"), and on the
# stock binary it exercises patch 02 (ILDG reader found_ildgLFN assert).
#
# Seed. HMC_SEED_OFFSET=k shifts the fixed seeds {11..15} (serial RNG) and
# {16..20} (parallel RNG) by k (driver source, RNG block). Both builds are
# configured RNG_SITMO (identical Config.h apart from GRID_HAVE_QUDA), which
# enables RNG_FAST_DISCARD: each site's generator is the master engine skipped
# by its GLOBAL index (Grid/lattice/Lattice_rng.h, SeedFixedIntegers, identical
# in fork and stock). The same seed therefore gives bit-identical momenta,
# pseudofermion noise and Metropolis random number in both binaries, and would
# even across decompositions. What differs is arithmetic after the draws (the
# GPU sum reduction changed between the fork's April base and 3d3eff86; build
# plan section 5), so agreement is to roundoff, not bitwise.
#
# Estimated wall time: the identical physics on the QUDA hybrid took 130 s for
# trajectory 1 on 4 GPUs (runs/2026_7_11_three_level_recheck_tol11_16/case_qf2.log);
# pure Grid CG is expected to be several times slower. Timeout is 60 min.
# =============================================================================
set -o pipefail

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
WORKFLOW=$BASE/grid-lqcd-workflow

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
USAGE="usage: SLURM_JOB_ID=<id> BIN=<binary> RUN=<name> bash $0"

# ---- 1. Inputs ---------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || die "SLURM_JOB_ID is not set (the id of an existing allocation). $USAGE"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be a numeric job id, got '$SLURM_JOB_ID'"
[ -n "${BIN:-}" ] || die "BIN is not set (path to the driver binary). $USAGE"
[ -n "${RUN:-}" ] || die "RUN is not set (run name, becomes runs/<date>_<RUN>). $USAGE"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN may contain only letters, digits, '.', '_' and '-', got '$RUN'"
{ [ -f "$BIN" ] && [ -x "$BIN" ]; } || die "BIN is not an executable file: $BIN"
BIN=$(readlink -f "$BIN")

# The allocation must already exist; this script only attaches srun steps to it.
# (squeue is read-only.)
JOB_STATE=$(squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null | head -1)
[ "$JOB_STATE" = RUNNING ] || die "allocation $SLURM_JOB_ID is not RUNNING (state: '${JOB_STATE:-not found}'). This script never allocates."

# ---- 2. Environment + loader check -------------------------------------------
# machines/perlmutter.sh pins cray-mpich 9.0.1 + cudatoolkit 12.9 and sets
# LD_LIBRARY_PATH from CRAY_LD_LIBRARY_PATH plus the MPFR lib. Without it the
# binary dies with libcudart.so.13 / libmpfr.so.4, exit 127 (machines/perlmutter.md,
# "Build and runtime environment"; _SUM.md perlmutter-ops).
# shellcheck source=/dev/null
source "$WORKFLOW/machines/perlmutter.sh" || die "sourcing $WORKFLOW/machines/perlmutter.sh failed"
set -u

LDD_OUT=$(ldd "$BIN" 2>&1) || die "ldd failed on $BIN: $LDD_OUT"
N_NOTFOUND=$(printf '%s\n' "$LDD_OUT" | grep -c 'not found' || true)
N_QUDA=$(printf '%s\n' "$LDD_OUT" | grep -ci 'libquda' || true)
if [ "$N_NOTFOUND" -ne 0 ]; then
  printf '%s\n' "$LDD_OUT" | grep 'not found' >&2
  die "ldd reports $N_NOTFOUND unresolved librar(y/ies) for $BIN (see machines/perlmutter.md)"
fi
[ "$N_QUDA" -eq 0 ] || die "ldd shows libquda in $BIN; this smoke is for pure-Grid builds only"

# ---- 3. Clear every knob this smoke does not set -----------------------------
# A stray variable from the caller's shell would silently change the run, or
# kill it: in a pure-Grid build HASEN_MG_RUNG / HASEN_QUDA_CG_RUNGS /
# HASEN_QUDA_CG_TAIL / HASEN_QUDA_FORCE_LOGDET exit(1) with "requires a QUDA
# build" (driver source). Families cleared:
#   QUDA_*, HASEN_*, HMC_MG_*, USE_HMC_MG  QUDA / MG routing (driver + fork headers)
#   FORCES_*                               FORCES_ONLY mode (NOT used here)
#   CKPT_*                                 checkpointer (off: no config written)
#   TXQCD_*, WCF_*                         fork-library env gates (all default off;
#                                          WCF_LOGDET_GPU/_DERIV_GPU also in the carried
#                                          LogDet header, off until audited, build plan s.6)
#   LAMBDA_MN2                             fork-only MN2 knob; stock Integrator_algorithm.h
#                                          has no such env
#   INTEGRATOR_VERBOSE_MEM                 MemoryManager::Print gate (patch 03)
#   STRANGE_EVEN, NO_METROP, IMPORT_CFG    reset, then set below as needed
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|STRANGE_EVEN$|NO_METROP$|IMPORT_CFG$)' | grep -v -E '^(HASEN_GRID_MG_|GRID_MG_)')
# Kept on purpose (not scrubbed): HASEN_GRID_MG_RUNGS and GRID_MG_* select and tune the
# pure-Grid multigrid rung solver (M2, 2026_09_29_pure_grid_m2_mg_solver_design.md).
# Unset = the plain-CG default binary behaviour; the ENV line below records them.
N_QUDA_ENV=$(compgen -e | grep -c '^QUDA_' || true)
[ "$N_QUDA_ENV" -eq 0 ] || die "QUDA_* variables still set after clearing"

# ---- 4. Physics (16^3x48 cl3 ensemble) ---------------------------------------
# Every value is exported even where it equals a compiled default, so neither
# binary's params.h (include/params.h, identical to Grid-TXQCD/production/params.h)
# nor a future source default can differ between the two runs.
export LATT=16.16.16.48                                   # include/params.h lattice_size()
export IMPORT_CFG=$BASE/data/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime   # 16^3 recipes; ILDG, 113 MB
[ -f "$IMPORT_CFG" ] || die "starting configuration not found: $IMPORT_CFG"
export BETA=6.1                                           # include/params.h
export CSW=1.24930970916466                               # include/params.h
# U0: the driver's LOCAL default is 1.0 and shadows params.h (0.8326...); the
# wrong U0 was the 48^3 blow-up root cause (driver source comment at the LW
# gauge action; __docs/_TOC.md u0-48cube). cl3 16^3 b6.1 value:
export U0=0.832605301399891
export MASS_LIGHT=-0.245                                  # include/params.h; must equal ladder[0]
# MASS_STRANGE: every 16^3 driver recipe uses -0.2050 (three_level_recheck_tol11_16.sh,
# gridcg_ladder_short_traj_16.sbatch), not params.h's -0.2450. Kept for comparability.
export MASS_STRANGE=-0.2050
export STOUT_RHO=0.125 STOUT_NSMEAR=1                     # include/params.h; ensemble metadata
                                                          # (run_probe_grid_mg.sh STOUT_* note)
# Strange RHMC bounds: the driver defaults, as used by every 16^3 recipe
# (48^3 needs 0.4/35, chroma_match_sop_48.sh; not 16^3).
export RAT_LO=1e-4 RAT_HI=100 RAT_DEGREE=20

# ---- 5. Hasenbusch ladder + integrator ----------------------------------------
# HASEN_LADDER format (driver source): comma-separated, no spaces needed,
# STRICTLY increasing light->heavy, >= 2 masses, first = MASS_LIGHT, last = the
# bare-det tail mass (no Pauli-Villars). Unset = single-level baseline (one Schur
# monomial + LogDet, no ratio rungs). This 3-mass ladder is the one the 16^3
# same-seed two-binary A/B at tol 1e-11/1e-12 used
# (submit_scripts/hmc48_compact/three_level_recheck_tol11_16.sh): 2 ratio rungs
# (PF0, PF1) + tail at -0.10, so the carried TwoFlavourSchurCloverRatioAction
# header is exercised too.
export HASEN_LADDER=-0.245,-0.20,-0.10
export HASEN_TAIL_LEVEL=outer                             # driver default; 16^3 recipe leaves it unset
# 2-level integrator (fermions / gauge). HASEN_STRANGE_LEVEL stays unset (cleared above).
export INTEGRATOR=ForceGradient                           # chroma_match_sop_48.sh + 16^3 A/B recipe
export GAUGE_INNER_MULT=2                                 # driver default = chroma_match_sop_48.sh
# TRAJL is MANDATORY: the driver exits "FATAL: TRAJL is not set" before the
# config load. sqrt(2)/4, the benchmark value (driver FATAL message; SOP):
export TRAJL=0.35355339059327379
# MDSTEPS=4: 16^3 recipes (MDSTEPS=20 timed out at 30 min on 1 GPU,
# gridcg_ladder_short_traj_16.sbatch header; the A/B recipe used 4).
export MDSTEPS=4
export N_TRAJ=1
# Metropolis ON: NO_METROP stays unset -> NoMetropolisUntil=0 (driver HMC block).
# NOTE: on a REJECT the "Plaquette: [ 1 ]" line is the START config, not the
# post-MD one; dH ("Total H after trajectory ... dH =") is printed either way.

# ---- 6. Seed + tolerances -----------------------------------------------------
# 200 = the seed of the 16^3 two-binary A/B (three_level_recheck_tol11_16.sh).
export HMC_SEED_OFFSET=200
# Production tolerances (__docs/_SUM.md conventions: 1e-11 force / 1e-12 action;
# chroma_match_sop_48.sh). Driver defaults are 1e-6 deriv / 1e-8 action / 1e-8
# strange, so all three are set. Strange 1e-9 = the 2026-07-13 production
# decision (chroma_match_sop_48.sh line 37).
export TUNE_CG_TOL_DERIV=1e-11
export TUNE_CG_TOL_ACTION=1e-12
export TUNE_CG_TOL_STRANGE=1e-9

# ---- 7. MPI / comms / threads -------------------------------------------------
# Pure-Grid comms set = the probe wrapper's
# (6_benchmarks/grid_quda_wilson_clover/perlmutter/run_probe_grid_mg.sh; port
# plan 2026_09_29_pure_grid_port_plan.md s.3.1 and s.7: never evaluate pure Grid
# under chroma_match_sop_48.sh's --comms-sequential --shm-mpi 1 settings), with
# MPICH GPU IPC + RDMA OFF for multi-rank GPU runs (port plan s.7 "CUDA IPC";
# _SUM.md perlmutter-ops). On one node with --shm-mpi 0 the halo goes through
# Grid's own nvlink shm, so the two MPICH knobs only guard against the IPC hang.
export SLURM_CPU_BIND=cores
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0 MPICH_RDMA_ENABLED_CUDA=0
export MPICH_OFI_NIC_POLICY=GPU                           # halo-nic fix (L122); inert on 1 node
export OMP_NUM_THREADS=8                                  # run_probe_grid_mg.sh
# Decomposition: 4 ranks split T, as every 4-rank 16^3 driver recipe does
# (three_level_recheck_tol11_16.sh: --mpi 1.1.1.4); local volume 16.16.16.12.
# MPI_GEOM is overridable (4 ranks in every case): the Grid-MG runs need 1.1.2.2, because
# 1.1.1.4 leaves a level-1 coarse T extent of 3 that the fp32 SIMD layout cannot split
# (test_grid_mg_odd.cc header). Runs to be compared must share the same MPI_GEOM.
MPI_GEOM=${MPI_GEOM:-1.1.1.4}
NTASKS=4
# Grid flags from run_probe_grid_mg.sh. No --device-mem: the cap exists for the
# MG PaddedCell raw allocations (run_probe_grid_mg.sh DEVICE_MEM_MB note), this
# run is CG only, the C2 probe ran with Grid's default, and the cache size does
# not change arithmetic. Add it at M2 when MG enters.
GRID_ARGS=(--grid "$LATT" --mpi "$MPI_GEOM" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap)

# ---- 8. Run directory (refuse to overwrite) -----------------------------------
DATE_TAG=$(date +%Y_%-m_%-d)                              # unpadded, runs/ convention
RUN_DIR=$BASE/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
LOG=$RUN_DIR/hmc.log

# ---- 9. select_gpu wrapper + srun step shape ----------------------------------
# 1:1 local rank -> GPU + NUMA domain (paboyle model), verbatim from
# run_probe_grid_mg.sh. It needs --gpu-bind=none: --gpu-bind=closest breaks CUDA
# IPC and hangs (_SUM.md halo-nic; port plan s.7). --cpus-per-task=32 with
# --cpu-bind=cores puts local rank i on NUMA domain i (4 tasks own the node).
SELECT_GPU=$RUN_DIR/select_gpu
printf '%s\n' \
  '#!/bin/bash' \
  '# 1:1 local rank -> GPU + NUMA binding; copied from run_probe_grid_mg.sh' \
  'export GPU=$SLURM_LOCALID' \
  'export NUMA=$SLURM_LOCALID' \
  'export CUDA_VISIBLE_DEVICES=$GPU' \
  'exec numactl -m $NUMA -N $NUMA "$@"' > "$SELECT_GPU"
chmod +x "$SELECT_GPU"

# --export=ALL passes this whole environment to every rank (an --export without
# ALL strips it: the ~10 s exit-1 "shell variable" failure, machines/perlmutter.md).
# SRUN_EXTRA (optional, space-separated) is appended verbatim, e.g. SRUN_EXTRA=--overlap
# when another step (a library build) already holds the node's cores and GPUs.
SRUN=(srun --jobid="$SLURM_JOB_ID" -N 1 -n "$NTASKS" --ntasks-per-node=4
      --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=cores --gpu-bind=none
      --export=ALL --chdir="$RUN_DIR" ${SRUN_EXTRA:+$SRUN_EXTRA})

# ---- 10. Gated NUMA binding ---------------------------------------------------
# HARD GATE, not a print (machines/perlmutter.md "Node binding"; L104; job
# 58201154 died because the bad binding was printed and not acted on). This
# runs exactly what select_gpu will run on every rank (numactl -m i -N i), the
# 4-rank form of `numactl -m 0 -N 0 true || exit 1`.
if ! "${SRUN[@]}" "$SELECT_GPU" true; then
  die "NUMA gate failed: a local rank cannot bind its own NUMA domain; select_gpu would fail. Nothing was run (rundir $RUN_DIR)."
fi
"${SRUN[@]}" --label "$SELECT_GPU" numactl --show 2>&1 | grep -E 'nodebind' > "$RUN_DIR/numa_binding.txt"

# ---- 11. Provenance: ENV line = line 1 of the log -----------------------------
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
# Grid embeds its configure-time GITHASH ("<sha>: (<refs>) [uncommited changes]")
# via Grid/util/version.cc; Grid_init prints it too ("Current Grid git commit hash=").
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
GRID_SHA=${GRID_HASH_LINE:0:40}
GRID_SHA=${GRID_SHA:-unknown}
case "$GRID_HASH_LINE" in *uncommit*) GRID_DIRTY=yes ;; *) GRID_DIRTY=no ;; esac

printf 'ENV SMOKE16_PUREGRID RUN=%s JOBID=%s BIN=%s BIN_SHA256=%s GRID_SHA=%s GRID_DIRTY=%s LATT=%s MPI=%s NTASKS=%s IMPORT_CFG=%s SEED=%s N_TRAJ=%s MDSTEPS=%s TRAJL=%s INTEGRATOR=%s GAUGE_INNER_MULT=%s LADDER=%s TAIL_LEVEL=%s MASS_LIGHT=%s MASS_STRANGE=%s CSW=%s BETA=%s U0=%s STOUT_RHO=%s STOUT_NSMEAR=%s RAT=%s/%s/%s TOL_DRV=%s TOL_ACT=%s TOL_STRANGE=%s METROP=on QUDA_ENV=%s MPICH_IPC=%s MPICH_RDMA=%s GRID_MG_RUNGS=%s GRID_MG_ENV="%s" GRID_FLAGS="%s"\n' \
  "$RUN" "$SLURM_JOB_ID" "$(basename "$BIN")" "$BIN_SHA256" "$GRID_SHA" "$GRID_DIRTY" \
  "$LATT" "$MPI_GEOM" "$NTASKS" "$(basename "$IMPORT_CFG")" "$HMC_SEED_OFFSET" "$N_TRAJ" \
  "$MDSTEPS" "$TRAJL" "$INTEGRATOR" "$GAUGE_INNER_MULT" "$HASEN_LADDER" "$HASEN_TAIL_LEVEL" \
  "$MASS_LIGHT" "$MASS_STRANGE" "$CSW" "$BETA" "$U0" "$STOUT_RHO" "$STOUT_NSMEAR" \
  "$RAT_LO" "$RAT_HI" "$RAT_DEGREE" "$TUNE_CG_TOL_DERIV" "$TUNE_CG_TOL_ACTION" \
  "$TUNE_CG_TOL_STRANGE" "$N_QUDA_ENV" "$MPICH_GPU_IPC_ENABLED" "$MPICH_RDMA_ENABLED_CUDA" \
  "${HASEN_GRID_MG_RUNGS:-none}" "$(compgen -e | grep '^GRID_MG_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
  "${GRID_ARGS[*]}" > "$LOG"
{
  printf 'GRID_HASH %s\n' "${GRID_HASH_LINE:-unknown}"
  printf 'BIN_PATH %s\n' "$BIN"
  sed 's/^/NUMA /' "$RUN_DIR/numa_binding.txt"
  printf 'CMD '
  printf '%q ' timeout -k 60s 60m "${SRUN[@]}" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}"
  printf '\n'
} >> "$LOG"

# ---- 12. Launch (one trajectory) ----------------------------------------------
# timeout wrapper: every run keeps one (_SUM.md perlmutter-ops); an aborted
# step can also leave srun hung (port plan s.7 "Hung srun after an abort").
echo "running $(basename "$BIN") -> $LOG"
timeout -k 60s 60m "${SRUN[@]}" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 13. Summary ----------------------------------------------------------------
# First "Plaquette: [ 1 ]" line = thin links, second = smeared links.
# The first [0][k] "Force average" lines are the first force call per level-0
# monomial: same config + same noise, the cleanest cross-binary force point.
# FORCES traj=1 is the per-monomial average over the whole trajectory.
echo "==== $RUN: exit=$rc  log: $LOG"
grep -a -E '^ENV ' "$LOG"
grep -a -E 'Current Grid git commit hash|Initial plaquette|CG tol:|Hasenbusch chain' "$LOG"
grep -a -E '\[0\]\[[0-9]+\] Force average' "$LOG" | head -6
grep -a -E 'Total H before trajectory|Total H after trajectory|exp\(-dH\)|Metropolis_test|Total time for trajectory|Plaquette: \[|FORCES traj=' "$LOG"
echo "log: $LOG"
exit "$rc"
