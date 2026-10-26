#!/bin/bash
# =============================================================================
# forces16_puregrid.sh -- ONE FORCES_ONLY pass of the pure-Grid compact-Schur
# driver (src/gen_qcd_hasenbusch_tune_compact_schur.cc) at 16^3x48 on one node
# (4 ranks), inside an EXISTING allocation: the plain force pass, or the stiffness
# screen with FORCES_STIFFNESS=1 (driver FORCES_ONLY block, 2026-10-14; handoff
# __docs/2026_10_14_handoff_stiffness_screen.md, step 1 item 7 = this gate).
# This script never allocates and never submits.
#
# Recipe = smoke16_puregrid.sh's, verbatim (same 16^3 config, ladder -0.245,-0.20,-0.10,
# tail outer, 2 levels, seed 200, production tolerances, same comms flags), so that a
# FORCES_STIFFNESS run with the same seed draws the same momenta and heatbath noise as
# a smoke16 trajectory and its undisplaced forces equal that trajectory's first
# update_P forces (the gate's comparison). The integrator knobs are inert in
# FORCES_ONLY (the driver returns before the Integrator is built) but exported so the
# [Ladder] / [Levels] banners and the ENV line match the trajectory's.
#
# Usage (one line):
#   SLURM_JOB_ID=<id> BIN=<binary> RUN=<name> bash forces16_puregrid.sh
# e.g. the plain pass:
#   SLURM_JOB_ID=<id> BIN=<hb>/bin/..._stock_m10 RUN=fo16_m10 bash forces16_puregrid.sh
# the stiffness screen, every direction, two displacements:
#   SLURM_JOB_ID=<id> BIN=<hb>/bin/..._stock_m10 RUN=stiff16_m10 FORCES_STIFFNESS=1 FORCES_STIFF_DIRS=each,P FORCES_STIFF_EPS=1e-3,2e-3 bash forces16_puregrid.sh
# The allocation (1 node) comes from `bash ~/.claude/bin/alloc_gpu.sh 1 <hours>` and is
# released with `release_alloc.sh <id>` (machines/perlmutter.md).
#
# Caller overrides, read BEFORE the scrub of section 3 (everything else is hard-coded):
#   FORCES_SAMPLES        default 1
#   FORCES_STIFFNESS=1    the stiffness screen instead of the plain pass; its knobs
#                         FORCES_STIFF_DIRS / _EPS / _ONESIDED / _ACTIONS / _GRAM pass
#                         through with the driver's semantics (its comment block). The
#                         binary must contain the FORCES_STIFFNESS string (m10 or later).
#   HMC_SEED_OFFSET       default 200 (the smoke16 seed)
#   MPI_GEOM              default 1.1.1.4; the Grid-MG route needs 1.1.2.2 (smoke16 s.7)
#   HASEN_GRID_*, GRID_MG_*   routes and tunables, not scrubbed (as in smoke16);
#                         HASEN_GRID_BATCH_SMEAR is inert here (never enters update_P)
#   SRUN_EXTRA            appended verbatim to srun (e.g. --overlap)
# FORCES_SKIP_* are not passed: the plain pass runs every monomial, the screen refuses them.
#
# Output: runs/<YYYY_M_D>_<RUN>/hmc.log (unpadded date). Line 1 is the ENV provenance
# line; per the conventions entry of __docs/_SUM.md that line, not this script, is the
# ground truth for what ran. Timeout 60 min (the screen with every direction and two
# displacements is ~20 force evaluations per monomial: minutes at 16^3).
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
# Caller overrides, captured before the scrub.
FO_SAMPLES=${FORCES_SAMPLES:-1}
FO_STIFF=${FORCES_STIFFNESS:-}
FO_STIFF_DIRS=${FORCES_STIFF_DIRS:-} FO_STIFF_EPS=${FORCES_STIFF_EPS:-} FO_STIFF_ONESIDED=${FORCES_STIFF_ONESIDED:-}
FO_STIFF_ACTIONS=${FORCES_STIFF_ACTIONS:-} FO_STIFF_GRAM=${FORCES_STIFF_GRAM:-}
FO_SEED=${HMC_SEED_OFFSET:-200}
[[ "$FO_SAMPLES" =~ ^[1-9][0-9]*$ ]] || die "FORCES_SAMPLES must be a positive integer, got '$FO_SAMPLES'"
[[ "$FO_SEED" =~ ^[0-9]+$ ]] || die "HMC_SEED_OFFSET must be a non-negative integer, got '$FO_SEED'"
[ -z "$FO_STIFF" ] || [ "$FO_STIFF" = 1 ] || die "FORCES_STIFFNESS must be 1 or unset, got '$FO_STIFF'"
if [ -n "$FO_STIFF" ]; then
  [ "$(grep -c -a -F FORCES_STIFFNESS "$BIN" || true)" -gt 0 ] || die "$BIN does not contain FORCES_STIFFNESS (a binary older than _stock_m10 would run the plain pass)"
fi

# The allocation must already exist; this script only attaches srun steps to it.
JOB_STATE=$(squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null | head -1)
[ "$JOB_STATE" = RUNNING ] || die "allocation $SLURM_JOB_ID is not RUNNING (state: '${JOB_STATE:-not found}'). This script never allocates."

# ---- 2. Environment + loader check -------------------------------------------
# machines/perlmutter.sh pins cray-mpich 9.0.1 + cudatoolkit 12.9 and sets LD_LIBRARY_PATH
# from CRAY_LD_LIBRARY_PATH (without it: libcudart.so.13, exit 127; machines/perlmutter.md).
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
[ "$N_QUDA" -eq 0 ] || die "ldd shows libquda in $BIN; this launcher is for pure-Grid builds only"

# ---- 3. Clear every knob this run does not set --------------------------------
# Same families and the same HASEN_GRID_ / GRID_MG_ exemption as smoke16_puregrid.sh
# (its section 3 explains each family). FORCES_* are cleared here and set in section 7.
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|NO_METROP$|IMPORT_CFG$)' | grep -v -E '^(HASEN_GRID_|GRID_MG_)')
N_QUDA_ENV=$(compgen -e | grep -c '^QUDA_' || true)
[ "$N_QUDA_ENV" -eq 0 ] || die "QUDA_* variables still set after clearing"
unset N_TRAJ
# Production defaults (2026-10-02, smoke16 section 3): every validated speed-up ON unless
# the caller says otherwise (VAR=0); a binary that predates a switch runs without it.
for v in HASEN_GRID_FUSED_CLOVER_FORCE HASEN_GRID_DEVICE_CB HASEN_GRID_BATCH_SMEAR HASEN_GRID_IMPORT_SKIP HASEN_GRID_SHARE_FIELDSTRENGTH HASEN_GRID_GPU_CLOVER_INV HASEN_GRID_CLOVER_STENCIL; do
  if [ -z "${!v:-}" ]; then
    if [ "$(grep -c -a -F "$v" "$BIN" || true)" -gt 0 ]; then
      export "$v=1"
    else
      echo "WARNING: $BIN predates $v: running without it" >&2
    fi
  fi
done

# ---- 4. Physics (16^3x48 cl3 ensemble) = smoke16 section 4 --------------------
export LATT=16.16.16.48
export IMPORT_CFG=$BASE/data/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime
[ -f "$IMPORT_CFG" ] || die "starting configuration not found: $IMPORT_CFG"
export BETA=6.1
export CSW=1.24930970916466
export U0=0.832605301399891
export MASS_LIGHT=-0.245
export MASS_STRANGE=-0.2050
export STOUT_RHO=0.125 STOUT_NSMEAR=1
export RAT_LO=1e-4 RAT_HI=100 RAT_DEGREE=20

# ---- 5. Hasenbusch ladder + integrator = smoke16 section 5 (legacy knobs) -----
export HASEN_LADDER=-0.245,-0.20,-0.10
export INTEGRATOR=ForceGradient
export HASEN_TAIL_LEVEL=outer
export GAUGE_INNER_MULT=2
export TRAJL=0.35355339059327379
export MDSTEPS=4

# ---- 6. Seed + tolerances = smoke16 section 6 ----------------------------------
export HMC_SEED_OFFSET=$FO_SEED
export TUNE_CG_TOL_DERIV=1e-11
export TUNE_CG_TOL_ACTION=1e-12
export TUNE_CG_TOL_STRANGE=1e-9

# ---- 7. FORCES_ONLY mode -------------------------------------------------------
export FORCES_ONLY=1
export FORCES_SAMPLES=$FO_SAMPLES
STIFF_TAG=none
if [ -n "$FO_STIFF" ]; then
  export FORCES_STIFFNESS=1
  [ -z "$FO_STIFF_DIRS" ]     || export FORCES_STIFF_DIRS=$FO_STIFF_DIRS
  [ -z "$FO_STIFF_EPS" ]      || export FORCES_STIFF_EPS=$FO_STIFF_EPS
  [ -z "$FO_STIFF_ONESIDED" ] || export FORCES_STIFF_ONESIDED=$FO_STIFF_ONESIDED
  [ -z "$FO_STIFF_ACTIONS" ]  || export FORCES_STIFF_ACTIONS=$FO_STIFF_ACTIONS
  [ -z "$FO_STIFF_GRAM" ]     || export FORCES_STIFF_GRAM=$FO_STIFF_GRAM
  STIFF_TAG="dirs=${FO_STIFF_DIRS:-default}/eps=${FO_STIFF_EPS:-default}/onesided=${FO_STIFF_ONESIDED:-0}/actions=${FO_STIFF_ACTIONS:-all}/gram=${FO_STIFF_GRAM:-P}"
fi

# ---- 8. MPI / comms / threads = smoke16 section 7 -------------------------------
export SLURM_CPU_BIND=cores
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=${MPICH_GPU_IPC_ENABLED:-1} MPICH_RDMA_ENABLED_CUDA=${MPICH_RDMA_ENABLED_CUDA:-1}
export MPICH_GPU_EAGER_REGISTER_HOST_MEM=${MPICH_GPU_EAGER_REGISTER_HOST_MEM:-0} MPICH_GPU_NO_ASYNC_MEMCPY=${MPICH_GPU_NO_ASYNC_MEMCPY:-0}
export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS=8
MPI_GEOM=${MPI_GEOM:-1.1.1.4}
NTASKS=4
GRID_ARGS=(--grid "$LATT" --mpi "$MPI_GEOM" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap)

# ---- 9. Run directory (refuse to overwrite) -------------------------------------
DATE_TAG=$(date +%Y_%-m_%-d)
RUN_DIR=$BASE/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
LOG=$RUN_DIR/hmc.log

# ---- 10. select_gpu wrapper + srun step shape = smoke16 section 9 ---------------
SELECT_GPU=$RUN_DIR/select_gpu
printf '%s\n' \
  '#!/bin/bash' \
  '# 1:1 local rank -> GPU + NUMA binding; copied from run_probe_grid_mg.sh' \
  'export GPU=$SLURM_LOCALID' \
  'export NUMA=$SLURM_LOCALID' \
  'export CUDA_VISIBLE_DEVICES=$GPU' \
  'exec numactl -m $NUMA -N $NUMA "$@"' > "$SELECT_GPU"
chmod +x "$SELECT_GPU"
SRUN=(srun --jobid="$SLURM_JOB_ID" -N 1 -n "$NTASKS" --ntasks-per-node=4
      --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=cores --gpu-bind=none
      --export=ALL --chdir="$RUN_DIR" ${SRUN_EXTRA:+$SRUN_EXTRA})

# ---- 11. Gated NUMA binding (machines/perlmutter.md "Node binding") ---------------
if ! "${SRUN[@]}" "$SELECT_GPU" true; then
  die "NUMA gate failed: a local rank cannot bind its own NUMA domain; select_gpu would fail. Nothing was run (rundir $RUN_DIR)."
fi
"${SRUN[@]}" --label "$SELECT_GPU" numactl --show 2>&1 | grep -E 'nodebind' > "$RUN_DIR/numa_binding.txt"

# ---- 12. Provenance: ENV line = line 1 of the log --------------------------------
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
GRID_SHA=${GRID_HASH_LINE:0:40}
GRID_SHA=${GRID_SHA:-unknown}
case "$GRID_HASH_LINE" in *uncommit*) GRID_DIRTY=yes ;; *) GRID_DIRTY=no ;; esac

printf 'ENV FORCES16_PUREGRID RUN=%s JOBID=%s BIN=%s BIN_SHA256=%s GRID_SHA=%s GRID_DIRTY=%s MODE=FORCES_ONLY FORCES_SAMPLES=%s FORCES_STIFFNESS=%s STIFF=%s LATT=%s MPI=%s NTASKS=%s IMPORT_CFG=%s SEED=%s MDSTEPS=%s TRAJL=%s INTEGRATOR=%s GAUGE_INNER_MULT=%s LADDER=%s TAIL_LEVEL=%s MASS_LIGHT=%s MASS_STRANGE=%s CSW=%s BETA=%s U0=%s STOUT_RHO=%s STOUT_NSMEAR=%s RAT=%s/%s/%s TOL_DRV=%s TOL_ACT=%s TOL_STRANGE=%s QUDA_ENV=%s MPICH_IPC=%s MPICH_RDMA=%s GRID_MG_RUNGS=%s GRID_MG_HB=%s GRID_MIXED=%s GRID_MIXED_HB=%s GRID_MG_ENV="%s" GRID_FLAGS="%s"\n' \
  "$RUN" "$SLURM_JOB_ID" "$(basename "$BIN")" "$BIN_SHA256" "$GRID_SHA" "$GRID_DIRTY" \
  "$FORCES_SAMPLES" "${FORCES_STIFFNESS:-0}" "$STIFF_TAG" \
  "$LATT" "$MPI_GEOM" "$NTASKS" "$(basename "$IMPORT_CFG")" "$HMC_SEED_OFFSET" \
  "$MDSTEPS" "$TRAJL" "$INTEGRATOR" "$GAUGE_INNER_MULT" "$HASEN_LADDER" "$HASEN_TAIL_LEVEL" \
  "$MASS_LIGHT" "$MASS_STRANGE" "$CSW" "$BETA" "$U0" "$STOUT_RHO" "$STOUT_NSMEAR" \
  "$RAT_LO" "$RAT_HI" "$RAT_DEGREE" "$TUNE_CG_TOL_DERIV" "$TUNE_CG_TOL_ACTION" \
  "$TUNE_CG_TOL_STRANGE" "$N_QUDA_ENV" "$MPICH_GPU_IPC_ENABLED" "$MPICH_RDMA_ENABLED_CUDA" \
  "${HASEN_GRID_MG_RUNGS:-none}" "${HASEN_GRID_MG_HEATBATH_RUNGS:-none}" \
  "${HASEN_GRID_MIXED_CG_RUNGS:-none}" "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}" \
  "$(compgen -e | grep -E '^(GRID_MG_|HASEN_GRID_|STRANGE_EVEN$)' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
  "${GRID_ARGS[*]}" > "$LOG"
{
  printf 'GRID_HASH %s\n' "${GRID_HASH_LINE:-unknown}"
  printf 'BIN_PATH %s\n' "$BIN"
  sed 's/^/NUMA /' "$RUN_DIR/numa_binding.txt"
  printf 'CMD '
  printf '%q ' timeout -k 60s 60m "${SRUN[@]}" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}"
  printf '\n'
} >> "$LOG"

# ---- 13. Launch (one FORCES_ONLY pass) --------------------------------------------
echo "running $(basename "$BIN") FORCES_ONLY samples=$FORCES_SAMPLES stiffness=${FORCES_STIFFNESS:-0} -> $LOG"
timeout -k 60s 60m "${SRUN[@]}" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 14. Summary ------------------------------------------------------------------
echo "==== $RUN: exit=$rc  log: $LOG"
grep -a -E '^ENV ' "$LOG"
grep -a -E 'Current Grid git commit hash|Initial plaquette|CG tol:|Hasenbusch chain|\[Ladder\]|\[Levels\]|Integrator: ' "$LOG"
grep -a -E 'FORCES_ONLY (level|samples=)|\[FORCES_ONLY\]' "$LOG"
grep -a -E 'FORCES_STIFF (meta|force|P |dir|gaugecheck|done)' "$LOG"
echo "log: $LOG"
exit "$rc"
