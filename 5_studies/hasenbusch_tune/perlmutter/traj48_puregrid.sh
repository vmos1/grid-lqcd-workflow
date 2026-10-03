#!/bin/bash
# =============================================================================
# traj48_puregrid.sh -- FULL HMC trajectories of the pure-Grid compact-Schur
# driver (src/gen_qcd_hasenbusch_tune_compact_schur.cc) at 48^3x96 on 4 GPU
# nodes (16 ranks, 1 GPU each), inside an EXISTING allocation. A gauge + RNG
# checkpoint is written after EVERY trajectory, and RESUME_FROM continues the
# Markov chain of a previous run of THIS launcher. Never allocates, never submits.
#
# Purpose: M3 step 2 of __docs/2026_09_29_pure_grid_stock_build_plan.md (section 4
# table), "the 48^3 trajectory launcher with checkpoint write + restart exercised",
# gate "a checkpoint written by 16 ranks reads back". C3 physics, production
# tolerances, Metropolis on; the same env block as forces48_puregrid.sh except that
# the integrator knobs are live here and the M3 step 1 solver routes are on.
#
# Usage (one line each):
#   SLURM_JOB_ID=<id> BIN=<binary> RUN=<name> bash traj48_puregrid.sh
#   SLURM_JOB_ID=<id> BIN=<binary> RUN=<name> RESUME_FROM=<previous run dir> bash traj48_puregrid.sh
# e.g.
#   SLURM_JOB_ID=<id> BIN=<hb>/bin/gen_qcd_hasenbusch_tune_compact_schur_stock RUN=traj48_c3_s1 bash traj48_puregrid.sh
#   SLURM_JOB_ID=<id> BIN=<hb>/bin/gen_qcd_hasenbusch_tune_compact_schur_stock RUN=traj48_c3_s2 RESUME_FROM=<base>/runs/<date>_traj48_c3_s1 bash traj48_puregrid.sh
# The allocation (4 nodes) comes from `bash ~/.claude/bin/alloc_gpu.sh 4 <hours>` and is
# released with `release_alloc.sh <id>` (machines/perlmutter.md). That wrapper asks for
# `-C gpu`, not `gpu&hbm80g`: the card size you got (40 or 80 GB) is recorded by Grid's
# totalGlobalMem line and nvidia-smi memory.total. Never merge 40 and 80 GB rows (L119).
#
# Caller overrides (everything else is hard-coded on purpose):
#   N_TRAJ              default 1: trajectories in THIS run
#   HMC_SEED_OFFSET     default 0; fresh start only (a resume restores the RNG state)
#   RESUME_FROM         run dir of an earlier traj48_puregrid.sh run: restart (section 11)
#   RESUME_TRAJ         default: RESUME_FROM's LAST complete checkpoint. An earlier one
#                       REPLAYS trajectories that run already made (the replay test)
#   HASEN_GRID_MG_RUNGS                default "0,1,2"  } rung solver routes, section 8.
#   HASEN_GRID_MG_HEATBATH_RUNGS       default "0,1"    } Each may be set EMPTY (off),
#   HASEN_GRID_MIXED_CG_RUNGS          default "3"      } e.g. HASEN_GRID_MG_HEATBATH_RUNGS=
#   HASEN_GRID_MIXED_CG_HEATBATH_RUNGS default "2,3"    }
#   GRID_MG_*           pass through untouched (GridMGParams::from_env rejects typos)
#   CHECKPOINT_ROOT     default $PSCRATCH/grid_pure_hmc/ckpt; "rundir" = <rundir>/ckpt on CFS
#   TIMEOUT_MIN         default 210 (3.5 h)
#   DEVICE_MEM_MB       default 8000 (--device-mem)
#   INTEGRATOR_VERBOSE_MEM  default 0 (off); 1 = MemoryManager::Print per force call
#   GPU_MON_MS          default 2000 (nvidia-smi sampling period); 0 = no sampler
#   SRUN_EXTRA          appended verbatim to srun (e.g. --overlap)
#   DIAG_TOL_STRANGE, DIAG_MDSTEPS, DIAG_TRAJL  diagnostic overrides (2026-10-02, physics review
#                       tests T1-T3): strange multishift tolerance, MD steps, trajectory length.
#                       Unset = production. The ENV line records the values actually used.
#
# Output: runs/<YYYY_M_D>_<RUN>/hmc.log (unpadded date). Line 1 is the ENV provenance
# line; per the conventions entry of __docs/_SUM.md that line, not this script, is the
# ground truth for what ran. <rundir>/ckpt -> the checkpoint directory (section 12),
# CKPT_LOCATION.txt, gpu_mem.<node>.csv + gpu_mem_peak.txt when GPU_MON_MS > 0.
#
# Wall time: a pure-Grid C3 trajectory is currently about 2 h (build plan s.6 projects
# ~8,000 s before M3 step 1's heatbath MG). Plus ~2-5 min of startup (6.1 GB config or
# checkpoint read, operator setup) and ~10 s per checkpoint write. Default timeout
# 210 min fits ONE trajectory. A timeout loses only the trajectory in flight: every
# completed one is checkpointed, and RESUME_FROM picks the chain up from there.
# =============================================================================
set -o pipefail

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
WORKFLOW=$BASE/grid-lqcd-workflow
SELF=$(readlink -f "$0")

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
USAGE="usage: SLURM_JOB_ID=<id> BIN=<binary> RUN=<name> [RESUME_FROM=<run dir>] bash $0"

# squeue %L ([D-]HH:MM:SS or MM:SS) -> whole minutes; prints nothing if not a duration.
to_minutes() {
  [[ "$1" =~ ^(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)$ ]] || return 0
  echo $(( 10#${BASH_REMATCH[2]:-0} * 1440 + 10#${BASH_REMATCH[4]:-0} * 60 + 10#${BASH_REMATCH[5]} ))
}

# Trajectory numbers whose checkpoint is COMPLETE in a log of this driver: both the
# "Written BINARY RNG <dir>/ckpoint_rng.<n>" and the "Written ILDG Configuration on
# <dir>/ckpoint_lat.<n>" line are present (the checkpointer prints each only after
# that file's write and close returned; ILDGCheckpointer.h TrajectoryComplete). One
# number per line, ascending. The smeared file (ckpoint_lat_smr.<n>) never matches.
complete_ckpts() {
  awk '/Written BINARY RNG / { f = $0; sub(/ checksum .*/, "", f)
                               if (f ~ /ckpoint_rng\.[0-9]+$/) { sub(/.*ckpoint_rng\./, "", f); r[f] = 1 } }
       /Written ILDG Configuration on / { f = $0; sub(/ checksum .*/, "", f)
                               if (f ~ /ckpoint_lat\.[0-9]+$/) { sub(/.*ckpoint_lat\./, "", f); l[f] = 1 } }
       END { for (n in l) if (n in r) print n }' "$1" | sort -n
}

# Per-trajectory table of one log, rows lo..hi (saved trajectory numbers). Columns:
# M = Metropolis (ACC/REJ); wall_s = "Total H before" -> "Total H after" (the MD, the
# PRIMARY Grid timing family, _SUM.md conventions); cycle_s = Grid's "Total time for
# trajectory" (adds heatbath, momentum refresh, Metropolis; NOT the checkpoint or the
# observables, which HMC.h:292 runs after that stamp at :283); ckpt_s = RNG write start
# -> "Written ILDG Configuration" (both Message-channel stamps); plaq = the unsmeared
# then the smeared "Plaquette: [ n ]" (on a REJECT these are the START config's).
traj_table() {
  awk -v lo="$2" -v hi="$3" '
    function after(key,   i) { for (i = 1; i < NF; i++) if ($i == key) return $(i + 1); return "" }
    /-- # Trajectory = /              { n = $NF + 1; if (!(n in seen)) { seen[n] = 1; ord[++cnt] = n } }
    /Total H before trajectory = /    { hb[n] = $NF; tb[n] = $5 }
    /Total H after trajectory /       { ha[n] = after("="); dh[n] = $NF; ta[n] = $5 }
    /exp\(-dH\) = /                   { ex[n] = after("=") }
    /Metropolis_test -- ACCEPTED/     { ac[n] = "ACC" }
    /Metropolis_test -- REJECTED/     { ac[n] = "REJ" }
    /Total time for trajectory \(s\): / { cy[n] = $NF }
    /Plaquette: \[ /                  { k = after("["); if (!(k in pu)) pu[k] = $NF; else if (!(k in ps)) ps[k] = $NF }
    /RNG write I\/O on file /         { f = $NF; if (f ~ /ckpoint_rng\.[0-9]+$/) { sub(/.*ckpoint_rng\./, "", f); cw[f] = $5 } }
    /Written ILDG Configuration on /  { f = $0; sub(/ checksum .*/, "", f)
                                        if (f ~ /ckpoint_lat\.[0-9]+$/) { sub(/.*ckpoint_lat\./, "", f); ck[f] = $5 } }
    END {
      printf "%-6s %-3s %-20s %-20s %-20s %-10s %-8s %-8s %-7s %-18s %-18s\n", "traj", "M", "H_before", "H_after", "dH", "exp(-dH)", "wall_s", "cycle_s", "ckpt_s", "plaq_thin", "plaq_smeared"
      na = 0; nd = 0
      for (j = 1; j <= cnt; j++) {
        n = ord[j]; if (n + 0 < lo + 0 || n + 0 > hi + 0) continue
        w = ((n in tb) && (n in ta)) ? sprintf("%.1f", ta[n] - tb[n]) : "-"
        c = ((n in cw) && (n in ck)) ? sprintf("%.1f", ck[n] - cw[n]) : "-"
        printf "%-6s %-3s %-20s %-20s %-20s %-10s %-8s %-8s %-7s %-18s %-18s\n", n, (n in ac) ? ac[n] : "-", (n in hb) ? hb[n] : "-", (n in ha) ? ha[n] : "-", (n in dh) ? dh[n] : "-", (n in ex) ? ex[n] : "-", w, (n in cy) ? cy[n] : "-", c, (n in pu) ? pu[n] : "-", (n in ps) ? ps[n] : "-"
        if (n in ac) { nd++; if (ac[n] == "ACC") na++ }
      }
      printf "Metropolis: %d accepted of %d decided\n", na, nd
    }' "$1"
}

# ---- 1. Inputs ---------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || die "SLURM_JOB_ID is not set (the id of an existing 4-node allocation). $USAGE"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be a numeric job id, got '$SLURM_JOB_ID'"
[ -n "${BIN:-}" ] || die "BIN is not set (path to the driver binary). $USAGE"
[ -n "${RUN:-}" ] || die "RUN is not set (run name, becomes runs/<date>_<RUN>). $USAGE"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN may contain only letters, digits, '.', '_' and '-', got '$RUN'"
{ [ -f "$BIN" ] && [ -x "$BIN" ]; } || die "BIN is not an executable file: $BIN"
BIN=$(readlink -f "$BIN")
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
DATE_TAG=$(date +%Y_%-m_%-d)                              # unpadded, runs/ convention

# The allocation must already exist and span 4 nodes; this script only attaches
# srun steps to it. (squeue is read-only.)
JOB_STATE=$(squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null | head -1)
[ "$JOB_STATE" = RUNNING ] || die "allocation $SLURM_JOB_ID is not RUNNING (state: '${JOB_STATE:-not found}'). This script never allocates."
JOB_NODES=$(squeue -h -j "$SLURM_JOB_ID" -o %D 2>/dev/null | head -1)
{ [[ "$JOB_NODES" =~ ^[0-9]+$ ]] && [ "$JOB_NODES" -ge 4 ]; } || die "allocation $SLURM_JOB_ID has '${JOB_NODES:-?}' node(s); this run needs 4"
ALLOC_LEFT_MIN=$(to_minutes "$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | head -1)")

# ---- 2. Caller overrides (read BEFORE the scrub in section 4) ----------------------
# The scrub clears the INTEGRATOR_VERBOSE_MEM and HASEN_* families, so every override
# is captured here first. "+x" / "-" where an EMPTY value means something (route off).
N_TRAJ=${N_TRAJ:-1}
SEED_REQ=${HMC_SEED_OFFSET:-}
RESUME_FROM=${RESUME_FROM:-}
RESUME_TRAJ_REQ=${RESUME_TRAJ:-}
TIMEOUT_MIN=${TIMEOUT_MIN:-210}
DEVICE_MEM_MB=${DEVICE_MEM_MB:-8000}
GPU_MON_MS=${GPU_MON_MS:-2000}
VERBOSE_MEM_REQ=${INTEGRATOR_VERBOSE_MEM:-0}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${PSCRATCH:+$PSCRATCH/grid_pure_hmc/ckpt}}
if [ -z "${HASEN_GRID_MG_RUNGS+x}" ]; then HASEN_GRID_MG_RUNGS=0,1,2; fi
if [ -z "${HASEN_GRID_MG_HEATBATH_RUNGS+x}" ]; then HASEN_GRID_MG_HEATBATH_RUNGS=0,1; fi
if [ -z "${HASEN_GRID_MIXED_CG_RUNGS+x}" ]; then HASEN_GRID_MIXED_CG_RUNGS=3; fi
if [ -z "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS+x}" ]; then HASEN_GRID_MIXED_CG_HEATBATH_RUNGS=2,3; fi
[[ "$N_TRAJ" =~ ^[1-9][0-9]*$ ]] || die "N_TRAJ must be a positive integer, got '$N_TRAJ'"
[ -z "$SEED_REQ" ] || [[ "$SEED_REQ" =~ ^[0-9]+$ ]] || die "HMC_SEED_OFFSET must be a non-negative integer, got '$SEED_REQ'"
[ -z "$RESUME_TRAJ_REQ" ] || [ -n "$RESUME_FROM" ] || die "RESUME_TRAJ=$RESUME_TRAJ_REQ needs RESUME_FROM=<run dir>"
[ -z "$RESUME_TRAJ_REQ" ] || [[ "$RESUME_TRAJ_REQ" =~ ^[0-9]+$ ]] || die "RESUME_TRAJ must be a trajectory number, got '$RESUME_TRAJ_REQ'"
[[ "$TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]] || die "TIMEOUT_MIN must be a positive integer (minutes), got '$TIMEOUT_MIN'"
[[ "$DEVICE_MEM_MB" =~ ^[1-9][0-9]*$ ]] || die "DEVICE_MEM_MB must be a positive integer (MB), got '$DEVICE_MEM_MB'"
[[ "$GPU_MON_MS" =~ ^[0-9]+$ ]] || die "GPU_MON_MS must be a non-negative integer (ms), got '$GPU_MON_MS'"
[ -n "$CHECKPOINT_ROOT" ] || die "CHECKPOINT_ROOT is empty and \$PSCRATCH is not set; give CHECKPOINT_ROOT=<absolute dir> or CHECKPOINT_ROOT=rundir"
[ "$CHECKPOINT_ROOT" = rundir ] || [[ "$CHECKPOINT_ROOT" = /* ]] || die "CHECKPOINT_ROOT must be an absolute path or 'rundir', got '$CHECKPOINT_ROOT'"

# Wall-time sanity (warnings only). If the allocation ends first, Slurm kills the step,
# possibly inside a checkpoint write; section 11 then skips that incomplete checkpoint.
if [ -n "$ALLOC_LEFT_MIN" ] && [ "$ALLOC_LEFT_MIN" -lt $(( TIMEOUT_MIN + 2 )) ]; then
  warn "allocation $SLURM_JOB_ID has ${ALLOC_LEFT_MIN} min left, less than TIMEOUT_MIN=${TIMEOUT_MIN}: Slurm, not the timeout, will end this run"
fi
if [ $(( N_TRAJ * 120 + 10 )) -gt "$TIMEOUT_MIN" ]; then
  warn "N_TRAJ=$N_TRAJ at ~120 min per trajectory will not all finish within TIMEOUT_MIN=$TIMEOUT_MIN; completed ones are checkpointed and resumable"
fi

# ---- 3. Environment + loader check -------------------------------------------
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
[ "$N_QUDA" -eq 0 ] || die "ldd shows libquda in $BIN; this launcher is for pure-Grid builds only"

# ---- 4. Clear every knob this run does not set -----------------------------------
# Same families as forces48_puregrid.sh / smoke16_puregrid.sh, with the exemption
# widened from HASEN_GRID_MG_ to HASEN_GRID_ (the M3 step 1 routes are HASEN_GRID_*;
# section 8 checks each against the binary). A stray variable from the caller's shell
# would silently change the run, or kill it: in a pure-Grid build HASEN_MG_RUNG /
# HASEN_QUDA_CG_RUNGS / HASEN_QUDA_CG_TAIL / HASEN_QUDA_FORCE_LOGDET exit(1) "requires a
# QUDA build", and any QUDA rung route beside HASEN_GRID_MG_RUNGS exits "a rung has ONE
# solver route" (driver source). Families cleared:
#   QUDA_*, HASEN_*, HMC_MG_*, USE_HMC_MG  QUDA / MG routing (driver + fork headers)
#   FORCES_*                               FORCES_ONLY mode: must be OFF for trajectories
#   CKPT_*                                 checkpointer: reset, then set in sections 11-12
#   TXQCD_*, WCF_*                         fork-library env gates (all default off; the
#                                          LogDet GPU paths default ON under CUDA, build
#                                          plan s.6, as in every production run)
#   LAMBDA_MN2                             fork-only MN2 knob
#   INTEGRATOR_VERBOSE_MEM                 MemoryManager::Print gate (patch 03): s.10
#   STRANGE_EVEN, NO_METROP, IMPORT_CFG    reset; NO_METROP stays unset (Metropolis on
#                                          from the first trajectory, the SOP); IMPORT_CFG
#                                          set in section 11 for a fresh start only
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|STRANGE_EVEN$|NO_METROP$|IMPORT_CFG$)' | grep -v -E '^(HASEN_GRID_|GRID_MG_)')
N_QUDA_ENV=$(compgen -e | grep -c '^QUDA_' || true)
[ "$N_QUDA_ENV" -eq 0 ] || die "QUDA_* variables still set after clearing"

# ---- 5. Physics (cl21 48^3x96 ensemble, Chroma cfg_2000 action) ---------------
# All from submit_scripts/hmc48_compact/chroma_match_sop_48.sh section A (confirmed
# 2026-07-13), identical to forces48_puregrid.sh. Every value is exported even where a
# default exists, so no params.h / driver default can differ from production.
export LATT=48.48.48.96
START_CFG=$BASE/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime   # ILDG, 6.1 GB
START_TRAJ=2000                                           # its trajectory number (section 11)
export BETA=6.3
# U0: the driver's LOCAL default is 1.0 and shadows params.h; the wrong U0 was the
# 48^3 blow-up root cause (__docs/_TOC.md u0-48cube). SOP value:
export U0=0.84570646270714
export CSW=1.20536588031793
export MASS_LIGHT=-0.2416                                 # must equal HASEN_LADDER[0]
export MASS_STRANGE=-0.2050
# STOUT: params.h defaults (0.125 / 1), which every production 48^3 run inherited.
export STOUT_RHO=0.125 STOUT_NSMEAR=1
export RAT_LO=0.4 RAT_HI=35.0 RAT_DEGREE=20               # strange RHMC bounds (48^3 values)

# ---- 6. C3 ladder, level layout, integrator (LIVE here) ---------------------------
# C3 = the w3 ladder with the tail on strange's level
# (submit_scripts/hasenbusch_tune/2026_8_21_lvlroute_w3_tailmiddle_extend_2011_2020.sh,
# on top of the SOP): 4 ratio rungs PF0..PF3 + bare-det tail at -0.1870, 3-level
# ForceGradient: light rungs + LogDets outer (MDSTEPS=6), strange + tail middle
# (x1), gauge inner (x2).
# LADDER_PROFILE=baseG (default since 2026-10-01) or c3 (the production C3 above). baseG is the
# SOP's baseline, the configuration of the Grid-vs-Chroma comparison 2026_08_24_grid_vs_chroma_comparison_summary.md:
# ladder -0.2416,-0.2400,-0.2320,-0.2180,-0.1870, tail on the inner (gauge) level, MDSTEPS=12,
# strange level at the driver default; chroma_match_sop_48.sh lines 23-30).
LADDER_PROFILE=${LADDER_PROFILE:-baseG}
case "$LADDER_PROFILE" in
  c3)
    export HASEN_LADDER=-0.2416,-0.2380,-0.2340,-0.2180,-0.1870
    export HASEN_STRANGE_LEVEL=middle
    export HASEN_STRANGE_INNER_MULT=1
    export HASEN_TAIL_LEVEL=middle
    export MDSTEPS=6                                      # C3 (SOP has 12)
    ;;
  baseG)
    export HASEN_LADDER=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870   # SOP: base+G
    unset HASEN_STRANGE_LEVEL HASEN_STRANGE_INNER_MULT
    export HASEN_TAIL_LEVEL=inner                         # SOP
    export MDSTEPS=12                                     # SOP
    ;;
  # c1 and c2 (2026-10-02): the other two tuned candidates of the c1c2c3 campaign, from the ENV
  # lines of the hybrid chains runs/2026_8_13_u1a_accept10_seed300_48 (C1 = u1a: four masses,
  # three ratio rungs, a heavy bare-det tail at +0.3044 on the light level, 2 levels, MDSTEPS 12)
  # and runs/2026_8_10_3level_w3_mdsteps5_48 (C2 = w3@5: C3's ladder in the 3-level layout with
  # the tail on the gauge level, MDSTEPS 5). The rung routes below assume four ratio rungs: with
  # c1 set HASEN_GRID_MIXED_CG_RUNGS= (empty) and HASEN_GRID_MIXED_CG_HEATBATH_RUNGS=2 yourself.
  c1)
    export HASEN_LADDER=-0.2416,-0.2250,-0.1870,0.3044
    unset HASEN_STRANGE_LEVEL HASEN_STRANGE_INNER_MULT
    export HASEN_TAIL_LEVEL=outer
    export MDSTEPS=12
    ;;
  c2)
    export HASEN_LADDER=-0.2416,-0.2380,-0.2340,-0.2180,-0.1870
    export HASEN_STRANGE_LEVEL=middle
    export HASEN_STRANGE_INNER_MULT=1
    export HASEN_TAIL_LEVEL=inner
    export MDSTEPS=5
    ;;
  *) die "LADDER_PROFILE must be c1, c2, c3 or baseG, got $LADDER_PROFILE" ;;
esac
export MDSTEPS=${DIAG_MDSTEPS:-$MDSTEPS}                        # DIAG_MDSTEPS: diagnostic override of the profile (header)
export INTEGRATOR=ForceGradient                           # SOP section B
export GAUGE_INNER_MULT=2                                 # SOP section B
# TRAJL is MANDATORY: the driver exits "FATAL: TRAJL is not set" before the config load.
export TRAJL=${DIAG_TRAJL:-0.35355339059327379}       # sqrt(2)/4, SOP section B. DIAG_TRAJL: diagnostic override (header)

# ---- 7. Trajectories, seed, tolerances --------------------------------------------
export N_TRAJ
# Seed: HMC_SEED_OFFSET=k shifts the fixed seeds {11..15} (serial) and {16..20}
# (parallel) by k; 0 == the historical fixed seeds (driver RNG block). It is exported
# in section 11: the caller's value on a fresh start, 0 (and irrelevant) on a resume,
# where CheckpointRestore overwrites both RNGs after seeding.
# PRODUCTION tolerances (chroma_match_sop_48.sh section C; __docs/_SUM.md conventions).
export TUNE_CG_TOL_DERIV=1e-11
export TUNE_CG_TOL_ACTION=1e-12
export TUNE_CG_TOL_STRANGE=${DIAG_TOL_STRANGE:-1e-9}   # DIAG_TOL_STRANGE: diagnostic override (header)
# Metropolis ON: NO_METROP unset (section 4) -> NoMetropolisUntil=0, MetropolisTest=true
# (driver HMCparameters block). NoMetropolisUntil counts from each run's own
# StartTrajectory (HMC.h evolve), so it would restart on every segment: never set it.

# ---- 8. Rung solver routes (pure Grid, no QUDA) -------------------------------------
# HASEN_GRID_MG_RUNGS (M2): these rungs' deriv + action solves use the fp32 three-level
#   Grid multigrid; the lowest listed rung donates the one hierarchy, built lazily at its
#   first solve (driver HASEN_GRID_MG_RUNGS block). 0,1,2 = the hybrid's C3 routing (MG 0
#   + shared 1,2; SOP section D).
# HASEN_GRID_MG_HEATBATH_RUNGS, HASEN_GRID_MIXED_CG_RUNGS, HASEN_GRID_MIXED_CG_HEATBATH_RUNGS
#   (M3 step 1, build plan s.4): MG heatbath on rungs 0,1 (the force-only SOP's validated
#   HASEN_MG_HEATBATH_RUNGS=0,1), mixed-precision CG for the remaining CG solves. These
#   are being added to the driver on 2026-09-30 and were NOT in the source when this
#   launcher was written: their exact semantics are the driver's; here each is only
#   validated as a list of C3 ratio rungs, checked against the binary, and exported.
# A binary that lacks a knob's name ignores it and silently runs Grid CG (the M2 stock
# binary of 2026-09-29 19:21 has none of the three M3 names). So every non-empty
# HASEN_GRID_* knob must appear as a string in $BIN, or the run is refused; set a knob
# EMPTY to run without that route. Every other HASEN_GRID_* the caller set passes through
# under the same check (a typo'd name is refused), and is recorded in the ENV line.
ROUTE_KNOBS=(HASEN_GRID_MG_RUNGS HASEN_GRID_MG_HEATBATH_RUNGS HASEN_GRID_MIXED_CG_RUNGS HASEN_GRID_MIXED_CG_HEATBATH_RUNGS)
for v in "${ROUTE_KNOBS[@]}"; do
  val=${!v:-}
  if [ -n "$val" ]; then
    [[ "$val" =~ ^[0-3](,[0-3])*$ ]] || die "$v must list ratio rungs 0..3 of the C3 ladder (PF0..PF3), got '$val'"
    N_STR=$(grep -c -a -F "$v" "$BIN" || true)
    [ "$N_STR" -gt 0 ] || die "$BIN has no $v support: it would silently run Grid CG on rung(s) $val. Use a binary built with it, or set $v= (empty)."
    export "$v"
  else
    unset "$v"
  fi
done
# One solver route per rung (the driver's rule), checked here before 4 nodes spend
# minutes reading 6.1 GB: deriv/action lists disjoint, heatbath lists disjoint.
overlap() {
  local a=",$1," k out=
  for k in ${2//,/ }; do case "$a" in *",$k,"*) out="$out $k" ;; esac; done
  printf '%s' "$out"
}
OV=$(overlap "${HASEN_GRID_MG_RUNGS:-}" "${HASEN_GRID_MIXED_CG_RUNGS:-}")
[ -z "$OV" ] || die "rung(s)$OV in both HASEN_GRID_MG_RUNGS and HASEN_GRID_MIXED_CG_RUNGS: a rung has ONE solver route"
OV=$(overlap "${HASEN_GRID_MG_HEATBATH_RUNGS:-}" "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-}")
[ -z "$OV" ] || die "rung(s)$OV in both HASEN_GRID_MG_HEATBATH_RUNGS and HASEN_GRID_MIXED_CG_HEATBATH_RUNGS: a rung has ONE heatbath route"
# ---- production defaults (2026-10-02): every validated speed-up ON unless the caller says
# otherwise. Override with VAR=0 for a diagnostic run (HASEN_GRID_BATCH_SMEAR=0 restores the
# per-monomial post-pullback "Force average" lines for kick-size work). A binary that predates a
# switch (no such name in it) runs without it, with a warning, instead of refusing. Measured
# together on base+G: 3,290 s wall vs the hybrid's 3,256 (force-cost analysis s.7.5; one-page
# summary 2026_10_02_pure_grid_speedup_summary.md).
for v in HASEN_GRID_FUSED_CLOVER_FORCE HASEN_GRID_DEVICE_CB HASEN_GRID_BATCH_SMEAR HASEN_GRID_IMPORT_SKIP HASEN_GRID_SHARE_FIELDSTRENGTH HASEN_GRID_GPU_CLOVER_INV HASEN_GRID_CLOVER_STENCIL; do
  if [ -z "${!v:-}" ]; then
    if [ "$(grep -c -a -F "$v" "$BIN" || true)" -gt 0 ]; then
      export "$v=1"
    else
      echo "WARNING: $BIN predates $v: running without it" >&2
    fi
  fi
done
HASEN_GRID_EXTRA=
while read -r v; do
  [ -n "$v" ] || continue
  HASEN_GRID_EXTRA="$HASEN_GRID_EXTRA$v=${!v} "
  if [ "${!v}" = 0 ]; then continue; fi   # an explicit off is harmless for a binary without the switch
  N_STR=$(grep -c -a -F "$v" "$BIN" || true)
  [ "$N_STR" -gt 0 ] || die "$v is set but $BIN does not contain that name (typo, or an older binary); unset it"
done < <(compgen -e | grep '^HASEN_GRID_' | grep -v -x -E 'HASEN_GRID_(MG_RUNGS|MG_HEATBATH_RUNGS|MIXED_CG_RUNGS|MIXED_CG_HEATBATH_RUNGS)')
GRID_MG_ENV=$(compgen -e | grep -E '^(GRID_MG_|HASEN_GRID_FUSED)' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)

# ---- 9. MPI / comms / threads -------------------------------------------------------
# The probe's pure-Grid flag set, exactly as forces48_puregrid.sh
# (6_benchmarks/grid_quda_wilson_clover/perlmutter/run_probe_grid_mg.sh). NOT the
# SOP's --shm-mpi 1 --comms-sequential: those inflate Grid's clover solve 5.16x (L130).
# MPICH GPU IPC + GPU-direct RDMA ON by default since 2026-10-02 (the 09-01 Grid-developer
# config, L187: 9-35% faster per force piece, forces identical, no hang under the select_gpu
# + --gpu-bind=none model; the June "=0" rule predated that model). Caller-overridable (=0).
# NIC policy GPU = the halo-nic fix (L122), live here: 4 nodes, inter-node halos.
export SLURM_CPU_BIND=cores
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=${MPICH_GPU_IPC_ENABLED:-1} MPICH_RDMA_ENABLED_CUDA=${MPICH_RDMA_ENABLED_CUDA:-1}
export MPICH_GPU_EAGER_REGISTER_HOST_MEM=${MPICH_GPU_EAGER_REGISTER_HOST_MEM:-0} MPICH_GPU_NO_ASYNC_MEMCPY=${MPICH_GPU_NO_ASYNC_MEMCPY:-0}
export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS=8                                  # run_probe_grid_mg.sh
# 1.2.2.4 = the SOP's and the MG probe's C3 split: local 48.24.24.24.
MPI_GEOM=1.2.2.4
NODES=4
NTASKS=16
# --device-mem 8000: what the 48^3 MG needed; 12000 OOMs (L119, L167). The 48^3
# FORCES_ONLY probe peaked at 25.5-26.3 GB per 40 GB GPU with it (build plan s.6).
GRID_ARGS=(--grid "$LATT" --mpi "$MPI_GEOM" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap --device-mem "$DEVICE_MEM_MB")

# ---- 10. Device-memory reporting ------------------------------------------------------
# INTEGRATOR_VERBOSE_MEM=1 re-enables patch 03's MemoryManager::Print around every force
# evaluation (Integrator.h update_P and S). LIVE in a trajectory run, and each print ends
# in cudaMemGetInfo, a device sync: such a run is a memory probe, not a timing to quote.
# Hence OFF by default here (forces48_puregrid.sh defaults it on because it is inert in
# FORCES_ONLY). The nvidia-smi sampler (GPU_MON_MS) costs nothing on the GPU streams.
case "$VERBOSE_MEM_REQ" in
  ''|0) VERBOSE_MEM=off ;;
  *)    export INTEGRATOR_VERBOSE_MEM=1; VERBOSE_MEM=on ;;
esac

# ---- 11. Start: fresh from cfg_2000, or RESUME from a checkpoint ------------------------
# The driver's checkpointer (driver "Checkpointer (roadmap D)" block; lines 311-360 as read
# 2026-09-30) is Grid's stock ILDGHmcCheckpointer, switched on by CKPT_DIR:
#   CKPT_DIR=<dir>         prefixes <dir>/ckpoint_lat, <dir>/ckpoint_lat_smr, <dir>/ckpoint_rng
#                          (the string is used verbatim, so an absolute path works);
#                          format IEEE64BIG; saveSmeared = false HARD-CODED (no env knob;
#                          Grid would write ckpoint_lat_smr.<n> if it were true). The smeared
#                          links are a deterministic function of the thin ones (stout 0.125 x1),
#                          so a restart needs only ckpoint_lat + ckpoint_rng.
#   CKPT_INTERVAL=<n>      save every n-th trajectory (default 1; set to 1 here).
#   CKPT_START_TRAJ=<n>    numbering offset of a FRESH start from IMPORT_CFG.
#   CKPT_RESUME_TRAJ=<n>   Ckpt->CheckpointRestore(n, Umu, sRNG, pRNG): reads the gauge field
#                          from <CKPT_DIR>/ckpoint_lat.<n> and BOTH RNGs (serial + parallel)
#                          from <CKPT_DIR>/ckpoint_rng.<n>, overwriting the seeded state;
#                          IMPORT_CFG is ignored; StartTrajectory := n. Needs CKPT_DIR (FATAL
#                          otherwise). The read and the write share ONE directory prefix.
# HMC.h evolve runs traj = StartTrajectory .. StartTrajectory+N_TRAJ-1 and calls every
# observable's TrajectoryComplete(traj+1) after the Metropolis step, ACCEPT OR REJECT (a
# reject re-saves the unchanged gauge field with the advanced RNG: the stream stays
# continuous). The checkpointer is the last observable, after FORCES / Plaquette /
# Polyakov: "Written BINARY RNG <dir>/ckpoint_rng.<n>" then "Written ILDG Configuration
# on <dir>/ckpoint_lat.<n>". Both lines print the RNG's checksums (Grid reuses the
# variables); the gauge file's own SciDAC checksum + plaquette are verified on read.
# HMCparameters "Starting type : ColdStart" in the banner is cosmetic: HybridMonteCarlo
# never reads it; the driver's gauge-field block decides the start.
# Numbering: fresh CKPT_START_TRAJ=2000 -> "-- # Trajectory = 2000", files and
# "Plaquette: [ 2001 ]" .. 2000+N_TRAJ, the numbering of the hybrid C3 chain from the same
# cfg_2000. Resume from n -> n+1 .. n+N_TRAJ.
# The driver HAS an RNG restart, so no IMPORT_CFG + new-seed fallback is needed: a resume
# continues the same random stream. It is the same Markov chain up to solver tolerance, not
# bitwise: the Grid-MG hierarchy is rebuilt from scratch at the resumed run's first solve,
# where an unbroken run carries it over (GRID_MG_SEED reseeds each build, so the HMC RNG
# stream itself is untouched). Validate with plaquette + H continuity, never bits (_SUM.md
# logdet-qforce "Do not repeat").
# Writer: the multi-rank ILDG write needs patch 02's boss-node guard in IldgWriter::
# writeLimeIldgLFN (L053/L054, 2026_07_15_checkpointer_boss_guard_fix.md); the stock stage
# carries it (PROVENANCE.txt patch 02; staged IldgIO.h). Pre-fix signature: a ~3.7 KB
# ckpoint_lat stub and exit 139/143 right after "Written BINARY RNG"; section 17 flags it.
V=1
for x in ${LATT//./ }; do V=$(( V * x )); done
LAT_MIN=$(( V * 4 * 9 * 16 ))     # ILDG payload: 4 links x 3x3 complex x 16 B (IEEE64BIG)
RNG_MIN=$(( 104 * (V + 1) ))      # RNG_SITMO (stock Config.h): 13 x uint64 per site
                                  # (Lattice_rng.h RngStateCount) + the serial RNG. The
                                  # hybrid chain's ckpoint_rng files are >= exactly this.
# Physics fingerprint: a resume must continue the same action and integrator.
PHYS="$LATT:$HASEN_LADDER:$MASS_LIGHT:$MASS_STRANGE:$CSW:$BETA:$U0:$STOUT_RHO/$STOUT_NSMEAR:$RAT_LO/$RAT_HI/$RAT_DEGREE:$INTEGRATOR:$MDSTEPS:$GAUGE_INNER_MULT:$TRAJL:${HASEN_STRANGE_LEVEL:-default}/${HASEN_STRANGE_INNER_MULT:-default}/$HASEN_TAIL_LEVEL:$TUNE_CG_TOL_DERIV/$TUNE_CG_TOL_ACTION/$TUNE_CG_TOL_STRANGE"

PREV= PREV_LOG= PREV_ENV= PREV_LAT= PREV_RNG= PREV_RNG_CSUM= PREV_PLAQ= RESUME_N=
prev_field() { printf '%s\n' "$PREV_ENV" | tr ' ' '\n' | sed -n "s/^$1=//p" | head -1; }

if [ -z "$RESUME_FROM" ]; then
  MODE=fresh
  [ -f "$START_CFG" ] || die "starting configuration not found: $START_CFG"
  export IMPORT_CFG=$START_CFG
  export CKPT_START_TRAJ=$START_TRAJ
  export HMC_SEED_OFFSET=${SEED_REQ:-0}
  SEED_TAG=$HMC_SEED_OFFSET
  START_TAG=$(basename "$START_CFG")
  IMPORT_TAG=$START_TAG
  CHAIN=${DATE_TAG}_${RUN}
  SEGMENT=1
else
  MODE=resume
  # (a) RESUME_FROM must be a finished run of THIS launcher (its ENV line + ckpt link).
  PREV=$(readlink -f "$RESUME_FROM") || die "RESUME_FROM does not resolve: $RESUME_FROM"
  [ -d "$PREV" ] || die "RESUME_FROM is not a directory: $RESUME_FROM"
  PREV_LOG=$PREV/hmc.log
  [ -f "$PREV_LOG" ] || die "no hmc.log in $PREV"
  PREV_ENV=$(head -1 "$PREV_LOG")
  case "$PREV_ENV" in
    "ENV TRAJ48_PUREGRID "*) ;;
    *) die "$PREV is not a traj48_puregrid.sh run (line 1 of its hmc.log is not 'ENV TRAJ48_PUREGRID ...')" ;;
  esac
  PREV_PHYS=$(prev_field PHYS)
  [ "$PREV_PHYS" = "$PHYS" ] || die "physics of $PREV differs from this launcher's; a resume must continue the same action and integrator. Theirs: '$PREV_PHYS'. Ours: '$PHYS'. Start a new chain instead."
  grep -q -a -E '^exit=' "$PREV_LOG" || warn "$PREV_LOG has no exit= line: that run may still be running (its launcher appends exit= when srun returns)"
  PREV_CKPT=$(readlink -f "$PREV/ckpt" || true)
  { [ -n "$PREV_CKPT" ] && [ -d "$PREV_CKPT" ]; } || die "checkpoint directory of $PREV is missing ($PREV/ckpt -> '${PREV_CKPT:-?}'); purged from \$PSCRATCH? see $PREV/CKPT_LOCATION.txt"
  # (b) The last complete checkpoint = the highest n with BOTH Written lines in its log;
  #     a trajectory killed during the write (timeout, allocation end) has at most one.
  COMPLETE=$(complete_ckpts "$PREV_LOG")
  if [ -z "$COMPLETE" ]; then
    HINT=
    [ "$(prev_field MODE)" = resume ] && HINT="; it resumed from $(prev_field RESUME_FROM) trajectory $(prev_field CKPT_RESUME_TRAJ), resume from that run instead"
    die "$PREV wrote no complete checkpoint (no trajectory with both 'Written BINARY RNG' and 'Written ILDG Configuration' lines)$HINT"
  fi
  if [ -n "$RESUME_TRAJ_REQ" ]; then
    printf '%s\n' "$COMPLETE" | grep -q -x -F "$RESUME_TRAJ_REQ" || die "RESUME_TRAJ=$RESUME_TRAJ_REQ is not a complete checkpoint of $PREV; complete: $(printf '%s ' $COMPLETE)"
    RESUME_N=$RESUME_TRAJ_REQ
  else
    RESUME_N=$(printf '%s\n' "$COMPLETE" | tail -1)
  fi
  # (c) Both files present and at least their payload size (a truncated or stub file is
  #     refused here rather than by a GRID_ASSERT on 16 ranks after a 6.1 GB read).
  PREV_LAT=$PREV_CKPT/ckpoint_lat.$RESUME_N
  PREV_RNG=$PREV_CKPT/ckpoint_rng.$RESUME_N
  { [ -f "$PREV_LAT" ] && [ -f "$PREV_RNG" ]; } || die "checkpoint $RESUME_N of $PREV is logged as written but a file is missing: $PREV_LAT / $PREV_RNG"
  LAT_SIZE=$(stat -L -c %s "$PREV_LAT")
  RNG_SIZE=$(stat -L -c %s "$PREV_RNG")
  [ "$LAT_SIZE" -ge "$LAT_MIN" ] || die "$PREV_LAT is $LAT_SIZE B, below the ILDG payload $LAT_MIN B: truncated"
  [ "$RNG_SIZE" -ge "$RNG_MIN" ] || die "$PREV_RNG is $RNG_SIZE B, below the RNG state size $RNG_MIN B: truncated"
  PREV_RNG_CSUM=$(grep -a -F 'Written BINARY RNG ' "$PREV_LOG" | grep -a -F "/ckpoint_rng.$RESUME_N checksum " | tail -1 | awk '{print $NF}')
  PREV_PLAQ=$(grep -a -F "Plaquette: [ $RESUME_N ] " "$PREV_LOG" | head -1 | awk '{print $NF}')
  PREV_SHA=$(prev_field BIN_SHA256)
  [ "$PREV_SHA" = "$BIN_SHA256" ] || warn "binary differs from the run being resumed (sha256 ${PREV_SHA:0:12} -> ${BIN_SHA256:0:12}); the ENV line records both"
  [ -z "$SEED_REQ" ] || [ "$SEED_REQ" = 0 ] || warn "HMC_SEED_OFFSET=$SEED_REQ is ignored on a resume: both RNGs are restored from ckpoint_rng.$RESUME_N"
  export CKPT_RESUME_TRAJ=$RESUME_N
  unset IMPORT_CFG CKPT_START_TRAJ
  export HMC_SEED_OFFSET=0
  SEED_TAG=restored
  START_TAG=ckpt:$RESUME_N
  IMPORT_TAG=none
  CHAIN=$(prev_field CHAIN)
  CHAIN=${CHAIN:-$(basename "$PREV")}
  PREV_SEG=$(prev_field SEGMENT)
  [[ "$PREV_SEG" =~ ^[0-9]+$ ]] || PREV_SEG=1
  SEGMENT=$(( PREV_SEG + 1 ))
fi

# ---- 12. Run directory + checkpoint directory (refuse to overwrite) ----------------------
# Checkpoints go to $PSCRATCH by default, linked from the run dir as <rundir>/ckpt:
#   - 6.1 GB gauge + 1.1 GB RNG = 7.2 GB per trajectory (ckpoint_lat 6,115,299,576 B and
#     ckpoint_rng 1,104,150,632 B in runs/2026_8_18_lvlroute_w3_tailmiddle_48/ckpt/);
#     144 GB for a 20-trajectory chain. The m4599 CFS quota is shared (67% of 20 TB and
#     62% of the 10 M inode limit on 2026-09-30; inodes hit 99% on 2026-08-17,
#     machines/perlmutter.md "Filesystems").
#   - The NERSC guidance for this workspace puts active job checkpoints on $SCRATCH;
#     Lustre also takes the 16-rank MPI-IO write.
#   - Run output (hmc.log, the ENV line, the link) stays under runs/ on CFS as usual.
#   $PSCRATCH is NOT backed up and purges files idle ~8 weeks: a configuration that must
#   be kept is copied to CFS or HPSS by hand. CKPT_LOCATION.txt in the run dir says so.
#   CHECKPOINT_ROOT=rundir keeps them on CFS in <rundir>/ckpt (the hybrid chain's layout).
# Each run gets its OWN checkpoint directory, also on a resume. The driver reads the
# resume point from its CKPT_DIR (one prefix for read and write), so the resumed run's
# directory gets two symlinks, ckpoint_{lat,rng}.<n> -> the previous run's files (Grid
# opens them by path: check_filename is an ifstream, the reads are MPI-IO/fopen). Unlike
# the hybrid chain's "SAME dir, append" resume, nothing is ever written into another run's
# directory, a replay (RESUME_TRAJ) cannot overwrite the trajectories it replays, and a
# failed resume is re-run without cleaning anything up.
RUN_DIR=$BASE/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
if [ "$CHECKPOINT_ROOT" = rundir ]; then
  CKPT_PHYS=$RUN_DIR/ckpt
else
  CKPT_PHYS=$CHECKPOINT_ROOT/${DATE_TAG}_${RUN}
  [ -e "$CKPT_PHYS" ] && die "checkpoint directory already exists, refusing to overwrite: $CKPT_PHYS"
fi
case "$CKPT_PHYS" in
  /pscratch/*)   CKPT_FS=pscratch ;;
  /global/cfs/*) CKPT_FS=cfs ;;
  *)             CKPT_FS=other ;;
esac
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
LOG=$RUN_DIR/hmc.log
if [ "$CHECKPOINT_ROOT" = rundir ]; then
  mkdir "$CKPT_PHYS" || die "cannot create $CKPT_PHYS"
else
  mkdir -p "$CHECKPOINT_ROOT" || die "cannot create $CHECKPOINT_ROOT"
  mkdir "$CKPT_PHYS" || die "cannot create $CKPT_PHYS"
  ln -s "$CKPT_PHYS" "$RUN_DIR/ckpt" || die "cannot link $RUN_DIR/ckpt -> $CKPT_PHYS"
fi
printf 'Run directory: %s\n' "$RUN_DIR" > "$CKPT_PHYS/RUN_DIR.txt"
if [ "$MODE" = resume ]; then
  RESUME_LAT=$(readlink -f "$PREV_LAT")
  RESUME_RNG=$(readlink -f "$PREV_RNG")
  ln -s "$RESUME_LAT" "$CKPT_PHYS/ckpoint_lat.$RESUME_N" || die "cannot link the resume gauge file into $CKPT_PHYS"
  ln -s "$RESUME_RNG" "$CKPT_PHYS/ckpoint_rng.$RESUME_N" || die "cannot link the resume RNG file into $CKPT_PHYS"
else
  RESUME_LAT=none
fi
export CKPT_DIR=$CKPT_PHYS
export CKPT_INTERVAL=1
{
  printf 'Checkpoints of this run: %s (filesystem: %s)\n' "$CKPT_PHYS" "$CKPT_FS"
  printf 'Driver CKPT_DIR=%s CKPT_INTERVAL=1: after EVERY trajectory (accept or reject) ckpoint_lat.<n> (ILDG IEEE64BIG, 6.1 GB) and ckpoint_rng.<n> (serial + parallel RNG, 1.1 GB).\n' "$CKPT_PHYS"
  printf 'A checkpoint is complete when hmc.log has both its "Written BINARY RNG" and "Written ILDG Configuration on" lines.\n'
  if [ "$MODE" = resume ]; then
    printf 'Resumed from %s trajectory %s: ckpoint_lat.%s -> %s, ckpoint_rng.%s -> %s (symlinks).\n' "$PREV" "$RESUME_N" "$RESUME_N" "$RESUME_LAT" "$RESUME_N" "$RESUME_RNG"
  fi
  if [ "$CKPT_FS" = pscratch ]; then
    printf '$PSCRATCH is not backed up and purges files idle ~8 weeks: copy any configuration that must be kept to CFS or HPSS.\n'
  fi
} > "$RUN_DIR/CKPT_LOCATION.txt"

# ---- 13. select_gpu (+ gpu_mon) wrappers + srun step shape ------------------------------
# 1:1 local rank -> GPU + NUMA domain (paboyle model), verbatim from
# run_probe_grid_mg.sh. It needs --gpu-bind=none: --gpu-bind=closest breaks CUDA
# IPC and hangs (_SUM.md halo-nic; port plan s.7). --cpus-per-task=32 with
# --cpu-bind=cores puts local rank i on NUMA domain i (4 tasks own each node).
SELECT_GPU=$RUN_DIR/select_gpu
printf '%s\n' \
  '#!/bin/bash' \
  '# 1:1 local rank -> GPU + NUMA binding; copied from run_probe_grid_mg.sh' \
  'export GPU=$SLURM_LOCALID' \
  'export NUMA=$SLURM_LOCALID' \
  'export CUDA_VISIBLE_DEVICES=$GPU' \
  'exec numactl -m $NUMA -N $NUMA "$@"' > "$SELECT_GPU"
chmod +x "$SELECT_GPU"

LAUNCH=("$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}")
if [ "$GPU_MON_MS" -gt 0 ]; then
  # Runs inside the main step (no second srun): nvidia-smi ignores CUDA_VISIBLE_DEVICES
  # and sees the step's four GPUs per node. Backgrounded with its own timeout as a
  # backstop; slurmstepd kills it when the step ends. Then exec's select_gpu unchanged.
  GPU_MON=$RUN_DIR/gpu_mon
  {
    printf '%s\n' '#!/bin/bash' \
      '# nvidia-smi memory sampler on local rank 0 of each node; written by traj48_puregrid.sh' \
      'if [ "${SLURM_LOCALID:-}" = 0 ]; then'
    printf '  timeout -k 10s %sm nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.total --format=csv,noheader,nounits -lms %s > "%s/gpu_mem.${SLURMD_NODENAME:-$(hostname)}.csv" 2>&1 < /dev/null &\n' "$(( TIMEOUT_MIN + 5 ))" "$GPU_MON_MS" "$RUN_DIR"
    printf '%s\n' 'fi' 'exec "$@"'
  } > "$GPU_MON"
  chmod +x "$GPU_MON"
  LAUNCH=("$GPU_MON" "${LAUNCH[@]}")
fi

# --export=ALL passes this whole environment to every rank (an --export without
# ALL strips it: the ~10 s exit-1 "shell variable" failure, machines/perlmutter.md).
SRUN=(srun --jobid="$SLURM_JOB_ID" -N "$NODES" -n "$NTASKS" --ntasks-per-node=4
      --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=cores --gpu-bind=none
      --export=ALL --chdir="$RUN_DIR" ${SRUN_EXTRA:+$SRUN_EXTRA})

# ---- 14. Gated NUMA binding -----------------------------------------------------------
# HARD GATE, not a print (machines/perlmutter.md "Node binding"; L104). Runs exactly
# what select_gpu will run on every one of the 16 ranks (numactl -m i -N i).
if ! "${SRUN[@]}" "$SELECT_GPU" true; then
  die "NUMA gate failed: a local rank cannot bind its own NUMA domain; select_gpu would fail. Nothing was run (rundir $RUN_DIR)."
fi
"${SRUN[@]}" --label "$SELECT_GPU" numactl --show 2>&1 | grep -E 'nodebind' > "$RUN_DIR/numa_binding.txt"

# ---- 15. Provenance: ENV line = line 1 of the log ---------------------------------------
# Grid embeds its configure-time GITHASH ("<sha>: (<refs>) [uncommited changes]").
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
GRID_SHA=${GRID_HASH_LINE:0:40}
GRID_SHA=${GRID_SHA:-unknown}
case "$GRID_HASH_LINE" in *uncommit*) GRID_DIRTY=yes ;; *) GRID_DIRTY=no ;; esac
# Fields parsed by a later RESUME (MODE CHAIN SEGMENT PHYS BIN_SHA256 RESUME_FROM
# CKPT_RESUME_TRAJ) carry no spaces; quoted fields come last.
ENV_KV=(
  "RUN=$RUN" "JOBID=$SLURM_JOB_ID" "MODE=$MODE" "CHAIN=$CHAIN" "SEGMENT=$SEGMENT"
  "START=$START_TAG" "N_TRAJ=$N_TRAJ" "SEED=$SEED_TAG" "METROP=on"
  "BIN=$(basename "$BIN")" "BIN_SHA256=$BIN_SHA256" "GRID_SHA=$GRID_SHA" "GRID_DIRTY=$GRID_DIRTY"
  "IMPORT_CFG=$IMPORT_TAG" "CKPT_START_TRAJ=${CKPT_START_TRAJ:-none}"
  "CKPT_RESUME_TRAJ=${CKPT_RESUME_TRAJ:-none}" "RESUME_FROM=${PREV:-none}" "RESUME_LAT=$RESUME_LAT"
  "RESUME_RNG_CSUM=${PREV_RNG_CSUM:-none}" "RESUME_PLAQ=${PREV_PLAQ:-none}"
  "CKPT_DIR=$CKPT_DIR" "CKPT_FS=$CKPT_FS" "CKPT_INTERVAL=$CKPT_INTERVAL" "SMEARED_CKPT=no"
  "LATT=$LATT" "MPI=$MPI_GEOM" "NODES=$NODES" "NTASKS=$NTASKS"
  "LADDER=$HASEN_LADDER" "STRANGE_LEVEL=${HASEN_STRANGE_LEVEL:-default}" "STRANGE_INNER_MULT=${HASEN_STRANGE_INNER_MULT:-default}"
  "TAIL_LEVEL=$HASEN_TAIL_LEVEL" "INTEGRATOR=$INTEGRATOR" "MDSTEPS=$MDSTEPS" "GAUGE_INNER_MULT=$GAUGE_INNER_MULT"
  "TRAJL=$TRAJL" "MASS_LIGHT=$MASS_LIGHT" "MASS_STRANGE=$MASS_STRANGE" "CSW=$CSW" "BETA=$BETA" "U0=$U0"
  "STOUT_RHO=$STOUT_RHO" "STOUT_NSMEAR=$STOUT_NSMEAR" "RAT=$RAT_LO/$RAT_HI/$RAT_DEGREE"
  "TOL_DRV=$TUNE_CG_TOL_DERIV" "TOL_ACT=$TUNE_CG_TOL_ACTION" "TOL_STRANGE=$TUNE_CG_TOL_STRANGE"
  "GRID_MG_RUNGS=${HASEN_GRID_MG_RUNGS:-none}" "GRID_MG_HEATBATH_RUNGS=${HASEN_GRID_MG_HEATBATH_RUNGS:-none}"
  "GRID_MIXED_CG_RUNGS=${HASEN_GRID_MIXED_CG_RUNGS:-none}" "GRID_MIXED_CG_HEATBATH_RUNGS=${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}"
  "QUDA_ENV=$N_QUDA_ENV" "MPICH_IPC=$MPICH_GPU_IPC_ENABLED" "MPICH_RDMA=$MPICH_RDMA_ENABLED_CUDA"
  "MPICH_NIC=$MPICH_OFI_NIC_POLICY" "OMP=$OMP_NUM_THREADS" "DEVICE_MEM_MB=$DEVICE_MEM_MB"
  "VERBOSE_MEM=$VERBOSE_MEM" "GPU_MON_MS=$GPU_MON_MS" "TIMEOUT_MIN=$TIMEOUT_MIN" "ALLOC_LEFT_MIN=${ALLOC_LEFT_MIN:-unknown}"
  "PHYS=$PHYS"
  "HASEN_GRID_EXTRA=\"$HASEN_GRID_EXTRA\"" "GRID_MG_ENV=\"$GRID_MG_ENV\"" "GRID_FLAGS=\"${GRID_ARGS[*]}\""
)
printf 'ENV TRAJ48_PUREGRID %s\n' "${ENV_KV[*]}" > "$LOG"
{
  printf 'GRID_HASH %s\n' "${GRID_HASH_LINE:-unknown}"
  printf 'BIN_PATH %s\n' "$BIN"
  sed 's/^/NUMA /' "$RUN_DIR/numa_binding.txt"
  printf 'CMD '
  printf '%q ' timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "${LAUNCH[@]}"
  printf '\n'
} >> "$LOG"

# ---- 16. Launch -----------------------------------------------------------------------------
# timeout wrapper: every run keeps one (_SUM.md perlmutter-ops); an aborted step can
# also leave srun hung (port plan s.7 "Hung srun after an abort"). exit=124 = timeout.
echo "running $(basename "$BIN") $MODE start=$START_TAG N_TRAJ=$N_TRAJ ckpt=$CKPT_PHYS -> $LOG"
timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "${LAUNCH[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 17. Summary ----------------------------------------------------------------------------
echo "==== $RUN: exit=$rc  log: $LOG"
[ "$rc" -eq 124 ] && echo "exit 124 = TIMEOUT_MIN ($TIMEOUT_MIN min) reached: the trajectory in flight is lost, completed ones are checkpointed"
grep -a -E '^ENV ' "$LOG"
grep -a -E 'Current Grid git commit hash|CG tol:|Hasenbusch chain|\[Ladder\]|N_TRAJ=|INTEGRATOR=|NoMetropolisUntil=|\[HMC parameters\] (Trajectories|Start trajectory|Metropolis test)|HMC_SEED_OFFSET=|IMPORT_CFG=|\[Ckpt\]|Initial plaquette' "$LOG" | grep -a -v -E '^(ENV|CMD) '
if [ "$MODE" = resume ]; then
  echo "---- checkpoint read-back (reader asserts SciDAC checksum + plaquette/link trace) ----"
  grep -a -E 'RNG read I/O on file|RNG file (nersc_checksum|scidac_checksum)|FieldNormMetaData|SciDAC checksums|Plaquette and link trace match|Read ILDG Configuration from' "$LOG"
  THIS_READ_CSUM=$(grep -a -F 'Read ILDG Configuration from ' "$LOG" | head -1 | awk '{print $NF}')
  THIS_INIT_PLAQ=$(grep -a -F 'Initial plaquette = ' "$LOG" | head -1 | awk '{print $NF}')
  echo "---- continuity with $PREV trajectory $RESUME_N ----"
  if [ -n "$THIS_READ_CSUM" ] && [ "$THIS_READ_CSUM" = "$PREV_RNG_CSUM" ]; then CS=MATCH; else CS=MISMATCH; fi
  printf 'RNG checksum: written %s, read %s -> %s\n' "${PREV_RNG_CSUM:-n/a}" "${THIS_READ_CSUM:-n/a}" "$CS"
  awk -v a="${PREV_PLAQ:-}" -v b="${THIS_INIT_PLAQ:-}" 'BEGIN {
        if (a == "" || b == "") { printf "plaquette: previous %s, initial %s -> n/a\n", a, b; exit }
        d = a - b; if (d < 0) d = -d
        printf "plaquette: previous Plaquette:[ n ] %s, this run Initial plaquette %s, |diff| %.2e -> %s\n", a, b, d, (d < 1e-9) ? "MATCH" : "MISMATCH" }'
fi
echo "---- per trajectory (wall = MD only, PRIMARY; cycle = Grid Total time; ckpt = RNG+gauge write) ----"
traj_table "$LOG" 0 999999999
if [ "$MODE" = resume ]; then
  # Replay (RESUME_TRAJ below the previous run's last trajectory): same numbers in both
  # logs. Expect equal RNG checksums and agreement to solver tolerance, not bits.
  OVL_HI=$(( RESUME_N + N_TRAJ ))
  if [ -n "$(complete_ckpts "$PREV_LOG" | awk -v lo="$RESUME_N" '$1 > lo')" ]; then
    echo "---- replay: the previous run's rows for trajectories $(( RESUME_N + 1 ))..$OVL_HI ----"
    traj_table "$PREV_LOG" $(( RESUME_N + 1 )) "$OVL_HI"
    echo "---- replay: written RNG checksums, previous then this run ----"
    for (( n = RESUME_N + 1; n <= OVL_HI; n++ )); do
      A=$(grep -a -F 'Written BINARY RNG ' "$PREV_LOG" | grep -a -F "/ckpoint_rng.$n checksum " | tail -1 | awk '{print $NF}')
      B=$(grep -a -F 'Written BINARY RNG ' "$LOG" | grep -a -F "/ckpoint_rng.$n checksum " | tail -1 | awk '{print $NF}')
      [ -n "$A" ] || [ -n "$B" ] || continue
      if [ -n "$A" ] && [ "$A" = "$B" ]; then S=MATCH; else S=DIFFER; fi
      printf 'traj %s: %s %s -> %s\n' "$n" "${A:-none}" "${B:-none}" "$S"
    done
  fi
fi
echo "---- FORCES traj= (per-monomial force averages over each trajectory) ----"
grep -a -E 'FORCES traj=' "$LOG"
echo "---- Polyakov loop ----"
grep -a -E 'Polyakov Loop: \[' "$LOG"
echo "---- [GridMG lines: every build / rebuild / hard-tier line (per-solve lines tallied below) ----"
grep -a -F '[GridMG' "$LOG" | grep -a -v -E '\] solve [0-9]/2 iters ' | head -200
grep -a -F '[GridMG' "$LOG" | awk '/ solve [0-9]\/2 iters / {
    t = $0; sub(/.*\[GridMG /, "", t); sub(/\].*/, "", t); it = 0; rr = 0; s = 0
    for (i = 1; i < NF; i++) { if ($i == "iters") it = $(i + 1) + 0; if ($i == "true_rel_residual") rr = $(i + 1) + 0; if ($i == "seconds") s = $(i + 1) + 0 }
    ns[t]++; ts[t] += s; if (it > mi[t]) mi[t] = it; if (rr > mr[t]) mr[t] = rr }
  END { for (t in ns) printf "[GridMG %s] solves=%d max_iters=%d max_true_rel_residual=%.3g total_solve_s=%.1f\n", t, ns[t], mi[t], mr[t], ts[t] }' | sort
echo "---- checkpoint lines ----"
grep -a -E 'Written BINARY RNG|Written ILDG Configuration on' "$LOG"
echo "---- checkpoints written by this run (complete = both lines + at least payload size) ----"
N_CKPT=0
while read -r n; do
  [ -n "$n" ] || continue
  N_CKPT=$(( N_CKPT + 1 ))
  LS=$(stat -L -c %s "$CKPT_PHYS/ckpoint_lat.$n" 2>/dev/null || echo 0)
  RS=$(stat -L -c %s "$CKPT_PHYS/ckpoint_rng.$n" 2>/dev/null || echo 0)
  if [ "$LS" -ge "$LAT_MIN" ] && [ "$RS" -ge "$RNG_MIN" ]; then ST=complete; else ST=TRUNCATED; fi
  printf 'traj %s: ckpoint_lat %s B, ckpoint_rng %s B -> %s\n' "$n" "$LS" "$RS" "$ST"
done < <(complete_ckpts "$LOG")
[ "$N_CKPT" -gt 0 ] || echo "no complete checkpoint in this run"
ls -l "$CKPT_PHYS"
if [ "$GPU_MON_MS" -gt 0 ]; then
  echo "---- nvidia-smi high-water, MiB (file, GPU index, peak memory.used, memory.total, samples) ----"
  if compgen -G "$RUN_DIR/gpu_mem.*.csv" > /dev/null; then
    awk -F', *' '$4 ~ /^[0-9]+$/ { f = FILENAME; sub(/.*\//, "", f); k = f " gpu" $2
                  if (!(k in m) || $4 + 0 > m[k]) m[k] = $4 + 0; t[k] = $5; n[k]++ }
                 END { for (k in m) printf "%s peak_used=%d total=%s samples=%d\n", k, m[k], t[k], n[k] }' \
      "$RUN_DIR"/gpu_mem.*.csv | sort | tee "$RUN_DIR/gpu_mem_peak.txt"
  else
    echo "no gpu_mem.*.csv written (sampler did not start)"
  fi
fi
echo "---- failures (assert / OOM / abort) ----"
grep -a -i -E 'assert|abort|out of memory|cudaMalloc failed|EvictVictims|oom[-_ ]?kill|segmentation fault|bus error|floating point exception|FATAL|Grid : Error|srun: error|not found\. Aborting' "$LOG" | head -60
echo "---- next segment (one line) ----"
echo "SLURM_JOB_ID=<id> BIN=$BIN RUN=<next name> RESUME_FROM=$RUN_DIR bash $SELF"
echo "log: $LOG"
exit "$rc"
