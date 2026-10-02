#!/bin/bash
# =============================================================================
# forces48_puregrid.sh -- ONE FORCES_ONLY (force-evaluation-only) run of the
# pure-Grid compact-Schur driver (src/gen_qcd_hasenbusch_tune_compact_schur.cc)
# at 48^3x96 on 4 GPU nodes (16 ranks, 1 GPU each), inside an EXISTING
# allocation. This script never allocates and never submits.
#
# Purpose: the "48^3 memory probe" of __docs/2026_09_29_pure_grid_stock_build_plan.md
# section 6 (last M2 item, 2026_09_29_pure_grid_m2_mg_solver_design.md s.4 item 4):
# does the fp32 Grid multigrid fit beside the HMC's operators on 4 nodes, and what
# does one force evaluation cost per monomial. Production C3 ladder and production
# tolerances; Grid-MG on rungs 0,1,2 by default, CG-only control on request.
#
# Usage (one line each):
#   SLURM_JOB_ID=<id> BIN=<binary> RUN=<name> bash forces48_puregrid.sh
# e.g.
#   SLURM_JOB_ID=<id> BIN=<hb>/bin/gen_qcd_hasenbusch_tune_compact_schur_stock RUN=forces48_mg012 bash forces48_puregrid.sh
#   SLURM_JOB_ID=<id> BIN=<hb>/bin/gen_qcd_hasenbusch_tune_compact_schur_stock RUN=forces48_cgonly HASEN_GRID_MG_RUNGS= bash forces48_puregrid.sh
# The allocation (4 nodes) comes from `bash ~/.claude/bin/alloc_gpu.sh 4 <hours>` and
# is released with `release_alloc.sh <id>` (machines/perlmutter.md). That wrapper asks
# for `-C gpu`, not `gpu&hbm80g`: which card size you got (40 or 80 GB) is recorded
# below (Grid's totalGlobalMem line, nvidia-smi memory.total). Never merge 40 and
# 80 GB rows (L119).
#
# Caller overrides (everything else is hard-coded on purpose):
#   HASEN_GRID_MG_RUNGS  default "0,1,2"; set EMPTY (HASEN_GRID_MG_RUNGS=) for the
#                        CG-only control. GRID_MG_* pass through untouched.
#   HASEN_GRID_MG_HEATBATH_RUNGS, HASEN_GRID_MIXED_CG_RUNGS,
#   HASEN_GRID_MIXED_CG_HEATBATH_RUNGS  M3 routes, default unset; see section 8.
#   DEVICE_MEM_MB        default 8000 (--device-mem, Grid's device lattice cache in MB)
#   FORCES_SAMPLES       default 2 (see section 7)
#   INTEGRATOR_VERBOSE_MEM  default 1; "" or 0 = off (see section 10: INERT here)
#   GPU_MON_MS           default 2000 (nvidia-smi sampling period); 0 = no sampler
#   SRUN_EXTRA           appended verbatim to srun (e.g. --overlap)
#
# Output: runs/<YYYY_M_D>_<RUN>/hmc.log (unpadded date, machines/perlmutter.md
# "Filesystems"). Line 1 is the ENV provenance line; per the conventions entry of
# __docs/_SUM.md, that line, not this script, is the ground truth for what ran.
# Also gpu_mem.<node>.csv (one per node) and gpu_mem_peak.txt when GPU_MON_MS > 0.
#
# Wall time: no pure-Grid 48^3 FORCES_ONLY timing exists yet. Per sample: nine
# monomials; every heatbath is fp64 Grid CG at 1e-12 (pure Grid has no MG heatbath),
# non-MG derivs are fp64 CG at 1e-11, the MG donor's first deriv carries the lazy
# hierarchy build (~84 s per build at 48^3, build plan s.6), plus a 6.1 GB config read.
# Timeout 90 min.
# =============================================================================
set -o pipefail

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
WORKFLOW=$BASE/grid-lqcd-workflow

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
USAGE="usage: SLURM_JOB_ID=<id> BIN=<binary> RUN=<name> bash $0"

# ---- 1. Inputs ---------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || die "SLURM_JOB_ID is not set (the id of an existing 4-node allocation). $USAGE"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be a numeric job id, got '$SLURM_JOB_ID'"
[ -n "${BIN:-}" ] || die "BIN is not set (path to the driver binary). $USAGE"
[ -n "${RUN:-}" ] || die "RUN is not set (run name, becomes runs/<date>_<RUN>). $USAGE"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN may contain only letters, digits, '.', '_' and '-', got '$RUN'"
{ [ -f "$BIN" ] && [ -x "$BIN" ]; } || die "BIN is not an executable file: $BIN"
BIN=$(readlink -f "$BIN")

# The allocation must already exist and span 4 nodes; this script only attaches
# srun steps to it. (squeue is read-only.)
JOB_STATE=$(squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null | head -1)
[ "$JOB_STATE" = RUNNING ] || die "allocation $SLURM_JOB_ID is not RUNNING (state: '${JOB_STATE:-not found}'). This script never allocates."
JOB_NODES=$(squeue -h -j "$SLURM_JOB_ID" -o %D 2>/dev/null | head -1)
{ [[ "$JOB_NODES" =~ ^[0-9]+$ ]] && [ "$JOB_NODES" -ge 4 ]; } || die "allocation $SLURM_JOB_ID has '${JOB_NODES:-?}' node(s); this run needs 4"

# Caller overrides, read BEFORE the scrub in section 3 (which clears the FORCES_*
# and INTEGRATOR_VERBOSE_MEM families). "-" not ":-" where empty means something.
if [ -z "${HASEN_GRID_MG_RUNGS+x}" ]; then HASEN_GRID_MG_RUNGS=0,1,2; fi
FO_SAMPLES=${FORCES_SAMPLES:-2}
VERBOSE_MEM_REQ=${INTEGRATOR_VERBOSE_MEM-1}
DEVICE_MEM_MB=${DEVICE_MEM_MB:-8000}
GPU_MON_MS=${GPU_MON_MS:-2000}
[[ "$FO_SAMPLES" =~ ^[1-9][0-9]*$ ]] || die "FORCES_SAMPLES must be a positive integer, got '$FO_SAMPLES'"
[[ "$DEVICE_MEM_MB" =~ ^[1-9][0-9]*$ ]] || die "DEVICE_MEM_MB must be a positive integer (MB), got '$DEVICE_MEM_MB'"
[[ "$GPU_MON_MS" =~ ^[0-9]+$ ]] || die "GPU_MON_MS must be a non-negative integer (ms), got '$GPU_MON_MS'"

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
[ "$N_QUDA" -eq 0 ] || die "ldd shows libquda in $BIN; this probe is for pure-Grid builds only"

# ---- 3. Clear every knob this probe does not set ------------------------------
# Same families as smoke16_puregrid.sh, same HASEN_GRID_/GRID_MG_ exemption. A
# stray variable from the caller's shell would silently change the run, or kill it:
# every QUDA_* / HASEN_QUDA_* / HASEN_MG_* / HMC_MG_* knob of chroma_match_sop_48.sh
# and forces_only_sop_48.sh goes here, and in a pure-Grid build HASEN_MG_RUNG /
# HASEN_QUDA_CG_RUNGS / HASEN_QUDA_CG_TAIL / HASEN_QUDA_FORCE_LOGDET exit(1) with
# "requires a QUDA build", while any of HASEN_MG_RUNG / _SHARED_RUNGS / QUDA_CG_RUNGS /
# _HEATBATH_RUNGS beside HASEN_GRID_MG_RUNGS exits "a rung has ONE solver route"
# (driver source). Families cleared:
#   QUDA_*, HASEN_*, HMC_MG_*, USE_HMC_MG  QUDA / MG routing (driver + fork headers)
#   FORCES_*                               FORCES_ONLY mode: reset, then set in s.7
#   CKPT_*                                 checkpointer (off: no config written)
#   TXQCD_*, WCF_*                         fork-library env gates (all default off; the
#                                          LogDet GPU paths default ON under CUDA, build
#                                          plan s.6, as in every production run)
#   LAMBDA_MN2                             fork-only MN2 knob
#   INTEGRATOR_VERBOSE_MEM                 MemoryManager::Print gate (patch 03): s.10
#   STRANGE_EVEN, NO_METROP, IMPORT_CFG    reset, then set below as needed
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|STRANGE_EVEN$|NO_METROP$|IMPORT_CFG$)' | grep -v -E '^(HASEN_GRID_|GRID_MG_)')
# Kept on purpose (not scrubbed): every HASEN_GRID_* switch of the pure-Grid rung solvers
# and GRID_MG_*, the multigrid tunables. HASEN_GRID_MG_RUNGS (M2,
# 2026_09_29_pure_grid_m2_mg_solver_design.md) and, since M3, HASEN_GRID_MG_HEATBATH_RUNGS,
# HASEN_GRID_MIXED_CG_RUNGS and HASEN_GRID_MIXED_CG_HEATBATH_RUNGS select the routes; the
# driver parses each strictly and exits on a bad or conflicting list.
# GridMGParams::from_env rejects an unknown GRID_MG_* name ("typo?"), so a stray one
# fails loudly rather than silently. The ENV line below records them.
N_QUDA_ENV=$(compgen -e | grep -c '^QUDA_' || true)
[ "$N_QUDA_ENV" -eq 0 ] || die "QUDA_* variables still set after clearing"
unset N_TRAJ   # inert in FORCES_ONLY; cleared so a caller value cannot reach the log banner

# ---- 4. Physics (cl21 48^3x96 ensemble, Chroma cfg_2000 action) ---------------
# All from submit_scripts/hmc48_compact/chroma_match_sop_48.sh section A (confirmed
# 2026-07-13; identical in forces_only_sop_48.sh). Every value is exported even where
# a default exists, so no params.h / driver default can differ from production.
export LATT=48.48.48.96
export IMPORT_CFG=$BASE/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime   # ILDG, 6.1 GB
[ -f "$IMPORT_CFG" ] || die "starting configuration not found: $IMPORT_CFG"
export BETA=6.3
# U0: the driver's LOCAL default is 1.0 and shadows params.h; the wrong U0 was the
# 48^3 blow-up root cause (__docs/_TOC.md u0-48cube). SOP value:
export U0=0.84570646270714
export CSW=1.20536588031793
export MASS_LIGHT=-0.2416                                 # must equal HASEN_LADDER[0]
export MASS_STRANGE=-0.2050
# STOUT: params.h defaults (0.125 / 1). The SOP does not set them, so every
# production 48^3 run inherited exactly these; exported to pin them.
export STOUT_RHO=0.125 STOUT_NSMEAR=1
export RAT_LO=0.4 RAT_HI=35.0 RAT_DEGREE=20               # strange RHMC bounds (48^3 values)

# ---- 5. C3 ladder + level layout ------------------------------------------------
# C3 = the w3 ladder with the tail on strange's level
# (submit_scripts/hasenbusch_tune/2026_8_21_lvlroute_w3_tailmiddle_extend_2011_2020.sh,
# on top of the SOP): 4 ratio rungs PF0..PF3 + bare-det tail at -0.1870.
# The integrator / level knobs below are INERT in FORCES_ONLY: the driver returns
# before the Integrator is constructed (driver FORCES_ONLY block). They are exported
# so the driver still validates the layout, prints it in the [Ladder] banner, and so
# this env block equals the HMC's. TRAJL is not required in FORCES_ONLY (driver exempts
# it); set for the same reason.
# LADDER_PROFILE=baseG (default since 2026-10-01) or c3. baseG is the SOP's baseline of the Grid-vs-Chroma
# comparison, 2026_08_24_grid_vs_chroma_comparison_summary.md: ladder
# -0.2416,-0.2400,-0.2320,-0.2180,-0.1870, tail inner, MDSTEPS=12; SOP lines 23-30).
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
  *) die "LADDER_PROFILE must be c3 or baseG, got $LADDER_PROFILE" ;;
esac
export INTEGRATOR=ForceGradient                           # SOP section B
export GAUGE_INNER_MULT=2                                 # SOP section B
export TRAJL=0.35355339059327379                          # SOP section B

# ---- 6. Seed + tolerances -------------------------------------------------------
# Seed: 0 == unset == the historical fixed seeds (driver RNG block: "Unset/0").
# forces_only_sop_48.sh unsets HMC_SEED_OFFSET for paired draws across candidates;
# here it makes the MG and CG-only arms draw identical heatbath noise, so their
# per-monomial forces must agree to solver tolerance.
export HMC_SEED_OFFSET=0
# PRODUCTION tolerances (chroma_match_sop_48.sh section C; __docs/_SUM.md conventions),
# NOT the forces-only SOP's screening pair (its DERIV 1e-8 is a ladder-screening
# speedup only). This probe measures what a production force evaluation costs and
# holds in memory, and the MG must reach the production 1e-11.
export TUNE_CG_TOL_DERIV=1e-11
export TUNE_CG_TOL_ACTION=1e-12
export TUNE_CG_TOL_STRANGE=1e-9

# ---- 7. FORCES_ONLY mode ----------------------------------------------------------
# Driver semantics (FORCES_ONLY block): smear cfg_2000 once, then for each of
# FORCES_SAMPLES samples, for each monomial in the order LightLogDet, PF0..PF3,
# LightSchurPF, Strange, StrangeLogDet, Gauge: heatbath (refresh) then ONE force
# evaluation (deriv), streamed as "FORCES_ONLY level <name> sample <s> avg= max=
# refresh=<heatbath s> time=<force s>", then "FORCES_ONLY samples=<n> avg:/max:/
# refresh:/time:" averages. No integrator, no Metropolis, no config written.
# Samples: 2 by default. The Grid-MG hierarchy is built lazily at the donor's first
# solve (driver HASEN_GRID_MG_RUNGS block), i.e. inside PF0's sample-0 deriv, so
# sample 0's PF0 time includes the build and sample 1 is the steady-state cost. The
# hierarchy and every operator stay alive from sample 0 on, so the memory high-water
# is set in sample 0. The driver default is 1, the forces-only SOP uses 8.
# Change with FORCES_SAMPLES=<n> in the caller's environment.
# Nothing skipped: the SOP's FORCES_SKIP_STRANGE / _GAUGE are ladder-screening
# shortcuts; here total memory and per-monomial cost are the question, and the
# strange multishift (RAT_DEGREE=20 shifted solutions) runs while the hierarchy is alive.
export FORCES_ONLY=1
export FORCES_SAMPLES=$FO_SAMPLES

# ---- 8. Grid multigrid route ------------------------------------------------------
# HASEN_GRID_MG_RUNGS=0,1,2: rung 0 (-0.2416) donates the one fp32 three-level
# hierarchy, rungs 1 (-0.2380) and 2 (-0.2340) share it (GCR at their own mass);
# rung 3 (-0.2180) and the tail stay on CG, as the QUDA hybrid routes C3
# (SOP section D: MG 0 + shared 1,2, CG 3 + tail). MG covers deriv and S only; every
# heatbath stays CG_action unless the caller sets the M3 lists (pass through untouched,
# recorded on the ENV line): HASEN_GRID_MG_HEATBATH_RUNGS (heatbath on the hierarchy; rungs
# must be MG rungs), HASEN_GRID_MIXED_CG_RUNGS (non-MG rungs' deriv/S in mixed-precision CG),
# HASEN_GRID_MIXED_CG_HEATBATH_RUNGS (the other heatbaths in mixed-precision CG).
# Empty = CG-only control (driver treats empty as unset).
if [ -n "$HASEN_GRID_MG_RUNGS" ]; then
  [[ "$HASEN_GRID_MG_RUNGS" =~ ^[0-3](,[0-3])*$ ]] || die "HASEN_GRID_MG_RUNGS must list ratio rungs 0..3 of the C3 ladder, got '$HASEN_GRID_MG_RUNGS'"
  # A binary built before M2 ignores the switch and would silently run CG
  # (the 15:03 fork reference binary has no such string).
  N_MG_STR=$(grep -c -a -F 'HASEN_GRID_MG_RUNGS' "$BIN" || true)
  [ "$N_MG_STR" -gt 0 ] || die "$BIN has no HASEN_GRID_MG_RUNGS support (pre-M2 build): it would silently run CG. Use the M2 stock binary or set HASEN_GRID_MG_RUNGS= for the CG-only control."
  export HASEN_GRID_MG_RUNGS
  MG_RUNGS_TAG=$HASEN_GRID_MG_RUNGS
else
  unset HASEN_GRID_MG_RUNGS
  MG_RUNGS_TAG=none
fi
# The M3 lists: validated by the driver (strict parse, subset/exclusion rules). A pre-M3
# binary ignores them and would silently run the M2 routes, so refuse that here.
for v in HASEN_GRID_MG_HEATBATH_RUNGS HASEN_GRID_MIXED_CG_RUNGS HASEN_GRID_MIXED_CG_HEATBATH_RUNGS; do
  if [ -n "${!v:-}" ]; then
    [ "$(grep -c -a -F "$v" "$BIN" || true)" -gt 0 ] || die "$BIN has no $v support (pre-M3 build): it would silently ignore $v=${!v}."
    export "$v"
  else
    unset "$v"
  fi
done
# ---- production defaults (2026-10-02): every validated speed-up ON unless the caller says
# otherwise; override with VAR=0. A binary that predates a switch runs without it, with a
# warning. (HASEN_GRID_BATCH_SMEAR is inert in FORCES_ONLY, which never enters update_P.)
# Force-cost analysis s.7.5; one-page summary 2026_10_02_pure_grid_speedup_summary.md.
for v in HASEN_GRID_FUSED_CLOVER_FORCE HASEN_GRID_DEVICE_CB HASEN_GRID_BATCH_SMEAR HASEN_GRID_IMPORT_SKIP HASEN_GRID_SHARE_FIELDSTRENGTH HASEN_GRID_GPU_CLOVER_INV HASEN_GRID_CLOVER_STENCIL; do
  if [ -z "${!v:-}" ]; then
    if [ "$(grep -c -a -F "$v" "$BIN" || true)" -gt 0 ]; then
      export "$v=1"
    else
      echo "WARNING: $BIN predates $v: running without it" >&2
    fi
  fi
done
GRID_MG_ENV=$(compgen -e | grep -E '^(GRID_MG_|HASEN_GRID_)' | grep -v -x -E 'HASEN_GRID_(MG_RUNGS|MG_HEATBATH_RUNGS|MIXED_CG_RUNGS|MIXED_CG_HEATBATH_RUNGS)' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)

# ---- 9. MPI / comms / threads -------------------------------------------------------
# The probe's pure-Grid flag set, exactly as smoke16_puregrid.sh
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
# Decomposition 1.2.2.4 = the SOP's and the MG probe's C3 split: local 48.24.24.24,
# level-1 local 12.6.6.6 (GRID_MG_BLOCK default 4.4.4.4), level-2 block 2.3.3.3 fits.
MPI_GEOM=1.2.2.4
NODES=4
NTASKS=16
# --device-mem caps Grid's managed device lattice cache (MemoryManager::DeviceMaxBytes).
# The MG needs headroom OUTSIDE that cache (PaddedCell halo buffers are raw device
# allocations; run_probe_grid_mg.sh DEVICE_MEM_MB note): 8000 is what the 48^3 MG
# probe needed, 12000 OOMs (L119, L167). Fields beyond the cap are evicted to host,
# not OOMed, so a too-small cap shows up as time, a too-large one as an OOM.
# GRID_EXTRA_ARGS (optional, space-separated) is appended verbatim, e.g.
# GRID_EXTRA_ARGS="--log Error,Warning,Message,Performance,Debug" for the M5 instrumented run.
GRID_ARGS=(--grid "$LATT" --mpi "$MPI_GEOM" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap --device-mem "$DEVICE_MEM_MB" ${GRID_EXTRA_ARGS:+$GRID_EXTRA_ARGS})

# ---- 10. Device-memory reporting ------------------------------------------------------
# INTEGRATOR_VERBOSE_MEM=1 re-enables patch 03's MemoryManager::Print around each
# force evaluation (bytes allocated / evictable / max on device, evictions). In a
# TRAJECTORY run that costs a device sync (cudaMemGetInfo) per force call, so such a
# run is a memory probe, not a timing to quote.
# CAVEAT: in FORCES_ONLY it is INERT. Its only call sites are Integrator.h
# (update_P and S) and FORCES_ONLY calls deriv() directly without building the
# Integrator, so no "Memory Manager" block will appear. Exported anyway so the ENV
# line records it and the same env prints the report in a later trajectory run.
# What this run does report: Grid_init's "MemoryManager Cache <DeviceMaxBytes> bytes"
# and per-device totalGlobalMem, any "cudaMalloc failed" / EvictVictims / GRID_ASSERT
# line, and (GPU_MON_MS > 0) an nvidia-smi sampler: local rank 0 of each node logs
# memory.used / memory.total of all four GPUs of its node every GPU_MON_MS ms to
# gpu_mem.<node>.csv. NVML queries do not synchronise CUDA streams.
case "$VERBOSE_MEM_REQ" in
  ''|0) VERBOSE_MEM=off ;;
  *)    export INTEGRATOR_VERBOSE_MEM=1; VERBOSE_MEM=on ;;
esac

# ---- 11. Run directory (refuse to overwrite) -------------------------------------------
DATE_TAG=$(date +%Y_%-m_%-d)                              # unpadded, runs/ convention
RUN_DIR=$BASE/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
LOG=$RUN_DIR/hmc.log

# ---- 12. select_gpu (+ gpu_mon) wrappers + srun step shape ------------------------------
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
      '# nvidia-smi memory sampler on local rank 0 of each node; written by forces48_puregrid.sh' \
      'if [ "${SLURM_LOCALID:-}" = 0 ]; then'
    printf '  timeout -k 10s 95m nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.total --format=csv,noheader,nounits -lms %s > "%s/gpu_mem.${SLURMD_NODENAME:-$(hostname)}.csv" 2>&1 < /dev/null &\n' "$GPU_MON_MS" "$RUN_DIR"
    printf '%s\n' 'fi' 'exec "$@"'
  } > "$GPU_MON"
  chmod +x "$GPU_MON"
  LAUNCH=("$GPU_MON" "${LAUNCH[@]}")
fi

# --export=ALL passes this whole environment to every rank (an --export without
# ALL strips it: the ~10 s exit-1 "shell variable" failure, machines/perlmutter.md).
# SRUN_EXTRA (optional, space-separated) is appended verbatim, e.g. SRUN_EXTRA=--overlap.
SRUN=(srun --jobid="$SLURM_JOB_ID" -N "$NODES" -n "$NTASKS" --ntasks-per-node=4
      --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=cores --gpu-bind=none
      --export=ALL --chdir="$RUN_DIR" ${SRUN_EXTRA:+$SRUN_EXTRA})

# ---- 13. Gated NUMA binding -----------------------------------------------------------
# HARD GATE, not a print (machines/perlmutter.md "Node binding"; L104). Runs exactly
# what select_gpu will run on every one of the 16 ranks (numactl -m i -N i).
if ! "${SRUN[@]}" "$SELECT_GPU" true; then
  die "NUMA gate failed: a local rank cannot bind its own NUMA domain; select_gpu would fail. Nothing was run (rundir $RUN_DIR)."
fi
"${SRUN[@]}" --label "$SELECT_GPU" numactl --show 2>&1 | grep -E 'nodebind' > "$RUN_DIR/numa_binding.txt"

# ---- 14. Provenance: ENV line = line 1 of the log ---------------------------------------
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
# Grid embeds its configure-time GITHASH ("<sha>: (<refs>) [uncommited changes]").
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
GRID_SHA=${GRID_HASH_LINE:0:40}
GRID_SHA=${GRID_SHA:-unknown}
case "$GRID_HASH_LINE" in *uncommit*) GRID_DIRTY=yes ;; *) GRID_DIRTY=no ;; esac

printf 'ENV FORCES48_PUREGRID RUN=%s JOBID=%s BIN=%s BIN_SHA256=%s GRID_SHA=%s GRID_DIRTY=%s MODE=FORCES_ONLY FORCES_SAMPLES=%s FORCES_SKIP=none LATT=%s MPI=%s NODES=%s NTASKS=%s IMPORT_CFG=%s SEED=%s LADDER=%s STRANGE_LEVEL=%s STRANGE_INNER_MULT=%s TAIL_LEVEL=%s INTEGRATOR=%s MDSTEPS=%s GAUGE_INNER_MULT=%s TRAJL=%s MASS_LIGHT=%s MASS_STRANGE=%s CSW=%s BETA=%s U0=%s STOUT_RHO=%s STOUT_NSMEAR=%s RAT=%s/%s/%s TOL_DRV=%s TOL_ACT=%s TOL_STRANGE=%s QUDA_ENV=%s MPICH_IPC=%s MPICH_RDMA=%s MPICH_NIC=%s OMP=%s GRID_MG_RUNGS=%s GRID_MG_HB=%s GRID_MIXED=%s GRID_MIXED_HB=%s GRID_MG_ENV="%s" DEVICE_MEM_MB=%s VERBOSE_MEM=%s GPU_MON_MS=%s GRID_FLAGS="%s"\n' \
  "$RUN" "$SLURM_JOB_ID" "$(basename "$BIN")" "$BIN_SHA256" "$GRID_SHA" "$GRID_DIRTY" \
  "$FORCES_SAMPLES" "$LATT" "$MPI_GEOM" "$NODES" "$NTASKS" "$(basename "$IMPORT_CFG")" \
  "$HMC_SEED_OFFSET" "$HASEN_LADDER" "${HASEN_STRANGE_LEVEL:-default}" "${HASEN_STRANGE_INNER_MULT:-default}" \
  "$HASEN_TAIL_LEVEL" "$INTEGRATOR" "$MDSTEPS" "$GAUGE_INNER_MULT" "$TRAJL" \
  "$MASS_LIGHT" "$MASS_STRANGE" "$CSW" "$BETA" "$U0" "$STOUT_RHO" "$STOUT_NSMEAR" \
  "$RAT_LO" "$RAT_HI" "$RAT_DEGREE" "$TUNE_CG_TOL_DERIV" "$TUNE_CG_TOL_ACTION" \
  "$TUNE_CG_TOL_STRANGE" "$N_QUDA_ENV" "$MPICH_GPU_IPC_ENABLED" "$MPICH_RDMA_ENABLED_CUDA" \
  "$MPICH_OFI_NIC_POLICY" "$OMP_NUM_THREADS" "$MG_RUNGS_TAG" \
  "${HASEN_GRID_MG_HEATBATH_RUNGS:-none}" "${HASEN_GRID_MIXED_CG_RUNGS:-none}" \
  "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}" "$GRID_MG_ENV" \
  "$DEVICE_MEM_MB" "$VERBOSE_MEM" "$GPU_MON_MS" "${GRID_ARGS[*]}" > "$LOG"
{
  printf 'GRID_HASH %s\n' "${GRID_HASH_LINE:-unknown}"
  printf 'BIN_PATH %s\n' "$BIN"
  sed 's/^/NUMA /' "$RUN_DIR/numa_binding.txt"
  printf 'CMD '
  printf '%q ' timeout -k 60s 90m "${SRUN[@]}" "${LAUNCH[@]}"
  printf '\n'
} >> "$LOG"

# ---- 15. Launch (one FORCES_ONLY pass) ----------------------------------------------------
# timeout wrapper: every run keeps one (_SUM.md perlmutter-ops); an aborted step can
# also leave srun hung (port plan s.7 "Hung srun after an abort").
echo "running $(basename "$BIN") FORCES_ONLY samples=$FORCES_SAMPLES MG=$MG_RUNGS_TAG -> $LOG"
timeout -k 60s 90m "${SRUN[@]}" "${LAUNCH[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 16. Summary ---------------------------------------------------------------------------
echo "==== $RUN: exit=$rc  log: $LOG"
grep -a -E '^ENV ' "$LOG"
grep -a -E 'Current Grid git commit hash|CG tol:|Hasenbusch chain|\[Ladder\]|Initial plaquette' "$LOG"
echo "---- FORCES_ONLY per monomial (refresh = heatbath s, time = force s; sample 0 PF0 includes the MG build) ----"
grep -a -E 'FORCES_ONLY (level|samples=)|\[FORCES_ONLY\]' "$LOG"
echo "---- [GridMG lines ----"
grep -a -F '[GridMG' "$LOG"
echo "---- device memory (MemoryManager / bytes / totalGlobalMem) ----"
grep -a -i -E 'bytes|MemoryManager|totalGlobalMem' "$LOG"
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
grep -a -i -E 'assert|abort|out of memory|cudaMalloc failed|EvictVictims|oom[-_ ]?kill|segmentation fault|FATAL|Grid : Error|srun: error' "$LOG" | head -60
echo "log: $LOG"
exit "$rc"
