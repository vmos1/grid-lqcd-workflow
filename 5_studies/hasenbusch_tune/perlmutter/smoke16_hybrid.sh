#!/bin/bash
# =============================================================================
# smoke16_hybrid.sh -- ONE 16^3x48 HMC trajectory of the HYBRID (Grid-TXQCD + QUDA)
# compact-Schur driver, inside an EXISTING allocation (never allocates/submits).
# Task H, 2026-10-02: validation of the strange log-det parity fix (L189).
#
# Physics, seed, tolerances, lattice, config: IDENTICAL to smoke16_puregrid.sh
# (grid-lqcd-workflow/5_studies/hasenbusch_tune/perlmutter/). QUDA environment:
# the 16^3 hybrid recipe whose log is runs/2026_7_11_three_level_recheck_tol11_16/case_qf2.log
# (log shows: QUDA CG + QUDA force assembly on rungs 0,1 and the tail, [Strange] QUDA_FORCE
# active, RECON_NO), plus HASEN_QUDA_FORCE_LOGDET=strange (the strange log-det's QUDA force
# class, QCDLogDetCompactCloverEOQudaForceAction). Comms flags and MPICH settings: the hybrid's
# (--shm-mpi 1 --comms-sequential, MPICH IPC/RDMA 0), srun shape: smoke16_puregrid.sh's
# (select_gpu wrapper, --gpu-bind=none, --export=ALL).
#
# Usage:
#   SLURM_JOB_ID=<id> BIN=<hybrid binary> RUN=<name> [HY_KERNEL_COMPARE=1]
#     [HY_QUDA_RESOURCE_PATH=<dir>] [HASEN_GRID_STRANGE_LOGDET_ODD=0|1] bash smoke16_hybrid.sh
# HY_KERNEL_COMPARE=1 -> QUDA_FORCE_KERNEL_COMPARE=1 (every QUDA force class also runs the Grid
# force and prints the comparison). HY_QUDA_RESOURCE_PATH: QUDA tunecache dir (default
# <rundir>/quda_resource); share it between runs that must take identical kernel launches.
# Output: runs/<YYYY_M_D>_<RUN>/hmc.log, line 1 = ENV provenance line.
# =============================================================================
set -o pipefail

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
WORKFLOW=$BASE/grid-lqcd-workflow

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -n "${SLURM_JOB_ID:-}" ] || die "SLURM_JOB_ID is not set"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be numeric"
[ -n "${BIN:-}" ] || die "BIN is not set"
[ -n "${RUN:-}" ] || die "RUN is not set"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad RUN '$RUN'"
{ [ -f "$BIN" ] && [ -x "$BIN" ]; } || die "BIN is not an executable file: $BIN"
BIN=$(readlink -f "$BIN")
JOB_STATE=$(squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null | head -1)
[ "$JOB_STATE" = RUNNING ] || die "allocation $SLURM_JOB_ID is not RUNNING ('${JOB_STATE:-not found}')"

source "$WORKFLOW/machines/perlmutter.sh" || die "sourcing machines/perlmutter.sh failed"
set -u

LDD_OUT=$(ldd "$BIN" 2>&1) || die "ldd failed on $BIN"
N_NOTFOUND=$(printf '%s\n' "$LDD_OUT" | grep -c 'not found' || true)
N_QUDA=$(printf '%s\n' "$LDD_OUT" | grep -ci 'libquda' || true)
[ "$N_NOTFOUND" -eq 0 ] || die "ldd reports $N_NOTFOUND unresolved libraries for $BIN"
[ "$N_QUDA" -gt 0 ] || die "ldd shows no libquda in $BIN; this smoke is for hybrid builds"
QUDA_LIB=$(printf '%s\n' "$LDD_OUT" | grep -i 'libquda' | awk '{print $3}' | head -1)

# Clear every knob (same families as smoke16_puregrid.sh), keep HASEN_GRID_* from the caller
# (only HASEN_GRID_STRANGE_LOGDET_ODD is meant to be passed here; the pure-Grid speed-up gates
# stay at their compiled default OFF, as in every hybrid production run).
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|STRANGE_EVEN$|NO_METROP$|IMPORT_CFG$)' | grep -v -E '^(HASEN_GRID_|GRID_MG_)')

# ---- physics: verbatim smoke16_puregrid.sh sections 4-6 ----
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
export HASEN_LADDER=-0.245,-0.20,-0.10
export HASEN_TAIL_LEVEL=outer
export INTEGRATOR=ForceGradient
export GAUGE_INNER_MULT=2
export TRAJL=0.35355339059327379
export MDSTEPS=4
export N_TRAJ=1
export HMC_SEED_OFFSET=200
export TUNE_CG_TOL_DERIV=1e-11
export TUNE_CG_TOL_ACTION=1e-12
export TUNE_CG_TOL_STRANGE=1e-9

# ---- QUDA environment: case_qf2's recipe + the strange log-det force class ----
export QUDA_FORCE=1 QUDA_FORCE_KERNEL=1 QUDA_FORCE_RECON_NO=1
export HASEN_QUDA_CG_RUNGS=0,1
export HASEN_QUDA_CG_TAIL=1
export HASEN_QUDA_FORCE_RUNGS=all HASEN_QUDA_FORCE_TAIL=1
export HASEN_QUDA_FORCE_LOGDET=strange
export QUDA_ENABLE_MPS=1 QUDA_ENABLE_P2P=0
if [ "${HY_KERNEL_COMPARE:-0}" = 1 ]; then export QUDA_FORCE_KERNEL_COMPARE=1; fi

# ---- comms / threads: the hybrid's ----
export SLURM_CPU_BIND=cores
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_RDMA_ENABLED_CUDA=0 MPICH_GPU_IPC_ENABLED=0
export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads
MPI_GEOM=1.1.1.4
NTASKS=4
GRID_ARGS=(--grid "$LATT" --mpi "$MPI_GEOM" --accelerator-threads 8 --shm 2048 --shm-mpi 1 --comms-sequential)

DATE_TAG=$(date +%Y_%-m_%-d)
RUN_DIR=$BASE/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
LOG=$RUN_DIR/hmc.log
cp "$0" "$RUN_DIR/smoke16_hybrid.sh"
export QUDA_RESOURCE_PATH=${HY_QUDA_RESOURCE_PATH:-$RUN_DIR/quda_resource}
mkdir -p "$QUDA_RESOURCE_PATH"

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
      --export=ALL --chdir="$RUN_DIR")

if ! "${SRUN[@]}" "$SELECT_GPU" true; then
  die "NUMA gate failed (rundir $RUN_DIR)"
fi

BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
printf 'ENV SMOKE16_HYBRID RUN=%s JOBID=%s BIN=%s BIN_SHA256=%s QUDA_LIB=%s LATT=%s MPI=%s NTASKS=%s IMPORT_CFG=%s SEED=%s N_TRAJ=%s MDSTEPS=%s TRAJL=%s INTEGRATOR=%s GAUGE_INNER_MULT=%s LADDER=%s TAIL_LEVEL=%s MASS_LIGHT=%s MASS_STRANGE=%s CSW=%s BETA=%s U0=%s STOUT_RHO=%s STOUT_NSMEAR=%s RAT=%s/%s/%s TOL_DRV=%s TOL_ACT=%s TOL_STRANGE=%s METROP=on QUDA_HASEN_ENV="%s" GRID_FLAGS="%s"\n' \
  "$RUN" "$SLURM_JOB_ID" "$(basename "$BIN")" "$BIN_SHA256" "$QUDA_LIB" \
  "$LATT" "$MPI_GEOM" "$NTASKS" "$(basename "$IMPORT_CFG")" "$HMC_SEED_OFFSET" "$N_TRAJ" \
  "$MDSTEPS" "$TRAJL" "$INTEGRATOR" "$GAUGE_INNER_MULT" "$HASEN_LADDER" "$HASEN_TAIL_LEVEL" \
  "$MASS_LIGHT" "$MASS_STRANGE" "$CSW" "$BETA" "$U0" "$STOUT_RHO" "$STOUT_NSMEAR" \
  "$RAT_LO" "$RAT_HI" "$RAT_DEGREE" "$TUNE_CG_TOL_DERIV" "$TUNE_CG_TOL_ACTION" "$TUNE_CG_TOL_STRANGE" \
  "$(compgen -e | grep -E '^(QUDA_|HASEN_|GRID_MG_|MPICH_|OMP_)' | sort | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
  "${GRID_ARGS[*]}" > "$LOG"
{
  printf 'GRID_HASH %s\n' "${GRID_HASH_LINE:-unknown}"
  printf 'BIN_PATH %s\n' "$BIN"
  printf 'CMD '
  printf '%q ' timeout -k 60s 30m "${SRUN[@]}" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}"
  printf '\n'
} >> "$LOG"

echo "running $(basename "$BIN") -> $LOG"
timeout -k 60s 30m "${SRUN[@]}" "$SELECT_GPU" "$BIN" "${GRID_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

echo "==== $RUN: exit=$rc  log: $LOG"
grep -a -E '^ENV ' "$LOG"
grep -a -E '\[StrangeLogDet\]|\[Strange\]|QCDLogDetCompactCloverEOAction.*\] S = |QudaForceCompare.*logdet' "$LOG" | head -20
grep -a -E 'Total H before trajectory|Total H after trajectory|exp\(-dH\)|Metropolis_test|Total time for trajectory|Plaquette: \[' "$LOG"
exit "$rc"
