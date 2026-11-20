#!/bin/bash
# =============================================================================
# traj48_lq.sh -- full HMC trajectories of the pure-Grid compact-Schur driver at 48^3x96 on lq,
# run INSIDE a Slurm job (never submits). The lq twin of ../perlmutter/traj48_puregrid.sh with
# its baseline profile (LADDER_PROFILE=baseG, the default there): physics, ladder, integrator,
# tolerances, solver routes, speed-up switches and Grid flags are that launcher's, value for
# value (its sections 5-9). A gauge + RNG checkpoint is written after every trajectory, and
# RESUME_FROM continues the chain of an earlier run of THIS launcher.
#
#   gpu (LQ_TARGET=gpu, default)  lq2: 4 nodes x 4 A100, one rank per GPU, --mpi 1.2.2.4 =
#                                 Perlmutter's split (local 48.24.24.24), --device-mem 8000.
#   cpu (LQ_TARGET=cpu)           lq3: 16 ranks x 4 cores per node (Grid-CPU's best layout,
#                                 /lustre2/nplqcd/Grid-CPU/AGENT_REFERENCE.md), default 1 node,
#                                 --mpi 1.2.2.4 (the same split).
# Launch: srun --mpi=pmix (lq2: the Grid-TXQCD recipe with gompi; lq3: Grid-CPU's OpenMPI),
# one step over all nodes.
#
# Usage (inside a job; job scripts traj48_m10_gpu_job.sh, traj48_m10_cpu_job.sh):
#   BIN=<binary> RUN=<name> [N_TRAJ=n] [HMC_SEED_OFFSET=k] [RESUME_FROM=<run dir>] bash traj48_lq.sh
# Options (everything else is fixed):
#   N_TRAJ           default 1
#   HMC_SEED_OFFSET  default 0 (Perlmutter's default); fresh start only, a resume restores the RNGs
#   RESUME_FROM      run dir of an earlier traj48_lq.sh run with the same physics: continue its
#                    chain from its last complete checkpoint
#   LQ_TARGET        gpu (default) or cpu
#   MPI_GEOM         default 1.2.2.4; the product must equal the job's ranks
#   TIMEOUT_MIN      default 210
#   DEVICE_MEM_MB    gpu only, default 8000 (Perlmutter's default; dH is bit-exact across caps,
#                    traj48_puregrid.sh section 9)
#   GPU_MON_MS       gpu only, default 2000: nvidia-smi memory sampler on local rank 0 of each node
#   HASEN_GRID_*, GRID_MG_*  pass through (recorded in the ENV line), as on Perlmutter
#
# Output: runs/<YYYY_M_D>_<RUN>/hmc.log, ckpt/ (checkpoints, 7.2 GB per trajectory, on Lustre),
# select_gpu or nothing (cpu), numa_binding.txt, gpu_mem.*.csv. Line 1 of hmc.log is the ENV line
# in Perlmutter's TRAJ48_PUREGRID format (same keys, same order; MPICH_* read "na"), then
# GRID_HASH, BIN_PATH, NUMA, CMD and one LAUNCH line with the lq facts. Refuses to overwrite.
# =============================================================================
set -o pipefail

ROOT=/lustre2/nplqcd/vayyar/grid_qcd
WF=$ROOT/grid-lqcd-workflow
SELF=$(readlink -f "$0")

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
USAGE="usage: BIN=<binary> RUN=<name> [N_TRAJ=n] [RESUME_FROM=<run dir>] bash $0 (inside a Slurm job)"

# Trajectory numbers whose checkpoint is complete in a log of this driver (both "Written" lines);
# traj48_puregrid.sh complete_ckpts, verbatim.
complete_ckpts() {
  awk '/Written BINARY RNG / { f = $0; sub(/ checksum .*/, "", f)
                               if (f ~ /ckpoint_rng\.[0-9]+$/) { sub(/.*ckpoint_rng\./, "", f); r[f] = 1 } }
       /Written ILDG Configuration on / { f = $0; sub(/ checksum .*/, "", f)
                               if (f ~ /ckpoint_lat\.[0-9]+$/) { sub(/.*ckpoint_lat\./, "", f); l[f] = 1 } }
       END { for (n in l) if (n in r) print n }' "$1" | sort -n
}

# Per-trajectory table (traj48_puregrid.sh traj_table, verbatim): wall_s = MD only (H before ->
# H after, the primary timing), cycle_s = Grid's "Total time for trajectory".
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
    END {
      printf "%-6s %-3s %-20s %-20s %-20s %-10s %-8s %-8s %-18s %-18s\n", "traj", "M", "H_before", "H_after", "dH", "exp(-dH)", "wall_s", "cycle_s", "plaq_thin", "plaq_smeared"
      na = 0; nd = 0
      for (j = 1; j <= cnt; j++) {
        n = ord[j]; if (n + 0 < lo + 0 || n + 0 > hi + 0) continue
        w = ((n in tb) && (n in ta)) ? sprintf("%.1f", ta[n] - tb[n]) : "-"
        printf "%-6s %-3s %-20s %-20s %-20s %-10s %-8s %-8s %-18s %-18s\n", n, (n in ac) ? ac[n] : "-", (n in hb) ? hb[n] : "-", (n in ha) ? ha[n] : "-", (n in dh) ? dh[n] : "-", (n in ex) ? ex[n] : "-", w, (n in cy) ? cy[n] : "-", (n in pu) ? pu[n] : "-", (n in ps) ? ps[n] : "-"
        if (n in ac) { nd++; if (ac[n] == "ACC") na++ }
      }
      printf "Metropolis: %d accepted of %d decided\n", na, nd
    }' "$1"
}

# ---- 1. Inputs -------------------------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || die "not inside a Slurm job (SLURM_JOB_ID unset). $USAGE"
[ -n "${BIN:-}" ] || die "BIN is not set. $USAGE"
[ -n "${RUN:-}" ] || die "RUN is not set. $USAGE"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN may contain only letters, digits, '.', '_' and '-', got '$RUN'"
{ [ -f "$BIN" ] && [ -x "$BIN" ]; } || die "BIN is not an executable file: $BIN"
BIN=$(readlink -f "$BIN")
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
DATE_TAG=$(date +%Y_%-m_%-d)
G_NTRAJ=${N_TRAJ:-1}
SEED_REQ=${HMC_SEED_OFFSET:-}
RESUME_FROM=${RESUME_FROM:-}
G_TARGET=${LQ_TARGET:-gpu}
G_MPI=${MPI_GEOM:-1.2.2.4}
TIMEOUT_MIN=${TIMEOUT_MIN:-210}
DEVICE_MEM_MB=${DEVICE_MEM_MB:-8000}
GPU_MON_MS=${GPU_MON_MS:-2000}
case "$G_TARGET" in gpu|cpu) ;; *) die "LQ_TARGET must be gpu or cpu, got '$G_TARGET'" ;; esac
export LQ_TARGET=$G_TARGET
[[ "$G_NTRAJ" =~ ^[1-9][0-9]*$ ]] || die "N_TRAJ must be a positive integer, got '$G_NTRAJ'"
[ -z "$SEED_REQ" ] || [[ "$SEED_REQ" =~ ^[0-9]+$ ]] || die "HMC_SEED_OFFSET must be a non-negative integer, got '$SEED_REQ'"
[[ "$G_MPI" =~ ^[1-9][0-9]*\.[1-9][0-9]*\.[1-9][0-9]*\.[1-9][0-9]*$ ]] || die "MPI_GEOM must look like 1.2.2.4, got '$G_MPI'"
[[ "$TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]] || die "TIMEOUT_MIN must be a positive integer (minutes), got '$TIMEOUT_MIN'"
[[ "$DEVICE_MEM_MB" =~ ^[1-9][0-9]*$ ]] || die "DEVICE_MEM_MB must be a positive integer (MB), got '$DEVICE_MEM_MB'"
[[ "$GPU_MON_MS" =~ ^[0-9]+$ ]] || die "GPU_MON_MS must be a non-negative integer (ms), got '$GPU_MON_MS'"
# Ranks = the job's tasks; the decomposition must use all of them.
NODES=${SLURM_JOB_NUM_NODES:?SLURM_JOB_NUM_NODES unset}
NTASKS=${SLURM_NTASKS:?SLURM_NTASKS unset (give the job --ntasks-per-node)}
PER_NODE=$(( NTASKS / NODES ))
[ $(( PER_NODE * NODES )) -eq "$NTASKS" ] || die "$NTASKS tasks do not split evenly over $NODES nodes"
IFS=. read -r M1 M2 M3 M4 <<< "$G_MPI"
[ $(( M1 * M2 * M3 * M4 )) -eq "$NTASKS" ] || die "MPI_GEOM=$G_MPI needs $(( M1 * M2 * M3 * M4 )) ranks, the job has $NTASKS"
CPT=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK unset (give the job --cpus-per-task)}

# ---- 2. Environment, loader and release checks (as gate16_puregrid.sh section 2) ----------------
# shellcheck source=/dev/null
source "$WF/machines/lq.sh" || die "sourcing $WF/machines/lq.sh failed"
set -u
LDD_OUT=$(ldd "$BIN" 2>&1) || die "ldd failed on $BIN: $LDD_OUT"
if printf '%s\n' "$LDD_OUT" | grep -q 'not found'; then
  printf '%s\n' "$LDD_OUT" | grep 'not found' >&2
  die "ldd reports unresolved libraries for $BIN on $(hostname)"
fi
printf '%s\n' "$LDD_OUT" | grep -qi 'libquda' && die "ldd shows libquda in $BIN; this launcher is for pure-Grid builds only"
LIBMPI=$(printf '%s\n' "$LDD_OUT" | grep -o -m1 '/[^ ]*libmpi\.so[^ ]*' || true)
case $G_TARGET/${LQ_MPI_STACK:-} in
  gpu/gompi)       STACK_DIR=/srv/software/el8/x86_64/eb/OpenMPI/ ;;
  cpu/lq3-ompi418) STACK_DIR=${LQ_CPU_MPI_ROOT:?LQ_CPU_MPI_ROOT unset}/ ;;
  *)               die "LQ_TARGET=$G_TARGET with LQ_MPI_STACK=${LQ_MPI_STACK:-} cannot be launched by srun (gpu: gompi, cpu: lq3-ompi418)" ;;
esac
[[ "$LIBMPI" == "$STACK_DIR"* ]] || die "LQ_MPI_STACK=$LQ_MPI_STACK but $BIN loads libmpi from '$LIBMPI'"
PROV=$(dirname "$(dirname "$BIN")")/PROVENANCE.txt
if [ -f "$PROV" ]; then
  WANT=$(grep -m1 '^exe:' "$PROV" | grep -o 'sha256=[0-9a-f]*' | cut -d= -f2)
  [ "$WANT" = "$BIN_SHA256" ] || die "sha256 of $BIN is $BIN_SHA256, its release record $PROV says '${WANT}'"
fi
if [ "$G_TARGET" = gpu ]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: not a GPU node ($(hostname))"
fi

# ---- 3. Clear every knob this run does not set (traj48_puregrid.sh section 4) ------------------
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|STRANGE_EVEN$|NO_METROP$|IMPORT_CFG$|N_TRAJ$|HMC_SEED_OFFSET$|RAT_DEGREE_MD$|TUNE_CG_TOL_STRANGE_MD$)' | grep -v -E '^(HASEN_GRID_|GRID_MG_)')
N_QUDA_ENV=$(compgen -e | grep -c '^QUDA_' || true)
[ "$N_QUDA_ENV" -eq 0 ] || die "QUDA_* variables still set after clearing"

# ---- 4. Physics (traj48_puregrid.sh section 5, value for value) ---------------------------------
export LATT=48.48.48.96
START_CFG=$ROOT/inputs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime   # ILDG, 6.1 GB
START_TRAJ=2000
export BETA=6.3
export U0=0.84570646270714           # the driver's local default 1.0 is wrong (u0-48cube)
export CSW=1.20536588031793
export MASS_LIGHT=-0.2416            # must equal HASEN_LADDER[0]
export MASS_STRANGE=-0.2050
export STOUT_RHO=0.125 STOUT_NSMEAR=1
export RAT_LO=7e-4 RAT_HI=35.0 RAT_DEGREE=20
export RAT_BOUNDS_CHECK_FREQ=10 RAT_BOUNDS_CHECK_TOL=1e-6 RAT_BOUNDS_CHECK_ABORT=1
RAT_BC_TAG=$RAT_BOUNDS_CHECK_FREQ/$RAT_BOUNDS_CHECK_TOL/$RAT_BOUNDS_CHECK_ABORT
grep -q -a -F RAT_BOUNDS_CHECK_FREQ "$BIN" || die "$BIN predates the strange bounds check (binary m8 or later needed)"

# ---- 5. Baseline ladder and integrator (traj48_puregrid.sh section 6, LADDER_PROFILE=baseG) -----
LADDER_PROFILE=baseG
export HASEN_LADDER=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870
export HASEN_TAIL_LEVEL=inner
export MDSTEPS=12
export INTEGRATOR=ForceGradient
export GAUGE_INNER_MULT=2
export TRAJL=1.0
LEVEL_MULT_TAG=none

# ---- 6. Trajectories and tolerances (section 7) -------------------------------------------------
export N_TRAJ=$G_NTRAJ
export TUNE_CG_TOL_DERIV=1e-11
export TUNE_CG_TOL_ACTION=1e-12
export TUNE_CG_TOL_STRANGE=1e-9

# ---- 7. Rung solver routes and production speed-ups (section 8) --------------------------------
# Routes default as on Perlmutter; a caller may set one EMPTY to switch it off. Every non-empty
# route and every HASEN_GRID_* switch must be named in $BIN (an older binary would silently run CG).
if [ -z "${HASEN_GRID_MG_RUNGS+x}" ]; then HASEN_GRID_MG_RUNGS=0,1,2; fi
if [ -z "${HASEN_GRID_MG_HEATBATH_RUNGS+x}" ]; then HASEN_GRID_MG_HEATBATH_RUNGS=0,1; fi
if [ -z "${HASEN_GRID_MIXED_CG_RUNGS+x}" ]; then HASEN_GRID_MIXED_CG_RUNGS=3; fi
if [ -z "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS+x}" ]; then HASEN_GRID_MIXED_CG_HEATBATH_RUNGS=2,3; fi
for v in HASEN_GRID_MG_RUNGS HASEN_GRID_MG_HEATBATH_RUNGS HASEN_GRID_MIXED_CG_RUNGS HASEN_GRID_MIXED_CG_HEATBATH_RUNGS; do
  if [ -n "${!v}" ]; then
    [[ "${!v}" =~ ^[0-3](,[0-3])*$ ]] || die "$v must list ratio rungs 0..3 of the baseline ladder, got '${!v}'"
    grep -q -a -F "$v" "$BIN" || die "$BIN has no $v support"
    export "$v"
  else
    unset "$v"
  fi
done
for v in HASEN_GRID_FUSED_CLOVER_FORCE HASEN_GRID_DEVICE_CB HASEN_GRID_BATCH_SMEAR HASEN_GRID_IMPORT_SKIP HASEN_GRID_SHARE_FIELDSTRENGTH HASEN_GRID_GPU_CLOVER_INV HASEN_GRID_CLOVER_STENCIL; do
  if [ -z "${!v:-}" ]; then
    grep -q -a -F "$v" "$BIN" || die "$BIN predates $v (m10 has every switch)"
    export "$v=1"
  fi
done
HASEN_GRID_EXTRA=
while read -r v; do
  [ -n "$v" ] || continue
  HASEN_GRID_EXTRA="$HASEN_GRID_EXTRA$v=${!v} "
  [ "${!v}" = 0 ] || grep -q -a -F "$v" "$BIN" || die "$v is set but $BIN does not contain that name"
done < <(compgen -e | grep '^HASEN_GRID_' | grep -v -x -E 'HASEN_GRID_(MG_RUNGS|MG_HEATBATH_RUNGS|MIXED_CG_RUNGS|MIXED_CG_HEATBATH_RUNGS)')
GRID_MG_ENV=$(compgen -e | grep -E '^(GRID_MG_|HASEN_GRID_FUSED)' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)

# ---- 8. Threads and Grid flags ------------------------------------------------------------------
if [ "$G_TARGET" = gpu ]; then
  export OMP_NUM_THREADS=8           # Perlmutter's
  GRID_ARGS=(--grid "$LATT" --mpi "$G_MPI" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap --device-mem "$DEVICE_MEM_MB")
else
  export OMP_NUM_THREADS=$CPT OMP_PLACES=cores OMP_PROC_BIND=close
  GRID_ARGS=(--grid "$LATT" --mpi "$G_MPI" --threads "$CPT" --shm 2048)
fi

# ---- 9. Start: fresh from cfg_2000, or resume (traj48_puregrid.sh section 11, shortened) --------
V=1
for x in ${LATT//./ }; do V=$(( V * x )); done
LAT_MIN=$(( V * 4 * 9 * 16 ))        # ILDG payload, IEEE64BIG
RNG_MIN=$(( 104 * (V + 1) ))         # RNG_SITMO state + the serial RNG
PHYS="$LATT:$HASEN_LADDER:$MASS_LIGHT:$MASS_STRANGE:$CSW:$BETA:$U0:$STOUT_RHO/$STOUT_NSMEAR:$RAT_LO/$RAT_HI/$RAT_DEGREE:$INTEGRATOR:$MDSTEPS:$GAUGE_INNER_MULT:$TRAJL:default/default/$HASEN_TAIL_LEVEL:$TUNE_CG_TOL_DERIV/$TUNE_CG_TOL_ACTION/$TUNE_CG_TOL_STRANGE"
PREV= PREV_ENV= PREV_LAT= PREV_RNG= PREV_RNG_CSUM= PREV_PLAQ= RESUME_N=
prev_field() { printf '%s\n' "$PREV_ENV" | tr ' ' '\n' | sed -n "s/^$1=//p" | head -1; }
if [ -z "$RESUME_FROM" ]; then
  MODE=fresh
  [ -f "$START_CFG" ] || die "starting configuration not found: $START_CFG"
  export IMPORT_CFG=$START_CFG CKPT_START_TRAJ=$START_TRAJ HMC_SEED_OFFSET=${SEED_REQ:-0}
  SEED_TAG=$HMC_SEED_OFFSET START_TAG=$(basename "$START_CFG") IMPORT_TAG=$(basename "$START_CFG")
  CHAIN=${DATE_TAG}_${RUN} SEGMENT=1
else
  MODE=resume
  PREV=$(readlink -f "$RESUME_FROM") || die "RESUME_FROM does not resolve: $RESUME_FROM"
  [ -f "$PREV/hmc.log" ] || die "no hmc.log in $PREV"
  PREV_ENV=$(head -1 "$PREV/hmc.log")
  case "$PREV_ENV" in "ENV TRAJ48_PUREGRID "*) ;; *) die "$PREV is not a 48^3 trajectory run (line 1 is not 'ENV TRAJ48_PUREGRID ...')" ;; esac
  [ "$(prev_field PHYS)" = "$PHYS" ] || die "physics of $PREV differs: theirs '$(prev_field PHYS)', ours '$PHYS'"
  [ -d "$PREV/ckpt" ] || die "no ckpt directory in $PREV"
  RESUME_N=$(complete_ckpts "$PREV/hmc.log" | tail -1)
  [ -n "$RESUME_N" ] || die "$PREV wrote no complete checkpoint"
  PREV_LAT=$PREV/ckpt/ckpoint_lat.$RESUME_N
  PREV_RNG=$PREV/ckpt/ckpoint_rng.$RESUME_N
  { [ -f "$PREV_LAT" ] && [ -f "$PREV_RNG" ]; } || die "checkpoint $RESUME_N of $PREV is logged but a file is missing"
  [ "$(stat -L -c %s "$PREV_LAT")" -ge "$LAT_MIN" ] || die "$PREV_LAT is truncated"
  [ "$(stat -L -c %s "$PREV_RNG")" -ge "$RNG_MIN" ] || die "$PREV_RNG is truncated"
  PREV_RNG_CSUM=$(grep -a -F 'Written BINARY RNG ' "$PREV/hmc.log" | grep -a -F "/ckpoint_rng.$RESUME_N checksum " | tail -1 | awk '{print $NF}')
  PREV_PLAQ=$(grep -a -F "Plaquette: [ $RESUME_N ] " "$PREV/hmc.log" | head -1 | awk '{print $NF}')
  [ "$(prev_field BIN_SHA256)" = "$BIN_SHA256" ] || warn "binary differs from the run being resumed"
  [ -z "$SEED_REQ" ] || warn "HMC_SEED_OFFSET is ignored on a resume: both RNGs come from ckpoint_rng.$RESUME_N"
  export CKPT_RESUME_TRAJ=$RESUME_N HMC_SEED_OFFSET=0
  SEED_TAG=restored START_TAG=ckpt:$RESUME_N IMPORT_TAG=none
  CHAIN=$(prev_field CHAIN)
  SEGMENT=$(( $(prev_field SEGMENT) + 1 ))
fi

# ---- 10. Run directory and checkpoints (on Lustre, inside the run dir) --------------------------
RUN_DIR=$ROOT/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
mkdir "$RUN_DIR/ckpt" || die "cannot create $RUN_DIR/ckpt"
LOG=$RUN_DIR/hmc.log
if [ "$MODE" = resume ]; then
  RESUME_LAT=$(readlink -f "$PREV_LAT")
  ln -s "$RESUME_LAT" "$RUN_DIR/ckpt/ckpoint_lat.$RESUME_N" || die "cannot link the resume gauge file"
  ln -s "$(readlink -f "$PREV_RNG")" "$RUN_DIR/ckpt/ckpoint_rng.$RESUME_N" || die "cannot link the resume RNG file"
else
  RESUME_LAT=none
fi
export CKPT_DIR=$RUN_DIR/ckpt CKPT_INTERVAL=1

# ---- 11. Binding and the srun step --------------------------------------------------------------
# gpu: local rank L -> GPU L (one per rank) and that GPU's NUMA domain, read from sysfs on this
# node (lq2 nodes are identical). cpu: srun binds each rank to CPT cores, memory first-touch.
if [ "$G_TARGET" = gpu ]; then
  IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
  [ "${#GPUS[@]}" -eq "$PER_NODE" ] || die "one rank per GPU: $PER_NODE ranks per node but ${#GPUS[@]} GPUs ('${CUDA_VISIBLE_DEVICES:-}')"
  NUMAS=() BUSES=()
  for g in "${GPUS[@]}"; do
    bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$g" | tr 'A-F' 'a-f')
    BUSES+=("$bus")
    NUMAS+=("$(cat "/sys/bus/pci/devices/${bus#0000}/numa_node")")
  done
  SELECT_GPU=$RUN_DIR/select_gpu
  printf '%s\n' \
    '#!/bin/bash' \
    '# srun local rank -> GPU + the NUMA domain of that GPU (written by traj48_lq.sh)' \
    "RANK_GPU=(${GPUS[*]})" \
    "RANK_NUMA=(${NUMAS[*]})" \
    'L=${SLURM_LOCALID:?select_gpu: not started by srun}' \
    'export CUDA_DEVICE_ORDER=PCI_BUS_ID' \
    'export CUDA_VISIBLE_DEVICES=${RANK_GPU[$L]}' \
    'exec numactl -m "${RANK_NUMA[$L]}" -N "${RANK_NUMA[$L]}" "$@"' > "$SELECT_GPU"
  chmod +x "$SELECT_GPU"
  WRAP=("$SELECT_GPU")
  PROBE_WRAP=("$SELECT_GPU")
  SRUN=(srun --mpi=pmix --nodes="$NODES" --ntasks="$NTASKS" --ntasks-per-node="$PER_NODE" --cpus-per-task="$CPT" --cpu-bind=none --chdir="$RUN_DIR")
  PROBE='echo "host=$(hostname -s) gpu=$CUDA_VISIBLE_DEVICES bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $CUDA_VISIBLE_DEVICES) $(numactl --show | grep -E "^(nodebind|membind)" | tr "\n" " ")"'
  if [ "$GPU_MON_MS" -gt 0 ]; then
    GPU_MON=$RUN_DIR/gpu_mon
    {
      printf '%s\n' '#!/bin/bash' '# nvidia-smi memory sampler on local rank 0 of each node (written by traj48_lq.sh)' 'if [ "${SLURM_LOCALID:-}" = 0 ]; then'
      printf '  timeout -k 10s %sm nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.total --format=csv,noheader,nounits -lms %s > "%s/gpu_mem.$(hostname -s).csv" 2>&1 < /dev/null &\n' "$(( TIMEOUT_MIN + 5 ))" "$GPU_MON_MS" "$RUN_DIR"
      printf '%s\n' 'fi' 'exec "$@"'
    } > "$GPU_MON"
    chmod +x "$GPU_MON"
    WRAP=("$GPU_MON" "$SELECT_GPU")
  fi
else
  WRAP=()
  PROBE_WRAP=()
  SRUN=(srun --mpi=pmix --nodes="$NODES" --ntasks="$NTASKS" --ntasks-per-node="$PER_NODE" --cpus-per-task="$CPT" --cpu-bind=cores --chdir="$RUN_DIR")
  PROBE='echo "host=$(hostname -s) cpus=$(taskset -cp $$ | cut -d: -f2 | tr -d " ") $(numactl --show | grep -E "^(nodebind|membind)" | tr "\n" " ")"'
fi
# Gate: every rank reports its host, GPU (bus id) or cores, and NUMA binding; a failure stops here.
if ! timeout -k 10s 5m "${SRUN[@]}" --label "${PROBE_WRAP[@]}" bash -c "$PROBE" > "$RUN_DIR/numa_binding.raw" 2>&1; then
  cat "$RUN_DIR/numa_binding.raw" >&2
  die "binding probe failed: srun could not start every rank. Nothing was run (rundir $RUN_DIR)."
fi
sed -n -E 's/^ *([0-9]+): /\1: /p' "$RUN_DIR/numa_binding.raw" | sort -n > "$RUN_DIR/numa_binding.txt"
[ "$(wc -l < "$RUN_DIR/numa_binding.txt")" -eq "$NTASKS" ] || die "binding probe returned $(wc -l < "$RUN_DIR/numa_binding.txt") of $NTASKS ranks"

# ---- 12. Provenance: ENV line in Perlmutter's TRAJ48_PUREGRID format, then the lq LAUNCH line ---
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
GRID_SHA=${GRID_HASH_LINE:0:40}
GRID_SHA=${GRID_SHA:-unknown}
case "$GRID_HASH_LINE" in *uncommit*) GRID_DIRTY=yes ;; *) GRID_DIRTY=no ;; esac
ENV_KV=(
  "RUN=$RUN" "JOBID=$SLURM_JOB_ID" "MODE=$MODE" "CHAIN=$CHAIN" "SEGMENT=$SEGMENT"
  "START=$START_TAG" "N_TRAJ=$N_TRAJ" "SEED=$SEED_TAG" "METROP=on"
  "BIN=$(basename "$BIN")" "BIN_SHA256=$BIN_SHA256" "GRID_SHA=$GRID_SHA" "GRID_DIRTY=$GRID_DIRTY"
  "IMPORT_CFG=$IMPORT_TAG" "CKPT_START_TRAJ=${CKPT_START_TRAJ:-none}"
  "CKPT_RESUME_TRAJ=${CKPT_RESUME_TRAJ:-none}" "RESUME_FROM=${PREV:-none}" "RESUME_LAT=$RESUME_LAT"
  "RESUME_RNG_CSUM=${PREV_RNG_CSUM:-none}" "RESUME_PLAQ=${PREV_PLAQ:-none}"
  "CKPT_DIR=$CKPT_DIR" "CKPT_FS=lustre2" "CKPT_INTERVAL=$CKPT_INTERVAL" "SMEARED_CKPT=no"
  "LATT=$LATT" "MPI=$G_MPI" "NODES=$NODES" "NTASKS=$NTASKS"
  "LADDER=$HASEN_LADDER" "STRANGE_LEVEL=default" "STRANGE_INNER_MULT=default"
  "TAIL_LEVEL=$HASEN_TAIL_LEVEL" "STRANGE_RUNGS=none" "LADDER_PROFILE=$LADDER_PROFILE"
  "INTEGRATOR=$INTEGRATOR" "MDSTEPS=$MDSTEPS" "GAUGE_INNER_MULT=$GAUGE_INNER_MULT"
  "LEVEL_LIGHT=none" "LEVEL_STRANGE=none" "LEVEL_TAIL=none" "LEVEL_GAUGE=none" "LEVEL_MULT=$LEVEL_MULT_TAG"
  "TRAJL=$TRAJL" "MASS_LIGHT=$MASS_LIGHT" "MASS_STRANGE=$MASS_STRANGE" "CSW=$CSW" "BETA=$BETA" "U0=$U0"
  "STOUT_RHO=$STOUT_RHO" "STOUT_NSMEAR=$STOUT_NSMEAR" "RAT=$RAT_LO/$RAT_HI/$RAT_DEGREE"
  "RAT_DEGREE_MD=$RAT_DEGREE" "TOL_STRANGE_MD=$TUNE_CG_TOL_STRANGE" "RAT_BOUNDS_CHECK=$RAT_BC_TAG"
  "TOL_DRV=$TUNE_CG_TOL_DERIV" "TOL_ACT=$TUNE_CG_TOL_ACTION" "TOL_STRANGE=$TUNE_CG_TOL_STRANGE"
  "GRID_MG_RUNGS=${HASEN_GRID_MG_RUNGS:-none}" "GRID_MG_HEATBATH_RUNGS=${HASEN_GRID_MG_HEATBATH_RUNGS:-none}"
  "GRID_MIXED_CG_RUNGS=${HASEN_GRID_MIXED_CG_RUNGS:-none}" "GRID_MIXED_CG_HEATBATH_RUNGS=${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}"
  "QUDA_ENV=$N_QUDA_ENV" "MPICH_IPC=na" "MPICH_RDMA=na"
  "MPICH_NIC=na" "OMP=$OMP_NUM_THREADS" "DEVICE_MEM_MB=$([ "$G_TARGET" = gpu ] && echo "$DEVICE_MEM_MB" || echo na)"
  "VERBOSE_MEM=off" "GPU_MON_MS=$([ "$G_TARGET" = gpu ] && echo "$GPU_MON_MS" || echo 0)" "TIMEOUT_MIN=$TIMEOUT_MIN" "ALLOC_LEFT_MIN=unknown"
  "PHYS=$PHYS"
  "HASEN_GRID_EXTRA=\"$HASEN_GRID_EXTRA\"" "GRID_MG_ENV=\"$GRID_MG_ENV\"" "GRID_FLAGS=\"${GRID_ARGS[*]}\""
)
printf 'ENV TRAJ48_PUREGRID %s\n' "${ENV_KV[*]}" > "$LOG"
{
  printf 'GRID_HASH %s\n' "${GRID_HASH_LINE:-unknown}"
  printf 'BIN_PATH %s\n' "$BIN"
  sed 's/^/NUMA /' "$RUN_DIR/numa_binding.txt"
  printf 'CMD '
  printf '%q ' timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "${WRAP[@]}" "$BIN" "${GRID_ARGS[@]}"
  printf '\n'
  printf 'LAUNCH host=%s target=%s launcher=srun mpi_stack=%s libmpi=%s ompi="%s" nodes=%s ranks_per_node=%s cpus_per_task=%s nodelist=%s' \
    "$(hostname)" "$G_TARGET" "$LQ_MPI_STACK" "$LIBMPI" "$(mpirun --version 2>&1 | head -1)" "$NODES" "$PER_NODE" "$CPT" "${SLURM_JOB_NODELIST:-?}"
  if [ "$G_TARGET" = gpu ]; then
    printf ' modules="%s" gpus=%s buses=%s gpu_numa=%s gpu="%s"' "$(module -t list 2>&1 | tr '\n' ' ')" \
      "$(IFS=,; echo "${GPUS[*]}")" "$(IFS=,; echo "${BUSES[*]}")" "$(IFS=,; echo "${NUMAS[*]}")" \
      "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader -i "${GPUS[0]}")"
  else
    printf ' cxx="%s" cpu="%s" omp="OMP_NUM_THREADS=%s OMP_PLACES=%s OMP_PROC_BIND=%s"' "$(mpicxx --version 2>&1 | head -1)" \
      "$(lscpu | sed -n 's/^Model name: *//p' | head -1)" "$OMP_NUM_THREADS" "$OMP_PLACES" "$OMP_PROC_BIND"
  fi
  printf ' mca="%s" ucx="%s"\n' \
    "$(compgen -e | grep '^OMPI_MCA_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
    "$(compgen -e | grep '^UCX_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)"
} >> "$LOG"

# ---- 13. Launch -----------------------------------------------------------------------------------
echo "running $(basename "$BIN") $MODE start=$START_TAG N_TRAJ=$N_TRAJ on $NODES node(s) x $PER_NODE ranks -> $LOG"
timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "${WRAP[@]}" "$BIN" "${GRID_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 14. Summary ----------------------------------------------------------------------------------
echo "==== $RUN: exit=$rc  log: $LOG"
[ "$rc" -eq 124 ] && echo "exit 124 = TIMEOUT_MIN ($TIMEOUT_MIN min) reached: the trajectory in flight is lost, completed ones are checkpointed"
grep -a -E '^(ENV|LAUNCH) ' "$LOG"
grep -a -E 'communicator of size|bus id|Current Grid git commit hash|CG tol:|Hasenbusch chain|\[Ladder\]|\[Levels\]|HMC_SEED_OFFSET=|IMPORT_CFG=|\[Ckpt\]|Initial plaquette' "$LOG" | grep -a -v -E '^(ENV|CMD|LAUNCH) '
if [ "$MODE" = resume ]; then
  INIT_PLAQ=$(grep -a -F 'Initial plaquette = ' "$LOG" | head -1 | awk '{print $NF}')
  printf 'continuity: previous Plaquette:[ %s ] %s, this run initial plaquette %s\n' "$RESUME_N" "${PREV_PLAQ:-n/a}" "${INIT_PLAQ:-n/a}"
fi
echo "---- per trajectory (wall = MD only, PRIMARY; cycle = Grid Total time) ----"
traj_table "$LOG" 0 999999999
echo "---- FORCES traj= (per-monomial force averages over each trajectory) ----"
grep -a -E 'FORCES traj=' "$LOG"
echo "---- checkpoints written by this run ----"
complete_ckpts "$LOG" | while read -r n; do
  printf 'traj %s: ckpoint_lat %s B, ckpoint_rng %s B\n' "$n" "$(stat -L -c %s "$CKPT_DIR/ckpoint_lat.$n" 2>/dev/null || echo 0)" "$(stat -L -c %s "$CKPT_DIR/ckpoint_rng.$n" 2>/dev/null || echo 0)"
done
if compgen -G "$RUN_DIR/gpu_mem.*.csv" > /dev/null; then
  echo "---- nvidia-smi high-water, MiB ----"
  awk -F', *' '$4 ~ /^[0-9]+$/ { f = FILENAME; sub(/.*\//, "", f); k = f " gpu" $2
                if (!(k in m) || $4 + 0 > m[k]) m[k] = $4 + 0; t[k] = $5 }
               END { for (k in m) printf "%s peak_used=%d total=%s\n", k, m[k], t[k] }' \
    "$RUN_DIR"/gpu_mem.*.csv | sort | tee "$RUN_DIR/gpu_mem_peak.txt"
fi
echo "---- failures (assert / OOM / abort) ----"
grep -a -i -E 'assert|abort|out of memory|cudaMalloc failed|oom[-_ ]?kill|segmentation fault|bus error|floating point exception|FATAL|Grid : Error|srun: error' "$LOG" | head -60
echo "---- next segment ----"
echo "BIN=$BIN RUN=<next name> RESUME_FROM=$RUN_DIR bash $SELF   (inside a job of the same shape)"
exit "$rc"
