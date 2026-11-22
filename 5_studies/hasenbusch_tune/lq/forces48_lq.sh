#!/bin/bash
# =============================================================================
# forces48_lq.sh -- one FORCES_ONLY (force-evaluation-only) run of the pure-Grid compact-Schur
# driver at 48^3x96 on lq, run INSIDE a Slurm job (never submits). The lq twin of
# ../perlmutter/forces48_puregrid.sh: from cfg_2000, for each of FORCES_SAMPLES samples, every
# monomial's heatbath (refresh) and one force evaluation, each timed and printed as
#   FORCES_ONLY level <name> sample <s> avg= max= refresh=<heatbath s> time=<force s>
# Minutes instead of the hours of a trajectory: the tool for speed A/B tests (Grid flags,
# solver routes, layouts) and for force screening.
#
# Physics, baseline ladder, tolerances and speed-up switches are traj48_lq.sh's (= Perlmutter's
# traj48_puregrid.sh baseG profile). Solver routes:
#   FO_ROUTES=traj (default)   the trajectory's routes (MG rungs 0,1,2 + MG heatbath 0,1, mixed CG
#                              rung 3 + mixed CG heatbath 2,3), so the timings price a trajectory
#   FO_ROUTES=production       forces48_puregrid.sh's production profile (MG rungs 0,1,2 only,
#                              CG heatbaths), for a like-for-like with a Perlmutter forces48 run
# A caller's own HASEN_GRID_* lists win in both. GRID_MG_* pass through.
#
# Usage (inside a job; job script forces48_m10_cpu_job.sh):
#   BIN=<binary> RUN=<name> [FORCES_SAMPLES=n] [GRID_EXTRA_ARGS="--dslash-asm"] bash forces48_lq.sh
# Options:
#   FORCES_SAMPLES   default 2 (sample 0 carries the lazy MG build in PF0's first force;
#                    sample 1 is the steady state to quote)
#   HMC_SEED_OFFSET  default 0 (Perlmutter's default for forces48)
#   LQ_TARGET        gpu (default) or cpu
#   MPI_GEOM         default 1.2.2.4; must use all the job's ranks and pass the MG geometry check
#   GRID_EXTRA_ARGS  appended verbatim to the Grid flags (e.g. --dslash-asm), recorded in GRID_FLAGS
#   DIAG_START_CFG   another ILDG configuration of the ensemble (default cfg_2000)
#   TIMEOUT_MIN      default 90
#   DEVICE_MEM_MB, GPU_MON_MS  gpu only, as in traj48_lq.sh
#
# MG geometry check (before anything is launched; job 1364148 died after 12 min on 4 lq3 nodes
# with --mpi 2.2.2.4). The multigrid's levels are the lattice / GRID_MG_BLOCK (default 4.4.4.4)
# and that / GRID_MG_BLOCK2 (default 2.3.3.3): 12.12.12.24 and 6.4.4.8 here. On every level each
# rank's extent must be whole, divisible by the fp32 SIMD layout 1.2.2.2 in y, z, t, and the
# red-black half of x (the checkerboard) must split evenly over the x ranks. With the default
# blocks that allows x in {1,3}, y and z in {1,2}, t in {1,2,4}: 16, 24 or 48 ranks, not 32 or 64.
#
# Output: runs/<YYYY_M_D>_<RUN>/hmc.log, line 1 the ENV line in Perlmutter's FORCES48_PUREGRID
# format (MPICH_* read "na"), then GRID_HASH, BIN_PATH, NUMA, CMD and the lq LAUNCH line.
# =============================================================================
set -o pipefail

ROOT=/lustre2/nplqcd/vayyar/grid_qcd
WF=$ROOT/grid-lqcd-workflow

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
USAGE="usage: BIN=<binary> RUN=<name> [FORCES_SAMPLES=n] bash $0 (inside a Slurm job)"

# ---- 1. Inputs -------------------------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || die "not inside a Slurm job (SLURM_JOB_ID unset). $USAGE"
[ -n "${BIN:-}" ] || die "BIN is not set. $USAGE"
[ -n "${RUN:-}" ] || die "RUN is not set. $USAGE"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN may contain only letters, digits, '.', '_' and '-', got '$RUN'"
{ [ -f "$BIN" ] && [ -x "$BIN" ]; } || die "BIN is not an executable file: $BIN"
BIN=$(readlink -f "$BIN")
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
FO_SAMPLES=${FORCES_SAMPLES:-2}
FO_SEED=${HMC_SEED_OFFSET:-0}
FO_ROUTES=${FO_ROUTES:-traj}
G_TARGET=${LQ_TARGET:-gpu}
G_MPI=${MPI_GEOM:-1.2.2.4}
G_EXTRA=${GRID_EXTRA_ARGS:-}
TIMEOUT_MIN=${TIMEOUT_MIN:-90}
DEVICE_MEM_MB=${DEVICE_MEM_MB:-8000}
GPU_MON_MS=${GPU_MON_MS:-2000}
START_CFG=${DIAG_START_CFG:-$ROOT/inputs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime}
case "$G_TARGET" in gpu|cpu) ;; *) die "LQ_TARGET must be gpu or cpu, got '$G_TARGET'" ;; esac
export LQ_TARGET=$G_TARGET
case "$FO_ROUTES" in traj|production) ;; *) die "FO_ROUTES must be traj or production, got '$FO_ROUTES'" ;; esac
[[ "$FO_SAMPLES" =~ ^[1-9][0-9]*$ ]] || die "FORCES_SAMPLES must be a positive integer, got '$FO_SAMPLES'"
[[ "$FO_SEED" =~ ^[0-9]+$ ]] || die "HMC_SEED_OFFSET must be a non-negative integer, got '$FO_SEED'"
[[ "$G_MPI" =~ ^[1-9][0-9]*\.[1-9][0-9]*\.[1-9][0-9]*\.[1-9][0-9]*$ ]] || die "MPI_GEOM must look like 1.2.2.4, got '$G_MPI'"
[[ "$TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]] || die "TIMEOUT_MIN must be a positive integer (minutes), got '$TIMEOUT_MIN'"
[[ "$DEVICE_MEM_MB" =~ ^[1-9][0-9]*$ ]] || die "DEVICE_MEM_MB must be a positive integer (MB), got '$DEVICE_MEM_MB'"
[[ "$GPU_MON_MS" =~ ^[0-9]+$ ]] || die "GPU_MON_MS must be a non-negative integer (ms), got '$GPU_MON_MS'"
START_CFG=$(readlink -f "$START_CFG") || die "starting configuration does not resolve"
[ -f "$START_CFG" ] || die "starting configuration not found: $START_CFG"
NODES=${SLURM_JOB_NUM_NODES:?SLURM_JOB_NUM_NODES unset}
NTASKS=${SLURM_NTASKS:?SLURM_NTASKS unset (give the job --ntasks-per-node)}
PER_NODE=$(( NTASKS / NODES ))
[ $(( PER_NODE * NODES )) -eq "$NTASKS" ] || die "$NTASKS tasks do not split evenly over $NODES nodes"
IFS=. read -r -a P <<< "$G_MPI"
[ $(( P[0] * P[1] * P[2] * P[3] )) -eq "$NTASKS" ] || die "MPI_GEOM=$G_MPI needs $(( P[0] * P[1] * P[2] * P[3] )) ranks, the job has $NTASKS"
CPT=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK unset (give the job --cpus-per-task)}

# ---- 2. Environment, loader and release checks (as traj48_lq.sh section 2) ----------------------
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
grep -q -a -F FORCES_SAMPLES "$BIN" || die "$BIN has no FORCES_ONLY mode"

# ---- 3. Clear every knob this run does not set (forces48_puregrid.sh section 3) ------------------
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|STRANGE_EVEN$|NO_METROP$|IMPORT_CFG$|N_TRAJ$|HMC_SEED_OFFSET$|RAT_DEGREE_MD$|TUNE_CG_TOL_STRANGE_MD$)' | grep -v -E '^(HASEN_GRID_|GRID_MG_)')
N_QUDA_ENV=$(compgen -e | grep -c '^QUDA_' || true)
[ "$N_QUDA_ENV" -eq 0 ] || die "QUDA_* variables still set after clearing"

# ---- 4. Physics, baseline ladder, tolerances (traj48_lq.sh sections 4-6; layout inert here) -----
export LATT=48.48.48.96
export IMPORT_CFG=$START_CFG
export BETA=6.3
export U0=0.84570646270714
export CSW=1.20536588031793
export MASS_LIGHT=-0.2416
export MASS_STRANGE=-0.2050
export STOUT_RHO=0.125 STOUT_NSMEAR=1
export RAT_LO=7e-4 RAT_HI=35.0 RAT_DEGREE=20
export RAT_BOUNDS_CHECK_FREQ=10 RAT_BOUNDS_CHECK_TOL=1e-6 RAT_BOUNDS_CHECK_ABORT=1
RAT_BC_TAG=$RAT_BOUNDS_CHECK_FREQ/$RAT_BOUNDS_CHECK_TOL/$RAT_BOUNDS_CHECK_ABORT
LADDER_PROFILE=baseG
export HASEN_LADDER=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870
export HASEN_TAIL_LEVEL=inner
export MDSTEPS=12 INTEGRATOR=ForceGradient GAUGE_INNER_MULT=2 TRAJL=1.0
export HMC_SEED_OFFSET=$FO_SEED
export TUNE_CG_TOL_DERIV=1e-11 TUNE_CG_TOL_ACTION=1e-12 TUNE_CG_TOL_STRANGE=1e-9
export FORCES_ONLY=1 FORCES_SAMPLES=$FO_SAMPLES

# ---- 5. Solver routes and production speed-ups -------------------------------------------------
if [ -z "${HASEN_GRID_MG_RUNGS+x}" ]; then HASEN_GRID_MG_RUNGS=0,1,2; fi
if [ "$FO_ROUTES" = traj ]; then
  if [ -z "${HASEN_GRID_MG_HEATBATH_RUNGS+x}" ]; then HASEN_GRID_MG_HEATBATH_RUNGS=0,1; fi
  if [ -z "${HASEN_GRID_MIXED_CG_RUNGS+x}" ]; then HASEN_GRID_MIXED_CG_RUNGS=3; fi
  if [ -z "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS+x}" ]; then HASEN_GRID_MIXED_CG_HEATBATH_RUNGS=2,3; fi
fi
for v in HASEN_GRID_MG_RUNGS HASEN_GRID_MG_HEATBATH_RUNGS HASEN_GRID_MIXED_CG_RUNGS HASEN_GRID_MIXED_CG_HEATBATH_RUNGS; do
  if [ -n "${!v:-}" ]; then
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
while read -r v; do
  [ -n "$v" ] || continue
  [ "${!v}" = 0 ] || grep -q -a -F "$v" "$BIN" || die "$v is set but $BIN does not contain that name"
done < <(compgen -e | grep '^HASEN_GRID_' | grep -v -x -E 'HASEN_GRID_(MG_RUNGS|MG_HEATBATH_RUNGS|MIXED_CG_RUNGS|MIXED_CG_HEATBATH_RUNGS)')
GRID_MG_ENV=$(compgen -e | grep -E '^(GRID_MG_|HASEN_GRID_)' | grep -v -x -E 'HASEN_GRID_(MG_RUNGS|MG_HEATBATH_RUNGS|MIXED_CG_RUNGS|MIXED_CG_HEATBATH_RUNGS)' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)

# ---- 6. MG geometry check (header) --------------------------------------------------------------
if [ -n "${HASEN_GRID_MG_RUNGS:-}" ]; then
  IFS=. read -r -a L <<< "$LATT"
  IFS=. read -r -a B1 <<< "${GRID_MG_BLOCK:-4.4.4.4}"
  IFS=. read -r -a B2 <<< "${GRID_MG_BLOCK2:-2.3.3.3}"
  S=(1 2 2 2)
  for d in 0 1 2 3; do
    for lvl in 0 1 2; do
      case $lvl in
        0) g=${L[$d]} ;;
        1) [ $(( L[d] % B1[d] )) -eq 0 ] || die "MG geometry: dim $d lattice ${L[$d]} not divisible by GRID_MG_BLOCK ${B1[$d]}"
           g=$(( L[d] / B1[d] )) ;;
        2) [ $(( g % B2[d] )) -eq 0 ] || die "MG geometry: dim $d level-1 extent $g not divisible by GRID_MG_BLOCK2 ${B2[$d]}"
           g=$(( g / B2[d] )) ;;
      esac
      [ $(( g % P[d] )) -eq 0 ] || die "MG geometry: dim $d level-$lvl extent $g not divisible by $G_MPI's ${P[$d]} ranks"
      [ $(( (g / P[d]) % S[d] )) -eq 0 ] || die "MG geometry: dim $d level-$lvl local extent $(( g / P[d] )) not divisible by the fp32 SIMD layout ${S[$d]} (change MPI_GEOM)"
      if [ "$d" = 0 ]; then
        { [ $(( g % 2 )) -eq 0 ] && [ $(( (g / 2) % P[0] )) -eq 0 ]; } || die "MG geometry: x level-$lvl extent $g: its red-black half does not split over ${P[0]} x ranks (x ranks must divide $(( g / 2 )); job 1364148)"
      fi
    done
  done
fi

# ---- 7. Threads and Grid flags ------------------------------------------------------------------
read -r -a EXTRA <<< "$G_EXTRA"
if [ "$G_TARGET" = gpu ]; then
  export OMP_NUM_THREADS=8
  GRID_ARGS=(--grid "$LATT" --mpi "$G_MPI" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap --device-mem "$DEVICE_MEM_MB" "${EXTRA[@]}")
else
  export OMP_NUM_THREADS=$CPT OMP_PLACES=cores OMP_PROC_BIND=close
  GRID_ARGS=(--grid "$LATT" --mpi "$G_MPI" --threads "$CPT" --shm 2048 "${EXTRA[@]}")
fi

# ---- 8. Run directory, binding and the srun step (as traj48_lq.sh sections 10-11) ---------------
DATE_TAG=$(date +%Y_%-m_%-d)
RUN_DIR=$ROOT/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir "$RUN_DIR" || die "cannot create $RUN_DIR"
LOG=$RUN_DIR/hmc.log
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
    '# srun local rank -> GPU + the NUMA domain of that GPU (written by forces48_lq.sh)' \
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
      printf '%s\n' '#!/bin/bash' '# nvidia-smi memory sampler on local rank 0 of each node (written by forces48_lq.sh)' 'if [ "${SLURM_LOCALID:-}" = 0 ]; then'
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
if ! timeout -k 10s 5m "${SRUN[@]}" --label "${PROBE_WRAP[@]}" bash -c "$PROBE" > "$RUN_DIR/numa_binding.raw" 2>&1; then
  cat "$RUN_DIR/numa_binding.raw" >&2
  die "binding probe failed: srun could not start every rank. Nothing was run (rundir $RUN_DIR)."
fi
sed -n -E 's/^ *([0-9]+): /\1: /p' "$RUN_DIR/numa_binding.raw" | sort -n > "$RUN_DIR/numa_binding.txt"
[ "$(wc -l < "$RUN_DIR/numa_binding.txt")" -eq "$NTASKS" ] || die "binding probe returned $(wc -l < "$RUN_DIR/numa_binding.txt") of $NTASKS ranks"

# ---- 9. Provenance: ENV line in Perlmutter's FORCES48_PUREGRID format, then the lq LAUNCH line --
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
GRID_SHA=${GRID_HASH_LINE:0:40}
GRID_SHA=${GRID_SHA:-unknown}
case "$GRID_HASH_LINE" in *uncommit*) GRID_DIRTY=yes ;; *) GRID_DIRTY=no ;; esac
printf 'ENV FORCES48_PUREGRID RUN=%s JOBID=%s BIN=%s BIN_SHA256=%s GRID_SHA=%s GRID_DIRTY=%s MODE=FORCES_ONLY FORCES_SAMPLES=%s FORCES_SKIP=%s FO_PROFILE=%s LATT=%s MPI=%s NODES=%s NTASKS=%s IMPORT_CFG=%s SEED=%s LADDER=%s STRANGE_LEVEL=%s STRANGE_INNER_MULT=%s TAIL_LEVEL=%s STRANGE_RUNGS=%s LADDER_PROFILE=%s INTEGRATOR=%s MDSTEPS=%s GAUGE_INNER_MULT=%s LEVEL_LIGHT=%s LEVEL_STRANGE=%s LEVEL_TAIL=%s LEVEL_GAUGE=%s LEVEL_MULT=%s TRAJL=%s MASS_LIGHT=%s MASS_STRANGE=%s CSW=%s BETA=%s U0=%s STOUT_RHO=%s STOUT_NSMEAR=%s RAT=%s/%s/%s TOL_DRV=%s TOL_ACT=%s TOL_STRANGE=%s RAT_DEGREE_MD=%s TOL_STRANGE_MD=%s RAT_BOUNDS_CHECK=%s QUDA_ENV=%s MPICH_IPC=%s MPICH_RDMA=%s MPICH_NIC=%s OMP=%s GRID_MG_RUNGS=%s GRID_MG_HB=%s GRID_MIXED=%s GRID_MIXED_HB=%s GRID_MG_ENV="%s" DEVICE_MEM_MB=%s VERBOSE_MEM=%s GPU_MON_MS=%s GRID_FLAGS="%s" FORCES_STIFFNESS=%s STIFF=%s\n' \
  "$RUN" "$SLURM_JOB_ID" "$(basename "$BIN")" "$BIN_SHA256" "$GRID_SHA" "$GRID_DIRTY" \
  "$FORCES_SAMPLES" none "$FO_ROUTES" "$LATT" "$G_MPI" "$NODES" "$NTASKS" "$(basename "$IMPORT_CFG")" \
  "$HMC_SEED_OFFSET" "$HASEN_LADDER" default default \
  "$HASEN_TAIL_LEVEL" none "$LADDER_PROFILE" "$INTEGRATOR" "$MDSTEPS" "$GAUGE_INNER_MULT" \
  none none none none none "$TRAJL" \
  "$MASS_LIGHT" "$MASS_STRANGE" "$CSW" "$BETA" "$U0" "$STOUT_RHO" "$STOUT_NSMEAR" \
  "$RAT_LO" "$RAT_HI" "$RAT_DEGREE" "$TUNE_CG_TOL_DERIV" "$TUNE_CG_TOL_ACTION" \
  "$TUNE_CG_TOL_STRANGE" "$RAT_DEGREE" "$TUNE_CG_TOL_STRANGE" "$RAT_BC_TAG" "$N_QUDA_ENV" na na \
  na "$OMP_NUM_THREADS" "${HASEN_GRID_MG_RUNGS:-none}" \
  "${HASEN_GRID_MG_HEATBATH_RUNGS:-none}" "${HASEN_GRID_MIXED_CG_RUNGS:-none}" \
  "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}" "$GRID_MG_ENV" \
  "$([ "$G_TARGET" = gpu ] && echo "$DEVICE_MEM_MB" || echo na)" off "$([ "$G_TARGET" = gpu ] && echo "$GPU_MON_MS" || echo 0)" "${GRID_ARGS[*]}" 0 none > "$LOG"
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

# ---- 10. Launch -----------------------------------------------------------------------------------
echo "running $(basename "$BIN") FORCES_ONLY samples=$FORCES_SAMPLES routes=$FO_ROUTES on $NODES node(s) x $PER_NODE ranks -> $LOG"
timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "${WRAP[@]}" "$BIN" "${GRID_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 11. Summary ----------------------------------------------------------------------------------
echo "==== $RUN: exit=$rc  log: $LOG"
[ "$rc" -eq 124 ] && echo "exit 124 = TIMEOUT_MIN ($TIMEOUT_MIN min) reached"
grep -a -E '^(ENV|LAUNCH) ' "$LOG"
grep -a -E 'communicator of size|Initial plaquette|SciDAC checksums' "$LOG"
echo "---- FORCES_ONLY per monomial (refresh = heatbath s, time = force s; sample 0 PF0 includes the MG build) ----"
grep -a -E 'FORCES_ONLY (level|samples=)|\[FORCES_ONLY\]' "$LOG"
echo "---- multigrid builds ----"
grep -a -E '\[GridMG [A-Za-z0-9]+ hierarchy\] build' "$LOG"
echo "---- failures (assert / OOM / abort) ----"
grep -a -i -E 'assert|abort|out of memory|cudaMalloc failed|oom[-_ ]?kill|segmentation fault|bus error|FATAL|Grid : Error|srun: error' "$LOG" | grep -a -v 'UCX  WARN' | head -60
echo "log: $LOG"
exit "$rc"
