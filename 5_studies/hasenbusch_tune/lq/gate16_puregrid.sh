#!/bin/bash
# =============================================================================
# gate16_puregrid.sh -- one 16^3x48 case of the pure-Grid compact-Schur driver
# (src/gen_qcd_hasenbusch_tune_compact_schur.cc) on one lq2 node, 4 ranks on the job's 4 A100
# (one per rank) or 2 (two ranks per GPU, the QOS test cap), or with LQ_TARGET=cpu on one lq3
# node (4 ranks x 16 cores), run INSIDE a Slurm job with OpenMPI mpirun. Never submits; the
# job scripts are gate16_m10_gpu_job.sh and gate16_m10_cpu_job.sh.
#   CASE=traj    one HMC trajectory   = the recipe of ../perlmutter/smoke16_puregrid.sh
#   CASE=forces  one FORCES_ONLY pass = the recipe of ../perlmutter/forces16_puregrid.sh
#
# Purpose: the 16^3 gate of an lq release against the Perlmutter reference
# (__docs/2026_11_18_handoff_lq_port.md section 7c). Physics, seed, tolerances, Grid flags
# and decomposition are the Perlmutter recipe's, value for value; the only lq changes are the
# launcher (mpirun, not srun), the GPU/NUMA wrapper and, with LQ_MPI_STACK=hpcx, the MPI library.
#
# Log format. Line 1 of hmc.log is the ENV line in the Perlmutter format: tag SMOKE16_PUREGRID
# (traj) or FORCES16_PUREGRID (forces), the same fields in the same order, so the two logs
# diff field by field. MPICH_IPC and MPICH_RDMA read "na" (Cray MPICH only). Then GRID_HASH,
# BIN_PATH, one NUMA line per rank and CMD, as on Perlmutter, and one LAUNCH line with the lq
# facts (host, OpenMPI, modules, GPUs, NUMA table, OMPI_MCA_/UCX_ variables, driver).
#
# Usage (inside a job; one line):
#   CASE=traj|forces BIN=<binary> RUN=<name> bash gate16_puregrid.sh
# Caller options, read before the scrub (everything else is fixed):
#   MPI_GEOM            default 1.1.1.4; the Grid-MG route needs 1.1.2.2 (at 1.1.1.4 the
#                       level-1 coarse T extent is 3, which the fp32 SIMD layout cannot split)
#   HMC_SEED_OFFSET     default 200 (the reference seed)
#   HASEN_GRID_*, GRID_MG_*   routes and tunables, passed through unscrubbed and recorded in
#                       the ENV line's GRID_MG_ENV, as on Perlmutter
#   CASE_MIN            timeout of the run in minutes, default 10
#   LQ_MPI_STACK        gompi (default) or hpcx, read by machines/lq.sh. gompi's OpenMPI puts
#                       every rank in a node communicator of size 1 (Grid's GPU-to-GPU halo
#                       path off); HPC-X groups the node's ranks. The launcher checks that the
#                       binary's libmpi and mpirun come from the chosen stack.
#   NRANKS              MPI ranks, default 4; must equal the product of MPI_GEOM (layout tests:
#                       NRANKS=2 MPI_GEOM=1.1.1.2 puts one rank on each of QOS test's 2 GPUs)
#   GPU_MPS             1 = start NVIDIA's MPS daemon for this case (pipe and log dirs in node
#                       /tmp, log copied to the rundir), so ranks sharing a GPU run concurrently
#                       instead of in time slices; default 0
#   LQ_TARGET           gpu (default, lq2) or cpu (lq3, read by machines/lq.sh too). cpu: no GPU
#                       table and no select_gpu wrapper; mpirun binds each rank to
#                       SLURM_CPUS_PER_TASK cores (ppr:N:node:PE=T --bind-to core) with
#                       OMP_NUM_THREADS=T; Grid flags --threads T --shm 2048 (the GPU-only
#                       --accelerator-threads, --shm-mpi, --comms-overlap dropped; Grid-CPU
#                       measured --comms-overlap 5-15% slower on one lq3 node). Same ENV line;
#                       its GRID_FLAGS field shows the CPU flags.
#
# Checks before anything runs: ldd (0 unresolved, no libquda) with machines/lq.sh loaded; the
# binary's sha256 equals its release's PROVENANCE.txt exe line (when the binary sits in
# <release>/bin/); the GPU count divides 4; every rank binds its GPU and that GPU's NUMA domain (a
# probe step through the same wrapper; a failure stops the case before the driver starts).
#
# Output: runs/<YYYY_M_D>_<RUN>/ with hmc.log, select_gpu, numa_binding.txt (unpadded date,
# the runs/ convention). Refuses to overwrite.
# =============================================================================
set -o pipefail

ROOT=/lustre2/nplqcd/vayyar/grid_qcd
WF=$ROOT/grid-lqcd-workflow

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
USAGE="usage: CASE=traj|forces BIN=<binary> RUN=<name> bash $0 (inside a Slurm job)"

# ---- 1. Inputs ----------------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || die "not inside a Slurm job (SLURM_JOB_ID unset). $USAGE"
case "${CASE:-}" in traj|forces) ;; *) die "CASE must be traj or forces, got '${CASE:-}'. $USAGE" ;; esac
[ -n "${BIN:-}" ] || die "BIN is not set. $USAGE"
[ -n "${RUN:-}" ] || die "RUN is not set. $USAGE"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN may contain only letters, digits, '.', '_' and '-', got '$RUN'"
{ [ -f "$BIN" ] && [ -x "$BIN" ]; } || die "BIN is not an executable file: $BIN"
BIN=$(readlink -f "$BIN")
G_MODE=$CASE
G_MPI=${MPI_GEOM:-1.1.1.4}
G_SEED=${HMC_SEED_OFFSET:-200}
G_MIN=${CASE_MIN:-10}
G_NRANKS=${NRANKS:-4}
G_MPS=${GPU_MPS:-0}
G_TARGET=${LQ_TARGET:-gpu}
case "$G_TARGET" in gpu|cpu) ;; *) die "LQ_TARGET must be gpu or cpu, got '$G_TARGET'" ;; esac
{ [ "$G_TARGET" = gpu ] || [ "$G_MPS" = 0 ]; } || die "GPU_MPS=1 needs LQ_TARGET=gpu"
export LQ_TARGET=$G_TARGET
[[ "$G_MPI" =~ ^[1-9]\.[1-9]\.[1-9]\.[1-9]$ ]] || die "MPI_GEOM must look like 1.1.1.4, got '$G_MPI'"
[[ "$G_SEED" =~ ^[0-9]+$ ]] || die "HMC_SEED_OFFSET must be a non-negative integer, got '$G_SEED'"
[[ "$G_MIN" =~ ^[1-9][0-9]*$ ]] || die "CASE_MIN must be a positive integer, got '$G_MIN'"
[[ "$G_NRANKS" =~ ^[1-9][0-9]*$ ]] || die "NRANKS must be a positive integer, got '$G_NRANKS'"
[[ "$G_MPS" =~ ^[01]$ ]] || die "GPU_MPS must be 0 or 1, got '$G_MPS'"
IFS=. read -r M1 M2 M3 M4 <<< "$G_MPI"
[ $((M1 * M2 * M3 * M4)) -eq "$G_NRANKS" ] || die "MPI_GEOM=$G_MPI needs $((M1 * M2 * M3 * M4)) ranks, NRANKS=$G_NRANKS"
[ "$G_NRANKS" -le "${SLURM_NTASKS:-$G_NRANKS}" ] || die "NRANKS=$G_NRANKS exceeds the job's $SLURM_NTASKS tasks"
NTASKS=$G_NRANKS

# ---- 2. Environment, loader and release checks ------------------------------------------
# shellcheck source=/dev/null
source "$WF/machines/lq.sh" || die "sourcing $WF/machines/lq.sh failed"
set -u
LDD_OUT=$(ldd "$BIN" 2>&1) || die "ldd failed on $BIN: $LDD_OUT"
N_NOTFOUND=$(printf '%s\n' "$LDD_OUT" | grep -c 'not found' || true)
N_QUDA=$(printf '%s\n' "$LDD_OUT" | grep -ci 'libquda' || true)
if [ "$N_NOTFOUND" -ne 0 ]; then
  printf '%s\n' "$LDD_OUT" | grep 'not found' >&2
  die "ldd reports $N_NOTFOUND unresolved librar(y/ies) for $BIN on $(hostname)"
fi
[ "$N_QUDA" -eq 0 ] || die "ldd shows libquda in $BIN; this launcher is for pure-Grid builds only"
LIBMPI=$(printf '%s\n' "$LDD_OUT" | grep -o -m1 '/[^ ]*libmpi\.so[^ ]*' || true)
MPIRUN_BIN=$(command -v mpirun || true)
case $G_TARGET/$LQ_MPI_STACK in
  gpu/gompi)       STACK_DIR=/srv/software/el8/x86_64/eb/OpenMPI/ ;;
  gpu/hpcx)        STACK_DIR=${HPCX_DIR:?lq.sh set LQ_MPI_STACK=hpcx but HPCX_DIR is unset}/ ;;
  cpu/lq3-ompi418) STACK_DIR=${LQ_CPU_MPI_ROOT:?lq.sh set LQ_TARGET=cpu but LQ_CPU_MPI_ROOT is unset}/ ;;
  *)               die "LQ_TARGET=$G_TARGET with LQ_MPI_STACK=$LQ_MPI_STACK is not a known stack" ;;
esac
[[ "$LIBMPI" == "$STACK_DIR"* ]] || die "LQ_MPI_STACK=$LQ_MPI_STACK but $BIN loads libmpi from '$LIBMPI'"
[[ "$MPIRUN_BIN" == "$STACK_DIR"* ]] || die "LQ_MPI_STACK=$LQ_MPI_STACK but mpirun is '$MPIRUN_BIN'"
BIN_SHA256=$(sha256sum "$BIN" | cut -d' ' -f1)
PROV=$(dirname "$(dirname "$BIN")")/PROVENANCE.txt
if [ -f "$PROV" ]; then
  WANT=$(grep -m1 '^exe:' "$PROV" | grep -o 'sha256=[0-9a-f]*' | cut -d= -f2)
  [ "$WANT" = "$BIN_SHA256" ] || die "sha256 of $BIN is $BIN_SHA256, its release record $PROV says '${WANT}'"
fi
if [ "$G_TARGET" = gpu ]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: not a GPU node ($(hostname))"
fi

# ---- 3. Clear every knob this run does not set (the Perlmutter families) ------------------
# QUDA_*, HASEN_* (except HASEN_GRID_*), HMC_MG_*, USE_HMC_MG, FORCES_*, CKPT_*, TXQCD_*,
# WCF_*, LAMBDA_MN2, INTEGRATOR_VERBOSE_MEM, NO_METROP, IMPORT_CFG; smoke16_puregrid.sh
# section 3 explains each. HASEN_GRID_* and GRID_MG_* pass through.
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|NO_METROP$|IMPORT_CFG$)' | grep -v -E '^(HASEN_GRID_|GRID_MG_)')
unset N_TRAJ
N_QUDA_ENV=$(compgen -e | grep -c '^QUDA_' || true)
[ "$N_QUDA_ENV" -eq 0 ] || die "QUDA_* variables still set after clearing"
# Production speed-ups ON unless the caller sets VAR=0 (smoke16 section 3, 2026-10-02).
for v in HASEN_GRID_FUSED_CLOVER_FORCE HASEN_GRID_DEVICE_CB HASEN_GRID_BATCH_SMEAR HASEN_GRID_IMPORT_SKIP HASEN_GRID_SHARE_FIELDSTRENGTH HASEN_GRID_GPU_CLOVER_INV HASEN_GRID_CLOVER_STENCIL; do
  if [ -z "${!v:-}" ]; then
    if [ "$(grep -c -a -F "$v" "$BIN" || true)" -gt 0 ]; then
      export "$v=1"
    else
      echo "WARNING: $BIN predates $v: running without it" >&2
    fi
  fi
done

# ---- 4. Physics: the 16^3x48 cl3 recipe (smoke16_puregrid.sh section 4) -------------------
export LATT=16.16.16.48
export IMPORT_CFG=$ROOT/inputs/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime   # sha256 62c5c61e...
[ -f "$IMPORT_CFG" ] || die "starting configuration not found: $IMPORT_CFG"
export BETA=6.1
export CSW=1.24930970916466
export U0=0.832605301399891          # the driver's local default 1.0 is wrong (u0-48cube)
export MASS_LIGHT=-0.245
export MASS_STRANGE=-0.2050
export STOUT_RHO=0.125 STOUT_NSMEAR=1
export RAT_LO=1e-4 RAT_HI=100 RAT_DEGREE=20

# ---- 5. Ladder and integrator (legacy knobs, no layout override) ---------------------------
export HASEN_LADDER=-0.245,-0.20,-0.10
export INTEGRATOR=ForceGradient
export HASEN_TAIL_LEVEL=outer
export GAUGE_INNER_MULT=2
export TRAJL=0.35355339059327379
export MDSTEPS=4
[ "$G_MODE" = forces ] || export N_TRAJ=1

# ---- 6. Seed and tolerances ---------------------------------------------------------------
export HMC_SEED_OFFSET=$G_SEED
export TUNE_CG_TOL_DERIV=1e-11
export TUNE_CG_TOL_ACTION=1e-12
export TUNE_CG_TOL_STRANGE=1e-9

# ---- 7. FORCES_ONLY mode (forces16_puregrid.sh section 7, plain pass) ------------------------
if [ "$G_MODE" = forces ]; then
  export FORCES_ONLY=1
  export FORCES_SAMPLES=1
fi

# ---- 8. Threads and Grid flags --------------------------------------------------------------
if [ "$G_TARGET" = gpu ]; then
  export OMP_NUM_THREADS=8
  GRID_ARGS=(--grid "$LATT" --mpi "$G_MPI" --accelerator-threads 8 --shm 2048 --shm-mpi 0 --comms-overlap)
else
  CPU_T=${SLURM_CPUS_PER_TASK:-16}
  [[ "$CPU_T" =~ ^[1-9][0-9]*$ ]] || die "SLURM_CPUS_PER_TASK must be a positive integer, got '$CPU_T'"
  export OMP_NUM_THREADS=$CPU_T OMP_PLACES=cores OMP_PROC_BIND=close
  GRID_ARGS=(--grid "$LATT" --mpi "$G_MPI" --threads "$CPU_T" --shm 2048)
fi

# ---- 9. Run directory (refuse to overwrite) ---------------------------------------------------
DATE_TAG=$(date +%Y_%-m_%-d)
RUN_DIR=$ROOT/runs/${DATE_TAG}_${RUN}
[ -e "$RUN_DIR" ] && die "run directory already exists, refusing to overwrite: $RUN_DIR"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
LOG=$RUN_DIR/hmc.log

# ---- 10. Binding: GPU and NUMA table with the select_gpu wrapper (gpu), cores (cpu) -----------
# gpu: the GPUs Slurm gave the job; each rank bound to its GPU's NUMA domain as read from sysfs
# (lq2: 2 sockets x 32 cores; the GPU-socket wiring is read, not assumed).
# CUDA_DEVICE_ORDER=PCI_BUS_ID makes the CUDA index equal the nvidia-smi index used here.
HAVE_NUMACTL=yes
command -v numactl >/dev/null 2>&1 || HAVE_NUMACTL=no
if [ "$G_TARGET" = gpu ]; then
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
fi
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU=${#GPUS[@]}
{ [ "$NGPU" -ge 1 ] && [ $((NTASKS % NGPU)) -eq 0 ]; } || die "the GPU count must divide $NTASKS, CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
GNUMA=()
BUSES=()
for g in "${GPUS[@]}"; do
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$g" | tr 'A-F' 'a-f')
  sys=/sys/bus/pci/devices/${bus#0000}/numa_node
  n=-1
  [ -r "$sys" ] && n=$(cat "$sys")
  [ "$HAVE_NUMACTL" = yes ] || n=-1
  BUSES+=("$bus")
  GNUMA+=("$n")
done
# Rank L -> GPU number L*NGPU/NTASKS: one rank per GPU with 4 GPUs; with fewer (QOS test caps a
# job at 2 GPUs), neighbouring ranks share a GPU. The decomposition (--mpi) is unchanged, so the
# arithmetic is too; only the halo path between ranks on one GPU differs.
# Under MPS the server renumbers the devices it was started with from 0, so a client names its
# GPU by position in the job's list, not by the Slurm index.
RANKS=()
NUMAS=()
for ((L = 0; L < NTASKS; L++)); do
  i=$((L * NGPU / NTASKS))
  if [ "$G_MPS" = 1 ]; then RANKS+=("$i"); else RANKS+=("${GPUS[$i]}"); fi
  NUMAS+=("${GNUMA[$i]}")
done
SELECT_GPU=$RUN_DIR/select_gpu
printf '%s\n' \
  '#!/bin/bash' \
  '# OpenMPI local rank -> GPU + the NUMA domain of that GPU (written by gate16_puregrid.sh)' \
  "RANK_GPU=(${RANKS[*]})" \
  "RANK_NUMA=(${NUMAS[*]})" \
  'L=${OMPI_COMM_WORLD_LOCAL_RANK:?select_gpu: not started by OpenMPI mpirun}' \
  'export CUDA_DEVICE_ORDER=PCI_BUS_ID' \
  'export CUDA_VISIBLE_DEVICES=${RANK_GPU[$L]}' \
  'N=${RANK_NUMA[$L]}' \
  '[ "$N" -lt 0 ] || exec numactl -m "$N" -N "$N" "$@"' \
  'exec "$@"' > "$SELECT_GPU"
chmod +x "$SELECT_GPU"
WRAP=("$SELECT_GPU")
MPIRUN=(mpirun -np "$NTASKS" --map-by "ppr:${NTASKS}:node" --bind-to none --wdir "$RUN_DIR")
PROBE='echo "gpu=$CUDA_VISIBLE_DEVICES $(numactl --show 2>/dev/null | grep -E "^(nodebind|membind)" | tr "\n" " ")"'
else
# cpu: mpirun places the ranks in order over the node and binds each to CPU_T cores (on lq3,
# ranks 0-1 on socket 0 and 2-3 on socket 1 with 4 x 16 cores); memory is first-touch local.
WRAP=()
MPIRUN=(mpirun -np "$NTASKS" --map-by "ppr:${NTASKS}:node:PE=${CPU_T}" --bind-to core --wdir "$RUN_DIR")
PROBE='echo "cpus=$(taskset -cp $$ | cut -d: -f2 | tr -d " ") $(numactl --show 2>/dev/null | grep -E "^(nodebind|membind)" | tr "\n" " ")"'
fi

# Gate: every rank reports its GPU or cores and its NUMA binding; a failure stops here.
if ! timeout -k 10s 3m "${MPIRUN[@]}" --tag-output "${WRAP[@]}" bash -c "$PROBE" > "$RUN_DIR/numa_binding.raw" 2>&1; then
  cat "$RUN_DIR/numa_binding.raw" >&2
  die "binding probe failed: mpirun could not start every rank. Nothing was run (rundir $RUN_DIR)."
fi
sed -n -E 's/^\[[0-9]+,([0-9]+)\]<stdout>:/\1: /p' "$RUN_DIR/numa_binding.raw" | sort -n > "$RUN_DIR/numa_binding.txt"
[ "$(wc -l < "$RUN_DIR/numa_binding.txt")" -eq "$NTASKS" ] || die "binding probe returned $(wc -l < "$RUN_DIR/numa_binding.txt") of $NTASKS ranks (see $RUN_DIR/numa_binding.raw)"

# Optional MPS daemon (GPU_MPS=1), started on the job's GPUs and stopped when this script exits.
# Its pipe directory must be on a local filesystem (named pipes), hence node /tmp.
COMPUTE_MODE=na
if [ "$G_TARGET" = gpu ]; then
  COMPUTE_MODE=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader -i "${GPUS[0]}" 2>/dev/null || echo unknown)
fi
MPS_STATE=off
stop_mps() {
  echo quit | nvidia-cuda-mps-control
  sleep 2
  cp -r "$CUDA_MPS_LOG_DIRECTORY" "$RUN_DIR/mps_log" 2>/dev/null || true
}
if [ "$G_MPS" = 1 ]; then
  command -v nvidia-cuda-mps-control >/dev/null 2>&1 || die "GPU_MPS=1 but nvidia-cuda-mps-control is not on PATH"
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_${SLURM_JOB_ID}_${RUN}
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${SLURM_JOB_ID}_${RUN}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" || die "cannot create the MPS directories in /tmp"
  CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-cuda-mps-control -d || die "the MPS control daemon did not start (compute mode $COMPUTE_MODE)"
  trap stop_mps EXIT
  MPS_STATE="on(pipe=$CUDA_MPS_PIPE_DIRECTORY)"
fi

# ---- 11. Provenance: ENV line = line 1 of the log (Perlmutter format) -------------------------
GRID_HASH_LINE=$(strings -a "$BIN" | grep -m1 -E '^[0-9a-f]{40}: ' || true)
GRID_SHA=${GRID_HASH_LINE:0:40}
GRID_SHA=${GRID_SHA:-unknown}
case "$GRID_HASH_LINE" in *uncommit*) GRID_DIRTY=yes ;; *) GRID_DIRTY=no ;; esac
MG_ENV=$(compgen -e | grep -E '^(GRID_MG_|HASEN_GRID_|STRANGE_EVEN$)' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)
if [ "$G_MODE" = traj ]; then
  printf 'ENV SMOKE16_PUREGRID RUN=%s JOBID=%s BIN=%s BIN_SHA256=%s GRID_SHA=%s GRID_DIRTY=%s LATT=%s MPI=%s NTASKS=%s IMPORT_CFG=%s SEED=%s N_TRAJ=%s MDSTEPS=%s TRAJL=%s INTEGRATOR=%s GAUGE_INNER_MULT=%s LADDER=%s TAIL_LEVEL=%s MASS_LIGHT=%s MASS_STRANGE=%s CSW=%s BETA=%s U0=%s STOUT_RHO=%s STOUT_NSMEAR=%s RAT=%s/%s/%s TOL_DRV=%s TOL_ACT=%s TOL_STRANGE=%s METROP=on QUDA_ENV=%s MPICH_IPC=%s MPICH_RDMA=%s GRID_MG_RUNGS=%s GRID_MG_HB=%s GRID_MIXED=%s GRID_MIXED_HB=%s GRID_MG_ENV="%s" GRID_FLAGS="%s" STRANGE_LEVEL=%s STRANGE_INNER_MULT=%s STRANGE_RUNGS=%s LEVEL_LIGHT=%s LEVEL_STRANGE=%s LEVEL_TAIL=%s LEVEL_GAUGE=%s LEVEL_MULT=%s\n' \
    "$RUN" "$SLURM_JOB_ID" "$(basename "$BIN")" "$BIN_SHA256" "$GRID_SHA" "$GRID_DIRTY" \
    "$LATT" "$G_MPI" "$NTASKS" "$(basename "$IMPORT_CFG")" "$HMC_SEED_OFFSET" "$N_TRAJ" \
    "$MDSTEPS" "$TRAJL" "$INTEGRATOR" "$GAUGE_INNER_MULT" "$HASEN_LADDER" "$HASEN_TAIL_LEVEL" \
    "$MASS_LIGHT" "$MASS_STRANGE" "$CSW" "$BETA" "$U0" "$STOUT_RHO" "$STOUT_NSMEAR" \
    "$RAT_LO" "$RAT_HI" "$RAT_DEGREE" "$TUNE_CG_TOL_DERIV" "$TUNE_CG_TOL_ACTION" \
    "$TUNE_CG_TOL_STRANGE" "$N_QUDA_ENV" na na \
    "${HASEN_GRID_MG_RUNGS:-none}" "${HASEN_GRID_MG_HEATBATH_RUNGS:-none}" \
    "${HASEN_GRID_MIXED_CG_RUNGS:-none}" "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}" \
    "$MG_ENV" "${GRID_ARGS[*]}" default default none none none none none none > "$LOG"
else
  printf 'ENV FORCES16_PUREGRID RUN=%s JOBID=%s BIN=%s BIN_SHA256=%s GRID_SHA=%s GRID_DIRTY=%s MODE=FORCES_ONLY FORCES_SAMPLES=%s FORCES_STIFFNESS=%s STIFF=%s LATT=%s MPI=%s NTASKS=%s IMPORT_CFG=%s SEED=%s MDSTEPS=%s TRAJL=%s INTEGRATOR=%s GAUGE_INNER_MULT=%s LADDER=%s TAIL_LEVEL=%s MASS_LIGHT=%s MASS_STRANGE=%s CSW=%s BETA=%s U0=%s STOUT_RHO=%s STOUT_NSMEAR=%s RAT=%s/%s/%s TOL_DRV=%s TOL_ACT=%s TOL_STRANGE=%s QUDA_ENV=%s MPICH_IPC=%s MPICH_RDMA=%s GRID_MG_RUNGS=%s GRID_MG_HB=%s GRID_MIXED=%s GRID_MIXED_HB=%s GRID_MG_ENV="%s" GRID_FLAGS="%s"\n' \
    "$RUN" "$SLURM_JOB_ID" "$(basename "$BIN")" "$BIN_SHA256" "$GRID_SHA" "$GRID_DIRTY" \
    "$FORCES_SAMPLES" 0 none \
    "$LATT" "$G_MPI" "$NTASKS" "$(basename "$IMPORT_CFG")" "$HMC_SEED_OFFSET" \
    "$MDSTEPS" "$TRAJL" "$INTEGRATOR" "$GAUGE_INNER_MULT" "$HASEN_LADDER" "$HASEN_TAIL_LEVEL" \
    "$MASS_LIGHT" "$MASS_STRANGE" "$CSW" "$BETA" "$U0" "$STOUT_RHO" "$STOUT_NSMEAR" \
    "$RAT_LO" "$RAT_HI" "$RAT_DEGREE" "$TUNE_CG_TOL_DERIV" "$TUNE_CG_TOL_ACTION" \
    "$TUNE_CG_TOL_STRANGE" "$N_QUDA_ENV" na na \
    "${HASEN_GRID_MG_RUNGS:-none}" "${HASEN_GRID_MG_HEATBATH_RUNGS:-none}" \
    "${HASEN_GRID_MIXED_CG_RUNGS:-none}" "${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}" \
    "$MG_ENV" "${GRID_ARGS[*]}" > "$LOG"
fi
{
  printf 'GRID_HASH %s\n' "${GRID_HASH_LINE:-unknown}"
  printf 'BIN_PATH %s\n' "$BIN"
  sed 's/^/NUMA /' "$RUN_DIR/numa_binding.txt"
  printf 'CMD '
  printf '%q ' timeout -k 60s "${G_MIN}m" "${MPIRUN[@]}" "${WRAP[@]}" "$BIN" "${GRID_ARGS[@]}"
  printf '\n'
  if [ "$G_TARGET" = gpu ]; then
  printf 'LAUNCH host=%s target=gpu mpi_stack=%s libmpi=%s mps=%s compute_mode="%s" ompi="%s" modules="%s" gpus=%s buses=%s gpu_numa=%s rank_gpu=%s rank_numa=%s numactl=%s gpu="%s" mca="%s" ucx="%s"\n' \
    "$(hostname)" "$LQ_MPI_STACK" "$LIBMPI" "$MPS_STATE" "$COMPUTE_MODE" "$(mpirun --version 2>&1 | head -1)" "$(module -t list 2>&1 | tr '\n' ' ')" \
    "$(IFS=,; echo "${GPUS[*]}")" "$(IFS=,; echo "${BUSES[*]}")" "$(IFS=,; echo "${GNUMA[*]}")" \
    "$(IFS=,; echo "${RANKS[*]}")" "$(IFS=,; echo "${NUMAS[*]}")" "$HAVE_NUMACTL" \
    "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader -i "${GPUS[0]}")" \
    "$(compgen -e | grep '^OMPI_MCA_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
    "$(compgen -e | grep '^UCX_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)"
  else
  printf 'LAUNCH host=%s target=cpu mpi_stack=%s libmpi=%s ompi="%s" cxx="%s" cpu="%s" cores=%s numa_nodes=%s threads_per_rank=%s omp="%s" numactl=%s mca="%s" ucx="%s"\n' \
    "$(hostname)" "$LQ_MPI_STACK" "$LIBMPI" "$(mpirun --version 2>&1 | head -1)" "$(mpicxx --version 2>&1 | head -1)" \
    "$(lscpu | sed -n 's/^Model name: *//p' | head -1)" "$(nproc)" "$(lscpu | sed -n 's/^NUMA node(s): *//p')" "$CPU_T" \
    "OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND" "$HAVE_NUMACTL" \
    "$(compgen -e | grep '^OMPI_MCA_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
    "$(compgen -e | grep '^UCX_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)"
  fi
} >> "$LOG"

# ---- 12. Launch -------------------------------------------------------------------------------
echo "running $(basename "$BIN") CASE=$G_MODE MPI=$G_MPI -> $LOG"
timeout -k 60s "${G_MIN}m" "${MPIRUN[@]}" "${WRAP[@]}" "$BIN" "${GRID_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 13. Summary ------------------------------------------------------------------------------
echo "==== $RUN: exit=$rc  log: $LOG"
grep -a -E '^(ENV|NUMA|LAUNCH) ' "$LOG"
grep -a -E 'communicator of size|Current Grid git commit hash|Initial plaquette|CG tol:|Hasenbusch chain|\[Ladder\]|\[Levels\]|Integrator: ' "$LOG"
if [ "$G_MODE" = traj ]; then
  grep -a -E '\[0\]\[[0-9]+\] Force average' "$LOG" | head -6
  grep -a -E 'Total H before trajectory|Total H after trajectory|exp\(-dH\)|Metropolis_test|Total time for trajectory|Plaquette: \[|FORCES traj=' "$LOG"
else
  grep -a -E 'FORCES_ONLY (level|samples=)|\[FORCES_ONLY\]' "$LOG"
fi
echo "log: $LOG"
exit "$rc"
