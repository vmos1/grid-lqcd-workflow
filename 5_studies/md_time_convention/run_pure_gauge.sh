#!/bin/bash
# Pure-gauge HMC step-size test, Grid vs Chroma (__docs/2026_10_05_grid_chroma_md_time_handoff.md §6).
#
# 8^4, single-level leapfrog, Metropolis on, two actions: Wilson beta 6.0 and the production
# Luscher-Weisz form (beta 6.3, u0 0.84570646270714). Each code thermalizes its own configuration
# from cold (NTHERM updates, the first NWARM without Metropolis), then runs NTRAJ trajectories per
# setting from it:
#   Chroma  tau0 1.0,    n_steps 4 6 8 12
#   Grid    TRAJL 0.7071 (the sqrt 2 map), MDSTEPS 4 6 8 12
#   Grid    TRAJL 1.0, 0.5, 0.25 (same nominal length; a factor 2; the factor-4 map), MDSTEPS 6 12
# The sqrt 2 map predicts Grid (0.7071, n) reproduces Chroma (1.0, n) in <dH>, <dH^2>, acceptance;
# the factor-4 map predicts Grid (0.25, n) does. Leapfrog <dH> ~ h^4, so the two Grid series differ
# by (2 sqrt 2)^4 ~ 64. <plaq> must agree between codes per action (gauge-action normalisation),
# <exp(-dH)> = 1 in each (correctness).
#
# One node, four lanes: each task is a 1-rank srun step (--overlap) pinned to one GPU with
# CUDA_VISIBLE_DEVICES; nothing binds NUMA, so the 1-task binding trap does not apply.
#
# Env: PHASE smoke | therm | scan | all (default all = therm then scan), OUT, GRID_BIN, NTRAJ (100),
#      NTHERM (200), NWARM (50), TASK_MIN (per-task timeout, default 40), THERM_NS (10),
#      CHROMA_NS ("4 6 8 12"), SERIES_NS ("6 12"), SERIES_TRAJLS ("1.0 0.5 0.25").
# The first run (runs/2026_10_5_pure_gauge, defaults) sat at dH 3-250, acceptance 0-30%: chains
# barely moved, so dH was measured on one configuration per code. The clean run uses
# THERM_NS=32 CHROMA_NS="16 24 32 48" SERIES_NS="24 48" (runs/2026_10_5_pure_gauge_fine).
# Usage: SLURM_JOB_ID=<id> bash run_pure_gauge.sh > <log> 2>&1

set -uo pipefail
: "${SLURM_JOB_ID:?attach to an allocation: SLURM_JOB_ID=<id> bash run_pure_gauge.sh}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORKFLOW=$(cd "$HERE/../.." && pwd)
BASE=$(cd "$WORKFLOW/.." && pwd)

OUT=${OUT:-$BASE/runs/2026_10_5_pure_gauge}
GRID_BIN=${GRID_BIN:-${PSCRATCH}/grid_pure_hmc/md_time/bin/pure_gauge_hmc_grid}
PHASE=${PHASE:-all}
NTRAJ=${NTRAJ:-100}
NTHERM=${NTHERM:-200}
NWARM=${NWARM:-50}
TASK_MIN=${TASK_MIN:-40}
THERM_NS=${THERM_NS:-10}                    # MD steps of the thermalization runs
CHROMA_NS=${CHROMA_NS:-"4 6 8 12"}          # Chroma tau0 1.0 and Grid TRAJL 0.7071 step counts
SERIES_NS=${SERIES_NS:-"6 12"}              # step counts of the Grid TRAJL series
SERIES_TRAJLS=${SERIES_TRAJLS:-"1.0 0.5 0.25"}
TEMPLATE=$HERE/pure_gauge_chroma.ini.xml.in
mkdir -p "$OUT"

SRUN=(srun --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 --cpus-per-task=32 --gpus-per-node=4)

act_params() {
  case $1 in
    wilson) GACT=WILSON_GAUGEACT;  BETA=6.0; U0=1.0 ;;
    lw)     GACT=LW_TREE_GAUGEACT; BETA=6.3; U0=0.84570646270714 ;;
  esac
}

# grid_task LANE NAME ACT TRAJL MDSTEPS NTRAJ NOMETROP START START_TRAJ SAVE CKPT_DIR [SEED]
grid_task() {
  local lane=$1 name=$2 act=$3 trajl=$4 md=$5 ntraj=$6 nometrop=$7 start=$8 st=$9 save=${10} ck=${11}
  local seed=${12:-0}
  local d=$OUT/$name
  mkdir -p "$d"
  act_params "$act"
  (
    source "$WORKFLOW/config.sh" > /dev/null 2>&1
    export OMP_NUM_THREADS=8
    PG_ACTION=$act PG_BETA=$BETA PG_U0=$U0 PG_TRAJL=$trajl PG_MDSTEPS=$md PG_NTRAJ=$ntraj \
    PG_NOMETROP=$nometrop PG_START=$start PG_START_TRAJ=$st PG_SAVE=$save PG_CKPT_DIR=${ck:-$d} PG_SEED=$seed \
      timeout -k 30s "${TASK_MIN}m" "${SRUN[@]}" --chdir="$d" --export=ALL \
      env CUDA_VISIBLE_DEVICES="$lane" "$GRID_BIN" --grid 8.8.8.8 --mpi 1.1.1.1 \
      --accelerator-threads 8 --shm 1024 > "$d/stdout.log" 2>&1
    rc=$?
    echo "$(date '+%H:%M:%S') lane $lane $name exit=$rc"
  )
}

# chroma_task LANE NAME ACT TAU NSTEPS START NUPDATES CFG_TYPE CFG_FILE SEED
chroma_task() {
  local lane=$1 name=$2 act=$3 tau=$4 ns=$5 start=$6 nup=$7 ctype=$8 cfile=$9 seed=${10}
  local d=$OUT/$name
  mkdir -p "$d"
  act_params "$act"
  sed -e "s|@CFG_TYPE@|$ctype|" -e "s|@CFG_FILE@|$cfile|" -e "s|@SEED@|$seed|" \
      -e "s|@START@|$start|" -e "s|@NWARMUP@|$NWARM|" -e "s|@NUPDATES@|$nup|" \
      -e "s|@GACT@|$GACT|" -e "s|@BETA@|$BETA|" -e "s|@U0@|$U0|" \
      -e "s|@TAU@|$tau|" -e "s|@NSTEPS@|$ns|" "$TEMPLATE" > "$d/in.xml"
  (
    source "$BASE/chroma/env_chroma_pm.sh" > /dev/null 2>&1
    export QUDA_RESOURCE_PATH=$d/quda_resource
    mkdir -p "$QUDA_RESOURCE_PATH"
    timeout -k 30s "${TASK_MIN}m" "${SRUN[@]}" --chdir="$d" --export=ALL \
      env CUDA_VISIBLE_DEVICES="$lane" "$CHROMA_HMC" -i "$d/in.xml" -o "$d/out.xml" -l "$d/log.xml" \
      -geom 1 1 1 1 > "$d/stdout.log" 2>&1
    rc=$?
    echo "$(date '+%H:%M:%S') lane $lane $name exit=$rc"
  )
}

# Run four lane lists concurrently; each list is a sequence of task command strings.
run_lanes() {
  local i lst
  for i in 0 1 2 3; do
    eval "lst=(\"\${LANE$i[@]}\")"
    ( for t in "${lst[@]}"; do eval "$t"; done ) &
  done
  wait
}

echo "=== pure gauge $(date '+%Y-%m-%d %H:%M:%S %Z') job $SLURM_JOB_ID PHASE=$PHASE"
echo "ENV PURE_GAUGE OUT=$OUT GRID_BIN=$GRID_BIN NTRAJ=$NTRAJ NTHERM=$NTHERM NWARM=$NWARM TASK_MIN=$TASK_MIN THERM_NS=$THERM_NS CHROMA_NS='$CHROMA_NS' SERIES_NS='$SERIES_NS' SERIES_TRAJLS='$SERIES_TRAJLS' LATT=8.8.8.8 INTEGRATOR=leapfrog"
"${SRUN[@]}" hostname || { echo "ERROR: cannot attach to allocation $SLURM_JOB_ID" >&2; exit 1; }
[ -x "$GRID_BIN" ] || { echo "ERROR: Grid binary missing: $GRID_BIN" >&2; exit 1; }

if [ "$PHASE" = smoke ]; then
  LANE0=("grid_task 0 smoke_grid_lw lw 0.70711 6 3 1 cold 0 1000000 ''")
  LANE1=("chroma_task 1 smoke_chroma_lw lw 1.0 6 0 3 UNIT none 41")
  LANE2=(); LANE3=()
  run_lanes
fi

if [ "$PHASE" = therm ] || [ "$PHASE" = all ]; then
  echo "=== therm $(date '+%H:%M:%S')"
  LANE0=("grid_task 0 grid_wilson_therm wilson 0.70711 $THERM_NS $((NTHERM - NWARM)) $NWARM cold 0 $NTHERM ''")
  LANE1=("grid_task 1 grid_lw_therm lw 0.70711 $THERM_NS $((NTHERM - NWARM)) $NWARM cold 0 $NTHERM ''")
  LANE2=("chroma_task 2 chroma_wilson_therm wilson 1.0 $THERM_NS 0 $NTHERM UNIT none 11")
  LANE3=("chroma_task 3 chroma_lw_therm lw 1.0 $THERM_NS 0 $NTHERM UNIT none 12")
  run_lanes
fi

if [ "$PHASE" = scan ] || [ "$PHASE" = all ]; then
  echo "=== scan $(date '+%H:%M:%S')"
  TASKS=()
  for act in wilson lw; do
    cfg=$OUT/chroma_${act}_therm/pg_cfg_${NTHERM}.lime
    ck=$OUT/grid_${act}_therm
    [ -f "$cfg" ] || echo "WARNING: missing Chroma thermalized config $cfg"
    [ -f "$ck/ckpoint_lat.$NTHERM" ] || echo "WARNING: missing Grid checkpoint $ck/ckpoint_lat.$NTHERM"
    seed=100
    for n in $CHROMA_NS; do
      seed=$((seed + 1))
      TASKS+=("chroma_task LANE chroma_${act}_tau1.0_n$n $act 1.0 $n $NTHERM $NTRAJ SZINQIO $cfg $seed")
      TASKS+=("grid_task LANE grid_${act}_trajl0.70711_n$n $act 0.70711 $n $NTRAJ 0 ckpt $NTHERM 1000000 $ck")
    done
    for n in $SERIES_NS; do
      for tl in $SERIES_TRAJLS; do
        TASKS+=("grid_task LANE grid_${act}_trajl${tl}_n$n $act $tl $n $NTRAJ 0 ckpt $NTHERM 1000000 $ck")
      done
    done
  done
  LANE0=(); LANE1=(); LANE2=(); LANE3=()
  i=0
  for t in "${TASKS[@]}"; do
    lane=$((i % 4))
    eval "LANE$lane+=(\"\${t/LANE/$lane}\")"
    i=$((i + 1))
  done
  run_lanes
fi
if [ "$PHASE" = replica ]; then
  # Residual check (handoff §7): NREP independent chains per (code, action, n), all from ONE shared
  # configuration, Grid's thermalized ckpoint_lat.$NTHERM in REP_SRC (Chroma reads it as NERSC);
  # Grid reseeded per replica (PG_START=reseed), Chroma a different seed per replica. Errors on
  # <dH^2> then come from the replica scatter, not a bootstrap over one chain.
  REP_SRC=${REP_SRC:?set REP_SRC to the run dir holding grid_<action>_therm/ckpoint_lat.$NTHERM}
  NREP=${NREP:-6}
  REP_NS=${REP_NS:-"24 32"}
  echo "=== replica $(date '+%H:%M:%S') REP_SRC=$REP_SRC NREP=$NREP REP_NS='$REP_NS'"
  TASKS=()
  for act in wilson lw; do
    ck=$REP_SRC/grid_${act}_therm
    [ -f "$ck/ckpoint_lat.$NTHERM" ] || { echo "ERROR: missing $ck/ckpoint_lat.$NTHERM" >&2; exit 1; }
    for n in $REP_NS; do
      for r in $(seq 1 "$NREP"); do
        TASKS+=("grid_task LANE grid_${act}_trajl0.70711_n${n}_r$r $act 0.70711 $n $NTRAJ 0 reseed $NTHERM 1000000 $ck $((1000 * r + n))")
        TASKS+=("chroma_task LANE chroma_${act}_tau1.0_n${n}_r$r $act 1.0 $n $NTHERM $NTRAJ NERSC $ck/ckpoint_lat.$NTHERM $((200 + 10 * r + n))")
      done
    done
  done
  # Shared queue (Grid tasks take ~5x Chroma's; a static round-robin left two lanes idle): every
  # lane walks the whole list, skips a task whose output already holds NTRAJ trajectories, and runs
  # it only if it wins the atomic mkdir claim $OUT/.claims/<name>. A rerun therefore resumes;
  # pre-create a claim to keep a task out (e.g. one still running from an earlier launch).
  mkdir -p "$OUT/.claims"
  task_done() {
    local d=$OUT/$1
    if [[ $1 == grid_* ]]; then
      [ "$(grep -c 'Total H after' "$d/stdout.log" 2> /dev/null)" -ge "$NTRAJ" ]
    else
      [ "$(grep -c '<deltaH>' "$d/log.xml" 2> /dev/null)" -ge "$NTRAJ" ]
    fi
  }
  for lane in 0 1 2 3; do
    (
      for t in "${TASKS[@]}"; do
        name=$(echo "$t" | awk '{print $3}')
        task_done "$name" && continue
        mkdir "$OUT/.claims/$name" 2> /dev/null || continue
        eval "${t/LANE/$lane}"
      done
    ) &
  done
  wait
fi
echo "=== done $(date '+%Y-%m-%d %H:%M:%S %Z')"
