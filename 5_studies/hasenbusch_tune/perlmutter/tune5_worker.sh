#!/bin/bash
#SBATCH --account=m4599_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/runs/slurm_%x_%j.out
# =============================================================================
# tune5_worker.sh -- the tune 5 worker (campaign tune5-autonomous,
# __docs/2026_10_28_handoff_tune5_autonomous.md "How it runs"; queue and use in
# __docs/2026_10_28_tune5_worker_queue_cost_model.md). Inside ONE 4-node job it takes
# candidate files from the queue one at a time, validates them, runs the existing
# launchers UNCHANGED (traj48_puregrid.sh for trajectories, forces48_puregrid.sh for
# force-only), checks the run's ENV line (and for trajectories the driver's [Levels]
# banner) against the request, and writes one results row per candidate. It never
# allocates, never submits, never resumes a chain.
#
# Reviewed by the user once, then write-protected from the agent: the planner (a Claude
# session on the login node) writes candidate files, never this script.
#
# Usage (one line each):
#   sbatch --time=08:00:00 -J t5w tune5_worker.sh      # batch: the job is the allocation
#   SLURM_JOB_ID=<id> bash tune5_worker.sh             # inside an interactive 4-node allocation
#   bash tune5_worker.sh --check <candidate file>      # validate only, print what would run
#   bash tune5_worker.sh --verify <candidate file> <hmc.log>   # check a run's log against a candidate
# --time is mandatory for sbatch (a missing one has defaulted to 10 min, machines/perlmutter.md).
#
# Queue ($QUEUE below):
#   staging/                   the planner's tool (autotune/tune5.py enqueue) writes a candidate here,
#                              runs this script's --check on it, then writes the same text to pending/
#                              (staging/ keeps the planner's record; this script never reads it)
#   pending/<NNN>_<name>.cand  taken in file-name order once 30 s old (a file written in place may be
#                              incomplete before that), the first one whose max_min fits the time left
#   running/                   claimed by rename (atomic), so two workers can share one queue
#   done/, failed/             the candidate + <NNN>_<name>.out (the launcher's stdout + stderr)
#   rejected/                  refused by validation, + <file>.why; nothing ran
#   results/                   000_header.tsv + one one-line TSV per candidate (cat results/*.tsv)
#   STOP                       if present, the worker exits before taking the next candidate
#   A candidate left in running/ after its job ended was interrupted by Slurm: re-queue it under a
#   new name (run names are single-use), never by moving the file back.
#   worker_<jobid>.log         this worker's events
#
# Candidate file: key=value lines and nothing else (no blank lines, no comments), at most 40
# lines and 4 KB, each key at most once, values only [A-Za-z0-9.,/_-]. It is never sourced or
# evaluated: each value is matched against its pattern, then handed to the launcher as one
# environment variable. Keys:
#   name=<id>          [A-Za-z0-9][A-Za-z0-9._-]{0,39}, equal to the file name's <name>; the run is
#                      RUN=t5_<name> (runs/<date>_t5_<name>); refused if that run name was used before
#   kind=traj|fo       trajectories (traj48_puregrid.sh) or one force-only pass (forces48_puregrid.sh)
#   ladder=<masses>    light -> tail, comma separated, the first -0.2416 (MASS_LIGHT), strictly
#                      increasing, below 1. At most 6 masses for traj (7 host-OOM in a trajectory on
#                      four 40 GB nodes, A3 10-20), 7 for fo (8 OOM, fo16_A7b)
#   mg, mg_hb, mixed, mixed_hb  ratio-rung lists (0-based, e.g. 0,1,2; may be empty) for
#                      HASEN_GRID_MG_RUNGS, _MG_HEATBATH_RUNGS, _MIXED_CG_RUNGS, _MIXED_CG_HEATBATH_RUNGS.
#                      Always given, so no launcher default applies. mg_hb inside mg, mg and mixed
#                      disjoint, mg_hb and mixed_hb disjoint (the driver's rules)
#   seed=<0..99999>    HMC_SEED_OFFSET of the fresh start
#   start=djm<s>_<t>   the starting configuration, stream s trajectory t: djm3_2000 = cfg_2000 in
#                      data/; any other from $CFG_DIR (the ensemble-cfg-spread copies on scratch)
#   max_min=<minutes>  the run's budget: started only with max_min + MARGIN_MIN left in the job.
#                      traj: the launcher's TIMEOUT_MIN, 30 + 45 x n_traj .. 480 (a floor, not an
#                      estimate: the planner sets it from the cost model). fo: 10 .. 90, the
#                      planner's estimate, used only to decide whether it fits: the force-only
#                      launcher's own timeout is a fixed 90 min, so an overrun near the end of the
#                      job is ended by Slurm
#   layout=L/S/T/G     traj only: steps per trajectory of light / strange / tail / gauge, each 2..256.
#                      The distinct values, sorted, are the integrator levels (0 = outermost =
#                      MDSTEPS); each is 2 x mult x the previous one, so every ratio must be even
#   n_traj=1..3        traj only; longer chains need the user's approval and do not go through here
#   fo_samples=2..9    fo only (9 = the 8-draw method, sample 0 dropped in the analysis)
#   pred_wall_s=<s>, pred_force_s=<s>, risk=light|strange|tail|gauge|cost|none, pair=<name>
#                      optional, the planner's prediction: copied to the results row, never used
# Everything else is the launchers' own, and checked on the ENV line: binary $BIN, cfg_2000
# physics, production tolerances, TRAJL 1.0, --device-mem 8000, four 40 GB nodes. Force-only
# runs carry the fo8 runs' inert layout (tail inner, MDSTEPS 12), so they read like fo8_base.
#
# Stops: the STOP file; nothing runnable for IDLE_MIN min (empty queue, or nothing fits the
# time left); MAX_FAILS failed runs in a row; any ENV or banner mismatch (the request -> launcher
# mapping itself is then suspect, so nothing more runs on it).
# =============================================================================
set -o pipefail
shopt -s nullglob

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
HB=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune
TRAJ_LAUNCHER=$HB/perlmutter/traj48_puregrid.sh
FO_LAUNCHER=$HB/perlmutter/forces48_puregrid.sh
BIN=$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m9   # tune 2's binary: CUSTOM_LEVEL_* family
QUEUE=$BASE/runs/2026_10_28_tune5_queue
CFG_DEFAULT=$BASE/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime
CFG_DIR=/pscratch/sd/v/vpa/cfgs48
CFG_PREFIX=cl21_48_96_b6p3_m0p2416_m0p2050
CFG_MIN_BYTES=6115000000          # ILDG 48^3x96 payload + headers; every cached config is 6,115,385,4xx B
MASS_LIGHT=-0.2416                # the launchers' MASS_LIGHT; the driver requires ladder[0] == it
GPU40_MARK='totalGlobalMem: 42404806656'   # Grid's banner on a 40 GB A100 (80 GB nodes print another value)
IDLE_MIN=15
MARGIN_MIN=10                     # launcher set-up before srun (~1-2 min) and summary after it
MAX_FAILS=3
ROW_HEADER='end	jobid	name	kind	status	exit	elapsed_s	run_dir	env_check	ladder	layout	n_traj	fo_samples	seed	start	dH	acc	wall_s	cycle_s	fo_percall_s	gpu_peak_mib	pred_wall_s	risk	pair	note'

JOBTAG=check
WLOG=
log() {
  local m
  m="$(date '+%F %T') [$JOBTAG] $*"
  printf '%s\n' "$m"
  [ -z "$WLOG" ] || printf '%s\n' "$m" >> "$WLOG"
}
die() { log "ERROR: $*"; exit 1; }

# squeue %L ([D-]HH:MM:SS or MM:SS) -> whole minutes (the launchers' to_minutes). Three tries
# 20 s apart (a Slurm controller stall must not end the window); 0 if still unreadable.
left_min() {
  local l try
  for try in 1 2 3; do
    l=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | head -1)
    if [[ "$l" =~ ^(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)$ ]]; then
      echo $(( 10#${BASH_REMATCH[2]:-0} * 1440 + 10#${BASH_REMATCH[4]:-0} * 60 + 10#${BASH_REMATCH[5]} ))
      return
    fi
    [ "$try" -eq 3 ] || sleep 20
  done
  echo 0
}

# move_to <file> <dir>: never overwrites; a name already in <dir> gets .<jobid>.<n> appended.
move_to() {
  local dst=$2/$(basename "$1") n=0
  while [ -e "$dst" ]; do n=$(( n + 1 )); dst=$2/$(basename "$1").$SLURM_JOB_ID.$n; done
  mv -- "$1" "$dst" && printf '%s' "$dst"
}

# ---- Validation ------------------------------------------------------------------------
# validate <file>: fills C[] (the request) and the derived run settings, or sets WHY and
# returns 1. Read-only: never moves the file.
declare -A C
ALLOWED=" name kind ladder mg mg_hb mixed mixed_hb seed start max_min layout n_traj fo_samples pred_wall_s pred_force_s risk pair "
NAME_RE='^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$'
want() { [[ "${C[$1]}" =~ $2 ]] || { WHY="$1='${C[$1]}': $3"; return 1; }; }
in_list() { [[ ",$2," == *",$1,"* ]]; }

validate() {
  local f=$1 b line k v n=0 x
  C=() WHY=
  b=$(basename "$f")
  [[ "$b" =~ ^[0-9]{3}_([A-Za-z0-9][A-Za-z0-9._-]{0,39})\.cand$ ]] || { WHY="file name is not <3 digits>_<name>.cand"; return 1; }
  local fname=${BASH_REMATCH[1]}
  { [ -f "$f" ] && [ ! -L "$f" ]; } || { WHY="not a regular file"; return 1; }
  [ "$(stat -c %s "$f")" -le 4096 ] || { WHY="larger than 4096 bytes"; return 1; }
  while IFS= read -r line || [ -n "$line" ]; do
    n=$(( n + 1 ))
    [ "$n" -le 40 ] || { WHY="more than 40 lines"; return 1; }
    [[ "$line" =~ ^([a-z_]+)=([A-Za-z0-9.,/_-]*)$ ]] || { WHY="line $n is not key=value with plain characters"; return 1; }
    k=${BASH_REMATCH[1]} v=${BASH_REMATCH[2]}
    [[ "$ALLOWED" == *" $k "* ]] || { WHY="unknown key '$k'"; return 1; }
    [ -z "${C[$k]+x}" ] || { WHY="key '$k' given twice"; return 1; }
    C[$k]=$v
  done < "$f"
  for k in name kind ladder mg mg_hb mixed mixed_hb seed start max_min; do
    [ -n "${C[$k]+x}" ] || { WHY="missing key '$k'"; return 1; }
  done
  want name "$NAME_RE" "letters, digits, . _ -, at most 40" || return 1
  [ "${C[name]}" = "$fname" ] || { WHY="name=${C[name]} differs from the file name's '$fname'"; return 1; }
  case "${C[kind]}" in
    traj) for k in layout n_traj; do [ -n "${C[$k]+x}" ] || { WHY="kind=traj needs '$k'"; return 1; }; done
          [ -z "${C[fo_samples]+x}" ] || { WHY="fo_samples is for kind=fo only"; return 1; } ;;
    fo)   [ -n "${C[fo_samples]+x}" ] || { WHY="kind=fo needs 'fo_samples'"; return 1; }
          for k in layout n_traj; do [ -z "${C[$k]+x}" ] || { WHY="'$k' is for kind=traj only"; return 1; }; done ;;
    *)    WHY="kind must be traj or fo, got '${C[kind]}'"; return 1 ;;
  esac

  # Ladder: light -> tail; masses as the launchers pass them, verbatim, to HASEN_LADDER.
  want ladder '^-?[0-9]\.[0-9]{1,6}(,-?[0-9]\.[0-9]{1,6})+$' "masses light -> tail, e.g. -0.2416,-0.2400,-0.1870" || return 1
  local -a M
  IFS=, read -r -a M <<< "${C[ladder]}"
  NMASS=${#M[@]}
  local maxm=6
  [ "${C[kind]}" = fo ] && maxm=7
  [ "$NMASS" -le "$maxm" ] || { WHY="ladder has $NMASS masses, at most $maxm for kind=${C[kind]} (host memory on four 40 GB nodes)"; return 1; }
  awk -v l="${C[ladder]}" -v ml="$MASS_LIGHT" 'BEGIN { n = split(l, m, ","); if (m[1] + 0 != ml + 0) exit 1
                                                       for (i = 2; i <= n; i++) if (m[i] + 0 <= m[i - 1] + 0 || m[i] + 0 >= 1) exit 2 }'
  case $? in
    0) ;;
    1) WHY="ladder must start at MASS_LIGHT $MASS_LIGHT"; return 1 ;;
    *) WHY="ladder masses must increase strictly and stay below 1"; return 1 ;;
  esac
  NRUNG=$(( NMASS - 1 ))

  # Solver routes: lists of ratio rungs 0..NRUNG-1, the driver's one-route-per-solve rules.
  for k in mg mg_hb mixed mixed_hb; do
    want "$k" '^([0-9](,[0-9])*)?$' "a comma list of ratio rungs, or empty" || return 1
    local seen=,
    for x in ${C[$k]//,/ }; do
      [ "$x" -lt "$NRUNG" ] || { WHY="$k lists rung $x; this ladder has rungs 0..$(( NRUNG - 1 ))"; return 1; }
      [[ "$seen" != *",$x,"* ]] || { WHY="$k lists rung $x twice"; return 1; }
      seen="$seen$x,"
    done
  done
  for x in ${C[mg_hb]//,/ }; do in_list "$x" "${C[mg]}" || { WHY="mg_hb rung $x is not in mg (the heatbath uses the multigrid rungs' hierarchy)"; return 1; }; done
  for x in ${C[mixed]//,/ }; do ! in_list "$x" "${C[mg]}" || { WHY="rung $x in both mg and mixed"; return 1; }; done
  for x in ${C[mixed_hb]//,/ }; do ! in_list "$x" "${C[mg_hb]}" || { WHY="rung $x in both mg_hb and mixed_hb"; return 1; }; done

  want seed '^(0|[1-9][0-9]{0,4})$' "an integer 0..99999, no leading zero" || return 1

  # Starting configuration.
  want start '^djm[1-4]_[0-9]{1,5}$' "djm<stream>_<trajectory>, e.g. djm3_2000" || return 1
  if [ "${C[start]}" = djm3_2000 ]; then
    CFG=$CFG_DEFAULT CFG_TRAJ=2000 CFG_DIAG=no
  else
    local s=${C[start]#djm}
    CFG=$CFG_DIR/${CFG_PREFIX}-djm-${s%%_*}_cfg_${s#*_}.lime CFG_TRAJ=${s#*_} CFG_DIAG=yes
  fi
  { [ -f "$CFG" ] && [ "$(stat -L -c %s "$CFG")" -ge "$CFG_MIN_BYTES" ]; } || { WHY="start=${C[start]}: no complete configuration at $CFG"; return 1; }

  want max_min '^[1-9][0-9]{0,2}$' "an integer number of minutes, no leading zero" || return 1
  if [ "${C[kind]}" = traj ]; then
    want n_traj '^[1-3]$' "1..3 (longer chains need the user's approval)" || return 1
    local lo=$(( 30 + 45 * ${C[n_traj]} ))
    { [ "${C[max_min]}" -ge "$lo" ] && [ "${C[max_min]}" -le 480 ]; } || { WHY="max_min=${C[max_min]} outside $lo..480 for n_traj=${C[n_traj]}"; return 1; }
    want layout '^[1-9][0-9]{0,2}(/[1-9][0-9]{0,2}){3}$' "L/S/T/G steps, e.g. 12/12/48/48" || return 1
    derive_layout || return 1
  else
    want fo_samples '^[2-9]$' "2..9" || return 1
    { [ "${C[max_min]}" -ge 10 ] && [ "${C[max_min]}" -le 90 ]; } || { WHY="max_min=${C[max_min]} outside 10..90 for kind=fo"; return 1; }
  fi

  # Prediction (copied to the row, never used to run).
  [ -z "${C[pred_wall_s]+x}" ] || want pred_wall_s '^[0-9]{1,6}$' "integer seconds" || return 1
  [ -z "${C[pred_force_s]+x}" ] || want pred_force_s '^[0-9]{1,6}$' "integer seconds" || return 1
  [ -z "${C[risk]+x}" ] || want risk '^(light|strange|tail|gauge|cost|none)$' "light|strange|tail|gauge|cost|none" || return 1
  [ -z "${C[pair]+x}" ] || want pair "$NAME_RE" "a candidate name" || return 1

  # A run name is used once: the launchers refuse an existing run dir only for the same date.
  local d
  [ "${VERIFY_ONLY:-no}" = yes ] || for d in "$BASE"/runs/*_t5_"${C[name]}"; do WHY="run name t5_${C[name]} was used before ($d)"; return 1; done
  return 0
}

# derive_layout: C[layout] = L/S/T/G -> levels. Sets STEPS[] (distinct steps, ascending = level
# 0..NLEV-1), MULT[] (MULT[l] = STEPS[l] / STEPS[l-1] / 2), LVL_LIGHT/STRANGE/TAIL/GAUGE, MDSTEPS.
derive_layout() {
  local -a G
  local g l i
  IFS=/ read -r -a G <<< "${C[layout]}"
  for g in "${G[@]}"; do
    { [ $(( 10#$g )) -ge 2 ] && [ $(( 10#$g )) -le 256 ]; } || { WHY="layout=${C[layout]}: steps must be 2..256"; return 1; }
  done
  mapfile -t STEPS < <(printf '%d\n' $(( 10#${G[0]} )) $(( 10#${G[1]} )) $(( 10#${G[2]} )) $(( 10#${G[3]} )) | sort -n -u)
  NLEV=${#STEPS[@]}
  MULT=(1)
  for (( l = 1; l < NLEV; l++ )); do
    if (( STEPS[l] % STEPS[l - 1] != 0 || (STEPS[l] / STEPS[l - 1]) % 2 != 0 )); then
      WHY="layout=${C[layout]}: ${STEPS[l]} / ${STEPS[l - 1]} is not an even integer (a nested level runs 2 x mult x its parent's steps)"
      return 1
    fi
    MULT[l]=$(( STEPS[l] / STEPS[l - 1] / 2 ))
  done
  local -a LV
  for g in 0 1 2 3; do
    for (( i = 0; i < NLEV; i++ )); do (( STEPS[i] == 10#${G[g]} )) && LV[g]=$i; done
  done
  LVL_LIGHT=${LV[0]} LVL_STRANGE=${LV[1]} LVL_TAIL=${LV[2]} LVL_GAUGE=${LV[3]}
  MDSTEPS=${STEPS[0]}
  return 0
}

# ---- Launcher environment ----------------------------------------------------------------
# launch_env: the exact KEY=VALUE list handed to the launcher (env(1) arguments), from C[].
launch_env() {
  LENV=(SLURM_JOB_ID="${SLURM_JOB_ID:-}" BIN="$BIN" RUN="t5_${C[name]}" LADDER_PROFILE=custom
        CUSTOM_LADDER="${C[ladder]}" HMC_SEED_OFFSET="${C[seed]}"
        HASEN_GRID_MG_RUNGS="${C[mg]}" HASEN_GRID_MG_HEATBATH_RUNGS="${C[mg_hb]}"
        HASEN_GRID_MIXED_CG_RUNGS="${C[mixed]}" HASEN_GRID_MIXED_CG_HEATBATH_RUNGS="${C[mixed_hb]}")
  if [ "${C[kind]}" = traj ]; then
    LENV+=(CUSTOM_MDSTEPS="$MDSTEPS" CUSTOM_LEVEL_LIGHT="$LVL_LIGHT" CUSTOM_LEVEL_STRANGE="$LVL_STRANGE"
           CUSTOM_LEVEL_TAIL="$LVL_TAIL" CUSTOM_LEVEL_GAUGE="$LVL_GAUGE"
           N_TRAJ="${C[n_traj]}" TIMEOUT_MIN="${C[max_min]}")
    local l
    for (( l = 1; l < NLEV; l++ )); do LENV+=(CUSTOM_LEVEL_MULT_$l="${MULT[l]}"); done   # only used levels: the driver refuses others
    [ "$CFG_DIAG" = no ] || LENV+=(DIAG_START_CFG="$CFG" DIAG_START_TRAJ="$CFG_TRAJ")
  else
    LENV+=(CUSTOM_TAIL_LEVEL=inner CUSTOM_MDSTEPS=12 FO_PROFILE=production
           FORCES_SAMPLES="${C[fo_samples]}" DEVICE_MEM_MB=8000)
    [ "$CFG_DIAG" = no ] || LENV+=(DIAG_START_CFG="$CFG")
  fi
}

# expected_env: the tokens the run's ENV line must carry (each as " KEY=VALUE ").
expected_env() {
  local mg=${C[mg]:-none} mghb=${C[mg_hb]:-none} mx=${C[mixed]:-none} mxhb=${C[mixed_hb]:-none}
  WANT=("BIN=$(basename "$BIN")" "LADDER=${C[ladder]}" "LADDER_PROFILE=custom" "SEED=${C[seed]}"
        "LATT=48.48.48.96" "NODES=4" "NTASKS=16" "TRAJL=1.0" "MASS_LIGHT=-0.2416" "MASS_STRANGE=-0.2050"
        "CSW=1.20536588031793" "BETA=6.3" "U0=0.84570646270714" "STOUT_RHO=0.125" "STOUT_NSMEAR=1"
        "RAT=7e-4/35.0/20" "RAT_DEGREE_MD=20" "TOL_STRANGE_MD=1e-9" "RAT_BOUNDS_CHECK=10/1e-6/1"
        "TOL_DRV=1e-11" "TOL_ACT=1e-12" "TOL_STRANGE=1e-9" "QUDA_ENV=0" "DEVICE_MEM_MB=8000"
        "INTEGRATOR=ForceGradient")
  if [ "${C[kind]}" = traj ]; then
    WANT+=("MODE=fresh" "METROP=on" "N_TRAJ=${C[n_traj]}" "START=$(basename "$CFG")" "MDSTEPS=$MDSTEPS"
           "LEVEL_LIGHT=$LVL_LIGHT" "LEVEL_STRANGE=$LVL_STRANGE" "LEVEL_TAIL=$LVL_TAIL" "LEVEL_GAUGE=$LVL_GAUGE"
           "LEVEL_MULT=${MULT[1]:-1},${MULT[2]:-1},${MULT[3]:-1}" "TIMEOUT_MIN=${C[max_min]}"
           "GRID_MG_RUNGS=$mg" "GRID_MG_HEATBATH_RUNGS=$mghb" "GRID_MIXED_CG_RUNGS=$mx" "GRID_MIXED_CG_HEATBATH_RUNGS=$mxhb")
  else
    WANT+=("MODE=FORCES_ONLY" "FORCES_SAMPLES=${C[fo_samples]}" "FORCES_SKIP=none" "FO_PROFILE=production"
           "FORCES_STIFFNESS=0" "IMPORT_CFG=$(basename "$CFG")"
           "GRID_MG_RUNGS=$mg" "GRID_MG_HB=$mghb" "GRID_MIXED=$mx" "GRID_MIXED_HB=$mxhb")
  fi
}

# env_check <hmc.log>: ENV line tokens, the [Levels] banner (traj), every GPU a 40 GB card.
# ENV_WHY = a HARD mismatch: what ran is not what was asked (wrong launcher, ENV token, level
# layout, node type); the request -> launcher mapping is suspect and the worker stops.
# ENV_MISSING = lines a run that died early never printed (no ENV line, no banner): a failed run.
# ENV_OK=yes only with neither.
env_check() {
  local log=$1 line w l gpus peak tag=TRAJ48_PUREGRID
  ENV_OK=no ENV_WHY= ENV_MISSING=
  [ "${C[kind]}" = fo ] && tag=FORCES48_PUREGRID
  line=$(head -1 "$log")
  if [[ "$line" != "ENV $tag "* ]]; then
    if [[ "$line" == "ENV "* ]]; then ENV_WHY="line 1 is not the ENV $tag line"; else ENV_MISSING="no ENV line"; fi
    return
  fi
  expected_env
  for w in "${WANT[@]}"; do [[ " $line " == *" $w "* ]] || ENV_WHY="$ENV_WHY ${w%%=*}"; done
  [ -z "$ENV_WHY" ] || ENV_WHY="ENV differs:$ENV_WHY"
  if [ "${C[kind]}" = traj ]; then
    if grep -a -q -F '[Levels] light=' "$log"; then
      grep -a -q -F "[Levels] light=L$LVL_LIGHT strange=L$LVL_STRANGE tail=L$LVL_TAIL gauge=L$LVL_GAUGE mults=" "$log" || ENV_WHY="$ENV_WHY; [Levels] group line differs"
      for (( l = 0; l < NLEV; l++ )); do
        grep -a -q -F "[Levels] L$l: mult ${MULT[l]}, steps/traj ${STEPS[l]}, " "$log" || ENV_WHY="$ENV_WHY; [Levels] L$l is not mult ${MULT[l]} steps ${STEPS[l]}"
      done
      [ "$(grep -a -c -E '\[Levels\] L[0-9]: ' "$log")" -eq "$NLEV" ] || ENV_WHY="$ENV_WHY; [Levels] does not show $NLEV levels"
    else
      ENV_MISSING="no [Levels] banner"
    fi
  fi
  gpus=$(grep -a -o -E 'totalGlobalMem: [0-9]+' "$log" | sort -u)
  if [ -z "$gpus" ]; then ENV_MISSING="${ENV_MISSING:+$ENV_MISSING, }no GPU banner"
  elif [ "$gpus" != "$GPU40_MARK" ]; then ENV_WHY="$ENV_WHY; not (only) 40 GB GPUs"
  fi
  # nvidia-smi sees every node's four cards (Grid's banner is rank 0's only).
  peak=$(dirname "$log")/gpu_mem_peak.txt
  [ ! -f "$peak" ] || ! grep -q -v -E ' total=40960 ' "$peak" || ENV_WHY="$ENV_WHY; nvidia-smi shows a card that is not 40 GB"
  ENV_WHY=${ENV_WHY#; }
  [ -n "$ENV_WHY$ENV_MISSING" ] || ENV_OK=yes
}

# ---- Results ------------------------------------------------------------------------------
clean() { printf '%s' "${1:--}" | tr '\t\n' '  ' | cut -c1-300; }
write_row() {   # <status> <exit> <elapsed> <run_dir> <env> <dH> <acc> <wall> <cycle> <fo> <gpu> <note>
  local f stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  f=$QUEUE/results/${stamp}_${SLURM_JOB_ID:-check}_$(printf '%s' "${C[name]:-$REJ_NAME}" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-40).tsv
  [ -e "$QUEUE/results/000_header.tsv" ] || printf '%s\n' "$ROW_HEADER" > "$QUEUE/results/000_header.tsv"
  local IFS=$'\t'
  local -a r=("$(date '+%F %T')" "${SLURM_JOB_ID:-}" "$(clean "${C[name]:-$REJ_NAME}")" "$(clean "${C[kind]}")"
              "$1" "$2" "$3" "$(clean "$4")" "$(clean "$5")" "$(clean "${C[ladder]}")" "$(clean "${C[layout]}")"
              "$(clean "${C[n_traj]}")" "$(clean "${C[fo_samples]}")" "$(clean "${C[seed]}")" "$(clean "${C[start]}")"
              "$(clean "$6")" "$(clean "$7")" "$(clean "$8")" "$(clean "$9")" "$(clean "${10}")" "$(clean "${11}")"
              "$(clean "${C[pred_wall_s]}")" "$(clean "${C[risk]}")" "$(clean "${C[pair]}")" "$(clean "${12}")")
  printf '%s\n' "${r[*]}" > "$f"
}

# traj_cols <hmc.log>: dH, Metropolis (A/R), wall (MD: H before -> H after), cycle, comma lists,
# '|'-separated (a tab is IFS whitespace: empty fields would collapse on read).
traj_cols() {
  awk '/Total H before trajectory = / { tb = $5 }
       /Total H after trajectory /    { dh = dh s $NF; w = w s sprintf("%.0f", $5 - tb) }
       /Metropolis_test -- ACCEPTED/  { a = a s "A" }
       /Metropolis_test -- REJECTED/  { a = a s "R" }
       /Total time for trajectory \(s\): / { c = c s sprintf("%.0f", $NF); s = "," }
       END { printf "%s|%s|%s|%s\n", dh, a, w, c }' "$1"
}
# fo_cols <hmc.log>: per-call force time per monomial, mean over samples >= 1 (L014).
fo_cols() {
  awk '/FORCES_ONLY level / { nm = ""; sm = -1; t = ""
         for (i = 1; i < NF; i++) { if ($i == "level") nm = $(i + 1); if ($i == "sample") sm = $(i + 1) + 0 }
         for (i = 1; i <= NF; i++) if ($i ~ /^time=/) t = substr($i, 6)
         if (nm != "" && sm >= 1 && t != "") { if (!(nm in n)) o[++k] = nm; s[nm] += t; n[nm]++ } }
       END { for (j = 1; j <= k; j++) printf "%s%s=%.2f", (j > 1 ? "," : ""), o[j], s[o[j]] / n[o[j]] }' "$1"
}
gpu_peak() {
  [ -f "$1" ] || { echo -; return; }
  awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^peak_used=/) { v = substr($i, 11) + 0; if (v > m) m = v } } END { print m + 0 }' "$1"
}

# ---- --check mode: validate one file and print what would run ------------------------------
if [ "${1:-}" = --check ]; then
  [ -n "${2:-}" ] || { echo "usage: bash $0 --check <candidate file>" >&2; exit 2; }
  if ! validate "$2"; then echo "REJECT $(basename "$2"): $WHY"; exit 1; fi
  launch_env
  expected_env
  echo "OK $(basename "$2"): kind=${C[kind]} launcher=$( [ "${C[kind]}" = traj ] && basename "$TRAJ_LAUNCHER" || basename "$FO_LAUNCHER") config=$CFG"
  [ "${C[kind]}" != traj ] || echo "levels: steps ${STEPS[*]}  mults ${MULT[*]}  light/strange/tail/gauge = L$LVL_LIGHT/L$LVL_STRANGE/L$LVL_TAIL/L$LVL_GAUGE"
  echo "env: ${LENV[*]}"
  echo "ENV line must carry: ${WANT[*]}"
  exit 0
fi
# ---- --verify mode: check an existing run's log against a candidate (read-only) -------------
if [ "${1:-}" = --verify ]; then
  [ -n "${2:-}" ] && [ -f "${3:-}" ] || { echo "usage: bash $0 --verify <candidate file> <hmc.log>" >&2; exit 2; }
  VERIFY_ONLY=yes
  if ! validate "$2"; then echo "REJECT $(basename "$2"): $WHY"; exit 1; fi
  env_check "$3"
  if [ "$ENV_OK" = yes ]; then echo "MATCH $3"; exit 0; fi
  echo "MISMATCH $3: ${ENV_WHY}${ENV_WHY:+${ENV_MISSING:+; }}${ENV_MISSING}"
  exit 1
fi

# ---- Start-up -----------------------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: SLURM_JOB_ID is not set: submit with sbatch, or SLURM_JOB_ID=<id> bash $0 inside a 4-node allocation" >&2; exit 1; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: SLURM_JOB_ID must be numeric, got '$SLURM_JOB_ID'" >&2; exit 1; }
JOBTAG=$SLURM_JOB_ID
[ -d "$QUEUE" ] || die "queue directory $QUEUE does not exist (the planner's tune5.py enqueue creates it)"
for d in pending running done failed rejected results; do mkdir -p "$QUEUE/$d" || die "cannot create $QUEUE/$d"; done
WLOG=$QUEUE/worker_$SLURM_JOB_ID.log
[ "$(squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null | head -1)" = RUNNING ] || die "job $SLURM_JOB_ID is not RUNNING"
JOB_NODES=$(squeue -h -j "$SLURM_JOB_ID" -o %D 2>/dev/null | head -1)
{ [[ "$JOB_NODES" =~ ^[0-9]+$ ]] && [ "$JOB_NODES" -ge 4 ]; } || die "job $SLURM_JOB_ID has '${JOB_NODES:-?}' node(s); the launchers need 4"
# Every timing row must come from 40 GB nodes (L119): refuse a job that was not constrained to them
# (an unconstrained allocation can mix 40 and 80 GB; alloc_gpu.sh needs its third argument, 40).
JOB_FEAT=$(squeue -h -j "$SLURM_JOB_ID" -o %f 2>/dev/null | head -1)
[[ "$JOB_FEAT" == *hbm40g* ]] || die "job $SLURM_JOB_ID has features '${JOB_FEAT:-?}', not hbm40g: use sbatch (header) or alloc_gpu.sh <4> <h> 40"
{ [ -f "$BIN" ] && [ -x "$BIN" ]; } || die "BIN is not an executable file: $BIN"
[ -f "$TRAJ_LAUNCHER" ] && [ -f "$FO_LAUNCHER" ] || die "a launcher is missing: $TRAJ_LAUNCHER / $FO_LAUNCHER"
# The batch environment carries SLURM_GPUS_PER_NODE=4 from the header, which srun reads beside the
# launchers' --gpus-per-task=1 (traj48_puregrid_batch.sh): drop it, as that wrapper does.
unset SLURM_GPUS_PER_NODE
# Nothing from the submitting shell may reach a launcher except what launch_env sets: every knob
# family the launchers or the driver read, plus the launchers' caller overrides.
while read -r v; do unset "$v"; done < <(compgen -e | grep -E '^(CUSTOM_|DIAG_|HASEN_|GRID_MG_|RESUME_|FORCES_|QUDA_|HMC_MG_|CKPT_|TXQCD_|WCF_)|^(GRID_EXTRA_ARGS|FO_PROFILE|LADDER_PROFILE|N_TRAJ|HMC_SEED_OFFSET|TIMEOUT_MIN|DEVICE_MEM_MB|GPU_MON_MS|SRUN_EXTRA|CHECKPOINT_ROOT|INTEGRATOR_VERBOSE_MEM|RAT_BOUNDS_CHECK_FREQ|RAT_BOUNDS_CHECK_TOL|RAT_BOUNDS_CHECK_ABORT|RAT_DEGREE_MD|TUNE_CG_TOL_STRANGE_MD|MPICH_GPU_IPC_ENABLED|MPICH_RDMA_ENABLED_CUDA|MPICH_GPU_EAGER_REGISTER_HOST_MEM|MPICH_GPU_NO_ASYNC_MEMCPY|USE_HMC_MG|LAMBDA_MN2|STRANGE_EVEN|NO_METROP|IMPORT_CFG|RUN)$')
log "start: worker sha256 $(sha256sum "$0" | cut -c1-16), traj launcher $(sha256sum "$TRAJ_LAUNCHER" | cut -c1-16), fo launcher $(sha256sum "$FO_LAUNCHER" | cut -c1-16), bin $(basename "$BIN"), queue $QUEUE, $(left_min) min left"

# ---- Main loop ----------------------------------------------------------------------------
FAILS=0
IDLE_SINCE=
REJ_NAME=
while :; do
  [ ! -e "$QUEUE/STOP" ] || { log "STOP file present: exiting"; break; }
  LEFT=$(left_min)
  [ "$LEFT" -ge $(( 10 + MARGIN_MIN )) ] || { log "$LEFT min left: exiting"; break; }

  # Pick: the first pending file (name order) that validates and fits; refused ones go to rejected/.
  PICK=
  NPEND=0 NYOUNG=0 NLONG=0
  for f in "$QUEUE"/pending/[0-9][0-9][0-9]_*.cand; do
    NPEND=$(( NPEND + 1 ))
    b=$(basename "$f")
    # A file written in place may be incomplete: one younger than 30 s waits for the next pass.
    if [ $(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || date +%s) )) -lt 30 ]; then NYOUNG=$(( NYOUNG + 1 )); continue; fi
    if ! validate "$f"; then
      if dst=$(move_to "$f" "$QUEUE/rejected" 2>/dev/null); then
        printf '%s\n' "$WHY" > "$dst.why"
        REJ_NAME=${b%.cand}; REJ_NAME=${REJ_NAME#[0-9][0-9][0-9]_}
        log "REJECTED $b: $WHY"
        write_row rejected - 0 - - - - - - - - "$WHY"
        REJ_NAME=
      fi
      continue
    fi
    if [ $(( ${C[max_min]} + MARGIN_MIN )) -gt "$LEFT" ]; then NLONG=$(( NLONG + 1 )); continue; fi
    [ ! -e "$QUEUE/running/$b" ] || continue          # same file name already running elsewhere
    mv -- "$f" "$QUEUE/running/$b" 2>/dev/null || continue   # another worker claimed it
    PICK=$b
    break
  done
  if [ -z "$PICK" ]; then
    # Valid candidates that need more time than is left will not fit later either: release the
    # nodes now. Wait (up to IDLE_MIN) only on an empty queue or a file still being written.
    if [ "$NLONG" -gt 0 ] && [ "$NYOUNG" -eq 0 ]; then log "$NLONG pending candidate(s), none fits the $LEFT min left: exiting"; break; fi
    NOW=$(date +%s)
    [ -n "$IDLE_SINCE" ] || { IDLE_SINCE=$NOW; log "nothing runnable ($NPEND pending, $LEFT min left): waiting up to $IDLE_MIN min"; }
    if [ $(( NOW - IDLE_SINCE )) -ge $(( IDLE_MIN * 60 )) ]; then log "nothing runnable for $IDLE_MIN min: exiting"; break; fi
    sleep 60
    continue
  fi
  IDLE_SINCE=

  # Re-validate the claimed copy: the run uses exactly what was checked.
  CF=$QUEUE/running/$PICK
  STEM=${PICK%.cand}
  if ! validate "$CF"; then
    dst=$(move_to "$CF" "$QUEUE/rejected") && printf '%s\n' "$WHY" > "$dst.why"
    log "REJECTED $PICK after claiming: $WHY"
    continue
  fi
  launch_env
  OUT=$QUEUE/running/$STEM.out
  LAUNCHER=$FO_LAUNCHER
  [ "${C[kind]}" = traj ] && LAUNCHER=$TRAJ_LAUNCHER
  log "RUN $PICK: kind=${C[kind]} ladder=${C[ladder]} ${C[layout]:+layout=${C[layout]} }seed=${C[seed]} start=${C[start]} max_min=${C[max_min]} ($LEFT min left)"
  T0=$SECONDS
  # Backstop only. Each launcher wraps its srun in its own `timeout` (TIMEOUT_MIN = max_min for
  # trajectories, a fixed 90 min for force-only), which runs in a process group of its own: this
  # outer timeout cannot reach that srun, so it must never fire first (a killed launcher would
  # leave its srun running beside the next candidate's). It fires only if a launcher hangs after
  # its srun has ended.
  BACKSTOP=$(( ${C[max_min]} + 8 ))
  [ "${C[kind]}" = traj ] || BACKSTOP=98
  timeout -k 60s "${BACKSTOP}m" env "${LENV[@]}" bash "$LAUNCHER" > "$OUT" 2>&1
  RC=$?
  EL=$(( SECONDS - T0 ))

  # Read back: the run dir from the launcher's own "running ... -> <log>" line.
  LOGF=$(sed -n 's/^running .* -> \(\/.*\/hmc\.log\)$/\1/p' "$OUT" | head -1)
  if [ -z "$LOGF" ]; then for d in "$BASE"/runs/*_t5_"${C[name]}"; do [ -f "$d/hmc.log" ] && LOGF=$d/hmc.log; done; fi
  RDIR=- DH=- ACC=- WALL=- CYC=- FOC=- GPU=- NOTE= ENV_OK=n/a ENV_WHY= ENV_MISSING=
  if [ -n "$LOGF" ] && [ -f "$LOGF" ]; then
    RDIR=$(dirname "$LOGF")
    env_check "$LOGF"
    GPU=$(gpu_peak "$RDIR/gpu_mem_peak.txt")
    if [ "${C[kind]}" = traj ]; then
      IFS='|' read -r DH ACC WALL CYC < <(traj_cols "$LOGF")
      NDONE=$(grep -a -c 'Total time for trajectory (s):' "$LOGF")
    else
      FOC=$(fo_cols "$LOGF")
      NDONE=$(grep -a -c 'FORCES_ONLY samples=' "$LOGF")
    fi
    NOTE=$(grep -a -m1 -E 'ERROR:|ABORT|oom_kill|out of memory|cudaMalloc failed|FATAL|srun: error' "$LOGF" "$OUT" | head -1)
  else
    NDONE=0
    NOTE=$(grep -a -m1 -E 'ERROR:' "$OUT" | head -1)
    NOTE=${NOTE:-no hmc.log}
  fi
  if [ -n "$ENV_WHY" ]; then STATUS=env_mismatch
  elif [ "$RC" -eq 124 ]; then STATUS=timeout
  elif [ "$RC" -ne 0 ]; then STATUS=fail
  elif [ "${C[kind]}" = traj ] && [ "$NDONE" -ne "${C[n_traj]}" ]; then STATUS=incomplete
  elif [ "${C[kind]}" = fo ] && [ "$NDONE" -lt 1 ]; then STATUS=incomplete
  elif [ "$ENV_OK" != yes ]; then STATUS=incomplete
  else STATUS=ok
  fi
  [ "$ENV_OK" = yes ] || NOTE="${ENV_WHY:+$ENV_WHY; }${ENV_MISSING:+$ENV_MISSING; }$NOTE"
  write_row "$STATUS" "$RC" "$EL" "$RDIR" "$ENV_OK" "$DH" "$ACC" "$WALL" "$CYC" "$FOC" "$GPU" "$NOTE"
  STATE=done
  [ "$STATUS" = ok ] || STATE=failed
  move_to "$CF" "$QUEUE/$STATE" > /dev/null
  move_to "$OUT" "$QUEUE/$STATE" > /dev/null
  log "END $PICK: $STATUS exit=$RC ${EL}s env=$ENV_OK dH=$DH acc=$ACC wall=$WALL run=$RDIR${NOTE:+ note: $(clean "$NOTE")}"

  case "$STATUS" in
    ok) FAILS=0 ;;
    env_mismatch) log "ENV or banner mismatch: the request -> launcher mapping is suspect, exiting"; break ;;
    *) FAILS=$(( FAILS + 1 ))
       [ "$FAILS" -lt "$MAX_FAILS" ] || { log "$FAILS failed runs in a row: exiting"; break; } ;;
  esac
done
log "exit"
exit 0
