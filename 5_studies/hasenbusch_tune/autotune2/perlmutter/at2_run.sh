#!/bin/bash
# =============================================================================
# at2_run.sh -- autotune 2: one run on 4 nodes of the batch job this session lives in.
# It never allocates or submits: it attaches srun steps to its own job (SLURM_JOB_ID) through
# the production launchers: Grid forces48_puregrid.sh and traj48_puregrid.sh, Chroma
# chroma_baseg_mds.sh (inputs from at2_chroma_inputs.py beside this file).
# A 4-node job takes one run at a time on its 4 nodes. An 8-node job has two lanes of 4 nodes,
# LANE=1 (nodes 1-4) and LANE=2 (nodes 5-8), one run in each at a time, side by side; every run
# is still a 4-node run, timed against the 4-node baseline. LANE=all (Chroma only) is one run on
# all 8 nodes, layout 2 2 2 4: its timings compare only with a baseline run on 8 nodes.
#
# Usage, one line each; start it with the Bash tool and run_in_background: true.
#   bash at2_run.sh fo      <name> <ladder> [KEY=VALUE ...]                          Grid force-only
#   bash at2_run.sh traj    <name> <ladder> <layout> <n_traj> <seed> [KEY=VALUE ...] Grid trajectories
#   bash at2_run.sh resume  <name> <run dir to continue> <ladder> <layout> <n_traj> [KEY=VALUE ...]
#   bash at2_run.sh ctraj   <name> <ladder> <layout> <n_traj> <seed> [EST_MIN=<m>] [LANE=<l>]  Chroma
#   bash at2_run.sh cresume <name> <restart xml> <n_traj> [EST_MIN=<m>] [LANE=<l>]    Chroma continuation
#   In an 8-node job every run needs LANE=1, LANE=2 or (Chroma) LANE=all.
#
#   name    letters, digits, _ ; the run directory is runs/<YYYY_M_D>_at2<chain>_<name> (Grid) or
#           runs/<YYYY_M_D>_chroma_baseg_mds<outer steps>_at2<chain>_<name> (Chroma)
#   ladder  masses lightest first, comma-separated, the last one is the tail; Grid 2 to 6 masses
#           (7 do not fit on 4 x 40 GB), Chroma 2 to 10 with the hmc of chroma/install_chrono24
#           (QUDA with 24 predictor channels; the old hmc stops at 6 masses).
#           Baseline: -0.2416,-0.2400,-0.2320,-0.2180,-0.1870
#   Chroma: Grid's solver rule (multigrid below -0.226, CG at and above), the input decoded and
#           checked against the request before the run (printed: ladder, levels, solvers, seed);
#           a configuration and restart XML saved after EVERY trajectory under
#           $PSCRATCH/at2_chroma_saves/at2<chain>_<name>/ (6.1 GB each). To continue a run that was
#           cut off, give cresume the last restart XML there (..._restart_<traj>.xml) and the
#           number of trajectories still to run. Seed S = the QDP++ seed words S S+1000 S+2000
#           S+3000; seed "template" keeps cfg_2000's metadata seed (reproductions only). Results: the run's *.xmldat (deltaH, AcceptP, seconds_for_trajectory, a full
#           trajectory: Chroma has no MD-only time); chroma_runs_csv.py tabulates them. The
#           launcher's own exit code can read 141 on a clean run: the log's last line exit=0 is
#           the truth.
#   layout  L/S/T/G = steps per trajectory of the light rungs, strange, tail, gauge (e.g.
#           8/16/32/64, 12/12/48/48). The distinct values must nest: each is 2 x an integer x
#           the next smaller one. Equal values share a level.
#   seed    HMC_SEED_OFFSET of a fresh start (302-305: baseline screens exist; 300: the
#           benchmark chains). A resume restores the RNG and takes no seed.
#   KEY=VALUE, only these:
#     FO_PROFILE=production|screen  fo: production = every force at production tolerance;
#                                   screen = strange and gauge skipped, deriv tol 1e-8 (faster)
#     FORCES_SAMPLES=<n>            fo: default 9 (sample 0 carries set-up; read samples 1..n-1)
#     DIAG_START_CFG=<file> DIAG_START_TRAJ=<n>   traj: a fresh start from another configuration
#     EST_MIN=<minutes>             your own estimate of the whole run (default: fo 22 production,
#                                   14 screen; traj 70 per trajectory + 10; ctraj and cresume 30
#                                   per trajectory + 10). With LANE, the only KEYs a Chroma run takes.
#     LANE=1|2|all                  8-node job: which lane (above); a 4-node job takes none (or 1)
#     HASEN_GRID_MG_RUNGS HASEN_GRID_MG_HEATBATH_RUNGS HASEN_GRID_MIXED_CG_RUNGS
#     HASEN_GRID_MIXED_CG_HEATBATH_RUNGS   solver routes; default from the ladder, as tune5.py:
#                                   multigrid for a rung whose lighter mass is below -0.226 (its
#                                   heatbath too if both masses are), mixed-precision CG otherwise
#
# Refuses (exit 2, nothing run): another run active in this job (4 nodes) or in the lane asked
# for (8 nodes); less time left than the run needs plus 3 min; a bad ladder, layout, name or KEY.
# Lane state: /tmp/at2_lanes_<job>/lane<l>.pid on the session's node (a lane is busy while that
# process is a live run); each start is logged in $AT2_JOBDIR/lanes.log. A trajectory run gets
# TIMEOUT_MIN = minutes left - 3, so the launcher, not Slurm, ends an overrun: completed
# trajectories are checkpointed and `resume` continues them in a later job.
# Line 1 of the run's hmc.log (the ENV line) is the truth of what ran.
# =============================================================================
set -u
die() { echo "at2_run: $*" >&2; exit 2; }

ROOT=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
HB=$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune
BIN=$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m9
MG_BELOW=-0.226
MAX_MASSES=6
AT2_BIN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHROMA_GEN=$AT2_BIN/at2_chroma_inputs.py
CHROMA_LAUNCHER=$ROOT/grid-lqcd-workflow/5_studies/chroma_inputs/perlmutter/chroma_baseg_mds.sh
CHROMA_HMC_BIN=$ROOT/chroma/install_chrono24/bin/hmc   # every autotune Chroma run, baseline included
CHROMA_INPUTS=$ROOT/chroma/inputs
CHROMA_SAVES=${PSCRATCH:-/pscratch/sd/v/vpa}/at2_chroma_saves

[ -n "${SLURM_JOB_ID:-}" ] || die "SLURM_JOB_ID is not set: run this inside the batch job"
CHAIN=${AT2_CHAIN:-}
[[ "$CHAIN" =~ ^[A-Z]$ ]] || die "AT2_CHAIN is not set to one capital letter (start_agent.sh exports it)"

mode=${1:-}
restart=""
case "$mode" in
  fo)      [ $# -ge 3 ] || die "usage: at2_run.sh fo <name> <ladder> [KEY=VALUE ...]"
           name=$2 ladder=$3 layout=12/12/48/48 n_traj=0 seed=0 prev=""; shift 3 ;;
  traj)    [ $# -ge 6 ] || die "usage: at2_run.sh traj <name> <ladder> <layout> <n_traj> <seed> [KEY=VALUE ...]"
           name=$2 ladder=$3 layout=$4 n_traj=$5 seed=$6 prev=""; shift 6 ;;
  resume)  [ $# -ge 6 ] || die "usage: at2_run.sh resume <name> <run dir> <ladder> <layout> <n_traj> [KEY=VALUE ...]"
           name=$2 prev=$3 ladder=$4 layout=$5 n_traj=$6 seed=0; shift 6 ;;
  ctraj)   [ $# -ge 6 ] || die "usage: at2_run.sh ctraj <name> <ladder> <layout> <n_traj> <seed> [EST_MIN=<m>]"
           name=$2 ladder=$3 layout=$4 n_traj=$5 seed=$6 prev=""; shift 6; MAX_MASSES=10 ;;
  cresume) [ $# -ge 4 ] || die "usage: at2_run.sh cresume <name> <restart xml> <n_traj> [EST_MIN=<m>]"
           name=$2 restart=$3 n_traj=$4 ladder="" layout="" seed=0 prev=""; shift 4
           [ -r "$restart" ] || die "restart XML not readable: $restart" ;;
  *)       die "first argument must be fo, traj, resume, ctraj or cresume" ;;
esac
case "$mode" in ctraj|cresume) CHROMA=1 ;; *) CHROMA=0 ;; esac

[[ "$name" =~ ^[A-Za-z0-9_]{1,60}$ ]] || die "name must be 1-60 letters, digits or _, got '$name'"
[[ "$n_traj" =~ ^[0-9]+$ ]] || die "n_traj must be a whole number, got '$n_traj'"
[ "$mode" = fo ] || [ "$n_traj" -ge 1 ] || die "n_traj must be at least 1"
if [ "$mode" = ctraj ] && [ "$seed" = template ]; then
  :   # Chroma only: keep the seed of cfg_2000's metadata (the earlier single-trajectory references)
else
  [[ "$seed" =~ ^[0-9]+$ ]] || die "seed must be a whole number, got '$seed'"
fi

# ---- KEY=VALUE (whitelist) ----------------------------------------------------------------
unset FO_PROFILE FORCES_SAMPLES DIAG_START_CFG DIAG_START_TRAJ EST_MIN LANE SRUN_EXTRA \
      CHROMA_NODES CHROMA_GEOM CHROMA_NODELIST \
      HASEN_GRID_MG_RUNGS HASEN_GRID_MG_HEATBATH_RUNGS HASEN_GRID_MIXED_CG_RUNGS HASEN_GRID_MIXED_CG_HEATBATH_RUNGS
route_given=0
for kv in "$@"; do
  key=${kv%%=*} val=${kv#*=}
  [ "$key" != "$kv" ] || die "extra argument '$kv' is not KEY=VALUE"
  [ "$CHROMA" = 0 ] || [ "$key" = EST_MIN ] || [ "$key" = LANE ] || die "a Chroma run takes only EST_MIN and LANE, not $key"
  case "$key" in
    FO_PROFILE)       [ "$mode" = fo ] || die "FO_PROFILE is for fo runs"
                      [[ "$val" =~ ^(production|screen)$ ]] || die "FO_PROFILE must be production or screen"
                      FO_PROFILE=$val ;;
    FORCES_SAMPLES)   [ "$mode" = fo ] || die "FORCES_SAMPLES is for fo runs"
                      [[ "$val" =~ ^[1-9][0-9]?$ ]] || die "FORCES_SAMPLES must be 1-99"
                      FORCES_SAMPLES=$val ;;
    DIAG_START_CFG)   [ "$mode" = traj ] || die "DIAG_START_CFG is for fresh traj runs"
                      [ -r "$val" ] || die "DIAG_START_CFG file not readable: $val"
                      DIAG_START_CFG=$val ;;
    DIAG_START_TRAJ)  [ "$mode" = traj ] || die "DIAG_START_TRAJ is for fresh traj runs"
                      [[ "$val" =~ ^[0-9]+$ ]] || die "DIAG_START_TRAJ must be a whole number"
                      DIAG_START_TRAJ=$val ;;
    EST_MIN)          [[ "$val" =~ ^[1-9][0-9]*$ ]] || die "EST_MIN must be a positive whole number of minutes"
                      EST_MIN=$val ;;
    LANE)             [[ "$val" =~ ^(1|2|all)$ ]] || die "LANE must be 1, 2 or all"
                      LANE=$val ;;
    HASEN_GRID_MG_RUNGS|HASEN_GRID_MG_HEATBATH_RUNGS|HASEN_GRID_MIXED_CG_RUNGS|HASEN_GRID_MIXED_CG_HEATBATH_RUNGS)
                      [[ "$val" =~ ^([0-9](,[0-9])*)?$ ]] || die "$key must be a comma list of rung numbers or empty"
                      printf -v "$key" '%s' "$val"; route_given=1 ;;
    *)                die "KEY '$key' is not allowed (see the header of this script)" ;;
  esac
done
{ [ -n "${DIAG_START_CFG:-}" ] && [ -z "${DIAG_START_TRAJ:-}" ]; } && die "DIAG_START_CFG needs DIAG_START_TRAJ"
{ [ -z "${DIAG_START_CFG:-}" ] && [ -n "${DIAG_START_TRAJ:-}" ]; } && die "DIAG_START_TRAJ needs DIAG_START_CFG"

# ---- ladder ---------------------------------------------------------------------------------
[ "$mode" = cresume ] && ladder=-0.2416,-0.1870
LADDER_CHECK=$(awk -v l="$ladder" -v maxm="$MAX_MASSES" 'BEGIN {
  n = split(l, m, ",")
  if (n < 2 || n > maxm) { print "ladder needs 2 to " maxm " masses, got " n; exit }
  for (i = 1; i <= n; i++) {
    if (m[i] !~ /^[+-]?[0-9]*\.[0-9]+$/) { print "mass " i " is not a decimal number: " m[i]; exit }
    if (i > 1 && m[i] + 0 <= m[i-1] + 0) { print "masses must increase (lightest first): " m[i-1] " then " m[i]; exit }
  }
  print "ok"
}')
[ "$LADDER_CHECK" = ok ] || die "$LADDER_CHECK"

# ---- solver routes from the ladder (unless given) ---------------------------------------------
if [ "$route_given" = 0 ]; then
  read -r R_MG R_MGHB R_MX R_MXHB < <(awk -v l="$ladder" -v b="$MG_BELOW" 'BEGIN {
    n = split(l, m, ",")
    for (k = 0; k < n - 1; k++) {
      lo = m[k+1] + 0; hi = m[k+2] + 0
      if (lo < b) mg = mg (mg == "" ? "" : ",") k; else mx = mx (mx == "" ? "" : ",") k
      if (lo < b && hi < b) mghb = mghb (mghb == "" ? "" : ",") k; else mxhb = mxhb (mxhb == "" ? "" : ",") k
    }
    printf "%s %s %s %s\n", (mg == "" ? "-" : mg), (mghb == "" ? "-" : mghb), (mx == "" ? "-" : mx), (mxhb == "" ? "-" : mxhb)
  }')
  for v in R_MG R_MGHB R_MX R_MXHB; do [ "${!v}" = "-" ] && printf -v "$v" '%s' ""; done
  HASEN_GRID_MG_RUNGS=$R_MG HASEN_GRID_MG_HEATBATH_RUNGS=$R_MGHB
  HASEN_GRID_MIXED_CG_RUNGS=$R_MX HASEN_GRID_MIXED_CG_HEATBATH_RUNGS=$R_MXHB
fi

# ---- layout -> integrator levels ---------------------------------------------------------------
[ "$mode" = cresume ] && layout=12/12/48/48
LEVELS=$(awk -v lay="$layout" 'BEGIN {
  if (split(lay, s, "/") != 4) { print "ERR layout must be L/S/T/G, got " lay; exit }
  for (i = 1; i <= 4; i++) if (s[i] !~ /^[1-9][0-9]*$/) { print "ERR layout entries must be positive whole numbers: " lay; exit }
  # distinct step counts, ascending
  nd = 0
  for (i = 1; i <= 4; i++) { seen = 0; for (j = 1; j <= nd; j++) if (d[j] == s[i]) seen = 1; if (!seen) d[++nd] = s[i] + 0 }
  for (i = 1; i <= nd; i++) for (j = i + 1; j <= nd; j++) if (d[j] < d[i]) { t = d[i]; d[i] = d[j]; d[j] = t }
  mult[1] = 1; mult[2] = 1; mult[3] = 1
  for (i = 2; i <= nd; i++) {
    if (d[i] % (2 * d[i-1]) != 0) { print "ERR " d[i] " is not 2 x an integer x " d[i-1] " (levels must nest)"; exit }
    mult[i-1] = d[i] / (2 * d[i-1])
  }
  for (i = 1; i <= 4; i++) for (j = 1; j <= nd; j++) if (d[j] == s[i]) lv[i] = j - 1
  printf "%d %d %d %d %d %d %d %d\n", d[1], lv[1], lv[2], lv[3], lv[4], mult[1], mult[2], mult[3]
}')
case "$LEVELS" in ERR*) die "${LEVELS#ERR }" ;; esac
read -r MDSTEPS LV_L LV_S LV_T LV_G MULT_1 MULT_2 MULT_3 <<< "$LEVELS"

# ---- Chroma: names and paths (the run itself is set up after the time checks) ----------------
if [ "$CHROMA" = 1 ]; then
  [ "$mode" = cresume ] && MDSTEPS=$(grep -m1 -o '<n_steps>[0-9]*</n_steps>' "$restart" | tr -dc 0-9)
  [[ "$MDSTEPS" =~ ^[1-9][0-9]*$ ]] || die "cannot read the outer n_steps of $restart"
  TAG=at2${CHAIN}_${name}
  C_INPUT=$CHROMA_INPUTS/hmc_baseg_mds${MDSTEPS}_${TAG}.ini.xml
  C_RUNDATE=$(date +%Y_%-m_%-d)
  C_OUT=$ROOT/runs/${C_RUNDATE}_chroma_baseg_mds${MDSTEPS}_${TAG}
  C_SAVE=$CHROMA_SAVES/$TAG
  [ ! -e "$C_INPUT" ] || die "input $C_INPUT exists: pick a new name (never overwrite)"
  [ ! -e "$C_OUT" ] || die "run dir $C_OUT exists: pick a new name (never overwrite)"
  if [ "$mode" = ctraj ] && [ -d "$C_SAVE" ] && [ -n "$(ls -A "$C_SAVE" 2> /dev/null)" ]; then
    die "save dir $C_SAVE is not empty: pick a new name"
  fi
fi

# ---- lanes: which of the job's nodes this run uses ------------------------------------------------
JOB_NODELIST=${SLURM_JOB_NODELIST:-$(squeue -h -j "$SLURM_JOB_ID" -o %N 2> /dev/null | head -1)}
mapfile -t HOSTS < <(scontrol show hostnames "$JOB_NODELIST" 2> /dev/null)
case "${#HOSTS[@]}" in
  4) [ "${LANE:-1}" = 1 ] || die "this job has 4 nodes: one run at a time on all of them (no LANE)"
     LANE="" ;;
  8) [ -n "${LANE:-}" ] || die "this job has 8 nodes: give LANE=1 (nodes 1-4), LANE=2 (nodes 5-8) or, Chroma only, LANE=all" ;;
  *) die "this job has ${#HOSTS[@]} nodes ($JOB_NODELIST); at2_run.sh knows 4 (one run) or 8 (two lanes of 4)" ;;
esac
RUN_NODELIST="" LANES=""
case "${LANE:-}" in
  1)   RUN_NODELIST=$(IFS=,; echo "${HOSTS[*]:0:4}") LANES=1 ;;
  2)   RUN_NODELIST=$(IFS=,; echo "${HOSTS[*]:4:4}") LANES=2 ;;
  all) [ "$CHROMA" = 1 ] || die "LANE=all is for Chroma only: the Grid launchers run 4 nodes, layout 1.2.2.4"
       RUN_NODELIST=$(IFS=,; echo "${HOSTS[*]}") LANES="1 2" ;;
esac
LANE_NOTE=${LANE:+ lane=$LANE nodes=$RUN_NODELIST}

# AT2_DRY=1: print what would be passed to the launcher and stop (no job or time checks).
[ "${AT2_DRY:-0}" = 1 ] && [ -n "${LANE:-}" ] && echo "DRY lane $LANE nodes $RUN_NODELIST"
if [ "${AT2_DRY:-0}" = 1 ] && [ "$CHROMA" = 1 ]; then
  if [ "$mode" = ctraj ]; then
    echo "DRY ctraj TAG=$TAG ladder=$ladder layout=$layout n_traj=$n_traj seed=$seed"
    echo "DRY outer steps $MDSTEPS (levels light=$LV_L strange=$LV_S tail=$LV_T gauge=$LV_G mult=$MULT_1,$MULT_2,$MULT_3)"
    echo "DRY saves $C_SAVE"
  else
    echo "DRY cresume TAG=$TAG restart=$restart n_traj=$n_traj outer steps $MDSTEPS (from the restart; saves continue under its SavePrefix)"
  fi
  echo "DRY input $C_INPUT"
  echo "DRY run dir $C_OUT"
  exit 0
fi
if [ "${AT2_DRY:-0}" = 1 ]; then
  echo "DRY $mode RUN=at2${CHAIN}_${name} ladder=$ladder layout=$layout n_traj=$n_traj seed=$seed prev=${prev:-none}"
  echo "DRY routes mg=${HASEN_GRID_MG_RUNGS:-none} mg_hb=${HASEN_GRID_MG_HEATBATH_RUNGS:-none} mixed=${HASEN_GRID_MIXED_CG_RUNGS:-none} mixed_hb=${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}"
  echo "DRY levels MDSTEPS=$MDSTEPS light=$LV_L strange=$LV_S tail=$LV_T gauge=$LV_G mult=$MULT_1,$MULT_2,$MULT_3"
  exit 0
fi

# ---- one run at a time (per lane in an 8-node job), enough time left ----------------------------
if [ -z "${LANE:-}" ]; then
  if pgrep -u "$(id -u)" -f "forces48_puregrid.sh|traj48_puregrid.sh|chroma_baseg_mds.sh" > /dev/null; then
    die "another run is active in this job (a launcher is running); wait for it to finish"
  fi
else
  # A lane is held by the process id in its file while that process is this wrapper or a launcher
  # (the wrapper execs the launcher, so the id stays the run's to the end). The check and the claim
  # happen under a lock; a run refused after its claim leaves a dead id, which frees the lane.
  LANE_DIR=${AT2_LANE_DIR:-/tmp/at2_lanes_${SLURM_JOB_ID}}   # AT2_LANE_DIR: tests only
  mkdir -p "$LANE_DIR" || die "cannot make $LANE_DIR"
  exec 9> "$LANE_DIR/claim.lock"
  flock -w 20 9 || die "cannot lock $LANE_DIR/claim.lock"
  for l in $LANES; do
    pid=$(cat "$LANE_DIR/lane$l.pid" 2> /dev/null)
    if [[ "$pid" =~ ^[0-9]+$ ]] && grep -qaE "at2_run.sh|forces48_puregrid.sh|traj48_puregrid.sh|chroma_baseg_mds.sh" "/proc/$pid/cmdline" 2> /dev/null; then
      die "lane $l is busy (process $pid: $(tr '\0' ' ' < "/proc/$pid/cmdline" 2> /dev/null | cut -c1-160)); wait for it to finish"
    fi
  done
  for l in $LANES; do echo "$$" > "$LANE_DIR/lane$l.pid"; done
  exec 9>&-
  [ -n "${AT2_JOBDIR:-}" ] && echo "$(date '+%F %T') pid=$$ lane=$LANE nodes=$RUN_NODELIST $mode $name" >> "$AT2_JOBDIR/lanes.log"
fi
to_min() { awk -v t="$1" 'BEGIN { d = 0
  if (t !~ /^[0-9]/) { print -1; exit }
  if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
  n = split(t, p, ":")
  if (n == 3) m = p[1] * 60 + p[2]; else if (n == 2) m = p[1]; else m = 0
  printf "%d\n", d * 1440 + m }'; }
LEFT=$(to_min "$(squeue -h -j "$SLURM_JOB_ID" -o %L 2> /dev/null | head -1)")
[ "$LEFT" -ge 0 ] 2> /dev/null || die "cannot read the time left of job $SLURM_JOB_ID"
if [ -z "${EST_MIN:-}" ]; then
  if [ "$mode" = fo ]; then
    [ "${FO_PROFILE:-production}" = screen ] && EST_MIN=14 || EST_MIN=22
  elif [ "$CHROMA" = 1 ]; then
    EST_MIN=$(( n_traj * 30 + 10 ))
  else
    EST_MIN=$(( n_traj * 70 + 10 ))
  fi
fi
[ $(( EST_MIN + 3 )) -le "$LEFT" ] || die "not enough time: the run needs ~${EST_MIN} min + 3, the job has ${LEFT} min left"

# ---- Chroma launch -----------------------------------------------------------------------------
if [ "$CHROMA" = 1 ]; then
  echo "at2_run: $mode TAG=$TAG n_traj=$n_traj est=${EST_MIN}min left=${LEFT}min$LANE_NOTE"
  if [ "$mode" = ctraj ]; then
    mkdir -p "$C_SAVE" || die "cannot make the save dir $C_SAVE"
    python3 -I "$CHROMA_GEN" new --out "$C_INPUT" --tag "baseg_mds${MDSTEPS}_${TAG}" --ladder="$ladder" \
      --layout "$layout" --ntraj "$n_traj" --seed "$seed" --save-dir "$C_SAVE" \
      || die "the Chroma input was not made or does not match the request: nothing run"
  else
    python3 -I "$CHROMA_GEN" resume --out "$C_INPUT" --restart "$restart" --ntraj "$n_traj" \
      || die "the Chroma resume input was not made: nothing run"
  fi
  unset JOBID LAUNCH_PREFIX
  export MDS=$MDSTEPS SUFFIX=_$TAG RUNDATE=$C_RUNDATE CHROMA_EXTRA_ARGS="-poolsize 20g" CHROMA_TIMEOUT_MIN=$(( LEFT - 3 )) CHROMA_HMC_BIN
  if [ -n "${LANE:-}" ]; then
    export CHROMA_NODELIST=$RUN_NODELIST
    [ "$LANE" = all ] && export CHROMA_NODES=8 CHROMA_GEOM="2 2 2 4"
  fi
  echo "at2_run: launching Chroma, run dir $C_OUT, CHROMA_TIMEOUT_MIN=$CHROMA_TIMEOUT_MIN"
  exec bash "$CHROMA_LAUNCHER"
fi

# ---- launch ------------------------------------------------------------------------------------
export BIN RUN="at2${CHAIN}_${name}" LADDER_PROFILE=custom CUSTOM_LADDER="$ladder"
export HASEN_GRID_MG_RUNGS HASEN_GRID_MG_HEATBATH_RUNGS HASEN_GRID_MIXED_CG_RUNGS HASEN_GRID_MIXED_CG_HEATBATH_RUNGS
export DEVICE_MEM_MB=8000
[ -z "${LANE:-}" ] || export SRUN_EXTRA="-w $RUN_NODELIST"

echo "at2_run: $mode RUN=$RUN ladder=$ladder layout=$layout n_traj=$n_traj seed=$seed est=${EST_MIN}min left=${LEFT}min$LANE_NOTE"
echo "at2_run: routes mg=${HASEN_GRID_MG_RUNGS:-none} mg_hb=${HASEN_GRID_MG_HEATBATH_RUNGS:-none} mixed=${HASEN_GRID_MIXED_CG_RUNGS:-none} mixed_hb=${HASEN_GRID_MIXED_CG_HEATBATH_RUNGS:-none}"

if [ "$mode" = fo ]; then
  # Layout is inert in FORCES_ONLY; fo8_base's placeholder layout keeps the ENV lines comparable.
  export CUSTOM_TAIL_LEVEL=inner CUSTOM_MDSTEPS=12
  export FO_PROFILE=${FO_PROFILE:-production} FORCES_SAMPLES=${FORCES_SAMPLES:-9}
  exec bash "$HB/perlmutter/forces48_puregrid.sh"
fi

export CUSTOM_MDSTEPS=$MDSTEPS
export CUSTOM_LEVEL_LIGHT=$LV_L CUSTOM_LEVEL_STRANGE=$LV_S CUSTOM_LEVEL_TAIL=$LV_T CUSTOM_LEVEL_GAUGE=$LV_G
export CUSTOM_LEVEL_MULT_1=$MULT_1 CUSTOM_LEVEL_MULT_2=$MULT_2 CUSTOM_LEVEL_MULT_3=$MULT_3
export N_TRAJ=$n_traj TIMEOUT_MIN=$(( LEFT - 3 ))
echo "at2_run: levels MDSTEPS=$MDSTEPS light=$LV_L strange=$LV_S tail=$LV_T gauge=$LV_G mult=$MULT_1,$MULT_2,$MULT_3 TIMEOUT_MIN=$TIMEOUT_MIN"
if [ "$mode" = resume ]; then
  export RESUME_FROM=$prev
else
  export HMC_SEED_OFFSET=$seed
  [ -z "${DIAG_START_CFG:-}" ] || export DIAG_START_CFG DIAG_START_TRAJ
fi
exec bash "$HB/perlmutter/traj48_puregrid.sh"
