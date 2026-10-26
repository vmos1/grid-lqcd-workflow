#!/bin/bash
# =============================================================================
# gate16_levels_m9.sh -- the 16^3 gate of the general integrator-level assignment
# (driver HASEN_LEVEL_* family, binary _stock_m9, handoff
# __docs/2026_10_13_handoff_general_level_driver.md task 1). Eight one-trajectory
# smoke16_puregrid.sh runs inside an EXISTING 1-node allocation, then a pass/fail
# table. This script never allocates and never submits.
#
# Usage (one line):
#   SLURM_JOB_ID=<id> bash gate16_levels_m9.sh
# Optional: M8=<old binary> M9=<new binary> TAG=<suffix for the run names, e.g. _r2>
#           GATE_CASE_MIN=<per-case cap in minutes, default 20>
# The allocation (1 node, interactive QOS) comes from `bash ~/.claude/bin/alloc_gpu.sh 1 2`
# and is released with `release_alloc.sh <id>` (machines/perlmutter.md). smoke16 refuses to
# overwrite a run directory, so a rerun on the same day needs a TAG.
#
# Cases (all: 16^3x48, ladder -0.245,-0.20,-0.10, MDSTEPS 4, seed 200, the smoke recipe):
#   a1  m8  default            2-level, tail outer, gauge x2            (legacy knobs)
#   a2  m9  default            same                                     (legacy knobs)
#   a3  m9  LEVEL light0 strange0 tail0 gauge1 mult_1=2                 (level family)
#   b1  m8  legacy 3-level     strange middle x1, tail middle, gauge x2 (legacy knobs)
#   b2  m9  same                                                        (legacy knobs)
#   b3  m9  LEVEL light0 strange1 tail1 gauge2 mults 1,2                (level family)
#   c   m9  LEVEL light0 strange1 tail2 gauge3 mults 1,1,1   4 levels: 4/8/16/32 steps
#   d   m9  LEVEL light0 strange0 tail1 gauge2 mults 1,1     the campaign layout: 4/8/16
# Pass criteria: a1 = a2 = a3 and b1 = b2 = b3 bit-exact on dH and on both "Plaquette: [ 1 ]"
# values (thin and smeared); c and d: force calls per level = 3n+1 (ForceGradient) for
# n = 4, 8, 16, 32 (c) and 4, 8, 16 (d), and |dH| < 1. Force calls per level are counted
# from the "[l][0] P update elapsed time" lines (GridLogMessage, one per update_P call on
# level l, action 0); the Integrator-channel "update_P : Level [l][0]" count is recorded too.
#
# Output: runs/<date>_gate16_<case><TAG>/ per case (smoke16's layout) and the summary
# runs/<date>_gate16_summary<TAG>.txt (appended per case, the table at the end). Each
# case also keeps smoke16's stdout as runs/<date>_gate16_summary<TAG>.<case>.out.
#
# Time: each smoke16 run is minutes on one node (its own timeout is 60 min); this script
# caps a case at GATE_CASE_MIN and goes on to the next one, so the eight cases fit a
# 2-hour allocation. On a cap the whole process group of the case (bash, timeout, srun)
# gets TERM then KILL, and the script waits for the job's steps to clear before continuing.
# =============================================================================
set -o pipefail

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
HB=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune
SMOKE=$HB/perlmutter/smoke16_puregrid.sh
M8=${M8:-$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m8}
M9=${M9:-$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m9}
TAG=${TAG:-}
GATE_CASE_MIN=${GATE_CASE_MIN:-20}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
USAGE="usage: SLURM_JOB_ID=<id> bash $0"

# ---- 1. Inputs -----------------------------------------------------------------
[ -n "${SLURM_JOB_ID:-}" ] || die "SLURM_JOB_ID is not set (the id of an existing 1-node allocation). $USAGE"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be a numeric job id, got '$SLURM_JOB_ID'"
[ -f "$SMOKE" ] || die "smoke launcher not found: $SMOKE"
{ [ -f "$M8" ] && [ -x "$M8" ]; } || die "M8 is not an executable file: $M8"
{ [ -f "$M9" ] && [ -x "$M9" ]; } || die "M9 is not an executable file: $M9"
[ "$(grep -c -a -F HASEN_LEVEL_LIGHT "$M9" || true)" -gt 0 ] || die "$M9 does not contain HASEN_LEVEL_LIGHT (not an m9-or-later build)"
[ -z "$TAG" ] || [[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || die "TAG may contain only letters, digits, '.', '_' and '-', got '$TAG'"
[[ "$GATE_CASE_MIN" =~ ^[1-9][0-9]*$ ]] || die "GATE_CASE_MIN must be a positive integer (minutes), got '$GATE_CASE_MIN'"
JOB_STATE=$(squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null | head -1)
[ "$JOB_STATE" = RUNNING ] || die "allocation $SLURM_JOB_ID is not RUNNING (state: '${JOB_STATE:-not found}'). This script never allocates."

# No layout override may leak from this shell into a case: every case passes its own.
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(CUSTOM_|DIAG_)' || true)

DATE_TAG=$(date +%Y_%-m_%-d)
SUMMARY=$BASE/runs/${DATE_TAG}_gate16_summary${TAG}.txt
mkdir -p "$BASE/runs" || die "cannot create $BASE/runs"
{
  echo "==== gate16_levels_m9 $(date '+%Y-%m-%d %H:%M %Z') job $SLURM_JOB_ID node $(squeue -h -j "$SLURM_JOB_ID" -o %N)"
  echo "M8=$M8"
  echo "M9=$M9"
  echo "M8_SHA256=$(sha256sum "$M8" | cut -d' ' -f1)"
  echo "M9_SHA256=$(sha256sum "$M9" | cut -d' ' -f1)"
} >> "$SUMMARY"

# ---- 2. Helpers ----------------------------------------------------------------
# Wait until the allocation has no running steps (after a capped case), at most 3 min.
wait_steps() {
  local i
  for i in $(seq 1 18); do
    [ -n "$(squeue -h -s -j "$SLURM_JOB_ID" 2>/dev/null)" ] || return 0
    sleep 10
  done
  echo "WARNING: steps of job $SLURM_JOB_ID still listed after 3 min; continuing" >&2
  return 0
}

# Per-case facts, parsed from hmc.log into the CASE_* variables.
declare -A CASE_RC CASE_DH CASE_PT CASE_PS CASE_CALLS CASE_IPCALLS CASE_BATCH CASE_LEVELS CASE_DIR
parse_case() {   # name log
  local name=$1 log=$2 l n
  CASE_DH[$name]=$(grep -a 'Total H after trajectory' "$log" | sed -n 's/.*dH = *//p' | head -1 | tr -d ' ')
  CASE_PT[$name]=$(grep -a 'Plaquette: \[ 1 \]' "$log" | sed -n 1p | awk '{print $NF}')
  CASE_PS[$name]=$(grep -a 'Plaquette: \[ 1 \]' "$log" | sed -n 2p | awk '{print $NF}')
  CASE_LEVELS[$name]=$(grep -a '\[Levels\]' "$log" | sed 's/^.*\[Levels\]/[Levels]/' | tr '\n' '|')
  CASE_CALLS[$name]=
  CASE_IPCALLS[$name]=
  CASE_BATCH[$name]=
  for l in 0 1 2 3; do
    n=$(grep -a -c "\[$l\]\[0\] P update elapsed time" "$log")
    [ "$n" -gt 0 ] || break
    CASE_CALLS[$name]+="L$l=$n "
    CASE_IPCALLS[$name]+="L$l=$(grep -a -c "update_P : Level \[$l\]\[0\]" "$log") "
    CASE_BATCH[$name]+="L$l=$(grep -a -c "\[$l\] batched smeared P update" "$log") "
  done
}

# run_case <name> <binary> [VAR=VALUE ...]: one smoke16 run, capped, then parsed + appended.
run_case() {
  local name=$1 bin=$2
  shift 2
  local run=gate16_${name}${TAG}
  local rundir=$BASE/runs/${DATE_TAG}_${run}
  local log=$rundir/hmc.log
  local out=$SUMMARY.$name.out
  local t0 t1 rc
  echo "=== case $name: $(basename "$bin") $* -> $rundir"
  t0=$(date +%s)
  env "$@" SLURM_JOB_ID="$SLURM_JOB_ID" BIN="$bin" RUN="$run" timeout -k 60s "${GATE_CASE_MIN}m" bash "$SMOKE" > "$out" 2>&1
  rc=$?
  t1=$(date +%s)
  [ "$rc" -ne 124 ] || { echo "case $name hit the ${GATE_CASE_MIN}-min cap" >&2; wait_steps; }
  CASE_RC[$name]=$rc
  CASE_DIR[$name]=$rundir
  if [ -f "$log" ]; then
    parse_case "$name" "$log"
  else
    CASE_DH[$name]= CASE_PT[$name]= CASE_PS[$name]= CASE_CALLS[$name]= CASE_IPCALLS[$name]= CASE_BATCH[$name]= CASE_LEVELS[$name]=
  fi
  {
    echo "---- case $name  bin=$(basename "$bin")  overrides: $*"
    echo "rundir=$rundir  exit=$rc  wall_s=$(( t1 - t0 ))"
    if [ -f "$log" ]; then
      grep -a -E '^ENV ' "$log"
      grep -a -E 'Integrator: |\[Ladder\]|\[Levels\]' "$log"
      grep -a -E 'Total H before trajectory|Total H after trajectory|Metropolis_test|Plaquette: \[|Total time for trajectory' "$log"
      echo "force_calls_per_level(P update elapsed time [l][0]): ${CASE_CALLS[$name]}"
      echo "force_calls_per_level(update_P : Level [l][0]):      ${CASE_IPCALLS[$name]}"
      echo "batched_smeared_pullbacks_per_level:                ${CASE_BATCH[$name]}"
      grep -a -E 'exit=' "$log" | tail -1
    else
      echo "no hmc.log (smoke16 refused or died before launching; see $out)"
      tail -5 "$out"
    fi
  } >> "$SUMMARY"
}

# ---- 3. The eight cases --------------------------------------------------------
run_case a1 "$M8"
run_case a2 "$M9"
run_case a3 "$M9" CUSTOM_LEVEL_LIGHT=0 CUSTOM_LEVEL_STRANGE=0 CUSTOM_LEVEL_TAIL=0 CUSTOM_LEVEL_GAUGE=1 CUSTOM_LEVEL_MULT_1=2
run_case b1 "$M8" CUSTOM_STRANGE_LEVEL=middle CUSTOM_STRANGE_INNER_MULT=1 CUSTOM_TAIL_LEVEL=middle
run_case b2 "$M9" CUSTOM_STRANGE_LEVEL=middle CUSTOM_STRANGE_INNER_MULT=1 CUSTOM_TAIL_LEVEL=middle
run_case b3 "$M9" CUSTOM_LEVEL_LIGHT=0 CUSTOM_LEVEL_STRANGE=1 CUSTOM_LEVEL_TAIL=1 CUSTOM_LEVEL_GAUGE=2 CUSTOM_LEVEL_MULT_1=1 CUSTOM_LEVEL_MULT_2=2
run_case c  "$M9" CUSTOM_LEVEL_LIGHT=0 CUSTOM_LEVEL_STRANGE=1 CUSTOM_LEVEL_TAIL=2 CUSTOM_LEVEL_GAUGE=3 CUSTOM_LEVEL_MULT_1=1 CUSTOM_LEVEL_MULT_2=1 CUSTOM_LEVEL_MULT_3=1
run_case d  "$M9" CUSTOM_LEVEL_LIGHT=0 CUSTOM_LEVEL_STRANGE=0 CUSTOM_LEVEL_TAIL=1 CUSTOM_LEVEL_GAUGE=2 CUSTOM_LEVEL_MULT_1=1 CUSTOM_LEVEL_MULT_2=1

# ---- 4. Verdicts -----------------------------------------------------------------
same3() {   # three case names: bit-exact dH + both plaquettes, all non-empty
  local a=$1 b=$2 c=$3
  [ -n "${CASE_DH[$a]}" ] && [ -n "${CASE_PT[$a]}" ] && [ -n "${CASE_PS[$a]}" ] || return 1
  [ "${CASE_DH[$a]}" = "${CASE_DH[$b]}" ] && [ "${CASE_DH[$a]}" = "${CASE_DH[$c]}" ] || return 1
  [ "${CASE_PT[$a]}" = "${CASE_PT[$b]}" ] && [ "${CASE_PT[$a]}" = "${CASE_PT[$c]}" ] || return 1
  [ "${CASE_PS[$a]}" = "${CASE_PS[$b]}" ] && [ "${CASE_PS[$a]}" = "${CASE_PS[$c]}" ] || return 1
  return 0
}
calls_ok() {   # case name, then the expected steps per level: calls must be 3n+1 each, no extra level
  local name=$1 expect n l=0
  shift
  expect=
  for n in "$@"; do
    expect+="L$l=$(( 3 * n + 1 )) "
    l=$(( l + 1 ))
  done
  [ "${CASE_CALLS[$name]}" = "$expect" ]
}
dh_sane() {   # |dH| < 1 and numeric
  local v=${CASE_DH[$1]}
  [ -n "$v" ] || return 1
  awk -v x="$v" 'BEGIN { if (x + 0 != x) exit 1; if (x < 0) x = -x; exit (x < 1) ? 0 : 1 }'
}
verdict() { if "$@"; then echo PASS; else echo FAIL; fi; }

V_A=$(verdict same3 a1 a2 a3)
V_B=$(verdict same3 b1 b2 b3)
V_C=$(verdict calls_ok c 4 8 16 32)
V_C2=$(verdict dh_sane c)
V_D=$(verdict calls_ok d 4 8 16)
V_D2=$(verdict dh_sane d)
{
  echo "==== gate16 table $(date '+%Y-%m-%d %H:%M %Z')"
  printf '%-4s %-5s %-22s %-20s %-20s %s\n' case exit dH plaq_thin plaq_smeared force_calls_per_level
  for k in a1 a2 a3 b1 b2 b3 c d; do
    printf '%-4s %-5s %-22s %-20s %-20s %s\n' "$k" "${CASE_RC[$k]:-?}" "${CASE_DH[$k]:-?}" "${CASE_PT[$k]:-?}" "${CASE_PS[$k]:-?}" "${CASE_CALLS[$k]:-?}"
  done
  echo "(a) 2-level: m8 default = m9 default = m9 level family, bit-exact dH + plaquettes: $V_A"
  echo "(b) 3-level: m8 legacy  = m9 legacy  = m9 level family, bit-exact dH + plaquettes: $V_B"
  echo "(c) 4-level light0/strange1/tail2/gauge3: force calls 13/25/49/97: $V_C   |dH| < 1: $V_C2"
  echo "(d) light+strange0/tail1/gauge2:          force calls 13/25/49:    $V_D   |dH| < 1: $V_D2"
  echo "run dirs:"
  for k in a1 a2 a3 b1 b2 b3 c d; do echo "  $k ${CASE_DIR[$k]:-?}"; done
} | tee -a "$SUMMARY"
echo "summary: $SUMMARY"
ALL=0
for v in "$V_A" "$V_B" "$V_C" "$V_C2" "$V_D" "$V_D2"; do [ "$v" = PASS ] || ALL=1; done
exit "$ALL"
