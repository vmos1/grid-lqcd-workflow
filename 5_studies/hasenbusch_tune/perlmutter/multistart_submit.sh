#!/bin/bash
# multistart_submit.sh -- submit one candidate ARM of campaign multistart-validation: 9 regular-QOS
# batch jobs, one per production start, 5 trajectories each, HMC_SEED_OFFSET=500, TRAJL 1.0, four
# 40 GB nodes (traj48_puregrid_batch.sh), paired start by start with the baseline arm run in round 1
# (2026-11-01; the baseline is not rerun). Procedure, verdict rule, round log:
# __docs/2026_10_28_handoff_multistart_validation.md. Analysis: data_extraction_scripts/regen_multistart.sh.
#
# Usage (DRY=1 prints the 9 commands and submits nothing; the global rule to ask before sbatch holds):
#   [DRY=1] bash multistart_submit.sh <arm> <HH:MM:SS> KEY=VALUE [KEY=VALUE ...]
# <arm>       run names become <arm>_<stream>_<traj>_ms500; letters, digits and _ only
# <HH:MM:SS>  --time per job: 5 x the arm's slowest expected full trajectory + ~30 min (the
#             baseline needed up to 5 h 46 min, so 06:30:00)
# KEY=VALUE   the launcher settings that define the arm, copied from the ENV line of the
#             candidate's own tuning run (LADDER_PROFILE, CUSTOM_*, BIN, ...). Commas are fine: they
#             go on the env prefix, not through --export. The solver routes default to round 1's
#             (MG 0,1,2; MG heatbath 0,1; mixed CG 3; mixed-CG heatbath 2,3) unless given here.
#             RUN, N_TRAJ, HMC_SEED_OFFSET, DIAG_TRAJL, DIAG_START_CFG, DIAG_START_TRAJ are this
#             script's and refused: they are what makes the pairing.
# Round 1, as this script would submit it:
#   baseline 06:30:00 LADDER_PROFILE=baseG
#   B2_t24   06:00:00 LADDER_PROFILE=custom CUSTOM_LADDER=-0.2416,-0.2400,-0.2320,-0.2180,0.1000
#            CUSTOM_LEVEL_LIGHT=0 CUSTOM_LEVEL_STRANGE=0 CUSTOM_LEVEL_TAIL=1 CUSTOM_LEVEL_GAUGE=2
#            CUSTOM_LEVEL_MULT_1=1 CUSTOM_LEVEL_MULT_2=1 CUSTOM_MDSTEPS=12 BIN=<bin>/..._stock_m9
# Starts on $PSCRATCH/cfgs48/ (purgeable; refetch with runs/2026_10_26_fo_cfgs_driver/get_cfgs.hsi).
# Every submitted job is appended to runs/multistart_jobs.txt (job id, run, time, settings).
BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
HB=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune
BATCH=$HB/perlmutter/traj48_puregrid_batch.sh
CFGS=/pscratch/sd/v/vpa/cfgs48
ENS=cl21_48_96_b6p3_m0p2416_m0p2050
SEED=500
JOBLOG=$BASE/runs/multistart_jobs.txt
# Noisy rung-1 starts first (2026_10_26_force_only_config_spread.md §6), then the rest.
STARTS="djm-3:1000 djm-1:1800 djm-2:1400 djm-1:1000 djm-1:2600 djm-2:700 djm-2:2100 djm-3:2700 djm-3:3300"

die() { echo "multistart_submit: $*" >&2; exit 1; }
[ $# -ge 3 ] || die "usage: [DRY=1] bash multistart_submit.sh <arm> <HH:MM:SS> KEY=VALUE [KEY=VALUE ...]"
ARM=$1 TIME=$2
shift 2
[[ "$ARM" =~ ^[A-Za-z0-9_]+$ ]] || die "arm '$ARM': letters, digits and _ only"
[[ "$ARM" == *_djm* ]] && die "arm '$ARM' must not contain '_djm' (the analysis splits labels there)"
[[ "$TIME" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || die "time '$TIME' is not HH:MM:SS"
[ "$ARM" != baseline ] || [ -n "${DRY:-}" ] || die "the baseline arm ran in round 1; rerun it only on purpose (new binary or solver), and under another arm name"

ROUTES=(HASEN_GRID_MG_RUNGS=0,1,2 HASEN_GRID_MG_HEATBATH_RUNGS=0,1 HASEN_GRID_MIXED_CG_RUNGS=3 HASEN_GRID_MIXED_CG_HEATBATH_RUNGS=2,3)
SETTINGS=()
for kv in "$@"; do
  [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=.*$ ]] || die "setting '$kv' is not KEY=VALUE"
  key=${kv%%=*}
  case $key in
    RUN|N_TRAJ|HMC_SEED_OFFSET|DIAG_TRAJL|DIAG_START_CFG|DIAG_START_TRAJ) die "$key is set by this script (it makes the pairing)" ;;
  esac
  for i in "${!ROUTES[@]}"; do [ "${ROUTES[$i]%%=*}" = "$key" ] && unset 'ROUTES[i]'; done
  SETTINGS+=("$kv")
done
[ -n "$(printf '%s\n' "${SETTINGS[@]}" | grep '^LADDER_PROFILE=')" ] || die "no LADDER_PROFILE given; copy the arm's settings from its run's ENV line"

for start in $STARTS; do
  stream=${start%:*} traj=${start#*:}
  cfg=$CFGS/${ENS}-${stream}_cfg_${traj}.lime
  run=${ARM}_${stream/-/}_${traj}_ms$SEED
  [ -f "$cfg" ] || die "missing start config $cfg (refetch with runs/2026_10_26_fo_cfgs_driver/get_cfgs.hsi); nothing more submitted"
  cmd=(env RUN=$run N_TRAJ=5 HMC_SEED_OFFSET=$SEED DIAG_TRAJL=1.0 DIAG_START_CFG=$cfg DIAG_START_TRAJ=$traj
       "${ROUTES[@]}" "${SETTINGS[@]}" sbatch --time=$TIME -J ${ARM}_${stream/-/}_${traj} --export=ALL $BATCH)
  echo "== $run"
  if [ -n "${DRY:-}" ]; then
    echo "${cmd[*]}"
  else
    out=$("${cmd[@]}") || die "sbatch failed for $run: $out"
    echo "$out"
    echo "$(date '+%Y-%m-%d %H:%M') ${out##* } $run time=$TIME ${SETTINGS[*]}" >> "$JOBLOG"
  fi
done
