#!/bin/bash
# Regenerate the multistart-validation comparison (every arm against the baseline, each from 9
# production configurations, 5 trajectories per start, seed 500) from the raw HMC logs
# (2026-11-02). Modelled on regen_tune1_b_comparison.sh.
#
# This script IS the manifest: which run directory is which arm and start. Runs are submitted by
# perlmutter/multistart_submit.sh (round 1 by runs/2026_10_30_multistart_driver/submit_ms.sh); a
# run directory is dated by the day its job started. To add a candidate, append its 9 lines below
# (label <arm>_<stream>_<start trajectory>); the arm list is read from the labels, baseline first.
# A start finished in a second segment (after a stall: DIAG_START_CFG=<ckpt>, new seed) is
# label=dir1,dir2 in trajectory order. A start rerun from its start with a new seed after a stall
# (amended rule, 2026-11-12) takes the rerun in RUNS and the stalled run in STALLED: the stall is
# reported as cleared. Two such reruns so far: B2_t24 on djm-3 1000 (round 1, rerun 11-13) and D_8
# on djm-2 700 (round 2, rerun 11-09).
#
# Worked output: __docs/2026_11_02_multistart_b2t24_vs_baseline.md (round 1),
# __docs/2026_11_09_multistart_round2_tune2.md (round 2). Current outdir
# runs/2026_11_13_multistart_analysis (amended rule, both reruns); runs/2026_11_12_multistart_analysis
# lacks B2_t24's rerun, and runs/2026_11_9_multistart_analysis predates the amendment.
#
# Outputs (outdir default runs/2026_11_2_multistart_analysis, outside the git tree; a cache):
#   ms.csv            per-trajectory CSV (compare_runs_multiwindow.py)
#   ms_tables.md      compare_runs_multiwindow.py's full report, per run
#   ms_summary.md     multistart_pairs.py: per arm, per start, and the pre-registered verdict
#
# Usage:  regen_multistart.sh [outdir]

set -euo pipefail
BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
R=$BASE/runs
S=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune/data_extraction_scripts
OUT=${1:-$R/2026_11_2_multistart_analysis}
mkdir -p "$OUT"

RUNS=(
  # ---- round 1 (10-31 → 11-01): the baseline, run once, and tune 1's B2_t24 ----
  # B2_t24 on djm-3 1000 stalled in its third trajectory (2026_10_31_..._ms500, in STALLED below);
  # its seed-600 rerun from the same start (11-13) ran 5 clean trajectories and stands in for it.
  baseline_djm1_1000=2026_11_1_baseline_djm1_1000_ms500   B2_t24_djm1_1000=2026_11_1_B2_t24_djm1_1000_ms500
  baseline_djm1_1800=2026_10_31_baseline_djm1_1800_ms500  B2_t24_djm1_1800=2026_11_1_B2_t24_djm1_1800_ms500
  baseline_djm1_2600=2026_11_1_baseline_djm1_2600_ms500   B2_t24_djm1_2600=2026_11_1_B2_t24_djm1_2600_ms500
  baseline_djm2_700=2026_11_1_baseline_djm2_700_ms500     B2_t24_djm2_700=2026_11_1_B2_t24_djm2_700_ms500
  baseline_djm2_1400=2026_11_1_baseline_djm2_1400_ms500   B2_t24_djm2_1400=2026_11_1_B2_t24_djm2_1400_ms500
  baseline_djm2_2100=2026_11_1_baseline_djm2_2100_ms500   B2_t24_djm2_2100=2026_11_1_B2_t24_djm2_2100_ms500
  baseline_djm3_1000=2026_10_31_baseline_djm3_1000_ms500  B2_t24_djm3_1000=2026_11_13_B2_t24_djm3_1000_ms600
  baseline_djm3_2700=2026_11_1_baseline_djm3_2700_ms500   B2_t24_djm3_2700=2026_11_1_B2_t24_djm3_2700_ms500
  baseline_djm3_3300=2026_11_1_baseline_djm3_3300_ms500   B2_t24_djm3_3300=2026_11_1_B2_t24_djm3_3300_ms500
  # ---- round 2 (11-05 → 11-08): tune 2's two layouts, against the round-1 baseline ----
  # D_8 on djm-2 700 stalled in its second trajectory (2026_11_6_..._ms500, in STALLED below); its
  # seed-600 rerun from the same start ran 5 clean trajectories and stands in for it. That start's
  # D_8 cells are therefore not same-seed with the baseline's.
  D_8_16_32_64_djm1_1000=2026_11_6_D_8_16_32_64_djm1_1000_ms500   D_10_40_80_djm1_1000=2026_11_7_D_10_40_80_djm1_1000_ms500
  D_8_16_32_64_djm1_1800=2026_11_5_D_8_16_32_64_djm1_1800_ms500   D_10_40_80_djm1_1800=2026_11_7_D_10_40_80_djm1_1800_ms500
  D_8_16_32_64_djm1_2600=2026_11_6_D_8_16_32_64_djm1_2600_ms500   D_10_40_80_djm1_2600=2026_11_8_D_10_40_80_djm1_2600_ms500
  D_8_16_32_64_djm2_700=2026_11_9_D_8_16_32_64_djm2_700_ms600     D_10_40_80_djm2_700=2026_11_8_D_10_40_80_djm2_700_ms500
  D_8_16_32_64_djm2_1400=2026_11_6_D_8_16_32_64_djm2_1400_ms500   D_10_40_80_djm2_1400=2026_11_7_D_10_40_80_djm2_1400_ms500
  D_8_16_32_64_djm2_2100=2026_11_6_D_8_16_32_64_djm2_2100_ms500   D_10_40_80_djm2_2100=2026_11_8_D_10_40_80_djm2_2100_ms500
  D_8_16_32_64_djm3_1000=2026_11_5_D_8_16_32_64_djm3_1000_ms500   D_10_40_80_djm3_1000=2026_11_7_D_10_40_80_djm3_1000_ms500
  D_8_16_32_64_djm3_2700=2026_11_6_D_8_16_32_64_djm3_2700_ms500   D_10_40_80_djm3_2700=2026_11_8_D_10_40_80_djm3_2700_ms500
  D_8_16_32_64_djm3_3300=2026_11_7_D_8_16_32_64_djm3_3300_ms500   D_10_40_80_djm3_3300=2026_11_8_D_10_40_80_djm3_3300_ms500
  # ---- later rounds: append <arm>_<stream>_<traj>=<run dir>[,<run dir>] here ----
)
# Stalled runs replaced in RUNS by a new-seed rerun of the same start: <label>=<stalled run dir>
STALLED=(
  B2_t24_djm3_1000=2026_10_31_B2_t24_djm3_1000_ms500
  D_8_16_32_64_djm2_700=2026_11_6_D_8_16_32_64_djm2_700_ms500
)
STALL_ARGS=()
for e in "${STALLED[@]}"; do
  [ -f "$R/${e#*=}/hmc.log" ] || { echo "ERROR: ${e%%=*}: no $R/${e#*=}/hmc.log" >&2; exit 1; }
  STALL_ARGS+=(--stalled "${e%%=*}=$R/${e#*=}/hmc.log")
done
RUN_ARGS=() LOG_ARGS=() ARMS=baseline
for e in "${RUNS[@]}"; do
  label=${e%%=*} logs=""
  IFS=, read -ra dirs <<< "${e#*=}"
  for d in "${dirs[@]}"; do
    [ -f "$R/$d/hmc.log" ] || { echo "ERROR: $label: no $R/$d/hmc.log" >&2; exit 1; }
    logs=${logs:+$logs,}$R/$d/hmc.log
  done
  RUN_ARGS+=(--run "$label=$logs")
  LOG_ARGS+=(--log "$label=$logs")
  arm=${label%_djm*}
  [[ ",$ARMS," == *",$arm,"* ]] || ARMS=$ARMS,$arm
done

python3 "$S/compare_runs_multiwindow.py" "${RUN_ARGS[@]}" --csv "$OUT/ms.csv" > "$OUT/ms_tables.md"
python3 "$S/multistart_pairs.py" "$OUT/ms.csv" --ref baseline --arms "$ARMS" "${LOG_ARGS[@]}" \
  "${STALL_ARGS[@]}" | tee "$OUT/ms_summary.md"
