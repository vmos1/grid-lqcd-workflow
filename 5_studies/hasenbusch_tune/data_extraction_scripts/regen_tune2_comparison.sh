#!/bin/bash
# Regenerate the tune2-vary-steps comparison (baseline vs the step-layout finalist 8/16/32/128 and,
# in Chroma, the hedge 10/40/80) from the raw HMC logs (2026-11-03). Modelled on
# regen_tune1_b_comparison.sh; same scripts, same conventions, same production reference.
#
# This script IS the manifest: which run directories make up each chain, in order. The Grid
# finalist chain is two checkpoint-resumed segments (2001-2010, 2011-2020); the second is picked up
# automatically once its run directory exists. Candidate names D_<light>_<strange>_<tail>_<gauge>,
# D for the default (baseline) ladder: D_8_16_32_64 = light rungs and log-dets at 8 steps, strange
# alone at 16, tail alone at 32, gauge alone at 64 (production form); its chains and flow ran with
# the gauge at 128 (Grid CUSTOM_LEVEL_* 0/1/2/3, MULT 1,1,2, MDSTEPS 8; Chroma spec l8s16t32g128),
# equivalent (L204). D_10_40_80 = light, log-dets and strange at 10, tail at 40, gauge at 80
# (Chroma spec l10t40g80; Grid only as five single draws).
#
# Worked output: __docs/2026_11_03_tune2_candidates_vs_baseline.md
#
# Outputs (outdir default runs/2026_11_03_tune2_comparison, outside the git tree; a cache):
#   grid.csv, chroma.csv              per-trajectory CSVs
#   grid_tables.md                    compare_runs_multiwindow.py's full report (per-sector times)
#   grid_summary.md, chroma_summary.md   candidate_tables.py, the doc's summary tables
#   __docs/figs/2026_11_03_tune2_{grid,chroma}_2x2.png   the figures (repo uv env: matplotlib)
#   piece_time.md                     time by piece (piece_time_table.py, Grid's own timers)
#   wflow_scales.md, __docs/figs/2026_11_03_tune2_wflow_scales.png   Wilson-flow scales, once the
#                                     flow logs exist (runs/*_wflow_t2fin_seg{1,2})
#   doc_check.txt                     check_doc_tables.py against the doc, once the doc exists
#
# Usage:  regen_tune2_comparison.sh [outdir]

set -euo pipefail
BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
R=$BASE/runs
S=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune/data_extraction_scripts
FIGS=$BASE/__docs/figs
OUT=${1:-$R/2026_11_03_tune2_comparison}
UV=/global/homes/v/vpa/.local/bin/uv
mkdir -p "$OUT" "$FIGS"

LADDER_BASE=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870

# ---- Grid chains (hmc.log of each segment, in trajectory order) ----
BASELINE=$R/2026_10_7_baseG_trajl1_seed300_20/hmc.log
FIN=$R/2026_11_2_t2fin_8_16_32_128_s300/hmc.log
FIN2=$(ls -d "$R"/*_t2fin_8_16_32_128_s300_r2 2>/dev/null | tail -1 || true)
if [ -n "$FIN2" ] && [ -f "$FIN2/hmc.log" ]; then
  FIN=$FIN,$FIN2/hmc.log
else
  echo "NOTE: finalist segment 2 (2011-2020) not found yet; D_8_16_32_64 has 10 trajectories" >&2
fi

python3 "$S/compare_runs_multiwindow.py" \
  --run "baseline=$BASELINE" --run "D_8_16_32_64=$FIN" \
  --ladder "baseline=$LADDER_BASE" --ladder "D_8_16_32_64=$LADDER_BASE" \
  --csv "$OUT/grid.csv" > "$OUT/grid_tables.md"

# ---- Chroma chains (run directories) ----
python3 "$S/chroma_runs_csv.py" \
  --run "baseline chain 1=$R/2026_10_8_chroma_baseg_mds12_mgall20_s300" \
  --run "baseline chain 2=$R/2026_10_26_chroma_baseg_mds12_base_20_s301" \
  --run "D_8_16_32_64=$R/2026_11_2_chroma_baseg_mds8_t2fin_20_s300" \
  --run "D_10_40_80=$R/2026_11_2_chroma_baseg_mds10_t2hedge_20_s300" \
  --csv "$OUT/chroma.csv"

# ---- the doc's summary tables ----
python3 "$S/candidate_tables.py" "$OUT/grid.csv" --ref baseline \
  --order "baseline,D_8_16_32_64" > "$OUT/grid_summary.md"
python3 "$S/candidate_tables.py" "$OUT/chroma.csv" --ref "baseline both chains" \
  --combine "baseline both chains=baseline chain 1+baseline chain 2" \
  --order "baseline chain 1,baseline chain 2,baseline both chains,D_8_16_32_64,D_10_40_80" > "$OUT/chroma_summary.md"

# ---- figures (matplotlib from the repo's uv environment) ----
# Band in panel B: the production ensemble's plaquette, mean ± one configuration's sd over the 9
# independent configurations of runs/2026_10_28_wflow_production_set (as the tune 1 comparison).
PBAND="0.5434195,0.0000297,production: mean ± 1 sd (9 configurations)"
$UV run --project "$BASE/grid-lqcd-workflow" python "$S/plot_candidates_2x2.py" "$OUT/grid.csv" \
  "$FIGS/2026_11_03_tune2_grid_2x2.png" --order "baseline,D_8_16_32_64" \
  --display "baseline=baseline (12/12/48/48);D_8_16_32_64=8/16/32/128 (light 8, strange 16, tail 32, gauge 128)" \
  --band "$PBAND" \
  --title "Grid, 48³×96: baseline vs the tune 2 finalist, from cfg_2000 (TRAJL 1.0)"
# Chroma: chain 1 in the baseline's slot, the layouts next, chain 2 last (as the tune 1 figure).
python3 - "$OUT/chroma.csv" "$OUT/chroma_plot.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
order = ["baseline chain 1", "D_8_16_32_64", "D_10_40_80", "baseline chain 2"]
rows.sort(key=lambda r: order.index(r["run"]))
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
EOF
$UV run --project "$BASE/grid-lqcd-workflow" python "$S/plot_candidates_2x2.py" "$OUT/chroma_plot.csv" \
  "$FIGS/2026_11_03_tune2_chroma_2x2.png" \
  --order "baseline chain 1,D_8_16_32_64,D_10_40_80,baseline chain 2" \
  --display "baseline chain 1=baseline, chain 1 (seed 300);baseline chain 2=baseline, chain 2 (seed 301);D_8_16_32_64=8/16/32/128;D_10_40_80=10/40/80 (light and strange at 10)" \
  --band "$PBAND" \
  --title "Chroma, 48³×96: baseline vs the tune 2 layouts, from cfg_2000 (τ0 1.414)"

# ---- time by piece from Grid's own per-call timers (L199) ----
python3 "$S/piece_time_table.py" --pieces \
  --run "baseline=$BASELINE" --run "D_8_16_32_64=$FIN" > "$OUT/piece_time.md"

# ---- Wilson-flow scales against the production ensemble, once the flow logs exist ----
WF=$(ls "$R"/*_wflow_t2fin_seg*/flow_*.log 2>/dev/null || true)
if [ -n "$WF" ]; then
  $UV run --project "$BASE/grid-lqcd-workflow" python "$S/plot_wflow_scales.py" \
    "$FIGS/2026_11_03_tune2_wflow_scales.png" \
    $WF "$R"/2026_10_28_wflow_tune1_set/flow_[0-3].log "$R"/2026_10_28_wflow_production_set/flow_[0-3].log \
    "$R/2026_10_26_wflow_validate_cheap_t42/test.log" > "$OUT/wflow_scales.md"
else
  echo "NOTE: no flow logs yet (runs/*_wflow_t2fin_seg*); wflow_scales.md not regenerated" >&2
fi

# ---- the doc against these outputs, cell by cell, once the doc exists ----
DOC=$BASE/__docs/2026_11_03_tune2_candidates_vs_baseline.md
if [ -f "$DOC" ]; then
  python3 "$S/check_doc_tables.py" "$DOC" \
    "2=$OUT/grid_summary.md" "3=$OUT/chroma_summary.md" "4=$OUT/piece_time.md" \
    $( [ -f "$OUT/wflow_scales.md" ] && echo "5=$OUT/wflow_scales.md" ) \
    | tee "$OUT/doc_check.txt"
fi

echo "outputs in $OUT and $FIGS/2026_11_03_tune2_*"
