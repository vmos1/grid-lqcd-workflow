#!/bin/bash
# Regenerate the tune1-perturbations candidate comparison (baseline vs B1, B2, B2 with the tail at
# 24 steps; Grid and Chroma) from the raw HMC logs (2026-10-27). Modelled on
# regen_c1c2c3_comparison.sh.
#
# This script IS the manifest: which run directories make up each chain, in order. Every Grid chain
# spans several checkpoint-resumed segments, and the B2 chains change seed at 2008 (fresh start from
# the 2008 configuration after the light-solver stall in 2009, DIAG_START_CFG); the B2 12/48 chain's
# last segment (2015-2020) is picked up automatically once its run directory exists.
#
# Worked output: __docs/2026_10_27_tune1_b_candidates_vs_baseline.md
#
# Outputs (outdir default runs/2026_10_27_tune1_b_comparison, outside the git tree; a cache):
#   grid.csv, chroma.csv            per-trajectory CSVs
#   grid_tables.md                  compare_runs_multiwindow.py's full report (per-sector times)
#   grid_summary.md, chroma_summary.md   candidate_tables.py, the doc's summary tables
#   __docs/figs/2026_10_27_tune1_b_{grid,chroma}_2x2.png   the figures (repo uv env: matplotlib)
#   __docs/figs/2026_10_29_tune1_wflow_scales.png, wflow_scales.md   Wilson-flow scales per stream
#
# Usage:  regen_tune1_b_comparison.sh [outdir]

set -euo pipefail
BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
R=$BASE/runs
S=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune/data_extraction_scripts
FIGS=$BASE/__docs/figs
OUT=${1:-$R/2026_10_27_tune1_b_comparison}
mkdir -p "$OUT" "$FIGS"

LADDER_BASE=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870
LADDER_B1=-0.2416,-0.2400,-0.2320,-0.2180,-0.0300
LADDER_B2=-0.2416,-0.2400,-0.2320,-0.2180,0.1000

# ---- Grid chains (hmc.log of each segment, in trajectory order) ----
BASELINE=$R/2026_10_7_baseG_trajl1_seed300_20/hmc.log
B1=$R/2026_10_19_B1_md12_s300/hmc.log,$R/2026_10_20_B1_md12_s300_r1/hmc.log,$R/2026_10_25_B1_md12_s300_r2/hmc.log,$R/2026_10_25_B1_md12_s300_r3/hmc.log
B2=$R/2026_10_19_B2_md12_s300/hmc.log,$R/2026_10_20_B2_md12_s300_r1/hmc.log,$R/2026_10_20_B2_md12_s300_r2/hmc.log,$R/2026_10_24_B2_md12_s302_from2008/hmc.log
B2_EXT=$(ls -d "$R"/*_B2_md12_s302_r2 2>/dev/null | tail -1 || true)
if [ -n "$B2_EXT" ] && [ -f "$B2_EXT/hmc.log" ]; then
  B2=$B2,$B2_EXT/hmc.log
else
  echo "NOTE: B2 12/48 extension (2015-2020) not found yet; B2 has 14 trajectories" >&2
fi
B2T24=$R/2026_10_20_B2_tail24_s300/hmc.log,$R/2026_10_20_B2_tail24_s300_r2/hmc.log,$R/2026_10_24_B2_tail24_s301_from2008/hmc.log,$R/2026_10_26_B2_tail24_s301_r2/hmc.log

# Run names (2026-10-27): the letter is the tail MASS (baseline −0.1870, B1 −0.0300, B2 +0.1000),
# the _tNN suffix the tail's STEP COUNT. The run directories keep their original names
# (B2_tail24_*, B2_md12_*, ...); the main doc's §1-§6 use the old names (name map at its top).
python3 "$S/compare_runs_multiwindow.py" \
  --run "baseline=$BASELINE" --run "B1_t48=$B1" --run "B2_t48=$B2" --run "B2_t24=$B2T24" \
  --ladder "baseline=$LADDER_BASE" --ladder "B1_t48=$LADDER_B1" --ladder "B2_t48=$LADDER_B2" \
  --ladder "B2_t24=$LADDER_B2" \
  --csv "$OUT/grid.csv" > "$OUT/grid_tables.md"

# ---- Chroma chains (run directories) ----
python3 "$S/chroma_runs_csv.py" \
  --run "baseline chain 1=$R/2026_10_8_chroma_baseg_mds12_mgall20_s300" \
  --run "baseline chain 2=$R/2026_10_26_chroma_baseg_mds12_base_20_s301" \
  --run "B1_t48=$R/2026_10_25_chroma_baseg_mds12_b1_20_s300" \
  --run "B2_t48=$R/2026_10_23_chroma_baseg_mds12_b2_20_s300" \
  --run "B2_t24=$R/2026_10_26_chroma_baseg_mds12_b2t24_20_s300" \
  --csv "$OUT/chroma.csv"

# ---- the doc's summary tables ----
python3 "$S/candidate_tables.py" "$OUT/grid.csv" --ref baseline \
  --order "baseline,B1_t48,B2_t48,B2_t24" > "$OUT/grid_summary.md"
python3 "$S/candidate_tables.py" "$OUT/chroma.csv" --ref "baseline both chains" \
  --combine "baseline both chains=baseline chain 1+baseline chain 2" \
  --order "baseline chain 1,baseline chain 2,baseline both chains,B1_t48,B2_t48,B2_t24" > "$OUT/chroma_summary.md"

# ---- figures (matplotlib from the repo's uv environment) ----
# Band in B and D: the production ensemble's plaquette, mean ± one configuration's sd over the 9
# independent configurations of runs/2026_10_28_wflow_production_set (plaq0 lines, 2026-10-29).
# D's running mean starts at 2005, past the relaxation from cfg_2000 (plaquette 0.543508).
PBAND="0.5434195,0.0000297,production: mean ± 1 sd (9 configurations)"
DISP="baseline=baseline (tail −0.1870, 12/48);B1_t48=B1_t48 (tail −0.0300, 12/48);B2_t48=B2_t48 (tail +0.1000, 12/48);B2_t24=B2_t24 (tail +0.1000, 12/24/48)"
uv run --project "$BASE/grid-lqcd-workflow" python "$S/plot_candidates_2x2.py" "$OUT/grid.csv" \
  "$FIGS/2026_10_27_tune1_b_grid_2x2.png" --order "baseline,B1_t48,B2_t48,B2_t24" --display "$DISP" \
  --band "$PBAND" --mean-from 2005 \
  --title "Grid, 48³×96: baseline vs B candidates, from cfg_2000 (TRAJL 1.0)"
# Chroma: chain 1 takes the baseline's slot (blue); chain 2 the fifth (pink); the B runs keep theirs.
python3 - "$OUT/chroma.csv" "$OUT/chroma_plot.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
order = ["baseline chain 1", "B1_t48", "B2_t48", "B2_t24", "baseline chain 2"]
rows.sort(key=lambda r: order.index(r["run"]))
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
EOF
uv run --project "$BASE/grid-lqcd-workflow" python "$S/plot_candidates_2x2.py" "$OUT/chroma_plot.csv" \
  "$FIGS/2026_10_27_tune1_b_chroma_2x2.png" \
  --order "baseline chain 1,B1_t48,B2_t48,B2_t24,baseline chain 2" \
  --display "baseline chain 1=baseline, chain 1 (seed 300);baseline chain 2=baseline, chain 2 (seed 301);B1_t48=B1_t48 (tail −0.0300);B2_t48=B2_t48 (tail +0.1000);B2_t24=B2_t24 (tail +0.1000, at 24 steps)" \
  --band "$PBAND" --mean-from 2005 \
  --title "Chroma, 48³×96: baseline vs B candidates, from cfg_2000 (τ0 1.414)"

# ---- Wilson-flow scales per stream against the production ensemble (§5, added 2026-10-29) ----
W=$R/2026_10_28_wflow
uv run --project "$BASE/grid-lqcd-workflow" python "$S/plot_wflow_scales.py" \
  "$FIGS/2026_10_29_tune1_wflow_scales.png" \
  "${W}_tune1_set"/flow_[0-3].log "${W}_b2t48_extra"/flow_[01].log "${W}_production_set"/flow_[0-3].log \
  "$R/2026_10_26_wflow_validate_cheap_t42/test.log" > "$OUT/wflow_scales.md"

echo "outputs in $OUT, $FIGS/2026_10_27_tune1_b_*_2x2.png and $FIGS/2026_10_29_tune1_wflow_scales.png"
