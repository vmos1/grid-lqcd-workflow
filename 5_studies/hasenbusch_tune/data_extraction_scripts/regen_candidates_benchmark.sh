#!/bin/bash
# Regenerate the HMC candidates benchmark (2026-11-16): the baseline and the three viable candidates
# of tunes 1-2 (B2_t24, D_10_40_80, D_8_16_32_64) side by side, at both scales the campaigns used:
# 20-trajectory chains from cfg_2000 (Grid and Chroma) and the 9 production starts of
# multistart-validation (Grid, 5 trajectories each, seed 500, paired with the baseline). Combines
# regen_tune1_b_comparison.sh, regen_tune2_comparison.sh and regen_multistart.sh; same scripts,
# same conventions, same production reference.
#
# This script IS the manifest. To benchmark a new candidate against the same data:
#   1. its 20-trajectory Grid chain from cfg_2000 (seed offset 300, TRAJL 1.0, four 40 GB nodes):
#      add a GRID_RUNS line (label=hmc.log[,hmc.log of the next segment]) and its ladder in LADDERS;
#   2. its Chroma chain, if any: a CHROMA_RUNS line;
#   3. its 9 multistart runs (perlmutter/multistart_submit.sh): 9 lines in regen_multistart.sh;
#   4. Wilson flow on the chain's 20 configurations (perlmutter/wflow48_batch.sh): its row in
#      plot_wflow_scales.py ROWS, its flow logs in WFLOW_LOGS and its row label in WFLOW_ROWS;
#   5. its name in ORDER (and in plot_candidates_2x2.NAMED for a fixed colour).
# The protocol, pass bars and reference numbers: __docs/2026_11_16_hmc_tuning_summary.md.
#
# Candidates (names as in the campaigns): B2_t24 = tail mass +0.1000 alone on its own level at 24
# steps, light and strange at 12, gauge at 48 (tune 1); D_10_40_80 = baseline ladder, light rungs,
# log-dets and strange at 10, tail 40, gauge 80; D_8_16_32_64 = baseline ladder, light 8, strange 16,
# tail 32, gauge 64 (tune 2). D_8_16_32_64's cfg_2000 chains and their flow ran with the gauge at
# 128, equivalent (L204); its multistart arm ran at 64. D_10_40_80's Grid chain (two segments,
# jobs 61425065/61425176) and its flow are picked up automatically once their run dirs exist.
#
# Worked output: __docs/2026_11_16_hmc_candidates_benchmark.md
#
# Outputs (outdir default runs/2026_11_16_candidates_benchmark, outside the git tree; a cache):
#   grid.csv, grid_tables.md, grid_summary.md      the Grid chains (doc §3)
#   chroma.csv, chroma_summary.md                  the Chroma chains (doc §4)
#   multistart/{ms.csv,ms_tables.md,ms_summary.md,ms_runs.txt}   regen_multistart.sh (doc §5)
#   piece_time_ms.md       time by piece over the 9 starts, 36 warm trajectories per arm (doc §6)
#   piece_time_chains.md   the same from the cfg_2000 chains (not in the doc; cfg_2000 is quiet)
#   force_by_monomial.md   force per monomial over the 9 starts, 45 trajectories per arm (doc §6)
#   wflow_scales.md        Wilson-flow scales of the Grid chains against production (doc §7)
#   __docs/figs/2026_11_16_bench_{grid,chroma}_2x2.png, _force_by_monomial.png, _wflow_scales.png
#   doc_check.txt          check_doc_tables.py: the doc against all of the above; exits non-zero
#                          if any doc cell differs
#
# Usage:  regen_candidates_benchmark.sh [outdir]

set -euo pipefail
BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
R=$BASE/runs
S=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune/data_extraction_scripts
FIGS=$BASE/__docs/figs
OUT=${1:-$R/2026_11_16_candidates_benchmark}
UV=/global/homes/v/vpa/.local/bin/uv
PY="$UV run --project $BASE/grid-lqcd-workflow python"
DOC=$BASE/__docs/2026_11_16_hmc_candidates_benchmark.md
mkdir -p "$OUT" "$FIGS"

ORDER="baseline,B2_t24,D_8_16_32_64,D_10_40_80"   # tune order, as regen_multistart.sh's arms
LADDER_BASE=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870
LADDER_B2=-0.2416,-0.2400,-0.2320,-0.2180,0.1000
declare -A LADDERS=([baseline]=$LADDER_BASE [B2_t24]=$LADDER_B2 [D_10_40_80]=$LADDER_BASE
                    [D_8_16_32_64]=$LADDER_BASE)

# ---- Grid chains from cfg_2000 (hmc.log of each segment, in trajectory order) ----
# B2_t24 changes seed at 2008 (fresh start from the 2008 configuration after the light-solver stall
# in 2009, DIAG_START_CFG; tune 1 §6).
GRID_RUNS=(
  "baseline=$R/2026_10_7_baseG_trajl1_seed300_20/hmc.log"
  "B2_t24=$R/2026_10_20_B2_tail24_s300/hmc.log,$R/2026_10_20_B2_tail24_s300_r2/hmc.log,$R/2026_10_24_B2_tail24_s301_from2008/hmc.log,$R/2026_10_26_B2_tail24_s301_r2/hmc.log"
  "D_8_16_32_64=$R/2026_11_2_t2fin_8_16_32_128_s300/hmc.log,$R/2026_11_3_t2fin_8_16_32_128_s300_r2/hmc.log"
)
D10=""
for seg in D_10_40_80_s300 D_10_40_80_s300_r2; do
  d=$(ls -d "$R"/*_"$seg" 2>/dev/null | tail -1 || true)
  if [ -n "$d" ] && [ -f "$d/hmc.log" ] && grep -aq "Total time for trajectory" "$d/hmc.log"; then
    D10=${D10:+$D10,}$d/hmc.log
  fi
done
if [ -n "$D10" ]; then
  GRID_RUNS+=("D_10_40_80=$D10")
else
  echo "NOTE: D_10_40_80 Grid chain (runs/*_D_10_40_80_s300[_r2]) has no trajectory yet; left out" >&2
fi
GRID_ARGS=() GRID_ORDER=""
for o in ${ORDER//,/ }; do
  for e in "${GRID_RUNS[@]}"; do
    [ "${e%%=*}" = "$o" ] || continue
    GRID_ARGS+=(--run "$e" --ladder "$o=${LADDERS[$o]}")
    GRID_ORDER=${GRID_ORDER:+$GRID_ORDER,}$o
  done
done
python3 "$S/compare_runs_multiwindow.py" "${GRID_ARGS[@]}" --csv "$OUT/grid.csv" > "$OUT/grid_tables.md"
python3 "$S/candidate_tables.py" "$OUT/grid.csv" --ref baseline --order "$GRID_ORDER" > "$OUT/grid_summary.md"
python3 "$S/piece_time_table.py" --pieces $(printf -- '--run %s ' "${GRID_RUNS[@]}") > "$OUT/piece_time_chains.md"

# ---- Chroma chains from cfg_2000 (run directories) ----
python3 "$S/chroma_runs_csv.py" \
  --run "baseline chain 1=$R/2026_10_8_chroma_baseg_mds12_mgall20_s300" \
  --run "baseline chain 2=$R/2026_10_26_chroma_baseg_mds12_base_20_s301" \
  --run "B2_t24=$R/2026_10_26_chroma_baseg_mds12_b2t24_20_s300" \
  --run "D_10_40_80=$R/2026_11_2_chroma_baseg_mds10_t2hedge_20_s300" \
  --run "D_8_16_32_64=$R/2026_11_2_chroma_baseg_mds8_t2fin_20_s300" \
  --csv "$OUT/chroma.csv"
CHROMA_ORDER="baseline chain 1,baseline chain 2,baseline both chains,B2_t24,D_8_16_32_64,D_10_40_80"
python3 "$S/candidate_tables.py" "$OUT/chroma.csv" --ref "baseline both chains" \
  --combine "baseline both chains=baseline chain 1+baseline chain 2" \
  --order "$CHROMA_ORDER" > "$OUT/chroma_summary.md"

# ---- figures of the chains (matplotlib from the repo's uv environment) ----
# Band in panel B: the production ensemble's plaquette, mean ± one configuration's sd over its 9
# independent configurations (runs/2026_10_28_wflow_production_set), as in the tune 1 and 2 docs.
PBAND="0.5434195,0.0000297,production: mean ± 1 sd (9 configurations)"
DISP="baseline=baseline (tail −0.1870, 12/12/48/48);B2_t24=B2_t24 (tail +0.1000, 12/12/24/48);D_10_40_80=D_10_40_80 (10/10/40/80);D_8_16_32_64=D_8_16_32_64 (8/16/32, gauge 128 in this chain)"
$PY "$S/plot_candidates_2x2.py" "$OUT/grid.csv" "$FIGS/2026_11_16_bench_grid_2x2.png" \
  --order "$GRID_ORDER" --display "$DISP" --band "$PBAND" --named-colors \
  --title "Grid, 48³×96: baseline and the three candidates, 20 trajectories from cfg_2000 (TRAJL 1.0)"
python3 - "$OUT/chroma.csv" "$OUT/chroma_plot.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
order = ["baseline chain 1", "B2_t24", "D_8_16_32_64", "D_10_40_80", "baseline chain 2"]
rows.sort(key=lambda r: order.index(r["run"]))
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
EOF
$PY "$S/plot_candidates_2x2.py" "$OUT/chroma_plot.csv" "$FIGS/2026_11_16_bench_chroma_2x2.png" \
  --order "baseline chain 1,B2_t24,D_8_16_32_64,D_10_40_80,baseline chain 2" --named-colors \
  --display "baseline chain 1=baseline, chain 1 (seed 300);baseline chain 2=baseline, chain 2 (seed 301);B2_t24=B2_t24 (tail +0.1000, at 24);D_10_40_80=D_10_40_80 (10/40/80);D_8_16_32_64=D_8_16_32_64 (8/16/32/128)" \
  --band "$PBAND" \
  --title "Chroma, 48³×96: baseline and the three candidates, 20 trajectories from cfg_2000 (τ0 1.414)"

# ---- the 9 production starts (multistart-validation) ----
bash "$S/regen_multistart.sh" "$OUT/multistart" > /dev/null
declare -A MS
while IFS='=' read -r label logs; do
  arm=${label%_djm*}
  MS[$arm]=${MS[$arm]:+${MS[$arm]},}$logs
done < "$OUT/multistart/ms_runs.txt"
MS_ARGS=()
for o in ${ORDER//,/ }; do
  [ -n "${MS[$o]:-}" ] || { echo "ERROR: no multistart arm $o in $OUT/multistart/ms_runs.txt" >&2; exit 1; }
  MS_ARGS+=(--run "$o=${MS[$o]}")
done
python3 "$S/piece_time_table.py" --pieces "${MS_ARGS[@]}" > "$OUT/piece_time_ms.md"
$PY "$S/plot_force_by_monomial.py" "$FIGS/2026_11_16_bench_force_by_monomial.png" "${MS_ARGS[@]}" \
  --starts --named-colors \
  --title "Grid: force per monomial over the 9 production starts (45 trajectories per arm)" \
  > "$OUT/force_by_monomial.md"

# ---- Wilson-flow scales of the Grid chains against the production ensemble ----
# Baseline and B2_t24: 4 configurations each on 10-28 (tune1_set), the other 16 each on 11-16
# (wflow_bench_base_b2t24); D_8_16_32_64: 20 (t2fin seg1/seg2); D_10_40_80: its chain's 20 once
# flowed (runs/*_wflow_bench_d10*), the five single draws until then.
WFLOW_LOGS=("$R"/2026_10_28_wflow_tune1_set/flow_[0-3].log "$R"/*_wflow_t2fin_seg[12]/flow_*.log)
WFLOW_LOGS+=($(ls "$R"/*_wflow_bench_base_b2t24/flow_*.log 2>/dev/null || true))
D10_FLOW=($(ls "$R"/*_wflow_bench_d10*/flow_*.log 2>/dev/null || true))
if [ ${#D10_FLOW[@]} -gt 0 ]; then
  WFLOW_LOGS+=("${D10_FLOW[@]}")
  WFLOW_ROWS="Grid baseline,Grid B2_t24,Grid D_8_16_32_64,Grid D_10_40_80"
else
  WFLOW_LOGS+=("$R"/*_wflow_t2hedge_draws/flow_*.log)
  WFLOW_ROWS="Grid baseline,Grid B2_t24,Grid D_8_16_32_64,Grid D_10_40_80 (5 single draws)"
  echo "NOTE: no D_10_40_80 chain flow yet (runs/*_wflow_bench_d10*); its five single draws stand in" >&2
fi
WFLOW_LOGS+=("$R"/2026_10_28_wflow_production_set/flow_[0-3].log "$R/2026_10_26_wflow_validate_cheap_t42/test.log")
$PY "$S/plot_wflow_scales.py" "$FIGS/2026_11_16_bench_wflow_scales.png" --rows "$WFLOW_ROWS" \
  --title "Wilson-flow scales: the Grid chains from cfg_2000 against the production ensemble (48³×96, clover E)" \
  "${WFLOW_LOGS[@]}" > "$OUT/wflow_scales.md"

# ---- the doc against these outputs, cell by cell, once the doc exists ----
if [ -f "$DOC" ]; then
  python3 "$S/check_doc_tables.py" "$DOC" \
    "3=$OUT/grid_summary.md" "4=$OUT/chroma_summary.md" "5=$OUT/multistart/ms_summary.md" \
    "6=$OUT/piece_time_ms.md" "6=$OUT/force_by_monomial.md" "7=$OUT/wflow_scales.md" \
    | tee "$OUT/doc_check.txt"
fi

echo "outputs in $OUT and $FIGS/2026_11_16_bench_*"
