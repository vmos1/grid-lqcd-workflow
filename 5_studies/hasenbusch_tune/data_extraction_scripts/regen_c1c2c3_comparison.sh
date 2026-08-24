#!/bin/bash
# Regenerate the base+G / C1 / C2 / C3 four-way comparison from the raw HMC logs.
#
# This script IS the manifest: which log windows make up each of the four runs,
# and which Hasenbusch ladder each ran. That mapping is the part worth keeping --
# every run spans 2-3 checkpoint-resumed windows, and `accept_run_tables.py --csv`
# cannot stitch them (it restarts trajectory numbering at 2001 per file), so the
# window list has to be passed to compare_runs_multiwindow.py explicitly.
#
# Worked output: __docs/2026_8_20_c1_c2_c3_vs_baseg_comparison.md
#
# Usage:  regen_c1c2c3_comparison.sh [outdir] [figure.png]
#         outdir defaults to $BASE/runs/2026_8_20_c1c2c3_baseg_comparison
#         figure defaults to the PNG the comparison doc embeds
#
# The figure step needs matplotlib+pandas (`module load python/3.12-26.1.0`);
# it is skipped with a warning if they are missing, since the tables above are
# stdlib-only and are the part that must always work.
#
# The CSV lands OUTSIDE the git tree on purpose (run/analysis data, per the
# project convention) -- but it is cheap to rebuild, so treat it as a cache,
# not an artifact.

set -euo pipefail
module load python/3.12-26.1.0 2>/dev/null || true

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
R=$BASE/runs
S=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune/data_extraction_scripts

OUT=${1:-$R/2026_8_20_c1c2c3_baseg_comparison}
mkdir -p "$OUT"
CSV=$OUT/c1c2c3_baseline.csv

cd "$S"

python3 compare_runs_multiwindow.py \
  --run "base+G=$R/2026_7_13_accept_baseG_seed300_48/accept_baseG_seed300.log,$R/2026_7_13_accept_baseG_seed300_48/accept_baseG_seed300_ext_2011_2020.log" \
  --run "C1 (u1a)=$R/2026_8_13_u1a_accept10_seed300_48/u1a_accept10_seed300.log,$R/2026_8_13_u1a_accept10_seed300_48/u1a_accept10_seed300_ext_2011_2020.log" \
  --run "C2 (w3@5)=$R/2026_8_10_3level_w3_mdsteps5_48/3level_w3_mdsteps5.log,$R/2026_8_10_3level_w3_mdsteps5_48/3level_w3_mdsteps5_ext_2011_2020.log" \
  --run "C3 (w3@6+mid)=$R/2026_8_18_lvlroute_w3_tailmiddle_48/lvlroute_w3_tailmiddle.log,$R/2026_8_18_lvlroute_w3_tailmiddle_48/lvlroute_w3_tailmiddle_ext_2003_2010.log,$R/2026_8_18_lvlroute_w3_tailmiddle_48/lvlroute_w3_tailmiddle_ext_2011_2020.log" \
  --ladder 'base+G=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870' \
  --ladder 'C1 (u1a)=-0.2416,-0.2250,-0.1870,0.3044' \
  --ladder 'C2 (w3@5)=-0.2416,-0.2380,-0.2340,-0.2180,-0.1870' \
  --ladder 'C3 (w3@6+mid)=-0.2416,-0.2380,-0.2340,-0.2180,-0.1870' \
  --csv "$CSV"

echo
echo "=================== autocorrelation-corrected observables ==================="
echo
# --pooled because all four runs share the action and trajectory length, so they
# share one true tau_int; pooling rho(t) over them is 4x less noisy and applies a
# single inflation factor to every arm. NEVER quote the naive z from the tables
# above -- it put C1's smeared plaquette at +3.28 sigma vs +1.69 corrected.
python3 obs_autocorr_compare.py "$CSV" --pooled

FIG=${2:-$BASE/__docs/figs/2026_8_20_c1_c2_c3_vs_baseg_20traj_2x2.png}
echo
if python3 -c "import matplotlib, pandas" 2>/dev/null; then
    mkdir -p "$(dirname "$FIG")"
    python3 plot_c1c2c3_baseline.py "$CSV" "$FIG"
else
    echo "WARNING: matplotlib/pandas unavailable -- figure NOT regenerated."
    echo "         module load python/3.12-26.1.0 and rerun, or:"
    echo "         python3 plot_c1c2c3_baseline.py $CSV $FIG"
fi

echo
echo "CSV: $CSV"
