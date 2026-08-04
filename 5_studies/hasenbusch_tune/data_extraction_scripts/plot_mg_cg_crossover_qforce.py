#!/usr/bin/env python3
"""Shared-MG vs QUDA-CG deriv cost per rung mass, WITH QUDA force assembly.

Single-panel companion to plot_mg_cg_crossover.py: same crossover-plot design
(donor diamond, crossing annotation) but for the ONE tolerance this scan
covers — production (deriv 1e-11, action 1e-12) — with the QUDA
force-assembly path on (HASEN_QUDA_FORCE_RUNGS=all), i.e. the numbers in
__docs/2026_7_11_mg_cg_crossover_qforce_tables.md.

Data = 48^3 FORCES_ONLY crossover scan on cfg_2000 (3-sample mean deriv
seconds per force evaluation; same seeds in both arms), job 55783709 (array
of 8) covering base/e1/x1 ladders plus the -0.2260/-0.2240/-0.2220 refinement.
Companion doc: __docs/2026_7_11_mg_cg_crossover_qforce_tables.md
Usage: module load python && python3 plot_mg_cg_crossover_qforce.py [outdir]
Writes 2026_7_11_mg_cg_crossover_qforce.{png,pdf} to outdir (default __docs/figs).
"""
import sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# rung mass : (shared-MG s/eval, QUDA-CG s/eval) -- deriv 1e-11, action 1e-12,
# QUDA force-assembly path on (HASEN_QUDA_FORCE_RUNGS=all)
TOL11_QFORCE = {
    -0.2400: (8.27, 48.66), -0.2320: (9.42, 18.38), -0.2280: (12.06, 14.89),
    -0.2260: (13.59, 13.68), -0.2240: (14.71, 12.71), -0.2220: (15.63, 12.08),
    -0.2180: (17.72, 10.75), -0.2143: (19.38, 9.99), -0.2007: (26.10, 8.44),
    -0.1870: (32.78, 7.54), -0.1400: (49.96, 6.48),
}
DONOR = -0.2416           # MG setup mass (rung 0, dedicated MG in every arm)
DONOR_MG = 10.43          # donor's own deriv (no CG reference exists)

def crossing(data):
    """Linear-interpolated mass where the MG and CG curves intersect."""
    ms = sorted(data)
    for a, b in zip(ms, ms[1:]):
        da, db = data[a][0] - data[a][1], data[b][0] - data[b][1]
        if da * db < 0:
            return a + (b - a) * (-da) / (db - da)
    return None

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ms = sorted(TOL11_QFORCE)
ax.plot(ms, [TOL11_QFORCE[m][0] for m in ms], "o-", color="tab:blue",
        label="shared-MG (donor setup at −0.2416)")
ax.plot(ms, [TOL11_QFORCE[m][1] for m in ms], "s-", color="tab:red",
        label="QUDA-CG")
ax.plot([DONOR], [DONOR_MG], "D", color="tab:blue", mfc="white", ms=8,
        label="donor rung (own MG; no CG ref)")
x = crossing(TOL11_QFORCE)
if x is not None:
    ax.axvline(x, color="gray", ls=":", lw=1)
    ax.annotate(f"m* ≈ {x:.4f}", (x, ax.get_ylim()[1]),
                xytext=(x + 0.003, 44), fontsize=9, color="gray")
ax.set_xlabel(r"rung mass $am_q^{bare}$")
ax.set_ylabel("deriv seconds / force evaluation")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Force evaluation time comparison: shared MG vs CG", fontsize=12)
fig.tight_layout()

outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parents[4] / "docs" / "figs"
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(outdir / f"2026_7_11_mg_cg_crossover_qforce.{ext}", dpi=300)
print(f"wrote {outdir}/2026_7_11_mg_cg_crossover_qforce.png/.pdf; "
      f"crossing m*={crossing(TOL11_QFORCE):.4f}")
