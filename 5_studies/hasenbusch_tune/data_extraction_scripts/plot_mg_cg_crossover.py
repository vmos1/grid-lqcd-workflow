#!/usr/bin/env python3
"""Shared-MG vs QUDA-CG deriv cost per rung mass, at both tolerances.

Data = 48^3 FORCES_ONLY crossover scans on cfg_2000 (3-sample mean deriv
seconds per force evaluation; same seeds in both arms of each pair):
  tol 1e-6  : runs/2026_7_9_mgshare_crossover_48       (jobs 55732524-31)
  tol 1e-11 : runs/2026_7_10_mgshare_tol11_crossover_48 (job 55747239)
              + refinement points -0.2260/-0.2240/-0.2220 from
              runs/2026_7_10_mgshare_tol11_refine_48 (job 55752081)
Companion doc: __docs/2026_7_10_mg_cg_crossover_tables.md
Usage: module load python && python3 plot_mg_cg_crossover.py [outdir]
Writes 2026_7_10_mg_cg_crossover.{png,pdf} to outdir (default __docs/figs).
"""
import sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# rung mass : (shared-MG s/eval, QUDA-CG s/eval)
TOL6 = {  # TUNE_CG_TOL_DERIV=1e-6 (default), action 1e-8
    -0.2400: (21.7, 31.5), -0.2320: (21.8, 24.1), -0.2280: (22.1, 23.8),
    -0.2180: (25.0, 21.7), -0.2143: (25.0, 22.4), -0.2007: (28.2, 21.6),
    -0.1870: (31.2, 20.7), -0.1400: (38.9, 20.4),
}
TOL11 = {  # TUNE_CG_TOL_DERIV=1e-11, TUNE_CG_TOL_ACTION=1e-12
    -0.2400: (23.0, 63.9), -0.2320: (24.4, 33.3), -0.2280: (27.1, 29.6),
    -0.2260: (28.1, 28.4), -0.2240: (29.2, 27.4), -0.2220: (30.1, 26.7),
    -0.2180: (33.0, 25.9), -0.2143: (34.3, 24.7), -0.2007: (41.2, 23.3),
    -0.1870: (47.1, 22.2), -0.1400: (64.0, 21.0),
}
DONOR = -0.2416          # MG setup mass (rung 0, dedicated MG in every arm)
DONOR_MG = {6: 21.9, 11: 24.3}   # donor's own deriv (no CG reference exists)

def crossing(data):
    """Linear-interpolated mass where the MG and CG curves intersect."""
    ms = sorted(data)
    for a, b in zip(ms, ms[1:]):
        da, db = data[a][0] - data[a][1], data[b][0] - data[b][1]
        if da * db < 0:
            return a + (b - a) * (-da) / (db - da)
    return None

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
for ax, data, tol in ((axes[0], TOL6, "1e-6"), (axes[1], TOL11, "1e-11")):
    ms = sorted(data)
    ax.plot(ms, [data[m][0] for m in ms], "o-", color="tab:blue",
            label="shared-MG (donor setup at −0.2416)")
    ax.plot(ms, [data[m][1] for m in ms], "s-", color="tab:red",
            label="QUDA-CG")
    ax.plot([DONOR], [DONOR_MG[6 if tol == "1e-6" else 11]], "D",
            color="tab:blue", mfc="white", ms=8,
            label="donor rung (own MG; no CG ref)")
    x = crossing(data)
    if x is not None:
        ax.axvline(x, color="gray", ls=":", lw=1)
        ax.annotate(f"m* ≈ {x:.4f}", (x, ax.get_ylim()[0]),
                    xytext=(x + 0.002, 24), fontsize=9, color="gray")
    ax.axhline(20, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.text(-0.15, 20.5, "~20 s force-assembly floor", fontsize=8,
            ha="right", alpha=0.7)
    ax.set_title(f"deriv tolerance {tol}")
    ax.set_xlabel("rung mass $am_q^{bare}$")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("deriv seconds / force evaluation")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Shared-MG vs QUDA-CG per-rung deriv cost, 48$^3$ cfg_2000 "
             "(3-sample means, same seeds)", fontsize=11)
fig.tight_layout()

outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parents[4] / "docs" / "figs"
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(outdir / f"2026_7_10_mg_cg_crossover.{ext}", dpi=300)
print(f"wrote {outdir}/2026_7_10_mg_cg_crossover.png/.pdf; "
      f"crossings: 1e-6 m*={crossing(TOL6):.4f}, 1e-11 m*={crossing(TOL11):.4f}")
