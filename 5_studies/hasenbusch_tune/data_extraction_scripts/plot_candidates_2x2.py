#!/usr/bin/env python3
"""Per-trajectory figure for a candidate-vs-baseline comparison (tune1-perturbations, 2026-10-27),
from the CSV of `compare_runs_multiwindow.py --csv` (Grid) or `chroma_runs_csv.py` (Chroma). Three
panels in a row (the name is historical: a 2x2 until 2026-10-29):

  (A) time per trajectory (full trajectory, cycle_s)   (B) plaquette (unsmeared)
  (C) dH (open marker = rejected)

Uses `plot_accept_runs.py`'s helpers (style map, panel drawing, accept/reject fill, axis styling)
unchanged, so colours, marker shapes and conventions match every other figure in __docs/figs/.
That module's fourth panel, the max kick, is left out: in these runs the per-trajectory maximum is
the gauge or the strange force (~0.30) in every candidate, so it does not separate them. A running
mean of the plaquette (the 10-27 panel D) was dropped 10-29: it only restates B.

--band MEAN,SD,LABEL shades a reference value (mean ± sd) in B, e.g. the production ensemble's
plaquette.

--named-colors (2026-11-16, candidates benchmark): a run named in NAMED takes that colour and marker
(the same per-name palette as plot_wflow_scales.SLOT), so a candidate keeps one colour across every
figure of a doc; other runs keep their --order slot. Off by default, so older figures regenerate
unchanged.

Colour follows the run, given in fixed order by --order (Okabe-Ito slots, validated CVD-safe; each
run also has its own marker shape, so identity never rests on colour alone). Pass the same labels in
the same slots for the Grid and the Chroma figure so an entity keeps its colour across both.

Usage:
  plot_candidates_2x2.py <in.csv> <out.png> --order 'L1,L2,...' [--title '...'] [--display 'L1=text;L2=text']
                         [--band 'MEAN,SD,LABEL'] [--named-colors]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accept_runs as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (P sets the Agg backend first)
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SLOTS = [("#0072B2", "o"), ("#E69F00", "s"), ("#009E73", "^"), ("#D55E00", "D"), ("#CC79A7", "v")]
# --named-colors: as plot_wflow_scales.SLOT, plus the Chroma baseline chains' labels
NAMED = {"baseline": ("#0072B2", "o"), "baseline chain 1": ("#0072B2", "o"),
         "baseline chain 2": ("#CC79A7", "v"), "B1_t48": ("#E69F00", "s"),
         "B2_t48": ("#009E73", "^"), "B2_t24": ("#D55E00", "D"),
         "D_8_16_32_64": ("#56B4E9", "P"), "D_10_40_80": ("#000000", "X")}
BAND = "#e6e6e6"


def _band(ax, band):
    if band:
        ax.axhspan(band[0] - band[1], band[0] + band[1], color=BAND, lw=0, zorder=0)
        ax.axhline(band[0], color=P.MUTED, ls="--", lw=0.9, zorder=1)


def main(argv):
    csv_path, out_png = argv[0], argv[1]
    order, title, display, band, named = None, "", {}, None, False
    i = 2
    while i < len(argv):
        if argv[i] == "--named-colors":
            named = True
            i += 1
            continue
        if argv[i] == "--order":
            order = argv[i + 1].split(",")
        elif argv[i] == "--title":
            title = argv[i + 1]
        elif argv[i] == "--display":
            display = dict(kv.split("=", 1) for kv in argv[i + 1].split(";"))
        elif argv[i] == "--band":
            m, sd, lab = argv[i + 1].split(",", 2)
            band = (float(m), float(sd), lab)
        i += 2
    df = pd.read_csv(csv_path)
    runs = order or list(dict.fromkeys(df["run"]))
    df = df[df["run"].isin(runs)].copy()
    df["run"] = pd.Categorical(df["run"], categories=runs, ordered=True)
    df = df.sort_values(["run", "traj_abs"])
    sty = {r: (NAMED[r] if named and r in NAMED else SLOTS[k]) for k, r in enumerate(runs)}

    ncol = len(runs) if len(runs) <= 3 else 3
    legend_rows = -(-(len(runs) + (1 if band else 0)) // ncol)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.subplots_adjust(top=0.80 - 0.045 * (legend_rows - 1), bottom=0.15,
                        left=0.06, right=0.985, wspace=0.3)

    # (A) full-trajectory time; cold start and checkpoint resumes ringed (excluded from the means)
    P._panel(axA, df, sty, "cycle_s", lambda g: g["cycle_s"],
             "(A) Time per trajectory (full trajectory)", "seconds")
    first = df[(df["cold"] == 1) | (df["resume"] == 1)]
    # The rings are explained in the doc's caption: an in-panel note lands on the data.
    axA.plot(first["traj_abs"], first["cycle_s"], marker="o", ms=13, mfc="none", mec=P.MUTED,
             mew=1.2, ls="none", zorder=5)

    # (B) plaquette, zoomed out to 3x the spread as in plot_accept_runs
    P._panel(axB, df, sty, "plaquette", lambda g: g["plaquette"], "(B) Plaquette (unsmeared)",
             "plaquette")
    axB.ticklabel_format(useOffset=False, axis="y")
    pmean, pspread = df["plaquette"].mean(), df["plaquette"].max() - df["plaquette"].min()
    pad = max(pspread * 0.8, 0.0001)
    axB.set_ylim(pmean - pad, pmean + pad)
    _band(axB, band)

    # (C) dH, linear: every run here stays O(1)
    P._line_acc_rej(axC, df, sty, "dH")
    axC.axhline(0.0, color=P.MUTED, lw=0.8, ls="--", alpha=0.6, zorder=1)
    lo, hi = df["dH"].min(), df["dH"].max()
    axC.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
    P._finish(axC, "(C) ΔH  (open marker = rejected)", "ΔH")

    xs = sorted(df["traj_abs"].unique())
    for ax in (axA, axB, axC):
        ax.set_xticks(xs[::2])
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("trajectory (from cfg_2000)", fontsize=9, color=P.MUTED)

    handles = [Line2D([0], [0], color=sty[r][0], marker=sty[r][1], lw=1.8, ms=7,
                      markeredgecolor="white", label=display.get(r, r)) for r in runs]
    if band:
        handles.append(Patch(facecolor=BAND, edgecolor="none", label=band[2]))
    fig.text(0.01, 0.98, title, fontsize=13, color=P.INK, ha="left", va="top")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False,
               fontsize=10, ncol=ncol)
    fig.savefig(out_png, dpi=200, facecolor="white")
    print("wrote %s" % out_png)


if __name__ == "__main__":
    main(sys.argv[1:])
