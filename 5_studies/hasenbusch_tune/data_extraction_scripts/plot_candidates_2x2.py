#!/usr/bin/env python3
"""2x2 per-trajectory figure for a candidate-vs-baseline comparison (tune1-perturbations, 2026-10-27),
from the CSV of `compare_runs_multiwindow.py --csv` (Grid) or `chroma_runs_csv.py` (Chroma):

  (A) time per trajectory (full trajectory, cycle_s)   (B) plaquette (unsmeared)
  (C) dH (open marker = rejected)                       (D) running mean of the plaquette

Uses `plot_accept_runs.py`'s helpers (style map, panel drawing, accept/reject fill, axis styling)
unchanged, so colours, marker shapes and conventions match every other figure in __docs/figs/.
Panel D replaces that module's max-kick panel: the fermion kicks are not in today's logs, and the
running mean shows directly whether the chains settle to a common plaquette.

Colour follows the run, given in fixed order by --order (Okabe-Ito slots, validated CVD-safe; each
run also has its own marker shape, so identity never rests on colour alone). Pass the same labels in
the same slots for the Grid and the Chroma figure so an entity keeps its colour across both.

Usage:
  plot_candidates_2x2.py <in.csv> <out.png> --order 'L1,L2,...' [--title '...'] [--display 'L1=text;L2=text']
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accept_runs as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (P sets the Agg backend first)
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SLOTS = [("#0072B2", "o"), ("#E69F00", "s"), ("#009E73", "^"), ("#D55E00", "D"), ("#CC79A7", "v")]


def main(argv):
    csv_path, out_png = argv[0], argv[1]
    order, title, display = None, "", {}
    i = 2
    while i < len(argv):
        if argv[i] == "--order":
            order = argv[i + 1].split(",")
        elif argv[i] == "--title":
            title = argv[i + 1]
        elif argv[i] == "--display":
            display = dict(kv.split("=", 1) for kv in argv[i + 1].split(";"))
        i += 2
    df = pd.read_csv(csv_path)
    runs = order or list(dict.fromkeys(df["run"]))
    df = df[df["run"].isin(runs)].copy()
    df["run"] = pd.Categorical(df["run"], categories=runs, ordered=True)
    df = df.sort_values(["run", "traj_abs"])
    sty = {r: SLOTS[k] for k, r in enumerate(runs)}

    ncol = len(runs) if len(runs) <= 3 else 3
    legend_rows = -(-len(runs) // ncol)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    fig.subplots_adjust(top=0.85 - 0.035 * (legend_rows - 1), bottom=0.08,
                        left=0.08, right=0.97, hspace=0.5, wspace=0.28)
    (axA, axB), (axC, axD) = axes

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

    # (C) dH, linear: every run here stays O(1)
    P._line_acc_rej(axC, df, sty, "dH")
    axC.axhline(0.0, color=P.MUTED, lw=0.8, ls="--", alpha=0.6, zorder=1)
    lo, hi = df["dH"].min(), df["dH"].max()
    axC.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
    P._finish(axC, "(C) ΔH  (open marker = rejected)", "ΔH")

    # (D) running mean of the plaquette along each chain
    P._panel(axD, df, sty, "plaquette",
             lambda g: g["plaquette"].expanding().mean(), "(D) Running mean of the plaquette",
             "mean plaquette, trajectories 2001…n")
    axD.ticklabel_format(useOffset=False, axis="y")

    xs = sorted(df["traj_abs"].unique())
    for ax in (axA, axB, axC, axD):
        ax.set_xticks(xs[::2])
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("trajectory (from cfg_2000)", fontsize=9, color=P.MUTED)

    handles = [Line2D([0], [0], color=sty[r][0], marker=sty[r][1], lw=1.8, ms=7,
                      markeredgecolor="white", label=display.get(r, r)) for r in runs]
    fig.text(0.01, 0.98, title, fontsize=13, color=P.INK, ha="left", va="top")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), frameon=False,
               fontsize=10, ncol=ncol)
    fig.savefig(out_png, dpi=200, facecolor="white")
    print("wrote %s" % out_png)


if __name__ == "__main__":
    main(sys.argv[1:])
