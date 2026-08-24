#!/usr/bin/env python3
"""2x2 per-trajectory comparison plots for the acceptance runs.

Reads the tidy CSV written by `accept_run_tables.py --csv` and draws one figure
with four panels vs trajectory number, one coloured line+marker series per run:

  (A) wall time (s)        (B) plaquette (unsmeared)
  (C) dH (symlog)          (D) max kick max|F.dt| (integrator limiter)

Panels C and D share the accept/reject marker fill (filled = accepted, open =
Metropolis-rejected), so the reader can see the rejections line up with the kick
spikes. `limiter_kick` = the single largest |F.dt| across ALL sectors and steps of
the trajectory (each sector's F.dt already uses that sector's own dt from the
nested integrator), i.e. the stability-limiting kick. Panel D used to plot the
summed force-solver TIME, but that is ~97% of the wall time and duplicated (A).

Design notes (colorblind-safe, per the dataviz skill):
  * Okabe-Ito categorical hues, assigned to runs in FIXED order (never cycled),
    validated CVD-safe (worst adjacent deutan dE 11.0).
  * Each run also gets a distinct MARKER SHAPE -> identity never rests on colour
    alone (covers the low contrast-vs-surface WARN on orange/pink, CVD, and print).
  * dH panel: FILLED marker = accepted, OPEN marker = Metropolis-rejected.
  * Trajectory 1 is the cold start (one-time init inflates wall / Sigma force); it
    is drawn with a ringed marker and annotated, not silently mixed in.

Adding runs later (3-level, the 2011-2020 extensions, mass-scan points) needs no
code change: just include their logs when building the CSV; unknown runs get the
next palette slot. Panel D's quantity is easily swapped for `limiter_kick` (also
in the CSV) if the max-kick view is wanted instead of total force cost.

Usage:
  plot_accept_runs.py <in.csv> [out.png]
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Fixed run -> (hue, marker) order. Okabe-Ito; extend the list, never recolour.
PALETTE = [
    ("base+G",  "#0072B2", "o"),
    ("u1",      "#E69F00", "s"),
    ("3-level", "#009E73", "^"),
    ("_extra1", "#D55E00", "D"),
    ("_extra2", "#CC79A7", "v"),
]
# Poster-facing display labels (legend text only) -- internal run names above
# (from accept_run_tables.py's run_name()) stay as-is so PALETTE/style_map
# keying and every other doc that reads these CSVs are unaffected.
DISPLAY = {"base+G": "1_baseline", "u1": "2_sparse-ladder", "3-level": "3_threelevel"}
# Figure title. Module-level so a caller that imports this as a module can
# override it (together with PALETTE/DISPLAY) for a different run set without
# forking the whole plotting routine -- e.g. runs that share a seed and start
# config but are NOT a matched draw, where "matched draw" would be wrong. The
# default is unchanged, so every existing CLI invocation renders identically.
TITLE = "HMC run comparison — per-trajectory diagnostics (matched draw, seed 300)"
# Panel C y-scale. "symlog" is right when a run blows up (the 2_sparse-ladder
# comparison had dH to +1022, which a linear axis flattens the rest against);
# "linear" is right when every run stays O(1), where symlog's linthresh=1
# instead squashes the entire signal into the linear stub around zero.
DH_SCALE = "symlog"
# Panel D shaded reference band, 0 -> KICK_BAND. Default 0.10 is the nominal
# healthy max|F.dt|. Set to a specific run's own proven-safe maximum (and give
# KICK_BAND_LABEL) when the point of the figure is "does this candidate stay
# inside what the baseline already demonstrated".
KICK_BAND = 0.10
KICK_BAND_LABEL = None
# Legend columns. None = auto: one row while it fits, two columns past three
# runs. These labels carry the integrator config ("C3 = w3 + tail on strange
# level (3-level, MDSTEPS=6)"), so four of them on one row overrun the 11"
# canvas and get clipped at BOTH edges -- and fig.legend gives no warning when
# it does. Wrapping costs one row of top margin, which main() reserves.
LEGEND_NCOL = None
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"

def style_map(runs):
    """Assign each present run a stable (colour, marker) by first appearance in
    PALETTE, then by leftover slots for names not in PALETTE."""
    known = {name: (c, m) for name, c, m in PALETTE}
    leftovers = [(c, m) for name, c, m in PALETTE if name.startswith("_")]
    out, li = {}, 0
    for r in runs:
        if r in known and not r.startswith("_"):
            out[r] = known[r]
        else:
            out[r] = leftovers[li % len(leftovers)]; li += 1
    return out

def _panel(ax, df, sty, ycol, transform, title, ylabel):
    for run, g in df.groupby("run", sort=False):
        c, mk = sty[run]
        x = g["traj_abs"].to_numpy()
        y = transform(g)
        ax.plot(x, y, "-", color=c, lw=1.8, alpha=0.9, zorder=2)
        ax.plot(x, y, marker=mk, ms=7, color=c, ls="none", zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
    _finish(ax, title, ylabel)

def _line_acc_rej(ax, df, sty, ycol):
    """Line + markers, FILLED where accepted and OPEN (white face) where the
    trajectory was Metropolis-rejected. Used for dH and the max-kick panel so a
    reject and its cause line up visually."""
    for run, g in df.groupby("run", sort=False):
        c, mk = sty[run]
        x = g["traj_abs"].to_numpy(); y = g[ycol].to_numpy()
        acc = (g["accepted"] == 1).to_numpy()
        ax.plot(x, y, "-", color=c, lw=1.8, alpha=0.9, zorder=2)
        ax.plot(x[acc], y[acc], marker=mk, ms=7, color=c, ls="none",
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        if (~acc).any():
            ax.plot(x[~acc], y[~acc], marker=mk, ms=8, mfc="white", mec=c,
                    mew=1.6, ls="none", zorder=4)

def _finish(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
    ax.set_xlabel("configuration # (continues cfg_2000)", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

def main(csv_path, out_png):
    df = pd.read_csv(csv_path)
    runs = list(dict.fromkeys(df["run"]))
    sty = style_map(runs)

    # constrained_layout is deliberately OFF: its auto text-metric layout
    # recomputes per savefig() call using each backend's own font-metrics
    # path, which measures text extents differently for a raster (PNG) vs
    # vector (PDF) target -- this previously caused the reserved top margin
    # (via a constrained_layout `rect`) to land correctly in the PNG but
    # overlap the title/legend in the PDF from the very same figure object.
    # Fixed figure-fraction margins below render byte-for-byte identically
    # across every output format.
    ncol = LEGEND_NCOL or (len(runs) if len(runs) <= 3 else 2)
    legend_rows = -(-len(runs) // ncol)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    fig.subplots_adjust(top=0.84 - 0.03 * (legend_rows - 1), bottom=0.08,
                         left=0.07, right=0.97, hspace=0.55, wspace=0.28)
    (axA, axB), (axC, axD) = axes

    # (A) wall time in seconds
    _panel(axA, df, sty, "wall_s", lambda g: g["wall_s"],
           "(A) Wall time per trajectory", "wall time (s)")
    # (B) plaquette (unsmeared) -- zoomed out a bit around the mean (3x the
    # actual spread) so the wobble is still clearly visible while the axis
    # range makes clear it's a small fluctuation, not drift. Auto-scaling
    # tight to the data would exaggerate the wobble; zooming out too far (was
    # tried at 25x) flattens it to invisible -- this is the middle ground.
    _panel(axB, df, sty, "plaquette", lambda g: g["plaquette"],
           "(B) Plaquette (unsmeared)", "plaquette")
    axB.ticklabel_format(useOffset=False, axis="y")
    pmean = df["plaquette"].mean()
    pspread = df["plaquette"].max() - df["plaquette"].min()
    pad = max(pspread * 3, 0.0003)
    axB.set_ylim(pmean - pad, pmean + pad)

    # (C) dH with accept/reject marker fill; see DH_SCALE for why the scale is
    # a knob rather than always symlog.
    _line_acc_rej(axC, df, sty, "dH")
    axC.axhline(0.0, color=MUTED, lw=0.8, ls="--", alpha=0.6, zorder=1)
    if DH_SCALE == "symlog":
        axC.set_yscale("symlog", linthresh=1.0)
        axC.set_ylim(-3, df["dH"].max() * 3)  # headroom so the largest reject marker isn't clipped
        _finish(axC, "(C) ΔH  (symlog; open marker = rejected)", "dH")
    else:
        lo, hi = df["dH"].min(), df["dH"].max()
        pad = max((hi - lo) * 0.15, 1e-6)
        axC.set_ylim(lo - pad, hi + pad)
        _finish(axC, "(C) ΔH  (open marker = rejected)", "dH")

    # (D) max kick |F.dt| per trajectory (the integrator limiter); same acc/rej fill
    _line_acc_rej(axD, df, sty, "limiter_kick")
    axD.axhspan(0.0, KICK_BAND, color=GRID, alpha=0.35, zorder=0)
    if KICK_BAND_LABEL:
        # Sit the label halfway DOWN the band, not on its top edge: the top edge
        # is exactly where the runs being compared against it cluster, so a label
        # there lands on the data. The band's interior is empty by construction.
        axD.annotate(KICK_BAND_LABEL, xy=(0.99, KICK_BAND * 0.5),
                     xycoords=("axes fraction", "data"),
                     fontsize=8, color=MUTED, ha="right", va="center")
    _finish(axD, "(D) Max kick  max|F·dt|  (open = rejected)", "max |F·dt|")

    # ring + annotate the cold trajectory on the wall-time panel only
    cold = df[df["cold"] == 1]
    axA.plot(cold["traj_abs"], cold["wall_s"], marker="o", ms=13,
             mfc="none", mec=MUTED, mew=1.2, ls="none", zorder=5)
    axA.annotate("traj 1 = cold\n(one-time init)", xy=(0.14, 0.92),
                 xycoords="axes fraction", fontsize=8, color=MUTED, ha="center")

    # ring + annotate the checkpoint-resume trajectory too, if the CSV has it
    # (older CSVs predating this column just won't draw anything here).
    if "resume" in df.columns:
        resume = df[df["resume"] == 1]
        if len(resume):
            axA.plot(resume["traj_abs"], resume["wall_s"], marker="o", ms=13,
                     mfc="none", mec=MUTED, mew=1.2, ls="none", zorder=5)
            rx = resume["traj_abs"].iloc[0]
            axA.annotate("resume\n(fresh solver\ncaches)", xy=(rx, resume["wall_s"].iloc[0]),
                         xytext=(rx + 0.4, resume["wall_s"].iloc[0]),
                         fontsize=8, color=MUTED, ha="left", va="center")

    # integer trajectory ticks
    xs = sorted(df["traj_abs"].unique())
    for ax in (axA, axB, axC, axD):
        ax.set_xticks(xs); ax.tick_params(axis="x", rotation=45)

    # one figure-level legend (colour + marker = run)
    handles = [Line2D([0], [0], color=sty[r][0], marker=sty[r][1], lw=1.8, ms=7,
                      markeredgecolor="white", label=DISPLAY.get(r, r)) for r in runs]
    fig.text(0.01, 0.98, TITLE, fontsize=13, color=INK, ha="left", va="top")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               frameon=False, fontsize=10, ncol=ncol)

    # No bbox_inches="tight": that crops to a per-backend-recomputed tight
    # bounding box (raster vs vector renderers measure text extents
    # differently), which is what caused the PNG and PDF from this same
    # figure to crop -- and so visually overlap the title/legend --
    # differently. The fixed subplots_adjust margins above already reserve
    # the right whitespace, so saving the full canvas as-is renders
    # identically in both formats.
    fig.savefig(out_png, dpi=300, facecolor="white")
    print(f"wrote {out_png}")
    # Vector PDF alongside the PNG: PNG is for inline doc/markdown preview
    # (not all viewers render PDF inline), PDF is what should actually be
    # pasted into Keynote for a poster -- it stays crisp at any print size
    # instead of the fixed-resolution raster softening when blown up.
    out_pdf = out_png[:-4] + ".pdf" if out_png.endswith(".png") else out_png + ".pdf"
    fig.savefig(out_pdf, facecolor="white")
    print(f"wrote {out_pdf}")

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "accept_runs.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "accept_runs_2x2.png"
    main(csv_path, out_png)
