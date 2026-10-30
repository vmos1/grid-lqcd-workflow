#!/usr/bin/env python3
"""Wilson-flow scales per stream against the production ensemble (tune1-perturbations, 2026-10-29):
one figure, two panels (t0/a², w0/a), one row per stream. Small markers are single configurations,
the large marker with its bar is the stream mean ± standard error (sd/√n); the light grey band is the
production mean ± one configuration's spread (sd over the production configurations), the darker
inner band ± sd/√4, the spread expected for a mean of 4 independent configurations (a stream's size;
a stream's configurations come from one short chain and are not fully independent, so its mean
belongs somewhere between the two bands).

Reads wflow_measure's output (`WFLOW <cfg> flow <step> <t> <t2E_clover> <t2E_plaq>` lines). t0 and w0
as in runs/2026_10_26_wflow_driver/wflow_streams.py: clover t²E = 0.3 and t d/dt t²E = 0.3, linear
interpolation, central differences. A configuration flowed in more than one log is used once.

Colour follows the run as in plot_candidates_2x2.py (Okabe-Ito slots); Grid filled, Chroma open;
production and cfg_2000 in grey. Rows are named on the axis, so identity never rests on colour.

On stdout, two markdown tables: the per-stream means (t0, w0) and, from 2026-10-30, the production
configurations one by one with their plaquettes, mean and spread, in the doc's §5 format (for
check_doc_tables.py).

Usage:
  plot_wflow_scales.py <out.png> <flow log> [<flow log> ...]
"""
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accept_runs as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (P sets the Agg backend first)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

GREY = "#8c8c8c"
BAND = "#e6e6e6"
BAND_IN = "#cfcfcf"
N_STREAM = 4  # configurations per tune 1 stream: the inner band is sd/sqrt(N_STREAM)
SLOT = {"baseline": ("#0072B2", "o"), "B1_t48": ("#E69F00", "s"), "B2_t48": ("#009E73", "^"),
        "B2_t24": ("#D55E00", "D"), "chain 2": ("#CC79A7", "v")}
ENS = "cl21_48_96_b6p3_m0p2416_m0p2050"
# (group, row label, path fragment, colour, marker, filled); top to bottom, first match wins
ROWS = [
    ("production", "production djm-1", "cfgs48/%s-djm-1_" % ENS, GREY, "o", True),
    ("production", "production djm-2", "cfgs48/%s-djm-2_" % ENS, GREY, "o", True),
    ("production", "production djm-3", "cfgs48/%s-djm-3_" % ENS, GREY, "o", True),
    ("start", "start: cfg_2000", "data/%s-djm-3_cfg_2000" % ENS, P.INK, "*", True),
    ("Grid", "Grid baseline", "2026_10_7_baseG_trajl1_seed300_20/", *SLOT["baseline"], True),
    ("Grid", "Grid B1_t48", "_B1_md12_s300", *SLOT["B1_t48"], True),
    ("Grid", "Grid B2_t48", "_B2_md12_s30", *SLOT["B2_t48"], True),
    ("Grid", "Grid B2_t24", "_B2_tail24_s30", *SLOT["B2_t24"], True),
    ("Chroma", "Chroma baseline chain 2", "chroma_cfgs/baseg_mds12_base_20_s301/", *SLOT["chain 2"], False),
    ("Chroma", "Chroma B1_t48", "chroma_baseg_mds12_b1_20_s300_cfg", *SLOT["B1_t48"], False),
    ("Chroma", "Chroma B2_t24", "chroma_baseg_mds12_b2t24_20_s300_cfg", *SLOT["B2_t24"], False),
]
pat_flow = re.compile(r"WFLOW (\S+) flow (\d+) (\S+) (\S+) (\S+)")
pat_plaq = re.compile(r"WFLOW (\S+) (plaq0|plaq0_smeared) (\S+)")
pat_prod = re.compile(r"-(djm-\d)_cfg_(\d+)\.lime")


def plaquettes(logs):
    """{cfg path: {"plaq0": x, "plaq0_smeared": y}} from the flow program's own lines."""
    out = {}
    for log in logs:
        for line in open(log, errors="replace"):
            m = pat_plaq.search(line)
            if m:
                out.setdefault(m.group(1), {}).setdefault(m.group(2), float(m.group(3)))
    return out


def sci1(x):
    """3.0e-5 -> '3 ×10⁻⁵' (one significant digit), as the doc prints a plaquette spread."""
    e = int(math.floor(math.log10(abs(x))))
    d = int(round(x / 10 ** e))
    if d == 10:
        d, e = 1, e + 1
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return "%d ×10%s" % (d, sup)


def production_table(vals, plaq):
    """The doc's §5 production table: one row per configuration, the mean of all with its
    standard error (one digit, as candidate_tables.one_digit), and one configuration's sd."""
    from candidate_tables import one_digit
    prod = []
    for cfg, (t0, w0) in vals.items():
        m = pat_prod.search(cfg)
        if m and "cfgs48/" in cfg and cfg in plaq:
            prod.append(((m.group(1), int(m.group(2))), plaq[cfg]["plaq0"],
                         plaq[cfg]["plaq0_smeared"], t0, w0))
    if not prod:
        return
    prod.sort()
    print()
    print("| production configuration | thin plaquette | smeared plaquette | t0/a² | w0/a |")
    print("|---|---|---|---|---|")
    for (s, n), p, ps, t0, w0 in prod:
        print("| %s %d | %.6f | %.6f | %.4f | %.4f |" % (s, n, p, ps, t0, w0))
    cols = [[r[k] for r in prod] for k in range(1, 5)]
    n = len(prod)
    print("| **mean of %d** | %s |" % (n, " | ".join(
        "**%s**" % one_digit(st.mean(c), st.stdev(c) / math.sqrt(n)) for c in cols)))
    sd = [st.stdev(c) for c in cols]
    print("| spread, one configuration (sd) | %s | %s | %.3f (%.1f%%) | %.3f (%.1f%%) |" % (
        sci1(sd[0]), sci1(sd[1]), sd[2], 100 * sd[2] / st.mean(cols[2]),
        sd[3], 100 * sd[3] / st.mean(cols[3])))


def crossing(ts, ys, target):
    for i in range(1, len(ys)):
        if (ys[i - 1] - target) * (ys[i] - target) <= 0 and ys[i] != ys[i - 1]:
            return ts[i - 1] + (target - ys[i - 1]) / (ys[i] - ys[i - 1]) * (ts[i] - ts[i - 1])
    return None


def scales(logs):
    """{cfg path: (t0, w0)}, each configuration once (first log that has it)."""
    out = {}
    for log in logs:
        flow = {}
        for line in open(log, errors="replace"):
            m = pat_flow.search(line)
            if m:
                flow.setdefault(m.group(1), []).append((float(m.group(3)), float(m.group(4))))
        for cfg, pts in flow.items():
            if cfg in out:
                continue
            pts.sort()
            t = [a for a, _ in pts]
            y = [b for _, b in pts]
            w = [t[i] * (y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1]) for i in range(1, len(t) - 1)]
            out[cfg] = (crossing(t, y, 0.3), math.sqrt(crossing(t[1:-1], w, 0.3)))
    return out


def main(out_png, logs):
    vals = scales(logs)
    rows = {r[1]: [] for r in ROWS}
    for cfg, tw in vals.items():
        hit = next((r for r in ROWS if r[2] in cfg), None)
        if hit is None:
            print("skipped (no row): %s" % cfg, file=sys.stderr)
            continue
        rows[hit[1]].append(tw)
    ROW = [r for r in ROWS if rows[r[1]]]

    # y positions: one unit per row, a gap between groups
    ys, y, prev = {}, 0.0, None
    seps = []
    for g, lab, *_ in ROW:
        if prev is not None and g != prev:
            seps.append(y + 0.2)  # midway between the last row (y + 1) and the next (y - 0.6)
            y -= 0.6
        ys[lab] = y
        y -= 1.0
        prev = g

    prod = [tw for r in ROW if r[0] == "production" for tw in rows[r[1]]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.8), sharey=True)
    fig.subplots_adjust(top=0.76, bottom=0.09, left=0.20, right=0.98, wspace=0.08)
    print("| code | stream | configurations | t0/a² | w0/a |")
    print("|---|---|---|---|---|")
    for k, (ax, name) in enumerate(zip(axes, ("t0/a²", "w0/a"))):
        if prod:
            pv = [tw[k] for tw in prod]
            m, sd = st.mean(pv), st.stdev(pv)
            ax.axvspan(m - sd, m + sd, color=BAND, zorder=0, lw=0)
            # inner band: the spread of a mean of N_STREAM independent configurations
            ax.axvspan(m - sd / math.sqrt(N_STREAM), m + sd / math.sqrt(N_STREAM), color=BAND_IN,
                       zorder=0, lw=0)
            ax.axvline(m, color=P.MUTED, ls="--", lw=0.9, zorder=1)
        for g, lab, _, c, mk, filled in ROW:
            v = [tw[k] for tw in rows[lab]]
            yy = ys[lab]
            face = c if filled else "white"
            ms = 13 if mk == "*" else 9
            if len(v) > 1:
                ax.plot(v, [yy - 0.28] * len(v), marker=mk, ms=4.5, ls="none", color=c,
                        mfc=face, mec=c, mew=1.0, alpha=0.75, zorder=3)
                se = st.stdev(v) / math.sqrt(len(v))
                ax.errorbar(st.mean(v), yy, xerr=se, fmt=mk, ms=ms, color=c, mfc=face, mec=c,
                            mew=1.6, capsize=3, elinewidth=1.4, zorder=4)
            else:
                se = None
                ax.plot(v, [yy], marker=mk, ms=ms, ls="none", color=c, mfc=face, mec=c, mew=1.6,
                        zorder=4)
            if k == 0:
                t0s = [tw[0] for tw in rows[lab]]
                w0s = [tw[1] for tw in rows[lab]]
                cell = (lambda x: "%.4f ± %.4f" % (st.mean(x), st.stdev(x) / math.sqrt(len(x)))
                        if len(x) > 1 else "%.4f" % x[0])
                stream = lab[len(g):].strip(" :") if lab.startswith(g) else lab
                print("| %s | %s | %d | %s | %s |" % (g, stream, len(v), cell(t0s), cell(w0s)))
        for s in seps:
            ax.axhline(s, color=P.GRID, lw=0.8, zorder=1)
        ax.set_title("(%s) %s" % ("AB"[k], "t0/a²  (t²E = 0.3)" if k == 0 else
                                  "w0/a  (t d/dt t²E = 0.3)"), fontsize=11, color=P.INK, pad=8,
                     loc="left")
        ax.set_xlabel(name, fontsize=9, color=P.MUTED)
        ax.grid(True, axis="x", color=P.GRID, lw=0.6, alpha=0.7, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(P.GRID)
        ax.tick_params(colors=P.MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ys[r[1]] for r in ROW])
    axes[0].set_yticklabels(["%s (%d)" % (r[1], len(rows[r[1]])) for r in ROW], fontsize=9,
                            color=P.INK)
    axes[0].set_ylim(min(ys.values()) - 0.8, max(ys.values()) + 0.6)

    handles = [
        Line2D([0], [0], marker="o", ms=9, color=GREY, ls="-", lw=1.4,
               label="stream mean ± standard error (n configurations in brackets)"),
        Line2D([0], [0], marker="o", ms=4.5, color=GREY, ls="none", alpha=0.75,
               label="single configuration"),
        Patch(facecolor=BAND, edgecolor="none",
              label="production: mean (dashed) ± sd of single configurations (%d)" % len(prod)),
        Patch(facecolor=BAND_IN, edgecolor="none",
              label="± sd/√%d: expected spread of a mean of %d independent configurations"
              % (N_STREAM, N_STREAM)),
        Line2D([0], [0], marker="o", ms=8, color=GREY, mfc="white", mew=1.6, ls="none",
               label="open marker = Chroma, filled = Grid"),
    ]
    fig.text(0.01, 0.98, "Wilson-flow scales: tune 1 streams against the production ensemble "
             "(48³×96, clover E)", fontsize=13, color=P.INK, ha="left", va="top")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.94), frameon=False,
               fontsize=9, ncol=2)
    fig.savefig(out_png, dpi=200, facecolor="white")
    production_table(vals, plaquettes(logs))
    print("wrote %s" % out_png, file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
