#!/usr/bin/env python3
"""Force per monomial, candidates vs baseline (tune1-perturbations, 2026-10-29), from Grid hmc.log.

Per trajectory and monomial, from the integrator's per-step lines:
  average force = mean over the MD steps of "Force average" (|F| RMS over sites, sqrt(Σ|F|²/V): L089)
  peak force    = max over the MD steps of "Force max" (largest single-site |F|)
Then over the completed trajectories of each run: mean ± standard error (sd/√n), and the largest
single-trajectory peak. Raw forces, before the stout pullback for the smeared fermion monomials;
the same in every run, so runs compare like with like.

Monomials are named by role, not by (level, index): B2_t24 has a different level layout. Rungs are
numbered from the light end (rung 4 ends on the tail mass, which differs between candidates); the two
log-dets are told apart by order of appearance (1-flavour first, as in accept_run_tables.py).
A trajectory with no "Total H after" line (an aborted solve) is left out and listed on stderr.

Output: a two-panel figure (A average, B peak; log scale) and, on stdout, the markdown table.
Colours and markers as plot_candidates_2x2.py (Okabe-Ito slots, fixed order).

Usage:
  plot_force_by_monomial.py <out.png> --run 'LABEL=log1,log2,...' [--run ...] [--title '...']
"""
import collections
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accept_runs as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (P sets the Agg backend first)
from matplotlib.lines import Line2D  # noqa: E402

SLOTS = [("#0072B2", "o"), ("#E69F00", "s"), ("#009E73", "^"), ("#D55E00", "D"), ("#CC79A7", "v")]
FORCE = re.compile(r"\[(\d)\]\[(\d)\] Force (average|max)\s*(?:\([^)]*\))?\s*:\s*([0-9.eE+-]+)\s*(.*)$")
TRAJ = re.compile(r"# Trajectory = (\d+)")
HAFT = re.compile(r"Total H after trajectory\s*=\s*\S+\s+dH = ([0-9.eE+-]+)")
ROLES = ["rung 1", "rung 2", "rung 3", "rung 4", "tail", "strange", "log-det 1f", "log-det 2f", "gauge"]


def role_map(names):
    """{(lv, ix): role} from the action names of one log."""
    out, rungs, ldets = {}, [], []
    for k, nm in names.items():
        if "Rational" in nm:
            out[k] = "strange"
        elif "LogDet" in nm:
            ldets.append(k)
        elif "PlaqPlusRectangle" in nm:
            out[k] = "gauge"
        elif "RatioAction" in nm:
            m = re.search(r"det\(([-0-9.]+)\) / det", nm)
            rungs.append((float(m.group(1)), k))
        elif "TwoFlavourSchurClover" in nm:
            out[k] = "tail"
    for i, (_, k) in enumerate(sorted(rungs)):  # lightest numerator first
        out[k] = "rung %d" % (i + 1)
    for i, k in enumerate(ldets):
        out[k] = "log-det 1f" if i == 0 else "log-det 2f"
    return out


def parse(paths):
    """{traj: {"avg": {role: [..]}, "max": {role: [..]}, "dH": x or None}}; aborted kept as 'N*'."""
    out = collections.OrderedDict()
    for p in paths:
        names, rows, cur = collections.OrderedDict(), [], None
        for line in open(p, errors="ignore"):
            m = TRAJ.search(line)
            if m:
                # "# Trajectory = N" opens the trajectory that writes configuration N+1
                cur = {"t": int(m.group(1)) + 1, "avg": collections.defaultdict(list),
                       "max": collections.defaultdict(list), "dH": None}
                rows.append(cur)
                continue
            if cur is None:
                continue
            m = FORCE.search(line)
            if m:
                k = (m.group(1), m.group(2))
                names.setdefault(k, m.group(5).strip())
                cur["avg" if m.group(3) == "average" else "max"][k].append(float(m.group(4)))
                continue
            m = HAFT.search(line)
            if m:
                cur["dH"] = float(m.group(1))
        roles = role_map(names)
        for r in rows:
            d = {"avg": {roles[k]: v for k, v in r["avg"].items()},
                 "max": {roles[k]: v for k, v in r["max"].items()}, "dH": r["dH"]}
            key = r["t"] if r["dH"] is not None else "%d*" % r["t"]
            out[key] = d  # a later segment's trajectory replaces an earlier one
    return out


def stats(data):
    """{role: (avg mean, avg se, peak mean, peak se, largest peak, its traj, n)} over completed."""
    res = {}
    done = [(t, d) for t, d in data.items() if d["dH"] is not None]
    for r in ROLES:
        a = [st.mean(d["avg"][r]) for _, d in done if d["avg"].get(r)]
        pk = [(max(d["max"][r]), t) for t, d in done if d["max"].get(r)]
        if not a:
            continue
        pv = [x for x, _ in pk]
        top, tt = max(pk)
        res[r] = (st.mean(a), st.stdev(a) / math.sqrt(len(a)), st.mean(pv),
                  st.stdev(pv) / math.sqrt(len(pv)), top, tt, len(pv))
    return res


def fmt(v, e):
    """Value with a one-digit error in brackets: 0.82(7), 10.69(3), 0.0032(1)."""
    if e <= 0:
        return "%.4g" % v
    d = max(0, -int(math.floor(math.log10(e))))
    ed = round(e * 10 ** d)
    if ed >= 10:
        d, ed = d - 1, round(e * 10 ** (d - 1))
    return "%.*f(%d)" % (max(d, 0), v, ed)


def main(argv):
    out_png, runs, title = argv[0], [], "Grid: force per monomial, baseline vs B candidates"
    i = 1
    while i < len(argv):
        if argv[i] == "--run":
            lab, paths = argv[i + 1].split("=", 1)
            runs.append((lab, paths.split(",")))
        elif argv[i] == "--title":
            title = argv[i + 1]
        i += 2
    allstats = []
    for lab, paths in runs:
        data = parse(paths)
        for t, d in data.items():
            if d["dH"] is None:
                print("%s: trajectory %s aborted, left out" % (lab, t), file=sys.stderr)
        allstats.append((lab, stats(data)))

    roles = [r for r in ROLES if any(r in s for _, s in allstats)]
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)
    fig.subplots_adjust(top=0.83, bottom=0.08, left=0.08, right=0.98, hspace=0.25)
    width = 0.17
    for j, (lab, s) in enumerate(allstats):
        c, mk = SLOTS[j]
        off = (j - (len(allstats) - 1) / 2) * width
        xs = [k + off for k, r in enumerate(roles) if r in s]
        rr = [r for r in roles if r in s]
        axA.errorbar(xs, [s[r][0] for r in rr], yerr=[s[r][1] for r in rr], fmt=mk, ms=6.5, color=c,
                     mec="white", mew=0.8, elinewidth=1.3, capsize=2.5, zorder=3)
        axB.errorbar(xs, [s[r][2] for r in rr], yerr=[s[r][3] for r in rr], fmt=mk, ms=6.5, color=c,
                     mec="white", mew=0.8, elinewidth=1.3, capsize=2.5, zorder=3)
        axB.plot(xs, [s[r][4] for r in rr], marker=mk, ms=6, ls="none", mfc="white", mec=c, mew=1.3,
                 zorder=2)
    for ax, ttl, yl in ((axA, "(A) Average force: |F| RMS over sites, mean over the MD steps",
                         "average |F|"),
                        (axB, "(B) Peak force (largest single-site |F| in a trajectory): mean and "
                              "max over the trajectories", "peak |F|")):
        ax.set_yscale("log")
        P._finish(ax, ttl, yl)
        ax.set_xlabel("")
        for k in range(1, len(roles)):
            ax.axvline(k - 0.5, color=P.GRID, lw=0.6, zorder=0)
        ax.grid(False, axis="x")
    axB.set_xticks(range(len(roles)))
    axB.set_xticklabels(roles, fontsize=9, color=P.INK)
    axB.set_xlim(-0.5, len(roles) - 0.5)

    handles = [Line2D([0], [0], color=SLOTS[j][0], marker=SLOTS[j][1], ls="none", ms=7,
                      mec="white", label=lab) for j, (lab, _) in enumerate(allstats)]
    handles += [Line2D([0], [0], color=P.MUTED, marker="o", ls="-", lw=1.2, ms=6,
                       label="mean over the trajectories ± standard error"),
                Line2D([0], [0], color=P.MUTED, marker="o", ls="none", ms=6, mfc="white",
                       label="(B) max over the trajectories")]
    fig.text(0.01, 0.98, title, fontsize=13, color=P.INK, ha="left", va="top")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), frameon=False,
               fontsize=9.5, ncol=3)
    fig.savefig(out_png, dpi=200, facecolor="white")
    print("wrote %s" % out_png, file=sys.stderr)

    # markdown table: peak force, mean ± standard error and the largest (trajectory)
    labs = [lab for lab, _ in allstats]
    print("| monomial | " + " | ".join(labs) + " |")
    print("|---|" + "---|" * len(labs))
    for r in roles:
        cells = []
        for _, s in allstats:
            if r not in s:
                cells.append("—")
                continue
            v = s[r]
            cells.append("%s, %.3g (%s)" % (fmt(v[2], v[3]), v[4], v[5]))
        print("| %s | %s |" % (r, " | ".join(cells)))
    print()
    print("| monomial (average force) | " + " | ".join(labs) + " |")
    print("|---|" + "---|" * len(labs))
    for r in roles:
        print("| %s | %s |" % (r, " | ".join(fmt(s[r][0], s[r][1]) if r in s else "—"
                                            for _, s in allstats)))


if __name__ == "__main__":
    main(sys.argv[1:])
