#!/usr/bin/env python3
"""Doc-format tables for the tune1-perturbations comparison doc §4 (2026-10-30), from Grid hmc.log
chains through compare_runs_multiwindow.Run (the same parser as every other Grid table):

  --pieces   time per trajectory by piece, seconds: Grid's own per-call timers (accept_run_tables,
             L199), mean over the warm trajectories (the first of each log window left out); then
             the tail's force calls and seconds per call (tail time over its calls).
  --dh       ΔH table for the step-count control: trajectories, RMS ΔH, largest abs ΔH, how many
             abs ΔH above 0.5, full-trajectory time against the first run.

The rows are printed exactly as the doc shows them, so check_doc_tables.py can compare the doc
against this output cell by cell.

Usage:
  piece_time_table.py --pieces --run 'LABEL=log1,log2,...' [--run ...]
  piece_time_table.py --dh --run 'LABEL=...' [--run ...]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_runs_multiwindow import Run, mean  # noqa: E402

GROUPS = [
    ("light rungs (4)", ["PF0", "PF1", "PF2", "PF3"]),
    ("log-dets", ["ldS", "ldL"]),
    ("strange", ["s"]),
    ("**tail**", ["tail"]),
    ("shared pullback (strange, log-dets, light rungs)", ["pb0"]),
    ("gauge", ["gauge"]),
    ("smearing", ["smear"]),
    ("other (link updates, final action)", ["other"]),
]


def comma(x):
    return "{:,.0f}".format(x)


def tail_calls(r):
    """Tail force calls per warm trajectory (all equal in a fixed layout; the mean otherwise)."""
    k = r.key_of["tail"]
    n = [sum(1 for ev in r.trajs[i]["evs"] if ev[1] == k) for i in r.warm_idx()]
    return mean(n)


def pieces(runs):
    print("| piece | " + " | ".join(r.label for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for lab, hs in GROUPS:
        v = [sum(r.sector_time(h) or 0.0 for h in hs) for r in runs]
        cells = [comma(x) for x in v]
        if lab.startswith("**"):
            cells = ["**%s**" % c for c in cells]
        print("| %s | %s |" % (lab, " | ".join(cells)))
    md = [mean([r.wall(i) for i in r.warm_idx()]) for r in runs]
    full = [mean([r.cycle(i) for i in r.warm_idx()]) for r in runs]
    print("| molecular dynamics | %s |" % " | ".join(comma(x) for x in md))
    print("| full trajectory | %s |" % " | ".join(comma(x) for x in full))
    print()
    print("| tail | " + " | ".join(r.label for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    calls = [tail_calls(r) for r in runs]
    print("| force calls per trajectory | %s |" % " | ".join("%.0f" % c for c in calls))
    print("| seconds per call | %s |" % " | ".join(
        "%.1f" % (r.sector_time("tail") / c) for r, c in zip(runs, calls)))


def dh(runs):
    print("| run | trajectories | ΔH spread (RMS) | largest abs ΔH | abs ΔH above 0.5 | time vs baseline |")
    print("|---|---|---|---|---|---|")
    ref = mean([runs[0].cycle(i) for i in runs[0].warm_idx()])
    for j, r in enumerate(runs):
        d = r.dHs()
        rms = math.sqrt(sum(x * x for x in d) / len(d))
        t = mean([r.cycle(i) for i in r.warm_idx()])
        tv = "—" if j == 0 else "%+.0f%%" % (100.0 * (t / ref - 1.0))
        print("| %s | %d | %.3f | %.2f | %d | %s |" % (
            r.label, len(d), rms, max(abs(x) for x in d), sum(1 for x in d if abs(x) > 0.5),
            tv.replace("-", "−")))


def main(argv):
    mode, runs, i = None, [], 0
    while i < len(argv):
        if argv[i] in ("--pieces", "--dh"):
            mode = argv[i]
            i += 1
        elif argv[i] == "--run":
            lab, paths = argv[i + 1].split("=", 1)
            runs.append(Run(lab, paths.split(",")))
            i += 2
        else:
            sys.exit("unknown argument %s" % argv[i])
    if mode == "--pieces":
        pieces(runs)
    elif mode == "--dh":
        dh(runs)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
