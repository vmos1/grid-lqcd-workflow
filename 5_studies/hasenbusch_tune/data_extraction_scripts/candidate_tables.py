#!/usr/bin/env python3
"""Candidate-vs-baseline summary table (markdown) from the per-trajectory CSV of
`compare_runs_multiwindow.py --csv` (Grid) or `chroma_runs_csv.py` (Chroma). Written for the
tune1-perturbations comparison doc (2026-10-27); the conventions are that doc's:

  trajectories / accepted   chain length and Metropolis accepts (rejects kept as repeats)
  dH spread                 RMS of dH;  largest |dH|
  expected acceptance       mean of min(1, e^-dH): less noisy than the accept count at N = 20
  <e^-dH>                   must be 1 for exact HMC; error = naive standard error
  time / trajectory         FULL trajectory (cycle_s), mean over warm trajectories (the first of
                            each log window, cold or resumed, excluded); % against the reference
  plaquette, smeared        mean and standard error, the error being the largest of the naive,
                            pair-binned and four-binned estimates; quoted with one digit of error

Usage:
  candidate_tables.py <csv> --ref LABEL [--combine 'NEW=LABEL1+LABEL2'] [--order L1,L2,...]
"""
import csv
import math
import statistics as st
import sys
from collections import OrderedDict


def blocked(x, b):
    nb = len(x) // b
    if nb < 2:
        return 0.0
    m = [st.mean(x[i * b:(i + 1) * b]) for i in range(nb)]
    return st.stdev(m) / math.sqrt(nb)


def err(x):
    return max(st.stdev(x) / math.sqrt(len(x)), blocked(x, 2), blocked(x, 4))


def one_digit(m, e):
    """0.5434377, 1.2e-5 -> '0.54344(1)'; the error rounded to one significant digit."""
    if e <= 0:
        return "%.6f" % m
    k = -int(math.floor(math.log10(e)))      # decimal place of the error's first digit
    d = round(e * 10 ** k)
    if d == 10:                               # 0.96e-5 rounds up to 1e-4
        k, d = k - 1, 1
    return "%.*f(%d)" % (k, round(m, k), d)


def main(argv):
    path, ref, combos, order = argv[0], None, [], None
    i = 1
    while i < len(argv):
        if argv[i] == "--ref":
            ref = argv[i + 1]
        elif argv[i] == "--combine":
            new, _, parts = argv[i + 1].partition("=")
            combos.append((new, parts.split("+")))
        elif argv[i] == "--order":
            order = argv[i + 1].split(",")
        else:
            sys.exit(__doc__)
        i += 2
    runs = OrderedDict()
    for r in csv.DictReader(open(path)):
        runs.setdefault(r["run"], []).append(r)
    for new, parts in combos:
        runs[new] = [r for p in parts for r in runs[p]]
    labels = order or list(runs)
    if ref is None or ref not in runs:
        sys.exit("--ref must name a run in %s: %s" % (path, ", ".join(runs)))

    def time(rows):
        warm = [float(r["cycle_s"]) for r in rows if r["cycle_s"] and r["cold"] == "0" and r["resume"] == "0"]
        return st.mean(warm)

    t_ref = time(runs[ref])
    smeared = any(r["plaquette_smeared"] for lab in labels for r in runs[lab])  # '—' where a run lacks it
    hdr = ["run", "trajectories", "accepted", "ΔH spread (RMS)", "largest abs ΔH", "expected acceptance",
           "⟨e^−ΔH⟩", "time per trajectory", "plaquette"] + (["smeared plaquette"] if smeared else [])
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for lab in labels:
        rows = runs[lab]
        dh = [float(r["dH"]) for r in rows]
        e = [math.exp(-x) for x in dh]
        t = time(rows)
        pct = "" if lab == ref else (" (%+.0f%%)" % (100 * (t / t_ref - 1))).replace("-", "−")
        plaq = [float(r["plaquette"]) for r in rows]
        cells = [lab, str(len(rows)), "%d/%d" % (sum(int(r["accepted"]) for r in rows), len(rows)),
                 "%.3f" % math.sqrt(st.mean([x * x for x in dh])), "%.2f" % max(abs(x) for x in dh),
                 "%.0f%%" % (100 * st.mean([min(1.0, x) for x in e])),
                 "%.2f ± %.2f" % (st.mean(e), st.stdev(e) / math.sqrt(len(e))),
                 "{:,.0f} s{}".format(t, pct), one_digit(st.mean(plaq), err(plaq))]
        if smeared:
            sm = [float(r["plaquette_smeared"]) for r in rows if r["plaquette_smeared"]]
            cells.append(one_digit(st.mean(sm), err(sm)) if len(sm) == len(rows) else "—")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
