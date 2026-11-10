#!/usr/bin/env python3
"""Per-start pairing and verdict for campaign multistart-validation (2026-11-02), from the
per-trajectory CSV of `compare_runs_multiwindow.py --csv` with one --run per arm and start,
labelled <arm>_<start> (e.g. baseline_djm3_1000, B2_t24_djm3_1000).

Every candidate is read against the reference arm start by start (same start, same seed):
  dH                  RMS, largest |dH|, count of |dH| > 1, <e^-dH> (naive error), expected
                      acceptance = mean of min(1, e^-dH). The accept COUNT is printed, but every
                      start shares one seed and so one Metropolis random sequence: accepts are
                      correlated across starts (one draw rejects every dH above its threshold).
                      Read the expected acceptance.
  stalls              lines "after a fresh hierarchy: ABORT" in each run's logs (--log, comma-
                      separated when a start was finished in a second segment); a stalled run's
                      CSV simply ends at its last finished trajectory
  time                warm trajectories only (the first of each run is a cold start); per start
                      the candidate / reference ratio of the warm means, then the mean and
                      standard error of those ratios over the starts; wall (MD only, primary)
                      and cycle (full trajectory)
  per accepted        the paired time ratio times P_ref / P_arm, P the arm's expected acceptance
                      over all its trajectories (not the accept count, see dH); the error adds the
                      ratio's and both acceptances' relative errors in quadrature (2026-11-09)
  plaquette, smeared  mean of the per-start means, error = sd of those means / sqrt(starts)
  plaquette drift     per start, the run's mean plaquette minus its "Initial plaquette" (first
                      --log); mean and standard error over the starts. Arms share seeds, so
                      their drifts are one observation, not independent ones (2026-11-09)

Verdict rule (2026_10_28_handoff_multistart_validation.md, written before the runs): a candidate
fails if (1) two or more trajectories with |dH| > 1 beyond the reference's on the same start,
(2) its dH RMS exceeds the reference's by more than 50%, or (3) more light-solver stalls than the
reference. Otherwise it passes and its paired cost is quoted.

Usage:
  multistart_pairs.py <csv> --ref ARM --arms ARM1,ARM2,... [--log RUNLABEL=hmc.log[,hmc.log] ...]
"""
import csv
import math
import os
import statistics as st
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidate_tables import one_digit

STALL = "after a fresh hierarchy: ABORT"
INITIAL = "Initial plaquette = "


def rms(x):
    return math.sqrt(st.mean([v * v for v in x]))


def sem(x):
    return st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def warm(rows, key):
    return [float(r[key]) for r in rows if r[key] and r["cold"] == "0" and r["resume"] == "0"]


def start_key(s):
    """djm3_1000 -> ('djm3', 1000), so starts sort by stream, then trajectory."""
    stream, _, traj = s.rpartition("_")
    return (stream, int(traj)) if traj.isdigit() else (s, 0)


def ratio_cell(rs):
    if not rs:
        return "—"
    return ("%+.1f%% ± %.1f%%" % (100 * (st.mean(rs) - 1), 100 * sem(rs))).replace("-", "−")


def main(argv):
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    path, ref, arms, logs = argv[0], None, None, {}
    i = 1
    while i < len(argv):
        if argv[i] == "--ref":
            ref = argv[i + 1]
        elif argv[i] == "--arms":
            arms = argv[i + 1].split(",")
        elif argv[i] == "--log":
            lab, _, p = argv[i + 1].partition("=")
            logs[lab] = p
        else:
            sys.exit(__doc__)
        i += 2
    if not ref or not arms or ref not in arms:
        sys.exit("--arms must list every arm and --ref must be one of them")

    # data[arm][start] = rows; longest arm name first so B2_t24 is not read as B2 + t24_...
    data = OrderedDict((a, OrderedDict()) for a in arms)
    by_len = sorted(arms, key=len, reverse=True)
    for r in csv.DictReader(open(path)):
        arm = next((a for a in by_len if r["run"].startswith(a + "_")), None)
        if arm is None:
            sys.exit("run %r matches no arm in --arms" % r["run"])
        data[arm].setdefault(r["run"][len(arm) + 1:], []).append(r)
    stalls, initial = {}, {}
    for lab, paths in logs.items():
        arm = next((a for a in by_len if lab.startswith(a + "_")), None)
        if arm is None:
            sys.exit("--log %r matches no arm in --arms" % lab)
        n = 0
        for k, p in enumerate(paths.split(",")):
            with open(p, errors="replace") as f:
                for line in f:
                    n += STALL in line
                    if k == 0 and INITIAL in line and (arm, lab[len(arm) + 1:]) not in initial:
                        initial[(arm, lab[len(arm) + 1:])] = float(line.split(INITIAL)[1].split()[0])
        stalls[(arm, lab[len(arm) + 1:])] = n
    starts = sorted({s for a in arms for s in data[a]}, key=start_key)

    def n_stalls(arm, s=None):
        return sum(v for (a, st_), v in stalls.items() if a == arm and (s is None or st_ == s))

    def paired(arm, key):
        out = []
        for s in starts:
            c, b = warm(data[arm].get(s, []), key), warm(data[ref].get(s, []), key)
            if c and b:
                out.append(st.mean(c) / st.mean(b))
        return out

    def expected_acceptance(arm):
        """Mean and standard error of min(1, e^-dH) over every trajectory of the arm."""
        p = [min(1.0, math.exp(-float(r["dH"]))) for s in data[arm] for r in data[arm][s]]
        return st.mean(p), sem(p)

    def per_accepted_cell(arm, rs):
        if not rs:
            return "—"
        (pr, epr), (pa, epa) = expected_acceptance(ref), expected_acceptance(arm)
        m = st.mean(rs) * pr / pa
        err = m * math.sqrt((sem(rs) / st.mean(rs)) ** 2 + (epr / pr) ** 2 + (epa / pa) ** 2)
        return ("%+.1f%% ± %.1f%%" % (100 * (m - 1), 100 * err)).replace("-", "−")

    def drift_cell(arm):
        d = [st.mean([float(r["plaquette"]) for r in data[arm][s]]) - initial[(arm, s)]
             for s in data[arm] if (arm, s) in initial]
        if len(d) < 2:
            return "—"
        return ("%+.1e ± %.1e" % (st.mean(d), sem(d))).replace("-", "−").replace("e−0", "e-0")

    # ---- physics per arm ----
    print("### Per arm: ΔH, stalls, observables\n")
    hdr = ["arm", "starts", "trajectories", "accepted", "expected acceptance", "ΔH spread (RMS)",
           "largest abs ΔH", "abs ΔH > 1", "⟨e^−ΔH⟩", "light-solver stalls", "plaquette", "smeared plaquette",
           "plaquette drift from start"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for a in arms:
        rows = [r for s in data[a] for r in data[a][s]]
        dh = [float(r["dH"]) for r in rows]
        e = [math.exp(-x) for x in dh]
        pm = [st.mean([float(r["plaquette"]) for r in data[a][s]]) for s in data[a]]
        sm = [st.mean([float(r["plaquette_smeared"]) for r in data[a][s]]) for s in data[a]
              if all(r["plaquette_smeared"] for r in data[a][s])]
        print("| " + " | ".join([
            a, str(len(data[a])), str(len(rows)),
            "%d/%d" % (sum(int(r["accepted"]) for r in rows), len(rows)),
            "%.1f%%" % (100 * st.mean([min(1.0, x) for x in e])),
            "%.3f" % rms(dh), "%.2f" % max(abs(x) for x in dh), str(sum(abs(x) > 1 for x in dh)),
            "%.3f ± %.3f" % (st.mean(e), sem(e)),
            str(n_stalls(a)) if logs else "—",
            one_digit(st.mean(pm), sem(pm)),
            one_digit(st.mean(sm), sem(sm)) if len(sm) == len(data[a]) else "—",
            drift_cell(a)]) + " |")

    # ---- cost per arm, paired per start ----
    print("\n### Per arm: time per trajectory (warm), paired with %s start by start\n" % ref)
    hdr = ["arm", "warm trajectories", "wall (MD only)", "cycle (full)", "wall vs %s, paired" % ref,
           "cycle vs %s, paired" % ref, "wall per accepted vs %s" % ref, "cycle per accepted vs %s" % ref,
           "starts paired"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for a in arms:
        rows = [r for s in data[a] for r in data[a][s]]
        w, c = warm(rows, "wall_s"), warm(rows, "cycle_s")
        pw, pc = paired(a, "wall_s"), paired(a, "cycle_s")
        print("| " + " | ".join([
            a, str(len(w)), "{:,.0f} s".format(st.mean(w)), "{:,.0f} s".format(st.mean(c)),
            "—" if a == ref else ratio_cell(pw), "—" if a == ref else ratio_cell(pc),
            "—" if a == ref else per_accepted_cell(a, pw), "—" if a == ref else per_accepted_cell(a, pc),
            "—" if a == ref else str(len(pw))]) + " |")
    print("\nPer accepted: the paired ratio times the reference's expected acceptance over the arm's "
          "(per-arm table); error from the ratio and both acceptances.")

    # ---- per start, each candidate against the reference ----
    for a in arms:
        if a == ref:
            continue
        print("\n### Per start: %s against %s (same start, same seed)\n" % (a, ref))
        hdr = ["start", "trajectories", "ΔH RMS", "largest abs ΔH", "stalls", "warm wall", "wall change"]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for s in starts:
            b, c = data[ref].get(s, []), data[a].get(s, [])

            def pair(f):
                return "%s / %s" % (f(b) if b else "—", f(c) if c else "—")

            wb, wc = warm(b, "wall_s"), warm(c, "wall_s")
            chg = (("%+.1f%%" % (100 * (st.mean(wc) / st.mean(wb) - 1))).replace("-", "−")
                   if wb and wc else "—")
            print("| " + " | ".join([
                s.replace("djm", "djm-").replace("_", " "),
                pair(lambda x: str(len(x))),
                pair(lambda x: "%.3f" % rms([float(r["dH"]) for r in x])),
                pair(lambda x: "%.2f" % max(abs(float(r["dH"])) for r in x)),
                "%s / %s" % (n_stalls(ref, s), n_stalls(a, s)) if logs else "—",
                pair(lambda x: "{:,.0f} s ({})".format(st.mean(warm(x, "wall_s")), len(warm(x, "wall_s")))
                     if warm(x, "wall_s") else "—"),
                chg]) + " |")
        print("\nCells are %s / %s; warm wall is the mean over the run's warm trajectories (count in "
              "brackets)." % (ref, a))

    # ---- the pre-registered verdict ----
    print("\n### Verdict (rule written before the runs)\n")
    hdr = ["candidate", "(1) abs ΔH > 1 beyond %s on the same start (fail at ≥ 2)" % ref,
           "(2) ΔH RMS / %s's (fail above 1.5)" % ref, "(3) stalls, %s vs candidate (fail if more)" % ref,
           "result"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    ref_rms = rms([float(r["dH"]) for s in data[ref] for r in data[ref][s]])
    for a in arms:
        if a == ref:
            continue
        extra = sum(max(0, sum(abs(float(r["dH"])) > 1 for r in data[a].get(s, []))
                       - sum(abs(float(r["dH"])) > 1 for r in data[ref].get(s, []))) for s in starts)
        q = rms([float(r["dH"]) for s in data[a] for r in data[a][s]]) / ref_rms
        sr, sa = n_stalls(ref), n_stalls(a)
        fails = [k for k, bad in (("1", extra >= 2), ("2", q > 1.5), ("3", logs and sa > sr)) if bad]
        print("| " + " | ".join([
            a, str(extra), "%.2f" % q, "%d vs %d" % (sr, sa) if logs else "— (no --log)",
            "**FAILS** (%s)" % ", ".join(fails) if fails else "passes"]) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
