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
                      CSV simply ends at its last finished trajectory. A start rerun with a new
                      seed after a stall takes the rerun in --run/--log and the stalled run's log
                      in --stalled: its stall is reported as cleared, and the start's cells are
                      then not same-seed with the reference (2026-11-12)
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

Verdict rule (2026_10_28_handoff_multistart_validation.md; written before the runs, amended
2026-11-12 after round 2): a candidate fails if
  (1) two or more trajectories with |dH| > 1 beyond the reference's on the same start;
  (2) its dH RMS / the reference's is above 1.5 across the whole 68% bootstrap range (bootstrap
      over starts, paired: a drawn start brings both arms' trajectories; 20,000 draws, seed 1);
      until 11-12 the point ratio alone decided;
  (3) more light-solver stalls than the reference that recur when the stalled start is rerun with
      a new seed; a stall not yet rerun leaves the verdict pending; until 11-12 every stall counted.
Otherwise it passes and its paired cost is quoted.

Usage:
  multistart_pairs.py <csv> --ref ARM --arms ARM1,ARM2,... [--log RUNLABEL=hmc.log[,hmc.log] ...]
                      [--stalled RUNLABEL=hmc.log[,hmc.log] ...]
"""
import csv
import math
import os
import random
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
    path, ref, arms, logs, stalled_logs = argv[0], None, None, {}, {}
    i = 1
    while i < len(argv):
        if argv[i] == "--ref":
            ref = argv[i + 1]
        elif argv[i] == "--arms":
            arms = argv[i + 1].split(",")
        elif argv[i] in ("--log", "--stalled"):
            lab, _, p = argv[i + 1].partition("=")
            (logs if argv[i] == "--log" else stalled_logs)[lab] = p
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
    cleared = {}  # (arm, start) -> stalls in the run that the rerun replaced
    for lab, paths in stalled_logs.items():
        arm = next((a for a in by_len if lab.startswith(a + "_")), None)
        if arm is None or lab not in logs:
            sys.exit("--stalled %r needs a matching arm and a --log for its rerun" % lab)
        n = 0
        for p in paths.split(","):
            with open(p, errors="replace") as f:
                n += sum(STALL in line for line in f)
        cleared[(arm, lab[len(arm) + 1:])] = n
    starts = sorted({s for a in arms for s in data[a]}, key=start_key)

    def n_stalls(arm, s=None):
        return sum(v for (a, st_), v in stalls.items() if a == arm and (s is None or st_ == s))

    def n_cleared(arm, s=None):
        return sum(v for (a, st_), v in cleared.items() if a == arm and (s is None or st_ == s))

    def stall_cell(arm, s=None):
        c = n_cleared(arm, s)
        return str(n_stalls(arm, s)) + (" (%d cleared on rerun)" % c if c else "")

    def boot_ratio(arm, n=20000):
        """dH RMS arm / ref, bootstrap over the shared starts, paired: (16th, 84th percentile,
        fraction of draws above 1.5). Fresh seed per arm, so an arm's range is independent of
        which other arms are listed."""
        sh = [s for s in sorted(data[ref]) if s in data[arm]]
        rng, qs = random.Random(1), []
        for _ in range(n):
            pick = [rng.choice(sh) for _ in sh]
            qs.append(rms([float(r["dH"]) for s in pick for r in data[arm][s]]) /
                      rms([float(r["dH"]) for s in pick for r in data[ref][s]]))
        qs.sort()
        return qs[int(0.16 * n)], qs[int(0.84 * n)], sum(q > 1.5 for q in qs) / n

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
            stall_cell(a) if logs else "—",
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
        print("\n### Per start: %s against %s (same start, same seed unless marked rerun)\n" % (a, ref))
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
                s.replace("djm", "djm-").replace("_", " ") + (" (rerun)" if (a, s) in cleared else ""),
                pair(lambda x: str(len(x))),
                pair(lambda x: "%.3f" % rms([float(r["dH"]) for r in x])),
                pair(lambda x: "%.2f" % max(abs(float(r["dH"])) for r in x)),
                "%s / %s" % (n_stalls(ref, s), stall_cell(a, s)) if logs else "—",
                pair(lambda x: "{:,.0f} s ({})".format(st.mean(warm(x, "wall_s")), len(warm(x, "wall_s")))
                     if warm(x, "wall_s") else "—"),
                chg]) + " |")
        print("\nCells are %s / %s; warm wall is the mean over the run's warm trajectories (count in "
              "brackets). A rerun start's candidate run has a new seed, so it is not same-seed with "
              "%s there." % (ref, a, ref))

    # ---- the verdict ----
    print("\n### Verdict (rule written before the runs, amended 2026-11-12)\n")
    hdr = ["candidate", "(1) abs ΔH > 1 beyond %s on the same start (fail at ≥ 2)" % ref,
           "(2) ΔH RMS / %s's, 68%% bootstrap range (fail if all above 1.5)" % ref,
           "(3) stalls, %s vs candidate (fail if more, after a rerun)" % ref, "result"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    ref_rms = rms([float(r["dH"]) for s in data[ref] for r in data[ref][s]])
    for a in arms:
        if a == ref:
            continue
        extra = sum(max(0, sum(abs(float(r["dH"])) > 1 for r in data[a].get(s, []))
                       - sum(abs(float(r["dH"])) > 1 for r in data[ref].get(s, []))) for s in starts)
        q = rms([float(r["dH"]) for s in data[a] for r in data[a][s]]) / ref_rms
        lo, hi, above = boot_ratio(a)
        sr, sa = n_stalls(ref), n_stalls(a)
        # a surplus stall on a start already rerun has recurred: fail; one not yet rerun: pending
        recur = [s for s in starts if n_stalls(a, s) > n_stalls(ref, s) and (a, s) in cleared]
        fresh = [s for s in starts if n_stalls(a, s) > n_stalls(ref, s) and (a, s) not in cleared]
        fails = [k for k, bad in (("1", extra >= 2), ("2", lo > 1.5),
                                  ("3", logs and sa > sr and recur)) if bad]
        if fails:
            result = "**FAILS** (%s)" % ", ".join(fails)
        elif logs and sa > sr and fresh:
            result = "pending (3): rerun %s with a new seed" % ", ".join(
                s.replace("djm", "djm-").replace("_", " ") for s in fresh)
        else:
            result = "passes"
        print("| " + " | ".join([
            a, str(extra), "%.2f (%.2f–%.2f; %.0f%% of draws above 1.5)" % (q, lo, hi, 100 * above),
            "%d vs %s" % (sr, stall_cell(a)) if logs else "— (no --log)", result]) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
