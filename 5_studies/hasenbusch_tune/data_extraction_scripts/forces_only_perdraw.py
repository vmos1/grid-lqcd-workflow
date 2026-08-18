#!/usr/bin/env python3
"""forces_only_perdraw.py -- paired per-draw force comparison across tolerance
(or any other) FORCES_ONLY arms.

`forces_only_tables.py ab` compares RUN-LEVEL avg/max, which averages away the
thing a tolerance scan is actually guarding against: one bad rare draw.  Since
HMC_SEED_OFFSET is unset (SOP), the gaussian eta is identical across arms, so
arm X's level-L sample-s force is directly comparable to the reference arm's
level-L sample-s force.  That gives 6 levels x N samples paired comparisons
instead of a single N-draw max.

This script reports the WORST deviation over all pairs, per level and overall,
against the campaign thresholds:

    |davg| < 0.1%   AND   |dmax| < 1%

NOTE those thresholds are a pre-registered sanity bound, NOT validated against
dH or acceptance.  A real acceptance run remains the only way to confirm a
FORCES_ONLY verdict.

It also prints the per-rung timing split (refresh vs deriv separately), because
FORCES_ONLY charges one refresh + one deriv per level while a real trajectory
does one refresh per level per TRAJECTORY and MDSTEPS x multiplicity derivs --
so the deriv column, not the total, is what predicts production benefit.

Usage:
    forces_only_perdraw.py <run_dir> [--ref R] [--samples-from 1]
    forces_only_perdraw.py <ref.log> <other.log> [<other.log> ...]
"""

import argparse
import os
import re
import sys
from glob import glob

RE = re.compile(
    r"FORCES_ONLY level (\S+) sample (\d+) avg=([\d.eE+-]+) max=([\d.eE+-]+) "
    r"refresh=([\d.eE+-]+) time=([\d.eE+-]+)"
)
RE_ENV = re.compile(r"TUNE_CG_TOL_ACTION=(\S+)\s+TUNE_CG_TOL_DERIV=(\S+)")

LEVEL_ORDER = ["LightLogDet", "PF0", "PF1", "PF2", "PF3", "LightSchurPF"]
THR_AVG, THR_MAX = 1e-3, 1e-2

AVG, MAX, REFRESH, DERIV = 0, 1, 2, 3


def load(path):
    """-> (records keyed by (level, sample), tolerance label)"""
    rec, tol = {}, "?"
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_ENV.search(line)
            if m and tol == "?":
                tol = f"{m.group(1)}/{m.group(2)}"
            m = RE.search(line)
            if m:
                rec[(m.group(1), int(m.group(2)))] = tuple(
                    float(m.group(i)) for i in (3, 4, 5, 6)
                )
    return rec, tol


def collect(args):
    """-> ordered [(name, path)], reference first."""
    if len(args.paths) == 1 and os.path.isdir(args.paths[0]):
        root = args.paths[0]
        found = []
        for d in sorted(os.listdir(root)):
            hits = glob(os.path.join(root, d, "*.log"))
            if hits:
                found.append((d, hits[0]))
        if not found:
            sys.exit(f"no <arm>/*.log under {root}")
        ref = args.ref or ("R" if any(n == "R" for n, _ in found) else found[0][0])
        if not any(n == ref for n, _ in found):
            sys.exit(f"reference arm {ref!r} not found in {root}")
        return [x for x in found if x[0] == ref] + [x for x in found if x[0] != ref]
    return [(os.path.basename(p).replace(".log", ""), p) for p in args.paths]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="a run dir with <arm>/*.log, or logs (ref first)")
    p.add_argument("--ref", default=None, help="reference arm name (default R)")
    p.add_argument("--samples-from", type=int, default=1,
                   help="drop early samples; sample 0 carries the MG setup premium")
    a = p.parse_args()

    arms = collect(a)
    data = {n: load(pth) for n, pth in arms}
    ref_name = arms[0][0]
    ref, ref_tol = data[ref_name]

    samples = sorted({s for (_, s) in ref})
    # Sample 0 is dropped from TIMING only -- it carries the one-time MG setup
    # premium.  Its FORCES are ordinary physics, so the accuracy comparison
    # keeps every draw; excluding it would discard exactly the kind of rare bad
    # draw this per-draw readout exists to catch.
    keep = [s for s in samples if s >= a.samples_from]
    levels = [l for l in LEVEL_ORDER if any(l == lv for lv, _ in ref)]
    npair = len(levels) * len(samples)

    print(f"reference: {ref_name}  ({ref_tol})")
    print(f"forces: all samples {samples}  ->  {len(levels)} levels x {len(samples)} draws "
          f"= {npair} paired comparisons per arm")
    print(f"timing: samples {keep} (sample 0 dropped, MG setup premium)")
    print(f"thresholds: |davg| < {THR_AVG:g}  |dmax| < {THR_MAX:g}   "
          f"(sanity bound, NOT dH-validated)")

    def mean(rec, lv, i):
        v = [rec[(lv, s)][i] for s in keep if (lv, s) in rec]
        return sum(v) / len(v) if v else float("nan")

    # ---- per-draw force deviation -----------------------------------------
    for name, _ in arms[1:]:
        rec, tol = data[name]
        print(f"\n{'='*78}\n{ref_name} vs {name}  ({tol})\n{'='*78}")
        print(f"{'level':<14}{'worst |davg|':>13}{'@s':>4}{'worst |dmax|':>13}{'@s':>4}"
              f"   avg     max")
        wa_all = wm_all = 0.0
        for lv in levels:
            wa = wm = 0.0
            sa = sm = -1
            for s in samples:
                if (lv, s) not in rec or (lv, s) not in ref:
                    continue
                for i in (AVG, MAX):
                    r0 = ref[(lv, s)][i]
                    if not r0:
                        continue
                    d = abs(rec[(lv, s)][i] - r0) / r0
                    if i == AVG and d > wa:
                        wa, sa = d, s
                    elif i == MAX and d > wm:
                        wm, sm = d, s
            wa_all, wm_all = max(wa_all, wa), max(wm_all, wm)
            print(f"{lv:<14}{wa:>13.2e}{sa:>4}{wm:>13.2e}{sm:>4}"
                  f"   {'PASS' if wa < THR_AVG else 'FAIL':<6}"
                  f"{'PASS' if wm < THR_MAX else 'FAIL'}")
        ok = wa_all < THR_AVG and wm_all < THR_MAX
        print(f"{'WORST-OF-'+str(npair):<14}{wa_all:>13.2e}{'':>4}{wm_all:>13.2e}")
        marg = ""
        if wa_all and wm_all:
            marg = (f"   (avg {THR_AVG/wa_all:.1f}x, max {THR_MAX/wm_all:.1f}x "
                    f"{'margin' if ok else 'OVER'})")
        print(f"  -> {'PASS' if ok else 'FAIL'}{marg}")

    # ---- timing ------------------------------------------------------------
    names = [n for n, _ in arms]
    print(f"\n{'='*78}\nTIMING per rung (s/sample, mean over samples {keep})\n{'='*78}")
    print(f"{'rung':<14}{'part':<9}" + "".join(f"{n:>9}" for n in names))
    tot = {n: 0.0 for n in names}
    sub = {n: {REFRESH: 0.0, DERIV: 0.0} for n in names}
    for lv in levels:
        for lbl, i in (("refresh", REFRESH), ("deriv", DERIV)):
            row = {}
            for n in names:
                v = mean(data[n][0], lv, i)
                row[n] = v
                tot[n] += v
                sub[n][i] += v
            print(f"{lv:<14}{lbl:<9}" + "".join(f"{row[n]:>9.2f}" for n in names))
    print("-" * (23 + 9 * len(names)))
    for lbl, i in (("refresh Σ", REFRESH), ("deriv Σ", DERIV)):
        print(f"{lbl:<23}" + "".join(f"{sub[n][i]:>9.2f}" for n in names))
    print(f"{'TOTAL':<23}" + "".join(f"{tot[n]:>9.2f}" for n in names))
    print(f"\n{'speedup vs '+ref_name:<23}" +
          "".join(f"{tot[ref_name]/tot[n]:>8.3f}x" for n in names))
    print(f"{'  deriv-only speedup':<23}" +
          "".join(f"{sub[ref_name][DERIV]/sub[n][DERIV]:>8.3f}x" for n in names))
    print("\nNOTE: deriv-only is the production-relevant number.  FORCES_ONLY charges")
    print("one refresh + one deriv per level; a real trajectory does one refresh per")
    print("level per TRAJECTORY but MDSTEPS x multiplicity derivs.")


if __name__ == "__main__":
    sys.exit(main())
