#!/usr/bin/env python3
"""solve_cost_split.py -- split every FORCES_ONLY heatbath and deriv time into
bare solve / QUDA per-call overhead / measured solve-free floor.

Companion to forces_only_time_budget.py.  That script stops at
"deriv -> solver / assembly / smearing"; this one goes one level deeper and
separates, for BOTH refresh() and deriv(), on BOTH the CG and MG paths:

    quoted  =  marginal solve  +  QUDA per-call  +  solve-free floor

  * marginal solve   = slope x N_iter, slope from the CG/MG cost model
  * solve-free floor = MEASURED per level, not assumed:
                         deriv     -> force assembly + smearing chain rule
                         refresh() -> the bare-det floor (LightSchurPF)
  * QUDA per-call    = whatever is left, i.e. setup + field transfer

Everything is read out of an ordinary FORCES_ONLY log -- no new instrumentation.
Solves are attributed to the heatbath or the deriv by the tolerance QUDA echoes
in its convergence line ("requested = ..."):  the action tolerance belongs to
refresh(), the deriv tolerance to deriv().  Those are auto-detected as the two
distinct non-MG-internal values, and can be overridden with --tol-action /
--tol-deriv when a run sets them equal.

Usage:
    solve_cost_split.py <log> [<log> ...] [--samples-from 1]
                        [--cg-slope 3.84] [--mg-slope 118]
                        [--tol-action 1e-12] [--tol-deriv 1e-11]
"""

import argparse
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------- line regexes

RE_LEVEL = re.compile(
    r"FORCES_ONLY level (\S+) sample (\d+) .*?refresh=([\d.eE+-]+) time=([\d.eE+-]+)"
)
RE_CHAIN = re.compile(r"chain rule took ([\d.eE+-]+) ms")
# Top-level solver convergence.  MG-internal lines are prefixed "MG level N" and
# are deliberately NOT matched: their iterations are already inside the outer
# GCR count, and counting them would double-count the V-cycle.
RE_CONV = re.compile(
    r"^\s*(GCR|CG|CA-GCR): Convergence at (\d+) iterations.*?requested = ([\d.eE+-]+)\)"
)
RE_ASSY = re.compile(
    r"\[(Quda\w+Force)/(.*?)\s*(?:\[QUDA force assembly\])?\]\s+(\d+) (?:op-force )?calls "
    r"\(ms/call\): (.*)$"
)
RE_MS = re.compile(r"=([\d.eE+-]+)")
RE_LADDER_DERIV = re.compile(r"\[Ladder\] rung (\d+) DerivativeSolver\S*.*?mass=(-[\d.]+)")

# Level order as emitted by the driver.
LEVEL_ORDER = ["LightLogDet", "PF0", "PF1", "PF2", "PF3", "LightSchurPF"]


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------- parser


def parse(path):
    """Return (per-level records, assembly table, ladder mass list)."""
    conv_buf = []      # solves seen since the last FORCES_ONLY line
    chain_buf = []     # chain-rule timings since the last FORCES_ONLY line
    recs = defaultdict(list)
    assy_raw = defaultdict(list)
    ladder = {}

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = RE_LADDER_DERIV.search(line)
            if m:
                ladder[int(m.group(1))] = m.group(2)
                continue

            m = RE_ASSY.search(line)
            if m:
                tag = f"{m.group(1)}/{m.group(2)}".strip()
                calls = int(m.group(3))
                ms = sum(float(v) for v in RE_MS.findall(m.group(4)))
                assy_raw[tag].append((calls, ms))
                continue

            m = RE_CHAIN.search(line)
            if m:
                chain_buf.append(float(m.group(1)) / 1000.0)
                continue

            # QUDA's convergence lines carry no Grid timestamp, so strip any
            # prefix before testing.
            m = RE_CONV.search(line.split(" : ")[-1] if " : " in line else line)
            if m and not line.lstrip().startswith("MG level"):
                conv_buf.append((m.group(1), int(m.group(2)), float(m.group(3))))
                continue

            m = RE_LEVEL.search(line)
            if m:
                level, sample = m.group(1), int(m.group(2))
                recs[level].append(
                    dict(
                        sample=sample,
                        refresh=float(m.group(3)),
                        deriv=float(m.group(4)),
                        conv=list(conv_buf),
                        chain=list(chain_buf),
                    )
                )
                conv_buf, chain_buf = [], []

    return recs, assy_raw, ladder


def detect_tols(recs):
    """The two distinct top-level tolerances = (action, deriv)."""
    tols = sorted({c[2] for rs in recs.values() for r in rs for c in r["conv"]})
    if len(tols) < 2:
        return (tols[0], tols[0]) if tols else (None, None)
    # Action tol is the tighter one in every run we have; keep both anyway.
    return tols[0], tols[1]


def assembly_per_sample(assy_raw, n_samples):
    """tag -> seconds of force assembly per sample (averaged over ranks)."""
    out = {}
    for tag, entries in assy_raw.items():
        calls = entries[0][0]
        ms = sum(e[1] for e in entries) / len(entries)
        # `calls` counts op-force calls over the whole run; a ratio level issues
        # two per sample (numerator + denominator), a plain det one.
        out[tag] = (calls / n_samples) * ms / 1000.0
    return out


def flag_assembly_outliers(by_level):
    """The assembly counters are CUMULATIVE over the whole run and are printed
    once at the end, so they cannot be restricted to samples >= 1.  Any one-time
    cost that lands inside a force call (e.g. the 33 s MG preconditioner build,
    which in an all-CG-heatbath arm is first triggered by rung 0's deriv) is
    smeared over every call and inflates that level.  Detect it rather than
    quietly reporting a contaminated number."""
    ratio = {k: v for k, v in by_level.items() if k.startswith("PF")}
    if len(ratio) < 3:
        return set()
    vals = sorted(ratio.values())
    med = vals[len(vals) // 2]
    return {k for k, v in ratio.items() if med > 0 and v > 3 * med}


def map_assembly(assy, ladder):
    """Attach the per-tag assembly cost to the level that produced it."""
    by_level = {}
    for tag, sec in assy.items():
        if "LogDet" in tag:
            by_level["LightLogDet"] = sec
        elif "RatioAction" in tag:
            m = re.search(r"det\((-[\d.]+)\) / det\((-[\d.]+)\)", tag)
            if not m:
                continue
            num = m.group(1)
            # rung k's deriv runs at the numerator mass
            for rung, mass in ladder.items():
                if float(mass) == float(num):
                    by_level[f"PF{rung}"] = sec
        else:
            by_level["LightSchurPF"] = sec
    return by_level


# ------------------------------------------------------------------- reporting


def report(path, args):
    recs, assy_raw, ladder = parse(path)
    if not recs:
        print(f"!! no FORCES_ONLY levels found in {path}")
        return

    samples = sorted({r["sample"] for rs in recs.values() for r in rs})
    keep = [s for s in samples if s >= args.samples_from]
    n_all = len(samples)

    tol_a, tol_d = args.tol_action, args.tol_deriv
    if tol_a is None or tol_d is None:
        tol_a, tol_d = detect_tols(recs)

    assy = map_assembly(assembly_per_sample(assy_raw, n_all), ladder)
    if args.assembly_from:
        ref_recs, ref_raw, ref_ladder = parse(args.assembly_from)
        ref_n = len({r["sample"] for rs in ref_recs.values() for r in rs})
        assy = map_assembly(assembly_per_sample(ref_raw, ref_n), ref_ladder)
    bad_assy = flag_assembly_outliers(assy)

    print(f"\n{'='*104}")
    print(f"  {path}")
    print(
        f"  samples {keep} of {samples} (sample 0 dropped: carries the MG setup premium)"
    )
    print(f"  action tol = {tol_a:g} -> refresh() ; deriv tol = {tol_d:g} -> deriv()")
    print(f"{'='*104}")

    # ---- floors -----------------------------------------------------------
    # refresh() floor: LightSchurPF is a bare det, its heatbath is one operator
    # application with no solve, so its refresh IS the floor.
    hb_floor = None
    if "LightSchurPF" in recs:
        vals = [r["refresh"] for r in recs["LightSchurPF"] if r["sample"] >= args.samples_from]
        hb_floor = sum(vals) / len(vals)

    rows_hb, rows_dv = [], []

    for level in LEVEL_ORDER:
        if level not in recs:
            continue
        rs = [r for r in recs[level] if r["sample"] >= args.samples_from]
        if not rs:
            continue

        n = len(rs)
        hb_t = sum(r["refresh"] for r in rs) / n
        dv_t = sum(r["deriv"] for r in rs) / n
        chain = sum(sum(r["chain"]) for r in rs) / n
        asm = assy.get(level, 0.0)

        hb_conv = [c for r in rs for c in r["conv"] if c[2] == tol_a]
        dv_conv = [c for r in rs for c in r["conv"] if c[2] == tol_d]
        hb_iters = sum(c[1] for c in hb_conv) / n
        dv_iters = sum(c[1] for c in dv_conv) / n
        hb_calls = len(hb_conv) / n
        dv_calls = len(dv_conv) / n
        hb_kind = hb_conv[0][0] if hb_conv else None
        dv_kind = dv_conv[0][0] if dv_conv else None

        def slope(kind):
            return (args.mg_slope / 1000.0) if kind and "GCR" in kind else (args.cg_slope / 1000.0)

        # --- heatbath: quoted = solve + per-call + bare-det floor
        if hb_calls:
            hb_solve = slope(hb_kind) * hb_iters
            hb_rest = hb_t - hb_solve - (hb_floor or 0.0)
            rows_hb.append(
                (level, f"{hb_kind} x{hb_calls:g}", hb_iters, hb_t, hb_solve,
                 hb_floor, hb_rest, 100 * hb_solve / hb_t)
            )
        else:
            rows_hb.append((level, "--", 0.0, hb_t, 0.0, hb_t, 0.0, 0.0))

        # --- deriv: floor is MEASURED (assembly + chain rule), not assumed
        dv_solve = slope(dv_kind) * dv_iters if dv_calls else 0.0
        dv_rest = dv_t - dv_solve - asm - chain
        rows_dv.append(
            (level, f"{dv_kind} x{dv_calls:g}" if dv_kind else "--", dv_iters, dv_t,
             dv_solve, asm, chain, dv_rest, 100 * dv_solve / dv_t if dv_t else 0.0)
        )

    hdr = f"{'level':<14}{'solver':<11}{'N_iter':>9}{'quoted':>9}{'solve':>9}{'floor':>9}{'QUDA/call':>11}{'solve%':>8}"
    print("\n-- HEATBATH  refresh()  [floor = bare-det refresh, measured on LightSchurPF]")
    print(hdr)
    print("-" * 80)
    for r in rows_hb:
        print(
            f"{r[0]:<14}{r[1]:<11}{r[2]:>9.0f}{r[3]:>9.2f}{fmt(r[4]):>9}"
            f"{fmt(r[5]):>9}{fmt(r[6]):>11}{r[7]:>7.0f}%"
        )
    print(
        f"{'SUM':<14}{'':<11}{'':>9}{sum(r[3] for r in rows_hb):>9.2f}"
        f"{sum(r[4] or 0 for r in rows_hb):>9.2f}{'':>9}{'':>11}"
    )

    hdr2 = (f"{'level':<14}{'solver':<11}{'N_iter':>9}{'quoted':>9}{'solve':>9}"
            f"{'assembly':>10}{'chainrule':>11}{'QUDA/call':>11}{'solve%':>8}")
    print("\n-- DERIV  deriv()  [floor = MEASURED assembly + MEASURED smearing chain rule]")
    print(hdr2)
    print("-" * 91)
    for r in rows_dv:
        mark = " !" if r[0] in bad_assy else ""
        print(
            f"{r[0]:<14}{r[1]:<11}{r[2]:>9.0f}{r[3]:>9.2f}{fmt(r[4]):>9}"
            f"{fmt(r[5])+mark:>10}{fmt(r[6]):>11}{fmt(r[7]):>11}{r[8]:>7.0f}%"
        )
    print(
        f"{'SUM':<14}{'':<11}{'':>9}{sum(r[3] for r in rows_dv):>9.2f}"
        f"{sum(r[4] for r in rows_dv):>9.2f}{sum(r[5] for r in rows_dv):>10.2f}"
        f"{sum(r[6] for r in rows_dv):>11.2f}{sum(r[7] for r in rows_dv):>11.2f}"
    )

    if bad_assy:
        print(
            f"\n   ! {sorted(bad_assy)}: assembly counter is cumulative over the run and\n"
            f"     carries a one-time cost (MG build) from sample 0.  Steady-state value is\n"
            f"     the other rungs'; re-run with --assembly-from <clean arm log>.  Its\n"
            f"     QUDA/call column absorbs the error and is correspondingly understated."
        )

    grand = sum(r[3] for r in rows_hb) + sum(r[3] for r in rows_dv)
    solve = sum(r[4] or 0 for r in rows_hb) + sum(r[4] for r in rows_dv)
    asmt = sum(r[5] for r in rows_dv)
    chnt = sum(r[6] for r in rows_dv)
    print(f"\n-- per-sample roll-up ({grand:.1f} s/sample)")
    print(f"   marginal solve            {solve:>7.1f} s   {100*solve/grand:>4.0f}%")
    print(f"   force assembly            {asmt:>7.1f} s   {100*asmt/grand:>4.0f}%")
    print(f"   smearing chain rule       {chnt:>7.1f} s   {100*chnt/grand:>4.0f}%")
    rest = grand - solve - asmt - chnt
    print(f"   QUDA per-call + floors    {rest:>7.1f} s   {100*rest/grand:>4.0f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--samples-from", type=int, default=1)
    p.add_argument("--cg-slope", type=float, default=3.84, help="ms per CG iteration")
    p.add_argument("--mg-slope", type=float, default=118.0, help="ms per MG(GCR) iteration")
    p.add_argument("--assembly-from", default=None,
                   help="take the force-assembly counters from this log instead "
                        "(use a clean arm when one level's counter is contaminated)")
    p.add_argument("--tol-action", type=float, default=None)
    p.add_argument("--tol-deriv", type=float, default=None)
    a = p.parse_args()
    for log in a.logs:
        report(log, a)


if __name__ == "__main__":
    sys.exit(main())
