#!/usr/bin/env python3
"""Decompose a FORCES_ONLY log into a per-level time budget.

Everything here is recovered from an EXISTING log -- no new instrumentation.
Three sources are combined:

  1. "FORCES_ONLY level <name> sample <s> ... refresh=<R> time=<D>"
     R = wall time of act->refresh()  (the heatbath)
     D = wall time of act->deriv()    (force, INCLUDING the smearing chain rule)

  2. "[QudaSchurOpForce/<label>] N op-force calls (ms/call): W/Z= pack= Wcall=
      sigmacall= unpack="  -- the QUDA force-ASSEMBLY cost inside D, printed
     once per rank at teardown.  Ranks are averaged (spread is <5%).

  3. "[QudaLogDetForce/...] N calls (ms/call): trace+deriv= unpack="
     Same, for the log-det level.

The residual D - assembly is then split into:
  * smearing chain rule -- the stout force pullback, charged to EVERY smeared
    level.  Measured directly off LightLogDet, which has NO solver in its
    deriv: its residual IS the smearing cost.  Assumed equal across levels
    (same smearing, same field size) -- this is the one modelled quantity in
    the table, and it is flagged as such.
  * solver -- whatever is left.

Sample 0 is reported separately from the steady state because the one-time MG
setup (~33 s) lands inside the donor rung's FIRST refresh, which would
otherwise smear a one-off cost across the per-sample average.

Usage:  forces_only_time_budget.py <log> [<log> ...]
"""
import re
import sys
from collections import defaultdict

LEVEL_RE = re.compile(
    r"FORCES_ONLY level (\S+) sample (\d+) avg=(\S+) max=(\S+) refresh=(\S+) time=(\S+)")
RATIO_RE = re.compile(
    r"\[QudaSchurOpForce/TwoFlavourSchurCloverRatioAction det\((-?[\d.]+)\) / det\((-?[\d.]+)\)[^]]*\]\]"
    r" (\d+) op-force calls \(ms/call\): W/Z=(\S+) pack=(\S+) Wcall=(\S+) sigmacall=(\S+) unpack=(\S+)")
TAIL_RE = re.compile(
    r"\[QudaSchurOpForce/TwoFlavourSchurCloverAction[^]]*\]\]"
    r" (\d+) op-force calls \(ms/call\): W/Z=(\S+) pack=(\S+) Wcall=(\S+) sigmacall=(\S+) unpack=(\S+)")
LOGDET_RE = re.compile(
    r"\[QudaLogDetForce/\S+\] (\d+) calls \(ms/call\): trace\+deriv=(\S+) unpack=(\S+)")
LADDER_RE = re.compile(r"LADDER=(\S+)")
MGSETUP_RE = re.compile(r"MG preconditioner built \((\d+) levels\) setup=(\S+) s")


def parse(path):
    with open(path, "rb") as fh:
        text = fh.read().replace(b"\0", b"").decode("utf-8", "replace")

    ladder = []
    m = LADDER_RE.search(text)
    if m:
        ladder = [float(x) for x in m.group(1).split(",")]

    # --- per-level per-sample refresh / deriv ------------------------------
    levels = defaultdict(dict)      # name -> sample -> (refresh, deriv)
    order = []
    for m in LEVEL_RE.finditer(text):
        name, s = m.group(1), int(m.group(2))
        if name not in order:
            order.append(name)
        levels[name][s] = (float(m.group(5)), float(m.group(6)))

    # --- QUDA force-assembly cost, averaged over ranks --------------------
    asm = defaultdict(list)         # level name -> [ms/call summed over stages]
    calls = {}
    for m in RATIO_RE.finditer(text):
        den, num = float(m.group(1)), float(m.group(2))
        n = int(m.group(3))
        tot = sum(float(m.group(i)) for i in range(4, 9))
        # rung k has DenOp=ladder[k], NumOp=ladder[k+1]
        name = None
        for k in range(max(0, len(ladder) - 1)):
            if abs(ladder[k] - den) < 1e-9 and abs(ladder[k + 1] - num) < 1e-9:
                name = "PF%d" % k
        if name is None:
            name = "det(%g)/det(%g)" % (den, num)
        asm[name].append(tot)
        calls[name] = n
    for m in TAIL_RE.finditer(text):
        asm["LightSchurPF"].append(sum(float(m.group(i)) for i in range(2, 7)))
        calls["LightSchurPF"] = int(m.group(1))
    for m in LOGDET_RE.finditer(text):
        asm["LightLogDet"].append(float(m.group(2)) + float(m.group(3)))
        calls["LightLogDet"] = int(m.group(1))

    mg = MGSETUP_RE.search(text)
    mg_setup = float(mg.group(2)) if mg else 0.0
    return order, levels, asm, calls, mg_setup


def report(path):
    order, levels, asm, calls, mg_setup = parse(path)
    if not order:
        print("  (no FORCES_ONLY level lines found)")
        return
    nsamp = max(max(d) for d in levels.values()) + 1

    def assembly(name):
        """QUDA force-assembly seconds per SAMPLE for this level."""
        if name not in asm or not asm[name]:
            return 0.0
        ms_per_call = sum(asm[name]) / len(asm[name])
        per_sample_calls = calls[name] / nsamp
        return ms_per_call * per_sample_calls / 1000.0

    # Smearing chain rule, measured off the solver-free log-det level.
    smear = None
    if "LightLogDet" in levels:
        d = [levels["LightLogDet"][s][1] for s in levels["LightLogDet"] if s > 0]
        if d:
            smear = sum(d) / len(d) - assembly("LightLogDet")

    print("  samples=%d   MG setup (one-time)=%.1f s   "
          "smearing chain rule=%s"
          % (nsamp, mg_setup,
             ("%.2f s/level (measured on LightLogDet)" % smear) if smear else "n/a"))
    print()
    hdr = ("  %-13s %9s %9s %9s %9s %9s %9s"
           % ("level", "heatbath", "deriv", " ->solver", " ->assy", " ->smear", "TOTAL"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    tot = defaultdict(float)
    for name in order:
        ss = [s for s in levels[name] if s > 0] or list(levels[name])
        r = sum(levels[name][s][0] for s in ss) / len(ss)
        d = sum(levels[name][s][1] for s in ss) / len(ss)
        a = assembly(name)
        sm = smear if smear is not None else 0.0
        if name == "LightLogDet":
            sm, solv = d - a, 0.0
        else:
            solv = d - a - sm
        print("  %-13s %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f"
              % (name, r, d, solv, a, sm, r + d))
        tot["hb"] += r
        tot["dv"] += d
        tot["solv"] += solv
        tot["asm"] += a
        tot["smear"] += sm
    print("  " + "-" * (len(hdr) - 2))
    grand = tot["hb"] + tot["dv"]
    print("  %-13s %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f"
          % ("TOTAL/sample", tot["hb"], tot["dv"], tot["solv"],
             tot["asm"], tot["smear"], grand))
    # heatbath is ~pure solve; add it to the solver row for the share
    solver_all = tot["hb"] + tot["solv"]
    print()
    print("  SHARE OF A STEADY-STATE SAMPLE (%.1f s):" % grand)
    print("    solver (heatbath %.1f + deriv %.1f) = %.1f s  (%.0f%%)"
          % (tot["hb"], tot["solv"], solver_all, 100 * solver_all / grand))
    print("    QUDA force assembly               = %.1f s  (%.0f%%)"
          % (tot["asm"], 100 * tot["asm"] / grand))
    print("    smearing chain rule               = %.1f s  (%.0f%%)"
          % (tot["smear"], 100 * tot["smear"] / grand))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=" * 78)
        print(p)
        print("=" * 78)
        report(p)
        print()
