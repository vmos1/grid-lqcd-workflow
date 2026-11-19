#!/usr/bin/env python3
"""Compare a 16^3 gate log against a reference log from another machine (or build): every
physics line the cross-machine gate looks at, as a table of reference value, new value and
relative difference |a-b|/max(|a|,|b|). It reports; it does not decide pass or fail.

Usage:
  compare_gate16_xmachine.py traj   <ref hmc.log> <new hmc.log>
  compare_gate16_xmachine.py forces <ref hmc.log> <new hmc.log>

traj (smoke16 / gate16_puregrid.sh CASE=traj logs):
  initial plaquette; solver iterations before H (heatbath, then initial action); the initial
  and final action S per monomial (the 'S [i][j] H =' lines, 6 digits); Total H before;
  first force call per monomial (avg / max; raw pre-pullback lines for the smeared fermions,
  the plain lines for the gauge); every raw pre-pullback line (compare_smoke_logs.py skips
  them) and every plain 'Force average/max' line, summarised per monomial; force calls per
  level; Total H after, dH; exp(-dH), Random, Metropolis result; final plaquettes.
forces (forces16 / CASE=forces logs): initial plaquette and every 'FORCES_ONLY level' line.

The ENV lines are diffed field by field first (fields expected to differ between machines
are listed, not hidden). Forces print with 6 significant digits, so a relative difference
below ~1e-6 there is at print precision.
"""
import re
import sys
from collections import OrderedDict

NUM = r"([-+0-9.eE]+|-?nan|-?inf)"
RE_ENV = re.compile(r"^ENV (\S+) (.*)$")
RE_PLAQ0 = re.compile(r"Initial plaquette = " + NUM)
RE_S = re.compile(r"S \[(\d+)\]\[(\d+)\] H = " + NUM)
RE_CG = re.compile(r"ConjugateGradient Converged on iteration (\d+)")
RE_MS = re.compile(r"All shifts have converged iteration (\d+)")
RE_HBEF = re.compile(r"Total H before trajectory = " + NUM)
RE_HAFT = re.compile(r"Total H after trajectory\s*= " + NUM + r"\s+dH = " + NUM)
RE_METRO = re.compile(r"exp\(-dH\) = " + NUM + r"\s+Random = " + NUM)
RE_ACC = re.compile(r"Metropolis_test -- (\w+)")
RE_PLAQ = re.compile(r"Plaquette: \[ \d+ \] " + NUM)
RE_RAW = re.compile(r"\[(\d+)\]\[(\d+)\] Force (average|max)\s*\(raw, pre-pullback\): " + NUM + r" (.*?)\s*$")
RE_PLAIN = re.compile(r"\[(\d+)\]\[(\d+)\] Force (average|max)\s*: " + NUM + r" (.*?)\s*$")
RE_FO = re.compile(r"FORCES_ONLY level (\S+) sample (\d+) avg=" + NUM + r" max=" + NUM)

# ENV fields that are expected to differ between two machines or builds.
ENV_EXPECTED = {"RUN", "JOBID", "BIN", "BIN_SHA256", "GRID_DIRTY", "MPICH_IPC", "MPICH_RDMA"}


def env_fields(line):
    out = OrderedDict()
    for m in re.finditer(r'(\w+)=("[^"]*"|\S+)', line):
        out[m.group(1)] = m.group(2)
    return out


def parse(path):
    d = {"env": None, "plaq0": None, "iters_pre": [], "S": [], "H_before": None,
         "H_after": None, "dH": None, "expmdH": None, "random": None, "metro": None,
         "plaq": [], "raw": OrderedDict(), "plain": OrderedDict(), "fo": OrderedDict(),
         "exit": None}
    before_h = True
    with open(path, errors="replace") as f:
        for line in f:
            if d["env"] is None:
                m = RE_ENV.match(line)
                if m:
                    d["env"] = (m.group(1), env_fields(m.group(2)))
                    continue
            if line.startswith("exit="):
                d["exit"] = line.strip()[5:]
            m = RE_PLAQ0.search(line)
            if m and d["plaq0"] is None:
                d["plaq0"] = float(m.group(1))
                continue
            if before_h:
                m = RE_CG.search(line) or RE_MS.search(line)
                if m:
                    kind = "CG" if "ConjugateGradient Converged" in line else "MultiShift"
                    d["iters_pre"].append((kind, int(m.group(1))))
                    continue
            m = RE_S.search(line)
            if m:
                d["S"].append(("[%s][%s]" % (m.group(1), m.group(2)), float(m.group(3))))
                continue
            m = RE_HBEF.search(line)
            if m:
                d["H_before"] = float(m.group(1))
                before_h = False
                continue
            m = RE_HAFT.search(line)
            if m:
                d["H_after"], d["dH"] = float(m.group(1)), float(m.group(2))
                continue
            m = RE_METRO.search(line)
            if m:
                d["expmdH"], d["random"] = float(m.group(1)), float(m.group(2))
                continue
            m = RE_ACC.search(line)
            if m:
                d["metro"] = m.group(1)
                continue
            m = RE_PLAQ.search(line)
            if m:
                d["plaq"].append(float(m.group(1)))
                continue
            m = RE_RAW.search(line)
            if m:
                key = ("[%s][%s]" % (m.group(1), m.group(2)), m.group(3), m.group(5))
                d["raw"].setdefault(key, []).append(float(m.group(4)))
                continue
            m = RE_PLAIN.search(line)
            if m:
                key = ("[%s][%s]" % (m.group(1), m.group(2)), m.group(3), m.group(5))
                d["plain"].setdefault(key, []).append(float(m.group(4)))
                continue
            m = RE_FO.search(line)
            if m:
                d["fo"][(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return d


def rel(a, b):
    if a is None or b is None:
        return None
    if a == b:
        return 0.0
    den = max(abs(a), abs(b))
    return abs(a - b) / den if den else 0.0


def fmt_rel(r):
    if r is None:
        return "missing"
    return "0 (identical)" if r == 0.0 else "%.2e" % r


ROWS = []


def row(name, a, b, r="auto"):
    if r == "auto":
        r = rel(a, b) if isinstance(a, float) and isinstance(b, float) else None
        rs = fmt_rel(r) if r is not None else ("same" if a == b else "DIFFERENT")
    else:
        rs = r
    ROWS.append((name, "%r" % (a,) if not isinstance(a, str) else a,
                 "%r" % (b,) if not isinstance(b, str) else b, rs))


def flush():
    w = [max(len(r[i]) for r in ROWS + [("quantity", "reference", "new", "rel diff")]) for i in range(4)]
    hdr = ("quantity", "reference", "new", "rel diff")
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
    print("|" + "|".join("-" * (w[i] + 2) for i in range(4)) + "|")
    for r in ROWS:
        print("| " + " | ".join(r[i].ljust(w[i]) for i in range(4)) + " |")
    del ROWS[:]


def env_diff(a, b):
    print("## ENV lines")
    if a["env"] is None or b["env"] is None:
        print("ENV line missing in %s" % ("reference" if a["env"] is None else "new log"))
        return
    print("tag: %s vs %s" % (a["env"][0], b["env"][0]))
    fa, fb = a["env"][1], b["env"][1]
    if list(fa) != list(fb):
        print("field lists differ: only ref %s, only new %s" % (
            [k for k in fa if k not in fb], [k for k in fb if k not in fa]))
    for k in fa:
        if k in fb and fa[k] != fb[k]:
            tag = "expected" if k in ENV_EXPECTED else "UNEXPECTED"
            print("  %-11s %-13s ref=%s  new=%s" % (tag, k, fa[k], fb[k]))
    print()


def series_summary(name, ra, rb):
    """Pair two dicts key -> list of values; one row per key with the max rel diff."""
    for key in ra:
        va, vb = ra[key], rb.get(key, [])
        mon = "%s %s %s" % key
        n = min(len(va), len(vb))
        if n == 0:
            row("%s (%s)" % (name, mon), "%d lines" % len(va), "%d lines" % len(vb), "COUNT DIFFERS")
            continue
        # A run cut short (timeout) is compared on the lines both logs have.
        rels = [rel(x, y) for x, y in zip(va[:n], vb[:n])]
        worst = max(range(len(rels)), key=lambda i: rels[i])
        nsame = sum(1 for r in rels if r == 0.0)
        note = "" if len(va) == len(vb) else " (first %d of %d; COUNT DIFFERS)" % (n, len(va))
        row("%s (%s)" % (name, mon), "%d lines" % len(va),
            "%d identical, worst call %d%s" % (nsame, worst + 1, note), fmt_rel(rels[worst]))
    for key in rb:
        if key not in ra:
            row("%s (%s %s %s)" % ((name,) + key), "absent", "%d lines" % len(rb[key]), "EXTRA")


def traj(a, b):
    env_diff(a, b)
    print("## Start, draws and initial action")
    row("exit", a["exit"] or "none", b["exit"] or "none")
    row("initial plaquette", a["plaq0"], b["plaq0"])
    ia = " ".join("%s:%d" % x for x in a["iters_pre"])
    ib = " ".join("%s:%d" % x for x in b["iters_pre"])
    row("solver iterations before H (heatbath, action)", ia, ib)
    n = len(a["S"]) // 2
    for i, ((ka, va), (kb, vb)) in enumerate(zip(a["S"][:n], b["S"][:n])):
        row("initial S %s" % ka, va, vb)
    row("Total H before", a["H_before"], b["H_before"])
    flush()
    print()
    print("## First force call per monomial")
    for src in ("raw", "plain"):
        for key, va in a[src].items():
            vb = b[src].get(key)
            row("first %s %s %s%s" % (key[0], key[1], key[2], " (raw)" if src == "raw" else ""),
                va[0], vb[0] if vb else None)
    flush()
    print()
    print("## Every force line (raw pre-pullback lines are not read by compare_smoke_logs.py)")
    series_summary("raw", a["raw"], b["raw"])
    series_summary("plain", a["plain"], b["plain"])
    l0a = len(next(iter(a["raw"].values()), []))
    l0b = len(next(iter(b["raw"].values()), []))
    l1a = len(next((v for k, v in a["plain"].items() if k[1] == "average"), []))
    l1b = len(next((v for k, v in b["plain"].items() if k[1] == "average"), []))
    row("force calls L0 / L1", "%d / %d" % (l0a, l1a), "%d / %d" % (l0b, l1b))
    nra = sum(len(v) for k, v in a["raw"].items() if k[1] == "average")
    nrb = sum(len(v) for k, v in b["raw"].items() if k[1] == "average")
    row("raw 'Force average' lines", "%d" % nra, "%d" % nrb)
    flush()
    print()
    print("## End of trajectory")
    for i, ((ka, va), (kb, vb)) in enumerate(zip(a["S"][n:], b["S"][n:])):
        row("final S %s" % ka, va, vb)
    row("Total H after", a["H_after"], b["H_after"])
    row("dH", a["dH"], b["dH"])
    if a["dH"] is not None and b["dH"] is not None:
        row("abs(dH_new - dH_ref)", "", "", "%.3e" % abs(a["dH"] - b["dH"]))
    row("exp(-dH)", a["expmdH"], b["expmdH"])
    row("Random", a["random"], b["random"])
    row("Metropolis", a["metro"] or "none", b["metro"] or "none")
    for i, (pa, pb) in enumerate(zip(a["plaq"], b["plaq"])):
        row("final plaquette %s" % ("thin" if i == 0 else "smeared" if i == 1 else str(i)), pa, pb)
    if len(a["plaq"]) != len(b["plaq"]):
        row("plaquette lines", "%d" % len(a["plaq"]), "%d" % len(b["plaq"]), "COUNT DIFFERS")
    flush()


def forces(a, b):
    env_diff(a, b)
    row("exit", a["exit"] or "none", b["exit"] or "none")
    row("initial plaquette", a["plaq0"], b["plaq0"])
    for key, (avg, mx) in a["fo"].items():
        nb = b["fo"].get(key)
        row("%s sample %s avg" % key, avg, nb[0] if nb else None)
        row("%s sample %s max" % key, mx, nb[1] if nb else None)
    for key in b["fo"]:
        if key not in a["fo"]:
            row("%s sample %s" % key, "absent", "present", "EXTRA")
    flush()


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("traj", "forces"):
        sys.exit(__doc__)
    a, b = parse(sys.argv[2]), parse(sys.argv[3])
    print("reference: %s\nnew:       %s\n" % (sys.argv[2], sys.argv[3]))
    (traj if sys.argv[1] == "traj" else forces)(a, b)


if __name__ == "__main__":
    main()
