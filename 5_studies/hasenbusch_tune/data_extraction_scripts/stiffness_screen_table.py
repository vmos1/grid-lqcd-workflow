#!/usr/bin/env python3
"""Tables from the stiffness screen (driver FORCES_ONLY=1 FORCES_STIFFNESS=1, 2026-10-14).

Reads the "FORCES_STIFF <kind> key=value ..." lines of one or more runs and prints:
  1. forces at U (pulled back and raw), the momentum norms, and the heatbath/force times;
  2. the gauge-action check per direction and eps (first and second differences of S
     against the force and the finite-difference stiffness);
  3. per direction v: the matrix T[a][b] = <F_b, M_a v> (unnormalised, vmax applied),
     |M_a v|^2, <v, M_a v>, <P, M_a v>, and the nonlinearity eps |D2| / (2 |M v|);
  4. with --levels, per integrator level l (step h_l = TRAJL / steps_l):
       FMF_l   = sum_{a in l} <F_{>=l}, M_a F_{>=l}>   (F_{>=l} = forces of level l and deeper)
       PMMP_l  = sum_{a,b in l} <M_a P, M_b P>         (from the Gram matrix of the P direction)
       PMF_l   = sum_{a in l} <P, M_a F_{>=l}>
     and their h_l^4 weights; FMF's weight times the pure-gauge calibration c = 2.20e-4
     (fg_stiffness_from_logs.py on runs/2026_10_9_pure_gauge_fg48: dH = c h^4 FMF to 1% from
     12 to 96 steps, FMF in this script's normalisation) is what the gauge's own-force
     coefficient would give for that level.  Caution (2026-10-14, 48^3 screens): applied to
     a nested outer level this overshoots the measured dH by ~50x, and the light level's
     h^4 contractions are ladder-independent (the level total telescopes), so these columns
     size the terms, they do not predict a level's piece; see
     __docs/2026_10_14_stiffness_screen_results.md.
  FMF_l needs a direction per group (or per action, FORCES_STIFF_DIRS=each) of every level
  >= l.  With group directions the group must sit on one level; --act-level overrides the
  group's level for one action (e.g. StrangeLogDet=0 in the 3-level layout).

Usage:
  python3 stiffness_screen_table.py [--levels light=0,strange=0,tail=1,gauge=1 --steps 12,48]
      [--trajl 1.0] [--act-level StrangeLogDet=0,...] [--csv out.csv] label=path/hmc.log ...
Groups: gauge = Gauge; tail = LightSchurPF; strange = Strange, StrangeLogDet;
        light = LightLogDet, PF<k>.
"""
import argparse
import math
import re
import sys

RE_LINE = re.compile(r"FORCES_STIFF (\w+) (.*)$")
C_GAUGE = 2.20e-4   # dH / (h^4 F M F), pure-gauge force-gradient scan, cfg_2000 (step 0 table, corrected normalisation)


def group_of(name):
    if name == "Gauge":
        return "gauge"
    if name == "LightSchurPF":
        return "tail"
    if name in ("Strange", "StrangeLogDet"):
        return "strange"
    return "light"


def kv(rest):
    out = {}
    for tok in rest.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def parse(path):
    runs = {}   # sample -> dict
    meta = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_LINE.search(line)
            if not m:
                continue
            kind, rest = m.group(1), m.group(2)
            d = kv(rest)
            if kind == "meta":
                meta = d
                continue
            if kind == "done":
                continue
            s = int(d.get("sample", 0))
            r = runs.setdefault(s, {"refresh": [], "force": {}, "P": {}, "gram0": {}, "dir": {},
                                    "stiff": {}, "gram": {}, "gaugecheck": {}, "order": []})
            if kind == "refresh":
                r["refresh"].append(d)
            elif kind == "force":
                r["force"][d["act"]] = d
                r["order"].append(d["act"])
            elif kind == "P":
                r["P"] = d
            elif kind == "gram0":
                r["gram0"][(d["a"], d["b"])] = d["dot"]
                r["gram0"][(d["b"], d["a"])] = d["dot"]
            elif kind == "dir":
                r["dir"][d["dir"]] = d
            elif kind == "stiff":
                r["stiff"][(d["dir"], d["eps"], d["act"])] = d
            elif kind == "gram":
                r["gram"][(d["dir"], d["eps"], d["a"], d["b"])] = d["dot"]
                r["gram"][(d["dir"], d["eps"], d["b"], d["a"])] = d["dot"]
            elif kind == "gaugecheck":
                r["gaugecheck"][(d["dir"], d["eps"])] = d
    return meta, runs


def fmt(x, w=12):
    if isinstance(x, str):
        return ("%" + str(w) + "s") % x
    return ("%+" + str(w) + ".4e") % x


def rel(a, b):
    den = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / den


def print_run(label, meta, s, r, args):
    acts = r["order"]
    print("# %s sample %d  gsites=%s central=%s dirs=%s eps=%s" % (
        label, s, meta.get("gsites"), meta.get("central"), meta.get("dirs"), meta.get("eps")))
    print("## forces at U (pulled back / raw, RMS per site and max site norm) and times")
    print("%-14s %12s %12s %12s %12s %10s %10s" % ("act", "avg", "max", "raw_avg", "raw_max", "force_s", "refresh_s"))
    rtime = {}
    for d in r["refresh"]:
        rtime[d["act"]] = rtime.get(d["act"], 0.0) + d["time"]
    for a in acts:
        d = r["force"][a]
        print("%-14s %12.6g %12.6g %12.6g %12.6g %10.2f %10.2f" % (
            a, d["avg"], d["max"], d["raw_avg"], d["raw_max"], d["time"], rtime.get(a, 0.0)))
    if r["P"]:
        print("%-14s norm2=%.6g avg=%.6g max=%.6g" % ("P", r["P"]["norm2"], r["P"]["avg"], r["P"]["max"]))
    print("## gauge-action check per direction / eps: dS/deps vs 2<v,F_g>, d2S/deps2 vs 2<v,M_g v>")
    print("%-9s %-8s %14s %14s %9s %14s %14s %9s" % ("dir", "eps", "dS1", "fd1", "rel", "dS2", "fd2", "rel"))
    for (dn, e), d in sorted(r["gaugecheck"].items()):
        dS2 = d.get("dS2", float("nan"))
        fd2 = d.get("fd2", float("nan"))
        print("%-9s %-8g %+14.6e %+14.6e %9.1e %+14.6e %+14.6e %9.1e" % (
            dn, e, d["dS1"], d["fd1"], rel(d["dS1"], d["fd1"]), dS2, fd2,
            rel(dS2, fd2) if not math.isnan(dS2) else float("nan")))
    # T[dir][eps][a] -> dict of unnormalised contractions
    dirs = list(r["dir"].keys())
    eps_vals = sorted({k[1] for k in r["stiff"]})
    print("## per direction v (unnormalised): |M_a v|^2, <v,M_a v>, <P,M_a v>, <F_b,M_a v> for b = a (self) and b = the level-force proxies; nonlinearity = eps|D2|/(2|Mv|)")
    for dn in dirs:
        vmax = r["dir"][dn]["vmax"]
        for e in eps_vals:
            rows = [(a, r["stiff"].get((dn, e, a))) for a in acts]
            rows = [(a, d) for a, d in rows if d is not None]
            if not rows:
                continue
            print("### dir=%s eps=%g vmax=%.6g |v|^2=%.6g" % (dn, e, vmax, r["dir"][dn]["norm2"]))
            hdr = "%-14s %12s %12s %12s %12s %10s" % ("act a", "|M_a v|^2", "<v,M_a v>", "<P,M_a v>", "<F_a,M_a v>", "nonlin")
            print(hdr + " " + " ".join("%12s" % ("<F_" + b[:7] + ",.>") for b in acts))
            for a, d in rows:
                n2 = d["norm2_Mv"] * vmax * vmax
                nonlin = e * math.sqrt(d.get("norm2_D2", 0.0)) / (2.0 * math.sqrt(d["norm2_Mv"])) if d["norm2_Mv"] > 0 else float("nan")
                line = "%-14s %12.4e %+12.4e %+12.4e %+12.4e %10.2e" % (
                    a, n2, d["dot_v"] * vmax * vmax, d["dot_P"] * vmax, d["dot_" + a] * vmax, nonlin)
                line += " " + " ".join("%+12.4e" % (d["dot_" + b] * vmax) for b in acts)
                print(line)
    # linearity across eps
    if len(eps_vals) >= 2:
        e0, e1 = eps_vals[0], eps_vals[1]
        print("## linearity: relative difference of <F_a, M_a v> and |M_a v|^2 between eps=%g and eps=%g" % (e0, e1))
        for dn in dirs:
            for a in acts:
                d0, d1 = r["stiff"].get((dn, e0, a)), r["stiff"].get((dn, e1, a))
                if d0 and d1:
                    print("  dir=%-8s act=%-14s d(<F_a,M_a v>)=%.2e  d(|M_a v|^2)=%.2e  d(<v,M_a v>)=%.2e" % (
                        dn, a, rel(d0["dot_" + a], d1["dot_" + a]), rel(d0["norm2_Mv"], d1["norm2_Mv"]),
                        rel(d0["dot_v"], d1["dot_v"])))
    if args.levels:
        level_table(label, s, r, acts, dirs, eps_vals, args)


def level_table(label, s, r, acts, dirs, eps_vals, args):
    glev = {}
    for tok in args.levels.split(","):
        g, l = tok.split("=")
        glev[g.strip()] = int(l)
    steps = [int(x) for x in args.steps.split(",")]
    alev = {a: glev[group_of(a)] for a in acts}
    if args.act_level:
        for tok in args.act_level.split(","):
            a, l = tok.split("=")
            alev[a.strip()] = int(l)
    nlev = max(alev.values()) + 1
    if len(steps) != nlev:
        sys.exit("--steps needs %d entries for %d levels" % (nlev, nlev))
    per_action_dirs = all(a in dirs for a in acts)
    # contraction C[a][b][c] = <F_b, M_a F_c> with c an action (per-action dirs) or a group
    def contr(e, a, b, c_items):
        tot = 0.0
        for c in c_items:
            d = r["stiff"].get((c, e, a))
            if d is None:
                return None
            tot += d["dot_" + b] * r["dir"][c]["vmax"]
        return tot
    def pmf(e, a, c_items):
        tot = 0.0
        for c in c_items:
            d = r["stiff"].get((c, e, a))
            if d is None:
                return None
            tot += d["dot_P"] * r["dir"][c]["vmax"]
        return tot
    print("## level pieces (%s sample %d): layout %s, steps %s, TRAJL %g; c_gauge = %.3g" % (
        label, s, args.levels, args.steps, args.trajl, C_GAUGE))
    print("%-5s %-4s %8s %12s %12s %12s %14s %14s %14s %12s" % (
        "eps", "lvl", "h", "FMF", "PMMP", "PMF", "h4*FMF", "h4*PMMP", "h4*PMF", "c*h4*FMF"))
    for e in eps_vals:
        for l in range(nlev):
            h = args.trajl / steps[l]
            in_l = [a for a in acts if alev[a] == l]
            deeper = [a for a in acts if alev[a] >= l]
            if per_action_dirs:
                c_items = deeper
            else:
                groups = sorted({group_of(a) for a in deeper})
                # every action of a used group must be on a level >= l, else the direction mixes levels
                for g in groups:
                    for a in acts:
                        if group_of(a) == g and alev[a] < l:
                            print("  WARNING: level %d: direction group '%s' contains %s on level %d (use FORCES_STIFF_DIRS=each)" % (l, g, a, alev[a]))
                c_items = groups
            fmf = 0.0
            pmf_tot = 0.0
            ok = True
            for a in in_l:
                for b in deeper:
                    v = contr(e, a, b, c_items)
                    if v is None:
                        ok = False
                    else:
                        fmf += v
                v = pmf(e, a, c_items)
                if v is None:
                    ok = False
                else:
                    pmf_tot += v
            pmmp = 0.0
            pok = "P" in dirs
            if pok:
                vP = r["dir"]["P"]["vmax"]
                for a in in_l:
                    for b in in_l:
                        g = r["gram"].get(("P", e, a, b))
                        if g is None:
                            pok = False
                        else:
                            pmmp += g * vP * vP
            print("%-5g %-4d %8.5f %12s %12s %12s %14s %14s %14s %12s" % (
                e, l, h, fmt(fmf) if ok else "n/a", fmt(pmmp) if pok else "n/a", fmt(pmf_tot) if ok else "n/a",
                fmt(h ** 4 * fmf) if ok else "n/a", fmt(h ** 4 * pmmp) if pok else "n/a",
                fmt(h ** 4 * pmf_tot) if ok else "n/a", fmt(C_GAUGE * h ** 4 * fmf) if ok else "n/a"))
        # per-action breakdown of FMF on each level
        print("   per action, eps=%g: <F_{>=l}, M_a F_{>=l}> and h^4 c times it" % e)
        for l in range(nlev):
            h = args.trajl / steps[l]
            in_l = [a for a in acts if alev[a] == l]
            deeper = [a for a in acts if alev[a] >= l]
            c_items = deeper if per_action_dirs else sorted({group_of(a) for a in deeper})
            for a in in_l:
                tot = 0.0
                ok = True
                for b in deeper:
                    v = contr(e, a, b, c_items)
                    if v is None:
                        ok = False
                    else:
                        tot += v
                self_v = contr(e, a, a, c_items)
                print("   L%d %-14s FMF=%s  self<F_a,M_a F>=%s  c*h4*FMF=%s" % (
                    l, a, fmt(tot) if ok else "n/a", fmt(self_v) if self_v is not None else "n/a",
                    fmt(C_GAUGE * h ** 4 * tot) if ok else "n/a"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=path/to/hmc.log")
    ap.add_argument("--levels", default=None, help="group=level list, e.g. light=0,strange=0,tail=1,gauge=1")
    ap.add_argument("--steps", default=None, help="steps per level, outermost first, e.g. 12,48")
    ap.add_argument("--trajl", type=float, default=1.0)
    ap.add_argument("--act-level", default=None, help="per-action level overrides, e.g. StrangeLogDet=0")
    args = ap.parse_args()
    if args.levels and not args.steps:
        sys.exit("--levels needs --steps")
    for spec in args.runs:
        if "=" not in spec:
            sys.exit("run spec must be label=path: %s" % spec)
        label, path = spec.split("=", 1)
        meta, runs = parse(path)
        if not runs:
            print("# %s: no FORCES_STIFF lines in %s" % (label, path))
            continue
        for s in sorted(runs):
            print_run(label, meta, s, runs[s], args)
            print()


if __name__ == "__main__":
    main()
