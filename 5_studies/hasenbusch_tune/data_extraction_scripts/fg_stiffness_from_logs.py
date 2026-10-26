#!/usr/bin/env python3
"""Stiffness along the level force, mined from force-gradient trajectory logs.

Every step of Grid's ForceGradient integrator (Integrator_algorithm.h, FG_update_P) evaluates
the force of every monomial on a level twice: once at the current links U (the update_P
group logged with dt = h^2/24, the "FG" displacement step) and once at the links displaced by
exp(-(h^2/12) F_level) U (the next update_P group of the same level, logged with
dt = (1 - 2 lambda) h = 2h/3).  F_level is the level's total pulled-back force; the
displacement is twice dt times the force because update_P applies the force with
HMC_MOMENTUM_DENOMINATOR = 2 (GaugeImplTypes.h) and update_U exponentiates that momentum.
With |F|^2 = avg^2 * gSites (avg = the logged "Force average", the RMS per-site norm), to
first order in the displacement

    avg1^2 - avg0^2  =  -(h^2/6) * (F_a . M_a . F_level) / gSites,

where M_a v = d F_a / d eps along U(eps) = exp(eps v) U (the stiffness screen's definition,
checked against the screen on 2026-10-14: the gauge's 1.507e9 / V = 142 per site on cfg_2000),

so each monomial's stiffness contracted with its own force on the left and the level force on
the right is available from logs we already have, with no run.  For smeared monomials under
HASEN_GRID_BATCH_SMEAR=1 the logged norms are RAW (pre-pullback), so the quantity is the raw
force's change along the thin-link displacement: a ranking proxy for the thin-link F M F, not
the full expression (handoff __docs/2026_10_14_handoff_stiffness_screen.md, step 0).

Output per run and per (level, action): the number of FG pairs, the mean and first-step
relative change of |F|^2, the derived per-site F M F_level = -6 * d(avg^2) / h^2, and the
level's h^4 weight of it, h^4 * FMF_site * gSites (the piece of dH up to the scheme constant;
pure gauge on cfg_2000: dH = 2.20e-4 * weight from 12 to 96 steps).

Usage:
    python3 fg_stiffness_from_logs.py [--csv out.csv] [--per-traj] label=path/to/hmc.log ...
The label is free text (e.g. baseG_12_48).  Only the Integrator-channel lines are read.
Six-digit norms limit the resolution of d(avg^2)/avg^2 to ~1e-5 per pair; the mean over the
3n+1 force calls of a trajectory beats that.
"""
import argparse
import math
import re
import sys

RE_UPDATE = re.compile(r"update_P : Level \[(\d+)\]\[(\d+)\] (.+?) dt ([0-9.eE+-]+)")
RE_AVG = re.compile(r"\[(\d+)\]\[(\d+)\] Force average\s*(?:\(raw, pre-pullback\))?\s*: ([0-9.eE+-]+)")
RE_MAX = re.compile(r"\[(\d+)\]\[(\d+)\] Force max\s*(?:\(raw, pre-pullback\))?\s*: ([0-9.eE+-]+)")
RE_FG = re.compile(r"FG update ([0-9.eE+-]+) ([0-9.eE+-]+)")
RE_DH = re.compile(r"dH = ([0-9.eE+-]+)")
RE_H0 = re.compile(r"Total H before trajectory")
RE_GRID = re.compile(r"--grid (\d+)\.(\d+)\.(\d+)\.(\d+)")
SUFFIX = " (smeared: raw force, pullback batched)"


def close(a, b, rtol=1e-6):
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def parse(path):
    """Return (gsites, pairs, dh_list).  pairs: list of dicts, one per FG pair."""
    gsites = None
    traj = 0
    pending = {}          # (level, a) -> partial event waiting for its avg/max lines
    windows = {}          # level -> {"fg": fg_dt, "ep": ep, "at": {a: event}}
    pairs = []
    dh = []
    with open(path, "r", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            if gsites is None:
                m = RE_GRID.search(line)
                if m:
                    gsites = 1
                    for g in m.groups():
                        gsites *= int(g)
            if RE_H0.search(line):
                traj += 1
                windows.clear()
                pending.clear()
                continue
            m = RE_DH.search(line)
            if m:
                dh.append(float(m.group(1)))
                continue
            m = RE_FG.search(line)
            if m:
                # The level is the one of the update_P lines that follow; open a window keyed
                # by the fg_dt value and resolve the level on the first event.
                windows["__open__"] = {"fg": float(m.group(1)), "ep": float(m.group(2)), "at": {}}
                continue
            m = RE_UPDATE.search(line)
            if m:
                level, a = int(m.group(1)), int(m.group(2))
                name = m.group(3)
                if name.endswith(SUFFIX):
                    name = name[: -len(SUFFIX)]
                pending[(level, a)] = {"traj": traj, "level": level, "a": a, "name": name,
                                       "dt": float(m.group(4)), "line": ln,
                                       "avg": None, "max": None}
                continue
            m = RE_AVG.search(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key in pending:
                    pending[key]["avg"] = float(m.group(3))
                continue
            m = RE_MAX.search(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key in pending and pending[key]["avg"] is not None:
                    ev = pending.pop(key)
                    ev["max"] = float(m.group(3))
                    level = ev["level"]
                    # Resolve a freshly opened window to this level.
                    if "__open__" in windows and close(ev["dt"], windows["__open__"]["fg"]):
                        windows[level] = windows.pop("__open__")
                    w = windows.get(level)
                    if w is None:
                        continue
                    if close(ev["dt"], w["fg"]):
                        w["at"][ev["a"]] = ev
                    elif close(ev["dt"], w["ep"]) and ev["a"] in w["at"]:
                        e0 = w["at"].pop(ev["a"])
                        h = 1.5 * w["ep"]
                        pairs.append({"traj": ev["traj"], "level": level, "a": ev["a"],
                                      "name": e0["name"], "h": h, "fg": w["fg"],
                                      "avg0": e0["avg"], "avg1": ev["avg"],
                                      "max0": e0["max"], "max1": ev["max"],
                                      "line0": e0["line"], "line1": ev["line"]})
                continue
    return gsites, pairs, dh


def summarise(label, gsites, pairs, dh, per_traj):
    rows = []
    keys = sorted({(p["level"], p["a"]) for p in pairs})
    for (level, a) in keys:
        sel = [p for p in pairs if p["level"] == level and p["a"] == a]
        trajs = sorted({p["traj"] for p in sel})
        groups = [(t, [p for p in sel if p["traj"] == t]) for t in trajs] if per_traj else [("all", sel)]
        for tlabel, grp in groups:
            h = grp[0]["h"]
            name = grp[0]["name"]
            rel = [(p["avg1"] ** 2 - p["avg0"] ** 2) / p["avg0"] ** 2 for p in grp]
            dav2 = [p["avg1"] ** 2 - p["avg0"] ** 2 for p in grp]
            relmax = [(p["max1"] - p["max0"]) / p["max0"] for p in grp]
            n = len(grp)
            mean_rel = sum(rel) / n
            sd_rel = math.sqrt(sum((r - mean_rel) ** 2 for r in rel) / max(n - 1, 1))
            mean_dav2 = sum(dav2) / n
            fmf_site = -6.0 * mean_dav2 / h ** 2
            fmf_first = -6.0 * dav2[0] / h ** 2
            weight = h ** 4 * fmf_site * (gsites or 0)
            avg0 = sum(p["avg0"] for p in grp) / n
            rows.append({"run": label, "traj": tlabel, "level": level, "a": a, "name": name,
                         "h": h, "n": n, "avg0": avg0, "rel_mean": mean_rel, "rel_sd": sd_rel,
                         "rel_first": rel[0], "relmax_mean": sum(relmax) / n,
                         "fmf_site": fmf_site, "fmf_first": fmf_first, "h4_weight": weight})
    return rows


def short(name):
    name = name.replace("TwoFlavourSchurCloverRatioAction ", "ratio ")
    name = name.replace("TwoFlavourSchurCloverAction", "tail")
    name = name.replace("OneFlavourSchurCloverRationalAction", "strange")
    name = name.replace("QCDLogDetCompactCloverEOAction", "logdet")
    name = name.replace("PlaqPlusRectangleAction", "gauge")
    name = name.replace("WilsonGaugeAction", "gauge")
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=path/to/hmc.log")
    ap.add_argument("--csv", default=None, help="write all rows as CSV")
    ap.add_argument("--per-traj", action="store_true", help="one row per trajectory instead of the mean")
    args = ap.parse_args()
    allrows = []
    for spec in args.runs:
        if "=" not in spec:
            sys.exit("run spec must be label=path: %s" % spec)
        label, path = spec.split("=", 1)
        gsites, pairs, dh = parse(path)
        rows = summarise(label, gsites, pairs, dh, args.per_traj)
        allrows.extend(rows)
        print("# %s: gSites=%s FG pairs=%d trajectories with dH=%d dH=%s" % (
            label, gsites, len(pairs), len(dh), " ".join("%+.4f" % x for x in dh[:8]) + (" ..." if len(dh) > 8 else "")))
        hdr = "%-8s %-5s %-34s %-9s %5s %10s %12s %10s %12s %10s %12s %12s" % (
            "traj", "lvl.a", "monomial", "h", "n", "avg0", "rel_mean", "rel_sd", "rel_first", "dmax/max", "FMF_site", "h4*FMF*V")
        print(hdr)
        for r in rows:
            print("%-8s %d.%-3d %-34s %-9.5f %5d %10.5g %+12.4e %10.1e %+12.4e %+10.2e %+12.4e %+12.4e" % (
                str(r["traj"]), r["level"], r["a"], short(r["name"])[:34], r["h"], r["n"], r["avg0"],
                r["rel_mean"], r["rel_sd"], r["rel_first"], r["relmax_mean"], r["fmf_site"], r["h4_weight"]))
        print()
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            for r in allrows:
                w.writerow(r)
        print("# wrote %s (%d rows)" % (args.csv, len(allrows)))


if __name__ == "__main__":
    main()
