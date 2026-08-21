#!/usr/bin/env python3
"""Per-trajectory acceptance tables from a Grid HMC log.

Companion to integrator_time_split.py (which produces grid-vs-quda per-sector
*averages*); this one emits the two per-trajectory tables the acceptance/ladder
docs use, for a SINGLE run:

  Table 1  physics + force-time split : dH, e^(-dH), Metropolis, plaquette
           (unsmeared+smeared), Polyakov, wall, and per-monomial force seconds.
  Table 2  maximum kick (Fdt)         : max |F·dt| per action piece per traj.

Force-time method (same as integrator_time_split.py): difference consecutive
`update_P` integrator timestamps within a trajectory and bucket each interval to
the (lv,ix) monomial whose update_P ENDS it -- MINUS any stout-smearing /
link-update time (Grid's own "Smearing in X ms" line) that fell inside that
gap, which goes to its own synthetic "smear" column instead (see SMEARKEY):
that cost is shared per-half-step integrator overhead, not any one monomial's,
and which bucket would otherwise absorb it depends only on which monomial
happens to co-occupy that integrator level in a given ladder -- splitting it
out is what makes gauge/tail costs comparable across ladders that place tail
differently. Max kick = max of the per-step "[lv][ix] Fdt max" lines over the
trajectory. Trajectory 1 is the cold one (its gauge bucket carries the ~650 s
one-time stencil/QUDA-autotune init) and is flagged, not dropped.

TIMING -- TWO CONVENTIONS, BOTH EMITTED. Never divide one by the other.
  wall (s)  = MD-ONLY: 'Total H before trajectory' -> 'Total H after
              trajectory'. The molecular-dynamics sweep alone -- the part
              MDSTEPS and the Hasenbusch ladder actually control. This is the
              campaign's PRIMARY metric: every headline speedup in the C1/C2/C3
              docs is in this convention.
  cycle (s) = TTT: Grid's own 'Total time for trajectory (s):' field. The full
              production cycle -- also covers checkpoint write, measurement,
              momentum refresh and the pseudofermion heatbath (itself a set of
              solves). Runs ~100-190 s longer than wall. This is the real cost
              of producing one configuration, and the only basis comparable
              with an external code (Chroma emits no MD-only interval).
Because the extra ~180 s is roughly candidate-independent, it is added to both
sides of any ratio and pulls speedups toward 1: base+G/C2 is 1.63x in wall but
1.59x in cycle. Report both; quote wall first. See __docs/README.md.

The monomial->column map is built AUTOMATICALLY from the update_P action names, so
this works for any Hasenbusch ladder (base+G, u1, and future mass-scan ladders):
  *RationalAction*                      -> s      (strange RHMC)
  *LogDet*                              -> ldS/ldL (in index order: 1st=ldS,2nd=ldL)
  *RatioAction det(A)/det(B)*           -> PF0,PF1,... (order of appearance)
  *TwoFlavourSchurCloverAction* (plain) -> tail   (single-det heaviest rung)
  *PlaqPlusRectangleAction*             -> gauge
Column order rendered: s, ldS, ldL, PF0.., tail, gauge (matches the docs).

Usage:
  accept_run_tables.py <hmc_log> [<hmc_log> ...]
  accept_run_tables.py --start 2000 run.log      # traj-number offset (cosmetic)
"""
import re, sys, os
from collections import OrderedDict

TS_UPDP = re.compile(r"Integrator : ([0-9.]+) s :\s*update_P : Level \[(\d)\]\[(\d)\] (.*?)(?: \[QUDA| dt |$)")
FDTMAX  = re.compile(r"\[(\d)\]\[(\d)\] Fdt max\s*:\s*([0-9.eE+-]+)")
FDTAVG  = re.compile(r"\[(\d)\]\[(\d)\] Fdt average\s*:\s*([0-9.eE+-]+)")
HBEF    = re.compile(r"Total H before trajectory")
HAFT    = re.compile(r"Total H after trajectory\s*=\s*[0-9.eE+-]+\s+dH = ([0-9.eE+-]+)")
TTOT    = re.compile(r"Total time for trajectory \(s\):\s*([0-9.eE+-]+)")
TSANY   = re.compile(r": ([0-9.]+) s :")
EXPDH   = re.compile(r"exp\(-dH\) = ([0-9.eE+-]+)")
METROP  = re.compile(r"Metropolis_test -- (\w+)")
PLAQ    = re.compile(r"Plaquette: \[\s*(\d+)\s*\] ([0-9.]+)")
POLY    = re.compile(r"Polyakov Loop: \[\s*(\d+)\s*\] \(([-0-9.eE]+),([-0-9.eE]+)\)")
SMEAR   = re.compile(r"Smearing in ([0-9.eE+-]+) ms")

# Synthetic bucket key for the stout-smearing / link-update cost. This is
# shared per-half-step integrator overhead (recomputing the smeared gauge
# field + its chain rule after every link update), NOT any one monomial's
# own force cost. Left unsplit, it gets misattributed entirely to whichever
# monomial happens to share that integrator level and print next -- e.g. it
# lands on "tail" when tail co-occupies level 1 with gauge, but lands on
# "gauge" when gauge is alone at that level (different ladders put tail at
# different levels), making cross-ladder gauge/tail comparisons meaningless
# unless it's split out. See __docs/2026_7_16_accept_baseG_u1_10traj_tables.md
# "Reading it" note under the single-trajectory table for the log evidence.
SMEARKEY = ("SM", "SM")

# Printed above Table 1 so that every doc which pastes this output is
# self-labelling by construction -- the cross-convention mistake this guards
# against is dividing a `cycle` number by a `wall` baseline.
TIMING_BANNER = (
    "Timing conventions (do not mix): **wall (s)** = MD-only, `Total H before "
    "trajectory` → `Total H after trajectory` — the MD sweep alone, and the "
    "campaign's primary metric. **cycle (s)** = Grid's `Total time for "
    "trajectory (s):` — the full production cycle, additionally covering "
    "checkpoint write, measurement, momentum refresh and the pseudofermion "
    "heatbath (~100–190 s more). Quote speedups as `wall` first, `cycle` "
    "alongside; never form a ratio across the two. See `__docs/README.md`.\n")

def parse(path):
    """Return (list-of-traj-dicts, {(lv,ix): action_name})."""
    trajs, cur, names = [], None, OrderedDict()
    with open(path, errors="ignore") as fh:
        for line in fh:
            if HBEF.search(line):
                t = TSANY.search(line)
                cur = {"t0": float(t.group(1)), "evs": [], "fdt": {}, "fda": {},
                       "plaq": [], "poly": None, "smear_pending": 0.0}
                continue
            if cur is None:
                continue
            ms = SMEAR.search(line)
            if ms:
                cur["smear_pending"] += float(ms.group(1)) / 1000.0
                continue
            mu = TS_UPDP.search(line)
            if mu:
                k = (mu.group(2), mu.group(3))
                cur["evs"].append((float(mu.group(1)), k, cur["smear_pending"]))
                cur["smear_pending"] = 0.0
                names.setdefault(k, mu.group(4).strip())
            mf = FDTMAX.search(line)
            if mf:
                k = (mf.group(1), mf.group(2)); v = float(mf.group(3))
                cur["fdt"][k] = max(cur["fdt"].get(k, 0.0), v)
            ma = FDTAVG.search(line)
            if ma:
                k = (ma.group(1), ma.group(2)); v = float(ma.group(3))
                cur["fda"].setdefault(k, []).append(v)  # per-step averages
            mh = HAFT.search(line)
            if mh:
                t = TSANY.search(line)
                cur["t1"] = float(t.group(1)); cur["dH"] = float(mh.group(1)); continue
            mt = TTOT.search(line)
            if mt:
                # Grid prints this AFTER Metropolis but BEFORE the Plaquette /
                # Polyakov lines that close the trajectory, so `cur` is still open.
                cur["ttt"] = float(mt.group(1)); continue
            me = EXPDH.search(line)
            if me and "exp" not in cur: cur["exp"] = float(me.group(1))
            mm = METROP.search(line)
            if mm and "metrop" not in cur: cur["metrop"] = mm.group(1)
            mp = PLAQ.search(line)
            if mp and "t1" in cur: cur["plaq"].append(float(mp.group(2)))
            mo = POLY.search(line)
            if mo and "t1" in cur and cur["poly"] is None:
                cur["poly"] = (float(mo.group(2)), float(mo.group(3)))
            if cur.get("poly") is not None and len(cur["plaq"]) >= 2:
                trajs.append(cur); cur = None
    if cur is not None and "t1" in cur:
        trajs.append(cur)
    return trajs, names

def build_columns(names):
    """Auto-map (lv,ix) -> column header from action names, in doc order."""
    strange = ldets = []
    s_keys, ld_keys, pf_keys, tail_keys, g_keys = [], [], [], [], []
    for k, nm in names.items():
        low = nm.lower()
        if "rational" in low:            s_keys.append(k)
        elif "logdet" in low:            ld_keys.append(k)
        elif "ratioaction" in low:       pf_keys.append(k)
        elif "plaqplusrectangle" in low: g_keys.append(k)
        elif "twoflavourschurclover" in low: tail_keys.append(k)  # plain (no Ratio)
    cols = []
    for k in s_keys:  cols.append((k, "s"))
    for i, k in enumerate(ld_keys): cols.append((k, "ldS" if i == 0 else "ldL"))
    for i, k in enumerate(pf_keys): cols.append((k, f"PF{i}"))
    for k in tail_keys: cols.append((k, "tail"))
    for k in g_keys:  cols.append((k, "gauge"))
    cols.append((SMEARKEY, "smear"))
    return cols

def buckets(tr):
    """Gap between consecutive update_P prints, credited to the ending
    monomial -- MINUS any stout-smearing/link-update time (reported
    directly by Grid's own "Smearing in X ms" line) that fell inside that
    gap, which is credited separately to SMEARKEY instead. Without this
    split, the smearing cost -- identical shared overhead in every ladder --
    gets misattributed entirely to whichever monomial happens to share its
    integrator level, making cross-ladder comparisons of that monomial
    meaningless (see SMEARKEY comment above)."""
    t, prev = {}, tr["t0"]
    for (ts, k, smear_s) in tr["evs"]:
        t[k] = t.get(k, 0.0) + (ts - prev - smear_s)
        t[SMEARKEY] = t.get(SMEARKEY, 0.0) + smear_s
        prev = ts
    return t

def render(path):
    trajs, names = parse(path)
    cols = build_columns(names)
    tag = os.path.basename(path)
    print(f"\n===== {tag}  ({len(trajs)} trajectories) =====")
    print("monomial map: " + ", ".join(f"[{k[0]}][{k[1]}]={h}" for k, h in cols))
    ch = [h for _, h in cols]
    print("\n## Table 1: physics + force-time split\n")
    print(TIMING_BANNER)
    print("| traj | dH | e^(-dH) | Metrop. | plaquette | smeared | Polyakov | wall (s) | "
          "cycle (s) | " + " | ".join(ch) + " | Σforce |")
    print("|---:|---:|---:|:--:|" + "---:|" * 5 + "--:|" * (len(cols) + 1))
    for i, tr in enumerate(trajs, 1):
        b = buckets(tr); wall = tr["t1"] - tr["t0"]
        cyc = tr.get("ttt")
        pl = sorted(tr["plaq"]); poly = tr["poly"]
        sig = sum(b.get(k, 0.0) for k, _ in cols)
        fc = " | ".join(f"{b.get(k,0.0):.0f}" for k, _ in cols)
        star = "*" if i == 1 else ""
        print(f"| {i}{star} | {tr['dH']:+.4f} | {tr.get('exp',float('nan')):.3f} | "
              f"{tr.get('metrop','?')[:3]} | {pl[0]:.7f} | {pl[-1]:.7f} | "
              f"({poly[0]:+.4f}, {poly[1]:+.4f}) | {wall:.0f} | "
              f"{'—' if cyc is None else f'{cyc:.0f}'} | {fc} | {sig:.0f} |")
    cols2 = [c for c in cols if c[0] != SMEARKEY]  # smearing has no Fdt kick -- drop it here
    ch2 = [h for _, h in cols2]
    print("\n## Table 2: kick (Fdt) per action piece — max / avg\n")
    print("Each cell is `max / avg` (Grid reports both over lattice SITES per step): "
          "max = peak single-site |F·dt| = the stability limiter; avg = site-averaged "
          "|F·dt| = the typical force magnitude. A large max/avg ratio means the force "
          "is spatially LOCALIZED (low-mode / near-critical dominated), not uniform.\n")
    print("| traj | " + " | ".join(ch2) + " |")
    print("|---:|" + "--:|" * len(cols2))
    for i, tr in enumerate(trajs, 1):
        cells = []
        for k, _ in cols2:
            mx = tr["fdt"].get(k, 0.0)
            av = tr["fda"].get(k, [])
            av = sum(av) / len(av) if av else 0.0
            cells.append(f"{mx:.4f} / {av:.4f}")
        print(f"| {i} | " + " | ".join(cells) + " |")
    dHs = [t["dH"] for t in trajs]; exps = [t.get("exp", float('nan')) for t in trajs]
    acc = sum(1 for t in trajs if t.get("metrop") == "ACCEPTED")
    warm = [t["t1"] - t["t0"] for t in trajs][1:] or [0]
    warmc = [t["ttt"] for t in trajs[1:] if t.get("ttt") is not None]
    cyc = (f", warm cycle {min(warmc):.0f}-{max(warmc):.0f}s "
           f"(mean {sum(warmc)/len(warmc):.0f})") if warmc else ""
    print(f"\n-- summary: mean dH={sum(dHs)/len(dHs):+.3f}, "
          f"<e^-dH>={sum(exps)/len(exps):.3f}, accepted={acc}/{len(trajs)}, "
          f"warm wall {min(warm):.0f}-{max(warm):.0f}s "
          f"(mean {sum(warm)/len(warm):.0f}){cyc}"
          "   [wall = MD-only; cycle = full trajectory]")

def run_name(path):
    """Infer a short run label from the log filename (extend as ladders grow)."""
    b = os.path.basename(path).lower()
    if "baseg" in b:                                   return "base+G"
    if "u1" in b:                                      return "u1"
    if any(t in b for t in ("three_level", "tl3", "3lvl", "3level")): return "3-level"
    return os.path.splitext(b)[0]

def write_csv(paths, out, start=2000):
    """Tidy one-row-per-(run,traj) CSV of the plot-relevant scalars.

    Columns: run, traj, traj_abs, accepted, cold, dH, exp_dH, plaquette,
    plaquette_smeared, poly_re, poly_im, wall_s, sigma_force_s, limiter_kick,
    cycle_s.
    limiter_kick = the trajectory's single largest |F·dt| across all monomials.
    wall_s = MD-only; cycle_s = full trajectory (see TIMING_BANNER). cycle_s is
    APPENDED last so existing by-name consumers (plot_accept_runs.py) are
    unaffected; it is empty for logs predating the 'Total time for trajectory'
    line.
    """
    import csv
    cols_hdr = ["run", "traj", "traj_abs", "accepted", "cold", "dH", "exp_dH",
                "plaquette", "plaquette_smeared", "poly_re", "poly_im",
                "wall_s", "sigma_force_s", "limiter_kick", "cycle_s"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols_hdr)
        for p in paths:
            rn = run_name(p)
            trajs, names = parse(p)
            mapped = [k for k, _ in build_columns(names)]
            for i, tr in enumerate(trajs, 1):
                b = buckets(tr)
                pl = sorted(tr["plaq"]); poly = tr["poly"]
                sig = sum(b.get(k, 0.0) for k in mapped)
                lim = max((tr["fdt"].get(k, 0.0) for k in mapped), default=0.0)
                w.writerow([
                    rn, i, start + i,
                    1 if tr.get("metrop") == "ACCEPTED" else 0,
                    1 if i == 1 else 0,
                    f"{tr['dH']:.6f}", f"{tr.get('exp', float('nan')):.6f}",
                    f"{pl[0]:.8f}", f"{pl[-1]:.8f}",
                    f"{poly[0]:.6f}", f"{poly[1]:.6f}",
                    f"{tr['t1'] - tr['t0']:.1f}", f"{sig:.1f}", f"{lim:.4f}",
                    "" if tr.get("ttt") is None else f"{tr['ttt']:.1f}"])
    print(f"wrote {out}  ({sum(len(parse(p)[0]) for p in paths)} rows)")

def main(argv):
    if "--csv" in argv:
        i = argv.index("--csv"); out = argv[i + 1]
        paths = [a for a in argv[:i] + argv[i + 2:] if not a.startswith("--")]
        write_csv(paths, out)
        return
    args = [a for a in argv if a != "--start" and not a.isdigit()]
    for p in args:
        render(p)

if __name__ == "__main__":
    main(sys.argv[1:])
