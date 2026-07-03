#!/usr/bin/env python3
"""Per-sector MD time split from Grid HMC logs — one table per code.

Differences consecutive `update_P` integrator timestamps within each trajectory
and buckets each interval by the monomial whose update_P *ends* it (the deriv runs
in the interval leading up to its update_P line — verified by the one-time gauge
init landing in traj 0's strange->gauge gap, and by per-call times matching the
independent FORCES_ONLY tables). Trajectory 0 is excluded (one-time gauge stencil
init + QUDA cold autotune). `update_U` (gauge-field exponentiation) happens only at
the finest integrator level, so it lands in the Gauge bucket -> "Gauge (+U)" here is
gauge-sector MD cost, slightly above the pure gauge-deriv in the FORCES tables. The
fermion buckets are clean deriv (= solver) times and match FORCES_ONLY.

Each code gets its own table reflecting its real integrator hierarchy:
  2plus1  - single-timescale fermions (lvl0) + gauge (lvl1)
  compact - 3-level multi-timescale; unpreconditioned light "Tail"; one Nf=3 logdet
  schur   - 3-level multi-timescale; EO-Schur light with a separate light logdet
Rows are ordered Light -> Strange -> Gauge with LIGHT/STRANGE sector subtotals.
grid and quda modes are shown side by side.

Usage: integrator_time_split.py <hmc_log> [<hmc_log> ...]   # any mix of the 6 logs
"""
import re, sys, os
from collections import defaultdict

# Per-code ordered monomial rows: (level, idx, label, sector, timescale-note).
CODE_ROWS = {
    "2plus1": [
        ("0", "0", "Light LogDet",      "LIGHT",   ""),
        ("0", "1", "Light Schur/Tail",  "LIGHT",   ""),
        ("0", "2", "Strange LogDet",    "STRANGE", ""),
        ("0", "3", "Strange RHMC",      "STRANGE", ""),
        ("1", "0", "Gauge (+U)",        "GAUGE",   ""),
    ],
    "compact": [
        ("1", "0", "Light Tail (unprec)",           "LIGHT",   "mid"),
        ("0", "0", "Strange RHMC",                   "STRANGE", "coarse"),
        ("0", "1", "Nf=3 LogDet (light+strange)",    "SHARED",  "coarse"),
        ("2", "0", "Gauge (+U)",                     "GAUGE",   "fine"),
    ],
    "schur": [
        ("1", "0", "Light LogDet",      "LIGHT",   "mid"),
        ("1", "1", "Light Schur/Tail",  "LIGHT",   "mid"),
        ("0", "1", "Strange LogDet",    "STRANGE", "coarse"),
        ("0", "0", "Strange RHMC",      "STRANGE", "coarse"),
        ("2", "0", "Gauge (+U)",        "GAUGE",   "fine"),
    ],
}
TITLE = {
    "2plus1":  "`2plus1` — single-timescale fermions (production-like)",
    "compact": "`compact` — 3-level multi-timescale, unpreconditioned light \"Tail\", single Nf=3 logdet",
    "schur":   "`schur` — 3-level multi-timescale, EO-Schur light (separate light logdet)",
}

TS   = re.compile(r"Integrator : ([0-9.]+) s :")
UPDP = re.compile(r"Integrator : ([0-9.]+) s :\s*update_P : Level \[(\d)\]\[(\d)\]")
HBEF = re.compile(r"Total H before trajectory")
HAFT = re.compile(r"Total H after trajectory")

def code_of(path):
    b = os.path.basename(path).lower()
    if "2plus1" in b: return "2plus1"
    if "schur"  in b: return "schur"
    if "compact" in b: return "compact"
    raise ValueError(f"cannot infer code from {path}")

def mode_of(path):
    return "quda" if "quda" in os.path.basename(path).lower() else "grid"

def parse(path):
    trajs, cur, last_ts = [], None, None
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = TS.search(line)
            if m: last_ts = float(m.group(1))
            if HBEF.search(line):
                t = re.search(r": ([0-9.]+) s :", line)
                cur = (float(t.group(1)) if t else last_ts, [])
                continue
            if HAFT.search(line) and cur is not None:
                t = re.search(r": ([0-9.]+) s :", line)
                trajs.append((cur[0], float(t.group(1)) if t else last_ts, cur[1]))
                cur = None
                continue
            mu = UPDP.search(line)
            if mu and cur is not None:
                cur[1].append((float(mu.group(1)), mu.group(2), mu.group(3)))
    return trajs

def analyze(path):
    """Return per-(lvl,idx) {s_traj, s_call, calls} and wall, averaged over steady trajs."""
    trajs = parse(path)
    steady = trajs[1:] if len(trajs) > 1 else trajs
    n = len(steady)
    t = defaultdict(float); c = defaultdict(int); walls = []
    for (t0, t1, evs) in steady:
        walls.append(t1 - t0); prev = t0
        for (ts, lv, ix) in evs:
            t[(lv, ix)] += ts - prev; prev = ts
            c[(lv, ix)] += 1
    wall = sum(walls) / n if n else 0.0
    out = {}
    for k in t:
        s_traj = t[k] / n; calls = c[k] / n
        out[k] = {"s_traj": s_traj, "calls": calls,
                  "s_call": s_traj / calls if calls else 0.0}
    return out, wall

def fmt(x, nd=1): return f"{x:.{nd}f}"

def render(code, g, gw, q, qw):
    rows = CODE_ROWS[code]
    print(f"\n### {TITLE[code]}\n")
    print("| Monomial | calls/traj | grid s/call | grid s/traj | % grid | quda s/call | quda s/traj | % quda |")
    print("|---|---|---|---|---|---|---|---|")
    sect_g = defaultdict(float); sect_q = defaultdict(float)
    def subtotal(sec):
        print(f"| **→ {sec}** | | | **{fmt(sect_g[sec])}** | **{fmt(100*sect_g[sec]/gw)}%** | "
              f"| **{fmt(sect_q[sec])}** | **{fmt(100*sect_q[sec]/qw)}%** |")
    for i, (lv, ix, label, sec, note) in enumerate(rows):
        gi = g.get((lv, ix), {"s_call":0,"s_traj":0,"calls":0})
        qi = q.get((lv, ix), {"s_call":0,"s_traj":0,"calls":0})
        calls = f"{int(round(gi['calls']))}" + (f" ({note})" if note else "")
        print(f"| {label} | {calls} | {fmt(gi['s_call'],2)} | {fmt(gi['s_traj'])} | "
              f"{fmt(100*gi['s_traj']/gw)}% | {fmt(qi['s_call'],2)} | {fmt(qi['s_traj'])} | "
              f"{fmt(100*qi['s_traj']/qw)}% |")
        if sec in ("LIGHT", "STRANGE"):
            sect_g[sec] += gi["s_traj"]; sect_q[sec] += qi["s_traj"]
            nxt = rows[i+1][3] if i+1 < len(rows) else None
            if nxt != sec:
                subtotal(sec)
    print(f"| **Wall/traj** | | | **{gw:.0f}** | | | **{qw:.0f}** | |")

def main(paths):
    byc = defaultdict(dict)
    for p in paths:
        byc[code_of(p)][mode_of(p)] = p
    for code in ("2plus1", "compact", "schur"):
        if code not in byc or "grid" not in byc[code] or "quda" not in byc[code]:
            continue
        g, gw = analyze(byc[code]["grid"])
        q, qw = analyze(byc[code]["quda"])
        render(code, g, gw, q, qw)

if __name__ == "__main__":
    main(sys.argv[1:])
