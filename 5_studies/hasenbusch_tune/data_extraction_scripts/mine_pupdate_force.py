#!/usr/bin/env python3
"""Per-monomial force time from a Grid HMC log's integrator lines
    '[l][a] P update elapsed time: X ms (force: Y ms)'
Prints, per [level][action] index: evaluations per trajectory, mean/min/max force seconds per
evaluation, and seconds per trajectory; plus the sum. Trajectories are counted by 'Total time for
trajectory' lines; the optional second argument skips that many leading trajectories (warm-up).
With HASEN_GRID_BATCH_SMEAR=1 the smeared monomials' times are the RAW derivative (the pullback
is logged once per level step as 'batched smeared pullback'), so compare like with like.

Usage: mine_pupdate_force.py LOG [skip_traj]
Used for the pure-grid-hmc tables of __docs/2026_09_30_pure_grid_force_cost_analysis.md
(s.6.1, s.6.2, s.7.5) and 2026_10_02_pure_grid_speedup_summary.md; the same script was run on the
pure-Grid, hybrid and comparison logs so the columns are commensurable."""
import re
import sys
from collections import defaultdict

log = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
pat = re.compile(r"\[(\d+)\]\[(\d+)\] P update elapsed time: ([\d.]+) ms \(force: ([\d.]+) ms\)")
traj = 0
force = defaultdict(list)
with open(log, "rb") as f:
    for raw in f:
        line = raw.decode("utf-8", "replace")
        if "Total time for trajectory" in line:
            traj += 1
            continue
        m = pat.search(line)
        if m and traj >= skip:
            force[(int(m.group(1)), int(m.group(2)))].append(float(m.group(4)) / 1000.0)
ntraj = traj - skip
print(f"{log}: trajectories={traj} used={ntraj}")
print(f"{'idx':>8} {'evals/traj':>10} {'mean s/eval':>12} {'min':>8} {'max':>8} {'s/traj':>9}")
tot = 0.0
for k in sorted(force):
    v = force[k]
    n = len(v) / ntraj if ntraj else len(v)
    mean = sum(v) / len(v)
    tot += mean * n
    print(f"[{k[0]}][{k[1]}] {n:10.1f} {mean:12.3f} {min(v):8.3f} {max(v):8.3f} {mean*n:9.1f}")
print(f"{'sum':>8} {'':>10} {'':>12} {'':>8} {'':>8} {tot:9.1f}")
