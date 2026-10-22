"""Per-stream summary of a gauge_config dump: count, trajectory range, save spacing, tape-write dates.

usage: python3 streams.py <gauge_config dump>

Dates are when files were written to HPSS, not when they were generated.
"""
import collections, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_common import load_dump

by = collections.defaultdict(list)
for r in load_dump(sys.argv[1]):
    m = r["metadata"]
    by[m["lqcd.stream"]].append((m["lqcd.trajectory"], m.get("lqcd.storage.mtime", "")))
for s in sorted(by):
    v = sorted(by[s])
    t = [x[0] for x in v]
    gaps = collections.Counter(b - a for a, b in zip(t, t[1:]))
    dates = sorted(x[1] for x in v if x[1])
    print(f"{s:8s} n={len(t):5d} traj {t[0]}..{t[-1]}  spacing {dict(gaps.most_common(4))}  "
          f"written {dates[0]}..{dates[-1]}")
