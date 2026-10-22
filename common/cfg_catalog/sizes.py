"""Classify configs by file size from a gauge_config dump.

usage: python3 sizes.py <dump> hist                 distinct sizes with counts per stream
       python3 sizes.py <dump> classify <cut>       per-stream runs of A (size < cut) / B by trajectory
       python3 sizes.py <dump> get <stream> <traj>  size of one config

Chroma embeds its input XML in every config, so configs written by different
HMC inputs differ in total size. For cl21_48_96_b6p3_m0p2416_m0p2050 the cut
6115370000 separates setup A (header ~54 KB) from setup B (~88 KB); a file far
below both groups is truncated. Size cannot see same-length edits (e.g. MG
blocking 4 4 4 4 vs 3 3 3 4): confirm with lime_head.py on a sample.
"""
import collections, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_common import load_dump

size = {}
for r in load_dump(sys.argv[1]):
    m = r["metadata"]
    size[(m["lqcd.stream"], m["lqcd.trajectory"])] = r["size"]
mode = sys.argv[2]

if mode == "hist":
    c = collections.Counter(size.values())
    for s, n in sorted(c.items()):
        streams = collections.Counter(k[0] for k, v in size.items() if v == s)
        print(f"{s}  n={n:5d}  {dict(streams)}")

elif mode == "classify":
    cut = int(sys.argv[3])
    by = collections.defaultdict(list)
    for (s, t), v in size.items():
        by[s].append((t, "A" if v < cut else "B"))
    for s in sorted(by):
        runs = []
        for t, c in sorted(by[s]):
            if runs and runs[-1][0] == c:
                runs[-1][2] = t
                runs[-1][3] += 1
            else:
                runs.append([c, t, t, 1])
        na = sum(r[3] for r in runs if r[0] == "A")
        print(f"{s:8s} A={na:4d} B={len(by[s]) - na:4d}  " +
              "  ".join(f"{c}:{a}-{b}({n})" for c, a, b, n in runs))

elif mode == "get":
    print(size[(sys.argv[3], int(sys.argv[4]))])
