"""Which configs have a downloaded restart XML, per stream.

usage: python3 coverage.py <gauge_config dump> <xml root, one subdir per stream>

Prints configs with and without a restart XML, the trajectory span of the
missing ones, and XMLs whose config is not in the catalog.
"""
import collections, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_common import load_dump

cfg = collections.defaultdict(set)
for r in load_dump(sys.argv[1]):
    m = r["metadata"]
    cfg[m["lqcd.stream"]].add(m["lqcd.trajectory"])
xml = collections.defaultdict(set)
for fn in glob.glob(os.path.join(sys.argv[2], "*", "*_restart_*.xml")):
    xml[os.path.basename(os.path.dirname(fn))].add(int(re.search(r"_restart_(\d+)\.xml$", fn).group(1)))
tot = 0
for s in sorted(cfg):
    miss = sorted(cfg[s] - xml[s])
    tot += len(miss)
    span = f"{miss[0]}..{miss[-1]}" if miss else "-"
    print(f"{s:8s} configs {len(cfg[s]):5d}  with XML {len(cfg[s] & xml[s]):5d}  "
          f"without {len(miss):4d} ({span})  XML-only {sorted(xml[s] - cfg[s])}")
print("configs without a restart XML:", tot)
