"""List configs that have no downloaded restart XML, with size and tape path.

usage: python3 noxml_paths.py <gauge_config dump> <xml root, one subdir per stream> [stream ...]

Prints `stream trajectory size path`, one per line, sorted; all streams if
none are given. The output is the list file that cache_status.py and
batch_heads.py read.
"""
import glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_common import load_dump

dump, root, streams = sys.argv[1], sys.argv[2], set(sys.argv[3:])
xml = {}
for fn in glob.glob(os.path.join(root, "*", "*_restart_*.xml")):
    xml.setdefault(os.path.basename(os.path.dirname(fn)), set()).add(
        int(re.search(r"_restart_(\d+)\.xml$", fn).group(1)))
rows = []
for r in load_dump(dump):
    m = r["metadata"]
    s, t = m["lqcd.stream"], m["lqcd.trajectory"]
    if (not streams or s in streams) and t not in xml.get(s, set()):
        rows.append((s, t, r["size"], m["lqcd.storage.path"]))
for s, t, size, p in sorted(rows):
    print(s, t, size, p)
