#!/usr/bin/env python3
"""Peak GPU memory per node from the `2026_8_20_mem_wrapper.sh` nvidia-smi traces.

Usage:  python3 memtrace_peaks.py <run_dir> [<run_dir> ...]

Each run dir holds one `memtrace_<node>.log` per node, written at a 2 s poll
with one GPU sampled per node (4 ranks/node, --gpus-per-task=1, so the sampled
GPU's usage is one rank's exclusive usage):

    1787259484.709 node=nid008228 0, 2841, 81920;
       epoch          node        gpu  used  total   (MiB)

TRAP: derive the node label with os.path.basename, NOT p.split("memtrace_")[1] --
the substring appears twice in these paths (run dir name AND file name), so the
naive split silently collapses all nodes onto one empty key and prints one
plausible-looking row.
"""
import glob
import os
import re
import sys

GRID_XFER = 1528823808 / 1048576.0  # the Grid Lattice alloc that fails at 4 nodes

ROW = re.compile(r"([\d.]+) node=(\S+) (\d+), (\d+), (\d+);")


def peaks(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "memtrace_*.log"))):
        node = os.path.basename(p)[len("memtrace_"):-len(".log")]
        rows = []
        for line in open(p):
            m = ROW.match(line)
            if m:
                rows.append((float(m.group(1)), int(m.group(4)), int(m.group(5))))
        if not rows:
            continue
        t0 = rows[0][0]
        pk = max(rows, key=lambda r: r[1])
        out.append((node, pk[1], pk[2], pk[0] - t0, pk[2] - pk[1]))
    return out


def main(dirs):
    print("Grid transient that fails at 4 nodes = %.0f MiB\n" % GRID_XFER)
    print("%-46s %-11s %9s %7s %8s %10s %6s"
          % ("run", "node", "peak MiB", "peak %", "t_peak", "free@peak", "fits?"))
    for d in dirs:
        lab = os.path.basename(os.path.normpath(d))
        rows = peaks(d)
        if not rows:
            print("%-46s  (no memtrace_*.log)" % lab)
            continue
        for node, used, tot, tpk, free in rows:
            print("%-46s %-11s %9d %6.1f%% %8.1f %10d %6s"
                  % (lab, node, used, used / tot * 100.0, tpk, free,
                     "yes" if free >= GRID_XFER else "NO"))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
