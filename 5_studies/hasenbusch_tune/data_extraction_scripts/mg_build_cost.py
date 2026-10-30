#!/usr/bin/env python3
"""Grid multigrid setup cost of a pure-Grid HMC run, from its hmc.log.

Reads the `[GridMG <name> hierarchy] build #N (<reason>, gen G): ... total T s` lines that
`src/grid_mg/grid_mg_hierarchy.h` Build() prints, and counts trajectories by `Total H after`.
Prints, per hierarchy and build reason, the number of builds and their summed and mean setup
time; the total setup per trajectory; the builds that fell inside each trajectory; and the
gauge imports (generations) per trajectory, i.e. how often a re-coarsen-on-every-import
cadence would fire.

Usage:
  mg_build_cost.py <hmc.log> [<hmc.log> ...]
"""
import collections
import re
import sys

BUILD = re.compile(r"\[GridMG (\S+) hierarchy\] build #(\d+) \(([^,]*), gen (\d+)\).*total ([0-9.]+) s")
TRAJ_END = re.compile(r"Total H after")


def summarise(path):
    n = collections.Counter()
    secs = collections.Counter()
    per_traj = collections.Counter()
    last_gen = {}
    ntraj = 0
    with open(path, errors="replace") as f:
        for line in f:
            if TRAJ_END.search(line):
                ntraj += 1
                continue
            m = BUILD.search(line)
            if not m:
                continue
            name, reason, gen, total = m.group(1), m.group(3), int(m.group(4)), float(m.group(5))
            n[(name, reason)] += 1
            secs[(name, reason)] += total
            per_traj[(ntraj, name)] += 1
            last_gen[name] = gen

    print(f"== {path}")
    print(f"trajectories (Total H after): {ntraj}")
    for key in sorted(n):
        print(f"  {key[0]:8s} {key[1]:22s} builds {n[key]:4d}  setup {secs[key]:8.0f} s"
              f"  mean {secs[key] / n[key]:6.1f} s")
    total = sum(secs.values())
    if ntraj:
        print(f"  total setup {total:.0f} s = {total / ntraj:.0f} s per trajectory")
    for name in sorted(last_gen):
        # Index t = builds before trajectory t+1 ended; the last entry is after the final one.
        counts = [per_traj[(t, name)] for t in range(ntraj + 1)]
        print(f"  {name} builds per trajectory: {counts}")
        if ntraj:
            print(f"  {name} gauge imports (generations) up to the last build: {last_gen[name]}"
                  f" (~{last_gen[name] / ntraj:.0f} per trajectory)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        summarise(p)
