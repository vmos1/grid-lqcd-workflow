#!/usr/bin/env python3
"""mg_printf_callers.py -- who calls printf inside the multigrid solves of an nsys report.

Written for grid-upstream-rebase gate 6 (2026-11-24: it found the CUDA driver's printf on
every MPI call, slowed by libquadmath's hooks); kept for grid-perf-profile.
Who calls the C library's printf family inside the multigrid solves? For the main-thread
samples that fall in idle GPU time inside the V-cycle intervals (as mg_vcycle_host.py) and
whose top-of-stack function belongs to printf, print how many carry a call stack and, for
those, the chain of frames from the printf function up to the first frame of the program.

  uv run --project grid-lqcd-workflow python grid-lqcd-workflow/common/perf_profile/mg_printf_callers.py REPORT.sqlite [--top 25]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nsys_force_split as nfs  # noqa: E402
import nsys_host_idle as nhi  # noqa: E402

PRINTF = ("printf", "__parse_one_spec", "_itoa", "strchrnul", "__vsnprintf", "__vfprintf", "__vsprintf",
          "_IO_", "__libc_scratch_buffer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--marker", default="blockProjectFused")
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    P = nfs.Profile(a.sqlite)
    ids = [i for i, n in enumerate(P.k_names) if a.marker in n]
    sel = np.isin(P.k_idx, ids)
    ms, me = P.k_s[sel], P.k_e[sel]
    order = np.argsort(ms)
    ms, me = ms[order], me[order]
    brk = np.nonzero(np.diff(ms) > a.gap)[0]
    iv = np.column_stack([ms[np.r_[0, brk + 1]], me[np.r_[brk, len(ms) - 1]]])
    periods = len(ms) - len(iv)
    idle = nfs.subtract(iv, P.busy)
    t_idle = nfs.Cover(idle).total()
    S = nhi.Samples(P)
    idx = np.nonzero(nhi.inside(idle, S.t) & (S.tid == S.main))[0]
    us = 1e6 * t_idle / periods / max(len(idx), 1)
    n_all = n_stack = 0
    depth_all = defaultdict(int)
    chains = defaultdict(int)
    callers = defaultdict(int)
    for i in idx:
        fr = S.frames(i)
        if len(fr) == 0:
            continue
        leaf = S.str.get(int(fr[0][0]), "?")
        if not any(k in leaf for k in PRINTF):
            continue
        n_all += 1
        depth_all[min(len(fr), 6)] += 1
        if len(fr) < 2:
            continue
        n_stack += 1
        chain = []
        prog = "(no frame of the program)"
        for sym, mod in fr:
            name = nhi.short_sym(S.str.get(int(sym), "?"))
            if S.binmod is not None and mod == S.binmod:
                prog = name
                break
            chain.append("%s:%s" % (S.modname(mod).split(".so")[0], name))
        # the frame the program called, and the module chain below it
        first_foreign = chain[-1] if chain else "?"
        callers["%s -> %s" % (prog, first_foreign)] += 1
        mods = []
        for c in chain:
            m = c.split(":")[0]
            if not mods or mods[-1] != m:
                mods.append(m)
        chains[" <- ".join(mods)] += 1
    print("# %s" % a.sqlite)
    print("%d V-cycle periods, %.1f us idle per V-cycle, %d main-thread samples in it (%.3f us per V-cycle each)" % (
        periods, 1e6 * t_idle / periods, len(idx), us))
    print("printf-family samples at the top of the stack: %d = %.1f us per V-cycle; with a call stack: %d (%.0f%%)" % (
        n_all, n_all * us, n_stack, 100.0 * n_stack / max(n_all, 1)))
    print("stack depth of those samples (6 = 6 or more): " + ", ".join("%d: %d" % kv for kv in sorted(depth_all.items())))
    w = n_all * us / max(n_stack, 1)
    print("\n## module chain from the printf function up to the program (us per V-cycle, stacks scaled to all printf samples)")
    for k, v in sorted(chains.items(), key=lambda x: -x[1])[:a.top]:
        print("%8.1f  %s" % (v * w, k))
    print("\n## function of the program and what it called")
    for k, v in sorted(callers.items(), key=lambda x: -x[1])[:a.top]:
        print("%8.1f  %s" % (v * w, k))


if __name__ == "__main__":
    main()
