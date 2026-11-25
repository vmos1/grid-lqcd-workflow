#!/usr/bin/env python3
"""mg_vcycle_host.py -- host time inside the multigrid solves of two nsys reports, per V-cycle.

Written for grid-upstream-rebase gate 6 (2026-11-24, old base against new base); kept for
grid-perf-profile. Where is the main thread while the GPU is idle inside the multigrid
solves, per V-cycle. Companion of mg_vcycle_compare.py (same intervals: runs of the marker
kernel ProbeMG::blockProjectFused less than --gap seconds apart).

The host call-stack samples that fall in idle GPU time inside those intervals are counted by
module, by function at the top of the stack, by innermost function of the program with what it
called, and by function anywhere on the stack. Counts are scaled so that all main-thread
samples together are the idle time per V-cycle (the main thread is on the CPU 98% of it).

Uses Profile / Cover / subtract of nsys_force_split.py and Samples of nsys_host_idle.py
(common/, imported, not changed). The report needs CPU sampling with call stacks.

  uv run --project grid-lqcd-workflow python grid-lqcd-workflow/common/perf_profile/mg_vcycle_host.py OLD.sqlite NEW.sqlite [--top 40]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nsys_force_split as nfs  # noqa: E402
import nsys_host_idle as nhi  # noqa: E402


def stats(path, marker, gap):
    P = nfs.Profile(path)
    ids = [i for i, n in enumerate(P.k_names) if marker in n]
    sel = np.isin(P.k_idx, ids)
    ms, me = P.k_s[sel], P.k_e[sel]
    order = np.argsort(ms)
    ms, me = ms[order], me[order]
    brk = np.nonzero(np.diff(ms) > gap)[0]
    first = np.r_[0, brk + 1]
    last = np.r_[brk, len(ms) - 1]
    iv = np.column_stack([ms[first], me[last]])
    periods = len(ms) - len(iv)
    idle = nfs.subtract(iv, P.busy)
    t_idle = nfs.Cover(idle).total()
    S = nhi.Samples(P)
    hit = nhi.inside(idle, S.t) & (S.tid == S.main)
    idx = np.nonzero(hit)[0]
    n = len(idx)
    us = 1e6 * t_idle / periods / max(n, 1)      # microseconds per V-cycle that one sample stands for
    out = {"path": path, "periods": periods, "wall": nfs.Cover(iv).total(), "idle": t_idle, "samples": n,
           "module": defaultdict(float), "leaf": defaultdict(float), "inner": defaultdict(float),
           "incl": defaultdict(float)}
    leaf = S.cc[S.first[idx], 2:4]
    for sym, mod in leaf:
        m = S.modname(mod)
        out["module"][m] += us
        name = nhi.short_sym(S.str.get(int(sym), "?"))
        if name.startswith("0x") or name == "?":
            name = "(no symbol)"
        out["leaf"]["%s : %s" % (m, name)] += us
    stk = idx[S.depth[idx] > 1]
    out["stacks"] = len(stk)
    w = us * n / max(len(stk), 1)                # the stacks stand for all samples
    for i in stk:
        p = S.path(i)
        own = [x for x in p if not x.startswith("[")]
        lastf = own[-1] if own else "(no frame of the program)"
        callee = p[-1] if p and p[-1].startswith("[") else "itself"
        out["inner"]["%s -> %s" % (lastf, callee)] += w
        for f in set(own):
            out["incl"][f] += w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--marker", default="blockProjectFused")
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    A, B = stats(a.old, a.marker, a.gap), stats(a.new, a.marker, a.gap)
    print("# host side of the multigrid solves, old against new, microseconds per V-cycle")
    for tag, S in (("old", A), ("new", B)):
        print("%s: %s\n     %d V-cycle periods, %.2f s wall, %.2f s idle GPU = %.1f us per V-cycle; "
              "%d main-thread samples in it, %d with a call stack" % (
                  tag, S["path"], S["periods"], S["wall"], S["idle"], 1e6 * S["idle"] / S["periods"],
                  S["samples"], S["stacks"]))
    for key, title, top in (("module", "By module of the function at the top of the stack", 12),
                            ("leaf", "By function at the top of the stack", a.top),
                            ("inner", "By innermost function of the program and what it called", a.top),
                            ("incl", "By function of the program anywhere on the stack (inclusive)", a.top)):
        names = set(A[key]) | set(B[key])
        rows = sorted(((n, A[key].get(n, 0.0), B[key].get(n, 0.0)) for n in names), key=lambda r: -abs(r[2] - r[1]))
        print("\n## %s, largest difference first\n| name | old us | new us | diff us |\n|---|---|---|---|" % title)
        for n, x, y in rows[:top]:
            print("| %s | %.1f | %.1f | %+.1f |" % (n.replace("|", "/"), x, y, y - x))
        if key in ("module", "leaf", "inner"):
            print("| all %d names | %.1f | %.1f | %+.1f |" % (len(rows), sum(r[1] for r in rows), sum(r[2] for r in rows),
                                                             sum(r[2] - r[1] for r in rows)))


if __name__ == "__main__":
    main()
