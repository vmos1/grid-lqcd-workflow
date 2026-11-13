#!/usr/bin/env python3
"""nsys_copy_stacks.py -- which code issues the host<->GPU copies and the GPU allocations of
Grid's force calls (grid-perf-profile step c, item 1(ii), 2026-11-13).

Input: a rank-0 Nsight Systems report taken with a call stack on the CUDA memory API calls
(nsys_rank.sh with NSYS_SAMPLE=process-tree NSYS_CUDABACKTRACE=memory:<ns>), exported to
sqlite, and the run's hmc.log. Every copy is joined to the API call that issued it and to
that call's stack. From the stack:

  path    the MemoryManager functions on it, innermost first. A device-to-host copy under
          EvictVictims is an eviction (the --device-mem cap made room); under CpuViewOpen it
          is host code reading a field that lives on the GPU. A host-to-device copy under
          AcceleratorViewOpen is a field uploaded because GPU code uses it: it was evicted
          before, or host code wrote it.
  caller  the first Grid functions outside the memory manager, innermost first.

The regions are those of nsys_force_split.py (force calls cut at their solves, same offset).

  uv run python common/nsys_copy_stacks.py <report.sqlite> <hmc.log> [--offset S] [--top N]
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsys_force_split import (PHASE_SEP, Profile, fit_offset, grid_categories, grid_windows,  # noqa: E402
                              merge, parse_grid_log)
from nsys_host_idle import inside, short_sym  # noqa: E402

MM = re.compile(r"MemoryManager::(\w+)")
WRAP = re.compile(r"View|acceleratorCopy|acceleratorAlloc|acceleratorFree|Lattice::Lattice|Lattice::~Lattice|^std::|"
                  r"alignedAllocator|devAllocator|uvmAllocator|LatticeAccelerator|Vector::|vector::|_M_")


class Chains:
    def __init__(self, P):
        self.P = P
        con = P.con
        cc = P._fetch("SELECT id, stackDepth, symbol, module FROM CUDA_CALLCHAINS ORDER BY id, stackDepth", 4)
        self.cc = cc
        uid, first, cnt = np.unique(cc[:, 0], return_index=True, return_counts=True)
        self.slice = {int(i): (int(f), int(c)) for i, f, c in zip(uid, first, cnt)}
        self.str = dict(con.execute("SELECT id, value FROM StringIds WHERE id IN "
                                    "(SELECT symbol FROM CUDA_CALLCHAINS UNION SELECT module FROM CUDA_CALLCHAINS)"))
        r = con.execute("SELECT c.module FROM CUDA_CALLCHAINS c JOIN StringIds s ON s.id = c.symbol "
                        "WHERE s.value = 'main' LIMIT 1").fetchone()
        self.binmod = r[0] if r else None
        self.memo = {}

    def describe(self, cid):
        """(memory-manager path, caller chain) of call chain cid."""
        if cid in self.memo:
            return self.memo[cid]
        if cid not in self.slice:
            out = ("(no call stack)", "")
        else:
            a, n = self.slice[cid]
            mm, own = [], []
            for sym, mod in self.cc[a:a + n, 2:4]:
                if self.binmod is not None and mod != self.binmod:
                    continue
                s = short_sym(self.str.get(int(sym), "?"))
                m = MM.search(s)
                if m:
                    if not mm or mm[-1] != m.group(1):
                        mm.append(m.group(1))
                elif not WRAP.search(s):
                    if not own or own[-1] != s:
                        own.append(s)
            out = (" < ".join(mm) or "(not through the memory manager)", " < ".join(x.replace("Grid::", "") for x in own[:3]))
        self.memo[cid] = out
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("log")
    ap.add_argument("--offset", type=float)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-gb", type=float, default=0.3, help="caller rows below this many GB are not printed")
    a = ap.parse_args()

    P = Profile(a.report)
    C = Chains(P)
    con = P.con
    win = P.window
    W = grid_windows(parse_grid_log(a.log))
    search = (P.launch - 2.0, P.launch + 20.0) if P.launch is not None else (-20.0, 40.0)
    d = a.offset if a.offset is not None else fit_offset(P, W, *search)
    cats = grid_categories(W, d, win)
    force = {k: iv for k, iv in cats.items() if k.startswith("force:")}

    def union(pred):
        return merge(np.vstack([iv for k, iv in force.items() if pred(k)] + [np.zeros((0, 2))]))
    regions = [("ALL force calls, outside solves", union(lambda k: True)),
               ("ALL force calls, before the first solve", union(lambda k: k.endswith(PHASE_SEP + "pre"))),
               ("ALL force calls, after the last solve", union(lambda k: k.endswith(PHASE_SEP + "post"))),
               ("solves", cats["solve"]), ("smearing pullback", cats["pullback"])]
    tail = [k for k in force if "[1][0]" in k]
    for k in sorted(tail):
        regions.append((k, force[k]))

    names = dict(con.execute("SELECT id, value FROM StringIds WHERE id IN (SELECT DISTINCT nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME)"))
    rt = P._fetch("SELECT correlationId, callchainId, nameId, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                  "WHERE callchainId IS NOT NULL", 5)
    corr2chain = dict(zip(rt[:, 0].tolist(), rt[:, 1].tolist()))
    mc = P._fetch("SELECT start, end, bytes, copyKind, COALESCE(correlationId, -1) FROM CUPTI_ACTIVITY_KIND_MEMCPY", 5)
    kinds = dict(con.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER").fetchall())
    ms, me = mc[:, 0] / 1e9, mc[:, 1] / 1e9
    print("# nsys copy stacks\nreport %s\nwindow %.1f s, offset %.3f s; %d copies, %.1f GB; %d CUDA API calls carry a call stack"
          % (a.report, win[1] - win[0], d, len(mc), mc[:, 2].sum() / 1e9, len(rt)))
    with_chain = np.array([c in corr2chain for c in mc[:, 4].tolist()])
    print("copies with a call stack: %d, holding %.1f GB (%.0f%% of the bytes)"
          % (with_chain.sum(), mc[with_chain, 2].sum() / 1e9, 100 * mc[with_chain, 2].sum() / max(mc[:, 2].sum(), 1)))

    for name, region in regions:
        sel = np.nonzero(inside(region, 0.5 * (ms + me)))[0]
        if len(sel) == 0:
            continue
        bypath = defaultdict(lambda: [0, 0.0, 0.0])
        bycall = defaultdict(lambda: [0, 0.0, 0.0, []])
        tot = 0.0
        for i in sel:
            gb = mc[i, 2] / 1e9
            tot += gb
            path, caller = C.describe(corr2chain.get(int(mc[i, 4]), -1))
            kd = str(kinds.get(int(mc[i, 3]), mc[i, 3])).replace("Host-to-Device", "to GPU").replace("Device-to-Host", "to host")
            x = bypath[(kd, path)]
            x[0] += 1
            x[1] += gb
            x[2] += me[i] - ms[i]
            y = bycall[(kd, path, caller)]
            y[0] += 1
            y[1] += gb
            y[2] += me[i] - ms[i]
            y[3].append(mc[i, 2] / 1e6)
        wall = float((region[:, 1] - region[:, 0]).sum())
        print("\n## %s: %.1f GB copied in %.2f s of wall" % (name, tot, wall))
        print("| direction | memory-manager path (innermost first) | copies | GB | share | copy s |\n|---|---|---|---|---|---|")
        for (kd, path), (n, gb, sec) in sorted(bypath.items(), key=lambda kv: -kv[1][1]):
            if gb >= 0.01 * tot:
                print("| %s | %s | %d | %.2f | %.0f%% | %.2f |" % (kd, path, n, gb, 100 * gb / tot, sec))
        print("\n| direction | path | caller (innermost first) | copies | MB each (median) | GB | copy s |\n|---|---|---|---|---|---|---|")
        for (kd, path, caller), (n, gb, sec, mbs) in sorted(bycall.items(), key=lambda kv: -kv[1][1])[:a.top]:
            if gb < a.min_gb:
                break
            print("| %s | %s | %s | %d | %.0f | %.2f | %.2f |" % (kd, path, caller, n, float(np.median(mbs)), gb, sec))

        # GPU allocations and frees with a stack, inside the region
        rs, re_ = rt[:, 3] / 1e9, rt[:, 4] / 1e9
        rsel = np.nonzero(inside(region, 0.5 * (rs + re_)))[0]
        acc = defaultdict(lambda: [0, 0.0])
        for i in rsel:
            nm = names.get(int(rt[i, 2]), "?")
            if not ("Malloc" in nm or "Free" in nm):
                continue
            path, caller = C.describe(int(rt[i, 1]))
            y = acc[(nm.split("_v")[0], path, caller)]
            y[0] += 1
            y[1] += re_[i] - rs[i]
        if acc:
            tsec = sum(v[1] for v in acc.values())
            print("\nGPU allocation calls over the stack threshold: %.2f s in all" % tsec)
            print("| call | path | caller (innermost first) | calls | s |\n|---|---|---|---|---|")
            for (nm, path, caller), (n, sec) in sorted(acc.items(), key=lambda kv: -kv[1][1])[:12]:
                if sec < 0.02 * tsec:
                    break
                print("| %s | %s | %s | %d | %.2f |" % (nm, path, caller, n, sec))


if __name__ == "__main__":
    main()
