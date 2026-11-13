#!/usr/bin/env python3
"""nsys_host_idle.py -- what the host is doing while the GPU is idle in Grid's force calls
(grid-perf-profile step c, item 1(i), 2026-11-13).

Input: a rank-0 Nsight Systems report taken with CPU sampling (nsys_rank.sh with
NSYS_SAMPLE=process-tree: an instruction-pointer sample every 2,000,000 CPU cycles per
thread, a dwarf call stack on every 4th, context switches traced), exported to sqlite, and
the run's hmc.log. The force calls are cut out of the log exactly as nsys_force_split.py
does it (same categories, same offset fit). Inside each region:

  idle      = the region minus every GPU kernel and copy; "long gaps" are idle stretches of
              at least --min-gap-ms (default 1 ms).
  samples   = the CPU samples whose time stamp falls in the idle part. They are counted per
              thread, per function at the top of the stack (all samples), and per call path
              (the samples that carry a call stack).
  on CPU    = from the context switches: how much of the idle time the main thread was
              running at all. Idle time with the main thread off the CPU is a blocking wait.

  uv run python common/nsys_host_idle.py <report.sqlite> <hmc.log> [--offset S] [--label REGEX]
  uv run python common/nsys_host_idle.py <report.sqlite> --whole-window      (no log: one region)

One sample stands for the same number of CPU cycles, so shares of samples are shares of CPU
time of that thread; seconds are samples times the main thread's measured time per sample.
Sampling slows the host code it measures: read shares, not absolute times (L128).
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsys_force_split import (PHASE_SEP, Cover, Profile, _strip_groups, fit_offset,  # noqa: E402
                              grid_categories, grid_windows, intersect, merge, parse_grid_log, subtract)

ROOT_NOISE = {"[Broken backtraces]", "_start", "__libc_start_call_main", "__libc_start_main@@GLIBC_2",
              "__libc_start_main", "start_thread", "clone3", "__clone3", "clone"}


def short_sym(s):
    """A function name without template arguments, argument lists and return type."""
    s = s.strip().replace("operator()", "operator_call")
    omp = " [omp]" if "omp_fn" in s else ""
    s = re.sub(r"\s*\[clone [^\]]+\]", "", s)
    s = _strip_groups(_strip_groups(s, "<", ">"), "(", ")")
    s = re.sub(r"operator\s+(\w)", r"operator_\1", s)
    s = re.sub(r"\s+const\b", "", s).strip()
    toks = s.split()
    return ((toks[-1] if toks else "?") + omp)[:110]


def inside(iv, t):
    """Mask: time stamps t that fall in the merged interval array iv."""
    if len(iv) == 0:
        return np.zeros(len(t), bool)
    k = np.searchsorted(iv[:, 0], t, side="right") - 1
    return (k >= 0) & (t < iv[np.clip(k, 0, None), 1])


class Samples:
    def __init__(self, P):
        self.P = P
        con = P.con
        ev = P._fetch("SELECT id, start, globalTid FROM COMPOSITE_EVENTS ORDER BY id", 3)
        self.id, self.t, self.tid = ev[:, 0], ev[:, 1] / 1e9, ev[:, 2]
        cc = P._fetch("SELECT id, stackDepth, COALESCE(symbol, -1), COALESCE(module, -1) "
                      "FROM SAMPLING_CALLCHAINS ORDER BY id, stackDepth", 4)
        self.cc = cc
        uid, first, cnt = np.unique(cc[:, 0], return_index=True, return_counts=True)
        pos = np.searchsorted(uid, self.id)
        ok = (pos < len(uid)) & (uid[np.clip(pos, 0, len(uid) - 1)] == self.id)
        self.first = np.where(ok, first[np.clip(pos, 0, len(uid) - 1)], -1)   # row of the leaf frame
        self.depth = np.where(ok, cnt[np.clip(pos, 0, len(uid) - 1)], 0)      # frames in the stack
        self.str = dict(con.execute("SELECT id, value FROM StringIds WHERE id IN "
                                    "(SELECT symbol FROM SAMPLING_CALLCHAINS UNION SELECT module FROM SAMPLING_CALLCHAINS)"))
        self.str[-1] = "?"
        # the program's own module: the one that holds main
        r = con.execute("SELECT c.module FROM SAMPLING_CALLCHAINS c JOIN StringIds s ON s.id = c.symbol "
                        "WHERE s.value = 'main' LIMIT 1").fetchone()
        self.binmod = r[0] if r else None
        # threads: main has tid == pid; names from ThreadNames when present
        self.pid = {int(g): (int(g) >> 24) & 0xFFFFFF for g in np.unique(self.tid)}
        self.tnum = {int(g): int(g) & 0xFFFFFF for g in np.unique(self.tid)}
        self.tname = {}
        if "ThreadNames" in P.tables:
            for g, v in con.execute("SELECT t.globalTid, s.value FROM ThreadNames t JOIN StringIds s ON s.id = t.nameId"):
                self.tname[int(g)] = v
        mains = [g for g in self.pid if self.pid[g] == self.tnum[g]]
        # several processes can be in the tree (the launcher shell): main = the busiest such thread
        self.main = max(mains, key=lambda g: int((self.tid == g).sum())) if mains else int(np.bincount(
            np.unique(self.tid, return_inverse=True)[1]).argmax())
        self._sched()

    def _sched(self):
        """On-CPU intervals per thread from the context switches."""
        P, w0, w1 = self.P, self.P.window[0], self.P.window[1]
        self.oncpu = {}
        if "SCHED_EVENTS" not in P.tables:
            return
        r = P._fetch("SELECT globalTid, start, isSchedIn FROM SCHED_EVENTS ORDER BY globalTid, start", 3)
        for g in np.unique(r[:, 0]):
            x = r[r[:, 0] == g]
            t, sin = x[:, 1] / 1e9, x[:, 2] == 1
            sel = sin[:-1] & ~sin[1:]
            iv = list(zip(t[:-1][sel], t[1:][sel]))
            if not sin[0]:
                iv.append((w0, t[0]))
            if sin[-1]:
                iv.append((t[-1], w1))
            self.oncpu[int(g)] = merge(iv)

    def oncpu_of(self, g):
        """A thread with samples but no context switch in the window ran the whole window."""
        if g in self.oncpu:
            return self.oncpu[g]
        return np.array([self.P.window], dtype=float)

    def label(self, g):
        role = "main" if g == self.main else self.tname.get(g, "thread")
        return "%s (tid %d)" % (role, self.tnum.get(g, 0))

    def frames(self, i):
        """Frames of sample i, leaf first: (symbol id, module id)."""
        a = self.first[i]
        return self.cc[a:a + self.depth[i], 2:4]

    def modname(self, m):
        return self.str.get(int(m), "?").rsplit("/", 1)[-1]

    def path(self, i):
        """Call path of sample i, root first. Every frame of the program is kept; a run of
        frames in other modules becomes one node, named by the function the program called."""
        fr = self.frames(i)[::-1]
        out, foreign = [], False
        for sym, mod in fr:
            name = self.str.get(int(sym), "?")
            if name in ROOT_NOISE:
                continue
            if self.binmod is not None and mod == self.binmod:
                out.append(short_sym(name))
                foreign = False
            elif not foreign:
                out.append("[%s] %s" % (self.modname(mod), short_sym(name)))
                foreign = True
        return out


def report_region(S, name, region, long_gaps, a, tree):
    P = S.P
    wall = Cover(region).total()
    if wall <= 0:
        return
    idle = subtract(region, P.busy)
    idle_long = intersect(region, long_gaps)
    t_idle, t_long = Cover(idle).total(), Cover(idle_long).total()
    print("\n## %s" % name)
    print("wall %.2f s; GPU idle %.2f s (%.0f%%), of which %.2f s in gaps of at least %g ms"
          % (wall, t_idle, 100 * t_idle / wall, t_long, a.min_gap_ms))
    use = idle_long if a.long_only else idle
    t_use = t_long if a.long_only else t_idle
    what = "idle gaps of at least %g ms" % a.min_gap_ms if a.long_only else "all idle time"
    if t_use <= 0:
        return
    sel = inside(use, S.t)
    main_on = Cover(intersect(use, S.oncpu_of(S.main))).total()
    nmain = int((sel & (S.tid == S.main)).sum())
    print("host, over %s (%.2f s): main thread on the CPU %.2f s (%.0f%%), blocked %.2f s; "
          "%d samples on it = %.2f s at the window rate of %.3f ms per sample"
          % (what, t_use, main_on, 100 * main_on / t_use, t_use - main_on, nmain, nmain * a.sps, 1e3 * a.sps))

    # threads
    print("\n| thread | samples in the idle time | CPU s | on CPU s (context switches) |\n|---|---|---|---|")
    tids, cnt = np.unique(S.tid[sel], return_counts=True)
    rest = 0
    for k, (g, c) in enumerate(sorted(zip(tids, cnt), key=lambda x: -x[1])):
        if k >= 12:
            rest += c
            continue
        on = Cover(intersect(use, S.oncpu_of(int(g)))).total()
        print("| %s | %d | %.2f | %.2f |" % (S.label(int(g)), c, c * a.sps, on))
    if rest:
        print("| other threads | %d | %.2f | |" % (rest, rest * a.sps))

    # host CUDA API calls overlapping the idle time (from the CUDA trace, not the samples)
    C = Cover(use)
    api = sorted(((float(C.overlap(s, e).sum()), k, len(s)) for k, (s, e) in P.host.items()), reverse=True)
    api = [x for x in api if x[0] >= 0.005 * t_use][:8]
    if api:
        print("\nCUDA API calls open during this idle time (CUDA trace; calls can nest and overlap): "
              + "; ".join("%s %.2f s" % (k.replace("cuda ", ""), v) for v, k, _ in api))

    for who, tsel in (("main thread", S.tid == S.main), ("all other threads", S.tid != S.main)):
        idx = np.nonzero(sel & tsel)[0]
        if len(idx) < 20:
            continue
        # top of stack, every sample
        leaf = S.cc[S.first[idx], 2:4]
        keys, c = np.unique(leaf, axis=0, return_counts=True)
        order = np.argsort(-c)
        print("\n### %s: function at the top of the stack (%d samples)" % (who, len(idx)))
        print("| share | CPU s | module | function |\n|---|---|---|---|")
        for j in order[:a.top]:
            if c[j] < a.min_frac * len(idx):
                break
            print("| %.1f%% | %.2f | %s | %s |" % (100 * c[j] / len(idx), c[j] * a.sps, S.modname(keys[j, 1]),
                                                 short_sym(S.str.get(int(keys[j, 0]), "?"))))
        bym = defaultdict(int)
        for (sym, mod), n in zip(keys, c):
            bym[S.modname(mod)] += n
        print("by module: " + "; ".join("%s %.1f%%" % (m, 100 * n / len(idx))
                                        for m, n in sorted(bym.items(), key=lambda x: -x[1])[:8]))
        # call paths, the samples that carry a stack
        stk = idx[S.depth[idx] > 1]
        if len(stk) < 20:
            continue
        w = len(idx) / len(stk)          # samples each stack stands for
        paths = [S.path(i) for i in stk]
        inner = defaultdict(int)
        incl = defaultdict(int)
        for p in paths:
            own = [x for x in p if not x.startswith("[")]
            last = own[-1] if own else "(no frame of the program)"
            callee = p[-1] if p and p[-1].startswith("[") else "itself"
            inner[(last, callee)] += 1
            for f in set(own):
                incl[f] += 1
        print("\n### %s: innermost Grid function and what it was in (%d call stacks, each stands for %.1f samples)"
              % (who, len(stk), w))
        print("| share | CPU s | Grid function | time is in |\n|---|---|---|---|")
        for (f, callee), n in sorted(inner.items(), key=lambda x: -x[1])[:a.top]:
            if n < a.min_frac * len(stk):
                break
            print("| %.1f%% | %.2f | %s | %s |" % (100 * n / len(stk), n * w * a.sps, f, callee))
        print("\n### %s: Grid functions by inclusive share (anywhere on the stack)" % who)
        print("| share | CPU s | function |\n|---|---|---|")
        for f, n in sorted(incl.items(), key=lambda x: -x[1])[:a.top]:
            if n < 2 * a.min_frac * len(stk):
                break
            print("| %.1f%% | %.2f | %s |" % (100 * n / len(stk), n * w * a.sps, f))
        if tree and who == "main thread":
            root = [0, {}]
            for p in paths:
                node = root
                node[0] += 1
                for f in p[:a.tree_depth]:
                    node = node[1].setdefault(f, [0, {}])
                    node[0] += 1
            print("\n### %s: call tree (share of call stacks; branches under %.0f%% dropped)" % (who, 100 * a.tree_frac))
            print("```")

            def walk(node, d):
                for f, ch in sorted(node[1].items(), key=lambda x: -x[1][0]):
                    if ch[0] < a.tree_frac * len(stk):
                        continue
                    print("%5.1f%% %s%s" % (100 * ch[0] / len(stk), " " * min(d, 30), f))
                    walk(ch, d + 1)
            walk(root, 0)
            print("```")


def _has(p, s):
    return any(s in f for f in p)


# Host hot spots of the force calls, first match wins (found in the 2026-11-13 profile).
HOT = [
    ("gauge import, stock path: clover inverse on the host (Eigen)", lambda p: _has(p, "CompactWilsonCloverHelpers::Invert")),
    ("gauge import, stock path: the rest", lambda p: _has(p, "CompactWilsonCloverFermion::ImportGauge")),
    ("gauge import, fork's guarded path", lambda p: _has(p, "GuardedImportGauge") or _has(p, "FastImportGauge")),
    ("gauge force: stencil table built per call (GeneralLocalStencil)", lambda p: _has(p, "GeneralLocalStencil::GeneralLocalStencil")),
    ("MPI", lambda p: bool(p) and p[-1].startswith(("[libmpi", "[libfabric", "[libcxi"))),
    ("CUDA driver and runtime", lambda p: bool(p) and p[-1].startswith(("[libcuda", "[libnvidia"))),
]
CALLERS = ["TwoFlavourSchurCloverActionMP::deriv", "GridMGSchurSolver::SetGauge", "MixedPrecCGRungSolver",
           "OneFlavourSchurCloverRationalActionMP::deriv", "TwoFlavourSchurCloverRatioAction"]
MEMFRAME = re.compile(r"MemoryManager|View|acceleratorCopy|accelerator|Lattice::Lattice|Lattice::~Lattice|std::")


def hot_of(p):
    for name, test in HOT:
        if test(p):
            if name.startswith("gauge import, stock"):
                via = next((c for c in CALLERS if _has(p, c)), "caller not on the stack")
                return "%s | %s" % (name, via)
            return "%s | " % name
    return "other host code | "


def detail(S, name, region, a):
    """Main thread only: its CPU time by host hot spot, in the idle part of the region and in
    the whole region; and the large copies of the region by the code that issued them."""
    P = S.P
    idle = subtract(region, P.busy)
    wall, t_idle = Cover(region).total(), Cover(idle).total()
    ismain = S.tid == S.main
    rows = defaultdict(lambda: [0.0, 0.0])
    for col, iv in ((0, idle), (1, region)):
        idx = np.nonzero(inside(iv, S.t) & ismain)[0]
        stk = idx[S.depth[idx] > 1]
        if len(stk) == 0:
            continue
        w = len(idx) / len(stk) * a.sps
        for i in stk:
            rows[hot_of(S.path(i))][col] += w
    print("\n## %s: the main thread's CPU time by host hot spot" % name)
    print("region wall %.2f s, GPU idle %.2f s. CPU seconds of the main thread from its call stacks; the part of the "
          "idle time it leaves is kernel time (not sampled) and blocking waits." % (wall, t_idle))
    print("| hot spot | called from | CPU s in the idle time | share of the idle time | CPU s in the whole region | share of the region |")
    print("|---|---|---|---|---|---|")
    tot = [0.0, 0.0]
    for k, (x, y) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        hs, via = k.split(" | ")
        tot[0] += x
        tot[1] += y
        print("| %s | %s | %.2f | %.0f%% | %.2f | %.0f%% |" % (hs, via, x, 100 * x / max(t_idle, 1e-9), y, 100 * y / wall))
    print("| **all sampled** | | %.2f | %.0f%% | %.2f | %.0f%% |" % (tot[0], 100 * tot[0] / max(t_idle, 1e-9), tot[1], 100 * tot[1] / wall))

    # large copies by issuing code: the main-thread call stack nearest to the copy
    mi = np.nonzero(ismain & (S.depth > 1))[0]
    mt = S.t[mi]
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    tot_gb = big_gb = 0.0
    for kind, (cs, ce, cb) in P.copies.items():
        if not kind.startswith("memcpy"):
            continue
        sel = inside(region, 0.5 * (cs + ce))
        tot_gb += cb[sel].sum() / 1e9
        for s, e, b in zip(cs[sel], ce[sel], cb[sel]):
            if b < a.copy_mb * 1e6:
                continue
            big_gb += b / 1e9
            lo, hi = np.searchsorted(mt, s - 0.01), np.searchsorted(mt, e + 0.01)
            if hi <= lo:
                key = "(no call stack within 10 ms)"
            else:
                j = lo + int(np.argmin(np.abs(mt[lo:hi] - 0.5 * (s + e))))
                p = S.path(mi[j])
                own = [f for f in p if not f.startswith("[") and not MEMFRAME.search(f)]
                hs = hot_of(p).split(" | ")[0]
                key = "%s: %s" % (hs, " < ".join(own[::-1][:3]) if own else "?")
            x = acc[(kind.replace("memcpy ", ""), int(round(b / 1e6)), key)]
            x[0] += 1
            x[1] += b / 1e9
            x[2] += e - s
    print("\n## %s: copies of at least %g MB, by the code the main thread was in" % (name, a.copy_mb))
    print("%.1f GB copied in the region, %.1f GB of it in copies this large. Innermost function first." % (tot_gb, big_gb))
    print("| direction | MB each | copies | GB | copy s | host hot spot: call chain |\n|---|---|---|---|---|---|")
    for (kind, mb, key), (n, gb, sec) in sorted(acc.items(), key=lambda kv: -kv[1][1])[:a.ncopy]:
        print("| %s | %d | %d | %.2f | %.2f | %s |" % (kind, mb, n, gb, sec, key))
    byhot = defaultdict(float)
    for (kind, mb, key), (n, gb, sec) in acc.items():
        byhot[key.split(":")[0]] += gb
    print("by hot spot: " + "; ".join("%s %.1f GB" % kv for kv in sorted(byhot.items(), key=lambda kv: -kv[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy-mb", type=float, default=30.0)
    ap.add_argument("--ncopy", type=int, default=30)
    ap.add_argument("--detail-only", action="store_true", help="print only the hot-spot and copy tables")
    ap.add_argument("report")
    ap.add_argument("log", nargs="?")
    ap.add_argument("--whole-window", action="store_true", help="no log: the whole collection window is the one region")
    ap.add_argument("--offset", type=float, help="profile time = log time + OFFSET (skips the fit)")
    ap.add_argument("--min-gap-ms", type=float, default=1.0)
    ap.add_argument("--long-only", action="store_true", help="use only the idle gaps of at least --min-gap-ms")
    ap.add_argument("--label", help="regex: also report every force-call piece whose name matches")
    ap.add_argument("--ncat", type=int, default=4, help="force-call pieces reported, by idle time")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--min-frac", type=float, default=0.01)
    ap.add_argument("--tree-frac", type=float, default=0.03)
    ap.add_argument("--tree-depth", type=int, default=40)
    a = ap.parse_args()

    P = Profile(a.report)
    S = Samples(P)
    win = P.window
    winv = np.array([win], dtype=float)
    gaps = subtract(winv, P.busy)
    long_gaps = gaps[(gaps[:, 1] - gaps[:, 0]) >= a.min_gap_ms / 1e3]
    on = Cover(S.oncpu_of(S.main)).total()
    nmain = int((S.tid == S.main).sum())
    a.sps = on / max(nmain, 1)
    print("# nsys host idle\nreport %s\nwindow %.1f s; %d CPU samples on %d threads, %d with a call stack; program module %s"
          % (a.report, win[1] - win[0], len(S.t), len(S.pid), int((S.depth > 1).sum()),
             S.modname(S.binmod) if S.binmod is not None else "NOT FOUND"))
    print("main thread %s: %d samples, on the CPU %.1f s of the window: %.3f ms of CPU per sample"
          % (S.label(S.main), nmain, on, 1e3 * a.sps))
    kern = [m for m in set(S.cc[:, 3]) if "kernel" in S.modname(m)]
    nk = int(np.isin(S.cc[S.first[S.first >= 0], 3], kern).sum()) if kern else 0
    print("samples taken in the kernel: %d (%.1f%%)" % (nk, 100 * nk / max(len(S.t), 1)))

    if a.whole_window:
        report_region(S, "whole window", winv, long_gaps, a, True)
        return
    if not a.log:
        raise SystemExit("give the hmc.log, or --whole-window")
    W = grid_windows(parse_grid_log(a.log))
    search = (P.launch - 2.0, P.launch + 20.0) if P.launch is not None else (-20.0, 40.0)
    d = a.offset if a.offset is not None else fit_offset(P, W, *search)
    cats = grid_categories(W, d, win)
    force = {k: iv for k, iv in cats.items() if k.startswith("force:")}
    ncall = sum(1 for x0, x1, _ in W["force"] if win[0] <= x0 + d and x1 + d <= win[1])
    print("offset %.3f s; %d force calls inside the window" % (d, ncall))

    def union(pred):
        return merge(np.vstack([iv for k, iv in force.items() if pred(k)] + [np.zeros((0, 2))]))
    detail(S, "ALL force calls, outside solves", union(lambda k: True), a)
    detail(S, "ALL force calls, before the first solve", union(lambda k: k.endswith(PHASE_SEP + "pre")), a)
    detail(S, "ALL force calls, after the last solve", union(lambda k: k.endswith(PHASE_SEP + "post")), a)
    detail(S, "solves", cats["solve"], a)
    idle_of = {k: Cover(subtract(iv, P.busy)).total() for k, iv in force.items()}
    picked = sorted(idle_of, key=lambda k: -idle_of[k])[:a.ncat]
    if a.label:
        picked += [k for k in force if re.search(a.label, k) and k not in picked]
    for k in picked:
        detail(S, k, force[k], a)
    if a.detail_only:
        return
    report_region(S, "ALL force calls, outside solves", union(lambda k: True), long_gaps, a, True)
    for ph, txt in (("pre", "before the first solve"), ("post", "after the last solve"),
                    ("mid", "between solves"), ("whole", "calls with no solve")):
        report_region(S, "ALL force calls, %s" % txt, union(lambda k, ph=ph: k.endswith(PHASE_SEP + ph)),
                      long_gaps, a, ph in ("pre", "post"))
    for k in picked:
        report_region(S, k, force[k], long_gaps, a, True)
    report_region(S, "solves (for comparison)", cats["solve"], long_gaps, a, False)


if __name__ == "__main__":
    main()
