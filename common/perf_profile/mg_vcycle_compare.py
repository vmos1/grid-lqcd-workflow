#!/usr/bin/env python3
"""mg_vcycle_compare.py -- multigrid solves of two nsys reports, per V-cycle.

Written for grid-upstream-rebase gate 6 (2026-11-24, old base against new base); kept for
grid-perf-profile. Compares the multigrid solves of two nsys reports at equal work.

The solves are found from the profile itself, not from log time stamps: every V-cycle runs one
fine-level projection kernel (ProbeMG::blockProjectFused). Marker kernels less than --gap
seconds apart belong to one solve (or to two solves that follow each other directly); the
interval taken is first marker start to last marker end. Inside those intervals everything is
divided by the number of V-cycle periods (markers minus intervals), so the two reports compare
per V-cycle whatever number of solves fell into each profiling window.

Uses Profile / Cover / census of common/nsys_force_split.py (imported, not changed).
The .sqlite files come from `nsys export --type sqlite` of each report.

  uv run --project grid-lqcd-workflow python grid-lqcd-workflow/common/perf_profile/mg_vcycle_compare.py OLD.sqlite NEW.sqlite [--marker blockProjectFused] [--gap 0.5] [--top 45]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nsys_force_split as nfs  # noqa: E402


def stats(path, marker, gap):
    P = nfs.Profile(path)
    ids = [i for i, n in enumerate(P.k_names) if marker in n]
    sel = np.isin(P.k_idx, ids)
    ms, me = P.k_s[sel], P.k_e[sel]
    order = np.argsort(ms)
    ms, me = ms[order], me[order]
    if len(ms) < 2:
        raise SystemExit("%s: fewer than 2 marker kernels (%s)" % (path, marker))
    brk = np.nonzero(np.diff(ms) > gap)[0]
    first = np.r_[0, brk + 1]
    last = np.r_[brk, len(ms) - 1]
    iv = np.column_stack([ms[first], me[last]])
    C = nfs.Cover(iv)
    c = nfs.census(P, iv, top=100000)
    S = {
        "path": path,
        "markers": len(ms),
        "intervals": len(iv),
        "periods": len(ms) - len(iv),
        "wall": c["wall"],
        "busy": c["busy"],
        "kern_sum": c["kern_sum"],
        "n_kern": c["n_kern"],
        "gaps": {g[0]: (g[1], g[2]) for g in c["gaps"]},
        "kern": {n: (t, k) for n, t, k in c["top"]},
        "copies": {k: (v[0], v[2]) for k, v in c["copies"].items() if v[2] > 0},
        "copy_bytes": {k: v[1] for k, v in c["copies"].items() if v[2] > 0},
        "host": {},
    }
    for k, (s, e) in P.host.items():
        o = C.overlap(s, e)
        if (o > 0).any():
            S["host"][k] = (float(o.sum()), int((o > 0).sum()))
    # the kernel launch calls, which Profile leaves out of its host table
    rt = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if rt in P.tables:
        for nid, name in P.con.execute(
                "SELECT id, value FROM StringIds WHERE value LIKE 'cudaLaunchKernel%' OR value LIKE 'cuLaunchKernel%'"):
            rows = P.con.execute("SELECT start, end FROM %s WHERE nameId=?" % rt, (nid,)).fetchall()
            if not rows:
                continue
            x = np.array(rows, dtype=np.int64) / 1e9
            o = C.overlap(x[:, 0], x[:, 1])
            if (o > 0).any():
                S["host"]["cuda " + name] = (float(o.sum()), int((o > 0).sum()))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--marker", default="blockProjectFused")
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=45)
    a = ap.parse_args()
    A, B = stats(a.old, a.marker, a.gap), stats(a.new, a.marker, a.gap)
    na, nb = A["periods"], B["periods"]

    print("# multigrid solves, old against new, per V-cycle (marker %s, gap %.2f s)" % (a.marker, a.gap))
    for tag, S in (("old", A), ("new", B)):
        print("%s: %s\n     %d marker kernels in %d intervals, %d V-cycle periods, %.2f s wall" % (
            tag, S["path"], S["markers"], S["intervals"], S["periods"], S["wall"]))

    def row(label, x, y, unit="ms", scale=1e3):
        d = y - x
        pct = 100 * d / x if x else float("nan")
        print("| %s | %.3f | %.3f | %+.3f | %+.1f |" % (label + " (" + unit + ")", scale * x, scale * y, scale * d, pct))

    print("\n## Per V-cycle\n| quantity | old | new | diff | % |\n|---|---|---|---|---|")
    row("wall", A["wall"] / na, B["wall"] / nb)
    row("GPU busy", A["busy"] / na, B["busy"] / nb)
    row("GPU idle", (A["wall"] - A["busy"]) / na, (B["wall"] - B["busy"]) / nb)
    row("kernel time", A["kern_sum"] / na, B["kern_sum"] / nb)
    row("kernels", A["n_kern"] / na, B["n_kern"] / nb, unit="count", scale=1.0)

    print("\n## Idle gaps per V-cycle\n| gap length | old ms | new ms | diff ms | old count | new count |\n|---|---|---|---|---|---|")
    for lab in nfs.GAP_LABELS:
        ta, ca = A["gaps"].get(lab, (0.0, 0))
        tb, cb = B["gaps"].get(lab, (0.0, 0))
        print("| %s | %.3f | %.3f | %+.3f | %.1f | %.1f |" % (lab, 1e3 * ta / na, 1e3 * tb / nb,
                                                           1e3 * (tb / nb - ta / na), ca / na, cb / nb))

    def table(title, key, top):
        names = set(A[key]) | set(B[key])
        rows = []
        for n in names:
            ta, ca = A[key].get(n, (0.0, 0))
            tb, cb = B[key].get(n, (0.0, 0))
            rows.append((n, ca / na, cb / nb, 1e6 * ta / na, 1e6 * tb / nb))
        rows.sort(key=lambda r: -abs(r[4] - r[3]))
        print("\n## %s per V-cycle, largest time difference first\n"
              "| name | old calls | new calls | old us | new us | diff us | old us/call | new us/call |\n|---|---|---|---|---|---|---|---|" % title)
        tot_a = tot_b = 0.0
        for n, ca, cb, ua, ub in rows:
            tot_a += ua
            tot_b += ub
        for n, ca, cb, ua, ub in rows[:top]:
            print("| %s | %.2f | %.2f | %.1f | %.1f | %+.1f | %.2f | %.2f |" % (
                n.replace("|", "/"), ca, cb, ua, ub, ub - ua, ua / ca if ca else 0.0, ub / cb if cb else 0.0))
        print("| all %d names | | | %.1f | %.1f | %+.1f | | |" % (len(rows), tot_a, tot_b, tot_b - tot_a))

    table("Kernels", "kern", a.top)
    table("Copies", "copies", 20)
    print("\ncopied MB per V-cycle (old, new): " + ", ".join(
        "%s %.2f, %.2f" % (k, A["copy_bytes"].get(k, 0.0) / na / 1e6, B["copy_bytes"].get(k, 0.0) / nb / 1e6)
        for k in sorted(set(A["copy_bytes"]) | set(B["copy_bytes"]))))
    table("Host CUDA calls (rank 0; a synchronise overlaps GPU work)", "host", 25)


if __name__ == "__main__":
    main()
