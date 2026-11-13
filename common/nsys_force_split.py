#!/usr/bin/env python3
"""nsys_force_split.py -- where the MD time outside the solves goes, from a rank-0 Nsight Systems
report (grid-perf-profile step b, third layer, 2026-11-10).

The fork build has no NVTX, so attribution is by time window. Grid mode takes the windows from
the run's hmc.log (force calls "[i][j] P update elapsed time", solves, MG builds, the batched
smearing pullback), shifts them onto the profile clock by a fitted offset, and splits each
category into kernels (by name), copies, MPI and GPU idle. Chroma mode has no usable log
windows: a QUDA solve is a run of quda:: kernels, gaps under --bridge seconds bridged.

  uv run python common/nsys_force_split.py grid   <report.sqlite> <hmc.log> [--offset S] [--ref hmc.log]
  uv run python common/nsys_force_split.py chroma <report.sqlite> [--t0 S --t1 S] [--bridge S]
  uv run python common/nsys_force_split.py log    <hmc.log>       (no profile: per-call solve vs rest)

The .sqlite comes from `nsys export --type sqlite --output X.sqlite X.nsys-rep` (on $PSCRATCH).

Categories (Grid), each instant assigned once, in this priority:
  mg_build  -- "[GridMG ...] build #k ... total T s"
  solve     -- MG, CG, mixed-precision CG and multishift solves (log intervals, unioned)
  pullback  -- "[i] batched smeared P update" (HASEN_GRID_BATCH_SMEAR)
  smearing  -- "Smearing in T ms" (update_U)
  force:<[i][j] action>  -- the rest of each force call
  md_other  -- the rest of the MD
  non_md    -- refresh, initial and final action
Offset: profile time = log time + offset. The fit needs no kernel names: kernels that only
run inside solves or only outside them make the right offset the one where each kernel name
falls cleanly on one side (score = sum over names of T_n |2 f_n - 1|).
Timestamps: Grid misprints the microseconds (L198), corrected with grid_log_time.StampFixer.
Profiler ratios between codes are not comparable (L128); this splits time within one code.
"""
import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "5_studies/hasenbusch_tune/data_extraction_scripts"))
from grid_log_time import StampFixer  # noqa: E402

NUM = r"([-+0-9.eE]+)"
STAMP = re.compile(r"Grid : \w+ : (\d+\.\d+) s : ")
# Lines that keep a multishift block open (its cleanup solves print all of these; "rsd norm" too).
SOLVERISH = re.compile(r"ConjugateGradient|ConjugageGradient|MixedPrecision|MultiShift|Solver Elapsed|rsd norm")
RX = {
    "mg_solve": re.compile(r"\[GridMG ([^\]]+)\] solve .*?seconds " + NUM),
    "mg_build": re.compile(r"\[GridMG ([^\]]+)\] build #(\d+).*total " + NUM + " s"),
    "cg": re.compile(r"Solver Elapsed\s+" + NUM + " s"),
    "mpcg": re.compile(r"MixedPrecisionConjugateGradient: Total time " + NUM + " s"),
    "ms_start": re.compile(r"ConjugateGradientMultiShiftMixedPrec: shift 0 target resid"),
    "force": re.compile(r"\[(\d+)\]\[(\d+)\] P update elapsed time: " + NUM + r" ms \(force: " + NUM + " ms\\)"),
    "action": re.compile(r"\[(\d+)\]\[(\d+)\] dt\s+: \S+ (.+)$"),
    "pullback": re.compile(r"\[(\d+)\] batched smeared P update elapsed time: " + NUM + r" ms \(pullback\+Ta: " + NUM),
    "smearing": re.compile(r"^Smearing in " + NUM + " ms"),
    "md_start": re.compile(r"Molecular Dynamics evolution"),
    "md_end": re.compile(r"Compute final action"),
    "traj_end": re.compile(r"Total H after trajectory"),
}


# ----------------------------------------------------------------------------- interval sets
def merge(iv):
    iv = np.asarray(iv, dtype=float).reshape(-1, 2)
    iv = iv[iv[:, 1] > iv[:, 0]]
    if len(iv) == 0:
        return iv
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    ends = np.maximum.accumulate(iv[:, 1])
    new = np.ones(len(iv), bool)
    new[1:] = iv[1:, 0] > ends[:-1]
    first = np.nonzero(new)[0]
    last = np.r_[first[1:] - 1, len(iv) - 1]
    return np.column_stack([iv[first, 0], ends[last]])


def subtract(A, B):
    A, B = merge(A), merge(B)
    out, j = [], 0
    for s, e in A:
        cur = s
        while j < len(B) and B[j, 1] <= cur:
            j += 1
        k = j
        while k < len(B) and B[k, 0] < e:
            if B[k, 0] > cur:
                out.append((cur, B[k, 0]))
            cur = max(cur, B[k, 1])
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return np.array(out, dtype=float).reshape(-1, 2)


def intersect(A, B):
    return subtract(A, subtract(A, B))


class Cover:
    """A union of intervals; overlap(s, e) is the measure of [s, e] inside it, vectorised."""

    def __init__(self, iv):
        iv = merge(iv)
        self.iv = iv
        self.a, self.b = iv[:, 0], iv[:, 1]
        self.c = np.concatenate([[0.0], np.cumsum(self.b - self.a)])

    def total(self):
        return float(self.c[-1])

    def M(self, x):
        x = np.asarray(x, dtype=float)
        if len(self.a) == 0:
            return np.zeros_like(x)
        k = np.searchsorted(self.a, x, side="right") - 1
        kk = np.clip(k, 0, None)
        v = self.c[kk] + np.clip(x - self.a[kk], 0.0, self.b[kk] - self.a[kk])
        return np.where(k < 0, 0.0, v)

    def overlap(self, s, e):
        return self.M(e) - self.M(s)


# ----------------------------------------------------------------------------- Grid log
def parse_grid_log(path):
    """Segments (time, text) with corrected stamps, in file order."""
    fx, segs = StampFixer(), []
    with open(path, errors="replace") as fh:
        for line in fh:
            ms = list(STAMP.finditer(line))
            for i, m in enumerate(ms):
                end = ms[i + 1].start() if i + 1 < len(ms) else len(line)
                h = {}
                fx.want(h, "t", m.group(1))
                fx.feed(m.group(1))
                segs.append((h, line[m.end():end].strip()))
    fx.finish()
    return [(h["t"], txt) for h, txt in segs]


def grid_windows(segs):
    W = defaultdict(list)          # kind -> list of (t0, t1, label)
    action = {}
    ms_open = None
    md_t0 = None
    for t, txt in segs:
        if ms_open is not None:
            if SOLVERISH.search(txt):
                ms_open[1] = t
            else:
                W["solve"].append((ms_open[0], ms_open[1], "multishift"))
                ms_open = None
        if RX["ms_start"].search(txt) and ms_open is None:
            ms_open = [t, t]
            continue
        m = RX["mg_solve"].search(txt)
        if m:
            W["solve"].append((t - float(m.group(2)), t, "MG " + m.group(1)))
            continue
        m = RX["mg_build"].search(txt)
        if m:
            W["mg_build"].append((t - float(m.group(3)), t, "build " + m.group(1)))
            continue
        m = RX["cg"].search(txt)
        if m:
            W["solve"].append((t - float(m.group(1)), t, "CG"))
            continue
        m = RX["mpcg"].search(txt)
        if m:
            W["solve"].append((t - float(m.group(1)), t, "MPCG"))
            continue
        m = RX["action"].search(txt)
        if m:
            action[(m.group(1), m.group(2))] = m.group(3).strip()
            continue
        m = RX["force"].search(txt)
        if m:
            key = (m.group(1), m.group(2))
            lab = "[%s][%s] %s" % (key[0], key[1], action.get(key, "?"))
            W["force"].append((t - float(m.group(3)) / 1e3, t, lab))
            continue
        m = RX["pullback"].search(txt)
        if m:
            W["pullback"].append((t - float(m.group(2)) / 1e3, t, "[%s] smear pullback" % m.group(1)))
            continue
        m = RX["smearing"].search(txt)
        if m:
            W["smearing"].append((t - float(m.group(1)) / 1e3, t, "smearing"))
            continue
        if RX["md_start"].search(txt):
            md_t0 = t
            continue
        if RX["md_end"].search(txt) and md_t0 is not None:
            W["md"].append((md_t0, t, "md"))
            md_t0 = None
            continue
        if RX["traj_end"].search(txt):
            W["traj_end"].append((t, t, "traj_end"))
    if md_t0 is not None:  # log ends inside an MD (killed run)
        W["md"].append((md_t0, segs[-1][0], "md (open)"))
    return W


def iv_of(W, kind, shift=0.0):
    return np.array([(a + shift, b + shift) for a, b, _ in W[kind]], dtype=float).reshape(-1, 2)


PHASE_SEP = " · "


def grid_categories(W, shift, window):
    """Disjoint category -> interval array, in profile seconds, clipped to `window`."""
    win = np.array([window], dtype=float)
    md = intersect(iv_of(W, "md", shift), win)
    allwin = win
    taken = np.zeros((0, 2))
    cats = {}
    for kind in ("mg_build", "solve", "pullback", "smearing"):
        iv = intersect(subtract(iv_of(W, kind, shift), taken), allwin)
        cats[kind] = iv
        taken = merge(np.vstack([taken, iv]))
    # Each force call is cut at its first and last solve: pre (before the first solve), mid
    # (between solves), post (after the last one); "whole" when the call has no solve.
    sv = np.array(sorted((a, b) for a, b, _ in W["solve"] + W["mg_build"]), dtype=float).reshape(-1, 2)
    byl = defaultdict(list)
    for a, b, lab in W["force"]:
        inside = sv[(sv[:, 1] > a) & (sv[:, 0] < b)] if len(sv) else sv
        if len(inside) == 0:
            byl[lab + PHASE_SEP + "whole"].append((a + shift, b + shift))
            continue
        s0, e1 = max(inside[:, 0].min(), a), min(inside[:, 1].max(), b)
        byl[lab + PHASE_SEP + "pre"].append((a + shift, s0 + shift))
        byl[lab + PHASE_SEP + "mid"].append((s0 + shift, e1 + shift))
        byl[lab + PHASE_SEP + "post"].append((e1 + shift, b + shift))
    for lab in sorted(byl):
        iv = intersect(subtract(np.array(byl[lab]), taken), allwin)
        cats["force:" + lab] = iv
        taken = merge(np.vstack([taken, iv]))
    cats["md_other"] = subtract(md, taken)
    taken = merge(np.vstack([taken, cats["md_other"]]))
    cats["non_md"] = subtract(allwin, taken)
    return cats


# ----------------------------------------------------------------------------- profile
def _strip_groups(s, o, c):
    """Remove every balanced o...c group, innermost first."""
    pat = re.compile("%s[^%s%s]*%s" % (re.escape(o), re.escape(o), re.escape(c), re.escape(c)))
    for _ in range(80):
        s2 = pat.sub("", s)
        if s2 == s:
            break
        s = s2
    return s


def _limit_depth(s, keep):
    """Keep template arguments down to nesting depth `keep`, drop anything deeper."""
    out, d = [], 0
    for ch in s:
        if ch == "<":
            d += 1
            if d <= keep:
                out.append(ch)
        elif ch == ">":
            if d <= keep:
                out.append(ch)
            d = max(d - 1, 0)
        elif d <= keep:
            out.append(ch)
    return "".join(out)


def short_name(n):
    """A readable kernel name.
    Grid: every accelerator_for loop is 'void Grid::LambdaApply<[RET] ENCLOSING<..>(..)::[lambda(..) (instance N)]>(..)';
    the name is the enclosing function, the lambda number, and the precisions that occur in the
    template arguments: 'Grid::blockZAXPY λ1 [f]'. Template instantiations of one function merge.
    Others (QUDA, thrust, QDP-JIT): template arguments kept to depth 2, argument lists dropped."""
    full = n.strip()
    prec = ("f" if "complex<float>" in full else "") + ("d" if "complex<double>" in full else "")
    tag = " [%s]" % prec if prec else ""
    m = re.search(r"\bLambdaApply\w*<", full)
    if m:
        body = full[m.end():]
        # the lambda is written '::[lambda(..) (instance N)]' by CUPTI, '::{lambda(..)#N}' by c++filt
        k = min([i for i in (body.find("::[lambda"), body.find("::{lambda")) if i >= 0], default=-1)
        lam = ""
        if k >= 0:
            lam = " λ" + ".".join(x or y for x, y in re.findall(r"\(instance (\d+)\)\]|#(\d+)\}", body[k:]))
            body = body[:k]
        # an assignment kernel is named by its expression template: keep the operation names
        ops = []
        for op in re.findall(r"Grid::((?:Binary|Unary|Trinary)\w+)", body):
            if op not in ops:
                ops.append(op)
        body = _strip_groups(_strip_groups(body, "<", ">"), "(", ")")
        body = re.sub(r"operator\s+", "operator", body)
        body = re.sub(r"\s+const\s*$", "", body).strip()
        toks = body.split()
        name = toks[-1] if toks else "?"
        if "operator=" in name and ops:
            name += "{" + ",".join(ops[:5]) + "}"
        return (name + lam + tag)[:140]
    body = _strip_groups(_limit_depth(full, 2), "(", ")")
    return re.sub(r"^(void|auto) ", "", body).strip()[:140]


class Profile:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.strings = {}
        self.window = self._window()
        self.launch = self._launch()
        self._load()

    def _str(self, i):
        if i not in self.strings:
            r = self.con.execute("SELECT value FROM StringIds WHERE id=?", (int(i),)).fetchone()
            self.strings[i] = r[0] if r else "?%d" % i
        return self.strings[i]

    def _window(self):
        """Event time stamps count from the start of collection, so the window is (0, duration).
        ANALYSIS_DETAILS.startTime is on another clock (202.5 s for --delay=200) and is not used."""
        r = self.con.execute("SELECT duration FROM ANALYSIS_DETAILS").fetchone()
        return (0.0, r[0] / 1e9)

    def _launch(self):
        """Event-clock time at which nsys launched the program (about -delay), or None."""
        if "DIAGNOSTIC_EVENT" not in self.tables:
            return None
        r = self.con.execute("SELECT timestamp FROM DIAGNOSTIC_EVENT WHERE text LIKE 'Process was launched%' "
                             "ORDER BY timestamp LIMIT 1").fetchone()
        return r[0] / 1e9 if r else None

    def _cols(self, t):
        return [r[1] for r in self.con.execute("PRAGMA table_info(%s)" % t)]

    def _fetch(self, query, ncol):
        """A query result as an int64 array, read in chunks (a Chroma report has ~1e8 rows)."""
        parts, cur = [], self.con.execute(query)
        while True:
            rows = cur.fetchmany(2_000_000)
            if not rows:
                break
            parts.append(np.array(rows, dtype=np.int64).reshape(-1, ncol))
        return np.vstack(parts) if parts else np.zeros((0, ncol), dtype=np.int64)

    def _load(self):
        k = "CUPTI_ACTIVITY_KIND_KERNEL"
        if k in self.tables:
            namecol = "demangledName" if "demangledName" in self._cols(k) else "shortName"
            a = self._fetch("SELECT start, end, %s FROM %s" % (namecol, k), 3)
        else:
            # Either no kernel ran in the window (program start-up), or the profiler lost its
            # kernel records (2026-11-13, allocation 61232844: one report of four had copies
            # and API calls but no kernel table, another had no CUDA data at all).
            print("WARNING: no GPU kernel records in this report; kernel, busy and idle figures below are not valid")
            a = np.zeros((0, 3), dtype=np.int64)
        self.k_s, self.k_e = a[:, 0] / 1e9, a[:, 1] / 1e9
        # raw_idx: one index per demangled name (used by the offset fit, which wants the finest
        # split); k_idx: one index per short name (template instantiations merged), for reporting.
        raw_ids, self.raw_idx = np.unique(a[:, 2], return_inverse=True)
        self.n_raw = len(raw_ids)
        self.raw_names = [self._str(i) for i in raw_ids]
        uniq, inv = np.unique(np.array([short_name(s) for s in self.raw_names]), return_inverse=True)
        self.k_idx = inv[self.raw_idx]
        self.k_names = list(uniq)
        self.k_ids = uniq
        self.copies = {}
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in self.tables:
            c = self._fetch("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY", 4)
            kinds = dict(self.con.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER").fetchall()) \
                if "ENUM_CUDA_MEMCPY_OPER" in self.tables else {}
            for ck in np.unique(c[:, 3]):
                sel = c[:, 3] == ck
                self.copies["memcpy " + str(kinds.get(int(ck), ck))] = (c[sel, 0] / 1e9, c[sel, 1] / 1e9, c[sel, 2])
        if "CUPTI_ACTIVITY_KIND_MEMSET" in self.tables:
            c = np.array(self.con.execute(
                "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET").fetchall(), dtype=np.int64).reshape(-1, 3)
            self.copies["memset"] = (c[:, 0] / 1e9, c[:, 1] / 1e9, c[:, 2])
        busy = [np.column_stack([self.k_s, self.k_e])] + [np.column_stack([s, e]) for s, e, _ in self.copies.values()]
        self.busy = merge(np.vstack(busy))
        # host-side: CUDA runtime (minus the launch calls) and MPI, by name
        self.host = {}
        rt = "CUPTI_ACTIVITY_KIND_RUNTIME"
        if rt in self.tables:
            skip = {i for i, v in self.con.execute(
                "SELECT id, value FROM StringIds WHERE value LIKE 'cudaLaunchKernel%' OR value LIKE 'cuLaunchKernel%'")}
            acc = defaultdict(list)
            cur = self.con.execute("SELECT start, end, nameId FROM %s" % rt)
            while True:
                rows = cur.fetchmany(1_000_000)
                if not rows:
                    break
                r = np.array(rows, dtype=np.int64)
                r = r[~np.isin(r[:, 2], list(skip))] if skip else r
                for nid in np.unique(r[:, 2]):
                    acc[int(nid)].append(r[r[:, 2] == nid, :2])
            for nid, parts in acc.items():
                x = np.vstack(parts)
                self.host["cuda " + self._str(nid)] = (x[:, 0] / 1e9, x[:, 1] / 1e9)
        for t in sorted(self.tables):
            if not t.startswith("MPI_"):
                continue
            cols = self._cols(t)
            nc = "textId" if "textId" in cols else ("nameId" if "nameId" in cols else None)
            if not ({"start", "end"} <= set(cols)) or nc is None:
                continue
            x = np.array(self.con.execute("SELECT start, end, %s FROM %s" % (nc, t)).fetchall(), dtype=np.int64).reshape(-1, 3)
            for nid in np.unique(x[:, 2]):
                y = x[x[:, 2] == nid]
                key = "mpi " + self._str(nid)
                if key in self.host:
                    s0, e0 = self.host[key]
                    self.host[key] = (np.r_[s0, y[:, 0] / 1e9], np.r_[e0, y[:, 1] / 1e9])
                else:
                    self.host[key] = (y[:, 0] / 1e9, y[:, 1] / 1e9)


GAP_EDGES = [0, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
GAP_LABELS = ["<10us", "10-100us", "0.1-1ms", "1-10ms", ">10ms"]


def census(P, iv, top=20):
    C = Cover(iv)
    wall = C.total()
    out = {"wall": wall}
    if wall == 0:
        return out
    ov = C.overlap(P.k_s, P.k_e)
    out["kern_sum"] = float(ov.sum())
    out["n_kern"] = int((ov > 0).sum())
    by = np.bincount(P.k_idx, weights=ov, minlength=len(P.k_ids))
    cnt = np.bincount(P.k_idx, weights=(ov > 0).astype(float), minlength=len(P.k_ids))
    order = np.argsort(-by)[:top]
    out["top"] = [(P.k_names[i], by[i], int(cnt[i])) for i in order if by[i] > 0]
    out["busy"] = float(C.overlap(P.busy[:, 0], P.busy[:, 1]).sum())
    out["copies"] = {}
    for k, (s, e, nb) in P.copies.items():
        o = C.overlap(s, e)
        d = np.maximum(e - s, 1e-12)
        hit = o > 0
        sizes, inv = np.unique(nb[hit], return_inverse=True)
        tot = np.bincount(inv, weights=nb[hit].astype(float)) if hit.any() else np.zeros(0)
        num = np.bincount(inv) if hit.any() else np.zeros(0, int)
        big = [(float(sizes[i]) / 1e6, int(num[i])) for i in np.argsort(-tot)[:4]]
        out["copies"][k] = (float(o.sum()), float((nb * o / d).sum()), int(hit.sum()), big)
    gs, ge = P.busy[:-1, 1], P.busy[1:, 0]
    go = C.overlap(gs, ge)
    glen = ge - gs
    b = np.digitize(glen, GAP_EDGES[1:-1])
    out["gaps"] = [(GAP_LABELS[i], float(go[b == i].sum()), int((go[b == i] > 0).sum())) for i in range(len(GAP_LABELS))]
    host = []
    for k, (s, e) in P.host.items():
        o = C.overlap(s, e)
        if o.sum() > 0:
            host.append((k, float(o.sum()), int((o > 0).sum())))
    out["host"] = sorted(host, key=lambda x: -x[1])[:12]
    return out


def copy_context(P, iv, min_mb=30.0, top=25):
    """Large host<->device copies inside `iv`, grouped by direction, size and the kernel that runs
    next. For an upload the next kernel is the first user of the field; for a download a long
    wait before the next kernel is host work on it. No call stacks are recorded, so this is the
    only handle on which code moves the data."""
    C = Cover(iv)
    if C.total() == 0:
        return
    order = np.argsort(P.k_s, kind="stable")
    ks, ke, kn = P.k_s[order], P.k_e[order], P.k_idx[order]
    rows = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for kind, (s, e, nb) in P.copies.items():
        if "Host" not in kind:
            continue
        sel = (nb >= min_mb * 1e6) & (C.overlap(s, e) > 0)
        for cs, ce, b in zip(s[sel], e[sel], nb[sel]):
            j = np.searchsorted(ks, ce, side="left")
            nxt = P.k_names[kn[j]] if j < len(ks) else "(none)"
            wait = ks[j] - ce if j < len(ks) else 0.0
            i = np.searchsorted(ks, cs, side="right") - 1
            since = cs - ke[i] if i >= 0 else 0.0
            r = rows[(kind.replace("memcpy ", ""), round(b / 1e6, 1), nxt)]
            r[0] += 1
            r[1] += b
            r[2] += wait
            r[3] += max(since, 0.0)
    if not rows:
        return
    print("\ncopies of at least %.0f MB, by direction, size and the next kernel "
          "(wait = copy end to that kernel's start, lead = previous kernel's end to copy start):" % min_mb)
    print("| direction | MB | n | GB | mean wait ms | mean lead ms | next kernel |\n|---|---|---|---|---|---|---|")
    for (kind, mb, nxt), (n, b, w, sn) in sorted(rows.items(), key=lambda x: -x[1][1])[:top]:
        print("| %s | %.1f | %d | %.2f | %.1f | %.1f | %s |" % (kind, mb, n, b / 1e9, 1e3 * w / n, 1e3 * sn / n,
                                                               nxt.replace("|", "/")))


def print_census(name, c, ncalls=None):
    w = c["wall"]
    if w == 0:
        return
    per = " (%d calls, %.3f s per call)" % (ncalls, w / ncalls) if ncalls else ""
    print("\n### %s: %.2f s wall%s" % (name, w, per))
    idle = w - c["busy"]
    print("GPU busy %.2f s (%.1f%%), idle %.2f s (%.1f%%); kernel time %.2f s in %d kernels (mean %.1f us)" % (
        c["busy"], 100 * c["busy"] / w, idle, 100 * idle / w, c["kern_sum"], c["n_kern"],
        1e6 * c["kern_sum"] / max(c["n_kern"], 1)))
    print("idle gaps by length: " + ", ".join("%s %.2f s (%d)" % g for g in c["gaps"] if g[1] > 0))
    cp = [(k, v) for k, v in c["copies"].items() if v[0] > 0]
    if cp:
        print("copies (time, bytes, count; largest sizes as MB x count): " + "; ".join(
            "%s %.3f s %.2f GB (%d; %s)" % (k, v[0], v[1] / 1e9, v[2],
                                           ", ".join("%.1fx%d" % b for b in v[3])) for k, v in cp))
    print("\n| kernel | s | % of wall | calls | mean us |\n|---|---|---|---|---|")
    for n, t, k in c["top"]:
        print("| %s | %.3f | %.1f | %d | %.1f |" % (n.replace("|", "/"), t, 100 * t / w, k, 1e6 * t / max(k, 1)))
    if c["host"]:
        print("\nhost (rank 0, overlaps GPU work): " + ", ".join("%s %.3f s (%d)" % h for h in c["host"]))


# ----------------------------------------------------------------------------- offset fit
def fit_offset(P, W, lo, hi):
    solve = merge(np.vstack([iv_of(W, "solve"), iv_of(W, "mg_build")]))
    md = merge(iv_of(W, "md"))
    if len(solve) == 0:
        raise SystemExit("no solves in the log")
    def score(d, sub):
        C = Cover(solve + d)
        Mx = Cover(md + d)
        s, e, ix = P.k_s[sub], P.k_e[sub], P.raw_idx[sub]
        inmd = Mx.overlap(s, e)
        ins = C.overlap(s, e)
        Tn = np.bincount(ix, weights=inmd, minlength=P.n_raw)
        In = np.bincount(ix, weights=np.minimum(ins, inmd), minlength=P.n_raw)
        ok = Tn > 0
        return float((Tn[ok] * np.abs(2 * In[ok] / Tn[ok] - 1)).sum()) / max(Tn.sum(), 1e-12)

    rng = np.random.default_rng(0)
    sub = rng.random(len(P.k_s)) < min(1.0, 2e5 / max(len(P.k_s), 1))
    grid = np.arange(lo, hi, 0.05)
    sc = np.array([score(d, sub) for d in grid])
    best = grid[np.argmax(sc)]
    fine = np.arange(best - 0.1, best + 0.1, 0.001)
    allk = np.ones(len(P.k_s), bool)
    scf = np.array([score(d, allk) for d in fine])
    d = float(fine[np.argmax(scf)])
    top = np.argsort(-sc)[:5]
    print("offset fit: best %.3f s, score %.4f (1 = every kernel name entirely on one side); "
          "coarse runners-up %s" % (d, scf.max(), ", ".join("%.2f:%.3f" % (grid[i], sc[i]) for i in top)))
    return d


# ----------------------------------------------------------------------------- log only
def log_split(W):
    """Per force-call label, per trajectory: wall, solve, MG build, rest (log clock only)."""
    solve = Cover(iv_of(W, "solve"))
    build = Cover(iv_of(W, "mg_build"))
    ends = [x[0] for x in W["traj_end"]]
    md = W["md"]
    print("\n## Log-only split of the force calls (per trajectory; solve = union of logged solve intervals)")
    print("rest = wall - solve - MG build; per call it splits into before the first solve, between solves, after the last.")
    print("| call | calls/traj | wall s/traj | solve s/traj | MG build s/traj | rest s/traj | rest s/call | before | between | after |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    ntraj = max(len(ends), 1)
    d = defaultdict(lambda: np.zeros(7))
    busy = merge(np.vstack([solve.iv, build.iv]))
    for a, b, lab in W["force"] + W["pullback"] + W["smearing"]:
        x = d[lab]
        inside = intersect(busy, np.array([[a, b]]))
        sb = float(build.overlap(a, b))
        rest = (b - a) - Cover(inside).total()
        pre = (inside[0, 0] - a) if len(inside) else rest
        post = (b - inside[-1, 1]) if len(inside) else 0.0
        x += [1, b - a, Cover(inside).total() - sb, sb, pre, rest - pre - post, post]
    tot = np.zeros(4)
    for lab in sorted(d):
        n, w, s, bl, pre, mid, post = d[lab]
        r = w - s - bl
        tot += [w, s, bl, r]
        print("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.3f | %.3f | %.3f | %.3f |" % (
            lab, n / ntraj, w / ntraj, s / ntraj, bl / ntraj, r / ntraj, r / n, pre / n, mid / n, post / n))
    print("| **all** | | %.1f | %.1f | %.1f | **%.1f** | | | | |" % tuple(v / ntraj for v in tot))
    mdw = sum(b - a for a, b, _ in md) / ntraj
    print("\nMD wall %.1f s/traj over %d trajectories; force + pullback + smearing windows cover %.1f s, "
          "the rest of the MD (update_U, logging, gaps) %.1f s" % (mdw, ntraj, tot[0] / ntraj, mdw - tot[0] / ntraj))


# ----------------------------------------------------------------------------- main
def overhead_table(W, window, ref):
    """Force-call wall per label (MG build time taken out) inside vs outside the profiled window
    (log clock), and in the reference run."""
    def per_label(Wx, sel):
        d = defaultdict(list)
        bc = Cover(iv_of(Wx, "mg_build"))
        for a, b, lab in Wx["force"] + Wx["pullback"]:
            if sel(a, b):
                d[lab].append(b - a - float(bc.overlap(a, b)))
        return d
    inside = per_label(W, lambda a, b: a >= window[0] and b <= window[1])
    outside = per_label(W, lambda a, b: b < window[0] or a > window[1])
    refd = per_label(ref, lambda a, b: True) if ref else {}
    print("\n## Force-call wall per call: profiled vs unprofiled (same run)%s" % (" vs reference run" if ref else ""))
    print("| call | profiled n | s/call | unprofiled n | s/call | ratio |%s\n|---|---|---|---|---|---|%s" % (
        " ref n | ref s/call |" if ref else "", "---|---|" if ref else ""))
    for lab in sorted(set(inside) | set(outside)):
        i, o = inside.get(lab, []), outside.get(lab, [])
        mi, mo = (np.mean(i) if i else np.nan), (np.mean(o) if o else np.nan)
        row = "| %s | %d | %.3f | %d | %.3f | %.3f |" % (lab, len(i), mi, len(o), mo, mi / mo)
        if ref:
            r = refd.get(lab, [])
            row += " %d | %.3f |" % (len(r), np.mean(r) if r else np.nan)
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["grid", "chroma", "log"])
    ap.add_argument("report", help="report .sqlite (log mode: the hmc.log)")
    ap.add_argument("log", nargs="?")
    ap.add_argument("--offset", type=float, help="profile time = log time + OFFSET (skips the fit)")
    ap.add_argument("--search", type=float, nargs=2, metavar=("LO", "HI"),
                    help="offset search range; default: 2 s before to 20 s after the program launch "
                         "recorded in the report (Grid's log clock starts shortly after launch)")
    ap.add_argument("--ref", help="a reference hmc.log for the per-call comparison (Grid)")
    ap.add_argument("--first-traj-only", action="store_true", help="reference: first trajectory only")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--bridge", type=float, default=0.005)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-frac", type=float, default=0.005,
                    help="print the detailed census only for categories of at least this fraction of the window")
    a = ap.parse_args()

    if a.mode == "log":
        log_split(grid_windows(parse_grid_log(a.report)))
        return
    P = Profile(a.report)
    win = (a.t0 if a.t0 is not None else P.window[0], a.t1 if a.t1 is not None else P.window[1])
    print("# nsys force split: %s\nreport %s\ncollection window %.1f-%.1f s (%.1f s); %d kernels, %d names" % (
        a.mode, a.report, win[0], win[1], win[1] - win[0], len(P.k_s), len(P.k_ids)))

    if a.mode == "grid":
        if not a.log:
            raise SystemExit("grid mode needs the hmc.log")
        W = grid_windows(parse_grid_log(a.log))
        print("log: %d force calls, %d solves, %d MG builds, %d pullbacks, %d MD segments" % (
            len(W["force"]), len(W["solve"]), len(W["mg_build"]), len(W["pullback"]), len(W["md"])))
        search = a.search
        if search is None:
            search = (P.launch - 2.0, P.launch + 20.0) if P.launch is not None else (-20.0, 40.0)
        print("program launch at %s s on the profile clock; offset search %.1f .. %.1f s" % (
            "%.3f" % P.launch if P.launch is not None else "unknown", search[0], search[1]))
        d = a.offset if a.offset is not None else fit_offset(P, W, *search)
        cats = grid_categories(W, d, win)
        ncalls = defaultdict(int)
        for x0, x1, lab in W["force"]:
            if win[0] <= x0 + d and x1 + d <= win[1]:
                ncalls["force:" + lab] += 1
        # aggregates over the force-call labels: everything outside solves, and each phase
        def union(pred):
            return merge(np.vstack([iv for k, iv in cats.items() if k.startswith("force:") and pred(k)]
                                   + [np.zeros((0, 2))]))
        agg = {"ALL force calls, outside solves": union(lambda k: True)}
        for ph in ("pre", "mid", "post", "whole"):
            agg["ALL force calls" + PHASE_SEP + ph] = union(lambda k, ph=ph: k.endswith(PHASE_SEP + ph))
        print("\n## Category totals inside the collection window (profile clock, log + %.3f s)" % d)
        print("force calls are cut at their solves: pre = before the first solve, mid = between solves, "
              "post = after the last, whole = a call with no solve")
        print("| category | calls | wall s | s per call | GPU busy % | kernel s | host-to-device GB | "
              "device-to-host GB | idle s | of which gaps >10 ms s |\n|---|---|---|---|---|---|---|---|---|---|")
        res = {}
        for k, iv in list(agg.items()) + list(cats.items()):
            res[k] = census(P, iv, a.top)
            c = res[k]
            if c["wall"] > 0:
                n = ncalls.get(k.split(PHASE_SEP)[0])
                cp = c["copies"]
                print("| %s | %s | %.2f | %s | %.1f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
                    k, n if n else "", c["wall"], "%.3f" % (c["wall"] / n) if n else "",
                    100 * c["busy"] / c["wall"], c["kern_sum"],
                    cp.get("memcpy Host-to-Device", (0, 0))[1] / 1e9, cp.get("memcpy Device-to-Host", (0, 0))[1] / 1e9,
                    c["wall"] - c["busy"], c["gaps"][-1][1]))
        wtot = win[1] - win[0]
        for k in res:
            if res[k]["wall"] >= a.min_frac * wtot:
                print_census(k, res[k], ncalls.get(k.split(PHASE_SEP)[0]))
                if k in agg or k in ("pullback", "smearing", "solve"):
                    copy_context(P, agg[k] if k in agg else cats[k])
        ref = None
        if a.ref:
            ref = grid_windows(parse_grid_log(a.ref))
            if a.first_traj_only and ref["traj_end"]:
                t_end = ref["traj_end"][0][0]
                ref = {k: [x for x in v if x[1] <= t_end] for k, v in ref.items()}
        overhead_table(W, (win[0] - d, win[1] - d), ref)
    else:
        qi = np.array([("quda::" in n) for n in P.k_names])
        sel = qi[P.k_idx]
        q = merge(np.column_stack([P.k_s[sel], P.k_e[sel]]))
        if len(q):
            gaps = np.column_stack([q[:-1, 1], q[1:, 0]])
            q = merge(np.vstack([q, gaps[(gaps[:, 1] - gaps[:, 0]) < a.bridge]]))
        winv = np.array([win], dtype=float)
        cats = {"quda_solve (quda:: runs, bridged %.0f ms)" % (1e3 * a.bridge): intersect(q, winv),
                "outside quda runs": subtract(winv, q)}
        for k, iv in cats.items():
            print_census(k, census(P, iv, a.top))


if __name__ == "__main__":
    main()
