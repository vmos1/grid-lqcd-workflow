#!/usr/bin/env python3
"""Cross-run comparison tables for N acceptance runs, each spanning several
checkpoint-resumed log WINDOWS (e.g. traj 2001-2010 + 2011-2020).

Why this and not `accept_run_tables.py` directly: that script renders ONE log
at a time and its `--csv` writer restarts the trajectory counter at 2001 for
every file, so a run made of a parent log plus a `_ext_2011_2020.log`
continuation cannot be expressed. This wrapper stitches a run's windows back
into one 20-trajectory chain, then emits the cross-run comparison tables that
a multi-candidate doc needs. It imports `accept_run_tables`' parser wholesale
(`parse`/`build_columns`/`buckets`) so every number still comes from the same,
already-validated log-parsing path -- nothing is reimplemented here.

"Warm" throughout = every trajectory EXCEPT the first of each window. The
first trajectory of window 1 is the cold start (one-time stencil/QUDA-autotune
init, ~+650 s in the gauge bucket); the first of each later window is a
checkpoint resume with cold solver caches. Both are excluded from timing
means and flagged, never silently averaged in.

Observable errors are NAIVE standard errors (sigma/sqrt(N)) over the full
Markov chain, rejected trajectories included as repeats of the previous
configuration -- which is what the ensemble average actually is. With ~20
correlated trajectories these UNDERSTATE the true error, so treat a small
z-score as solid evidence of agreement and a large one as needing a longer
chain before it means anything.

Usage:
  compare_runs_multiwindow.py --run 'LABEL=log1,log2' [--run ...] \
      [--ladder 'LABEL=m1,m2,...'] [--csv out.csv]

The FIRST --run given is the reference the z-scores are taken against.
"""
import os
import sys
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accept_run_tables import parse, build_columns, buckets, SMEARKEY

SECT_ORDER = ["s", "ldS", "ldL", "PF0", "PF1", "PF2", "PF3", "PF4", "tail", "gauge", "smear"]


def mean(v):
    return sum(v) / len(v)


def sem(v):
    """Naive standard error of the mean. Returns 0.0 for a single sample."""
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def zscore(a, ea, b, eb):
    d = sqrt(ea ** 2 + eb ** 2)
    return (a - b) / d if d > 0 else float("nan")


class Run:
    """One candidate: several log windows stitched into a single chain."""

    def __init__(self, label, paths, ladder=None):
        self.label, self.paths, self.ladder = label, paths, ladder
        self.trajs, self.cols, self.boundaries = [], None, []
        for p in paths:
            tr, names = parse(p)
            cols = build_columns(names)
            if self.cols is None:
                self.cols = cols
            elif [h for _, h in cols] != [h for _, h in self.cols]:
                sys.exit(f"{label}: monomial layout differs between windows:\n"
                         f"  {[h for _, h in self.cols]}\n  {[h for _, h in cols]}")
            self.boundaries.append(len(self.trajs))   # index of each window's first traj
            self.trajs.extend(tr)
        self.headers = [h for _, h in self.cols]
        self.key_of = {h: k for k, h in self.cols}

    # -- per-trajectory scalars -------------------------------------------
    def wall(self, i):
        """MD-only: 'Total H before' -> 'Total H after'. PRIMARY metric."""
        t = self.trajs[i]
        return t["t1"] - t["t0"]

    def cycle(self, i):
        """Full trajectory: Grid's 'Total time for trajectory (s):'. Adds
        checkpoint write, measurement, momentum refresh and the pseudofermion
        heatbath on top of wall(). None for logs predating that line."""
        return self.trajs[i].get("ttt")

    def has_cycle(self):
        return all(self.cycle(i) is not None for i in range(len(self.trajs)))

    def is_warm(self, i):
        return i not in self.boundaries

    def warm_idx(self):
        return [i for i in range(len(self.trajs)) if self.is_warm(i)]

    def sigma_force(self, i):
        b = buckets(self.trajs[i])
        return sum(b.get(k, 0.0) for k, _ in self.cols)

    def limiter(self, i):
        """Largest |F.dt| over every monomial and step of this trajectory."""
        t = self.trajs[i]
        return max((t["fdt"].get(k, 0.0) for k, h in self.cols if k != SMEARKEY),
                   default=0.0)

    def plaq(self, i):
        return sorted(self.trajs[i]["plaq"])[0]

    def plaq_sm(self, i):
        return sorted(self.trajs[i]["plaq"])[-1]

    def dHs(self):
        return [t["dH"] for t in self.trajs]

    def exps(self):
        return [t.get("exp", float("nan")) for t in self.trajs]

    def n_acc(self):
        return sum(1 for t in self.trajs if t.get("metrop") == "ACCEPTED")

    # -- per-sector aggregates --------------------------------------------
    def sector_time(self, h):
        """Mean seconds in bucket `h` over warm trajectories."""
        k = self.key_of.get(h)
        if k is None:
            return None
        return mean([buckets(self.trajs[i]).get(k, 0.0) for i in self.warm_idx()])

    def sector_fdt_max(self, h):
        """Largest |F.dt| for this monomial over the WHOLE chain."""
        k = self.key_of.get(h)
        if k is None or k == SMEARKEY:
            return None
        return max(t["fdt"].get(k, 0.0) for t in self.trajs)

    def sector_fdt_avg(self, h):
        """Site-averaged |F.dt| for this monomial, meaned over the chain."""
        k = self.key_of.get(h)
        if k is None or k == SMEARKEY:
            return None
        vals = []
        for t in self.trajs:
            a = t["fda"].get(k, [])
            if a:
                vals.append(mean(a))
        return mean(vals) if vals else None

    def rung_label(self, h):
        """`m_i -> m_{i+1}` for a PFk rung, the bare mass for the tail."""
        if not self.ladder:
            return ""
        if h.startswith("PF"):
            i = int(h[2:])
            if i + 1 < len(self.ladder):
                return f"{self.ladder[i]} → {self.ladder[i + 1]}"
        if h == "tail":
            return f"{self.ladder[-1]}"
        return ""


def fmt(x, spec=".0f", dash="—"):
    return dash if x is None else format(x, spec)


def row(cells):
    print("| " + " | ".join(cells) + " |")


def sep(n, first="---:"):
    print("|" + first + "|" + "---:|" * (n - 1))


# ------------------------------------------------------------------ tables
def table_config(runs):
    print("\n### Run configuration\n")
    row(["item"] + [r.label for r in runs])
    sep(len(runs) + 1, ":---")
    row(["log windows"] + [str(len(r.paths)) for r in runs])
    row(["trajectories"] + [str(len(r.trajs)) for r in runs])
    row(["ladder"] + [", ".join(r.ladder) if r.ladder else "—" for r in runs])
    row(["monomials"] + [", ".join(h for h in r.headers if h != "smear") for r in runs])


def table_summary(runs):
    print("\n### Summary comparison (over all trajectories)\n")
    ref = runs[0]
    row(["Metric"] + [r.label for r in runs])
    sep(len(runs) + 1, ":---")
    row(["Trajectories run"] + [str(len(r.trajs)) for r in runs])
    row(["**Acceptance rate**"] +
        [f"**{100.0 * r.n_acc() / len(r.trajs):.1f}% ({r.n_acc()}/{len(r.trajs)})**" for r in runs])
    row(["Avg wall time, warm *(MD-only)*"] +
        [f"{mean([r.wall(i) for i in r.warm_idx()]):.0f} s" for r in runs])
    row(["Warm wall spread"] +
        [f"{min(r.wall(i) for i in r.warm_idx()):.0f}–{max(r.wall(i) for i in r.warm_idx()):.0f} s"
         for r in runs])
    row(["Avg cycle time, warm *(full traj)*"] +
        [f"{mean([r.cycle(i) for i in r.warm_idx()]):.0f} s" if r.has_cycle() else "—"
         for r in runs])
    row(["Avg ΔH"] + [f"{mean(r.dHs()):+.3f}" for r in runs])
    row(["Median ΔH"] + [f"{median(r.dHs()):+.3f}" for r in runs])
    row(["Max ΔH"] + [f"{max(r.dHs()):+.3f}" for r in runs])
    row(["Min ΔH"] + [f"{min(r.dHs()):+.3f}" for r in runs])
    row(["**⟨e^(−ΔH)⟩** (must be 1)"] +
        [f"**{mean(r.exps()):.3f} ± {sem(r.exps()):.3f}**" for r in runs])
    row(["deviation from 1"] +
        [f"{abs(mean(r.exps()) - 1.0) / sem(r.exps()):.1f}σ" if sem(r.exps()) > 0 else "—"
         for r in runs])
    row(["Avg max\\|F·dt\\| (limiter)"] +
        [f"{mean([r.limiter(i) for i in range(len(r.trajs))]):.4f}" for r in runs])
    row(["Max \\|F·dt\\| observed"] +
        [f"{max(r.limiter(i) for i in range(len(r.trajs))):.4f}" for r in runs])
    row(["limiting sector"] + [limiting_sector(r) for r in runs])
    row(["Catastrophic blowups (\\|ΔH\\| > 10)"] +
        [str(sum(1 for d in r.dHs() if abs(d) > 10)) for r in runs])
    del ref


def limiting_sector(r):
    best, bh = -1.0, "—"
    for h in r.headers:
        v = r.sector_fdt_max(h)
        if v is not None and v > best:
            best, bh = v, h
    return f"{bh} ({best:.4f})"


def table_efficiency(runs):
    print("\n### Efficiency and totals\n")
    ref = runs[0]
    refeff = mean([ref.wall(i) for i in ref.warm_idx()]) * len(ref.trajs) / ref.n_acc()
    row([""] + [r.label for r in runs])
    sep(len(runs) + 1, ":---")
    row(["Total wall time (as run, incl. cold + resume)"] +
        [f"{sum(r.wall(i) for i in range(len(r.trajs))):.0f} s" for r in runs])
    row(["Warm wall / trajectory"] +
        [f"{mean([r.wall(i) for i in r.warm_idx()]):.0f} s" for r in runs])
    effs = []
    for r in runs:
        e = mean([r.wall(i) for i in r.warm_idx()]) * len(r.trajs) / r.n_acc()
        effs.append(e)
    row(["**Efficiency per accepted config**"] + [f"**{e:.0f} s**" for e in effs])
    row(["**Speedup vs " + ref.label + "** *(MD-only — headline)*"] +
        [("1.00×" if i == 0 else f"**{refeff / e:.2f}×**") for i, e in enumerate(effs)])
    row(["Wall-time change"] +
        [("—" if i == 0 else f"{100.0 * (e - refeff) / refeff:+.1f}%") for i, e in enumerate(effs)])
    # Same efficiency measure in the full-trajectory convention. Reported so a
    # reader never has to reach across docs (and so nobody forms a ratio with an
    # MD-only numerator and a full-trajectory denominator, or vice versa).
    if all(r.has_cycle() for r in runs):
        ceffs = [mean([r.cycle(i) for i in r.warm_idx()]) * len(r.trajs) / r.n_acc()
                 for r in runs]
        row(["Efficiency per accepted config *(full traj)*"] + [f"{e:.0f} s" for e in ceffs])
        row(["Speedup vs " + ref.label + " *(full traj)*"] +
            [("1.00×" if i == 0 else f"{ceffs[0] / e:.2f}×") for i, e in enumerate(ceffs)])
    row(["Σ force / wall (warm)"] +
        [f"{100.0 * mean([r.sigma_force(i) for i in r.warm_idx()]) / mean([r.wall(i) for i in r.warm_idx()]):.1f}%"
         for r in runs])


def table_observables(runs):
    print("\n### Physical observables (whole chain, rejects kept as repeats)\n")
    ref = runs[0]
    obs = [
        ("plaquette (unsmeared)", lambda r: [r.plaq(i) for i in range(len(r.trajs))], ".7f"),
        ("plaquette (stout-smeared)", lambda r: [r.plaq_sm(i) for i in range(len(r.trajs))], ".7f"),
        ("Polyakov loop Re", lambda r: [r.trajs[i]["poly"][0] for i in range(len(r.trajs))], "+.5f"),
        ("Polyakov loop Im", lambda r: [r.trajs[i]["poly"][1] for i in range(len(r.trajs))], "+.5f"),
    ]
    row(["observable"] + [r.label for r in runs] +
        [f"z vs {ref.label}: {r.label}" for r in runs[1:]])
    sep(len(runs) * 2, ":---")
    for name, getter, spec in obs:
        vals = [getter(r) for r in runs]
        cells = [f"{mean(v):{spec}} ± {sem(v):.7f}" for v in vals]
        zs = [f"{zscore(mean(vals[i]), sem(vals[i]), mean(vals[0]), sem(vals[0])):+.2f}σ"
              for i in range(1, len(runs))]
        row([name] + cells + zs)


def table_sectors(runs):
    print("\n### Per-sector comparison (time = warm-trajectory mean; "
          "force = max/avg over the whole chain)\n")
    present = [h for h in SECT_ORDER if any(h in r.headers for r in runs)]
    hdr = ["piece"]
    for r in runs:
        hdr.append(f"{r.label}<br>mass split")
    for r in runs:
        hdr.append(f"{r.label}<br>time (s)")
    for r in runs:
        hdr.append(f"{r.label}<br>max\\|F·dt\\|")
    for r in runs:
        hdr.append(f"{r.label}<br>avg\\|F·dt\\|")
    row(hdr)
    sep(len(hdr), ":---")
    for h in present:
        cells = [h]
        for r in runs:
            cells.append(r.rung_label(h) if h in r.headers else "—")
        for r in runs:
            cells.append(fmt(r.sector_time(h), ".0f"))
        for r in runs:
            cells.append(fmt(r.sector_fdt_max(h), ".4f"))
        for r in runs:
            cells.append(fmt(r.sector_fdt_avg(h), ".4f"))
        row(cells)
    cells = ["**Σforce / wall**"] + ["" for _ in runs]
    for r in runs:
        sf = mean([r.sigma_force(i) for i in r.warm_idx()])
        w = mean([r.wall(i) for i in r.warm_idx()])
        cells.append(f"**{sf:.0f} / {w:.0f}**")
    cells += ["" for _ in runs] + ["" for _ in runs]
    row(cells)


def table_per_traj(r, start=2000):
    print(f"\n### {r.label} — Table 1: physics + force-time split\n")
    ch = r.headers
    row(["traj", "cfg", "ΔH", "e^(−ΔH)", "Metrop.", "plaquette", "smeared",
         "Polyakov", "wall (s)"] + ch + ["Σforce"])
    sep(10 + len(ch))
    for i, t in enumerate(r.trajs):
        b = buckets(t)
        pl = sorted(t["plaq"])
        mark = "*" if i == 0 else ("†" if i in r.boundaries else "")
        row([f"{i + 1}{mark}", str(start + i + 1), f"{t['dH']:+.4f}",
             f"{t.get('exp', float('nan')):.3f}", t.get("metrop", "?")[:3],
             f"{pl[0]:.7f}", f"{pl[-1]:.7f}",
             f"({t['poly'][0]:+.4f}, {t['poly'][1]:+.4f})", f"{r.wall(i):.0f}"]
            + [f"{b.get(k, 0.0):.0f}" for k, _ in r.cols]
            + [f"{r.sigma_force(i):.0f}"])
    print(f"\n`*` = cold start (one-time init).  `†` = checkpoint resume "
          f"(cold solver caches).  Both excluded from warm timing means.")

    print(f"\n### {r.label} — Table 2: kick \\|F·dt\\| per action piece — max / avg\n")
    cols2 = [c for c in r.cols if c[0] != SMEARKEY]
    row(["traj", "cfg"] + [h for _, h in cols2])
    sep(2 + len(cols2))
    for i, t in enumerate(r.trajs):
        cells = []
        for k, _ in cols2:
            mx = t["fdt"].get(k, 0.0)
            a = t["fda"].get(k, [])
            cells.append(f"{mx:.4f} / {(mean(a) if a else 0.0):.4f}")
        row([str(i + 1), str(start + i + 1)] + cells)


def write_csv(runs, out, start=2000):
    import csv
    cols_hdr = ["run", "traj", "traj_abs", "accepted", "cold", "resume", "dH", "exp_dH",
                "plaquette", "plaquette_smeared", "poly_re", "poly_im",
                "wall_s", "sigma_force_s", "limiter_kick"]
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols_hdr)
        for r in runs:
            for i, t in enumerate(r.trajs):
                pl = sorted(t["plaq"])
                w.writerow([
                    r.label, i + 1, start + i + 1,
                    1 if t.get("metrop") == "ACCEPTED" else 0,
                    1 if i == 0 else 0,
                    1 if (i in r.boundaries and i != 0) else 0,
                    f"{t['dH']:.6f}", f"{t.get('exp', float('nan')):.6f}",
                    f"{pl[0]:.8f}", f"{pl[-1]:.8f}",
                    f"{t['poly'][0]:.6f}", f"{t['poly'][1]:.6f}",
                    f"{r.wall(i):.1f}", f"{r.sigma_force(i):.1f}", f"{r.limiter(i):.4f}"])
                n += 1
    sys.stderr.write(f"wrote {out}  ({n} rows)\n")


def main(argv):
    runs_spec, ladders, csv_out = [], {}, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--run":
            label, _, paths = argv[i + 1].partition("=")
            runs_spec.append((label, paths.split(",")))
            i += 2
        elif a == "--ladder":
            label, _, masses = argv[i + 1].partition("=")
            ladders[label] = masses.split(",")
            i += 2
        elif a == "--csv":
            csv_out = argv[i + 1]
            i += 2
        else:
            sys.exit(f"unknown argument {a!r}\n{__doc__}")
    if not runs_spec:
        sys.exit(__doc__)
    runs = [Run(l, p, ladders.get(l)) for l, p in runs_spec]
    table_config(runs)
    table_summary(runs)
    table_efficiency(runs)
    table_observables(runs)
    table_sectors(runs)
    for r in runs:
        table_per_traj(r)
    if csv_out:
        write_csv(runs, csv_out)


if __name__ == "__main__":
    main(sys.argv[1:])
