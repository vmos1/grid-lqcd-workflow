"""Table of dH per trajectory for the pure-gauge ForceGradient step scan on cfg_2000.

Reads every <OUT>/<steps>_s<seed>/hmc.log written by run_pure_gauge_fg.sh (directories with a
prefix, e.g. smoke_48_s0, are listed separately and kept out of the statistics) and prints, per
step count: the individual dH values in trajectory order, rms dH, mean dH, <exp(-dH)>, accepted
count; then the rms ratio between successive step counts with the implied power of h
(log ratio / log step ratio; ForceGradient is 4th order, rms dH ~ h^4 in the smooth regime) and
the same power from the per-draw |dH| ratios (the same seed draws the same momenta at every step
count, so trajectory k of two step counts shares its momentum draw); finally rms dH at 48 steps as
a fraction of the full-action baseline sigma(dH) = 0.200 at TRAJL 1.0 (12 outer / 48 inner).

Usage (from the grid-lqcd-workflow repo root):
    uv run python 5_studies/md_time_convention/pure_gauge_fg_table.py [OUT]
OUT defaults to <workspace>/runs/2026_10_9_pure_gauge_fg48.
"""

import math
import re
import statistics
import sys
from pathlib import Path

FULL_ACTION_SIGMA = 0.200  # full-action sigma(dH), TRAJL 1.0, base+G 12 outer / 48 inner
PRODUCTION_PLAQ = 0.5435082104  # "Initial plaquette" of runs/2026_10_7_baseG_trajl1_seed300_20

RE_DIR = re.compile(r"^(?P<prefix>.*?)(?P<steps>\d+)_s(?P<seed>\d+)$")
RE_ENV = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")
RE_DH = re.compile(r"Total H after trajectory\s*=\s*(\S+)\s+dH\s*=\s*(\S+)")
RE_ACC = re.compile(r"Metropolis_test -- (ACCEPTED|REJECTED)")
RE_PLAQ = re.compile(r"Initial plaquette\s*=\s*(\S+)")
RE_TTRAJ = re.compile(r"Total time for trajectory \(s\):\s*(\S+)")
RE_EXIT = re.compile(r"^exit=(\d+)")


def parse_log(path):
    env, dh, acc, ttraj = {}, [], [], []
    plaq, rc = None, None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("ENV "):
                env = {k: v.strip('"') for k, v in RE_ENV.findall(line)}
            elif m := RE_DH.search(line):
                dh.append(float(m.group(2)))
            elif m := RE_ACC.search(line):
                acc.append(m.group(1) == "ACCEPTED")
            elif m := RE_PLAQ.search(line):
                plaq = float(m.group(1))
            elif m := RE_TTRAJ.search(line):
                ttraj.append(float(m.group(1)))
            elif m := RE_EXIT.match(line):
                rc = int(m.group(1))
    return dict(env=env, dh=dh, acc=acc, plaq=plaq, ttraj=ttraj, rc=rc)


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[3] / "runs" / "2026_10_9_pure_gauge_fg48")
    runs, extra = {}, []
    for log in sorted(out.glob("*/hmc.log")):
        m = RE_DIR.match(log.parent.name)
        if not m:
            continue
        r = parse_log(log)
        r["name"] = log.parent.name
        if m.group("prefix"):
            extra.append(r)
            continue
        runs.setdefault(int(m.group("steps")), []).append((int(m.group("seed")), r))
    if not runs:
        sys.exit(f"no <steps>_s<seed>/hmc.log under {out}")

    print(f"Pure-gauge LW HMC from cfg_2000, step scan; run dir {out}")
    first = next(iter(runs.values()))[0][1]["env"]
    print(f"integrator {first.get('INTEGRATOR')}  TRAJL {first.get('TRAJL')}  "
          f"BIN_SHA256 {first.get('BIN_SHA256', '?')[:16]}...  CFG {Path(first.get('CFG', '?')).name}")
    print()

    # Provenance and plaquette checks
    print("run           exit  initial plaquette   d(plaq) vs production   s/traj (mean)")
    for steps in sorted(runs, reverse=True):
        for seed, r in sorted(runs[steps]):
            p = r["plaq"]
            dp = f"{p - PRODUCTION_PLAQ:+.1e}" if p is not None else "n/a"
            tt = f"{statistics.mean(r['ttraj']):.1f}" if r["ttraj"] else "n/a"
            print(f"{r['name']:<12}  {str(r['rc']):>4}  {p!s:<18}  {dp:<22}  {tt}")
    for r in extra:
        print(f"{r['name']:<12}  {str(r['rc']):>4}  {r['plaq']!s:<18}  (not in statistics)  "
              f"dH {', '.join(f'{x:.6g}' for x in r['dh'])}")
    print()

    # Per step count table
    rows = {}
    print(f"{'steps':>5} {'h':>8} {'n':>2} {'acc':>4}  {'rms dH':>10} {'mean dH':>11} "
          f"{'<exp(-dH)>':>10}  dH per trajectory (in order)")
    for steps in sorted(runs, reverse=True):
        dh, acc, per_seed = [], [], {}
        for seed, r in sorted(runs[steps]):
            dh += r["dh"]
            acc += r["acc"]
            per_seed[seed] = r["dh"]
        h = float(first.get("TRAJL", 1.0)) / steps
        rows[steps] = dict(dh=dh, rms=rms(dh), per_seed=per_seed)
        mean = statistics.mean(dh) if dh else float("nan")
        eav = statistics.mean(math.exp(-x) for x in dh) if dh else float("nan")
        print(f"{steps:>5} {h:>8.5f} {len(dh):>2} {sum(acc):>2}/{len(acc):<1}  {rms(dh):>10.3e} "
              f"{mean:>+11.3e} {eav:>10.5f}  {'  '.join(f'{x:+.4e}' for x in dh)}")
    print()

    # Scaling between successive step counts
    order = sorted(rows, reverse=True)
    print("step ratio        rms ratio   implied power (rms)   implied power per draw (|dH| ratios)")
    for a, b in zip(order, order[1:]):
        ra, rb = rows[a]["rms"], rows[b]["rms"]
        ratio = rb / ra if ra else float("nan")
        p_rms = math.log(ratio) / math.log(a / b) if ratio > 0 else float("nan")
        draws = []
        for seed, da in rows[a]["per_seed"].items():
            db = rows[b]["per_seed"].get(seed, [])
            for xa, xb in zip(da, db):
                if xa != 0 and xb != 0:
                    draws.append(math.log(abs(xb) / abs(xa)) / math.log(a / b))
        pd = "  ".join(f"{p:.2f}" for p in draws)
        med = f"median {statistics.median(draws):.2f}: " if draws else ""
        print(f"{a:>3} -> {b:<3} ({a / b:.3f})  {ratio:>9.3f}   {p_rms:>8.2f}             {med}{pd}")
    print()

    # h^4 extrapolation from the finest step count and the full-action comparison
    ref = order[0]
    print(f"h^4 prediction from {ref} steps (rms(n) = rms({ref}) * ({ref}/n)^4) vs measured:")
    for s in order:
        pred = rows[ref]["rms"] * (ref / s) ** 4
        print(f"  {s:>3} steps: predicted {pred:.3e}  measured {rows[s]['rms']:.3e}  "
              f"measured/full-action sigma {rows[s]['rms'] / FULL_ACTION_SIGMA:.3f}")
    if 48 in rows:
        r48 = rows[48]["rms"]
        print(f"\nGauge rms dH at 48 steps = {r48:.3e} = {r48 / FULL_ACTION_SIGMA:.4f} x the full-action "
              f"sigma(dH) {FULL_ACTION_SIGMA} (TRAJL 1.0, 12 outer / 48 inner); "
              f"in variance {(r48 / FULL_ACTION_SIGMA) ** 2:.2e}")


if __name__ == "__main__":
    main()
