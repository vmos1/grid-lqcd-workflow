#!/usr/bin/env python3
"""Autocorrelation-aware observable comparison across runs, from the tidy CSV
written by `compare_runs_multiwindow.py --csv`.

Why this exists: the naive standard error sigma/sqrt(N) assumes independent
samples. Consecutive HMC trajectories at tau=0.354 on a 48^3 lattice are
strongly correlated, so the naive error UNDERSTATES the true one and will
happily report a 3-sigma "disagreement" between two chains that are in fact
sampling the same distribution. Since the whole point of the C1/C2 comparison
is to show the observables AGREE with the baseline, using an error bar that is
biased small would make the agreement claim look worse than it is -- and, more
importantly, would be wrong.

Reports per run and observable:
  sd            per-trajectory standard deviation (the fluctuation scale)
  naive         sd/sqrt(N)                      -- lower bound, ignores tau
  tau_int       integrated autocorrelation time, Madras-Sokal automatic
                windowing (W = first w with w >= C*tau_int(w), C=5)
  corrected     naive * sqrt(2*tau_int)         -- the error to quote
  binned(b)     error from N/b non-overlapping block means, b=2,4,5

Pairwise z-scores against the FIRST run in the CSV are given with both the
naive and the corrected error, so the size of the correction is visible.

CAVEAT, stated here because it governs how the output should be read: with
N=20 the tau_int estimate is itself noisy (its own relative error is roughly
sqrt(2(2W+1)/N) ~ 60-80%), so `corrected` is an order-of-magnitude error bar,
not a precision one. It is used here only to answer "is the difference
resolvable at all", which is robust to that noise -- not to quote a final
uncertainty on the plaquette.

Usage:
  obs_autocorr_compare.py <csv> [obs_column ...]
"""
import sys
from math import sqrt


def mean(v):
    return sum(v) / len(v)


def var(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def autocorr(v, t):
    """Normalised autocorrelation at lag t (biased 1/N estimator)."""
    n, m = len(v), mean(v)
    c0 = sum((x - m) ** 2 for x in v) / n
    if c0 == 0:
        return 0.0
    ct = sum((v[i] - m) * (v[i + t] - m) for i in range(n - t)) / n
    return ct / c0


def tau_int(v, c=5.0):
    """Madras-Sokal integrated autocorrelation time with automatic windowing.
    tau = 0.5 + sum_{t=1..W} rho(t), W = first w satisfying w >= c*tau(w)."""
    n = len(v)
    tau, w = 0.5, 0
    for t in range(1, n // 2):
        tau += autocorr(v, t)
        w = t
        if tau <= 0:
            tau = 0.5
            break
        if t >= c * tau:
            break
    return max(tau, 0.5), w


def binned_err(v, b):
    """SEM of non-overlapping block means of length b (drops a short tail)."""
    nb = len(v) // b
    if nb < 2:
        return None
    blocks = [mean(v[i * b:(i + 1) * b]) for i in range(nb)]
    return sqrt(var(blocks) / nb)


def main(csv_path, obs_cols):
    import csv as _csv
    data = {}
    order = []
    with open(csv_path) as fh:
        for r in _csv.DictReader(fh):
            run = r["run"]
            if run not in data:
                data[run] = {c: [] for c in obs_cols}
                order.append(run)
            for c in obs_cols:
                data[run][c].append(float(r[c]))

    ref = order[0]
    for c in obs_cols:
        print(f"\n### {c}\n")
        print("| run | N | mean | sd | naive err | τ_int (W) | **corrected err** "
              "| binned b=2 | b=4 | b=5 |")
        print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        stats = {}
        for run in order:
            v = data[run][c]
            n = len(v)
            sd = sqrt(var(v))
            naive = sd / sqrt(n)
            t, w = tau_int(v)
            corr = naive * sqrt(2.0 * t)
            stats[run] = (mean(v), naive, corr)
            b2, b4, b5 = (binned_err(v, b) for b in (2, 4, 5))
            print(f"| {run} | {n} | {mean(v):.7f} | {sd:.2e} | {naive:.2e} "
                  f"| {t:.2f} ({w}) | **{corr:.2e}** "
                  f"| {b2:.2e} | {b4:.2e} | {b5:.2e} |")
        print(f"\n| difference vs {ref} | Δ | z (naive err) | **z (corrected err)** |")
        print("|:---|---:|---:|---:|")
        m0, n0, c0 = stats[ref]
        for run in order[1:]:
            m1, n1, c1 = stats[run]
            d = m1 - m0
            zn = d / sqrt(n0 ** 2 + n1 ** 2)
            zc = d / sqrt(c0 ** 2 + c1 ** 2)
            print(f"| {run} | {d:+.2e} | {zn:+.2f}σ | **{zc:+.2f}σ** |")


if __name__ == "__main__":
    cols = sys.argv[2:] or ["plaquette", "plaquette_smeared", "poly_re", "poly_im"]
    main(sys.argv[1], cols)
