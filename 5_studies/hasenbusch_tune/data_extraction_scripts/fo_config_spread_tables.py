"""Force-only numbers across configurations of the ensemble: the tables of
__docs/2026_10_26_force_only_config_spread.md (markdown on stdout).

Input: FORCES_ONLY runs of forces48_puregrid.sh, one per configuration (DIAG_START_CFG), same
ladder and settings; draws = samples 1..N (sample 0 carries first-call setup, L014). The
configuration is read from the IMPORT_CFG field of each run's ENV line.

1. Per configuration and piece: RMS and peak, mean over draws +- standard error (sd/sqrt(N)).
2. Spread per piece: pooled draw-to-draw sd, the noise that leaves in an N-draw mean, the
   config-to-config sd of those means, and their ratio (near 1 = configs differ by noise only).
3. Per stream: mean of the configuration means +- error from their spread (sd/sqrt(n configs)),
   draw-noise-only error in brackets.
4. Rung-1 draws with a peak above the threshold, per configuration.

Usage: python3 fo_config_spread_tables.py <runs dir> <threshold> <run> [<run> ...]
"""
import math
import re
import sys
from collections import defaultdict

RUNS, THR, NAMES = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
pat = re.compile(r"FORCES_ONLY level (\S+) sample (\d+) avg=(\S+) max=(\S+) refresh=(\S+) time=(\S+)")
PIECES = ["PF0", "PF1", "PF2", "PF3", "LightSchurPF", "Strange", "LightLogDet", "StrangeLogDet", "Gauge"]
LABEL = {"PF0": "rung 1", "PF1": "rung 2", "PF2": "rung 3", "PF3": "rung 4", "LightSchurPF": "tail",
         "Strange": "strange", "LightLogDet": "light log-det", "StrangeLogDet": "strange log-det",
         "Gauge": "gauge"}
NOISY = set(PIECES[:6])   # log-dets and gauge have no noise field: identical every draw
REF = ("djm-3", 2000)     # cfg_2000, the configuration of every earlier force-only run


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0)


def fmt(m, e, p):
    """value +- error to the error's second significant figure; 4 figures if exact."""
    if p not in NOISY or e <= 0:
        return "%.4g" % m
    dec = max(0, 1 - int(math.floor(math.log10(e))))
    return "%.*f ± %.*f" % (dec, m, dec, e)


def parse(run):
    cfg, d = None, defaultdict(lambda: defaultdict(list))
    for line in open("%s/%s/hmc.log" % (RUNS, run), errors="replace"):
        if cfg is None and line.startswith("ENV FORCES48_PUREGRID"):
            s, t = re.search(r" IMPORT_CFG=\S*-(djm-\d+)_cfg_(\d+)\.lime", line).groups()
            cfg = (s, int(t))
        m = pat.search(line)
        if m and int(m.group(2)) >= 1:
            d[m.group(1)]["rms"].append(float(m.group(3)))
            d[m.group(1)]["peak"].append(float(m.group(4)))
    return cfg, d


def name(cfg):
    return "%s %d%s" % (cfg[0], cfg[1], " (cfg_2000)" if cfg == REF else "")


data = sorted(parse(r) for r in NAMES)
head = " | ".join(LABEL[p] for p in PIECES) + " |"
rule = "|---" * len(PIECES) + "|"

for meas in ("rms", "peak"):
    print("\n### 1. %s per configuration (mean of the draws ± standard error)\n" % meas.upper())
    print("| config | " + head)
    print("|---" + rule)
    for cfg, d in data:
        xs = [d[p][meas] for p in PIECES]
        print("| %s | " % name(cfg) + " | ".join(
            fmt(mean_sd(x)[0], mean_sd(x)[1] / math.sqrt(len(x)), p) for x, p in zip(xs, PIECES)) + " |")

for meas in ("rms", "peak"):
    print("\n### 2. %s spread per piece\n" % meas.upper())
    print("| piece | draw-to-draw sd | noise in a mean of the draws | config-to-config sd | ratio | lowest / highest config |")
    print("|---|---|---|---|---|---|")
    for p in PIECES:
        runs = [d[p][meas] for _, d in data]
        means = [mean_sd(x)[0] for x in runs]
        mu, between = mean_sd(means)
        if p in NOISY:
            pooled = math.sqrt(sum(mean_sd(x)[1] ** 2 for x in runs) / len(runs))
            noise = pooled / math.sqrt(sum(len(x) for x in runs) / len(runs))
            cells = ("%.2g (%.1f%%)" % (pooled, 100 * pooled / mu), "%.2g" % noise, "%.1f" % (between / noise))
        else:
            cells = ("none", "none", "no noise")
        print("| %s | %s | %s | %.2g (%.1f%%) | %s | %.4g / %.4g |" % (
            LABEL[p], cells[0], cells[1], between, 100 * between / mu, cells[2], min(means), max(means)))

for meas in ("rms", "peak"):
    print("\n### 3. %s per stream (mean of config means ± error from their spread [draw noise only])\n" % meas.upper())
    print("| stream | configs | " + head)
    print("|---|---" + rule)
    groups = defaultdict(list)
    for (s, _), d in data:
        groups[s].append(d)
        groups["all"].append(d)
    for s in sorted(groups, key=lambda k: (k == "all", k)):
        ds, cells = groups[s], []
        for p in PIECES:
            means = [mean_sd(d[p][meas])[0] for d in ds]
            mu, sd_cfg = mean_sd(means)
            c = fmt(mu, sd_cfg / math.sqrt(len(ds)), p)
            if p in NOISY:
                c += " [%.2g]" % (math.sqrt(sum(mean_sd(d[p][meas])[1] ** 2 / len(d[p][meas]) for d in ds)) / len(ds))
            cells.append(c)
        print("| %s | %d | " % (s, len(ds)) + " | ".join(cells) + " |")

print("\n### 4. Rung-1 draws with a peak above %.2f\n" % THR)
print("| config | peaks above %.2f | RMS of those draws | draws above | largest peak |" % THR)
print("|---|---|---|---|---|")
for cfg, d in data:
    pk, rm = d["PF0"]["peak"], d["PF0"]["rms"]
    hits = [(i + 1, pk[i], rm[i]) for i in range(len(pk)) if pk[i] > THR]
    print("| %s | %s | %s | %d of %d | %.3f |" % (
        name(cfg), ", ".join("%.3f (draw %d)" % (h[1], h[0]) for h in hits) or "—",
        ", ".join("%.4f" % h[2] for h in hits) or "—", len(hits), len(pk), max(pk)))
