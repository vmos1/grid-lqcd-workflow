#!/usr/bin/env python3
"""Analyse run_pure_gauge.sh output (__docs/2026_10_05_grid_chroma_md_time_handoff.md §6).

Per run: N, <dH>, rms dH = sqrt<dH^2>, <exp(-dH)> (must be 1), acceptance, <plaq> (binned error).
Time factor: at fixed MD step count n, the Grid TRAJL whose <dH^2> equals Chroma's at tau0 1.0
(log-log interpolation along Grid's TRAJL 0.25 / 0.5 / 0.7071 / 1.0 points), factor = 1 / TRAJL;
bootstrap error over trajectories of both codes. The sqrt 2 map predicts 1.414, the factor-4 map 4.
Usage: python3 analyze_pure_gauge.py [run dir, default runs/2026_10_5_pure_gauge]
"""
import glob
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../../../runs/2026_10_5_pure_gauge')
rng = np.random.default_rng(7)


def grid_run(d):
    txt = open(os.path.join(d, 'stdout.log'), errors='replace').read()
    dh = [float(x) for x in re.findall(r'Total H after trajectory\s*=\s*\S+\s+dH = (\S+)', txt)]
    acc = [m == 'ACCEPTED' for m in re.findall(r'Metropolis_test -- (ACCEPTED|REJECTED)', txt)]
    plaq = {}
    for m in re.finditer(r'Plaquette: \[ (\d+) \] (\S+)', txt):
        plaq.setdefault(int(m.group(1)), float(m.group(2)))
    return np.array(dh), np.array(acc), np.array([plaq[k] for k in sorted(plaq)])


def chroma_run(d):
    log = open(os.path.join(d, 'log.xml'), errors='replace').read()
    dh = [float(x) for x in re.findall(r'<deltaH>([^<]+)</deltaH>', log)]
    acc = [x.strip() == 'true' for x in re.findall(r'<AcceptP>([^<]+)</AcceptP>', log)]
    out = open(os.path.join(d, 'out.xml'), errors='replace').read()
    plaq = {}
    for m in re.finditer(r'<Plaquette>\s*<update_no>(\d+)</update_no>\s*<w_plaq>([^<]+)</w_plaq>', out):
        plaq.setdefault(int(m.group(1)), float(m.group(2)))
    return np.array(dh), np.array(acc), np.array([plaq[k] for k in sorted(plaq)])


def binned_err(x, nbin=20):
    nb = len(x) // nbin
    if nb < 2:
        return float('nan')
    b = x[:nb * nbin].reshape(nb, nbin).mean(axis=1)
    return b.std(ddof=1) / math.sqrt(nb)


runs = {}
for d in sorted(glob.glob(os.path.join(OUT, '*_n*'))):
    name = os.path.basename(d)
    try:
        runs[name] = grid_run(d) if name.startswith('grid_') else chroma_run(d)
    except OSError as e:
        print(f'{name}: unreadable ({e})')

print(f'{"run":34s} {"N":>5s} {"<dH>":>18s} {"rms dH":>9s} {"<exp(-dH)>":>16s} {"acc":>6s} {"<plaq>":>20s}')
for name, (dh, acc, pl) in runs.items():
    if len(dh) == 0:
        print(f'{name:34s} no trajectories')
        continue
    e = np.exp(-dh)
    print(f'{name:34s} {len(dh):5d} {dh.mean():9.5f}±{dh.std(ddof=1)/math.sqrt(len(dh)):7.5f} '
          f'{math.sqrt((dh**2).mean()):9.5f} {e.mean():8.4f}±{e.std(ddof=1)/math.sqrt(len(e)):6.4f} '
          f'{acc.mean() if len(acc) else float("nan"):6.3f} {pl.mean():.6f}±{binned_err(pl):.6f}')

print('\nEquilibrium plaquette per code (pooled over scan runs; Grid runs share a start and RNG):')
for act in ('wilson', 'lw'):
    for code in ('chroma', 'grid'):
        pl = [runs[n][2] for n in runs if n.startswith(f'{code}_{act}_') and len(runs[n][2])]
        if pl:
            allp = np.concatenate(pl)
            print(f'  {act:6s} {code:6s} <plaq> = {allp.mean():.6f}  (runs {len(pl)}, per-run binned err ~{np.mean([binned_err(p) for p in pl]):.6f})')


def ms(dh):
    return (dh ** 2).mean()


def steps_of(prefix):
    return sorted({int(m.group(1)) for k in runs for m in [re.match(re.escape(prefix) + r'_n(\d+)$', k)] if m})


print('\nTime factor from <dH^2> at matched step count (Grid TRAJL scan vs Chroma tau0 1.0):')
for act in ('wilson', 'lw'):
    for n in sorted(set(steps_of(f'grid_{act}_trajl0.25')) & set(steps_of(f'chroma_{act}_tau1.0'))):
        ch = runs.get(f'chroma_{act}_tau1.0_n{n}')
        pts = []
        for tl in ('0.25', '0.5', '0.70711', '1.0'):
            g = runs.get(f'grid_{act}_trajl{tl}_n{n}')
            if g is not None and len(g[0]):
                pts.append((float(tl), g[0]))
        if ch is None or len(ch[0]) == 0 or len(pts) < 2:
            print(f'  {act} n={n}: incomplete')
            continue
        x = np.log([p[0] for p in pts])

        def factor(chd, gds):
            y = np.log([ms(g) for g in gds])
            t = math.log(ms(chd))
            if not np.all(np.diff(y) > 0) or t < y[0] or t > y[-1]:
                return float('nan')        # non-monotonic, or Chroma outside Grid's range
            return 1.0 / math.exp(np.interp(t, y, x))

        f0 = factor(ch[0], [p[1] for p in pts])
        boots = []
        for _ in range(400):
            cb = rng.choice(ch[0], len(ch[0]))
            gb = [rng.choice(p[1], len(p[1])) for p in pts]
            boots.append(factor(cb, gb))
        boots = np.array([b for b in boots if not math.isnan(b)])
        grid_ms = ', '.join(f'{p[0]}: {ms(p[1]):.3g}' for p in pts)
        print(f'  {act:6s} n={n:2d}: Chroma <dH^2> {ms(ch[0]):.3g}; Grid <dH^2> by TRAJL {{{grid_ms}}}')
        if math.isnan(f0):
            print('            factor: Chroma outside Grid range or non-monotonic')
        else:
            print(f'            factor tau0/TRAJL = {f0:.4f} ± {boots.std():.4f}  ({len(boots)} bootstraps)')

print('\nDirect check, sqrt 2 map: <dH^2> ratio Grid(0.7071, n) / Chroma(1.0, n), expect 1:')
for act in ('wilson', 'lw'):
    for n in steps_of(f'chroma_{act}_tau1.0'):
        ch = runs.get(f'chroma_{act}_tau1.0_n{n}')
        g = runs.get(f'grid_{act}_trajl0.70711_n{n}')
        if ch is None or g is None or not len(ch[0]) or not len(g[0]):
            continue
        r = ms(g[0]) / ms(ch[0])
        rb = [ms(rng.choice(g[0], len(g[0]))) / ms(rng.choice(ch[0], len(ch[0]))) for _ in range(400)]
        print(f'  {act:6s} n={n:2d}: {r:.3f} ± {np.std(rb):.3f}   (acc Grid {g[1].mean():.3f}, Chroma {ch[1].mean():.3f})')
