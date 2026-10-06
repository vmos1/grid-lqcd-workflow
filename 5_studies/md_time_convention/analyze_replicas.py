#!/usr/bin/env python3
"""Replica analysis of run_pure_gauge.sh PHASE=replica (__docs/2026_10_05_grid_chroma_md_time_handoff.md §8).

Runs <code>_<action>_<setting>_n<N>_r<R>: independent chains from one shared configuration. Per
replica, after dropping SKIP trajectories: <dH^2>, <dH>, acceptance, <plaq>. Per (action, n, code):
mean and standard error over replicas. Ratio R = <dH^2>_Grid(0.7071) / <dH^2>_Chroma(1.0) and the
implied factor c = sqrt(2) R^(1/4) (leapfrog <dH^2> ~ h^4), errors from the replica scatter.
Usage: python3 analyze_replicas.py <run dir> [SKIP, default 100]
"""
import glob
import math
import os
import re
import sys

import numpy as np

OUT = sys.argv[1]
SKIP = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def grid_run(d):
    txt = open(os.path.join(d, 'stdout.log'), errors='replace').read()
    dh = [float(x) for x in re.findall(r'Total H after trajectory\s*=\s*\S+\s+dH = (\S+)', txt)]
    acc = [m == 'ACCEPTED' for m in re.findall(r'Metropolis_test -- (ACCEPTED|REJECTED)', txt)]
    plaq = {}
    for m in re.finditer(r'Plaquette: \[ (\d+) \] (\S+)', txt):
        plaq.setdefault(int(m.group(1)), float(m.group(2)))
    return np.array(dh), np.array(acc, float), np.array([plaq[k] for k in sorted(plaq)])


def chroma_run(d):
    log = open(os.path.join(d, 'log.xml'), errors='replace').read()
    dh = [float(x) for x in re.findall(r'<deltaH>([^<]+)</deltaH>', log)]
    acc = [x.strip() == 'true' for x in re.findall(r'<AcceptP>([^<]+)</AcceptP>', log)]
    out = open(os.path.join(d, 'out.xml'), errors='replace').read()
    plaq = {}
    for m in re.finditer(r'<Plaquette>\s*<update_no>(\d+)</update_no>\s*<w_plaq>([^<]+)</w_plaq>', out):
        plaq.setdefault(int(m.group(1)), float(m.group(2)))
    return np.array(dh), np.array(acc, float), np.array([plaq[k] for k in sorted(plaq)])


groups = {}
for d in sorted(glob.glob(os.path.join(OUT, '*_r*'))):
    m = re.match(r'(grid|chroma)_(wilson|lw)_\S+_n(\d+)_r(\d+)$', os.path.basename(d))
    if not m:
        continue
    code, act, n, r = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    try:
        dh, acc, pl = grid_run(d) if code == 'grid' else chroma_run(d)
    except OSError:
        print(f'{os.path.basename(d)}: unreadable')
        continue
    if len(dh) <= SKIP:
        print(f'{os.path.basename(d)}: only {len(dh)} trajectories')
        continue
    dh, acc, pl = dh[SKIP:], acc[SKIP:], pl[SKIP:]
    groups.setdefault((act, n, code), []).append(
        dict(ms=(dh ** 2).mean(), mdh=dh.mean(), acc=acc.mean(), plaq=pl.mean(), ntraj=len(dh), ex=np.exp(-dh).mean()))


def mse(vals):
    v = np.array(vals)
    return v.mean(), (v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float('nan'))


print(f'SKIP={SKIP}. Per (action, n, code): mean ± s.e. over replicas')
print(f'{"action":6s} {"n":>3s} {"code":6s} {"rep":>3s} {"<dH^2>":>18s} {"<dH>":>18s} {"acc":>14s} {"<exp(-dH)>":>16s} {"<plaq>":>20s}')
for key in sorted(groups):
    g = groups[key]
    ms, msd = mse([x['ms'] for x in g])
    md, mdd = mse([x['mdh'] for x in g])
    ac, acd = mse([x['acc'] for x in g])
    ex, exd = mse([x['ex'] for x in g])
    pl, pld = mse([x['plaq'] for x in g])
    print(f'{key[0]:6s} {key[1]:3d} {key[2]:6s} {len(g):3d} {ms:9.5f}±{msd:7.5f} {md:9.5f}±{mdd:7.5f} '
          f'{ac:6.4f}±{acd:6.4f} {ex:8.5f}±{exd:6.5f} {pl:.6f}±{pld:.6f}')

print('\nRatio R = <dH^2> Grid(TRAJL 0.7071) / Chroma(tau0 1.0); factor c = sqrt(2) R^(1/4):')
cs, ws = [], []
for act in ('wilson', 'lw'):
    for n in sorted({k[1] for k in groups if k[0] == act}):
        g, c = groups.get((act, n, 'grid')), groups.get((act, n, 'chroma'))
        if not g or not c:
            continue
        mg, sg = mse([x['ms'] for x in g])
        mc, sc = mse([x['ms'] for x in c])
        R = mg / mc
        sR = R * math.sqrt((sg / mg) ** 2 + (sc / mc) ** 2)
        f = math.sqrt(2) * R ** 0.25
        sf = f * 0.25 * sR / R
        cs.append(f)
        ws.append(1 / sf ** 2)
        print(f'  {act:6s} n={n:2d}: R = {R:.3f} ± {sR:.3f}   c = {f:.4f} ± {sf:.4f}')
if cs:
    cw = np.average(cs, weights=ws)
    print(f'  weighted mean c = {cw:.4f} ± {1 / math.sqrt(sum(ws)):.4f}   (sqrt 2 = 1.4142)')

print('\nPlaquette per action and code (all n pooled, replica scatter):')
for act in ('wilson', 'lw'):
    for code in ('chroma', 'grid'):
        v = [x['plaq'] for k, g in groups.items() if k[0] == act and k[2] == code for x in g]
        if v:
            m, s = mse(v)
            print(f'  {act:6s} {code:6s} {m:.6f} ± {s:.6f}  ({len(v)} replicas)')
