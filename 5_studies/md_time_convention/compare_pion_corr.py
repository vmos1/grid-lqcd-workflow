#!/usr/bin/env python3
"""Compare Grid's PROPCORR (pion_corr_grid) with Chroma's <prop_corr> (PROPAGATOR) on the same
configuration (__docs/2026_10_05_grid_chroma_md_time_handoff.md §8). Equal C(t) at every t means the
two Dirac operators agree in mass, csw, smearing, boundary and normalisation (a constant factor c in
M would give C_grid / C_chroma = 1/c^2 at every t; a mass or boundary mismatch, a t-dependent ratio).
Usage: python3 compare_pion_corr.py [run dir, default runs/2026_10_5_pion_corr]
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../../../runs/2026_10_5_pion_corr')

g_txt = open(os.path.join(OUT, 'grid', 'stdout.log'), errors='replace').read()
g = {int(t): float(v) for t, v in re.findall(r'PROPCORR (\d+) (\S+)', g_txt)}
grid = np.array([g[t] for t in sorted(g)])
for line in re.findall(r'(GAUGE .*|SOLVE .*)', g_txt):
    print('grid  ', line)

c_txt = open(os.path.join(OUT, 'chroma', 'out.xml'), errors='replace').read()
m = re.search(r'<prop_corr>([^<]+)</prop_corr>', c_txt)
chroma = np.array([float(x) for x in m.group(1).split()])
for tag in ('w_plaq', 'link'):
    mm = re.search(rf'<{tag}>([^<]+)</{tag}>', c_txt)
    if mm:
        print(f'chroma {tag} {mm.group(1)}')

nt = min(len(grid), len(chroma))
print(f'\nNt grid {len(grid)}, chroma {len(chroma)}')
print(f'{"t":>3s} {"Grid C(t)":>24s} {"Chroma C(t)":>24s} {"Grid/Chroma - 1":>16s}')
rel = grid[:nt] / chroma[:nt] - 1
for t in range(nt):
    print(f'{t:3d} {grid[t]:24.15e} {chroma[t]:24.15e} {rel[t]:16.3e}')
print(f'\nmax |Grid/Chroma - 1| = {np.abs(rel).max():.3e} (t = {int(np.abs(rel).argmax())});'
      f' mean ratio - 1 = {rel.mean():.3e}')
