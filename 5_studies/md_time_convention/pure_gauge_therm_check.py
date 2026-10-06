"""Thermalization check for run_pure_gauge.sh's therm phase: plaquette in blocks of 100 updates and
acceptance after warm-up, for each code and action. Run it before the scan phase.

Usage: python3 pure_gauge_therm_check.py runs/<date>_pure_gauge[_fine]
"""
import os
import re
import sys

import numpy as np

if len(sys.argv) != 2:
    sys.exit(__doc__)
OUT = sys.argv[1]
for name in ('grid_wilson_therm', 'grid_lw_therm', 'chroma_wilson_therm', 'chroma_lw_therm'):
    d = os.path.join(OUT, name)
    try:
        if name.startswith('grid'):
            t = open(os.path.join(d, 'stdout.log'), errors='replace').read()
            p = {}
            for a, b in re.findall(r'Plaquette: \[ (\d+) \] (\S+)', t):
                p.setdefault(int(a), float(b))
            acc = [x == 'ACCEPTED' for x in re.findall(r'Metropolis_test -- (\w+)', t)]
        else:
            t = open(os.path.join(d, 'out.xml'), errors='replace').read()
            p = {}
            for a, b in re.findall(r'<Plaquette>\s*<update_no>(\d+)</update_no>\s*<w_plaq>([^<]+)</w_plaq>', t):
                p.setdefault(int(a), float(b))
            lg = open(os.path.join(d, 'log.xml'), errors='replace').read()
            acc = [x.strip() == 'true' for x in re.findall(r'<AcceptP>([^<]+)</AcceptP>', lg)][50:]
    except OSError:
        print(f'{name}: not ready')
        continue
    v = np.array([p[k] for k in sorted(p)])
    blocks = [round(float(v[i:i + 100].mean()), 5) for i in range(0, len(v), 100)]
    print(f'{name:22s} n={len(v):4d} plaq by 100: {blocks}  acc after warm-up {np.mean(acc) if acc else float("nan"):.3f}')
