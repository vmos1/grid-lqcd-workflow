"""rms dH per MD step count for Grid and Chroma HMC runs, as a side-by-side markdown table.

Each positional argument is <code>:<steps>:<file>, code grid or chroma. Grid files are hmc.log
(every "Total H after trajectory ... dH = x" line is one trajectory); Chroma files are the
.xmldat output (every <deltaH>). Per (code, steps): trajectories n, rms dH = sqrt(<dH^2>) with a
bootstrap error (n >= 2: 10,000 resamples, fixed seed), accepted count, and the expected
acceptance <min(1, e^-dH)>. Step sizes: Grid eps = TRAJL/steps, Chroma dt = tau0/steps.

Usage (from the grid-lqcd-workflow root, paths relative to the workspace runs/):
  uv run python 5_studies/md_time_convention/dh_vs_step_table.py [--trajl 1.0] [--tau0 1.414] \
      grid:12:../runs/<run>/hmc.log ... chroma:12:../runs/<run>/<run>.xmldat ...
"""
import argparse
import math
import re
import sys
from collections import defaultdict

import numpy as np

GRID_DH = re.compile(r'Total H after trajectory\s*=\s*\S+\s+dH = (\S+)')
GRID_ACC = re.compile(r'Metropolis_test -- (ACCEPTED|REJECTED)')
CHROMA_DH = re.compile(r'<deltaH>([^<]+)</deltaH>')
CHROMA_ACC = re.compile(r'<AcceptP>([^<]+)</AcceptP>')


def read(code, path):
    text = open(path, errors='replace').read()
    if code == 'grid':
        dh = [float(x) for x in GRID_DH.findall(text)]
        acc = [x == 'ACCEPTED' for x in GRID_ACC.findall(text)]
    else:
        dh = [float(x) for x in CHROMA_DH.findall(text)]
        acc = [x.strip() == 'true' for x in CHROMA_ACC.findall(text)]
    if not dh or len(acc) != len(dh):
        sys.exit(f'{path}: {len(dh)} dH values, {len(acc)} accept flags')
    return dh, acc


def rms_with_error(x, rng):
    x = np.asarray(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    if len(x) < 2:
        return rms, None
    idx = rng.integers(0, len(x), size=(10000, len(x)))
    return rms, float(np.sqrt(np.mean(x[idx] ** 2, axis=1)).std(ddof=1))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--trajl', type=float, default=1.0, help='Grid TRAJL')
ap.add_argument('--tau0', type=float, default=1.414, help='Chroma tau0')
ap.add_argument('runs', nargs='+', help='<grid|chroma>:<steps>:<file>')
args = ap.parse_args()

data = defaultdict(lambda: ([], []))
for spec in args.runs:
    code, steps, path = spec.split(':', 2)
    if code not in ('grid', 'chroma'):
        sys.exit(f'unknown code {code!r} in {spec}')
    dh, acc = read(code, path)
    data[(code, int(steps))][0].extend(dh)
    data[(code, int(steps))][1].extend(acc)

rng = np.random.default_rng(20261007)


def cells(code, steps):
    if (code, steps) not in data:
        return ['—', '—', '—']
    dh, acc = data[(code, steps)]
    rms, err = rms_with_error(dh, rng)
    exp_acc = sum(min(1.0, math.exp(-d)) for d in dh) / len(dh)
    rms_s = f'{rms:.1f}' if err is None else f'{rms:.2f} ± {err:.2f}'
    return [str(len(dh)), rms_s, f'{exp_acc:.2f} ({sum(acc)}/{len(acc)})']


print('| steps | Grid ε | Chroma dt | Grid n | Grid rms ΔH | Grid acceptance, expected (observed) '
      '| Chroma n | Chroma rms ΔH | Chroma acceptance, expected (observed) |')
print('|---|---|---|---|---|---|---|---|---|')
for steps in sorted({s for _, s in data}):
    row = [str(steps), f'{args.trajl / steps:.3f}', f'{args.tau0 / steps:.3f}']
    row += cells('grid', steps) + cells('chroma', steps)
    print('| ' + ' | '.join(row) + ' |')
