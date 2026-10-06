"""Free-drift result from run_free_drift.sh: the Grid TRAJL whose plaquette equals Chroma's at tau0,
and the conversion factor tau0 / TRAJL. Interpolates log(1 - plaq) against log(TRAJL) along Grid's
curve (1 - plaq ~ c t^2 at small t), per Grid seed and seed-averaged. Also prints <|P|^2> per link
of every draw (Grid 2 H_kin / N, Chroma 4 + KE_old / N; the force is zero, so H = H_kin).

Usage: python3 free_drift_factor.py runs/<date>_free_drift/run.log
"""
import math
import re
import sys

import numpy as np

L = 8  # run_free_drift.sh lattice, L^4
N = 4 * L**4

if len(sys.argv) != 2:
    sys.exit(__doc__)
log = open(sys.argv[1]).read()
summ = log[log.index('=== summary'):]
chroma, grid, hk, ke = {}, {}, {}, {}
for m in re.finditer(r'chroma_tau([\d.]+)_seed(\d): <w_plaq>([\d.]+) <KE_old>([-\d.eE+]+)', summ):
    chroma[(float(m[1]), int(m[2]))] = float(m[3])
    ke[(float(m[1]), int(m[2]))] = float(m[4])
for m in re.finditer(r'grid_trajl([\d.]+)_seed(\d): Plaquette: \[ 1 \] ([\d.]+) \| Total H before trajectory = ([\d.eE+]+)', summ):
    grid[(float(m[1]), int(m[2]))] = float(m[3])
    hk[(float(m[1]), int(m[2]))] = float(m[4])

print('per-link <|P|^2>: Grid', sorted({round(2 * v / N, 4) for v in hk.values()}),
      ' Chroma', sorted({round(4 + v / N, 4) for v in ke.values()}))

ts = sorted({t for t, _ in grid})
seeds = sorted({s for _, s in grid})


def match(target, curve):
    x = np.log(ts)
    y = np.log([1 - curve[t] for t in ts])
    return math.exp(np.interp(math.log(1 - target), y, x))


for tau in sorted({t for t, _ in chroma}):
    pc = [chroma[k] for k in sorted(chroma) if k[0] == tau]
    pcm = sum(pc) / len(pc)
    curves = [(f'grid seed{s}', {t: grid[(t, s)] for t in ts}) for s in seeds]
    curves.append(('grid mean ', {t: sum(grid[(t, s)] for s in seeds) / len(seeds) for t in ts}))
    print(f'Chroma tau0={tau}: plaq ' + ', '.join(f'{p:.5f}' for p in pc) + f' (mean {pcm:.5f})')
    for label, curve in curves:
        tm = match(pcm, curve)
        print(f'   {label}: matching TRAJL {tm:.5f}  ->  factor tau0/TRAJL = {tau / tm:.4f}')
print('sqrt(2) =', round(math.sqrt(2), 4))
