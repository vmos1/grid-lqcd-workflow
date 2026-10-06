"""Per-link momentum norm <|P|^2> at the start of each HMC trajectory, read from existing logs.

Grid (hmc.log / stdout.log): H_kin = (Total H before trajectory) - (sum of the monomial actions
printed just before it). With CPS_MD_TIME, H_kin = sum |P|^2 / 2, so <|P|^2> per link =
2 H_kin / N_links. Monomial actions print with 6 significant digits: resolution ~1e-4 per link.
Chroma (.xmldat / out.xml): KE_old = sum over links of (|P|^2 - 4), so <|P|^2> = 4 + KE_old / N_links.
N_links = 4 x volume, from Grid's --grid argument or Chroma's <nrow>.
Expected per link: 8 for Grid with CPS_MD_TIME, 4 for Chroma.

Usage: python3 momentum_norm_from_logs.py FILE...   (*.xml, *.xmldat -> Chroma, anything else -> Grid)
A momentum draw printed in several files (restarts, copies) is counted once.
"""
import math
import re
import sys

PAT_S = re.compile(r'S \[\d+\]\[\d+\] H = ([-0-9.eE+]+)')
PAT_H = re.compile(r'Total H before trajectory = ([-0-9.eE+]+)')
PAT_GRID = re.compile(r'--grid (\d+)\.(\d+)\.(\d+)\.(\d+)')
PAT_NROW = re.compile(r'<nrow>\s*(\d+) (\d+) (\d+) (\d+)\s*</nrow>')
PAT_KE = re.compile(r'<KE_old>([-0-9.eE+]+)</KE_old>')


def links(dims):
    n = 4
    for d in dims:
        n *= int(d)
    return n


def grid_draws(path):
    s_sum, n_s, n_links, out = 0.0, 0, None, {}
    for line in open(path, errors='replace'):
        if n_links is None:
            g = PAT_GRID.search(line)
            if g:
                n_links = links(g.groups())
        if 'Total H after' in line:
            s_sum, n_s = 0.0, 0
            continue
        m = PAT_S.search(line)
        if m:
            s_sum += float(m.group(1))
            n_s += 1
            continue
        m = PAT_H.search(line)
        if m:
            h = float(m.group(1))
            if n_s and n_links:
                out[round(h, 3)] = 2 * (h - s_sum) / n_links
            s_sum, n_s = 0.0, 0
    if n_links is None:
        print(f'warning: no --grid line in {path}, skipped', file=sys.stderr)
    return out


def chroma_draws(path):
    text = open(path, errors='replace').read()
    m = PAT_NROW.search(text)
    if not m:
        print(f'warning: no <nrow> in {path}, skipped', file=sys.stderr)
        return {}
    n_links = links(m.groups())
    return {float(k): 4 + float(k) / n_links for k in PAT_KE.findall(text)}


if len(sys.argv) < 2:
    sys.exit(__doc__)
draws = {'Grid': {}, 'Chroma': {}}
for path in sys.argv[1:]:
    if path.endswith(('.xml', '.xmldat')):
        draws['Chroma'].update(chroma_draws(path))
    else:
        draws['Grid'].update(grid_draws(path))
for code, d in draws.items():
    v = list(d.values())
    if not v:
        continue
    mean = sum(v) / len(v)
    err = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1) / len(v)) if len(v) > 1 else float('nan')
    print(f'{code}: {len(v)} distinct draws, <|P|^2> per link = {mean:.5f} +- {err:.5f}'
          f' (draws {min(v):.4f} to {max(v):.4f})')
