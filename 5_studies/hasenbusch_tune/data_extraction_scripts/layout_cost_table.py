#!/usr/bin/env python3
"""Cost and risk of every nested step layout for the baseline ladder, by arithmetic.

Tune 2 (2026_10_20_handoff_tune2_vary_steps.md). No runs: every number here is derived
from measured per-call times and from the step scans already made. Python 3.6 compatible.

Layout = steps per trajectory of light / strange / tail / gauge. A nested level runs
2 x multiplier x its parent's steps, so adjacent levels differ by an even integer (2, 4, 6,
8) or share a level (ratio 1). Monomials are ordered light <= strange <= tail <= gauge.

Cost model (2026_10_07_baseg_force_breakdown_trajl1.md, TRAJL 1.0, four 40 GB nodes):
  force calls per level = 3N + 1 (force-gradient); per-call times from the force-only run
  (PF0 10.18 s ... gauge 0.15 s); at 12/12/48/48 the pieces give 2,424 s.
  wall = force time + NONFORCE (MG rebuilds, cache copies, updates; T24: 3,166 - 2,048 = 1,118 s,
  baseline ~1,150-1,230 s). Wall savings are therefore ~2/3 of force-time savings.

dH model (pieces add; spreads add in quadrature; shifts add to the mean):
  gauge: momentum-independent SHIFT, exactly h^4, +0.0604 at 48 (2026_10_12_pure_gauge_fg_step_scan.md).
  tail (-0.1870): SPREAD sigma 0.47 at 24 (T24: RMS 0.51 over 3, baseline 0.195 removed in
        quadrature; 2026_10_19_force_only_candidates_like_with_like.md), scaled as h^4 (marginal
        for the tail: 2026_10_14_stiffness_screen_results.md).
  light rungs (baseline ladder): SPREAD sigma 0.43 at 6 without the control's +2.06 spike, 0.77
        with it (control at 6, RMS 0.47 / 0.79 over 10), scaled as h^4 (NOT established for the
        light rungs). Both are shown; the measured control at 6 (+5% per accepted trajectory)
        sits between them (-5% / +11%), so rows are ranked by the mean of the two. Light 6 itself
        is excluded: measured failure. cfg_2000 is a quiet configuration for rung 1
        (2026_10_26_force_only_config_spread.md), so the light piece is a lower bound.
  residual 0.191: the baseline's 0.195 with the light(12) and tail(48) pieces removed; holds the
        strange at 12, the log-dets and cross terms. Kept unchanged whatever the strange step, which
        is OPTIMISTIC for strange < 12 (untested) and pessimistic for strange > 12.
  acceptance = erfc(sigma / 2 sqrt 2) * exp(-(gauge shift - 0.0604)), capped at 1
        (the baseline's 0.195 gives 0.92 by this formula; measured 20/20, expected 0.94).
  cost per accepted trajectory = wall / acceptance; the headline column.

Per-piece detail (--detail): force x step per monomial = the 10-07 force-only value at the
baseline's step scaled by (baseline step / new step), since the ladder and so the forces are
unchanged; linear-range ratio = per-step link motion h |P|_max = 9.63 / N over the monomial's
linear range from the stiffness screen (gauge 1.9, log-dets 1.5, strange 0.67, tail 0.28,
PF3 0.13, PF2 0.055, PF1 0.015, PF0 0.0098; ratio > 1 = beyond the regime where an h^4
expansion means anything; the light rungs are always far beyond and still work at 12).

Usage: python3 layout_cost_table.py [--all] [--min-saving PCT] [--top N] [--light-min 6]
                                    [--csv FILE] [--detail 8/8/32/128,9/9/36/72,...]
"""
import argparse
import csv
import itertools
import math
import sys

# piece: (group, per-call time s, force x step avg at base, force x step peak at base, linear range)
PIECES = [
    ('PF0 -0.2416/-0.2400', 'light', 10.18, 0.0003, 0.029, 0.0098),
    ('PF1 -0.2400/-0.2320', 'light', 7.01, 0.001, 0.054, 0.015),
    ('PF2 -0.2320/-0.2180', 'light', 5.64, 0.001, 0.044, 0.055),
    ('PF3 -0.2180/-0.1870', 'light', 7.43, 0.002, 0.033, 0.13),
    ('light log-det', 'light', 1.10, 0.005, 0.014, 1.5),
    ('strange log-det', 'light', 1.02, 0.003, 0.007, 1.5),
    ('strange', 'strange', 12.04, 0.051, 0.197, 0.67),
    ('tail -0.1870', 'tail', 5.22, 0.028, 0.134, 0.28),
    ('gauge', 'gauge', 0.15, 0.143, 0.216, 1.9),
]
BASE = {'light': 12, 'strange': 12, 'tail': 48, 'gauge': 48}
PC = {g: sum(p[2] * (3 * BASE[g] + 1) for p in PIECES if p[1] == g) for g in BASE}   # s per trajectory at base
NONFORCE = 1150.0
GAUGE_SHIFT_48 = 0.0604
TAIL_SIGMA_24 = 0.47
LIGHT_SIGMA_6 = 0.43
LIGHT_SIGMA_6_SPIKE = 0.77
RESIDUAL = 0.191
P_MAX = 9.63                  # |P|_max per link, so per-step link motion = P_MAX / N
RATIOS = (1, 2, 4, 6, 8)      # 1 = shares the parent's level
TAIL_MIN, GAUGE_MIN = 32, 32  # tail at 24 and gauge at 24 both fail; 32 is the first step above
STRANGE_MAX, TAIL_MAX, GAUGE_MAX = 48, 96, 192
ORDER = ('light', 'strange', 'tail', 'gauge')


def calls(n):
    return 3 * n + 1


def force_time(lay):
    return {g: PC[g] * calls(lay[g]) / calls(BASE[g]) for g in BASE}


def gauge_shift(n):
    return GAUGE_SHIFT_48 * (48.0 / n) ** 4


def tail_sigma(n):
    return TAIL_SIGMA_24 * (24.0 / n) ** 4


def light_sigma(n, spike=False):
    return (LIGHT_SIGMA_6_SPIKE if spike else LIGHT_SIGMA_6) * (6.0 / n) ** 4


def acceptance(sigma, shift):
    return min(1.0, math.erfc(sigma / (2 * math.sqrt(2))) * math.exp(-(shift - GAUGE_SHIFT_48)))


def evaluate(lay):
    ft = force_time(lay)
    force = sum(ft.values())
    wall = force + NONFORCE
    st, sg = tail_sigma(lay['tail']), gauge_shift(lay['gauge'])
    out = dict(lay=lay, ft=ft, force=force, wall=wall, st=st, sg=sg,
               levels=len(set(lay.values())), strange_untested=lay['strange'] < 12)
    for tag, spike in (('', False), ('_spike', True)):
        sl = light_sigma(lay['light'], spike)
        sigma = math.sqrt(RESIDUAL ** 2 + sl ** 2 + st ** 2)
        acc = acceptance(sigma, sg)
        out['sl' + tag], out['sigma' + tag], out['acc' + tag], out['per_acc' + tag] = sl, sigma, acc, wall / acc
    out['per_acc_mean'] = 0.5 * (out['per_acc'] + out['per_acc_spike'])
    return out


def layouts(light_min):
    for nl in range(light_min, 13):
        for rs, rt, rg in itertools.product(RATIOS, RATIOS, RATIOS):
            ns, nt = nl * rs, nl * rs * rt
            ng = nt * rg
            if ns > STRANGE_MAX or nt < TAIL_MIN or nt > TAIL_MAX or ng < GAUGE_MIN or ng > GAUGE_MAX:
                continue
            lay = {'light': nl, 'strange': ns, 'tail': nt, 'gauge': ng}
            if len(set(lay.values())) > 4:
                continue
            yield lay


def knobs(lay):
    """CUSTOM_LEVEL_* values: level index per group, multipliers between consecutive levels, MDSTEPS."""
    steps = sorted(set(lay[g] for g in ORDER))
    idx = {g: steps.index(lay[g]) for g in ORDER}
    mult = []
    for k in range(1, len(steps)):
        r = steps[k] / steps[k - 1]
        if r != int(r) or int(r) % 2:
            return None
        mult.append(int(r) // 2)
    return dict(idx=idx, mult=mult, mdsteps=steps[0])


def parse(s):
    nl, ns, nt, ng = (int(x) for x in s.split('/'))
    return {'light': nl, 'strange': ns, 'tail': nt, 'gauge': ng}


def fmt(lay):
    return '%d/%d/%d/%d' % (lay['light'], lay['strange'], lay['tail'], lay['gauge'])


def pct(x, ref):
    return 100 * (x / ref - 1)


def print_detail(lays):
    base = evaluate(dict(BASE))
    cols = [dict(BASE)] + lays
    head = '| piece (level steps) | ' + ' | '.join(fmt(l) for l in cols) + ' |'
    sep = '|---|' + '---|' * len(cols)
    print('### Force x step, peak (avg), per monomial; x = ratio to the same piece in the baseline')
    print()
    print(head)
    print(sep)
    for name, g, tcall, fs_avg, fs_pk, lin in PIECES:
        cells = []
        for l in cols:
            r = BASE[g] / l[g]
            cells.append('%.3f (%.4f) x%.2f @%d' % (fs_pk * r, fs_avg * r, r, l[g]))
        print('| %s | %s |' % (name, ' | '.join(cells)))
    print()
    print('### Per-step link motion over the linear range (> 1 = beyond the h^4 regime); strange, tail, gauge are the meaningful rows')
    print()
    print(head)
    print(sep)
    for name, g, tcall, fs_avg, fs_pk, lin in PIECES:
        if g == 'light' and 'log-det' not in name:
            continue
        print('| %s | %s |' % (name, ' | '.join('%.2f @%d' % (P_MAX / l[g] / lin, l[g]) for l in cols)))
    print()
    print('### Force time per trajectory, s (calls = 3N + 1 per level)')
    print()
    print(head)
    print(sep)
    for name, g, tcall, fs_avg, fs_pk, lin in PIECES:
        print('| %s | %s |' % (name, ' | '.join('%.0f (%d calls)' % (tcall * calls(l[g]), calls(l[g])) for l in cols)))
    print('| **total force** | %s |' % ' | '.join('**%.0f** (%+.0f%%)' % (evaluate(l)['force'], pct(evaluate(l)['force'], base['force'])) for l in cols))
    print('| wall (force + %.0f) | %s |' % (NONFORCE, ' | '.join('%.0f' % evaluate(l)['wall'] for l in cols)))
    print()
    print('### dH pieces by the model, acceptance, cost per accepted trajectory, and the launcher knobs')
    print()
    print(head)
    print(sep)
    rows = [('gauge shift (exact h^4)', lambda e: '%+.3f' % e['sg']),
            ('tail sigma (h^4 from T24)', lambda e: '%.3f' % e['st']),
            ('light sigma, no spike (spike)', lambda e: '%.3f (%.3f)' % (e['sl'], e['sl_spike'])),
            ('strange', lambda e: 'untested (%d)' % e['lay']['strange'] if e['strange_untested'] else 'ok (%d)' % e['lay']['strange']),
            ('sigma total, no spike (spike)', lambda e: '%.3f (%.3f)' % (e['sigma'], e['sigma_spike'])),
            ('acceptance, no spike (spike)', lambda e: '%.2f (%.2f)' % (e['acc'], e['acc_spike'])),
            ('per accepted, s', lambda e: '%.0f (%.0f)' % (e['per_acc'], e['per_acc_spike'])),
            ('vs baseline', lambda e: '%+.0f%% (%+.0f%%)' % (pct(e['per_acc'], base['per_acc']), pct(e['per_acc_spike'], base['per_acc_spike'])))]
    for label, f in rows:
        print('| %s | %s |' % (label, ' | '.join(f(evaluate(l)) for l in cols)))
    kn = []
    for l in cols:
        k = knobs(l)
        kn.append('L/S/T/G=%d/%d/%d/%d MULT=%s MDSTEPS=%d' % (k['idx']['light'], k['idx']['strange'], k['idx']['tail'], k['idx']['gauge'],
                                                           ','.join(str(m) for m in k['mult']) or '-', k['mdsteps']) if k else 'invalid')
    print('| CUSTOM_LEVEL_* | %s |' % ' | '.join(kn))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--all', action='store_true', help='print every layout, not only the savers')
    ap.add_argument('--min-saving', type=float, default=8.0,
                    help='keep layouts whose mean cost per accepted trajectory beats the baseline by at least this %% (default 8)')
    ap.add_argument('--top', type=int, default=25, help='rows to print (default 25)')
    ap.add_argument('--light-min', type=int, default=7, help='smallest light step to enumerate (default 7; 6 is a measured failure)')
    ap.add_argument('--csv', help='write every layout to this CSV')
    ap.add_argument('--detail', help='comma-separated layouts L/S/T/G for the per-piece tables (baseline is always first)')
    args = ap.parse_args()

    if args.detail:
        print_detail([parse(s) for s in args.detail.split(',')])
        return

    base = evaluate(dict(BASE))
    rows = [evaluate(l) for l in layouts(args.light_min)]
    rows.sort(key=lambda r: r['per_acc_mean'])
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['layout', 'levels', 'force_s', 'force_vs_base_pct', 'wall_s', 'gauge_shift',
                        'sigma_tail', 'sigma_light', 'sigma_light_spike', 'acceptance', 'acceptance_spike',
                        'per_accepted_s', 'per_accepted_spike_s', 'vs_base_pct', 'vs_base_spike_pct',
                        'strange_untested'])
            for r in rows:
                w.writerow([fmt(r['lay']), r['levels'], '%.0f' % r['force'], '%+.1f' % pct(r['force'], base['force']),
                            '%.0f' % r['wall'], '%.3f' % r['sg'], '%.3f' % r['st'], '%.3f' % r['sl'],
                            '%.3f' % r['sl_spike'], '%.3f' % r['acc'], '%.3f' % r['acc_spike'],
                            '%.0f' % r['per_acc'], '%.0f' % r['per_acc_spike'],
                            '%+.1f' % pct(r['per_acc'], base['per_acc']),
                            '%+.1f' % pct(r['per_acc_spike'], base['per_acc_spike']),
                            int(r['strange_untested'])])
        print('wrote', args.csv, len(rows), 'layouts', file=sys.stderr)

    print('baseline %s: force %.0f s, wall %.0f s, sigma %.3f, acceptance %.3f, per accepted %.0f s; %d layouts enumerated'
          % (fmt(BASE), base['force'], base['wall'], base['sigma'], base['acc'], base['per_acc'], len(rows)))
    print('model: wall = force + %.0f s; sigma_light %.2f (%.2f with spike) x (6/N)^4, sigma_tail %.2f x (24/N)^4,'
          ' residual %.3f; gauge shift %.4f x (48/N)^4; strange < 12 untested (residual kept, optimistic);'
          ' ranked by the mean of the two per-accepted costs'
          % (NONFORCE, LIGHT_SIGMA_6, LIGHT_SIGMA_6_SPIKE, TAIL_SIGMA_24, RESIDUAL, GAUGE_SHIFT_48))
    print()
    print('| rank | layout L/S/T/G | levels | force s | vs base | wall s | gauge shift | σ tail | σ light (spike) | accept. (spike) | per accepted vs base (spike) | strange |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    n = 0
    for r in rows:
        saving = -pct(r['per_acc_mean'], base['per_acc'])
        if not args.all and saving < args.min_saving:
            continue
        n += 1
        print('| %d | %s | %d | %.0f | %+.0f%% | %.0f | %+.3f | %.3f | %.3f (%.3f) | %.2f (%.2f) | %+.0f%% (%+.0f%%) | %s |'
              % (n, fmt(r['lay']), r['levels'], r['force'], pct(r['force'], base['force']), r['wall'], r['sg'],
                 r['st'], r['sl'], r['sl_spike'], r['acc'], r['acc_spike'],
                 pct(r['per_acc'], base['per_acc']), pct(r['per_acc_spike'], base['per_acc_spike']),
                 'untested (%d)' % r['lay']['strange'] if r['strange_untested'] else 'ok'))
        if n >= args.top:
            break
    print()
    print('%d of %d layouts at or above %.0f%% saving; the rest are below it' % (
        sum(1 for r in rows if -pct(r['per_acc_mean'], base['per_acc']) >= args.min_saving), len(rows), args.min_saving))


if __name__ == '__main__':
    main()
