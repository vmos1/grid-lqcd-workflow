#!/usr/bin/env python3
"""Cost of a Hasenbusch ladder in nested step layouts, from its 8-draw force-only run (tune 3).

Per-call force times are read from a force-only run's log (FORCES_ONLY lines, samples >= 1, L014)
and are layout-independent; force seconds per trajectory = sum over pieces of t_call x (3 N + 1),
N the steps of the piece's level (2026_10_07_baseg_force_breakdown_trajl1.md). Two wall models:
  tune 2 (10-27):  wall = 1.47 x force
  tune 5 (11-04):  wall = 1.217 x force + 1.387 s x smeared calls + 364.1 s   (73 runs, LOO 2.7%)
  smeared calls = sum of (3 N + 1) over the distinct step counts of the light, strange and tail
  levels (the batched smearing pullback runs once per force call on each such level).
Risk column: force x step of each piece relative to the SAME piece of the baseline ladder at the
baseline's step (like with like, tune 1): ratio = (F / F_base) x (N_base / N). Rung forces are the
8-draw mean of the per-draw peak; the tail, strange and gauge likewise. Python 3.6, stdlib only.

Usage:
  python3 ladder_layout_cost.py --fo base=2026_10_16_fo8_base --fo B2=2026_10_16_fo8_B2 \
      --layout 12/12/48/48 --layout 8/16/32/64 [--sweep] [--runs DIR]
The first --fo is the reference ladder; its 12/12/48/48 is the 100% of every percentage.
"""
import argparse
import itertools
import re
from collections import defaultdict

RUNS = '/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/runs'
FO_RE = re.compile(r'FORCES_ONLY level (\S+) sample (\d+) avg=(\S+) max=(\S+) refresh=(\S+) time=(\S+)')
GROUPS = ('light', 'strange', 'tail', 'gauge')
BASE_LAY = {'light': 12, 'strange': 12, 'tail': 48, 'gauge': 48}
K2 = 1.47                               # tune 2 wall / force
T5 = (1.208, 1.415, 377.8)              # tune 5 wall model, refit 11-04 on 76 runs (tune5.py price)


def group_of(name):
    if name == 'Strange':
        return 'strange'
    if name == 'LightSchurPF':
        return 'tail'
    if name == 'Gauge':
        return 'gauge'
    return 'light'


def parse_fo(rundir):
    ladder, acc = None, defaultdict(lambda: defaultdict(list))
    for line in open(rundir + '/hmc.log', errors='replace'):
        if ladder is None and line.startswith('ENV FORCES48_PUREGRID'):
            ladder = [float(x) for x in re.search(r' LADDER=(\S+)', line).group(1).split(',')]
        m = FO_RE.search(line)
        if m and int(m.group(2)) >= 1:
            acc[m.group(1)]['rms'].append(float(m.group(3)))
            acc[m.group(1)]['peak'].append(float(m.group(4)))
            acc[m.group(1)]['t'].append(float(m.group(6)))
    out = {}
    for lv, kv in acc.items():
        n = len(kv['t'])
        out[lv] = dict(rms=sum(kv['rms']) / n, peak=sum(kv['peak']) / n, t=sum(kv['t']) / n, n=n)
    return ladder, out


def calls(n):
    return 3 * n + 1


def force_time(pc, lay):
    per = dict((g, 0.0) for g in GROUPS)
    for name, v in pc.items():
        per[group_of(name)] += calls(lay[group_of(name)]) * v['t']
    return sum(per.values()), per


def smeared_calls(lay):
    return sum(calls(n) for n in set(lay[g] for g in ('light', 'strange', 'tail')))


def walls(force, lay):
    return K2 * force, T5[0] * force + T5[1] * smeared_calls(lay) + T5[2]


def parse_layout(s):
    v = [int(x) for x in s.split('/')]
    return dict(zip(GROUPS, v))


def fmt_lay(lay):
    return '/'.join(str(lay[g]) for g in GROUPS)


def nests(lay):
    st = sorted(set(lay.values()))
    for i in range(1, len(st)):
        if st[i] % st[i - 1] or (st[i] // st[i - 1]) % 2:
            return False
    return lay['light'] <= lay['strange'] <= lay['tail'] <= lay['gauge']


def knobs(lay):
    steps = sorted(set(lay[g] for g in GROUPS))
    idx = dict((g, steps.index(lay[g])) for g in GROUPS)
    mult = [steps[k] // steps[k - 1] // 2 for k in range(1, len(steps))]
    return 'L/S/T/G=%d/%d/%d/%d MULT=%s MDSTEPS=%d' % (
        idx['light'], idx['strange'], idx['tail'], idx['gauge'],
        ','.join(str(m) for m in mult) or '-', steps[0])


def rung_rows(ladder, pc):
    return [('PF%d' % i, ladder[i], ladder[i + 1], pc['PF%d' % i]) for i in range(len(ladder) - 1)]


def risk(ladder, pc, lay, bl_ladder, bl_pc):
    """Force x step ratios to the baseline's counterpart at the baseline's step: worst rung, rung 1,
    tail, strange (peak-based)."""
    bl = rung_rows(bl_ladder, bl_pc)
    bl_worst = max(r[3]['peak'] for r in bl)
    rr = rung_rows(ladder, pc)
    s = 12.0 / lay['light']
    worst = max(r[3]['peak'] for r in rr) / bl_worst * s
    rung1 = rr[0][3]['peak'] / bl[0][3]['peak'] * s
    tail = pc['LightSchurPF']['peak'] / bl_pc['LightSchurPF']['peak'] * 48.0 / lay['tail']
    strange = pc['Strange']['peak'] / bl_pc['Strange']['peak'] * 12.0 / lay['strange']
    gauge = 48.0 / lay['gauge']
    return worst, rung1, tail, strange, gauge


def sweep_layouts(light_min=8, light_max=12, strange_min=10, tail_min=24, gauge_min=48, gauge_max=128):
    for nl in range(light_min, light_max + 1):
        for rs, rt, rg in itertools.product((1, 2, 4), (1, 2, 4), (1, 2, 4)):
            ns = nl * rs
            nt = ns * rt
            ng = nt * rg
            lay = {'light': nl, 'strange': ns, 'tail': nt, 'gauge': ng}
            if ns < strange_min or nt < tail_min or ng < gauge_min or ng > gauge_max or ns > 24:
                continue
            if len(set(lay.values())) > 4:
                continue
            yield lay


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', default=RUNS)
    ap.add_argument('--fo', action='append', required=True, help='LABEL=run_dir_name (first = reference)')
    ap.add_argument('--layout', action='append', default=[], help='L/S/T/G (repeatable)')
    ap.add_argument('--sweep', action='store_true', help='every nested layout, light 8-12, strange >= 10, tail >= 24, gauge 48-128')
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    lads = []
    for spec in args.fo:
        label, run = spec.split('=', 1)
        ladder, pc = parse_fo(args.runs + '/' + run)
        lads.append((label, run, ladder, pc))
    ref_label, ref_run, ref_ladder, ref_pc = lads[0]
    ref_force, _ = force_time(ref_pc, BASE_LAY)
    ref_w2, ref_w5 = walls(ref_force, BASE_LAY)

    print('## Per-call force times (s), 8-draw force-only, samples >= 1')
    print()
    names = ['PF0', 'PF1', 'PF2', 'PF3', 'PF4', 'LightLogDet', 'StrangeLogDet', 'Strange', 'LightSchurPF', 'Gauge']
    names = [n for n in names if any(n in pc for _, _, _, pc in lads)]
    print('| ladder | masses | ' + ' | '.join(names) + ' | force at 12/12/48/48 |')
    print('|---|---|' + '---|' * len(names) + '---|')
    for label, run, ladder, pc in lads:
        f, _ = force_time(pc, BASE_LAY)
        print('| %s | %s | %s | %.0f s |' % (label, ','.join('%+.4f' % m for m in ladder),
                                             ' | '.join('%.2f' % pc[n]['t'] if n in pc else '-' for n in names), f))
    print()
    print('## Peak forces (mean of per-draw peaks) and RMS, like with like')
    print()
    print('| ladder | rung 1 peak (RMS) | worst rung peak (RMS) | tail peak (RMS) | strange peak | gauge peak |')
    print('|---|---|---|---|---|---|')
    for label, run, ladder, pc in lads:
        rr = rung_rows(ladder, pc)
        w = max(rr, key=lambda r: r[3]['peak'])
        print('| %s | %.3f (%.4f) | %s %.3f (%.4f) | %.3f (%.3f) | %.3f | %.3f |' % (
            label, rr[0][3]['peak'], rr[0][3]['rms'], w[0], w[3]['peak'], w[3]['rms'],
            pc['LightSchurPF']['peak'], pc['LightSchurPF']['rms'], pc['Strange']['peak'], pc['Gauge']['peak']))
    print()

    def row(label, ladder, pc, lay):
        f, per = force_time(pc, lay)
        w2, w5 = walls(f, lay)
        wr, r1, tl, st, gg = risk(ladder, pc, lay, ref_ladder, ref_pc)
        return (label, fmt_lay(lay), f, 100 * (f / ref_force - 1), w2, 100 * (w2 / ref_w2 - 1), w5, 100 * (w5 / ref_w5 - 1),
                per['light'], per['strange'], per['tail'], per['gauge'], wr, r1, tl, st, gg, knobs(lay))

    hdr = ('| ladder | layout L/S/T/G | force s | vs ref | wall (1.47x) | vs ref | wall (tune 5) | vs ref | light | strange | tail | gauge '
           '| worst rung x | rung 1 x | tail x | strange x | gauge x | knobs |')
    sep = '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'
    fmt = '| %s | %s | %.0f | %+.1f%% | %.0f | %+.1f%% | %.0f | %+.1f%% | %.0f | %.0f | %.0f | %.0f | %.2f | %.2f | %.2f | %.2f | %.2f | %s |'
    print('## Cost per layout; reference = %s at 12/12/48/48: force %.0f s, wall %.0f s (1.47x) / %.0f s (tune 5)' % (
        ref_label, ref_force, ref_w2, ref_w5))
    print('force x step ratios ("x" columns) are to the same piece of the reference at its step; > 1 = coarser than the reference ever ran that piece')
    print()
    if args.layout:
        print(hdr)
        print(sep)
        for label, run, ladder, pc in lads:
            for s in args.layout:
                lay = parse_layout(s)
                if not nests(lay):
                    print('| %s | %s | invalid nesting |' % (label, s))
                    continue
                print(fmt % row(label, ladder, pc, lay))
        print()
    if args.sweep:
        for label, run, ladder, pc in lads:
            rows = [row(label, ladder, pc, lay) for lay in sweep_layouts()]
            rows.sort(key=lambda r: r[6])
            print('### %s: every nested layout, ranked by the tune 5 wall (top %d)' % (label, args.top))
            print()
            print(hdr)
            print(sep)
            for r in rows[:args.top]:
                print(fmt % r)
            print()


if __name__ == '__main__':
    main()
