#!/usr/bin/env python3
"""FORCES_ONLY candidate scoring: screened force -> predicted real F*dt -> trajectory cost.

``forces_only_tables.py`` reports what a FORCES_ONLY run measured.  This script
answers the next question -- *is the candidate worth running for real* -- which
needs three things the raw table does not have:

1. **Calibration.**  FORCES_ONLY and a real trajectory do not rank the rungs the
   same way.  On the C2 ladder the screen calls PF2 the peak (0.566) while the
   confirmed 20-trajectory run calls it PF0 (0.116).  All light rungs share one
   ``dt``, so this is a genuine method disagreement, not ``dt`` bookkeeping.  The
   fix is a per-rung factor ``k = realFdt / screenedF`` taken from a ladder that
   has BOTH a FORCES_ONLY log and a validated trajectory run.  Ranking on raw
   screened force scored a 49% regression as a 1% one (C2 round 1, y1b).

2. **Quantisation.**  The calibrated peak is a HEADROOM metric.  The only channel
   from headroom to wall time is the integer MDSTEPS, so a candidate that lowers
   the peak by 10% but does not cross a step-count boundary is worth exactly 0 s.
   Every C2 round-2 candidate was in that position.

3. **Trajectory cost model.**  ``t(m) = a + b*m``, calibrated from two measured
   points of a real run.  All integrator levels are locked multiples of MDSTEPS
   (``strange = 2*sm*MDSTEPS``, ``gauge = 2*gm*strange``), so they coarsen
   together and the whole trajectory scales with ``m``.

DEFAULTS ARE C2-SPECIFIC.  ``k``, the baseline peak and ``t(m)`` all come from
w3@MDSTEPS=5 (`runs/2026_8_10_3level_w3_mdsteps5_48/`, 20 trajectories, 19/20
accepted).  They are NOT portable to another ladder family -- a 4-rung light
ladder on a different base mass needs its own calibration.  Pass ``--k`` or
``--real-fdt`` + ``--calib-log`` to recalibrate; the script prints which
calibration it used on every run so a stale one cannot pass unnoticed.

Usage::

    # score candidates (dirs holding scan_<dirname>.log, or logs directly)
    forces_only_ladder_cost.py runs/.../y2a runs/.../y2b ...

    # recalibrate from a different validated run
    forces_only_ladder_cost.py --calib-log runs/.../w3/scan_w3.log \\
        --real-fdt PF0=0.116,PF1=0.093,PF2=0.092,PF3=0.061 <cands>

    # power-law exponents F_PFk ~ gap_k^a, fitted across the candidates given
    forces_only_ladder_cost.py --exponents <cands>

``--exponents`` sorts candidates by each rung's own gap (taken from the ladder
in the log's ENV header) and reports adjacent-pair exponents -- the fit that
drives candidate generation each round, previously done by hand.
"""

import argparse
import math
import os
import re
import sys

# ── C2 defaults, provenance in the module docstring ──────────────────────────
K_DEFAULT = {'PF0': 0.2306, 'PF1': 0.3019, 'PF2': 0.1625, 'PF3': 0.1432}
BASELINE_PEAK = 0.1180        # w3 (y1ctl) calibrated peak
REF_MDSTEPS = 5               # MDSTEPS at which the real F*dt was measured
TRAJ_A, TRAJ_B = 152.0, 369.0  # t(m) = a + b*m; t(5)=1998 s measured

SUM_RE = r'FORCES_ONLY samples=(\d+) %s: (.*)'
SAMPLE_RE = re.compile(
    r'FORCES_ONLY level (\S+) sample (\d+) .*?avg=([-\d.e+]+) max=([-\d.e+]+)'
    r'(?: refresh=([-\d.e+]+))?(?: time=([-\d.e+]+))?')
KV_RE = re.compile(r'(\w+)=([-\d.e+]+)')
TERMINAL_OK = (0, 134, 143)   # 134/143 = cosmetic QUDA teardown abort


def log_path(arg):
    """Accept either a candidate dir (holding scan_<dirname>.log) or a log."""
    if os.path.isdir(arg):
        name = os.path.basename(os.path.normpath(arg))
        return name, os.path.join(arg, 'scan_%s.log' % name)
    name = os.path.basename(arg)
    name = re.sub(r'^scan_|\.log$', '', name)
    return name, arg


def parse(path):
    """-> dict with avg/max/refresh/time per level, ladder, samples, exit."""
    try:
        txt = open(path, errors='replace').read()
    except OSError as e:
        return {'error': str(e)}
    out = {'ladder': None, 'n': 0, 'exit': None, 'partial': False}
    m = re.search(r'HASEN_LADDER=(\S+)', txt)
    if m:
        out['ladder'] = [float(x) for x in m.group(1).split(',')]
    m = re.search(r'^exit=(\d+)', txt, re.M)
    if m:
        out['exit'] = int(m.group(1))

    for kind in ('avg', 'max', 'refresh', 'time'):
        m = re.search(SUM_RE % kind, txt)
        if m:
            out['n'] = int(m.group(1))
            out[kind] = {k: float(v) for k, v in KV_RE.findall(m.group(2))}

    if 'max' not in out:                      # no summary line -> mean the samples
        acc = {}
        for lvl, s, a, mx, rf, tm in SAMPLE_RE.findall(txt):
            d = acc.setdefault(lvl, {'avg': [], 'max': [], 'refresh': [], 'time': []})
            d['avg'].append(float(a)); d['max'].append(float(mx))
            if rf:
                d['refresh'].append(float(rf))
            if tm:
                d['time'].append(float(tm))
        if not acc:
            return {'error': 'no FORCES_ONLY output in %s' % path}
        out['partial'] = True
        out['n'] = max(len(d['max']) for d in acc.values())
        for kind in ('avg', 'max', 'refresh', 'time'):
            out[kind] = {lvl: (sum(d[kind])/len(d[kind]) if d[kind] else 0.0)
                         for lvl, d in acc.items()}
    return out


def rungs_of(d):
    return sorted((l for l in d.get('max', {}) if re.fullmatch(r'PF\d+', l)),
                  key=lambda s: int(s[2:]))


def gap(d, rung):
    """gap_k = ladder[k+1] - ladder[k]; PFk's force is driven by it."""
    k = int(rung[2:])
    L = d.get('ladder')
    if not L or k + 1 >= len(L):
        return None
    return L[k+1] - L[k]


def score(d, K, baseline, safe, ref_m, a, b):
    cal = {r: d['max'][r] * K[r] for r in rungs_of(d) if r in K}
    if not cal:
        return None
    peak = max(cal.values())
    who = max(cal, key=cal.get)
    m = max(1, math.ceil(round(peak * ref_m / safe, 9)))
    t = a + b * m
    t_ref = a + b * ref_m
    return dict(cal=cal, peak=peak, who=who, mdsteps=m, traj=t,
                d_peak=100*(peak-baseline)/baseline, d_wall=100*(t-t_ref)/t_ref)


def validity(d):
    if 'error' in d:
        return 'ERROR: ' + d['error']
    bits = []
    if d['partial']:
        bits.append('NO SUMMARY (mean of %d samples)' % d['n'])
    elif d['n'] != 8:
        bits.append('%d samples' % d['n'])
    if d['exit'] is not None and d['exit'] not in TERMINAL_OK:
        bits.append('exit=%d' % d['exit'])
    if not bits:
        note = 'VALID'
        if d['exit'] in (134, 143):
            note += ' (exit=%d cosmetic)' % d['exit']
        return note
    return 'CHECK: ' + '; '.join(bits)


def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cands', nargs='+', help='candidate dirs or FORCES_ONLY logs')
    p.add_argument('--k', help='per-rung calibration, e.g. PF0=0.2306,PF1=0.3019,...')
    p.add_argument('--real-fdt', help='real per-rung F*dt from a validated traj run')
    p.add_argument('--calib-log', help='FORCES_ONLY log of that same ladder')
    p.add_argument('--baseline', help='candidate name to use as the reference row')
    p.add_argument('--baseline-peak', type=float, default=BASELINE_PEAK)
    p.add_argument('--safe', type=float, default=None,
                   help='max tolerable real F*dt on any light rung '
                        '(default: the baseline peak, so the reference maps to '
                        '--ref-mdsteps exactly)')
    p.add_argument('--ref-mdsteps', type=int, default=REF_MDSTEPS)
    p.add_argument('--traj-model', default='%g,%g' % (TRAJ_A, TRAJ_B),
                   help='a,b in t(m)=a+b*m  [default %(default)s]')
    p.add_argument('--exponents', action='store_true',
                   help='fit F_PFk ~ gap_k^a across the candidates given')
    args = p.parse_args(argv)

    K = dict(K_DEFAULT)
    calib_note = 'C2 default (w3@MDSTEPS=5, 20 traj)'
    if args.k:
        K = {k: float(v) for k, v in KV_RE.findall(args.k)}
        calib_note = 'explicit --k'
    elif args.real_fdt or args.calib_log:
        if not (args.real_fdt and args.calib_log):
            p.error('--real-fdt and --calib-log must be given together')
        real = {k: float(v) for k, v in KV_RE.findall(args.real_fdt)}
        cd = parse(log_path(args.calib_log)[1])
        if 'error' in cd:
            p.error(cd['error'])
        K = {r: real[r]/cd['max'][r] for r in real if r in cd.get('max', {})}
        calib_note = 'derived from %s' % os.path.basename(args.calib_log)

    a, b = (float(x) for x in args.traj_model.split(','))
    rows = []
    for c in args.cands:
        name, path = log_path(c)
        rows.append((name, parse(path)))

    baseline_peak = args.baseline_peak
    if args.baseline:
        for n, d in rows:
            if n == args.baseline and 'error' not in d:
                s = score(d, K, baseline_peak, baseline_peak, args.ref_mdsteps, a, b)
                if s:
                    baseline_peak = s['peak']
    safe = args.safe if args.safe is not None else baseline_peak

    print('# calibration: %s' % calib_note)
    print('#   k = %s' % ', '.join('%s=%.4f' % (r, K[r]) for r in sorted(K)))
    print('#   baseline peak %.4f @ MDSTEPS=%d ; safe F*dt <= %.4f ; t(m) = %g + %g*m'
          % (baseline_peak, args.ref_mdsteps, safe, a, b))
    if calib_note != 'C2 default (w3@MDSTEPS=5, 20 traj)' and not args.baseline:
        print('# WARNING: k was recalibrated but --baseline was not given, so the '
              'baseline peak\n#   above is still the C2 default. Pass --baseline '
              '<name> (the reference\n#   candidate, included in the argument list) '
              'so both come from the same run.')
    print()

    allr = sorted({r for _, d in rows for r in rungs_of(d)},
                  key=lambda s: int(s[2:]))
    hdr = (['run'] + ['%s avg/max' % r for r in allr]
           + ['deriv s', 'refresh s', 'cal. peak', 'vs base', 'min MDSTEPS',
              'est. traj s', 'vs base wall', 'valid'])
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '---|' * len(hdr))
    for name, d in rows:
        if 'error' in d:
            print('| %s |%s %s |' % (name, ' — |' * (len(hdr)-2), validity(d)))
            continue
        s = score(d, K, baseline_peak, safe, args.ref_mdsteps, a, b)
        cells = ['%.4f/%.3f' % (d['avg'][r], d['max'][r]) if r in d['max'] else '—'
                 for r in allr]
        dsum = sum(d.get('time', {}).values())
        rsum = sum(d.get('refresh', {}).values())
        print('| %s | %s | %.1f | %.1f | %.4f (%s) | %+.1f%% | %d | %.0f | %+.1f%% | %s |'
              % (name, ' | '.join(cells), dsum, rsum, s['peak'], s['who'],
                 s['d_peak'], s['mdsteps'], s['traj'], s['d_wall'], validity(d)))

    if args.exponents:
        print('\n## power-law exponents  F_PFk ~ gap_k^a  (adjacent pairs, sorted by gap)')
        for r in allr:
            pts = sorted((gap(d, r), d['max'][r], n) for n, d in rows
                         if 'error' not in d and gap(d, r) is not None
                         and r in d['max'])
            pts = [q for q in pts if q[0] > 0]
            if len(pts) < 2:
                continue
            out = []
            for (g1, f1, n1), (g2, f2, n2) in zip(pts, pts[1:]):
                if g1 <= 0 or g2 <= 0 or g1 == g2 or f1 <= 0 or f2 <= 0:
                    continue
                out.append('%.3f (%s->%s)' % (math.log(f2/f1)/math.log(g2/g1), n1, n2))
            if out:
                print('  %-4s gap %.5f..%.5f : %s' % (r, pts[0][0], pts[-1][0],
                                                      '  '.join(out)))
        print('\n  NB an exponent is only meaningful if the OTHER masses bounding')
        print('  that rung were held fixed across the pair -- a sweep that moves a')
        print('  rung\'s own solve mass conflates conditioning with gap width.')


if __name__ == '__main__':
    main(sys.argv[1:])
