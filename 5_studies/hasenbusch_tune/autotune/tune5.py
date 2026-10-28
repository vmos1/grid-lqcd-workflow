#!/usr/bin/env python3
"""Tune 5 planner tool: per-call cost model, wall calibration, pricing, and the worker's queue.

Campaign tune5-autonomous (__docs/2026_10_28_handoff_tune5_autonomous.md; this tool and the
queue: __docs/2026_10_28_tune5_worker_queue_cost_model.md). No compute: it reads run logs and
writes candidate files; the worker (perlmutter/tune5_worker.sh) runs them. Stdlib only, Python
3.6 compatible, so `python3` and `uv run python` both work.

Subcommands:
  fit       per-call force time of every monomial in the force-only runs (samples >= 1, L014),
            fitted against the rung's lighter mass, with leave-one-run-out residuals
  calib     wall / model force per trajectory run (pure Grid, TRAJL 1.0, 40 GB, four nodes)
  price     cost of given (ladder, layout) pairs against a reference
  sweep     every nested layout of one ladder, ranked by predicted wall
  enqueue   write a candidate file into the worker's queue (checked by the worker's --check)
  status    queue counts, worker logs, results rows

Cost model (all from logs, refit on every call):
  force per trajectory = sum over levels of (3 N_l + 1) x sum of that level's per-call times,
  N_l the level's steps per trajectory (FG: 3 force calls per step + 1; the driver's [Levels]
  banner prints the count). Levels: light group = the two log-dets + every ratio rung; strange;
  tail (bare det); gauge.
  per-call time: MEASURED for a (ladder, routes) that has a force-only run; otherwise the fit
    multigrid rung   T = c0 + c1 / (a - m_c)^2   a = the rung's lighter mass (its solves are at a)
    mixed-CG rung    T = c0 + c1 / (a - m_c)
    tail (bare det)  T = c0 + c1 / (t - m_c)
    log-dets, strange, gauge: the mean over the runs (mass-independent here)
  m_c = -0.2445 (tune 1's rung-peak fit, 10-15/16); `fit --scan-mc` refits it from the times.
  wall = K x force, K from `calib` (median over runs); cycle = wall + the measured cycle - wall.
Routes default to the rule every run so far used (multigrid for a rung whose lighter mass is
below --mg-below -0.226, its heatbath on multigrid if the heavier one is): a HYPOTHESIS (July,
QUDA solvers); `fit` prints where the two fitted curves cross in pure Grid.
"""
import argparse
import glob
import math
import os
import re
import subprocess
import sys
import time

BASE = '/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd'
RUNS = BASE + '/runs'
QUEUE = os.environ.get('TUNE5_QUEUE', RUNS + '/2026_10_28_tune5_queue')   # the worker's QUEUE; override only to test
WORKER = BASE + '/grid-lqcd-workflow/5_studies/hasenbusch_tune/perlmutter/tune5_worker.sh'
GPU40 = 'totalGlobalMem: 42404806656'
MC = -0.2445
EXP = {'mg': 2, 'mixed': 1, 'tail': 1}
MG_BELOW = -0.226
FIXED = ('LightLogDet', 'StrangeLogDet', 'Strange', 'Gauge')
FO_GLOBS = ('2026_10_16_fo8_*', '2026_10_26_fo8_A5*', '*_t5_*')
TRAJ_GLOBS = ('2026_1[0-2]_*',)
TRUST_FROM = (2026, 10, 6)   # trust rule: pure Grid inputs only after the 10-06 fixes (handoff)
BASELINE_LADDER = '-0.2416,-0.2400,-0.2320,-0.2180,-0.1870'
BASELINE_LAYOUT = '12/12/48/48'
K_DEFAULT = 1.47          # tune 2's wall / force (10-27); used only if calib finds no run
MAX_MASSES_TRAJ = 6       # 7 host-OOM in a trajectory on four 40 GB nodes (A3, 10-20)
GROUPS = ('light', 'strange', 'tail', 'gauge')

ENV_RE = re.compile(r'^ENV (TRAJ48_PUREGRID|FORCES48_PUREGRID) ')
FO_RE = re.compile(r'FORCES_ONLY level (\S+) sample (\d+) .*time=(\S+)')
TS_RE = re.compile(r'^Grid : \S+ : +([0-9.]+) s : ')


# ---- log parsing ------------------------------------------------------------------------------
def env_fields(line):
    out = {}
    for m in re.finditer(r'(\w+)=("[^"]*"|\S*)', line):
        out.setdefault(m.group(1), m.group(2).strip('"'))
    return out


def read_head(path):
    with open(path, errors='replace') as f:
        return f.readline()


def rung_lists(env, fo):
    """(mg, mg_hb, mixed, mixed_hb) as sets of ints, from either launcher's ENV line."""
    keys = (('GRID_MG_RUNGS', 'GRID_MG_HB', 'GRID_MIXED', 'GRID_MIXED_HB') if fo else
            ('GRID_MG_RUNGS', 'GRID_MG_HEATBATH_RUNGS', 'GRID_MIXED_CG_RUNGS', 'GRID_MIXED_CG_HEATBATH_RUNGS'))
    out = []
    for k in keys:
        v = env.get(k, 'none')
        out.append(set() if v in ('none', '') else set(int(x) for x in v.split(',')))
    return tuple(out)


def masses(ladder):
    return [float(x) for x in ladder.split(',')]


def route_key(ladder, routes):
    return ladder + '|' + '|'.join(','.join(str(x) for x in sorted(s)) for s in routes)


def parse_fo(rundir):
    """Force-only run -> dict(ladder, routes, cfg, times {monomial: mean s over samples >= 1})."""
    log = rundir + '/hmc.log'
    head = read_head(log)
    if not head.startswith('ENV FORCES48_PUREGRID '):
        return None
    env = env_fields(head)
    if env.get('FO_PROFILE') != 'production' or env.get('FORCES_SKIP') != 'none' or env.get('FORCES_STIFFNESS', '0') != '0':
        return None
    acc, gpu = {}, set()
    with open(log, errors='replace') as f:
        for line in f:
            m = FO_RE.search(line)
            if m and int(m.group(2)) >= 1:
                acc.setdefault(m.group(1), []).append(float(m.group(3)))
            elif 'totalGlobalMem:' in line:
                gpu.add(line.strip().split('AcceleratorCudaInit[0]:')[-1].strip())
    if not acc or gpu != {GPU40} or env.get('NODES') != '4' or env.get('DEVICE_MEM_MB') != '8000':
        return None
    return dict(run=os.path.basename(rundir), ladder=env['LADDER'], routes=rung_lists(env, True),
                bin=env.get('BIN', ''), cfg=env.get('IMPORT_CFG', ''), n=min(len(v) for v in acc.values()),
                times=dict((k, sum(v) / len(v)) for k, v in acc.items()))


def legacy_levels(env):
    """Levels of a run that used the legacy knobs (driver mapping, gen_qcd_..._compact_schur.cc)."""
    md = int(env['MDSTEPS'])
    gm = int(env['GAUGE_INNER_MULT']) if env.get('GAUGE_INNER_MULT', 'none') != 'none' else 2
    tl = env.get('TAIL_LEVEL', 'none')
    tl = 'outer' if tl == 'none' else tl
    if env.get('STRANGE_LEVEL') == 'middle':
        sm = int(env['STRANGE_INNER_MULT']) if env.get('STRANGE_INNER_MULT', 'default') != 'default' else 1
        lv = {'light': 0, 'strange': 1, 'tail': {'outer': 0, 'middle': 1, 'inner': 2}[tl], 'gauge': 2}
        mult = [1, sm, gm]
    else:
        lv = {'light': 0, 'strange': 0, 'tail': 1 if tl == 'inner' else 0, 'gauge': 1}
        mult = [1, gm]
    steps = [md]
    for m in mult[1:]:
        steps.append(steps[-1] * 2 * m)
    return dict((g, steps[lv[g]]) for g in GROUPS)


def run_layout(env):
    """{group: steps per trajectory} from a TRAJ48 ENV line (either knob family)."""
    if env.get('LEVEL_LIGHT', 'none') == 'none':
        return legacy_levels(env)
    lv = dict((g, int(env['LEVEL_' + g.upper()])) for g in GROUPS)
    mult = [1] + [int(x) for x in env['LEVEL_MULT'].split(',')]
    steps = [int(env['MDSTEPS'])]
    for l in range(1, max(lv.values()) + 1):
        steps.append(steps[-1] * 2 * mult[l])
    return dict((g, steps[lv[g]]) for g in GROUPS)


def parse_traj(rundir):
    log = rundir + '/hmc.log'
    try:
        head = read_head(log)
    except (IOError, OSError):
        return None
    if not head.startswith('ENV TRAJ48_PUREGRID '):
        return None
    env = env_fields(head)
    ok = (env.get('TRAJL') == '1.0' and env.get('RAT') == '7e-4/35.0/20' and env.get('QUDA_ENV') == '0'
          and env.get('NODES') == '4' and env.get('DEVICE_MEM_MB') == '8000' and env.get('TOL_DRV') == '1e-11'
          and env.get('INTEGRATOR') == 'ForceGradient' and '_stock_' in env.get('BIN', ''))
    if not ok:
        return None
    walls, cycles, gpu, tb, last_ts, first_hb = [], [], set(), None, 0.0, None
    with open(log, errors='replace') as f:
        for line in f:
            m = TS_RE.match(line)
            if m:
                last_ts = float(m.group(1))
            if 'Total H before trajectory = ' in line:
                tb = float(line.split()[4])
                if first_hb is None:
                    first_hb = tb
            elif 'Total H after trajectory ' in line and tb is not None:
                walls.append(float(line.split()[4]) - tb)
                tb = None
            elif 'Total time for trajectory (s):' in line:
                cycles.append(float(line.split()[-1]))
            elif 'totalGlobalMem:' in line:
                gpu.add(line.strip().split('AcceleratorCudaInit[0]:')[-1].strip())
    if not walls or gpu != {GPU40}:
        return None
    n = min(len(walls), len(cycles))
    return dict(run=os.path.basename(rundir), ladder=env['LADDER'], routes=rung_lists(env, False),
                layout=run_layout(env), walls=walls, cycles=cycles[:n],
                startup=first_hb or 0.0, total=last_ts, bin=env.get('BIN', ''))


def fo_runs(globs):
    out = []
    for g in globs:
        for d in sorted(glob.glob(RUNS + '/' + g)):
            if os.path.isfile(d + '/hmc.log'):
                r = parse_fo(d)
                if r:
                    out.append(r)
    seen, uniq = set(), []
    for r in out:
        if r['run'] not in seen:
            seen.add(r['run'])
            uniq.append(r)
    return uniq


# ---- per-call model ---------------------------------------------------------------------------
def classify(fo):
    """Data points (run, class, mass, seconds) of one force-only run."""
    m = masses(fo['ladder'])
    mg, _, mixed, _ = fo['routes']
    pts = []
    for k in range(len(m) - 1):
        t = fo['times'].get('PF%d' % k)
        if t is None:
            continue
        cls = 'mg' if k in mg else 'mixed' if k in mixed else 'cg'
        pts.append((fo['run'], cls, m[k], t))
    if 'LightSchurPF' in fo['times']:
        pts.append((fo['run'], 'tail', m[-1], fo['times']['LightSchurPF']))
    for name in FIXED:
        if name in fo['times']:
            pts.append((fo['run'], name, 0.0, fo['times'][name]))
    return pts


def lsq(pts, cls, mc):
    """Relative least squares of T = c0 + c1 x, x = 1/(a - mc)^p; returns (c0, c1) or None."""
    p = EXP[cls]
    rows = [(1.0 / (a - mc) ** p, t) for _, c, a, t in pts if c == cls and a > mc]
    if len(rows) < 2:
        return None
    sw = swx = swxx = swy = swxy = 0.0
    for x, y in rows:
        w = 1.0 / (y * y)
        sw += w; swx += w * x; swxx += w * x * x; swy += w * y; swxy += w * x * y
    det = sw * swxx - swx * swx
    if abs(det) < 1e-300:
        return None
    return ((swxx * swy - swx * swxy) / det, (sw * swxy - swx * swy) / det)


class Model(object):
    def __init__(self, fos, mc=MC):
        self.fos, self.mc = fos, mc
        self.pts = [p for fo in fos for p in classify(fo)]
        self.coef = dict((c, lsq(self.pts, c, mc)) for c in EXP)
        self.fixed = {}
        for name in FIXED:
            v = [t for _, c, _, t in self.pts if c == name]
            if v:
                self.fixed[name] = sum(v) / len(v)
        self.range = {}
        for c in EXP:
            a = [x for _, cc, x, _ in self.pts if cc == c]
            if a:
                self.range[c] = (min(a), max(a))
        self.measured = {}
        for fo in fos:
            self.measured.setdefault(route_key(fo['ladder'], fo['routes']), []).append(fo)

    def t(self, cls, a):
        c = self.coef.get(cls)
        if c is None or a <= self.mc:
            return float('nan')
        return c[0] + c[1] / (a - self.mc) ** EXP[cls]

    def percall(self, ladder, routes):
        """{monomial: seconds}, source ('measured: runs' or 'fit'), extrapolation notes."""
        key = route_key(ladder, routes)
        if key in self.measured:
            runs = self.measured[key]
            names = set(n for fo in runs for n in fo['times'])
            out = dict((n, sum(fo['times'][n] for fo in runs if n in fo['times']) /
                        sum(1 for fo in runs if n in fo['times'])) for n in names)
            return out, 'measured ' + '+'.join(fo['run'] for fo in runs), []
        m = masses(ladder)
        mg, _, mixed, _ = routes
        out, notes = dict(self.fixed), []
        for k in range(len(m) - 1):
            cls = 'mg' if k in mg else 'mixed' if k in mixed else 'cg'
            if cls == 'cg':
                out['PF%d' % k] = float('nan')
                notes.append('PF%d plain CG: no data' % k)
                continue
            out['PF%d' % k] = self.t(cls, m[k])
            lo, hi = self.range.get(cls, (0, 0))
            if not lo - 1e-9 <= m[k] <= hi + 1e-9:
                notes.append('PF%d %s at %.4f outside data %.4f..%.4f' % (k, cls, m[k], lo, hi))
        out['LightSchurPF'] = self.t('tail', m[-1])
        lo, hi = self.range.get('tail', (0, 0))
        if not lo - 1e-9 <= m[-1] <= hi + 1e-9:
            notes.append('tail at %.4f outside data %.4f..%.4f' % (m[-1], lo, hi))
        return out, 'fit', notes


def group_of(name):
    if name == 'Strange':
        return 'strange'
    if name == 'LightSchurPF':
        return 'tail'
    if name == 'Gauge':
        return 'gauge'
    return 'light'


def force(pc, layout):
    """Force seconds per trajectory, total and per group, from per-call times and a layout."""
    per = dict((g, 0.0) for g in GROUPS)
    for name, t in pc.items():
        g = group_of(name)
        per[g] += (3 * layout[g] + 1) * t
    return sum(per.values()), per


def auto_routes(ladder, below=MG_BELOW):
    m = masses(ladder)
    mg, mghb, mx, mxhb = set(), set(), set(), set()
    for k in range(len(m) - 1):
        (mg if m[k] < below else mx).add(k)
        if m[k] < below and m[k + 1] < below:
            mghb.add(k)
        else:
            mxhb.add(k)
    return (mg, mghb, mx, mxhb)


def parse_layout(s):
    v = [int(x) for x in s.split('/')]
    if len(v) != 4:
        raise ValueError('layout is L/S/T/G, got %s' % s)
    return dict(zip(GROUPS, v))


def layout_ok(lay):
    """'' if the four step counts nest (each distinct one 2 x mult x the previous), else why."""
    st = sorted(set(lay.values()))
    for i in range(1, len(st)):
        if st[i] % st[i - 1] or (st[i] // st[i - 1]) % 2:
            return '%d/%d not even' % (st[i], st[i - 1])
    return ''


def fmt_lay(lay):
    return '/'.join(str(lay[g]) for g in GROUPS)


def fmt_routes(r):
    return ' '.join('%s=%s' % (n, ','.join(str(x) for x in sorted(s)) or '-')
                    for n, s in zip(('mg', 'mg_hb', 'mixed', 'mixed_hb'), r))


# ---- calibration -----------------------------------------------------------------------------
def run_date(rundir):
    m = re.match(r'(\d{4})_(\d{1,2})_(\d{1,2})_', os.path.basename(rundir))
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def smeared_calls(layout):
    """Force calls per trajectory on levels holding a smeared monomial (every group but the gauge):
    the batched smearing pullback runs once per such call (driver [Levels] banner)."""
    return sum(3 * n + 1 for n in set(layout[g] for g in ('light', 'strange', 'tail')))


def calibrate(model, globs=TRAJ_GLOBS, measured_only=True):
    """One row per trajectory run. measured_only: runs whose (ladder, routes) has a force-only run,
    so the wall model is not fitted on the per-call fit's errors."""
    rows = []
    for g in globs:
        for d in sorted(glob.glob(RUNS + '/' + g)):
            if run_date(d) < TRUST_FROM:
                continue
            r = parse_traj(d)
            if not r:
                continue
            pc, src, notes = model.percall(r['ladder'], r['routes'])
            if measured_only and src == 'fit':
                continue
            f, _ = force(pc, r['layout'])
            if not f == f:
                continue
            w = sum(r['walls']) / len(r['walls'])
            cw = [c - w0 for c, w0 in zip(r['cycles'], r['walls'])]
            rows.append(dict(r, force=f, wall=w, k=w / f, src=src, sc=smeared_calls(r['layout']),
                             cyc_minus_wall=(sum(cw) / len(cw)) if cw else float('nan')))
    return rows


def solve(a, b):
    """Gaussian elimination for a small dense system a x = b."""
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-300:
            return None
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


# wall models: features of a calibration row; fitted by relative least squares
WALL_MODELS = (
    ('K x force', lambda r: [r['force']]),
    ('a x force + b', lambda r: [r['force'], 1.0]),
    ('a x force + c x smeared calls + b', lambda r: [r['force'], r['sc'], 1.0]),
)


def fit_wall(rows, feat):
    if len(rows) < len(feat(rows[0])) + 1:
        return None
    n = len(feat(rows[0]))
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for r in rows:
        x, w = feat(r), 1.0 / r['wall'] ** 2
        for i in range(n):
            atb[i] += w * x[i] * r['wall']
            for j in range(n):
                ata[i][j] += w * x[i] * x[j]
    return solve(ata, atb)


def wall_pred(coef, feat, r):
    return sum(c * x for c, x in zip(coef, feat(r)))


def wall_models(rows):
    """[(name, feat, coef, rel. RMS, leave-one-run-out rel. RMS)] for every model in WALL_MODELS."""
    out = []
    for name, feat in WALL_MODELS:
        coef = fit_wall(rows, feat)
        if not coef:
            continue
        res = [wall_pred(coef, feat, r) / r['wall'] - 1 for r in rows]
        loo = []
        for i in range(len(rows)):
            c = fit_wall(rows[:i] + rows[i + 1:], feat)
            if c:
                loo.append(wall_pred(c, feat, rows[i]) / rows[i]['wall'] - 1)
        rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v)) if v else float('nan')
        out.append((name, feat, coef, rms(res), rms(loo), max(abs(x) for x in loo) if loo else float('nan')))
    return out


def median(v):
    v = sorted(v)
    n = len(v)
    return float('nan') if not n else v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def calib_summary(rows, k=None):
    """Calibration in use: the wall model with the lowest leave-one-out RMS (or K x force with a
    given k), cycle - wall and startup medians."""
    ks = [r['k'] for r in rows]
    cw = [r['cyc_minus_wall'] for r in rows if r['cyc_minus_wall'] == r['cyc_minus_wall']]
    st = [r['startup'] for r in rows if r['startup'] > 0]
    s = dict(k=median(ks) if ks else K_DEFAULT, kmin=min(ks) if ks else float('nan'),
             kmax=max(ks) if ks else float('nan'), cw=median(cw) if cw else 150.0,
             startup=median(st) if st else 300.0, n=len(rows))
    if k is not None:
        s.update(k=k, wall_name='K x force, K = %.3f (given)' % k, feat=WALL_MODELS[0][1], coef=[k], loo=float('nan'))
        return s
    models = wall_models(rows) if rows else []
    if not models:
        s.update(wall_name='K x force, K = %.3f (default)' % K_DEFAULT, feat=WALL_MODELS[0][1], coef=[K_DEFAULT], loo=float('nan'))
        return s
    name, feat, coef, _, loo, _ = min(models, key=lambda m: m[4])
    s.update(wall_name='%s (%s; leave-one-out %.1f%%)' % (name, ', '.join('%.4g' % c for c in coef), 100 * loo),
             feat=feat, coef=coef, loo=loo)
    return s


# ---- commands --------------------------------------------------------------------------------
def build(args):
    fos = fo_runs(args.fo_globs or FO_GLOBS)
    if not fos:
        sys.exit('no force-only runs matched %s' % (args.fo_globs or FO_GLOBS,))
    return Model(fos, args.mc)


def cmd_fit(args):
    model = build(args)
    print('force-only runs: %d (production profile, 40 GB, four nodes, samples >= 1)' % len(model.fos))
    for fo in model.fos:
        print('  %-22s %-14s %d samples  %s  %s  routes %s' % (fo['run'], fo['bin'][-6:], fo['n'], fo['cfg'][-15:],
                                                             fo['ladder'], fmt_routes(fo['routes'])))
    print()
    print('m_c = %.4f; T = c0 + c1 / (a - m_c)^p' % model.mc)
    print('| class | p | c0 s | c1 | points | mass range | rel. RMS | leave-one-run-out max |')
    print('|---|---|---|---|---|---|---|---|')
    for cls in EXP:
        c = model.coef[cls]
        pts = [p for p in model.pts if p[1] == cls]
        if not c:
            print('| %s | %d | - | - | %d | - | - | - |' % (cls, EXP[cls], len(pts)))
            continue
        res = [(model.t(cls, a) - t) / t for _, _, a, t in pts]
        loo = []
        for run in sorted(set(p[0] for p in pts)):
            cc = lsq([p for p in model.pts if p[0] != run], cls, model.mc)
            if not cc:
                continue
            for _, _, a, t in [p for p in pts if p[0] == run]:
                loo.append((abs(cc[0] + cc[1] / (a - model.mc) ** EXP[cls] - t) / t, run, a))
        lmax = max(loo) if loo else (float('nan'), '-', 0)
        lo, hi = model.range[cls]
        print('| %s | %d | %.3f | %.4g | %d | %.4f .. %.4f | %.1f%% | %.1f%% (%s, %.4f) |' % (
            cls, EXP[cls], c[0], c[1], len(pts), lo, hi,
            100 * math.sqrt(sum(r * r for r in res) / len(res)), 100 * lmax[0], lmax[1], lmax[2]))
    print()
    print('mass-independent per-call times (mean over runs): ' +
          ', '.join('%s %.2f s' % (k, v) for k, v in model.fixed.items()))
    if args.points:
        print()
        print('| run | class | lighter mass | measured s | fit s | rel. |')
        print('|---|---|---|---|---|---|')
        for run, cls, a, t in model.pts:
            if cls in EXP:
                f = model.t(cls, a)
                print('| %s | %s | %.4f | %.2f | %.2f | %+.1f%% |' % (run, cls, a, t, f, 100 * (f / t - 1)))
    # where the multigrid and mixed-CG curves cross (the -0.226 routing rule is a July QUDA number)
    if model.coef.get('mg') and model.coef.get('mixed'):
        # Upper crossing: from the lightest multigrid point upward, the first mass where mixed CG
        # is the cheaper one (right at m_c the 1/(a - m_c)^2 term wins and the order flips back).
        lo, hi = model.range['mg']
        cross = None
        a = lo
        while a < 0.0:
            if model.t('mixed', a) <= model.t('mg', a):
                cross = a
                break
            a += 1e-4
        print()
        print('multigrid vs mixed-CG per call: the fitted curves cross at a = %s (multigrid data stop at %.4f,'
              ' so above that the multigrid curve is an extrapolation); the routing rule in use is %.3f.'
              % ('%.4f' % cross if cross else 'none below 0', hi, MG_BELOW))
        for a in (-0.230, -0.226, -0.222, -0.218, -0.214, -0.210, -0.205, -0.200):
            print('  a=%.3f  multigrid %.2f s  mixed CG %.2f s' % (a, model.t('mg', a), model.t('mixed', a)))
    if args.scan_mc:
        print()
        print('m_c scan (pooled relative RMS of the three fitted classes):')
        best = None
        mc = -0.2470
        while mc <= -0.2420 + 1e-12:
            m2 = Model(model.fos, mc)
            res = []
            for cls in EXP:
                if not m2.coef[cls]:
                    continue
                res += [(m2.t(cls, a) - t) / t for _, c, a, t in m2.pts if c == cls]
            rms = math.sqrt(sum(r * r for r in res) / len(res))
            best = min(best, (rms, mc)) if best else (rms, mc)
            print('  m_c %.4f  rms %.2f%%' % (mc, 100 * rms))
            mc += 0.0005
        print('best m_c %.4f (rms %.2f%%)' % (best[1], 100 * best[0]))


def cmd_calib(args):
    model = build(args)
    rows = calibrate(model, args.traj_globs or TRAJ_GLOBS)
    print('trajectory runs: %d (pure Grid, TRAJL 1.0, RAT 7e-4, 40 GB, four nodes, --device-mem 8000)' % len(rows))
    print('| run | ladder | layout L/S/T/G | traj | wall s (mean) | model force s | K = wall/force | cycle - wall s | startup s | per-call source |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print('| %s | %s | %s | %d | %.0f | %.0f | %.3f | %.0f | %.0f | %s |' % (
            r['run'], r['ladder'], fmt_lay(r['layout']), len(r['walls']), r['wall'], r['force'], r['k'],
            r['cyc_minus_wall'], r['startup'], r['src'].replace('measured ', 'm: ')))
    s = calib_summary(rows)
    print()
    print('K median %.3f (min %.3f, max %.3f) over %d runs; cycle - wall median %.0f s; startup (log start -> first'
          ' H before) median %.0f s' % (s['k'], s['kmin'], s['kmax'], s['n'], s['cw'], s['startup']))
    print()
    print('| wall model (relative least squares, one row per run) | coefficients | rel. RMS | leave-one-out RMS | leave-one-out max |')
    print('|---|---|---|---|---|')
    for name, feat, coef, rms, loo, lmax in wall_models(rows):
        print('| %s | %s | %.1f%% | %.1f%% | %.1f%% |' % (name, ', '.join('%.4g' % c for c in coef), 100 * rms, 100 * loo, 100 * lmax))
    print('in use for pricing: %s' % s['wall_name'])
    # The whole chain on ladders with no force-only run: fitted per-call times, then the wall model.
    fitrows = [r for r in calibrate(model, args.traj_globs or TRAJ_GLOBS, measured_only=False) if r['src'] == 'fit']
    if fitrows:
        print()
        print('check, ladders without a force-only run (per-call from the fit, then the wall model in use):')
        print('| run | ladder | layout | wall s | predicted s | error |')
        print('|---|---|---|---|---|---|')
        for r in fitrows:
            p = wall_pred(s['coef'], s['feat'], r)
            print('| %s | %s | %s | %.0f | %.0f | %+.1f%% |' % (r['run'], r['ladder'], fmt_lay(r['layout']), r['wall'], p, 100 * (p / r['wall'] - 1)))
    if args.holdout:
        test = [r for r in rows if args.holdout in r['run']]
        train = [r for r in rows if args.holdout not in r['run']]
        print()
        print('holdout: fit on the %d runs without "%s", predict the %d with it' % (len(train), args.holdout, len(test)))
        print('| wall model | ' + ' | '.join(r['run'].split('_', 3)[-1] for r in test) + ' | RMS | max |')
        print('|---|' + '---|' * (len(test) + 2))
        for name, feat in WALL_MODELS:
            coef = fit_wall(train, feat) if train else None
            if not coef or not test:
                continue
            e = [wall_pred(coef, feat, r) / r['wall'] - 1 for r in test]
            print('| %s | %s | %.1f%% | %.1f%% |' % (name, ' | '.join('%+.1f%%' % (100 * x) for x in e),
                                                   100 * math.sqrt(sum(x * x for x in e) / len(e)), 100 * max(abs(x) for x in e)))


def price_one(model, cal, ladder, lay, routes):
    pc, src, notes = model.percall(ladder, routes)
    f, per = force(pc, lay)
    wall = wall_pred(cal['coef'], cal['feat'], dict(force=f, sc=smeared_calls(lay)))
    return dict(ladder=ladder, lay=lay, routes=routes, force=f, per=per, wall=wall, cycle=wall + cal['cw'],
                src=src, notes=notes, why=layout_ok(lay),
                big=len(masses(ladder)) > MAX_MASSES_TRAJ)


def cmd_price(args):
    model = build(args)
    cal = calib_summary(calibrate(model, args.traj_globs or TRAJ_GLOBS), args.k)
    ref = price_one(model, cal, args.ref_ladder, parse_layout(args.ref_layout), auto_routes(args.ref_ladder, args.mg_below))
    ladders = args.ladder or [args.ref_ladder]
    print('wall model: %s, %d runs; reference %s at %s: force %.0f s, wall %.0f s' % (
        cal['wall_name'], cal['n'], ref['ladder'], fmt_lay(ref['lay']), ref['force'], ref['wall']))
    print('| ladder | layout | routes | light s | strange s | tail s | gauge s | force s | wall s | vs ref | per-call | notes |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for lad in ladders:
        for ls in args.layout or [BASELINE_LAYOUT]:
            r = price_one(model, cal, lad, parse_layout(ls), auto_routes(lad, args.mg_below))
            note = '; '.join(([r['why']] if r['why'] else []) + (['> %d masses: no trajectory on 4 nodes' % MAX_MASSES_TRAJ] if r['big'] else []) + r['notes'])
            print('| %s | %s | %s | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %+.1f%% | %s | %s |' % (
                lad, ls, fmt_routes(r['routes']), r['per']['light'], r['per']['strange'], r['per']['tail'],
                r['per']['gauge'], r['force'], r['wall'], 100 * (r['wall'] / ref['wall'] - 1),
                r['src'].replace('measured ', 'm: '), note or '-'))


def layouts(light_min, light_max, max_steps=256):
    out = []
    ratios = (1, 2, 4, 6, 8)
    for nl in range(light_min, light_max + 1):
        for rs in ratios:
            for rt in ratios:
                for rg in ratios:
                    ns, nt = nl * rs, nl * rs * rt
                    ng = nt * rg
                    if ng > max_steps:
                        continue
                    out.append({'light': nl, 'strange': ns, 'tail': nt, 'gauge': ng})
    return out


def cmd_sweep(args):
    model = build(args)
    cal = calib_summary(calibrate(model, args.traj_globs or TRAJ_GLOBS), args.k)
    ref = price_one(model, cal, args.ref_ladder, parse_layout(args.ref_layout), auto_routes(args.ref_ladder, args.mg_below))
    routes = auto_routes(args.ladder, args.mg_below)
    rows = [price_one(model, cal, args.ladder, lay, routes) for lay in layouts(args.light_min, args.light_max)]
    rows = [r for r in rows if r['lay']['strange'] >= args.strange_min and r['lay']['tail'] >= args.tail_min
            and r['lay']['gauge'] >= args.gauge_min]
    rows.sort(key=lambda r: r['wall'])
    print('ladder %s, routes %s; wall model %s; reference %s at %s wall %.0f s; per-call %s%s' % (
        args.ladder, fmt_routes(routes), cal['wall_name'], ref['ladder'], fmt_lay(ref['lay']), ref['wall'], rows[0]['src'] if rows else '-',
        ('; ' + '; '.join(rows[0]['notes'])) if rows and rows[0]['notes'] else ''))
    print('floors: strange >= %d, tail >= %d, gauge >= %d (set them from measured step failures, not by this tool)' % (
        args.strange_min, args.tail_min, args.gauge_min))
    print('| rank | layout L/S/T/G | levels | force s | wall s | vs ref | admitted (>= %.0f%%) |' % args.min_saving)
    print('|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows[:args.top]):
        sv = 100 * (1 - r['wall'] / ref['wall'])
        print('| %d | %s | %d | %.0f | %.0f | %+.1f%% | %s |' % (i + 1, fmt_lay(r['lay']), len(set(r['lay'].values())),
                                                         r['force'], r['wall'], -sv, 'yes' if sv >= args.min_saving else 'no'))


def next_number():
    n = 0
    for d in ('pending', 'running', 'done', 'failed', 'rejected', 'staging'):
        for f in glob.glob('%s/%s/[0-9][0-9][0-9]_*' % (QUEUE, d)):
            n = max(n, int(os.path.basename(f)[:3]))
    return n + 1


def cmd_enqueue(args):
    routes = auto_routes(args.ladder, args.mg_below) if args.routes == 'auto' else tuple(
        set(int(x) for x in s.split(',') if x != '') for s in args.routes.split('/'))
    if len(routes) != 4:
        sys.exit('--routes is auto or mg/mg_hb/mixed/mixed_hb, e.g. 0,1,2/0,1/3/2,3')
    kv = [('name', args.name), ('kind', args.kind), ('ladder', args.ladder)]
    kv += [(k, ','.join(str(x) for x in sorted(s))) for k, s in zip(('mg', 'mg_hb', 'mixed', 'mixed_hb'), routes)]
    kv += [('seed', str(args.seed)), ('start', args.start)]
    model = build(args)
    cal = calib_summary(calibrate(model, args.traj_globs or TRAJ_GLOBS))
    if args.kind == 'traj':
        lay = parse_layout(args.layout)
        r = price_one(model, cal, args.ladder, lay, routes)
        mm = args.max_min or int(math.ceil((cal['startup'] + args.n_traj * (r['cycle'] + 60)) * 1.3 / 60.0)) + 10
        mm = max(mm, 30 + 45 * args.n_traj)
        kv += [('max_min', str(mm)), ('layout', args.layout), ('n_traj', str(args.n_traj))]
        kv += [('pred_wall_s', str(int(round(r['wall'])))), ('pred_force_s', str(int(round(r['force']))))]
        pred = 'force %.0f s, wall %.0f s, cycle %.0f s, budget %d min (wall model %s; per-call %s)' % (
            r['force'], r['wall'], r['cycle'], mm, cal['wall_name'], r['src'])
    else:
        kv += [('max_min', str(args.max_min or 60)), ('fo_samples', str(args.fo_samples))]
        pred = 'force-only, %d samples' % args.fo_samples
    if args.risk:
        kv.append(('risk', args.risk))
    if args.pair:
        kv.append(('pair', args.pair))
    for d in ('pending', 'running', 'done', 'failed', 'rejected', 'results', 'staging'):
        if not os.path.isdir(QUEUE + '/' + d):
            os.makedirs(QUEUE + '/' + d)
    fn = '%03d_%s.cand' % (args.number or next_number(), args.name)
    stage = '%s/staging/%s' % (QUEUE, fn)
    text = ''.join('%s=%s\n' % kv_ for kv_ in kv)
    for d in ('staging', 'pending', 'running', 'done', 'failed', 'rejected'):
        if os.path.exists('%s/%s/%s' % (QUEUE, d, fn)):
            sys.exit('%s already exists in %s/: pick another --name or --number' % (fn, d))
    with open(stage, 'w') as f:
        f.write(text)
    p = subprocess.Popen(['bash', WORKER, '--check', stage], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True)
    out = p.communicate()[0]
    print(out.rstrip())
    if p.returncode != 0:
        print('NOT queued: the worker refuses it; the file stays in %s' % stage)
        sys.exit(1)
    print('prediction: ' + pred)
    if args.dry_run:
        print('dry run: left in %s' % stage)
        return
    # A second write, not a move: staging/ keeps the checked copy as the planner's record, and the
    # worker takes a pending file only once it is 30 s old (never half-written).
    with open('%s/pending/%s' % (QUEUE, fn), 'w') as f:
        f.write(text)
    print('queued %s/pending/%s' % (QUEUE, fn))


def cmd_status(args):
    if not os.path.isdir(QUEUE):
        print('no queue yet (%s)' % QUEUE)
        return
    for d in ('pending', 'running', 'done', 'failed', 'rejected', 'staging'):
        fs = sorted(os.path.basename(f) for f in glob.glob('%s/%s/*.cand' % (QUEUE, d)))
        print('%-9s %2d  %s' % (d, len(fs), ' '.join(fs)))
    if os.path.exists(QUEUE + '/STOP'):
        print('STOP file present: workers exit before their next candidate')
    for wl in sorted(glob.glob(QUEUE + '/worker_*.log')):
        with open(wl, errors='replace') as f:
            last = f.readlines()[-1:]
        print('%s: %s' % (os.path.basename(wl), last[0].rstrip() if last else '(empty)'))
    rows = sorted(glob.glob(QUEUE + '/results/*.tsv'))
    if rows:
        print()
        for fn in rows:
            with open(fn, errors='replace') as f:
                sys.stdout.write(f.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mc', type=float, default=MC, help='critical mass of the fit (default %.4f)' % MC)
    ap.add_argument('--fo-globs', nargs='*', help='force-only run dirs under runs/ (default %s)' % ' '.join(FO_GLOBS))
    ap.add_argument('--traj-globs', nargs='*', help='trajectory run dirs under runs/ (default %s, filtered by the ENV line)' % ' '.join(TRAJ_GLOBS))
    ap.add_argument('--mg-below', type=float, default=MG_BELOW, help='routing rule: multigrid below this mass (default %.3f, a hypothesis)' % MG_BELOW)
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('fit')
    p.add_argument('--points', action='store_true', help='every data point against the fit')
    p.add_argument('--scan-mc', action='store_true', help='refit m_c from the per-call times')
    p = sub.add_parser('calib')
    p.add_argument('--holdout', help='fit without the runs whose name contains this, report their prediction errors')
    for name in ('price', 'sweep'):
        p = sub.add_parser(name)
        p.add_argument('--ref-ladder', default=BASELINE_LADDER)
        p.add_argument('--ref-layout', default=BASELINE_LAYOUT)
        p.add_argument('--k', type=float, help='wall / force instead of the calib median')
        if name == 'price':
            p.add_argument('--ladder', action='append', help='masses light -> tail (repeatable)')
            p.add_argument('--layout', action='append', help='L/S/T/G steps per trajectory (repeatable)')
        else:
            p.add_argument('--ladder', required=True)
            p.add_argument('--light-min', type=int, default=4)
            p.add_argument('--light-max', type=int, default=16)
            p.add_argument('--strange-min', type=int, default=1)
            p.add_argument('--tail-min', type=int, default=1)
            p.add_argument('--gauge-min', type=int, default=1)
            p.add_argument('--min-saving', type=float, default=5.0)
            p.add_argument('--top', type=int, default=30)
    p = sub.add_parser('enqueue')
    p.add_argument('--name', required=True)
    p.add_argument('--kind', choices=('traj', 'fo'), required=True)
    p.add_argument('--ladder', required=True)
    p.add_argument('--routes', default='auto', help="auto, or mg/mg_hb/mixed/mixed_hb lists, e.g. '0,1,2/0,1/3/2,3'")
    p.add_argument('--seed', type=int, default=300)
    p.add_argument('--start', default='djm3_2000')
    p.add_argument('--layout', help='traj: L/S/T/G')
    p.add_argument('--n-traj', type=int, default=1)
    p.add_argument('--fo-samples', type=int, default=9)
    p.add_argument('--max-min', type=int, help='budget in minutes (default: from the cost model)')
    p.add_argument('--risk', choices=('light', 'strange', 'tail', 'gauge', 'cost', 'none'))
    p.add_argument('--pair', help='name of the paired reference candidate')
    p.add_argument('--number', type=int, help='queue position (default: next free)')
    p.add_argument('--dry-run', action='store_true', help='check and price, leave the file in staging/')
    sub.add_parser('status')
    args = ap.parse_args()
    if args.cmd == 'enqueue' and args.kind == 'traj' and not args.layout:
        ap.error('enqueue --kind traj needs --layout')
    {'fit': cmd_fit, 'calib': cmd_calib, 'price': cmd_price, 'sweep': cmd_sweep, 'enqueue': cmd_enqueue,
     'status': cmd_status}.get(args.cmd, lambda a: ap.print_help())(args)


if __name__ == '__main__':
    main()
