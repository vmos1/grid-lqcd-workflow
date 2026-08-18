#!/usr/bin/env python3
"""FORCES_ONLY log tables: per-level forces/timings, candidate-scan, and A/B diff.

Parses the ``FORCES_ONLY`` mode output of the compact_schur hasenbusch drivers
(one heatbath + force eval per action piece, no MD), i.e. logs containing::

    FORCES_ONLY level PF0 sample 0 avg=... max=... refresh=... time=...
    FORCES_ONLY samples=3 avg: LightLogDet=... PF0=... ... LightSchurPF=...
    FORCES_ONLY samples=3 max: ...          (and refresh: / time:)
    [Ladder] rung 0 DerivativeSolver/ActionSolver = QUDA MG (3 levels), mass=-0.2416

Every number is copied verbatim from the log; the only computed quantities are
sample means for incomplete runs (summary lines missing after a timeout), the
A/B relative differences / refresh ratios, and the optional tail F*eps.

Level semantics (compact_schur ladder driver): ``PFk`` is the Hasenbusch ratio
det(m_{k+1})/det(m_{k+2}) whose solves run at mass m_{k+1} (= the ``[Ladder]
rung k ... mass=`` line); ``LightSchurPF`` is the bare-determinant tail at the
last ladder mass; ``LightLogDet`` is the clover EE logdet.  The binary does NOT
print the tail mass, so either pass the ladder explicitly or (better) have the
submit script echo ``HASEN_LADDER=...`` as the log's first line — this parser
picks that up automatically (droprung_evensplit_debug.sh onward does this).

That same ``ENV`` header line should also carry the ROUTING knobs
(``FORCES_SKIP_RUNGS=``, ``HASEN_MG_HEATBATH_RUNGS=``, ``none`` when unset;
forces_only_sop_48.sh consumers do this from 2026-08-11).  They contribute no
number to any table, but scan and ab modes print a ⚠ block when the runs being
compared disagree on one — skipping changes which levels exist at all, and the
MG-vs-CG heatbath choice changes refresh seconds without changing forces.

Modes::

    # one log -> per-level table (+ per-sample detail with --samples)
    # NB masses are negative: write --ladder=-0.2416,... (equals form),
    # else argparse mistakes the value for an option
    python forces_only_tables.py single <log> [--ladder=m1,m2,...] [--samples]

    # candidate dirs (each holding scan_<dirname>.log) -> the interleaved
    # mass+force table of __docs/2026_7_8_hasen_tail_force_scan.md section 1,
    # plus refresh-seconds and deriv-seconds tables
    python forces_only_tables.py scan <dir> [<dir> ...] \\
        [--cands <file>] [--ladder name=m1,m2,...] [--eps X]

    # two same-seed logs -> per-level force rel-diffs + refresh speedups
    # (e.g. the QUDA-vs-Grid heatbath A/B: ab <base_log> <hb_log>)
    python forces_only_tables.py ab <logA> <logB> [--labels A,B] [--tol 1e-3]

``--cands`` accepts any file with lines of the form ``name m1,m2,... [rest]``
— in particular the CANDS=( ... ) block of the scan submit scripts works
as-is (e.g. --cands submit_scripts/hmc48_compact/hasen_tail_force_scan_48.sh).
``--eps`` (scan mode) adds a tail F*eps column, eps = the tail's MD kick step.
"""

import argparse
import re
import sys
from pathlib import Path

SAMPLE_RE = re.compile(
    r'FORCES_ONLY level (\S+) sample (\d+) '
    r'avg=(\S+) max=(\S+) refresh=(\S+) time=(\S+)')
SUMMARY_RE = re.compile(r'FORCES_ONLY samples=(\d+) (avg|max|refresh|time): (.+)')
PAIR_RE = re.compile(r'(\S+)=(\S+)')
RUNG_RE = re.compile(r'\[Ladder\] rung (\d+) .*=\s*(.+), mass=(\S+)')
LADDER_ECHO_RE = re.compile(r'HASEN_LADDER=([-+0-9.eE,]+)')
TOL_RE = re.compile(r'CG tol: (.+)')
EXIT_RE = re.compile(r'^exit=(\d+)')
STATS = ('avg', 'max', 'refresh', 'time')

# Routing knobs the submit script echoes on the ENV header line.  These do not
# enter any number in the tables, but two runs compared side by side must share
# them or the comparison is apples-to-oranges: FORCES_SKIP_RUNGS changes WHICH
# levels exist, and HASEN_MG_HEATBATH_RUNGS changes the refresh timings (never
# the forces).  Value 'none' = knob unset in that run.
ROUTING_RE = {
    'FORCES_SKIP_RUNGS': re.compile(r'FORCES_SKIP_RUNGS=(\S+)'),
    'HASEN_MG_HEATBATH_RUNGS': re.compile(r'HASEN_MG_HEATBATH_RUNGS=(\S+)'),
    # Tolerances are the ONE knob that changes force VALUES rather than just
    # timings -- everything else here (skip lists, MG-vs-CG heatbath routing)
    # was validated bit-identical.  Run w0 was compared against 1e-12/1e-11
    # runs for days before anyone spotted it had run at 1e-8/1e-6, so a
    # mismatch on these two is the most important thing this block can catch.
    'TUNE_CG_TOL_ACTION': re.compile(r'TUNE_CG_TOL_ACTION=(\S+)'),
    'TUNE_CG_TOL_DERIV': re.compile(r'TUNE_CG_TOL_DERIV=(\S+)'),
}


def parse_forces_log(path):
    """Parse one FORCES_ONLY log.

    Returns a dict:
      levels   : level names in log order, e.g. [LightLogDet, PF0..PF3, LightSchurPF]
      samples  : {level: {sample_index: {avg,max,refresh,time}}}
      summary  : {stat: {level: value}} — verbatim if the log printed the
                 samples=N summary lines, else means over the per-sample lines
                 (flagged by 'complete': False)
      complete : True when the binary's own summary lines were found
      nsamples : sample count (from summary, else max seen)
      rung_solver / rung_mass : {rung_index: str/float} from [Ladder] lines
      ladder   : [m1..mN] if a HASEN_LADDER= echo line is present, else None
      routing  : {knob: value} for the ROUTING_RE knobs found on the ENV header
                 line (absent key = the submit script didn't record it, which is
                 NOT the same as the knob being unset — that echoes 'none')
      cg_tol   : the 'CG tol: ...' line text (or None);  exit : int or None
    """
    levels, samples = [], {}
    summary = {}
    rung_solver, rung_mass = {}, {}
    routing = {}
    ladder = cg_tol = exit_code = None
    nsamples = 0
    with open(path, errors='replace') as fh:
        for line in fh:
            if line.startswith('ENV'):
                for key, rx in ROUTING_RE.items():
                    mm = rx.search(line)
                    if mm:
                        routing[key] = mm.group(1)
            m = SAMPLE_RE.search(line)
            if m:
                lvl, s = m.group(1), int(m.group(2))
                if lvl not in samples:
                    samples[lvl] = {}
                    levels.append(lvl)
                samples[lvl][s] = {k: float(v) for k, v in
                                   zip(STATS, m.group(3, 4, 5, 6))}
                nsamples = max(nsamples, s + 1)
                continue
            m = SUMMARY_RE.search(line)
            if m:
                nsamples = int(m.group(1))
                summary[m.group(2)] = {k: float(v)
                                       for k, v in PAIR_RE.findall(m.group(3))}
                continue
            m = RUNG_RE.search(line)
            if m:
                rung_solver[int(m.group(1))] = m.group(2).strip()
                rung_mass[int(m.group(1))] = float(m.group(3))
                continue
            m = LADDER_ECHO_RE.search(line)
            if m and ladder is None:
                ladder = [float(x) for x in m.group(1).split(',')]
                continue
            m = TOL_RE.search(line)
            if m:
                cg_tol = m.group(1).strip()
                continue
            m = EXIT_RE.match(line)
            if m:
                exit_code = int(m.group(1))

    complete = len(summary) == len(STATS)
    if not complete:                      # timed-out run: mean over what landed
        summary = {stat: {lvl: sum(d[stat] for d in samples[lvl].values())
                          / len(samples[lvl])
                          for lvl in levels if samples[lvl]}
                   for stat in STATS}
    return {'path': str(path), 'levels': levels, 'samples': samples,
            'summary': summary, 'complete': complete, 'nsamples': nsamples,
            'rung_solver': rung_solver, 'rung_mass': rung_mass,
            'ladder': ladder, 'routing': routing, 'cg_tol': cg_tol,
            'exit': exit_code}


def level_masses(run, ladder=None):
    """{level: mass or None}. PFk mass from its [Ladder] rung line; the tail
    (LightSchurPF) mass only from the ladder (echo line or --ladder), which is
    cross-checked against the rung lines when both are present."""
    ladder = ladder or run['ladder']
    if ladder:
        for k, m in run['rung_mass'].items():
            if k < len(ladder) and abs(ladder[k] - m) > 1e-10:
                print(f"# WARNING {run['path']}: --ladder[{k}]={ladder[k]} "
                      f"!= log rung mass {m}", file=sys.stderr)
    out = {}
    for lvl in run['levels']:
        if lvl.startswith('PF'):
            k = int(lvl[2:])
            out[lvl] = ladder[k] if ladder and k < len(ladder) \
                else run['rung_mass'].get(k)
        elif lvl == 'LightSchurPF':
            out[lvl] = ladder[-1] if ladder else None
        else:
            out[lvl] = None
    return out


def fnum(v, fmt='.4g'):
    return '—' if v is None else format(v, fmt)


def run_status(run):
    tag = f"samples={run['nsamples']}"
    if not run['complete']:
        tag += ' INCOMPLETE(summary=sample-means)'
    if run['exit'] not in (None, 0):
        tag += f" exit={run['exit']}"
    return tag


def routing_str(run):
    """One-line rendering of the recorded routing knobs ('' if none recorded)."""
    return ' '.join(f'{k}={run["routing"][k]}'
                    for k in ROUTING_RE if k in run['routing'])


def routing_mismatch(named_runs):
    """[(knob, {value: [names]})] for knobs that DIFFER across the given runs.

    Only knobs recorded in every run are checked — an older log that predates
    the ENV echo simply can't be compared, and silently pretending it matches
    would be worse than saying nothing.  Returns [] when everything agrees.
    """
    out = []
    for knob in ROUTING_RE:
        vals = {}
        for name, run in named_runs:
            if knob not in run['routing']:
                vals = {}
                break
            vals.setdefault(run['routing'][knob], []).append(name)
        if len(vals) > 1:
            out.append((knob, vals))
    return out


def print_routing_warning(named_runs):
    """Emit a warning block when the runs being compared were not run under the
    same routing.  FORCES_SKIP_RUNGS changes which levels were computed at all;
    HASEN_MG_HEATBATH_RUNGS changes refresh timings but not forces."""
    bad = routing_mismatch(named_runs)
    if not bad:
        return
    print('> ⚠ **ROUTING MISMATCH — these runs are not directly comparable:**')
    for knob, vals in bad:
        detail = '; '.join(f'`{v}` ({", ".join(names)})'
                           for v, names in vals.items())
        note = ('different levels were computed'
                if knob == 'FORCES_SKIP_RUNGS'
                else 'heatbath timings differ by solver, not by physics')
        print(f'> - `{knob}`: {detail} — {note}')
    print()


# ---------------------------------------------------------------- single mode

def print_single(run, ladder=None, show_samples=False):
    masses = level_masses(run, ladder)
    s = run['summary']
    print(f"# {run['path']}  ({run_status(run)})")
    if run['cg_tol']:
        print(f"# CG tol: {run['cg_tol']}")
    if routing_str(run):
        print(f"# routing: {routing_str(run)}")
    print()
    print('| level | mass | solver | F avg | F max | refresh s | deriv s |')
    print('|---|---|---|---|---|---|---|')
    tot = {'refresh': 0.0, 'time': 0.0}
    for lvl in run['levels']:
        solver = run['rung_solver'].get(int(lvl[2:]), '') \
            if lvl.startswith('PF') else ''
        cells = [fnum(s[st].get(lvl)) for st in STATS]
        print(f"| {lvl} | {fnum(masses[lvl])} | {solver} | "
              + ' | '.join(cells) + ' |')
        for st in ('refresh', 'time'):
            tot[st] += s[st].get(lvl, 0.0)
    print(f"| **Σ** |  |  |  |  | **{tot['refresh']:.4g}** "
          f"| **{tot['time']:.4g}** |")
    if show_samples:
        print('\n#### per-sample F max (draw-to-draw spread)\n')
        hdr = '| sample | ' + ' | '.join(run['levels']) + ' |'
        print(hdr)
        print('|---' * (len(run['levels']) + 1) + '|')
        for i in range(run['nsamples']):
            row = [fnum(run['samples'][lvl].get(i, {}).get('max'))
                   for lvl in run['levels']]
            print(f'| {i} | ' + ' | '.join(row) + ' |')


# ------------------------------------------------------------------ scan mode

def load_cands_file(path):
    """name -> [masses] from any file with lines 'name m1,m2,... [rest]'
    (the CANDS=( "..." ) block of the scan submit scripts parses as-is)."""
    out = {}
    pat = re.compile(r'^\s*"?(\w[\w-]*)\s+(-?\d?\.?\d[-+0-9.eE,]*,[-+0-9.eE,]+)')
    with open(path, errors='replace') as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                try:
                    out[m.group(1)] = [float(x) for x in m.group(2).split(',')]
                except ValueError:
                    pass
    return out


def print_scan(runs, ladders, eps=None):
    """runs: [(name, parsed)] — interleaved mass+force table (doc section-1
    layout) + refresh and deriv timing tables."""
    per = []
    max_pf = 0
    for name, run in runs:
        masses = level_masses(run, ladders.get(name))
        pfs = sorted((l for l in run['levels'] if l.startswith('PF')),
                     key=lambda l: int(l[2:]))
        max_pf = max(max_pf, len(pfs))
        per.append((name, run, masses, pfs))

    # Table 1: cand | m1 F(PF0) | m2 F(PF1) | ... | m_tail F(tail) [| tail F*eps]
    hdr = ['cand']
    for k in range(max_pf):
        hdr += [f'm{k + 1}', f'F(PF{k})']
    hdr += ['m_tail', 'F(tail)']
    if eps:
        hdr.append(f'tail F·ε (ε={eps:g})')
    print_routing_warning(runs)
    print('#### max force per rung (summary over samples)\n')
    print('| ' + ' | '.join(hdr) + ' |')
    print('|---' * len(hdr) + '|')
    for name, run, masses, pfs in per:
        fmax = run['summary']['max']
        row = [name if run['complete'] else name + ' ⚠']
        for k in range(max_pf):
            lvl = f'PF{k}'
            if lvl in pfs:
                row += [fnum(masses[lvl]), fnum(fmax.get(lvl), '.3g')]
            else:
                row += ['—', '—']
        tail = fmax.get('LightSchurPF')
        row += [fnum(masses.get('LightSchurPF')), fnum(tail, '.3g')]
        if eps:
            row.append(fnum(tail * eps if tail is not None else None, '.3g'))
        print('| ' + ' | '.join(row) + ' |')

    for stat, label in (('refresh', 'heatbath (refresh) seconds'),
                        ('time', 'deriv (force-eval) seconds')):
        cols = ['LightLogDet'] + [f'PF{k}' for k in range(max_pf)] \
            + ['LightSchurPF']
        print(f'\n#### {label} per level\n')
        print('| cand | ' + ' | '.join(cols) + ' | Σ |')
        print('|---' * (len(cols) + 2) + '|')
        for name, run, _, _ in per:
            vals = [run['summary'][stat].get(c) for c in cols]
            tot = sum(v for v in vals if v is not None)
            print(f'| {name} | ' + ' | '.join(fnum(v) for v in vals)
                  + f' | **{tot:.4g}** |')
    print('\n_(⚠ = incomplete run: summary is the mean of the samples that '
          'landed, not the binary’s own summary line)_')


# -------------------------------------------------------------------- ab mode

def rel(a, b):
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b))
    return 0.0 if denom == 0 else abs(a - b) / denom


def print_ab(ra, rb, la, lb, tol):
    print(f"# A = {la}: {ra['path']}  ({run_status(ra)})")
    print(f"# B = {lb}: {rb['path']}  ({run_status(rb)})")
    print(f"# force PASS threshold: rel diff < {tol:g} on avg and max\n")
    # An MG-vs-CG heatbath A/B is EXPECTED to differ here -- that is the point of
    # the test -- but an unnoticed difference silently reinterprets the speedup
    # column, so say it out loud either way.
    print_routing_warning([(la, ra), (lb, rb)])
    lvls = [l for l in ra['levels'] if l in rb['levels']]
    only = [l for l in ra['levels'] + rb['levels'] if l not in lvls]
    if only:
        print(f"# levels not in both logs (skipped): {', '.join(only)}\n")

    print('#### forces (same seeds ⇒ should match to solver tolerance)\n')
    print(f'| level | F avg {la} | F avg {lb} | rel | F max {la} '
          f'| F max {lb} | rel | verdict |')
    print('|---|---|---|---|---|---|---|---|')
    worst = 0.0
    for lvl in lvls:
        row, rels = [], []
        for st in ('avg', 'max'):
            a = ra['summary'][st].get(lvl)
            b = rb['summary'][st].get(lvl)
            r = rel(a, b)
            rels.append(r)
            row += [fnum(a, '.6g'), fnum(b, '.6g'), fnum(r, '.1e')]
        bad = any(r is None or r >= tol for r in rels)
        worst = max(worst, *(r for r in rels if r is not None))
        print(f"| {lvl} | " + ' | '.join(row)
              + f" | {'**DIFF**' if bad else 'match'} |")
    print(f'\n#### timings\n')
    print(f'| level | refresh {la} | refresh {lb} | speedup | deriv {la} '
          f'| deriv {lb} |')
    print('|---|---|---|---|---|---|')
    tots = {k: [0.0, 0.0] for k in ('refresh', 'time')}
    for lvl in lvls:
        a_r = ra['summary']['refresh'].get(lvl)
        b_r = rb['summary']['refresh'].get(lvl)
        a_t = ra['summary']['time'].get(lvl)
        b_t = rb['summary']['time'].get(lvl)
        for k, a, b in (('refresh', a_r, b_r), ('time', a_t, b_t)):
            tots[k][0] += a or 0.0
            tots[k][1] += b or 0.0
        speed = f'{a_r / b_r:.1f}×' if a_r and b_r else '—'
        print(f'| {lvl} | {fnum(a_r)} | {fnum(b_r)} | {speed} '
              f'| {fnum(a_t)} | {fnum(b_t)} |')
    sp = (f"{tots['refresh'][0] / tots['refresh'][1]:.1f}×"
          if tots['refresh'][1] else '—')
    print(f"| **Σ** | **{tots['refresh'][0]:.4g}** | **{tots['refresh'][1]:.4g}**"
          f" | **{sp}** | **{tots['time'][0]:.4g}** | **{tots['time'][1]:.4g}** |")
    verdict = 'PASS' if worst < tol else 'FAIL'
    print(f'\n**{verdict}** — worst force rel diff {worst:.2e} '
          f'(threshold {tol:g})')
    return verdict


# ------------------------------------------------------------------------ CLI

def parse_ladder_csv(text):
    return [float(x) for x in text.split(',')]


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode')   # required= kwarg needs py3.7
    sub.required = True

    p1 = sub.add_parser('single', help='per-level table for one log')
    p1.add_argument('log')
    p1.add_argument('--ladder', help='m1,m2,... (light→heavy) for mass labels')
    p1.add_argument('--samples', action='store_true',
                    help='also print per-sample F max')

    p2 = sub.add_parser('scan', help='candidate comparison tables')
    p2.add_argument('dirs', nargs='+',
                    help='candidate dirs, each holding scan_<dirname>.log')
    p2.add_argument('--cands',
                    help='file with "name m1,m2,..." lines (a scan submit '
                         'script CANDS block works as-is)')
    p2.add_argument('--ladder', action='append', default=[],
                    metavar='NAME=m1,m2,...',
                    help='per-candidate ladder override (repeatable)')
    p2.add_argument('--eps', type=float,
                    help='tail MD kick step → adds a tail F·ε column')

    p3 = sub.add_parser('ab', help='same-seed A/B force+timing diff')
    p3.add_argument('logA')
    p3.add_argument('logB')
    p3.add_argument('--labels', default='A,B', help='labels, e.g. grid,quda')
    p3.add_argument('--tol', type=float, default=1e-3,
                    help='force rel-diff PASS threshold (default 1e-3)')

    args = ap.parse_args(argv[1:])

    if args.mode == 'single':
        print_single(parse_forces_log(args.log),
                     ladder=parse_ladder_csv(args.ladder) if args.ladder
                     else None,
                     show_samples=args.samples)
    elif args.mode == 'scan':
        ladders = load_cands_file(args.cands) if args.cands else {}
        for spec in args.ladder:
            name, _, csv = spec.partition('=')
            ladders[name] = parse_ladder_csv(csv)
        runs = []
        for d in args.dirs:
            d = Path(d)
            log = d / f'scan_{d.name}.log'
            if not log.exists():
                print(f'# skipping {d}: no {log.name}', file=sys.stderr)
                continue
            runs.append((d.name, parse_forces_log(log)))
        if not runs:
            sys.exit('no scan_<name>.log found in any given dir')
        print_scan(runs, ladders, eps=args.eps)
    else:
        la, _, lb = args.labels.partition(',')
        verdict = print_ab(parse_forces_log(args.logA),
                           parse_forces_log(args.logB),
                           la or 'A', lb or 'B', args.tol)
        return 0 if verdict == 'PASS' else 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
