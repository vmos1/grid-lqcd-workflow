#!/usr/bin/env python3
"""Parse a single hasenbusch_tune HMC log into per-trajectory records.

This targets the log format emitted by the force-validation HMC drivers in
``5_studies/hasenbusch_tune`` (gen_qcd_cfgs_2plus1, gen_qcd_hasenbusch_tune_compact,
gen_qcd_hasenbusch_tune_compact_schur). Each trajectory prints, in order::

    HMC     : Total H before trajectory = <H_before>
    HMC     : Total H after trajectory  = <H_after>  dH = <dH>
    HMC     : Metropolis_test -- ACCEPTED|REJECTED
    HMC     : Total time for trajectory (s): <t>
    Message : Unsmeared plaquette
    Message : Plaquette: [ n ] <plaq_unsmeared>
    Message : Smeared plaquette
    Message : Plaquette: [ n ] <plaq_smeared>
    Message : Polyakov Loop: [ n ] (<re>,<im>)

The config label ``n`` (1-based) equals the checkpoint index: trajectory 0
produces label 1. So ``traj == n`` and "value after trajectory k" == label k+1.

NOTE: this is deliberately separate from ``hmc/extract.py`` in the parent
package, which parses a *different* Grid log layout (``-- # Trajectory = N``
markers) and is wired into the notebooks / ``hmc_obs`` CLI. This module is
dependency-free (pure stdlib) and tolerant of the non-UTF8 bytes that show up
in these particular Grid logs.

CLI::

    python hmc_log_parser.py <log> [<log> ...]
"""

import re
import sys

# A signed int/float, optionally in scientific notation.
_NUM = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'

_RE_HBEFORE = re.compile(r'Total H before trajectory\s*=\s*(' + _NUM + r')')
_RE_HAFTER  = re.compile(r'Total H after trajectory\s*=\s*(' + _NUM + r')\s+dH\s*=\s*(' + _NUM + r')')
_RE_METRO   = re.compile(r'Metropolis_test\s*--\s*(ACCEPTED|REJECTED)')
_RE_TIME    = re.compile(r'Total time for trajectory \(s\):\s*(' + _NUM + r')')
_RE_PLAQ    = re.compile(r'Plaquette:\s*\[\s*(\d+)\s*\]\s*(' + _NUM + r')')
_RE_PLOOP   = re.compile(r'Polyakov Loop:\s*\[\s*(\d+)\s*\]\s*\((' + _NUM + r')\s*,\s*(' + _NUM + r')\)')
_RE_INITPL  = re.compile(r'Initial plaquette\s*=\s*(' + _NUM + r')')

# Per-trajectory fields, all default None so missing observables are explicit.
_FIELDS = ('traj', 'H_before', 'H_after', 'dH', 'accepted',
           'traj_time_s', 'plaq', 'plaq_smeared', 'poly_re', 'poly_im')


def _blank():
    return dict.fromkeys(_FIELDS)


def parse_log(path):
    """Parse one log file.

    Returns ``{'path': str, 'initial_plaquette': float|None,
               'trajectories': [ {field: value, ...}, ... ]}``
    with one trajectory dict per HMC trajectory, ordered as written.
    """
    trajectories = []
    meta = {'path': str(path), 'initial_plaquette': None}
    cur = None
    plaq_is_smeared = False

    def flush():
        nonlocal cur
        if cur is not None:
            if cur['traj'] is None:          # loggerless run: index by position
                cur['traj'] = len(trajectories) + 1
            trajectories.append(cur)
        cur = None

    # errors='replace' — these Grid logs carry stray non-UTF8 bytes.
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            m = _RE_HBEFORE.search(line)
            if m:
                flush()
                cur = _blank()
                cur['H_before'] = float(m.group(1))
                plaq_is_smeared = False
                continue

            if cur is None:                  # preamble (before first trajectory)
                m = _RE_INITPL.search(line)
                if m:
                    meta['initial_plaquette'] = float(m.group(1))
                continue

            m = _RE_HAFTER.search(line)
            if m:
                cur['H_after'] = float(m.group(1))
                cur['dH'] = float(m.group(2))
                continue

            m = _RE_METRO.search(line)
            if m:
                cur['accepted'] = (m.group(1) == 'ACCEPTED')
                continue

            m = _RE_TIME.search(line)
            if m:
                cur['traj_time_s'] = float(m.group(1))
                continue

            if 'Unsmeared plaquette' in line:
                plaq_is_smeared = False
                continue
            if 'Smeared plaquette' in line:
                plaq_is_smeared = True
                continue

            m = _RE_PLAQ.search(line)
            if m:
                cur['traj'] = int(m.group(1))
                val = float(m.group(2))
                # First plaquette of the trajectory is unsmeared; second smeared.
                # The header lines above set plaq_is_smeared; the "already set"
                # fallback keeps this correct even if a header is missing.
                if plaq_is_smeared or cur['plaq'] is not None:
                    cur['plaq_smeared'] = val
                else:
                    cur['plaq'] = val
                continue

            m = _RE_PLOOP.search(line)
            if m:
                cur['traj'] = int(m.group(1))
                cur['poly_re'] = float(m.group(2))
                cur['poly_im'] = float(m.group(3))
                continue

    flush()
    return {**meta, 'trajectories': trajectories}


def _fmt(v, spec):
    return 'n/a' if v is None else format(v, spec)


def _print_table(parsed):
    print(f"# {parsed['path']}")
    if parsed['initial_plaquette'] is not None:
        print(f"#   initial plaquette = {parsed['initial_plaquette']:.10g}")
    hdr = ('traj', 'dH', 'H_after', 'accept', 'time_s', 'plaq', 'polyakov(re,im)')
    print('  {:>4} {:>14} {:>20} {:>6} {:>9} {:>13}  {}'.format(*hdr))
    for t in parsed['trajectories']:
        acc = 'n/a' if t['accepted'] is None else ('acc' if t['accepted'] else 'rej')
        poly = 'n/a'
        if t['poly_re'] is not None:
            poly = f"({t['poly_re']:+.6e}, {t['poly_im']:+.6e})"
        print('  {:>4} {:>14} {:>20} {:>6} {:>9} {:>13}  {}'.format(
            _fmt(t['traj'], 'd'),
            _fmt(t['dH'], '+.6g'),
            _fmt(t['H_after'], '.8f'),
            acc,
            _fmt(t['traj_time_s'], '.1f'),
            _fmt(t['plaq'], '.10g'),
            poly,
        ))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for path in argv[1:]:
        _print_table(parse_log(path))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
