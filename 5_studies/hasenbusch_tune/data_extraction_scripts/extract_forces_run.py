#!/usr/bin/env python3
"""Build the Test-3 grid-vs-quda comparison tables from a 6-run HMC directory.

A Test-3 run directory (produced by ``submit_scripts/forces_mpi_validation/
run_hmc16.sh``) holds six logs, one per (code, mode)::

    {2plus1,compact,schur}_{GRID,QUDA}.log

This script parses all six with ``hmc_log_parser`` and emits, as Markdown ready
to paste into ``__docs/2026_6_24_test_results.md``:

  * Table 3  — dH / H-after / acceptance / steady wall-per-trajectory / speedup
  * Table 3b — unsmeared plaquette + Polyakov loop (per trajectory)

"Steady" trajectory time is the mean over trajectories *after* the first
(``--skip-first``, default 1): the first trajectory carries the one-time QUDA
autotune + first-call force init, so it is excluded from the per-traj timing and
the grid/quda speedup. Every physics number is copied verbatim from the logs;
the only computed quantity is the speedup ratio (grid_steady / quda_steady).

Usage::

    python extract_forces_run.py <run_dir> [--skip-first N] [--csv out.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

from hmc_log_parser import parse_log

# Doc/table order: code rows, grid then quda within each code.
CODES = ('compact', 'schur', '2plus1')
CODE_LABEL = {'compact': 'compact', 'schur': 'schur', '2plus1': '2+1'}
MODES = ('grid', 'quda')          # filename token is uppercased


def load_run(run_dir):
    """Return {(code, mode): parsed_log} for the six logs found in run_dir."""
    run_dir = Path(run_dir)
    runs = {}
    for code in CODES:
        for mode in MODES:
            log = run_dir / f"{code}_{mode.upper()}.log"
            if log.exists():
                runs[(code, mode)] = parse_log(log)
    if not runs:
        raise FileNotFoundError(f"No <code>_<MODE>.log files found in {run_dir}")
    return runs


def steady_time(parsed, skip_first):
    """Mean traj_time_s over trajectories after the first `skip_first`."""
    times = [t['traj_time_s'] for t in parsed['trajectories'][skip_first:]
             if t['traj_time_s'] is not None]
    return sum(times) / len(times) if times else None


def _cells(trajs, key, fmt):
    """Format the field `key` for the first three trajectories as table cells."""
    out = []
    for i in range(3):
        v = trajs[i][key] if i < len(trajs) else None
        out.append('—' if v is None else format(v, fmt))
    return out


def table3(runs, skip_first):
    lines = [
        "| code | mode | dH₀ | dH₁ | dH₂ | H after 0 | H after 1 | H after 2 | accept | traj time | speedup |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for code in CODES:
        st = {m: steady_time(runs[(code, m)], skip_first)
              for m in MODES if (code, m) in runs}
        for mode in MODES:
            if (code, mode) not in runs:
                continue
            tr = runs[(code, mode)]['trajectories']
            dH = _cells(tr, 'dH', '+.6g')
            Ha = _cells(tr, 'H_after', '.8f')
            n = len(tr)
            nacc = sum(1 for t in tr if t['accepted'])
            t_s = st.get(mode)
            time_cell = '—' if t_s is None else f"{t_s:.0f} s"
            speed = '—'
            if mode == 'quda' and st.get('grid') and st.get('quda'):
                speed = f"**{st['grid'] / st['quda']:.1f}×**"
            code_cell = f"**{CODE_LABEL[code]}**" if mode == 'grid' else ''
            lines.append(
                f"| {code_cell} | {mode} | {dH[0]} | {dH[1]} | {dH[2]} "
                f"| {Ha[0]} | {Ha[1]} | {Ha[2]} | {nacc}/{n} | {time_cell} | {speed} |"
            )
    return '\n'.join(lines)


def _poly_cell(t):
    if t['poly_re'] is None:
        return '—'
    return f"({t['poly_re'] * 1e3:+.5f}, {t['poly_im'] * 1e3:+.5f})"


def table3b(runs):
    lines = [
        "| code | mode | plaq after 0 | plaq after 1 | plaq after 2 "
        "| Poly after 0 (×10⁻³) | Poly after 1 (×10⁻³) | Poly after 2 (×10⁻³) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for code in CODES:
        for mode in MODES:
            if (code, mode) not in runs:
                continue
            tr = runs[(code, mode)]['trajectories']
            plaq = _cells(tr, 'plaq', '.10g')
            poly = [_poly_cell(tr[i]) if i < len(tr) else '—' for i in range(3)]
            code_cell = f"**{CODE_LABEL[code]}**" if mode == 'grid' else ''
            lines.append(
                f"| {code_cell} | {mode} | {plaq[0]} | {plaq[1]} | {plaq[2]} "
                f"| {poly[0]} | {poly[1]} | {poly[2]} |"
            )
    return '\n'.join(lines)


def write_csv(runs, path):
    cols = ['code', 'mode', 'traj', 'dH', 'H_before', 'H_after', 'accepted',
            'traj_time_s', 'plaq', 'plaq_smeared', 'poly_re', 'poly_im']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for code in CODES:
            for mode in MODES:
                if (code, mode) not in runs:
                    continue
                for t in runs[(code, mode)]['trajectories']:
                    w.writerow([code, mode] + [t[c] for c in cols[2:]])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', help='Test-3 run directory with the six logs')
    ap.add_argument('--skip-first', type=int, default=1,
                    help='trajectories to exclude from steady timing (default 1)')
    ap.add_argument('--csv', help='also write a tidy per-trajectory CSV here')
    args = ap.parse_args(argv[1:])

    runs = load_run(args.run_dir)
    have = ', '.join(f"{c}/{m}" for (c, m) in runs)
    print(f"# parsed {len(runs)} logs from {args.run_dir}\n#   ({have})\n")
    print("### Table 3 — ΔH / H / acceptance / wall-per-trajectory\n")
    print(table3(runs, args.skip_first))
    print(f"\n_(traj time = mean over trajectories after the first "
          f"{args.skip_first}; speedup = grid_steady / quda_steady)_\n")
    print("### Table 3b — gauge observables (unsmeared plaquette + Polyakov loop)\n")
    print(table3b(runs))

    if args.csv:
        write_csv(runs, args.csv)
        print(f"\n# wrote {args.csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
