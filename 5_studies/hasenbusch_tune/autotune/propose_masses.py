#!/usr/bin/env python3
"""
propose_masses.py — CPU-only helper for the Hasenbusch autotune loop.

Called by autotune.sbatch after each FORCES_ONLY run.  Reads the forces log
and current masses, proposes updated masses via coordinate equalization, and
writes state.json.  Prints NEXT_LADDER=<comma-list> on stdout for the calling
script to capture.

Normal usage (one call per iteration):
  python3 propose_masses.py \
      --log   <iter_NNN/forces.log> \
      --masses <comma-separated current masses> \
      --state  <run_dir/state.json> \
      [--target-balance 5.0] [--step-frac 0.3] \
      [--tail-max 0.5]       [--min-gap 0.005]

Summary-only usage (at end of job):
  python3 propose_masses.py --summary --state <run_dir/state.json>
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_forces_log(log_path):
    """Return {level_name: avg_force} from a FORCES_ONLY log."""
    text = Path(log_path).read_text()
    m = re.search(r'FORCES_ONLY samples=\d+ avg:\s*(.+)', text)
    if not m:
        raise ValueError(f"No 'FORCES_ONLY samples=N avg:' line in {log_path}")
    forces = {}
    for token in m.group(1).split():
        name, val = token.split('=')
        forces[name] = float(val)
    return forces


def forces_to_vec(forces_dict, n_masses):
    """Return [F_PF0, ..., F_PF{n-2}, F_Tail] ordered to match the mass ladder."""
    vec = []
    for i in range(n_masses - 1):
        key = f'PF{i}'
        if key not in forces_dict:
            raise ValueError(f"Missing {key} in forces: {forces_dict}")
        vec.append(forces_dict[key])
    if 'Tail' not in forces_dict:
        raise ValueError(f"Missing Tail in forces: {forces_dict}")
    vec.append(forces_dict['Tail'])
    return vec


# ---------------------------------------------------------------------------
# Mass proposal
# ---------------------------------------------------------------------------

def propose_new_masses(masses, forces, step_frac, tail_max, min_gap):
    """
    Coordinate update rule for N masses (masses[0] fixed = m_light).

    For each free mass m_k (k = 1 .. N-1):
      - F_left  = forces[k-1]  : level whose right boundary is m_k
      - F_right = forces[k]    : level whose left boundary is m_k (or Tail)

    Moving m_k in the +ve direction (heavier, toward 0 or beyond) decreases
    F_right and increases F_left.  So the signed imbalance

        imb = (F_right - F_left) / (F_right + F_left)

    is positive when the right level dominates → move m_k right (+ve step).

        Δm_k = step_frac × imb × span

    where span = right_bound − left_bound (always positive).

    For the Tail mass (k = N-1) the right bound is tail_max since the Tail has
    no ratio level above it.
    """
    N = len(masses)
    new_masses = list(masses)

    for k in range(1, N):
        F_left  = forces[k - 1]
        F_right = forces[k]
        left_bound  = masses[k - 1]
        right_bound = masses[k + 1] if k < N - 1 else tail_max
        span = right_bound - left_bound          # always > 0 (heavier − lighter)
        imb  = (F_right - F_left) / (F_right + F_left)
        new_masses[k] = masses[k] + step_frac * imb * span

    # Enforce minimum gap: forward pass (light → heavy)
    for k in range(1, N):
        new_masses[k] = max(new_masses[k], new_masses[k - 1] + min_gap)

    # Cap Tail mass below tail_max, then back-propagate gap constraint
    new_masses[-1] = min(new_masses[-1], tail_max - min_gap)
    for k in range(N - 2, 0, -1):
        new_masses[k] = min(new_masses[k], new_masses[k + 1] - min_gap)

    return new_masses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def balance_ratio(forces):
    return max(forces) / min(forces)


def fmt_ladder(masses):
    return ','.join(f'{m:.6f}' for m in masses)


def fmt_forces(forces):
    return '  '.join(f'{f:.6f}' for f in forces)


# ---------------------------------------------------------------------------
# Summary (--summary mode)
# ---------------------------------------------------------------------------

def print_summary(state_path):
    state = json.loads(Path(state_path).read_text())
    print(f"Converged : {state.get('converged', False)}")
    final_masses = state.get('masses', [])
    print(f"Final masses : {fmt_ladder(final_masses)}")
    print(f"Final balance: {state.get('balance', 'N/A')}")
    print()
    print(f"{'iter':>4}  {'balance':>8}  masses")
    for h in state.get('history', []):
        ms = ','.join(f'{m:.4f}' for m in h['masses'])
        print(f"{h['iter']:>4}  {h['balance']:>8.1f}  [{ms}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',    help='Path to FORCES_ONLY log')
    ap.add_argument('--masses', help='Comma-separated current masses')
    ap.add_argument('--state',  required=True, help='Path to state.json (read/write)')
    ap.add_argument('--target-balance', type=float, default=5.0,
                    help='Converge when max/min force ratio < this (default 5.0)')
    ap.add_argument('--step-frac', type=float, default=0.3,
                    help='Step fraction of available span per iteration (default 0.3)')
    ap.add_argument('--tail-max', type=float, default=0.5,
                    help='Upper bound for Tail mass (default 0.5)')
    ap.add_argument('--min-gap', type=float, default=0.005,
                    help='Minimum gap between adjacent ladder masses (default 0.005)')
    ap.add_argument('--summary', action='store_true',
                    help='Print history from state.json and exit (no log needed)')
    args = ap.parse_args()

    if args.summary:
        print_summary(args.state)
        return

    if not args.log or not args.masses:
        print('ERROR: --log and --masses are required unless --summary is given',
              file=sys.stderr)
        sys.exit(1)

    masses = [float(x) for x in args.masses.split(',')]

    try:
        forces_dict = parse_forces_log(args.log)
        forces      = forces_to_vec(forces_dict, len(masses))
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    ratio     = balance_ratio(forces)
    converged = ratio < args.target_balance

    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {'iteration': 0, 'history': []}
    iteration = state.get('iteration', 0)

    # Print iteration report
    print(f'iter={iteration}  balance={ratio:.1f}  (target < {args.target_balance})')
    print(f'  masses : {fmt_ladder(masses)}')
    print(f'  forces : {fmt_forces(forces)}')

    state['history'].append({
        'iter':    iteration,
        'masses':  masses,
        'forces':  forces,
        'balance': round(ratio, 2),
    })

    if converged:
        state.update(converged=True, masses=masses, forces=forces,
                     balance=round(ratio, 2))
        print('  CONVERGED')
        print(f'NEXT_LADDER={fmt_ladder(masses)}')
    else:
        new_masses = propose_new_masses(
            masses, forces,
            step_frac=args.step_frac,
            tail_max=args.tail_max,
            min_gap=args.min_gap,
        )
        state.update(converged=False, masses=new_masses, iteration=iteration + 1)
        print(f'  proposed : {fmt_ladder(new_masses)}')
        print(f'  tail_mass: {masses[-1]:.6f} -> {new_masses[-1]:.6f}')
        print(f'NEXT_LADDER={fmt_ladder(new_masses)}')

    state_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
