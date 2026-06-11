#!/usr/bin/env python3
"""
propose_masses.py — CPU-only helper for the Hasenbusch autotune loop.

Called by autotune.sbatch after each FORCES_ONLY run.  Reads the forces/times
log and current masses, proposes updated masses, and writes state.json.
Prints NEXT_LADDER=<comma-list> on stdout for the calling script to capture.

Cost metric
-----------
The integrator runs every PF/Tail/Strange level on one shared MD step size,
so the step is bounded by the stiffest (max-force) term, and one MD step
costs (roughly) the sum of all per-level force-compute times:

    Cost(ladder) = max(F_max_light, F_strange_max) * (sum(t_light) + t_strange)

F_strange_max and t_strange are CONSTANTS: the strange sector (RHMC + logdet)
does not depend on HASEN_LADDER, so they are measured once via a full-HMC run
(see hasenbusch_autotune.md) and passed in via --strange-fmax/--strange-time
rather than recomputed every iteration (FORCES_SKIP_STRANGE=1 stays on).

Two regimes fall out of this:
  - F_max_light > F_strange_max: F_max is set by the light ladder, so reducing
    it reduces Cost directly -> "force-balance" phase (equalize forces, as
    before).
  - F_max_light <= F_strange_max: F_max is pinned at F_strange_max, so
    Cost ~ F_strange_max * sum(t_light) -> "minimize-time" phase (push free
    masses heavier, away from the critical mass, to cut CG solve time).

The phase is decided fresh each iteration from the just-measured forces, so
if a "minimize-time" step pushes F_max_light back above F_strange_max, the
next iteration reverts to "force-balance" and pulls it back -- the ladder
self-stabilizes near the F_max_light ~= F_strange_max boundary, which is
where Cost is minimized.

Normal usage (one call per iteration):
  python3 propose_masses.py \
      --log   <iter_NNN/forces.log> \
      --masses <comma-separated current masses> \
      --state  <run_dir/state.json> \
      [--strange-fmax 2.20] [--strange-time 392.0] [--cost-tol 0.02] \
      [--step-frac 0.3]     [--time-step-frac 0.15] \
      [--tail-max 0.5]      [--min-gap 0.005]

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

def _parse_kv_line(line):
    d = {}
    for token in line.split():
        name, val = token.split('=')
        d[name] = float(val)
    return d


def parse_forces_log(log_path):
    """Return (forces_avg_dict, forces_max_dict, times_dict) from a FORCES_ONLY log.

    forces_avg_dict: {level_name: avg_force}, from 'FORCES_ONLY samples=N avg: ...'
    forces_max_dict: {level_name: max_force}, from 'FORCES_ONLY samples=N max: ...'
    times_dict:      {level_name: force_compute_time_seconds}, from
                      'FORCES_ONLY samples=N time: ...'
    """
    text = Path(log_path).read_text()

    m_avg = re.search(r'FORCES_ONLY samples=\d+ avg:\s*(.+)', text)
    if not m_avg:
        raise ValueError(f"No 'FORCES_ONLY samples=N avg:' line in {log_path}")

    m_max = re.search(r'FORCES_ONLY samples=\d+ max:\s*(.+)', text)
    if not m_max:
        raise ValueError(f"No 'FORCES_ONLY samples=N max:' line in {log_path}")

    m_time = re.search(r'FORCES_ONLY samples=\d+ time:\s*(.+)', text)
    if not m_time:
        raise ValueError(
            f"No 'FORCES_ONLY samples=N time:' line in {log_path} "
            "(binary needs rebuilding with per-level force-time instrumentation)")

    return (_parse_kv_line(m_avg.group(1)), _parse_kv_line(m_max.group(1)),
            _parse_kv_line(m_time.group(1)))


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


def times_to_vec(times_dict, n_masses):
    """Return [t_PF0, ..., t_PF{n-2}, t_Tail] (seconds), same order as forces."""
    vec = []
    for i in range(n_masses - 1):
        key = f'PF{i}'
        if key not in times_dict:
            raise ValueError(f"Missing {key} in times: {times_dict}")
        vec.append(times_dict[key])
    if 'Tail' not in times_dict:
        raise ValueError(f"Missing Tail in times: {times_dict}")
    vec.append(times_dict['Tail'])
    return vec


# ---------------------------------------------------------------------------
# Cost metric
# ---------------------------------------------------------------------------

def compute_metrics(forces_max, times, strange_fmax, strange_time):
    """Return dict with cost, f_max_light, f_max, t_light, t_total, phase.

    forces_max: per-level MAX (sup-norm) forces -- this is the quantity that
    physically bounds the integrator step size, and is the convention
    strange_fmax (~2.20) was measured in. (Phase-1's force-balance move rule
    uses AVG forces separately -- see propose_force_balance.)
    """
    f_max_light = max(forces_max)
    f_max = max(f_max_light, strange_fmax)
    t_light = sum(times)
    t_total = t_light + strange_time
    cost = f_max * t_total
    phase = 'force-balance' if f_max_light > strange_fmax else 'minimize-time'
    return dict(cost=cost, f_max_light=f_max_light, f_max=f_max,
                 t_light=t_light, t_total=t_total, phase=phase)


# ---------------------------------------------------------------------------
# Mass proposal
# ---------------------------------------------------------------------------

def propose_force_balance(masses, forces, step_frac, tail_max, min_gap):
    """
    Phase 1 (F_max_light > F_strange_max): coordinate update rule for N masses
    (masses[0] fixed = m_light), equalizing adjacent forces.

    For each free mass m_k (k = 1 .. N-1):
      - F_left  = forces[k-1]  : level whose right boundary is m_k
      - F_right = forces[k]    : level whose left boundary is m_k (or Tail)

    Moving m_k in the +ve direction (heavier, toward 0 or beyond) decreases
    F_right and increases F_left.  So the signed imbalance

        imb = (F_right - F_left) / (F_right + F_left)

    is positive when the right level dominates -> move m_k right (+ve step).

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

    return _enforce_bounds(new_masses, tail_max, min_gap)


def propose_minimize_time(masses, times, step_frac, tail_max, min_gap):
    """
    Phase 2 (F_max_light <= F_strange_max): F_max is pinned by the strange
    sector, so Cost ~ F_strange_max * (sum(t_light) + t_strange) -- minimize
    sum(t_light) by pushing each free mass m_k (k = 1 .. N-1) toward its right
    neighbour (heavier == further from the critical light mass m_light ==
    cheaper CG solve for the level whose lighter operand is m_k, namely
    PF{k} for k < N-1, or the Tail's own bare-det solve for k = N-1).

    Step size for m_k is weighted by times[k] / max(times[1:]) so the level
    contributing most to sum(t_light) (excluding the fixed-cost PF0, whose
    lighter operand is m_light and cannot be tuned) gets the largest nudge.

    This is self-limiting: if pushing masses heavier drives F_max_light back
    above F_strange_max, the next iteration's phase check reverts to
    propose_force_balance, which pulls the dominant mass back down.
    """
    N = len(masses)
    new_masses = list(masses)

    movable_times = times[1:]
    t_max = max(movable_times) if movable_times else 1.0
    if t_max <= 0:
        t_max = 1.0

    for k in range(1, N):
        right_bound = masses[k + 1] if k < N - 1 else tail_max
        weight = times[k] / t_max
        new_masses[k] = masses[k] + step_frac * weight * (right_bound - masses[k])

    return _enforce_bounds(new_masses, tail_max, min_gap)


def _enforce_bounds(new_masses, tail_max, min_gap):
    N = len(new_masses)

    # Enforce minimum gap: forward pass (light -> heavy)
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


def fmt_vals(vals):
    return '  '.join(f'{f:.6f}' for f in vals)


# ---------------------------------------------------------------------------
# Summary (--summary mode)
# ---------------------------------------------------------------------------

def print_summary(state_path):
    state = json.loads(Path(state_path).read_text())
    print(f"Converged    : {state.get('converged', False)}")
    print(f"Final masses : {fmt_ladder(state.get('masses', []))}")
    print(f"Best masses  : {fmt_ladder(state.get('best_masses', []))}"
          f"  (cost={state.get('best_cost', 'N/A')}, iter={state.get('best_iter', 'N/A')})")
    print(f"Strange consts: F_max={state.get('strange_fmax', 'N/A')}"
          f"  t_strange={state.get('strange_time', 'N/A')}s")
    print()
    print(f"{'iter':>4}  {'phase':>14}  {'cost':>10}  {'F_max':>7}  "
          f"{'t_light':>8}  {'balance':>8}  masses")
    for h in state.get('history', []):
        ms = ','.join(f'{m:.4f}' for m in h['masses'])
        print(f"{h['iter']:>4}  {h.get('phase', '?'):>14}  {h.get('cost', 0):>10.1f}  "
              f"{h.get('f_max', 0):>7.3f}  {h.get('t_light', 0):>8.1f}  "
              f"{h.get('balance', 0):>8.1f}  [{ms}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',    help='Path to FORCES_ONLY log')
    ap.add_argument('--masses', help='Comma-separated current masses')
    ap.add_argument('--state',  required=True, help='Path to state.json (read/write)')

    ap.add_argument('--strange-fmax', type=float, default=2.20,
                    help='Strange-sector max force (constant). Once F_max_light '
                         'drops to/below this, F_max is pinned here. Default '
                         '2.20, measured on cfg_2000 (job 54256171, eval#1).')
    ap.add_argument('--strange-time', type=float, default=392.0,
                    help='Strange-sector force-compute time in seconds '
                         '(StrangeLogDet + StrangeRHMC, constant). Default '
                         '392.0, measured on cfg_2000 (job 54256171, eval#1).')
    ap.add_argument('--cost-tol', type=float, default=0.02,
                    help='Converge when |cost - prev_cost|/prev_cost < this '
                         '(default 0.02 = 2%%)')

    ap.add_argument('--step-frac', type=float, default=0.3,
                    help='Phase-1 (force-balance) step fraction of available '
                         'span per iteration (default 0.3)')
    ap.add_argument('--time-step-frac', type=float, default=0.15,
                    help='Phase-2 (minimize-time) step fraction of available '
                         'span per iteration (default 0.15)')
    ap.add_argument('--tail-max', type=float, default=0.5,
                    help='Upper bound for Tail mass (default 0.5)')
    ap.add_argument('--min-gap', type=float, default=0.005,
                    help='Minimum gap between adjacent ladder masses (default 0.005)')

    ap.add_argument('--target-balance', type=float, default=5.0,
                    help='(legacy, informational only -- recorded in state.json '
                         'but no longer gates convergence; superseded by '
                         '--cost-tol on the Cost metric)')

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
        forces_avg_dict, forces_max_dict, times_dict = parse_forces_log(args.log)
        forces     = forces_to_vec(forces_avg_dict, len(masses))   # avg -- Phase-1 move rule
        forces_max = forces_to_vec(forces_max_dict, len(masses))   # max -- Cost / phase decision
        times      = times_to_vec(times_dict, len(masses))
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(forces_max, times, args.strange_fmax, args.strange_time)
    ratio = balance_ratio(forces)

    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {'iteration': 0, 'history': [],
         'strange_fmax': args.strange_fmax, 'strange_time': args.strange_time,
         'target_balance': args.target_balance}
    iteration = state.get('iteration', 0)
    prev_cost = state['history'][-1]['cost'] if state['history'] else None

    print(f'iter={iteration}  phase={metrics["phase"]}  cost={metrics["cost"]:.1f}  '
          f'(F_max={metrics["f_max"]:.4f}  t_total={metrics["t_total"]:.1f}s)')
    print(f'  masses     : {fmt_ladder(masses)}')
    print(f'  forces avg : {fmt_vals(forces)}   (balance={ratio:.1f})')
    print(f'  forces max : {fmt_vals(forces_max)}'
          f'   (F_max_light={metrics["f_max_light"]:.4f},'
          f' F_strange_max={args.strange_fmax})')
    print(f'  times  [s] : {fmt_vals(times)}'
          f'   (t_light={metrics["t_light"]:.1f}s, t_strange={args.strange_time:.1f}s)')

    entry = {'iter': iteration, 'masses': masses, 'forces': forces,
             'forces_max': forces_max, 'times': times, 'balance': round(ratio, 2)}
    entry.update(metrics)
    state['history'].append(entry)

    if 'best_cost' not in state or metrics['cost'] < state['best_cost']:
        state['best_cost']   = metrics['cost']
        state['best_masses'] = masses
        state['best_iter']   = iteration

    converged = (prev_cost is not None
                  and abs(metrics['cost'] - prev_cost) / abs(prev_cost) < args.cost_tol)

    if converged:
        state.update(converged=True, masses=masses)
        print(f'  CONVERGED (|Δcost|/cost < {args.cost_tol*100:.0f}%, phase={metrics["phase"]})')
        print(f'NEXT_LADDER={fmt_ladder(masses)}')
    else:
        if metrics['phase'] == 'force-balance':
            new_masses = propose_force_balance(masses, forces, args.step_frac,
                                                 args.tail_max, args.min_gap)
        else:
            new_masses = propose_minimize_time(masses, times, args.time_step_frac,
                                                 args.tail_max, args.min_gap)
        state.update(converged=False, masses=new_masses, iteration=iteration + 1)
        print(f'  proposed   : {fmt_ladder(new_masses)}')
        print(f'NEXT_LADDER={fmt_ladder(new_masses)}')

    state_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
