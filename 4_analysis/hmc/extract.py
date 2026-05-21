"""Parse Grid HMC log files into a tidy DataFrame.

Grid output structure for each trajectory N:
    HMC : -- # Trajectory = N
    HMC : Total H after trajectory = ...  dH = X.XXX
    Message : Plaquette: [ N+1 ] X.XXX          (config produced by traj N)
    Message : Plaquette: [ N+1 ] X.XXX           (second: smeared; skip)
    Message : Polyakov Loop: [ N+1 ] (re, im)
    HMC : Metropolis_test -- ACCEPTED/REJECTED   (absent for warmup trajs)
    HMC : -- # Trajectory = N+1
    ...

The config label N+1 in Plaquette/Polyakov matches the checkpoint filename
ckpoint_lat.N+1, so it is used as the primary trajectory index.
"""

import math
import re
from pathlib import Path

import pandas as pd

_RE_TRAJ  = re.compile(r'-- # Trajectory = (\d+)')
_RE_DH    = re.compile(r'\bdH = ([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)')
_RE_PLAQ  = re.compile(r'Plaquette: \[ (\d+) \] ([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)')
_RE_PLOOP = re.compile(r'Polyakov Loop: \[ (\d+) \] \(([^,]+),([^)]+)\)')
_RE_METRO = re.compile(r'Metropolis_test -- (ACCEPTED|REJECTED)')


def _parse_file(path):
    """Parse one HMC log file. Returns a list of per-trajectory dicts."""
    records = {}       # label -> dict
    pending_dH = None
    current_label = None  # label of the record currently being built

    with open(path) as fh:
        for line in fh:

            if _RE_TRAJ.search(line):
                # New trajectory starting: reset transient state.
                pending_dH = None
                current_label = None
                continue

            m = _RE_DH.search(line)
            if m:
                pending_dH = float(m.group(1))
                continue

            m = _RE_PLAQ.search(line)
            if m:
                label = int(m.group(1))
                if label not in records:   # first occurrence only (skip smeared)
                    dH = pending_dH
                    exp_dH = (math.exp(-dH) if dH < 700 else 0.0) if dH is not None else None
                    records[label] = {
                        'traj'     : label,
                        'dH'       : dH,
                        'exp_dH'   : exp_dH,
                        'plaquette': float(m.group(2)),
                        'accepted' : None,   # filled by METRO line below
                    }
                    current_label = label
                continue

            m = _RE_PLOOP.search(line)
            if m:
                label = int(m.group(1))
                if label in records:
                    re_v = float(m.group(2))
                    im_v = float(m.group(3))
                    records[label].update({
                        'polyakov_re' : re_v,
                        'polyakov_im' : im_v,
                        'polyakov_abs': math.hypot(re_v, im_v),
                    })
                continue

            m = _RE_METRO.search(line)
            if m and current_label is not None:
                records[current_label]['accepted'] = (m.group(1) == 'ACCEPTED')
                current_label = None
                continue

    return list(records.values())


def load_run(run_dir, pattern='hmc_traj*.log'):
    """
    Load all HMC log files matching *pattern* from *run_dir*.

    Files are sorted by filename (trajectory range) before parsing, so
    extended runs with multiple log files are merged in the correct order.
    Duplicate trajectory labels across files are deduplicated (first wins).

    Parameters
    ----------
    run_dir : str or Path
        Directory containing the log files.
    pattern : str
        Glob pattern (default 'hmc_traj*.log'). Use '*.log' for legacy
        directories where all logs share the same folder.

    Returns
    -------
    pd.DataFrame
        Columns: traj, dH, exp_dH, plaquette, polyakov_re, polyakov_im,
                 polyakov_abs, accepted
        Sorted by traj, one row per config label.
    """
    run_dir = Path(run_dir)
    logs = sorted(run_dir.glob(pattern))
    if not logs:
        raise FileNotFoundError(f"No files matching '{pattern}' in {run_dir}")

    seen = set()
    rows = []
    for log in logs:
        for rec in _parse_file(log):
            t = rec['traj']
            if t not in seen:
                seen.add(t)
                rows.append(rec)

    df = pd.DataFrame(rows).sort_values('traj').reset_index(drop=True)
    return df


def save_csv(df, path):
    """Write observable DataFrame to CSV."""
    df.to_csv(path, index=False, float_format='%.10g')
