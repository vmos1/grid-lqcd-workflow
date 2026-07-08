#!/usr/bin/env python3
"""Per-MD-step timing and force tables from a single (real, non-FORCES_ONLY)
HMC log -- one pair of Markdown tables (timing, force max) per trajectory,
plus a sector-totals summary and the trajectory's H_before/H_after/dH/
Metropolis outcome when available.

Built for the diagnostic workflow used on the compact_schur 48^3 runaway
investigation (docs/2026_7_3_compact_schur_48cube_optimized_traj_timing_forces.md):
"is a rung's force/solve-time blowing up over the course of a trajectory, and
which sector actually dominates wall time." Works on any of the hasenbusch_tune
drivers (2plus1 / compact / compact_schur) and any Hasenbusch ladder length --
it does NOT hardcode a level/action layout per driver (unlike
integrator_time_split.py, which is a paired grid-vs-quda campaign averager and
needs one). Instead it parses the log's own
``update_P : Level [L][I] ActionName ...`` blocks directly and auto-detects
the MD-sub-step boundary:

  1. Count total occurrences of each (level, idx) key across the trajectory.
  2. The single most-frequent key is the finest (innermost, most sub-stepped)
     level -- typically gauge. It's excluded from boundary detection; its
     repeated calls within one step are summed (time) / maxed (force).
  3. Among the remaining keys, the most-frequent one is the natural "MD
     sub-step" cycle marker (e.g. the light sector's first monomial in a
     3-level strange/light/gauge integrator, or a fermion monomial in a flat
     2-level integrator). A new step starts each time this key repeats.
  4. Any other key (e.g. a coarser strange level in a 3-level integrator)
     just appears in whichever step it falls into, sparsely.

This mirrors exactly how the tables in the doc above were built by hand.

Before the two step tables, a "Setup / one-time costs" block reports
everything that happens ONCE per trajectory rather than once per MD step (so
it doesn't distort the per-step numbers/percentages):

  - Heatbath (refresh) per action, from the log's ``refresh [L][I]
    ActionName`` markers -- the pseudofermion draw at the start of the
    trajectory. Governed by a normal-equation CG solve for a Hasenbusch ratio
    rung (see docs/2026_7_2_hasenbusch_heatbath_solve_derivation.md), so it
    can be substantial for a near-critical rung even though it's a one-time
    cost.
  - Initial action eval (S()) per action, from ``S [L][I] action eval``
    markers -- needed once to compute ``Total H before trajectory``.
  - MG full preconditioner builds, from ``QudaCloverInverter: MG
    preconditioner built (N levels) setup=X s`` -- attributed to whichever
    (level, idx) block was open when the line appeared. Subsequent
    "MG thin-updated" calls (near-zero cost) are NOT one-time and are left in
    the regular per-step table.
  - Any per-step sector whose FIRST occurrence is >3x the median of its own
    later occurrences is flagged as carrying a hidden one-time cost (e.g.
    gauge's first call always includes the one-time stencil/padded-cell
    build, memory gauge-force-48cube-slow) -- the step table itself is left
    unmodified (so raw numbers stay trustworthy), just annotated.

Usage:
  traj_step_tables.py <hmc_log> [--traj N [N ...]] [--md]

  --traj N [N ...]   only show these trajectory indices (0-based, in the order
                      they appear in the log -- NOT necessarily Grid's own
                      trajectory numbering, which may skip if resuming).
                      Default: show every trajectory found in the log.
  --md                write pipe-style Markdown tables (default: on when
                      stdout is not a terminal, else a plain aligned table).
"""
import re
import sys
import argparse
import statistics
from collections import defaultdict

UPDATE_P_RE = re.compile(r'update_P : Level \[(\d+)\]\[(\d+)\] (\S+)(.*)$')
TIME_RE     = re.compile(r'\[(\d+)\]\[(\d+)\] P update elapsed time: ([\d.]+) ms')
FMAX_RE     = re.compile(r'\[(\d+)\]\[(\d+)\] Force max\s*: ([\d.eE+-]+)')
FAVG_RE     = re.compile(r'\[(\d+)\]\[(\d+)\] Force average: ([\d.eE+-]+)')
TRAJ_RE     = re.compile(r'-- # Trajectory = (-?\d+)')
HBEF_RE     = re.compile(r': ([\d.]+) s : Total H before trajectory\s*=\s*([\d.eE+-]+)')
HAFT_RE     = re.compile(r'Total H after trajectory\s*=\s*([\d.eE+-]+)\s+dH\s*=\s*([\d.eE+-]+)')
METRO_RE    = re.compile(r'Metropolis_test -- (\w+)')
SKIP_RE     = re.compile(r'Skipping Metropolis test')
REFRESH_RE  = re.compile(r': ([\d.]+) s : refresh \[(\d+)\]\[(\d+)\] (\S+)(.*)$')
SEVAL_RE    = re.compile(r': ([\d.]+) s : S \[(\d+)\]\[(\d+)\] action eval')
MGBUILD_RE  = re.compile(r': ([\d.]+) s : QudaCloverInverter: MG preconditioner built '
                          r'\((\d+) levels\) setup=([\d.]+) s')

# Solver convergence lines carry NO [L][I] tag of their own, so they can only
# be attributed to "whichever rung is currently pending" (see the parse()
# loop) -- which is reliable ONLY for Grid-native solvers. Grid's own
# ConjugateGradient/MixedPrecisionConjugateGradient print through the same
# timestamped, ordered iostream as update_P/Force/P-update-elapsed, so their
# position in the log is trustworthy, same guarantee as the timing/force
# columns. QUDA's own convergence lines (GCR:/CG: Convergence/MultiShiftCG:)
# carry NO timestamp at all -- confirmed by direct inspection that they can
# appear inside the WRONG rung's window (e.g. two `GCR: Convergence` lines
# were found inside a plain log-det block that involves no CG/MG solve at
# all) -- QUDA prints straight to stdout from the C library, independently
# buffered from Grid's own synchronized output. So on principle (this
# script's data must be reliable, not just plausible-looking) we do NOT
# attempt to extract QUDA-native iteration counts at all; ITER_RES only
# covers patterns proven safe. Any rung whose solve actually went through
# QUDA (any HASEN_MG_RUNG/HASEN_QUDA_CG_RUNGS rung, or a QUDA-accelerated
# strange) will show "see raw log" in the iterations table instead of a
# number -- see print_table's raw_log_fallback.
ITER_RES = [
    re.compile(r'ConjugateGradient Converged on iteration (\d+)'),  # Grid-native CG
]
MPCG_ITER_RE = re.compile(
    r'MixedPrecisionConjugateGradient: Inner CG iterations (\d+) '
    r'Restarts \d+ Final CG iterations (\d+)')


def short_label(level, idx, action, extra):
    """Compact column label: 'det(m0)/det(m1)' for a Hasenbusch ratio rung
    (from the trailing context Grid prints after the class name), else the
    class name with common suffixes trimmed."""
    extra = extra.strip()
    m = re.search(r'det\(([-\d.]+)\)\s*/\s*det\(([-\d.]+)\)', extra)
    if 'Ratio' in action and m:
        return f"[{level}][{idx}] det({m.group(1)})/det({m.group(2)})"
    short = action
    for junk in ('TwoFlavourSchurClover', 'OneFlavourSchurClover',
                 'QCDLogDetCompactClover', 'QCDLogDetClover', 'Action'):
        short = short.replace(junk, '')
    return f"[{level}][{idx}] {short or action}"


def parse(path):
    """Returns a list of trajectory dicts: {traj, h_before, h_before_t,
    h_after, dH, metropolis, events, refresh, seval, mg_builds}.
    `events` feeds group_into_steps() (per-MD-step table); `refresh`/`seval`
    are ordered (timestamp, label) lists for the one-time heatbath/action-eval
    costs; `mg_builds` is a list of (timestamp, levels, setup_s, label)."""
    trajs = []
    cur = None

    def new_traj():
        nonlocal cur
        if cur is not None and (cur['events'] or cur['h_before'] is not None):
            trajs.append(cur)
        cur = {'traj': None, 'h_before': None, 'h_before_t': None,
               'h_after': None, 'dH': None, 'metropolis': None,
               'events': [], 'refresh': [], 'seval': [], 'mg_builds': []}
    new_traj()

    last_label = None  # most recent (level,idx) context, for MG-build attribution
    pending_key = None
    with open(path, errors='ignore') as fh:
        for line in fh:
            mtraj = TRAJ_RE.search(line)
            if mtraj:
                new_traj()
                cur['traj'] = int(mtraj.group(1))
                continue
            mhb = HBEF_RE.search(line)
            if mhb:
                cur['h_before_t'] = float(mhb.group(1))
                cur['h_before'] = float(mhb.group(2))
                continue
            mha = HAFT_RE.search(line)
            if mha:
                cur['h_after'] = float(mha.group(1))
                cur['dH'] = float(mha.group(2))
                continue
            mm = METRO_RE.search(line)
            if mm:
                cur['metropolis'] = mm.group(1)
                continue
            if SKIP_RE.search(line):
                cur['metropolis'] = 'SKIPPED'
                continue
            mr = REFRESH_RE.search(line)
            if mr:
                t, lvl, idx, action, extra = mr.groups()
                last_label = short_label(lvl, idx, action, extra)
                cur['refresh'].append((float(t), last_label))
                continue
            ms = SEVAL_RE.search(line)
            if ms:
                t, lvl, idx = ms.groups()
                # label unknown at this line (class name isn't printed here);
                # reuse whatever label this (level,idx) had from refresh/update_P.
                last_label = f"[{lvl}][{idx}]"
                for src in (cur['refresh'], cur['events']):
                    for item in src:
                        lab = item[1]
                        if lab.startswith(f"[{lvl}][{idx}]"):
                            last_label = lab
                cur['seval'].append((float(t), last_label))
                continue
            mg = MGBUILD_RE.search(line)
            if mg:
                t, levels, setup = mg.groups()
                cur['mg_builds'].append((float(t), levels, float(setup), last_label))
                continue
            mu = UPDATE_P_RE.search(line)
            if mu:
                lvl, idx, action, extra = mu.groups()
                pending_key = (lvl, idx)
                last_label = short_label(lvl, idx, action, extra)
                cur['events'].append([pending_key, last_label, 0.0, None, None, None])
                continue
            if pending_key is None or not cur['events']:
                continue
            mt = TIME_RE.search(line)
            if mt and (mt.group(1), mt.group(2)) == pending_key:
                cur['events'][-1][2] += float(mt.group(3)) / 1000.0
                continue
            mfx = FMAX_RE.search(line)
            if mfx and (mfx.group(1), mfx.group(2)) == pending_key:
                cur['events'][-1][3] = float(mfx.group(3))
                continue
            mfa = FAVG_RE.search(line)
            if mfa and (mfa.group(1), mfa.group(2)) == pending_key:
                cur['events'][-1][4] = float(mfa.group(3))
                continue
            # Solver convergence lines carry no [L][I] tag -- attribute to
            # whichever rung is currently pending (see ITER_RES comment).
            mmp = MPCG_ITER_RE.search(line)
            if mmp:
                cur['events'][-1][5] = int(mmp.group(1)) + int(mmp.group(2))
                continue
            for iter_re in ITER_RES:
                mi = iter_re.search(line)
                if mi:
                    cur['events'][-1][5] = int(mi.group(1))
                    break
    new_traj()
    return trajs


def group_into_steps(events):
    """Auto-detect the MD-sub-step cycle key (see module docstring) and fold
    the flat event list into a list of {label: {time, fmax, iters}} step
    dicts. `iters` is the solver's own reported iteration count (see
    ITER_RES/MPCG_ITER_RE) -- None when a step's solver didn't print one
    (e.g. gauge, which has no iterative solve)."""
    if not events:
        return []
    counts = defaultdict(int)
    for key, *_ in events:
        counts[key] += 1
    finest_key = max(counts, key=lambda k: counts[k])
    rest = {k: v for k, v in counts.items() if k != finest_key}
    cycle_key = max(rest, key=lambda k: rest[k]) if rest else finest_key

    steps = []
    cur = {}
    for key, label, t, fmax, favg, iters in events:
        if key == cycle_key and cur:
            steps.append(cur)
            cur = {}
        d = cur.setdefault(label, {'time': 0.0, 'fmax': None, 'iters': None})
        d['time'] += t
        if fmax is not None:
            d['fmax'] = fmax if d['fmax'] is None else max(d['fmax'], fmax)
        if iters is not None:
            d['iters'] = iters
    if cur:
        steps.append(cur)
    return steps


def durations(ordered, end_time):
    """ordered: list of (timestamp, label), already in file order. Returns
    [(label, duration_s), ...] via consecutive-timestamp differencing, last
    one ending at `end_time` (None entries dropped)."""
    out = []
    for i, (t, label) in enumerate(ordered):
        tnext = ordered[i + 1][0] if i + 1 < len(ordered) else end_time
        if tnext is None:
            continue
        out.append((label, tnext - t))
    return out


def print_setup_costs(t, md):
    refresh_end = t['seval'][0][0] if t['seval'] else t['h_before_t']
    ref = durations(t['refresh'], refresh_end)
    sev = durations(t['seval'], t['h_before_t'])
    if not (ref or sev or t['mg_builds']):
        return
    print("\n### Setup / one-time costs (once per trajectory, not per MD step)\n")
    if ref:
        print("Heatbath (refresh), per action:")
        for label, d in ref:
            print(f"  {label:<40} {d:>9.2f}s")
        print(f"  {'TOTAL heatbath':<40} {sum(d for _, d in ref):>9.2f}s")
    if sev:
        print("\nInitial action eval (S(), for H_before), per action:")
        for label, d in sev:
            print(f"  {label:<40} {d:>9.2f}s")
        print(f"  {'TOTAL action eval':<40} {sum(d for _, d in sev):>9.2f}s")
    if t['mg_builds']:
        print("\nMG full preconditioner builds:")
        for tt, levels, setup, label in t['mg_builds']:
            print(f"  {label or '(unknown rung)':<40} {levels} levels, setup={setup:.2f}s")


def flag_one_time_inflation(steps, cols):
    """For each column, compare its first occurrence's time to the median of
    its later occurrences; flag if first > 3x median (e.g. gauge's one-time
    stencil/padded-cell build on the first call)."""
    flags = []
    for c in cols:
        vals = [s[c]['time'] for s in steps if c in s]
        if len(vals) < 4:
            continue
        first, rest = vals[0], vals[1:]
        med = statistics.median(rest)
        if med > 0 and first > 3 * med:
            flags.append((c, first, med))
    return flags


def ordered_columns(steps):
    """Column order: first-seen order across all steps."""
    seen = []
    for s in steps:
        for label in s:
            if label not in seen:
                seen.append(label)
    return seen


RAW_LOG_TIME_THRESHOLD_S = 1.0  # a real solve is presumed to cost at least this;
                                 # below it, an empty cell is presumed genuinely
                                 # N/A (no iterative solve involved) rather than
                                 # "count exists but wasn't safely extractable"


def print_table(title, steps, cols, field, fmt, md, raw_log_fallback=False):
    """raw_log_fallback: when True (only used for the iterations table),
    an empty cell whose sector cost >RAW_LOG_TIME_THRESHOLD_S prints
    'see raw log' instead of blank -- see the ITER_RES comment for why: only
    Grid-native solver counts are safe to extract, so a QUDA-routed rung's
    (real, nonzero) iteration count is deliberately NOT guessed at."""
    sep = ' | ' if md else '  '
    edge = '|' if md else ''
    header = [f"{edge}{'step':>4}"] + [f"{c:>14}" for c in cols]
    print(f"\n{title}\n")
    print((sep.join(header) + (edge if md else '')))
    if md:
        print('|' + '---|' * (len(cols) + 1))
    for i, s in enumerate(steps, 1):
        row = [f"{edge}{i:>4}"]
        for c in cols:
            cell = s.get(c, {})
            v = cell.get(field)
            if v is not None:
                row.append(f"{fmt(v):>14}")
            elif raw_log_fallback and cell.get('time', 0.0) > RAW_LOG_TIME_THRESHOLD_S:
                row.append(f"{'see raw log':>14}")
            else:
                row.append(f"{'':>14}")
        print(sep.join(row) + (edge if md else ''))


def print_totals(steps, cols):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for s in steps:
        for c in cols:
            if c in s:
                totals[c] += s[c]['time']
                counts[c] += 1
    grand = sum(totals.values()) or 1.0
    print("\nSector totals (this trajectory)\n")
    for c in sorted(cols, key=lambda c: -totals[c]):
        print(f"  {c:<28} calls={counts[c]:<4} total={totals[c]:>9.1f}s  "
              f"{100*totals[c]/grand:>5.1f}%")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log')
    ap.add_argument('--traj', type=int, nargs='+', default=None,
                     help='0-based trajectory indices to show (default: all)')
    ap.add_argument('--md', action='store_true', help='force Markdown table output')
    args = ap.parse_args(argv)

    md = args.md or not sys.stdout.isatty()
    trajs = parse(args.log)
    if not trajs:
        print(f"No trajectories found in {args.log}", file=sys.stderr)
        return 1

    indices = args.traj if args.traj is not None else range(len(trajs))
    for i in indices:
        if i >= len(trajs):
            print(f"(trajectory index {i} not found -- only {len(trajs)} in this log)",
                  file=sys.stderr)
            continue
        t = trajs[i]
        steps = group_into_steps(t['events'])
        cols = ordered_columns(steps)
        traj_no = t['traj'] if t['traj'] is not None else i
        print(f"\n{'='*70}\nTrajectory {traj_no}  "
              f"({len(steps)} MD sub-steps captured)")
        if t['h_before'] is not None:
            print(f"H_before = {t['h_before']}")
        if t['h_after'] is not None:
            print(f"H_after  = {t['h_after']}   dH = {t['dH']}")
        else:
            print("H_after  = (trajectory did not complete in this log)")
        if t['metropolis']:
            print(f"Metropolis: {t['metropolis']}")

        print_setup_costs(t, md)

        flags = flag_one_time_inflation(steps, cols)
        if flags:
            print("\nNote: one-time cost detected in the per-step table below "
                  "(not removed from the raw numbers, just flagged):")
            for c, first, med in flags:
                print(f"  {c:<28} first occurrence {first:.1f}s vs steady "
                      f"median {med:.1f}s  (+{first-med:.1f}s one-time, "
                      f"likely a first-call init cost)")

        print_table("### Timing per sector per step (s)", steps, cols, 'time',
                     lambda v: f"{v:.1f}", md)
        print_table("### Force max per sector per step", steps, cols, 'fmax',
                     lambda v: f"{v:.4f}", md)
        # Solver-iteration extraction was attempted and REMOVED: even
        # Grid-native diagnostic lines (e.g. MixedPrecisionConjugateGradient)
        # were found to land inside the WRONG rung's [L][I] window (a
        # structural print-ordering issue, not just QUDA's unstamped lines --
        # see git history / session notes). Rather than ship a heuristic that
        # only partially works, this table is not produced until there's a
        # source of iteration counts that's actually tied to [L][I] (e.g.
        # instrumenting the driver itself to print it alongside Force max).
        print_totals(steps, cols)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
