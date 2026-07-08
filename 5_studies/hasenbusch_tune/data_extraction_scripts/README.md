# data_extraction_scripts — hasenbusch_tune force-validation logs

Pull H / ΔH / acceptance / per-trajectory timing / plaquette / Polyakov loop out
of the HMC logs produced by the force-validation drivers in
`5_studies/hasenbusch_tune`, and assemble the grid-vs-quda comparison tables used
in `docs/2026_6_24_test_results.md` (Tables 3 and 3b).

Every physics number is copied **verbatim** from the logs (Grid computes them in
double precision). The only quantity these scripts *compute* is the steady-state
speedup ratio `grid_time / quda_time` — which is exactly the hand-arithmetic step
this tooling exists to remove.

## Why a separate parser from `4_analysis/hmc/extract.py`?

The package parser `4_analysis/hmc/extract.py` (used by the notebooks and the `hmc_obs` CLI)
targets a different Grid log layout — it keys off `-- # Trajectory = N` markers
and infers timing from timestamp diffs. The hasenbusch_tune drivers instead print
explicit `Total H before/after`, `Total time for trajectory (s):`, and a separate
`Metropolis_test` line, and never emit the `-- # Trajectory =` markers. Those logs
also contain stray non-UTF8 bytes that crash the package parser. So this is a
dedicated, dependency-free (stdlib-only), encoding-tolerant parser for *this*
driver format; it does not touch the notebook-facing code.

## Files

| file | role |
|---|---|
| `hmc_log_parser.py` | single-log parser — importable (`parse_log`) **and** a CLI |
| `extract_forces_run.py` | 6-run grid-vs-quda comparator → Table 3 / 3b Markdown |
| `integrator_time_split.py` | paired grid-vs-quda per-sector MD time split, averaged over steady trajectories — one table per driver's known level layout (hardcoded per code) |
| `traj_step_tables.py` | single-log, per-MD-step timing + force-max tables (see below) — driver-agnostic, no hardcoded level layout |
| `forces_only_tables.py` | `FORCES_ONLY`-mode logs (heatbath + one force eval per piece, no MD): per-level force/timing table for one log, candidate-scan comparison tables, and same-seed A/B diff (see below) |

## Environment

Needs only Python ≥ 3.6 (stdlib). On Perlmutter any of these works:

```bash
module load python/3.12-26.1.0     # or: cray-python/3.11.7
```

(No numpy/pandas needed. `4_analysis/requirements.txt` is for the parent `hmc` package.)

## Usage

Single log — quick per-trajectory dump (debugging / spot checks):

```bash
python hmc_log_parser.py <run_dir>/compact_QUDA.log
```

Whole 6-run set — emit the Markdown tables (and optionally a tidy CSV):

```bash
python extract_forces_run.py <run_dir> [--skip-first N] [--csv obs.csv]
```

`<run_dir>` is a Test-3 directory holding `{2plus1,compact,schur}_{GRID,QUDA}.log`,
e.g. `runs/2026_6_24_campaign/test3/hmc16_1222_20260625_102029/`.

`--skip-first N` (default 1) excludes the first N trajectories from the steady
per-trajectory time and the speedup: the first trajectory carries the one-time
QUDA autotune + first-call force init and is not representative of steady cost.

Per-MD-step timing + force tables for ONE log (real, non-`FORCES_ONLY` trajectory
mode) — "is a rung's force/solve-time blowing up over the course of a
trajectory, and which sector actually dominates wall time":

```bash
python traj_step_tables.py <hmc_log> [--traj N [N ...]] [--md]
```

Unlike `integrator_time_split.py` (which needs a paired grid+quda log per driver
and a hardcoded per-driver level layout), this works on a single log from any
driver/ladder-length: it parses the log's own `update_P : Level [L][I]
ActionName ...` blocks directly and auto-detects the MD-sub-step boundary from
the (level, idx) call-frequency pattern (see the script's docstring for the
exact rule). Prints, per trajectory found in the log: H_before/H_after/dH/
Metropolis outcome (when present — an incomplete/timed-out trajectory just
won't have H_after), a timing-per-sector-per-step table, a force-max-per-
sector-per-step table, and a sector-totals/percentage summary. `--traj N ...`
restricts to specific 0-based trajectory indices (default: all trajectories in
the log). Built for and validated against the compact_schur 48³ runaway
investigation, see
`docs/2026_7_3_compact_schur_48cube_optimized_traj_timing_forces.md`.

FORCES_ONLY logs (the ladder force-balance scans, `submit_scripts/hmc48_compact/
hasen_tail_force_scan_48.sh` and friends) — three sub-modes:

```bash
# one log → per-level table: mass | solver | F avg | F max | refresh s | deriv s
# (--samples adds the per-sample F-max table = draw-to-draw spread;
#  masses are negative, so use the equals form --ladder=-0.2416,...)
python forces_only_tables.py single <run_dir>/scan_base.log [--ladder=m1,...] [--samples]

# candidate dirs (each holding scan_<dirname>.log) → the interleaved
# mass+max-force table of docs/2026_7_8_hasen_tail_force_scan.md §1 plus
# heatbath-refresh and deriv-time tables. --cands takes any file with
# "name m1,m2,..." lines — the CANDS=( ... ) block of the scan submit
# script works verbatim. --eps adds a tail F·ε column.
python forces_only_tables.py scan runs/<scan>/{base,a1,...} \
    --cands submit_scripts/hmc48_compact/hasen_tail_force_scan_48.sh [--eps 0.0295]

# two same-seed logs → per-level force rel-diffs (match/DIFF verdict vs --tol,
# default 1e-3) + refresh speedups; exit code 0=PASS 1=FAIL. Built for the
# QUDA-vs-Grid heatbath A/B: ab <grid_heatbath_log> <quda_heatbath_log>
python forces_only_tables.py ab <logA> <logB> [--labels grid,quda] [--tol 1e-3]
```

The binary never prints the tail (LightSchurPF) mass, so mass labels come
from (in priority order) an `HASEN_LADDER=` echo on the log's first line —
the 2026-07-08+ scan submit scripts write this header — then `--ladder` /
`--cands`, then the `[Ladder] rung` lines (which cover every rung but the
tail). Incomplete runs (debug-queue timeout before the binary's `samples=N`
summary lines) degrade gracefully: the summary becomes the mean over the
samples that landed, flagged `INCOMPLETE`/`⚠`. Δcost/traj is deliberately NOT
computed — it embeds an integrator cost model (evals/traj); do that arithmetic
in the doc from the Σ refresh / Σ deriv columns.

## Importing the parser

```python
from hmc_log_parser import parse_log
run = parse_log('compact_QUDA.log')
run['initial_plaquette']            # float or None
for t in run['trajectories']:
    t['traj'], t['dH'], t['H_before'], t['H_after'], t['accepted']
    t['traj_time_s'], t['plaq'], t['plaq_smeared'], t['poly_re'], t['poly_im']
```

Missing observables are `None` (e.g. a loggerless run yields `plaq=None` but still
parses dH / H / timing / acceptance), so the parser degrades gracefully.
