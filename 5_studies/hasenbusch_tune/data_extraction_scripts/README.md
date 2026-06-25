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
