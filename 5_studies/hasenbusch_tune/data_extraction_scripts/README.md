# data_extraction_scripts — hasenbusch_tune force-validation logs

Pull H / ΔH / acceptance / per-trajectory timing / plaquette / Polyakov loop out
of the HMC logs produced by the force-validation drivers in
`5_studies/hasenbusch_tune`, and assemble the grid-vs-quda comparison tables used
in `__docs/2026_6_24_test_results.md` (Tables 3 and 3b).

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
| `forces_only_ladder_cost.py` | **decision layer on top of the above**: applies a per-rung calibration `k = realFdt/screenedF` to turn screened forces into predicted real `F·dt`, then into the minimum integer MDSTEPS a candidate supports and the resulting trajectory seconds. Also fits `F_PFk ~ gap_k^a` across a candidate set (`--exponents`). Answers "is this candidate worth a trajectory run", which the raw table cannot |

`forces_only_tables.py` reports *what was measured*; `forces_only_ladder_cost.py`
reports *what it is worth*. The distinction matters because the two disagree:
ranking on raw screened force scored a 49% regression as a 1% one (C2 round 1),
and a candidate can cut the calibrated peak 10% while being worth **0 seconds**
because MDSTEPS is an integer. Its defaults are calibrated to w3@MDSTEPS=5 and
are **not portable to another ladder family** — pass `--k`, or `--real-fdt` +
`--calib-log` + `--baseline`, to recalibrate. The calibration in force is
printed on every run.

```bash
python3 forces_only_ladder_cost.py runs/<scan>/{cand1,cand2,...} --baseline <ref>
python3 forces_only_ladder_cost.py runs/<scan>/* --baseline <ref> --exponents
```

## Acceptance-run family (real HMC trajectories, not FORCES_ONLY)

Everything above reads `FORCES_ONLY` screening logs. These read **real
acceptance-run logs** — full trajectories with a Metropolis step — and answer a
different question: "did this candidate actually work, and how fast was it".

| file | role |
|---|---|
| `accept_run_tables.py` | the log parser for acceptance runs, importable (`parse`, `build_columns`, `buckets`) and a CLI. Per-trajectory physics + force-time split (Table 1) and per-piece `max/avg` `\|F·dt\|` (Table 2). `--csv` writes a tidy CSV |
| `compare_runs_multiwindow.py` | **N-run cross-comparison**, each run spanning several checkpoint-resumed log windows. Imports the parser above — no parsing is reimplemented. Emits config / summary / efficiency / observable / per-sector tables plus every run's Table 1 + Table 2 |
| `obs_autocorr_compare.py` | autocorrelation-corrected observable comparison from that CSV: `τ_int` (Madras–Sokal automatic windowing), corrected errors, block errors, and pairwise z-scores vs the first run. `--pooled` adds a second report that averages ρ(t) **across** runs before windowing (see below) |
| `plot_accept_runs.py` | the 2×2 per-trajectory figure (wall time / plaquette / ΔH / max kick) from the tidy CSV. Writes PNG + PDF |
| `regen_c1c2c3_comparison.sh` | **one command that rebuilds the whole base+G / C1 / C2 / C3 comparison** — tables, pooled-τ report and figure — from the raw logs. It *is* the run manifest: which log windows make up each run and which ladder each ran, which nothing else records in executable form |

Two things worth knowing before reaching for these:

* **`accept_run_tables.py --csv` cannot stitch a parent run to its
  `CKPT_RESUME_TRAJ` extension** — it restarts trajectory numbering at 2001 per
  file, and `run_name()` collapses distinct labels (`u1a`→`u1`,
  `3level`→`3-level`). Use `compare_runs_multiwindow.py` for any run that spans
  more than one log.
* **Do not quote naive σ/√N errors on observables from these chains.**
  Consecutive trajectories at τ=0.354 are correlated, so the naive error is
  biased small and will report a spurious multi-σ disagreement between two
  chains that sample the same distribution. This is not hypothetical: it put
  C1's smeared plaquette at +3.28σ from the baseline, versus +1.69σ once
  corrected. Run `obs_autocorr_compare.py` and quote its corrected column.
* **`--pooled` when — and only when — the runs share an action.** At N=20 the
  per-run τ_int is itself noisy, and it is noisy in a way *correlated* with the
  mean it corrects, so it can inflate one arm's error more than another's for no
  physical reason. If the runs share an action and trajectory length they share
  one true τ_int, so pooling ρ(t) over them is less noisy and applies a single
  inflation factor to every arm. That holds for C1/C2/C3 vs base+G (same action,
  different integrator); it does **not** hold across codebases, so the
  Grid-vs-Chroma comparison uses the per-run mode.

`plot_accept_runs.py` exposes module-level knobs so a caller can retarget it
without forking: `PALETTE`, `DISPLAY`, `TITLE`, `DH_SCALE` (`"symlog"` default,
set `"linear"` when no run blows up — symlog's `linthresh=1` squashes O(1) ΔH
into the linear stub around zero), `KICK_BAND` / `KICK_BAND_LABEL` (panel D's
shaded reference, default the nominal 0.10; set it to a specific run's own
proven-safe max when that is the comparison being made) and `LEGEND_NCOL`
(default auto: one row up to 3 runs, 2 columns past that — four of these labels
on one row overrun the canvas and are silently clipped at both edges).
`plot_c1c2c3_baseline.py` is a worked example of such a driver.

The whole four-run comparison, in one command:

```bash
./regen_c1c2c3_comparison.sh              # tables + pooled-τ report + figure
```

or step by step, if only one piece is needed:

```bash
python3 compare_runs_multiwindow.py \
    --run 'base+G=<parent.log>,<ext.log>' \
    --run 'C1 (u1a)=<parent.log>,<ext.log>' \
    --ladder 'base+G=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870' \
    --csv out.csv
python3 obs_autocorr_compare.py out.csv --pooled
python3 plot_c1c2c3_baseline.py out.csv <fig>.png
```

Worked output: `__docs/2026_8_20_c1_c2_c3_vs_baseg_comparison.md`.

Only `plot_accept_runs.py` (and drivers over it) needs third-party packages —
`pandas` + `matplotlib`, via `module load python/3.12-26.1.0`. Everything else
is stdlib-only, which is why `regen_c1c2c3_comparison.sh` skips just the figure
step with a warning if they are missing rather than failing.

## GPU memory traces (not a log parser)

`memtrace_peaks.py` reads the `nvidia-smi` polling traces written by
`submit_scripts/hasenbusch_tune/2026_8_20_mem_wrapper.sh` — one
`memtrace_<node>.log` per node, one GPU sampled per node at a 2 s poll. With
`--gpus-per-task=1` and 4 ranks/node the sampled GPU's usage *is* one rank's
exclusive usage, so the peak it reports is a clean per-rank figure.

```bash
python3 memtrace_peaks.py <run_dir> [<run_dir> ...]
```

Prints per node: peak MiB used, card total, seconds-into-run at which the peak
landed, and headroom left. Written for the x5a2b 4-node OOM diagnosis, where it
established that memory climbs ~5667 MiB per Hasenbusch rung and that the
device-memory pool costs a flat ~10.4 GiB regardless of ladder — see
`__docs/2026_8_21_x5a2b_oom_diagnosis.md`.

Two traps it encodes, both of which produced plausible-looking wrong answers
first: the node label must come from `os.path.basename` (the string
`memtrace_` appears in both the run-dir name and the file name, so a naive
`split` collapses every node onto one key), and a **host**-RAM OOM never says
"out of memory" in the log — it surfaces as an OFI transport cascade, and only
`sacct` reveals it.

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
`__docs/2026_7_3_compact_schur_48cube_optimized_traj_timing_forces.md`.

FORCES_ONLY logs (the ladder force-balance scans, `submit_scripts/hmc48_compact/
hasen_tail_force_scan_48.sh` and friends) — three sub-modes:

```bash
# one log → per-level table: mass | solver | F avg | F max | refresh s | deriv s
# (--samples adds the per-sample F-max table = draw-to-draw spread;
#  masses are negative, so use the equals form --ladder=-0.2416,...)
python forces_only_tables.py single <run_dir>/scan_base.log [--ladder=m1,...] [--samples]

# candidate dirs (each holding scan_<dirname>.log) → the interleaved
# mass+max-force table of __docs/2026_7_8_hasen_tail_force_scan.md §1 plus
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
