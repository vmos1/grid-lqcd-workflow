# 4_analysis — HMC observable analysis toolkit

Python package and CLI for extracting physics observables from Grid HMC log files
and computing statistical quantities.

---

## Setup

```bash
module load mambaforge/23.1.0-4
conda activate /lustre2/nplqcd/vayyar/conda-envs/hmc-analysis
# or: pip install -r requirements.txt
```

---

## CLI — `hmc_obs`

Three subcommands, all operating on a run directory containing `hmc_traj*.log` files:

```bash
# Parse logs, print observable table, optionally save to CSV
hmc_obs extract <run_dir> [-o obs.csv]

# Integrated autocorrelation time for one observable (Gamma/UW method)
hmc_obs autocorr <obs.csv> <column> [--burnin N] [--S 1.5]

# Full report: accept rate, auto burn-in, tau_int for all observables
hmc_obs summary <run_dir> [--burnin N]
```

For legacy log directories (not yet using the `hmc_traj*.log` naming):

```bash
hmc_obs extract /path/to/logs --pattern 'mobius-eofa-*.log'
```

---

## Python module

```python
import sys
sys.path.insert(0, '/lustre2/nplqcd/vayyar/grid_qcd/grid-lqcd-workflow/4_analysis')
from hmc import load_run, gamma_method, suggest_burnin

df = load_run('runs/mobius_eofa/hmc')       # returns a pandas DataFrame
print(df[['traj', 'plaquette', 'dH', 'accepted']].head())

# Integrated autocorrelation time
result = gamma_method(df['plaquette'].dropna().values)
print(f"tau_int = {result['tau_int']:.3f} +/- {result['tau_int_err']:.3f}")

# Suggested burn-in trajectory
burnin = suggest_burnin(df)
df_eq  = df[df['traj'] >= burnin]
```

---

## Modules

### `hmc/extract.py`

Parses Grid HMC log files into a tidy DataFrame with one row per trajectory.

Extracted columns: `traj`, `dH`, `exp_dH`, `plaquette`, `polyakov_re`,
`polyakov_im`, `polyakov_abs`, `accepted`.

Handles multiple log files from extended runs — sorts by trajectory number
extracted from file content, not filename, so SLURM job ID ordering is
irrelevant.

### `hmc/autocorr.py` — Gamma / UW method

Computes the integrated autocorrelation time τ_int using the automatic-windowing
algorithm of Madras & Sokal (1988), as described in:

> U. Wolff (ALPHA Collaboration),  
> *"Monte Carlo errors with less errors"*,  
> Comput. Phys. Commun. **156** (2004) 143, [hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017)

**τ_int** measures how many HMC trajectories are needed between statistically
independent samples. The naive error on the mean underestimates the true error
by a factor of √(2τ_int). A value of τ_int ≈ 3 means you effectively have
N/6 independent samples from N trajectories.

The window parameter `S` (default 1.5) controls the bias/variance tradeoff of
the window estimator. Increase it if τ_int appears underestimated.

### `hmc/equilibrate.py`

Suggests a burn-in cut: the first trajectory after which the rolling mean of
an observable stabilises within a fractional tolerance of the long-run mean.
This is a heuristic — always verify by eye by plotting the observable vs trajectory.

---

## Planned additions

- Plotting: plaquette history, ACF, running mean overlay
- Autocorrelation matrix across multiple observables
- Formal equilibration tests (Gelman-Rubin, stationarity)
