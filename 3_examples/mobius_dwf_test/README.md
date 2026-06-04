# mobius_dwf_test — 1-flavor Möbius DWF: EOFA vs RHMC

Validates the 1-flavor Möbius domain wall fermion HMC implementation by comparing
two independent algorithms that represent the same fermion determinant.

Both codes represent **det(M_phys) / det(M_PV)** where M_PV has mass=1.0 (Pauli-Villars
regulator). They must produce identical expectation values for all observables if
the implementations are correct.

| Binary | Algorithm | Pseudofermion action |
|---|---|---|
| `dweofa_mobius` | Exact One Flavour Ratio (EOFA) | `ExactOneFlavourRatioPseudoFermionAction` |
| `dwrhmc_mobius` | Rational HMC (RHMC) | `OneFlavourEvenOddRatioRationalPseudoFermionAction` |

**Boundary conditions:** `{1, 1, 1, -1}` — periodic in x,y,z, antiperiodic in t.

---

## Build

```bash
cd 3_examples/mobius_dwf_test
bash build.sh          # compiles against install-txqcd-gpu by default
```

Override the Grid install:
```bash
GRID_CONFIG=/path/to/grid-config bash build.sh
```

---

## Run (lq cluster)

Both algorithms share the same `input.xml`. Run directories are separate so their
checkpoints don't collide.

```bash
# First run (cold start):
mkdir -p $BASE_DIR/runs/mobius_eofa/hmc  $BASE_DIR/runs/mobius_rhmc/hmc
cp inputs/ip_hmc_test.xml $BASE_DIR/runs/mobius_eofa/hmc/input.xml
cp inputs/ip_hmc_test.xml $BASE_DIR/runs/mobius_rhmc/hmc/input.xml
sbatch $HOME/projects/grid_qcd/jobs/run-eofa.sbatch
sbatch $HOME/projects/grid_qcd/jobs/run-rhmc.sbatch
```

Each job writes to its run directory:
- `hmc_traj{START}-{END}.log` — Grid physics output
- `slurm-<jobid>.log`         — job wrapper / error output
- `run_info_traj{N}.txt`      — provenance record
- `ckpoint_lat.*`, `ckpoint_rng.*` — gauge configs and RNG state

---

## Extending a run

Use `hmc_extend` to safely prepare the next job. It finds the last complete
checkpoint automatically — safe even if a job was killed mid-run.

```bash
cd $BASE_DIR/grid-lqcd-workflow/4_analysis
python hmc_extend $BASE_DIR/runs/mobius_eofa/hmc --trajectories 150
python hmc_extend $BASE_DIR/runs/mobius_rhmc/hmc --trajectories 150
sbatch $HOME/projects/grid_qcd/jobs/run-eofa.sbatch
sbatch $HOME/projects/grid_qcd/jobs/run-rhmc.sbatch
```

`hmc_extend` always sets `NoMetropolisUntil=0` and `StartingType=CheckpointStart`
for continuation runs, and archives the old `input.xml` as `input_traj{N}.xml`.

---

## Analysis and comparison

```bash
cd $BASE_DIR/grid-lqcd-workflow/4_analysis
module load mambaforge/23.1.0-4
conda activate /lustre2/nplqcd/vayyar/conda-envs/hmc-analysis

# Summary for one run (burn-in, tau_int, observables):
python hmc_obs summary $BASE_DIR/runs/mobius_eofa/hmc

# Side-by-side comparison (means, tau_int, consistency pull):
python hmc_compare $BASE_DIR/runs/mobius_eofa/hmc \
                   $BASE_DIR/runs/mobius_rhmc/hmc \
                   --label1 EOFA --label2 RHMC \
                   --info "1-flavor Mobius DWF beta=5.4 Ls=8 m=0.1"
```

For extended runs combining original and continuation logs, load both and merge:
```python
import pandas as pd, sys
sys.path.insert(0, '.')
from hmc import load_run, save_csv

BASE = '/lustre2/nplqcd/vayyar/grid_qcd'
LOGS = f'{BASE}/run_scripts/logs'

df_eofa = pd.concat([
    load_run(LOGS, pattern='mobius-eofa-1278542.log'),   # original 0-210
    load_run(f'{BASE}/runs/mobius_eofa/hmc'),            # continuations
]).drop_duplicates('traj').sort_values('traj').reset_index(drop=True)
```

---

## What constitutes a good comparison

- **Plaquette pull < 2σ** — primary agreement test; requires ~300+ equilibrated trajectories
- **⟨exp(−dH)⟩ ≈ 1** for both runs — confirms detailed balance (integrator sanity check)
- **Accept rate** — both should be similar (same MD parameters); 99% expected on 4³×8
- **τ_int** — measures relative efficiency, not correctness; EOFA and RHMC may differ

---

## Known issues

- **RHMC `OFRp.hi`** — hardcoded to 100.0 in `src/dwrhmc_mobius.cc`. The power method
  measured λ_max ≈ 87 on a 4³×8 lattice. If running larger lattices, measure λ_max
  from the log (`Approximation of largest eigenvalue`) and increase `hi` accordingly,
  then rebuild.
- **High acceptance rate (~99%)** — expected on this small lattice with MDsteps=8.
  Not a bug; for production, tune MDsteps to get ~75% acceptance.

## Results

See [RESULTS.md](RESULTS.md) for the comparison outcome on a 4³×8 test lattice.
