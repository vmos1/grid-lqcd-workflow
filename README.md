# grid-lqcd-workflow

Scripts to build, test, run, and analyse [Grid](https://github.com/paboyle/Grid) lattice QCD.
Supports two Grid forks and two platforms via a shared workflow structure.

| Profile | Repo | Description |
|---|---|---|
| `txqcd` | [mlwagman/Grid-TXQCD](https://github.com/mlwagman/Grid-TXQCD) | TXQCD physics extensions |
| `mainline` | [paboyle/Grid](https://github.com/paboyle/Grid) | Upstream Grid |

| Platform | Status |
|---|---|
| macOS Apple Silicon | CPU build via Homebrew |
| lq (Fermilab, AlmaLinux 8, NVIDIA A100) | GPU build via NVHPC/CUDA |

---

## Directory structure

```
grid_qcd/                              ← top-level working directory
├── grid-lqcd-workflow/                ← this repo
│   ├── config.sh                      ← profile and path configuration
│   ├── 1_build_grid/                  ← build scripts
│   │   ├── build_grid_homebrew.sh     ← macOS: Grid + Homebrew deps
│   │   ├── build_grid_lq.sh           ← lq: GPU build via NVHPC/CUDA
│   │   └── build_grid_fromscratch.sh  ← all deps from source
│   ├── 2_test_grid/                   ← correctness tests and benchmarks
│   │   ├── test_grid.sh               ← runs Test_* binaries
│   │   └── benchmark_grid.sh          ← runs Benchmark_* binaries
│   ├── 3_examples/                    ← self-contained physics examples
│   │   ├── mobius_dwf_test/           ← 1f Möbius DWF: EOFA vs RHMC (TXQCD build)
│   │   ├── mobius_2f_test/            ← 2f Möbius DWF: exact CG vs EOFA (mainline build)
│   │   └── wclover_2p1_test/          ← 2+1f Wilson-clover HMC (TXQCD build)
│   │       ├── src/                   ← wclover_2p1_eo.cc, wclover_2p1_rhmc.cc,
│   │       │                              wclover_hasenbusch_tune_no_eo.cc
│   │       ├── bin/                   ← compiled binaries (git-ignored)
│   │       ├── inputs/                ← XML inputs for each binary
│   │       └── build.sh               ← compile all binaries
│   ├── 4_analysis/                    ← HMC observable analysis toolkit
│   ├── 5_studies/                     ← ongoing physics studies (pre-production tuning)
│   │   └── hasenbusch_tune/           ← Hasenbusch mass tuning for cl21_48_96_b6p3
│   │       ├── src/                   ← gen_qcd_hasenbusch_tune.cc (TXQCD build)
│   │       ├── bin/                   ← compiled binary (git-ignored)
│   │       └── build.sh               ← compile against TXQCD install
│   │   ├── hmc/                       ← Python package
│   │   │   ├── extract.py             ← parse Grid logs → DataFrame
│   │   │   ├── autocorr.py            ← integrated autocorrelation (Gamma/UW)
│   │   │   └── equilibrate.py         ← burn-in detection
│   │   ├── hmc_obs                    ← CLI: extract / autocorr / summary
│   │   ├── requirements.txt
│   │   └── README.md
│   └── common/
│       └── run_exec.sh                ← general MPI runner (macOS)
│
├── Grid-TXQCD/                        ← txqcd source (cloned by build script)
├── Grid-mainline/                     ← mainline source (cloned by build script)
├── build-txqcd-gpu/                   ← txqcd GPU build tree (lq)
├── build-grid-gpu/                    ← mainline GPU build tree (lq)
├── install-txqcd-gpu/                 ← txqcd install: headers, lib, bin (lq)
├── install-grid-gpu/                  ← mainline install: headers, lib, bin (lq)
├── deps/                              ← from-scratch dependency builds
└── runs/                              ← HMC simulation output
    └── <ensemble>/
        ├── hmc/                       ← configs, logs, input.xml
        └── meas/                      ← measurement output
```

---

## Switching profiles

Edit `config.sh`:

```bash
export GRID_PROFILE="txqcd"     # or "mainline"
```

All paths update automatically. Both profiles can coexist.

---

## Step 1 — Build Grid

### macOS (Homebrew)

```bash
./1_build_grid/build_grid_homebrew.sh
```

### lq (GPU, NVHPC/CUDA)

```bash
MACHINE=lq ./1_build_grid/build_grid_lq.sh
```

Loads required modules, configures with `--enable-unified=yes --enable-simd=GPU`,
and installs to `../install-${GRID_PROFILE}-gpu/`.

**When to re-run:** after `git pull` on the Grid source, or after changing
compiler flags. Re-run `build.sh` in any `3_examples/` directory afterwards
to relink against the fresh install.

---

## Step 2 — Test the build

Tests and benchmarks run from `2_test_grid/` on macOS. On lq, submit via the
SLURM scripts in `$HOME/projects/grid_qcd/jobs/` (not in this repo —
cluster-specific paths and account names).

```bash
# macOS
./2_test_grid/test_grid.sh
./2_test_grid/benchmark_grid.sh

# lq
sbatch $HOME/projects/grid_qcd/jobs/test-txqcd.sbatch
sbatch $HOME/projects/grid_qcd/jobs/benchmark-txqcd.sbatch
```

**Known failures on lq GPU builds** (not regressions):
- `Test_general_stencil` — requires nvlink hugepages (not configured on lq)
- `Test_innerproduct_norm` — single-precision GPU rounding vs CPU reference

---

## Step 3 — Run an example: Möbius DWF EOFA vs RHMC

Compiles and runs two 1-flavour Möbius DWF HMC algorithms against the same
input for direct comparison.

```bash
# Compile both binaries against the TXQCD install
source /lustre2/nplqcd/vayyar/grid_qcd/env.sh
./3_examples/mobius_dwf_test/build.sh

# Set up a run directory with input.xml, then submit (lq)
mkdir -p $BASE_DIR/runs/mobius_eofa/hmc
cp 3_examples/mobius_dwf_test/inputs/ip_hmc_test.xml \
   $BASE_DIR/runs/mobius_eofa/hmc/input.xml
sbatch $HOME/projects/grid_qcd/jobs/run-eofa.sbatch
sbatch $HOME/projects/grid_qcd/jobs/run-rhmc.sbatch
```

Each job writes to its run directory:

```
runs/mobius_eofa/hmc/
  input.xml                  active input (edit StartTrajectory to extend)
  input_traj{N}.xml          archived on continuation
  run_info_traj{N}.txt       provenance: date, job ID, node, XML params
  hmc_traj{START}-{END}.log  Grid physics output
  slurm-<jobid>.log          job wrapper output
  ckpoint_lat.{N}            gauge configs (every 10 trajectories)
  ckpoint_rng.{N}            RNG state
```

**Extending a run:** update `StartTrajectory` in `input.xml` and resubmit.
The script auto-archives the old `input.xml` and names the new log correctly.

**RHMC note:** `OFRp.hi` in `dwrhmc_mobius.cc` must exceed the largest
eigenvalue of M†M. Measured λ_max ≈ 87 on a 4³×8 lattice; currently set to
100. Check and adjust for larger lattices.

---

## Step 4 — Analyse

```bash
module load mambaforge/23.1.0-4
conda activate /lustre2/nplqcd/vayyar/conda-envs/hmc-analysis
cd 4_analysis

# Full summary: accept rate, auto burn-in, tau_int for all observables
python hmc_obs summary $BASE_DIR/runs/mobius_eofa/hmc

# Save observables to CSV, then compute autocorrelation time
python hmc_obs extract $BASE_DIR/runs/mobius_eofa/hmc -o obs_eofa.csv
python hmc_obs autocorr obs_eofa.csv plaquette --burnin 50
```

See [`4_analysis/README.md`](4_analysis/README.md) for the Python module API
and details of the Gamma/UW autocorrelation method.

---

## Step 5 — Physics studies

`5_studies/` contains pre-production tuning workflows — short jobs run to
determine optimal parameters before committing to a production ensemble.

### Hasenbusch mass tuning (`5_studies/hasenbusch_tune/`)

Tunes the 4 intermediate masses for Hasenbusch preconditioning of the
`cl21_48_96_b6p3_m0p2416_m0p2050` 2+1f Wilson-clover ensemble.

Requires the TXQCD build (`install-txqcd-gpu`) for EO clover actions.
See `hasenbusch.md` in `$HOME/projects/grid_qcd/` for the full tuning guide.

```bash
source /lustre2/nplqcd/vayyar/grid_qcd/env.sh
cd 5_studies/hasenbusch_tune
bash build.sh

# Run with Chroma baseline masses (from cfg_2000 metadata):
export MASS_LIGHT=-0.2416 MASS_STRANGE=-0.2050 CSW=1.20537 BETA=6.3 LATT=48.48.48.96 U0=1.0
export HASEN_LADDER="-0.2416,-0.2400,-0.2320,-0.2180,-0.1870"
export IMPORT_CFG=/lustre2/nplqcd/cfgs/.../cfg_2000.lime N_TRAJ=5
# submit via $HOME/projects/grid_qcd/jobs/tune-hasenbusch.sbatch
```

---

## Configuration reference (`config.sh`)

| Variable | Description |
|---|---|
| `GRID_PROFILE` | Active profile: `txqcd` or `mainline` |
| `GRID_SRC` | Grid source directory |
| `GRID_BUILD` | Out-of-tree build directory |
| `GRID_INSTALL` | Installed library and executables |
| `DEPS_DIR` | From-scratch dependency builds (shared) |
| `RUNS_DIR` | Simulation output root |
| `MPICXX` | MPI C++ compiler (macOS) |

Set `MACHINE=lq` before sourcing to activate lq-specific overrides (modules,
CUDA paths, GPU architecture `sm_80`, `-gpu` path suffixes).
