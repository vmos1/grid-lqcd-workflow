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
| Perlmutter (NERSC, A100) | GPU build via Cray PE / CUDA 12.9 |

---

## Directory structure

```
grid_qcd/                              ← top-level working directory
├── grid-lqcd-workflow/                ← this repo
│   ├── config.sh                      ← profile and path configuration
│   ├── machines/                      ← machine-specific environment (sourced by config.sh)
│   │   ├── perlmutter.sh             ← NERSC Perlmutter (Cray PE, CUDA 12.9)
│   │   ├── lq.sh                     ← lq cluster (NVHPC, CUDA 12.2)
│   │   └── macos.sh                  ← macOS (Homebrew)
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
│   │       ├── src/                   ← gen_qcd_hasenbusch_tune.cc
│   │       ├── bin/                   ← compiled binary (git-ignored)
│   │       ├── lq/
│   │       │   └── build.sh           ← compile on lq
│   │       └── perlmutter/
│   │           ├── build_puregrid.sh  ← compile pure-Grid (no QUDA)
│   │           └── build_quda.sh      ← compile with QUDA (Phase 2)
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

## Environment

Sourcing `config.sh` loads the full environment for the detected machine — modules,
compiler, and CUDA/MPI/HDF5 paths (from `machines/<machine>.sh`). The machine is
auto-detected (`$NERSC_HOST` / `/lustre2/nplqcd` / `uname`), or force it with `MACHINE=<name>`:

```bash
source config.sh
```

The cluster build scripts (`1_build_grid/build_grid_*.sh`) and the SLURM submit scripts
already do this for you. You only need it explicitly when compiling an example by hand
(`3_examples/*/build.sh`) — source `config.sh` first.

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

Tests and benchmarks run from `2_test_grid/` on macOS. On clusters, submit via that
machine's SLURM scripts (not in this repo — cluster-specific paths and account names):
on lq they live in `$BASE_DIR/submit_scripts/`, on Perlmutter under the relevant
`*/perlmutter/` example directory.

```bash
# macOS
./2_test_grid/test_grid.sh
./2_test_grid/benchmark_grid.sh

# lq
sbatch $BASE_DIR/submit_scripts/test-txqcd.sbatch
sbatch $BASE_DIR/submit_scripts/benchmark-txqcd.sbatch
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
./3_examples/mobius_dwf_test/build.sh

# Set up a run directory with input.xml, then submit (lq)
mkdir -p $BASE_DIR/runs/mobius_eofa/hmc
cp 3_examples/mobius_dwf_test/inputs/ip_hmc_test.xml \
   $BASE_DIR/runs/mobius_eofa/hmc/input.xml
sbatch $BASE_DIR/submit_scripts/run-eofa.sbatch    # lq; see that machine's run notes
sbatch $BASE_DIR/submit_scripts/run-rhmc.sbatch
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
See [`5_studies/hasenbusch_tune/README.md`](5_studies/hasenbusch_tune/README.md) for the
full tuning guide (physics, env-var interface, tuning loop).

```bash
# lq
cd 5_studies/hasenbusch_tune && bash lq/build.sh

# Perlmutter (pure-Grid, no QUDA)
cd 5_studies/hasenbusch_tune && bash perlmutter/build_puregrid.sh
```

Submit via that machine's sbatch scripts (on lq: `$BASE_DIR/submit_scripts/`; on
Perlmutter: `5_studies/hasenbusch_tune/perlmutter/`).

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

Machine-specific environment is loaded from `machines/<machine>.sh`. The machine
is auto-detected (`$NERSC_HOST` for Perlmutter, `/lustre2/nplqcd` for lq, `uname`
for macOS), or forced explicitly:

```bash
MACHINE=lq source config.sh
```

Adding a new machine: create `machines/<name>.sh` and add a detection rule in `config.sh`.
