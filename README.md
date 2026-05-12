# grid-lqcd-workflow

Scripts to build, test, and run [Grid](https://github.com/paboyle/Grid) lattice QCD
on macOS Apple Silicon. Supports two Grid forks via a single `GRID_PROFILE` variable:

| Profile | Repo | Description |
|---|---|---|
| `txqcd` | [mlwagman/Grid-TXQCD](https://github.com/mlwagman/Grid-TXQCD) | TXQCD physics extensions |
| `mainline` | [paboyle/Grid](https://github.com/paboyle/Grid) | Upstream Grid |

---

## Directory structure

```
grid_qcd/                              ← top-level working directory
├── grid-lqcd-workflow/                ← this repo
│   ├── config.sh                      ← set GRID_PROFILE here
│   ├── 1_build_grid/                  ← build the Grid library
│   │   ├── build_grid_homebrew.sh     ← recommended for macOS
│   │   └── build_grid_fromscratch.sh  ← build Grid + all deps from source
│   ├── 2_test_grid/                   ← verify the Grid build
│   │   ├── test_grid.sh               ← correctness tests
│   │   └── benchmark_grid.sh          ← performance benchmarks
│   ├── 3_examples/                    ← self-contained physics examples
│   │   └── dweofa_mobius/             ← Domain Wall EOFA Mobius HMC
│   │       ├── src/                   ← source code
│   │       ├── bin/                   ← compiled binary (git-ignored)
│   │       ├── inputs/                ← XML parameter files
│   │       ├── build.sh               ← compile the binary
│   │       └── run_test.sh            ← smoke test (3 trajectories, ~1 min)
│   └── common/
│       └── run_exec.sh                ← general MPI runner with timestamped output
│
├── Grid-TXQCD/                        ← txqcd source (cloned by build script)
├── Grid-mainline/                     ← mainline source (cloned by build script)
├── build-txqcd/                       ← txqcd build artifacts
├── build-mainline/                    ← mainline build artifacts
├── install-txqcd/                     ← txqcd installed library + executables
├── install-mainline/                  ← mainline installed library + executables
├── deps/                              ← from-scratch dependency builds (shared)
└── runs/                              ← timestamped simulation output
```

Grid source, build, install, and run output all live *outside* the workflow repo —
`git pull` on Grid never touches your files, and vice versa.

---

## Switching profiles

Edit `config.sh` and change one line:

```bash
export GRID_PROFILE="txqcd"     # or "mainline"
```

All paths (`GRID_SRC`, `GRID_BUILD`, `GRID_INSTALL`) update automatically.
Both profiles can be built and installed simultaneously — switching is instant.

---

## Prerequisites

> **Platform note:** these instructions cover macOS Apple Silicon only. Linux support is not yet documented.

- macOS with Apple Silicon
- [Homebrew](https://brew.sh)
- GCC and the OpenMP runtime — install once when setting up a new machine:
  ```bash
  brew install gcc libomp
  ```

All Grid-specific dependencies (`open-mpi`, `gmp`, `mpfr`, `fftw`, etc.) are installed automatically by the build script.

---

## Step 1 — Build Grid

### Recommended: Homebrew (faster, macOS-optimised)

```bash
./1_build_grid/build_grid_homebrew.sh
```

### Alternative: build all dependencies from source

```bash
./1_build_grid/build_grid_fromscratch.sh
```

The from-scratch script builds GMP, MPFR, HDF5, OpenSSL, FFTW, and LIME from source
into `../deps/local/` before building Grid. Use this for HPC systems or when you need
precise control over dependency versions.

**When to re-run:** after `git pull` on the Grid source repo.

---

## Step 2 — Test the Grid build

### Correctness tests

```bash
./2_test_grid/test_grid.sh
```

Runs 7 correctness tests from the Grid build tree. All should pass on macOS.
Takes ~10 seconds.

### Performance benchmarks

```bash
./2_test_grid/benchmark_grid.sh
```

Runs DWF, Wilson, SU(3), and memory bandwidth benchmarks.

---

## Step 3 — Run an example

Each example under `3_examples/` is self-contained: build it, then run the smoke test.

### Domain Wall EOFA Mobius

```bash
# Compile
./3_examples/dweofa_mobius/build.sh

# Smoke test: 4^4 lattice, Ls=4, 3 trajectories (~1 min)
./3_examples/dweofa_mobius/run_test.sh

# Full production run via the general runner
./common/run_exec.sh dweofa_mobius_HSDM_v3 '' 4.4.4.4 1.1.1.1 1
```

The production XML (`inputs/ip_hmc_mobius.xml`) uses Ls=16 and 10 trajectories.
Pass it with `--ParameterFile`:

```bash
mpirun -np 1 3_examples/dweofa_mobius/bin/dweofa_mobius_HSDM_v3 \
    --grid 16.16.16.32 --mpi 1.1.1.1 \
    --ParameterFile 3_examples/dweofa_mobius/inputs/ip_hmc_mobius.xml
```

---

## General runner

`common/run_exec.sh` runs any built executable with MPI and creates an auto-numbered,
timestamped output directory under `../runs/`:

```bash
./common/run_exec.sh <executable> [param_file] [grid] [mpi] [nproc]
```

| Argument | Description | Default |
|---|---|---|
| `executable` | binary name | required |
| `param_file` | XML file from an `inputs/` directory (pass `''` to skip) | — |
| `grid` | lattice geometry | `4.4.4.8` |
| `mpi` | MPI decomposition | `1.1.1.1` |
| `nproc` | number of MPI ranks | `1` |

Each run creates:

```
runs/run_001_20260512_122525/
├── dweofa_mobius_HSDM_v3   ← copy of the executable
└── run.log                  ← all output
```

---

## Configuration reference

All settings are in [`config.sh`](config.sh):

| Variable | Description |
|---|---|
| `GRID_PROFILE` | Active profile: `txqcd` or `mainline` |
| `GRID_REPO` | Git URL of the Grid fork (set automatically by profile) |
| `GRID_SRC` | Path to Grid source directory |
| `GRID_BUILD` | Path to out-of-tree build directory |
| `GRID_INSTALL` | Path to installed library and executables |
| `DEPS_DIR` | Path to from-scratch dependency builds |
| `RUNS_DIR` | Path to simulation output directories |
| `MPICXX` | MPI C++ compiler wrapper |
| `OMPI_CXX` | Underlying C++ compiler used by MPI wrapper (auto-detected) |
