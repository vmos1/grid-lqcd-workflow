# grid-lqcd-workflow

Scripts to build, compile, and run [Grid](https://github.com/paboyle/Grid) lattice QCD
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
│   ├── 1_build_grid/
│   │   ├── build_grid_homebrew.sh     ← build Grid using Homebrew deps (recommended for macOS)
│   │   └── build_grid_fromscratch.sh  ← build Grid + all deps from source
│   ├── 2_compile/
│   │   ├── build_exec.sh              ← compile production/ programs or custom .cc files
│   │   ├── Makefile
│   │   └── src/                       ← place custom .cc files here
│   └── 3_run/
│       ├── run_exec.sh                ← run executables with MPI
│       └── inputs/                    ← XML parameter files
│
├── Grid-TXQCD/                        ← txqcd source (cloned by build script)
├── Grid-mainline/                     ← mainline source (cloned by build script)
├── build-txqcd/                       ← txqcd build artifacts
├── build-mainline/                    ← mainline build artifacts
├── install-txqcd/                     ← txqcd installed library + executables
├── install-mainline/                  ← mainline installed library + executables
├── deps/                              ← from-scratch dependency builds (shared)
└── runs/                              ← simulation output directories
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

- macOS with Apple Silicon
- [Homebrew](https://brew.sh)

All other dependencies are installed by the build script.

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

## Step 2 — Compile executables

### Build all production/ programs (txqcd profile only)

```bash
./2_compile/build_exec.sh
```

### Compile a custom .cc file

```bash
# Place your file in 2_compile/src/, then:
./2_compile/build_exec.sh myprogram.cc
# → produces 2_compile/bin/myprogram
```

Changing your `.cc` file only requires re-running step 2 — Grid does not need to be rebuilt.

---

## Step 3 — Run

```bash
./3_run/run_exec.sh <executable> [param_file] [grid] [mpi] [nproc]
```

| Argument | Description | Default |
|---|---|---|
| `executable` | binary name | required |
| `param_file` | file from `inputs/` (optional, pass `''` to skip) | — |
| `grid` | lattice geometry | `4.4.4.8` |
| `mpi` | MPI decomposition | `1.1.1.1` |
| `nproc` | number of MPI ranks | `1` |

Each run creates an auto-numbered, timestamped directory under `../runs/`:

```
runs/run_001_20260424_150852/
├── TXQCD_Wilson_small    ← copy of the executable
├── ip_hmc_mobius.xml     ← copy of the parameter file (if used)
└── run.log               ← all output
```

### Examples

```bash
# Run a TXQCD HMC on a 4^4 lattice, 1 MPI rank
./3_run/run_exec.sh TXQCD_Wilson_small '' 4.4.4.4 1.1.1.1 1

# Run with a parameter file
./3_run/run_exec.sh TXQCD_Wilson_small ip_hmc_mobius.xml 4.4.4.4 1.1.1.1 1

# Run a production config generator on 8^3×16, 4 ranks
./3_run/run_exec.sh gen_txqcd_cfgs '' 8.8.8.16 2.2.1.1 4
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
| `OMPI_CXX` | Underlying C++ compiler used by MPI wrapper |
