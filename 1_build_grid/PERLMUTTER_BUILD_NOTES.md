# Grid-TXQCD on Perlmutter — build notes & open issues

Working record of building `Grid-TXQCD` (mlwagman's fork) on NERSC Perlmutter for the
Hasenbusch mass-tuning study (`5_studies/hasenbusch_tune/`) and, later, auxiliary-field
(TXQCD) work. Companion to `PERLMUTTER_PORTING.md` in the work-dir root.

Date started: 2026-06-02/03. Built against `Grid-TXQCD` commit **`3b9ee2a7`** (branch `develop`).

---

## TL;DR status

| Piece | State |
|---|---|
| Core Grid library (`libGrid.a`, 186 MB) | ✅ **builds clean** |
| `grid-config` | ✅ works |
| Hasenbusch tune binary (pure-Grid, no QUDA) | 🔵 building / validating |
| QUDA strange force (`QUDA_FORCE=1`) | ⬜ not built yet (Phase 2) |
| **TXQCD auxiliary-field code** (`gen_txqcd_cfgs`, TXQCD HMC, `meas_*_txqcd`) | ❌ **does NOT compile on this toolchain — see §4, fix after Hasenbusch** |

**Bottom line:** the library and all *pure-QCD* binaries build on the current Perlmutter
toolchain. Only the *TXQCD σ-field* binaries are blocked, by compiler-strictness errors
(§4). Those must be resolved before any aux-field test, but are deferrable.

---

## 1. Environment

- **Machine:** NERSC Perlmutter, A100 GPU stack, project **`m4599`** (GPU charge code `m4599_g`).
- **Build location:** **CFS, in place** — `/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/Grid-TXQCD`
  (not `$PSCRATCH`: CFS is permanent/backed-up; PSCRATCH purges ~8 wk. This workload is a
  login-node compile + tiny HMC jobs with no HDF5, so PSCRATCH's speed edge doesn't matter.)
- **Toolchain (current default login modules — supersedes the fork's stale pins):**
  `PrgEnv-gnu/8.6.0`, `gcc-native/14` (g++ 14.3.0), `cray-mpich/9.0.1`,
  `cudatoolkit/12.9`, `craype-accel-nvidia80`, `gpu/1.0`.
- **GMP / MPFR:** Grid needs both (RHMC Remez). GMP is complete in `/usr` (auto-detected,
  no flag). MPFR has no dev header in `/usr`; use the complete Cray PE copy at
  **`/opt/cray/pe/gcc/mpfr/3.1.4`** via `--with-mpfr` (and on `LD_LIBRARY_PATH` at runtime).

### Machine environment (`machines/perlmutter.sh`)
The Perlmutter environment now lives in `grid-lqcd-workflow/machines/perlmutter.sh`
(sourced automatically by `config.sh` when `$NERSC_HOST=perlmutter`, or directly by
build and sbatch scripts). It replaces the ad-hoc untracked `Grid-TXQCD/sourceme.sh`
created during initial porting.

```bash
export CRAY_ACCEL_TARGET=nvidia80
module load PrgEnv-gnu
module load cudatoolkit/12.9
module load craype-accel-nvidia80
export GRID_MPFR_PREFIX=/opt/cray/pe/gcc/mpfr/3.1.4
export LD_LIBRARY_PATH="${GRID_MPFR_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CLIME_ROOT=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/lime-install
export CUDA_ARCH=sm_80
```
This matches upstream `paboyle/Grid/systems/Perlmutter` intent (which uses `cudatoolkit/12.0`
+ spack GMP/MPFR); we use `12.9` + system GMP + Cray MPFR.

---

## 2. Reproducible build recipe (library)

```bash
cd /global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/Grid-TXQCD
source /global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/grid-lqcd-workflow/machines/perlmutter.sh
./bootstrap.sh                      # fetches Eigen 3.4.0, generates ./configure
mkdir -p build && cd build
../configure \
    --enable-comms=mpi --enable-simd=GPU --enable-shm=nvlink \
    --enable-gen-simd-width=64 --enable-accelerator=cuda --enable-setdevice \
    --disable-fermion-reps --disable-unified --disable-gparity \
    --with-mpfr=$GRID_MPFR_PREFIX \
    CXX=nvcc LDFLAGS="-cudart shared" \
    CXXFLAGS="-ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 -cudart shared"
make -j 4                           # NOTE: -j 4, not -j 32 — see §3 (memory cap)
```
**Verify:** `build/grid-config --cxx` prints `nvcc -std=c++17 -x cu`; `build/Grid/libGrid.a`
exists (~186 MB). Configure summary should show `SIMD: GPU`, `Acceleration: cuda`,
`Communications: mpi3`, `GMP: yes`.

> Build only the library if that's all you need: `make -j 4 -C Grid`. The full `make all`
> also builds `benchmarks/`, `tests/`, `examples/`, `HMC/` — none of which the out-of-tree
> production binaries require (and `HMC/` currently fails, §4).

---

## 3. Issues encountered & RESOLVED

### 3.1 Stale fork config (`systems/Perlmutter/`)
The fork's `config-command`/`sourceme.sh` are behind upstream:
`cudatoolkit/11.4` (stale → `12.9`), `-std=c++14` (Grid mandates C++17 →
`AX_CXX_COMPILE_STDCXX(17)`; the trailing `-std=c++14` made the C++17 probe *fail* → use
`-std=c++17`), and `--with-gmp=$DIR/../Prequisites/install` pointing at a non-existent
`Prequisites/` tree (dropped; GMP auto-detected from `/usr`, MPFR via `--with-mpfr`).
Also dropped `--disable-accelerator-cshift` (unrecognized at this commit). The corrected
flags (§2) match upstream `paboyle/Grid/systems/Perlmutter`.

### 3.2 `-j 32` OOM-kills the compilers
Login nodes impose a **per-user cgroup memory cap of 30 GiB**
(`user.slice/.../memory.max = 32212254720`), *shared with everything you run on the node*
(including the Claude Code agent). `-j 32` runs 32 memory-hungry `cudafe++` (nvcc front-end)
processes → exceeds the cap → kernel SIGKILLs them:
```
nvcc error : 'cudafe++' died due to signal 9 (Kill signal)
```
**Fix:** use **`-j 4`** on a login node (peak ~16–24 GiB, comfortably under 30). The heavy
library TUs only used ~2 GiB each in practice. For a faster build, use a compute node
(`salloc`, 256 GiB) at high `-j`.

### 3.3 `__rdtsc`/`__rdpmc` overload error (`Grid/perfmon/PerfCount.h`)
Under `GRID_CUDA`, `Grid/perfmon/PerfCount.h:54-55` defined stubs returning `uint64_t`
(`unsigned long`), but `<x86intrin.h>` (pulled in elsewhere) declares the real ones
returning `unsigned long long` → *"cannot overload functions distinguished by return type
alone"* (fails compiling `Stat.cc`/`PerfCount.cc`).
**Fix applied:** comment out those two stub lines — exactly **upstream Grid's form**
(upstream has them commented; the fork un-commented them). `Stat.cc` then uses the real
`x86intrin.h` versions.
```c
#ifdef GRID_CUDA
//accelerator_inline uint64_t __rdtsc(void) {  return 0; }
//accelerator_inline uint64_t __rdpmc(int ) {  return 0; }
#else
#include <x86intrin.h>
#endif
```
> **Hand back to lq/collaborator:** revert-to-upstream of these two lines (§10 of porting doc).

---

## 4. OUTSTANDING — TXQCD aux-field code does NOT compile (fix after Hasenbusch)

`make all` fails in `build/HMC/` compiling `TXQCD_Wilson_small.o`. Root error is in the
auxiliary-field pseudofermion header:

```
Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h:165:
  error: need 'typename' before 'std::remove_cv<...iScalar<...GpuComplex<double2>...>>::type'
         because '...' is a dependent scope
  error: 'SSitePerLane' was not declared in this scope
  error: there are no arguments to 'g_acc' that depend on a template parameter, so a
         declaration of 'g_acc' must be available [-fpermissive]
  error: 'g_acc' was not declared in this scope; did you mean 'acc'?
```

**Impact:** blocks every TXQCD-theory binary — `gen_txqcd_cfgs(_2plus1/_nf2)`, the TXQCD HMC
drivers, `meas_conn_txqcd`, `meas_disco_txqcd`, `meas_aux_txqcd`, `eigspec_diag`. Pure-QCD
binaries (Hasenbusch tune, `gen_qcd_cfgs*`, `meas_*_qcd`, `compute_plaq`) are unaffected —
they link the clean `libGrid.a`.

**Likely cause:** compiler-strictness drift. This code built for the collaborator on the
older `cudatoolkit/11.4` + older gcc; gcc-14 / nvcc-12.9 enforce two-phase name lookup more
strictly. The **`[-fpermissive]`** tag on the `g_acc` error is strong evidence it compiled
under looser settings before — i.e. this is a *port*, not a logic bug.

**Suggested fix order (cheapest first; do after Hasenbusch tuning is done):**
1. **Try `gcc-native/12`** (keep `cudatoolkit/12.x`). gcc-14 tightened template diagnostics;
   gcc-12 may compile the TXQCD headers as-is. If so, rebuild `libGrid` with gcc-12 for a
   uniformly clean, aux-field-capable build with ~no source edits. *(Quick to test.)*
2. **`-fpermissive`** on the TXQCD translation units (blunt; downgrades the flagged errors).
3. **Targeted source fixes** — add `typename` / `this->` / qualify `g_acc`,`SSitePerLane` at
   `TXQCDWilsonPseudoFermionAction.h:165` and any cascading sites; same *class* as the
   `__rdtsc` port (§3.3). Contribute upstream to the fork.
4. If deeper than the above, **escalate to the collaborator** (it's their TXQCD physics code).

> A ~15–20 min scope-check (try #1 and #2, enumerate distinct errors) will bound this before
> committing real effort.

---

## 5. Hasenbusch tune binary (pure-Grid, no QUDA)

Built out-of-tree against `grid-config`, **without** `-DGRID_HAVE_QUDA` (the source's
`#ifdef GRID_HAVE_QUDA` guards then drop the QUDA strange-force path; `QUDA_FORCE` falls back
to Grid's mixed-precision rational CG):

```bash
GC=$GRID_TXQCD/build/grid-config ; PROD=$GRID_TXQCD/production
HB=.../grid-lqcd-workflow/5_studies/hasenbusch_tune
$($GC --cxx) $($GC --cxxflags) -I$PROD -o $HB/bin/gen_qcd_hasenbusch_tune \
    $HB/src/gen_qcd_hasenbusch_tune.cc $($GC --ldflags) $($GC --libs)
```
(log: `5_studies/hasenbusch_tune/build_pure_grid.log`.) Smoke test: `perlmutter/smoke.sbatch`
(8³×16 cold start, 1 GPU) — runs pure-Grid with `QUDA_FORCE` unset.

---

## 6. QUDA (Phase 2 — not built yet)

Needed only for `QUDA_FORCE=1` (strange RHMC force via
`OneFlavourSchurCloverQudaForceRationalActionMP`). Options:
- **Reuse** a prebuilt Perlmutter QUDA that matches this commit's wrapper headers (ask the
  collaborators for a path; must be sm_80, clover+multishift+clover-force, CUDA 12.x).
- **Build our own**, porting `vmos1/Staggered_multigrid_build/.../build_quda.sh` (Frontier/AMD):
  `QUDA_TARGET_TYPE=HIP→CUDA`, `gfx90a→sm_80`, drop ROCm/`hipcc`; **`QUDA_DIRAC_STAGGERED` →
  `QUDA_DIRAC_WILSON + QUDA_DIRAC_CLOVER`** (we need Wilson-clover); `QUDA_MULTIGRID=OFF`
  (we use no MG); keep `QUDA_DOWNLOAD_USQCD=ON` (provides `libqmp`). Start from QUDA `develop`
  and confirm the wrapper headers compile/link at the binary step.

---

## Key paths
- Grid-TXQCD build: `/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/Grid-TXQCD/build`
- `grid-config`: `…/Grid-TXQCD/build/grid-config` · `libGrid.a`: `…/build/Grid/libGrid.a`
- `params.h`: `…/Grid-TXQCD/production/params.h`
- tune binary src/bin: `…/grid-lqcd-workflow/5_studies/hasenbusch_tune/{src,bin}`
- MPFR runtime lib (keep on `LD_LIBRARY_PATH`): `/opt/cray/pe/gcc/mpfr/3.1.4/lib`

---

## Progress update — end of session 1 (2026-06-03)

Supersedes the "building/TODO" states above:

- **LIME: now ENABLED.** Built USQCD `c-lime` → `…/grid_qcd/lime-install` (`limeCreateReader` present),
  reconfigured Grid `--with-lime=…/lime-install` → `LIME: yes`, rebuilt `libGrid.a`. `grid-config --libs`
  now carries `-llime`. (Needed for `IldgReader` in the binary, and for the full-volume `.lime` validation.)
- **Pure-Grid tune binary: BUILT** (`5_studies/hasenbusch_tune/bin/gen_qcd_hasenbusch_tune`, 30 MB, no QUDA).
  `grid-config` alone is **insufficient** for an un-installed Grid — must add in-tree flags:
  `CXXFLAGS += -I$PROD -I$GRID_TXQCD -I$BUILD/Grid`, `LDFLAGS += -L$BUILD/Grid` (mirrors `production/Makefile`).
  The repo's `build.sh` / `perlmutter/build.sh` lack these — **fix before the QUDA build**.
- **Smoke (8³×16 cold, 1 GPU): pipeline VALIDATED** — runs to completion and prints the `Hasenbusch chain`
  + 5 `FORCES traj=` lines **only with `STOUT_NSMEAR=0`**.
- **NEW blocker — cold start + stout smearing = NaN.** On an exactly-cold (unit) config F=0, and the stout
  matrix-exponential hits 0/0 → NaN smeared links → every fermion solve is `-nan` (seen in the first
  Hasenbusch-ratio heatbath; the strange Remez is fine). **Fix:** start from a HOT (random) config so F≠0,
  smearing ON. Add a `HOT_START` env knob to `gen_qcd_hasenbusch_tune.cc` (~L162,
  `SU<Nc>::ColdConfiguration`→`HotConfiguration`), recompile. **Applies to the proxy tuning too** (also cold-start).
- Smoke job: `perlmutter/smoke_puregrid.sbatch` (no-QUDA variant; runnable via `bash` or `sbatch`).
- **Pure-Grid build script written: `perlmutter/build_driver_puregrid.sh`** — encapsulates the working
  in-tree compile (recompile = `bash build_driver_puregrid.sh`). The QUDA `perlmutter/build.sh` is still
  unfixed: it mandates `QUDA_PREFIX` and lacks the in-tree `-I/-L` flags — fix both before Phase 2.

