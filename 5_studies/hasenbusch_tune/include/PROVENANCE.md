# include/ provenance

## What this directory is
Updated on 2026-09-29 15:06 CDT · perlmutter

Project-authored additive headers that the pure-Grid HMC driver
`src/gen_qcd_hasenbusch_tune_compact_schur.cc` needs and that do not exist in
stock Grid. They are carried here, beside the driver, so the driver can be
compiled against a stock (patched) Grid tree without an include path into the
Grid-TXQCD fork. See `perlmutter/build_driver_stock.sh`.

These are NOT Grid patches. Nothing here modifies or overrides a file that
exists in stock Grid: at stock commit `3d3eff86f366` none of the six
`Grid/qcd/action/pseudofermion/` headers below exists, so placing
`-I<this dir>` ahead of Grid's include flags only adds files. The `Grid/...`
sub-path is kept so the driver's `#include <Grid/qcd/action/pseudofermion/...>`
lines resolve unchanged. `params.h` is found by the driver's quoted
`#include "params.h"` through the same `-I` flag.

Upstream of record: the Grid-TXQCD fork copy remains the authoritative version
of every file here until the fork is retired. Edit there first and re-copy, or,
once the fork is retired, declare this copy the upstream and say so here.

## Files
Updated on 2026-09-29 15:06 CDT · perlmutter

Source paths are relative to the fork root
`/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/Grid-TXQCD` (fork HEAD at copy
time: `b2c7d9d2`, 2026-07-22; the fork working tree was clean for all seven
files). "Fork commit" is the last commit touching the file
(`git log -1 --format='%h %ad' --date=short -- <path>`). Copies are
byte-identical (`cp -p`); the sha256 was checked on both source and copy.

| Copied file (under `include/`) | Source path in fork | Fork commit | sha256 | Copied |
|---|---|---|---|---|
| `Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h` | `Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h` | `351e3d55` 2026-04-26 | `7daa91883b2899bf8958a462dc5d232bce3beca71e043fcff7220ac0407ef192` | 2026-09-29 |
| `Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioAction.h` | `Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioAction.h` | `0ed7c199` 2026-07-14 | `2f8a0459bf3229fea89955d7b06804f32c40855583489d50487c0760fcd91bc7` | 2026-09-29 |
| `Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h` | `Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h` | `351e3d55` 2026-04-26 | `70f770802ab4506182a8285cc5ad62606ac0878a99c307d1a80cb95050b53988` | 2026-09-29 |
| `Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h` | `Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h` | `351e3d55` 2026-04-26 | `26b8e695471a2e0708cd5486ade424c771da747bf34d040083824c8be9d4acbd` | 2026-09-29 |
| `Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h` | `Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h` | `448c1df8` 2026-05-02 | `2224a7647527b1fa9561f5916701783eb1903c34a6e4af05f8bc028bbb37656a` | 2026-09-29 |
| `Grid/qcd/action/pseudofermion/QCDLogDetCompactCloverEOAction.h` | `Grid/qcd/action/pseudofermion/QCDLogDetCompactCloverEOAction.h` | `0ed7c199` 2026-07-14 | `ebcd9069fc25e0758d2b1f1a0f276982ff292ed99ee7b54be59d7f2184921fd7` | 2026-09-29 |
| `params.h` | `production/params.h` | `240607d7` 2026-06-01 | `9d0a6d811b194e2fdcffa73e7245c87c01bc3f60dcc25877563787f7f46861a3` | 2026-09-29 |

Re-verify a copy against the fork with `sha256sum` on both paths; a mismatch
means one side was edited and this table is stale.

## Local edits: M5 fused clover force (2026-10-01)
Updated on 2026-10-01 12:07 CDT · perlmutter

Three copies were edited here and now differ from the fork on purpose; the
fork was not touched. For these three files this copy is the upstream from
now on, and the sha256 values in the table above are the copy-time ones.

| File (under `include/`) | Edit | sha256 after the edit |
|---|---|---|
| `Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioAction.h` | fused branch in `deriv` | `ae71973efdaadc916bd1c62f8abcf466c619e7129efd80fd589d14af85bfa401` |
| `Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h` | fused branch in `deriv` | `9e31b1c263735807c2089603fab04b85ad09118111a44da8fffd8427ddd41829` |
| `Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h` | fused branch in `deriv` | `c8c263711ce8c4e32aa520e680801eb79ca06d31e600de0f81361b50ed24a7b5` |

Each edit only adds code: an `#include "clover_force/fused_clover_force.h"`
and, in `deriv`, an `if (FusedCloverForceEnabled())` branch that accumulates
the MooDeriv/MeeDeriv outer products and runs one clover pass
(`src/clover_force/fused_clover_force.h`, switch
`HASEN_GRID_FUSED_CLOVER_FORCE=1`, parsed by the driver). With the switch
off, which is the default, the original statements run unchanged and in the
same order. The include resolves through `-I$HB/src`, which
`build_driver_stock.sh`, `build_test_grid_mg.sh` and
`build_test_fused_clover.sh` all pass, so these three headers now need it.
They still include no path that exists only in the fork.

## Local edits: item D clover derivative (2026-10-02)
Updated on 2026-10-02 08:59 CDT · perlmutter

`Grid/qcd/action/pseudofermion/QCDLogDetCompactCloverEOAction.h` (sha256 before
`3248272c68f23a516023689c3a974ecbf719fab5c135317bf8fa6386d0ec242a`, after
`f281d8f0aaa68cda63b85f4a12d2c388487c19030b07af9339df870efd6664df`): adds
`#include "clover_force/clover_cmunu.h"`; in `deriv_cpu` and `deriv_gpu` the
`Cmunu` loop gains an `if (CloverStapleCacheEnabled())` branch
(`HASEN_GRID_CLOVER_STAPLE_CACHE=1`, cached link products, bit-identical) and
`deriv_gpu` an `if (CloverStencilEnabled())` branch (`HASEN_GRID_CLOVER_STENCIL=1`,
PaddedCell + GeneralLocalStencil, bit-identical); the two `setCheckerboard` calls become
`CloverSetCheckerboard`, which is `acceleratorSetCheckerboard` under
`HASEN_GRID_DEVICE_CB=1` (patch 06's switch, read the same way) and the original
call otherwise. All switches default off; off, the original statements run.

## Dependencies on Grid
Updated on 2026-09-29 15:06 CDT · perlmutter

No file here includes a fork-only path (nothing under `Grid/qcd/action/txqcd/`,
`Grid/util/Quda*`, or `Grid/algorithms/iterative/Quda*`). Every Grid header
they include exists in stock Grid at `3d3eff86f366`:
`Grid/Grid.h`, `Grid/qcd/action/fermion/WilsonCloverFermion.h`,
`Grid/qcd/action/fermion/CompactWilsonCloverFermion.h`,
`Grid/qcd/action/fermion/CloverHelpers.h`,
`Grid/algorithms/blas/BatchedBlas.h`,
`Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h`.
`QCDLogDetCompactCloverEOAction.h` also includes `<cublas_v2.h>` under
`#ifdef GRID_CUDA` (CUDA toolkit, not the fork).

The fork's `CloverHelpers.h` and `BatchedBlas.h` differ from stock (an opt-in
GPU clover-inverse path, and cuBLAS host pointer mode). The headers here use
only `GridBLAS::Init()` and `GridBLAS::gridblasHandle`, which stock provides
(`InstantiateGPU` appears in comments only), so the difference is expected to
affect performance only. A compile against stock has not been run yet.

## Local edits 2026-10-05 (not in the fork)
Updated on 2026-10-05 13:07 CDT · perlmutter

The three strange-action headers now diverge from the fork copy (pure-Grid only; the fork
headers are not edited, and the driver refuses the new knobs when built against them):
`OneFlavourSchurRationalExtras` (separate MD degree and MD tolerance for the x^−1/2 force
approximation, a run-time bounds check modelled on stock `Bounds.h`), selected by the macro
`HASEN_STRANGE_RATIONAL_EXTRAS` and the driver variables `RAT_DEGREE_MD`,
`TUNE_CG_TOL_STRANGE_MD`, `RAT_BOUNDS_CHECK_FREQ`, `RAT_BOUNDS_CHECK_TOL`,
`RAT_BOUNDS_CHECK_ABORT` (all default to the previous behaviour; binary `_stock_m8` is
bit-identical to `_stock_m7b` with them unset). Reason: ledger L191 (the strange interval missed
the smeared spectrum unnoticed). Validation: `runs/2026_10_5_ratmd2_*` (16³).
sha256 after the edit:
- `OneFlavourSchurCloverRationalAction.h` 01ea58287e542f5f4c97eae8f3815ff37e42705f11ddc91ff8042dc50ee013a0
- `OneFlavourSchurCloverRationalActionMP.h` ac489dd72a6924fbf42b82d7cbcd2266d5a1c8c9051c54c099be994d2089291a
- `OneFlavourSchurCloverRationalActionEven.h` 860b1e4404685c5f76d3ee785213efce61ab6de2d7e9c1c62cfa23e0e6cebd95
