# src/grid_mg/ provenance

## What this directory is
Updated on 2026-09-30 16:56 CDT · perlmutter

The Grid multigrid rung solver of the pure-Grid HMC (campaign `pure-grid-hmc`, milestone
M2): the gq-mg campaign's three-level fp32 multigrid (`best2` at M2, GMG3 by default since
M3, see below), lifted out of the benchmark probe and packaged as an `OperatorFunction` that
a Hasenbusch ratio action can use for its derivative and action solves. Spec:
`__docs/2026_09_29_pure_grid_m2_mg_solver_design.md`, section 2 (these files) and section 4
item 1 (the test). The driver integration (spec section 3) is a separate step.

M3 (2026-09-30) adds, in the same directory: the MG heatbath (a `GridMGSchurSolver` in a
rung's heatbath slot), a mixed-precision CG rung solver (`mixed_cg_rung_solver.h`) for the
ratio rungs' CG solves, and the GMG3 default. Section "M3 additions" below.

Header-only; nothing here modifies Grid. Compiled against the staged stock tree
`$PSCRATCH/grid_pure_hmc/stock-grid/3d3eff86f366.../Grid` with patches 01-04. Patch 04
(parity-agnostic `CoarsenOperator`) is required: without it an Odd hierarchy aborts in
`CBFromExpression`.

## Sources
Updated on 2026-09-29 19:00 CDT · perlmutter

Probe sources were taken from the COMMITTED revision with
`git -C grid-lqcd-workflow show HEAD:<path>`, never from the working tree (a parallel session
had uncommitted edits there). Workflow repo HEAD at copy time:
`9c6e99683d34f7b850eeffad3b515fdbe005024a`. Fork headers are from the Grid-TXQCD fork at HEAD
`b2c7d9d2`; the fork working tree was clean for both files.

| Source (path) | Last commit touching it | sha256 of the source | Used for |
|---|---|---|---|
| `6_benchmarks/grid_quda_wilson_clover/src/probe_mg_solvers.h` | `abab700` 2026-09-28 | `7f3abae67beb7784e22731913a1dc2637f3da3b38c65fbad8cc04a4259638819` | `mg_solvers.h` |
| `6_benchmarks/grid_quda_wilson_clover/src/probe_grid_mg_schur_clover.cc` | `5bf4a98a` 2026-09-29 | `3ba43bba7e305006f3e1a9cf460f6925001f9baa0ad05f6f7d59839dc81f33d8` | `mg_components.h`, `grid_mg_hierarchy.h`, `grid_mg_params.h` defaults, the test's checks |
| fork `Grid/algorithms/iterative/QudaRungSolverBase.h` | `0ed7c199` 2026-07-14 | `518170e85ef2a59e44d411afd3524a36fe5c33c16bdc9f9627c758b3bb3ec438` | `rung_solver_base.h` |
| fork `Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioActionQuda.h` | `0ed7c199` 2026-07-14 | `16ec743cbe8f556f705e0d7fdf1f3818445f6c2bede1d96f88634910c9d82ccf` | `ratio_action_rung_solver.h` |

`best2` itself is the `grid=(...)` block of `runs/2026_9_29_mg_final_bench_c3/c3_final_bench.sh`.

## Files: verbatim and adapted
Updated on 2026-09-30 16:56 CDT · perlmutter

(As copied at M2. The M3 edits to `grid_mg_params.h`, `grid_mg_hierarchy.h`,
`grid_mg_schur_solver.h` and `ratio_action_rung_solver.h` are listed in "M3 additions".)

- `mg_solvers.h`: the WHOLE `probe_mg_solvers.h`, verbatim, `ProbeMG` namespace kept, so
  `diff` against the probe shows only the edits. The probe file has no `bench_nvtx.h` include
  and no NVTX macro, so nothing was stripped. Unused parts (Chebyshev generator, power method,
  `MrhsCoarseApply`, `check_coarse_apply`) were kept for diffability; they are templates and
  cost nothing unless instantiated. Two edits, both in `create_subspace_gcr` and marked
  `M2 ADAPTATION`:
  1. The four work fields (`noise`, `src`, `guess`, `Mn`) take `Agg.Checkerboard()`. The probe
     constructed them Even, so on an Odd aggregation it drew Even noise and, the Schur
     operators being parity-agnostic, produced null vectors of the EVEN Schur complement. For
     every case the probe ran (Even aggregation, or a full grid at level 2) this assigns the
     default 0: no behaviour change.
  2. A trailing defaulted argument `gcr_verbose = 1` sets the fast GCR's per-step log. Default
     = the probe. The hierarchy passes 0 unless `GRID_MG_VERBOSE >= 3`.
- `mg_components.h`: `ShiftedSchurOperator`, `PrecisionChangeAdaptor`, `MGPreconditioner`,
  verbatim (probe lines 86-391, comments included). Only change: they sit in `namespace ProbeMG`
  instead of the probe's anonymous namespace (an anonymous namespace in a header would give
  every translation unit its own copy). The benchmark counters were already optional:
  `MGPreconditioner`'s stage timers tick only with `instrument` (default false, no barriers).
- `grid_mg_params.h`: new. `struct GridMGParams`, `best2` defaults, `from_env()`.
- `grid_mg_hierarchy.h`: new; the construction sequence is the probe's fp32 path
  (probe lines 1448-1671), minus the fp64 subspace, fp64 coarse operator and fp64 V-cycle.
- `rung_solver_base.h`: renamed copy (`QudaRungSolverBase` -> `RungSolverBase`), comment
  rewritten, code identical.
- `grid_mg_schur_solver.h`: new (the two-solve gamma5 contract of the QUDA hybrid).
- `ratio_action_rung_solver.h`: renamed copy (`TwoFlavourSchurCloverRatioActionQuda` ->
  `TwoFlavourSchurCloverRatioActionRungSolver`, `QudaRungSolverBase` -> `RungSolverBase`);
  `#include <Grid/Grid.h>` added first so the header stands alone; code otherwise identical.
- `test_grid_mg_odd.cc`: new; its Galerkin checks follow the probe's (lines 1211-1233 and
  1570-1585), in fp32 on the hierarchy's own objects.

## Where this differs from the probe or the spec, and why
Updated on 2026-09-29 19:00 CDT · perlmutter

Spec-mandated departures from the probe's `best2` row: the hierarchy coarsens
`SchurDiagMooeeOperator` (the operator the ratio action solves), not the probe's `SCHUR=one`;
the outer restart cap is 1000 (the benchmark ran 200); no fp64 subspace or coarse operator.

Implementation choices where the spec was silent or had to be made concrete:

1. Null-vector parity (above, `mg_solvers.h` edit 1). Without it the "Odd" hierarchy would
   be built from Even vectors and the Galerkin check would fail by construction.
2. Setup RNGs: owned by the hierarchy and RESEEDED at every `Build()` from `params.seed`
   (fine RNG on the full fp32 grid: seed..seed+3; level-1-grid RNG for level-2 noise:
   seed+4..seed+7). A build is then a function of the gauge field and parameters only,
   independent of how many rebuilds preceded it.
3. `GridMGHierarchy`'s operator argument is `GridMGHierarchy::WCF_f` =
   `CompactWilsonCloverFermion<WilsonImplF, CompactCloverHelpers<WilsonImplF>>`, the driver's
   `WCF_f` (the probe held a `WilsonFermionF` base reference; same arithmetic).
4. Level-1 stencil fallback: if the level-1 geometry has `npoint != 9` (an extent-2 coarse
   dimension) the shifted general apply is used, as the probe already did at level 2.
5. Outer solver: fresh `FlexibleGCR` per Mpc solve with the probe's outer settings
   (`zero_guess = false`, one Op on the first cycle); the probe's `verify_residual` block is
   off because the solver computes the true residual itself with the one extra `Op`.
6. Logging: inner solvers (null-vector GCR, smoothers, coarse solves) are silent unless
   `GRID_MG_VERBOSE >= 3`; the probe left the fine smoother and level-1 solve at verbose 1,
   which prints tens of lines per V-cycle. No numerical effect. The outer per-step log needs
   `GRID_MG_VERBOSE >= 2`.
7. Operator counters: always wrapped as in the probe (`CountingLinearOperator`), one
   `OpCounts` per hierarchy; the per-solve log line reports the V-cycle fine/coarse applies.
8. `from_env()` reads every tunable (`GRID_MG_<FIELD>`, full list in the header), not only
   the six the spec names, and aborts on a malformed value or an unknown `GRID_MG_*` name.
9. The hard tier compares with `!(residual <= bound)`, so a NaN residual also triggers it.
   `Build()` before any `SetGauge()` asserts.
10. The level-1 and level-2 general `GeneralCoarsenedMatrix` objects stay alive after their
    stencil copies are made (the probe's structure; `MGPreconditioner` holds a reference it
    never applies). Freeing them is a memory trim, deferred with the others (spec section 6).
11. The test's `--block2 auto` (default): best2's `2.3.3.3` cannot divide the 16^3 level-1
    lattice `4.4.4.12` on any decomposition, so the test lowers each dimension to the largest
    usable value and prints it (`2.2.2.3` on 1 GPU, as the probe's C2 runs; `2.2.1.3` on
    `--mpi 1.1.2.2`). `--mpi 1.1.1.4` (the M1 smoke's layout) admits NO MG geometry at 16^3:
    the level-1 local T extent is 3 and the fp32 SIMD layout is `1.2.2.2`.
12. Pass thresholds of the test (the spec gives expected values, not cuts): level-1 Galerkin
    `< 1e-1` (expected ~5e-2 at hops 1) with the probe's `Mpc^dag Mpc` control `> 1e-3`;
    level-1 stencil vs general apply `< 1e-4`; level-2 Galerkin `< 1e-4` (the probe's gate,
    expected ~1e-6); normal-equation true residual `<= 1e-10`.

## M3 additions
Updated on 2026-09-30 16:56 CDT · perlmutter

All new code; nothing copied. Every addition is env-gated in the driver: with
`HASEN_GRID_MG_HEATBATH_RUNGS`, `HASEN_GRID_MIXED_CG_RUNGS` and
`HASEN_GRID_MIXED_CG_HEATBATH_RUNGS` unset, none of it is constructed.

- GMG3 default (`grid_mg_params.h`). The gq-mg campaign's adopted recipe GMG3 is `best2`
  with a 4-step fine post-smoother (`SMOOTHER_NSTEP=4`), 3.05 s vs 3.19 s per C3 Mpc solve
  at 38 vs 33 outer iterations (ledger L172). `smoother_nstep` default 8 -> 4; every other
  default is still best2's. `Summary()` now starts with `recipe <name>` (`Recipe()`: GMG3,
  best2 = GMG3 with `GRID_MG_SMOOTHER_NSTEP=8`, or custom; `GRID_MG_VERBOSE` ignored); the
  value list moved to `Values()`, text unchanged. The M2 validation runs of 2026-09-29 ran
  best2: reproduce them with `GRID_MG_SMOOTHER_NSTEP=8`.
- `gauge_fingerprint.h` (new): `GaugeTraceFingerprint(U)` = per-Lorentz-component
  `sum_x tr U_mu(x)`, one `TraceIndex` + one global sum, compared bit for bit. Used by the two
  items below.
- `grid_mg_hierarchy.h`: `TrackGauge()`, `Carries(U)`, `SetGaugeIfNew(U)`. The M2 `SetGauge`
  body moved verbatim to the private `ImportGaugeF`; with tracking off (every run without
  the MG heatbath) `SetGauge` is exactly the M2 code. With tracking on, each import records
  the fingerprint of U and of `opF.Umu` (the fp32 doubled links) tagged with the generation
  it produced; `SetGaugeIfNew` imports unless the CURRENT generation carries U and the links
  are still that import's.
- `grid_mg_schur_solver.h`: `ImportIfHierarchyStale()`, the heatbath mode of a non-donor
  solver: its `SetGauge` calls `H.SetGaugeIfNew(U)`. Needed because a sharing rung's
  heatbath can run before the donor's `SetGauge` (driver comment at the M3 block:
  `HASEN_STRANGE_RUNGS` putting donor and sharer on different integrator levels), and at the
  first trajectory the hierarchy would have no gauge (Build() asserts).
- `mixed_cg_rung_solver.h` (new): `MixedPrecCGRungSolver`, Grid's
  `MixedPrecisionConjugateGradient` constructed as the driver's `MixedPrecCGWrapper` (the
  tail's `CG_light_md`), plus a `SetGauge` that imports U into the fp32 operator. The
  skip-redundant-import test is keyed to the operator (a registry per fp32 operator plus a
  fingerprint of its current links), not to the solver: a per-solver test would be wrong
  after a rejected trajectory, when another importer of the same fp32 operator (the next
  rung's deriv solver, the tail's MP deriv, the hierarchy) has changed it.
- `ratio_action_rung_solver.h`: the renamed fork copy is unchanged apart from one comment
  line; a second, new class `TwoFlavourSchurCloverRatioActionHeatbathRung` keeps the plain
  CG deriv / S solvers (separate objects, the base's tolerances) and syncs only a
  `RungSolverBase` heatbath's gauge in `refresh`.

## Build trap found while building
Updated on 2026-09-29 19:00 CDT · perlmutter

The staged stock tree has an INSTALL (`.../stock-grid/<sha>/install/include`, made 15:27)
that predates patch 04 (applied 16:07 to the source tree only), and `grid-config --cxxflags`
contains `-I<install>/include`. Any build that places grid-config's flags before
`-I$GRID_SOURCE` compiles the UNPATCHED `GeneralCoarsenedMatrix.h` (the only header that
differs; `diff -rq` 2026-09-29). `build_driver_stock.sh` does exactly that, which is harmless
for the CG-only driver but not once the driver includes these headers.
`perlmutter/build_test_grid_mg.sh` puts the source tree first and verifies, from the
compiler's own dependency list, that the patched copy was compiled.
