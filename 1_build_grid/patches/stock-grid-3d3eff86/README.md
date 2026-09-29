# Patches for stock Grid 3d3eff86 (pure-Grid HMC)

Applied in sorted order by `1_build_grid/build_grid_stock_hmc.sh patch` onto a clean clone of
`paboyle/Grid` at commit `3d3eff86f3664f4068af9814ae92150406bb812d` (develop, 2026-06-11).
Each file is `git diff` output rooted at the Grid repo top level (`git apply` from the repo root).
Extracted from Grid-TXQCD with `git diff 34b44d1f HEAD -- <files>`, where `34b44d1f` is the fork's
upstream base. Every hunk dry-applies cleanly on `3d3eff86` and on develop `87baaf1` (2026-09-29).

| Patch | Files | Mandatory | What |
|---|---|---|---|
| `01-clover-eo-force-MooDeriv-MeeDeriv.patch` | `Grid/qcd/action/fermion/implementation/WilsonCloverFermionImplementation.h`, `.../CompactWilsonCloverFermionImplementation.h` | yes | Bodies for `MooDeriv`/`MeeDeriv` (the even-odd clover force). Stock ships `GRID_ASSERT(0)`; the EO/Schur HMC aborts at the first force call without them. Fork commits `cecb7ad6` (non-compact) and `0ed7c199` (compact). Upstream-PR candidate. |
| `02-ildg-reader-lfn-and-writer-boss-guard.patch` | `Grid/parallelIO/IldgIO.h`, `Grid/parallelIO/BinaryIO.h` | yes | Reader: remove `GRID_ASSERT(found_ildgLFN)`; the ILDG logical-filename record is optional and Chroma configs omit it (fork `f664128a`). Writer: boss-node guard for multi-rank checkpoint writing (fork `b2c7d9d2`). |
| `03-hmc-dH-determinism-memprint-gate-schur-parity.patch` | `Grid/qcd/action/gauge/GaugeImplTypes.h`, `Grid/qcd/hmc/integrators/Integrator.h`, `Grid/qcd/action/pseudofermion/EvenOddSchurDifferentiable.h` | default on | Kinetic term via `norm2` (deterministic reduction, same value); `MemoryManager::Print` in the force loop behind `INTEGRATOR_VERBOSE_MEM` (it ends in `cudaMemGetInfo`, a device sync, twice per force evaluation); Schur force accepts even-parity inputs (needed only by the `STRANGE_EVEN` path). |

Not carried, on purpose: the fork's batched-BLAS host-pointer-mode change (upstream develop
adopted it later; at `3d3eff86` the handle is in DEVICE mode), the non-compact clover GPU inverse,
the `LAMBDA_MN2` env override, the `SymanzikGaugeAction(beta,u0)` constructor, `PerfCount.h`
(identical to stock), and all TXQCD and QUDA files.

Plan and rationale: `__docs/2026_09_29_pure_grid_stock_build_plan.md`.
