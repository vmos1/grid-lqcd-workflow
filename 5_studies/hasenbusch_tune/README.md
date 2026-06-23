# Hasenbusch mass tuning — `cl21_48_96_b6p3_m0p2416_m0p2050`

Tunes the intermediate Hasenbusch masses for 2+1f Wilson-clover HMC, with the goal of
demonstrating a wall-time speedup over the Chroma baseline that generated this ensemble.
The baseline chain is extracted directly from the `cfg_2000` metadata.

> **Machine-specific details** (how to build/submit on each cluster) live outside this
> file: build via `lq/build.sh` or `perlmutter/build_driver_puregrid.sh` (below); operational
> notes are in the lq runbook (`$BASE_DIR/CLAUDE.md`) and `1_build_grid/PERLMUTTER_BUILD_NOTES.md`.
> This README is machine-indifferent: physics, the binary's env-var interface, and the tuning loop.

> **Status (2026-06-09/10):** the action was corrected to drop the hand-inserted
> Pauli-Villars ratio and a double-counted light log-det — see
> `../../../hasenbusch_action_correction.md`. An automated mass-tuning loop
> (`5_studies/hasenbusch_tune/autotune/`) now drives the search — see
> `../../../hasenbusch_autotune.md`. The env-var interface and Chroma-baseline table
> below reflect the corrected (no-PV) action.

---

## Ensemble

| Parameter | Value |
|-----------|-------|
| Lattice | 48³ × 96 |
| β | 6.3 |
| m₀ light | −0.2416 (κ = 0.13302) |
| m₀ strange | −0.2050 |
| csw | 1.20537 |
| Gauge action | Lüscher-Weisz (`LW_TREE_GAUGEACT`, u0 = 1.0 tree-level) |
| Stout smearing | ρ = 0.125, 1 level |
| Reference config | `cfg_2000.lime` (thermalized, Oct 2019, plaq ≈ 0.54351) |

Validation: the equilibrium plaquette from a tuned run must converge to the cfg_2000
reference value (≈ 0.54351).

---

## Chroma baseline (extracted from `cfg_2000.lime` metadata)

The original run used a 5-level Hasenbusch chain with masses **clustered near m_l**:

| Level | m_i (lighter) | m_{i+1} (heavier) | Δm |
|-------|---------------|-------------------|----|
| PF0 | −0.2416 (m_l) | −0.2400 | 0.0016 |
| PF1 | −0.2400 | −0.2320 | 0.0080 |
| PF2 | −0.2320 | −0.2180 | 0.0140 |
| PF3 | −0.2180 | −0.1870 | 0.0310 |
| Tail | det M(−0.1870)² (bare det, no ratio — caps the chain) | — | — |

Steps widen away from the critical mass — the spectrum is densest near m_l, so fine
spacing is needed there. These masses are the **tuning starting point and baseline**.

---

## The binary: `gen_qcd_hasenbusch_tune` (+ `_compact` variant)

Standalone, env-var driven (no XML, no HDF5, no checkpointing) — designed for short tuning
runs. Source: `src/gen_qcd_hasenbusch_tune.cc`. Prints one grep-able `FORCES traj=N …` line
per trajectory for per-level force-balance analysis.

Two binaries are built from the same physics, differing only in clover storage:

| Binary | Light/strange operator | Clover storage | Host footprint |
|--------|------------------------|----------------|----------------|
| `gen_qcd_hasenbusch_tune` | `WilsonCloverFermion` | full fields | ~6× larger |
| `gen_qcd_hasenbusch_tune_compact` | `CompactWilsonCloverFermion` | packed Diagonal[6]+Triangle[15] | ~6× smaller |

The forces are **identical** between the two (validated: A/B max relative difference
~1e-10, the worst case being the strange RHMC mixed-precision CG rounding). The compact
binary exists because this Grid build keeps a full-volume *host* shadow of every lattice
field, so the non-compact operator's host shadow is what drives memory; the compact layout
cuts it enough that the full 48³×96 tune fits on far fewer nodes. Use whichever the machine
notes prescribe — the tuning signal (forces) is the same. The compact path uses
`QCDLogDetCompactCloverEOAction` for the det(M_ee) log-det terms (it reconstructs the even
clover block on demand via `ConvertLayout`); see `wclover_eo_compact_a_vs_b_comparison.md`
for the full A-vs-B writeup and `wclover_eo_compact_theory.tex` for the physics.

### Env-var interface

| Var | Meaning |
|-----|---------|
| `HASEN_LADDER` | Comma-separated masses light→heavy, e.g. `-0.2416,-0.2400,-0.2320,-0.2180,-0.1870`. Lightest **must** equal `MASS_LIGHT`; heaviest is the bare-det **Tail** mass (no Pauli-Villars appended — the ratio chain + Tail exactly reproduce det(M_l)²). Unset → single-level baseline (for timing comparison). |
| `MASS_LIGHT` / `MASS_STRANGE` | Quark masses (defaults −0.2416 / −0.2050 for this ensemble). |
| `CSW`, `BETA`, `U0` | Clover coeff, gauge coupling, mean-field u0 (use `U0=1.0` for LW tree-level). |
| `STOUT_NSMEAR` | Stout smearing levels (default 1). **Set `0` for cold-start smoke tests** — see "Cold-start" below. |
| `N_TRAJ`, `MDSTEPS` | Trajectories and MD steps per trajectory. |
| `GAUGE_INNER_MULT` | Gauge sub-steps per fermion step (default 4). |
| `IMPORT_CFG` | Starting gauge config (NERSC or Chroma/ILDG LIME). Unset → cold start. |
| `QUDA_FORCE` | `1` to use QUDA for the strange RHMC force (requires a QUDA-enabled build — Phase 2). |
| `FORCES_ONLY` | `1` → skip the integrator entirely: heatbath each action and print one `deriv()` force norm per level, then exit. This is the **fast tuning mode** — no completed trajectory needed (see "Fast tuning" below). |
| `FORCES_SAMPLES` | (FORCES_ONLY) number of heatbath/force draws to average over (default 1). |
| `FORCES_SKIP_STRANGE` | (FORCES_ONLY) `1` omits the expensive strange RHMC force eval. |
| `TUNE_CG_TOL_ACTION` | CG tolerance for the action/heatbath solves (default `1e-8`, production). Loosen (e.g. `1e-4`) for faster tuning iterations. |
| `TUNE_CG_TOL_DERIV` | CG tolerance for the force (`deriv`) solves (default `1e-6`). Loosen for faster tuning. |

```bash
# Chroma baseline chain:
export HASEN_LADDER="-0.2416,-0.2400,-0.2320,-0.2180,-0.1870"
# Single-level baseline (disable Hasenbusch):
unset HASEN_LADDER
```

### Cold-start caveat (NaN)

On an exactly-cold (unit) config the force is 0, and the stout matrix-exponential hits
0/0 → NaN smeared links → every solve returns `-nan`. For a cold smoke test set
`STOUT_NSMEAR=0`. With an imported, thermalized config (the real tuning runs) the force
is nonzero and smearing is fine — no action needed.

---

## Implementation notes

### Action classes for the ratio levels

For each ratio det(M_i)/det(M_{i+1}):
- Use `TwoFlavourRatioPseudoFermionAction` (full-lattice, proven correct), with
  mixed-precision CG for the force solve, and `is_smeared = true` (matches Chroma's
  `STOUT_FERM_STATE`).
- **Do *not* use `TwoFlavourEvenOddRatioPseudoFermionAction`** — it hard-asserts
  `NumOp.ConstEE() == 1 && DenOp.ConstEE() == 1`, but `WilsonCloverFermion::ConstEE()`
  always returns 0 (M_ee depends on the gauge field), so it aborts at the first force
  evaluation.

The ratio masses are close (Δm ≈ 0.001–0.03), so CG converges fast regardless of EO
preconditioning. Full-lattice ratio is fine; an EO Schur ratio action is a Phase 2 item.

### Building

```bash
# lq:
bash lq/build.sh
# Perlmutter (pure-Grid, no QUDA):
bash perlmutter/build_driver_puregrid.sh
```

Both compile out-of-tree against the corresponding Grid-TXQCD build's `grid-config`
(plus the in-tree `-I/-L` flags those scripts add). Requires `install-txqcd-gpu` (EO
clover, mixed-precision CG). QUDA strange force is Phase 2 — see `perlmutter/build_driver_quda.sh`.

The build scripts produce both `gen_qcd_hasenbusch_tune` and `gen_qcd_hasenbusch_tune_compact`
(override `SRC`/`BIN` to build just one). The compact binary additionally requires the
Grid-TXQCD `CompactWilsonCloverFermion::MooDeriv/MeeDeriv` force terms and
`QCDLogDetCompactCloverEOAction.h` to be present in that Grid-TXQCD build.

---

## Tuning loop

1. **Run with the Chroma baseline masses** (short: ~5 trajectories) from a thermalized
   config (`IMPORT_CFG=…/cfg_2000.lime`).
2. **Measure force balance** from the `FORCES traj=…` lines (per-level `deriv_norm_average`).
   Balance metric = max(PF force) / min(PF force); target < 1.3.
3. **Measure wall time per trajectory** vs the single-level run (`HASEN_LADDER` unset).
   A balanced chain should be ~2–5× faster on a near-physical ensemble.
4. **Iterate masses if imbalanced** — if the bottom level (PF0) dominates, its gap is too
   wide: move m_H1 closer to m_l. If the top (PF4) dominates, raise m_H4.
5. **Validate plaquette** converges to ≈ 0.54351.

### Fast tuning with `FORCES_ONLY`

For mass balancing you only need the per-level force *magnitudes*, not completed
trajectories — and on this ensemble a full trajectory is hours-long. `FORCES_ONLY=1`
heatbaths each pseudofermion and evaluates a single `deriv()` per level on the imported
config, printing:

```
FORCES_ONLY samples=N avg: PF0=.. PF1=.. PF2=.. PF3=.. Tail=.. Strange=.. Gauge=..
FORCES_ONLY samples=N max: ...
```

(There is deliberately no `LogDet` field for the light sector — the ratio chain + Tail
already represent the full det(M_l)²; see the action-correction doc linked above. The
strange sector's EO log-det is part of the action set but not part of this force list.)

Read the `avg:` line → adjust `HASEN_LADDER` → repeat (each iteration is minutes, not hours).
Combine with `FORCES_SKIP_STRANGE=1` and loose `TUNE_CG_TOL_ACTION/DERIV` (e.g. `1e-4`) to
cut the dominant light-quark solve time. The forces are identical between the compact and
non-compact binaries, so tune on whichever fits the machine.

The pseudofermion levels (`PF*`, `Tail`, `Strange`) are stochastic (depend on the heatbath
draw), so average over a few `FORCES_SAMPLES` if you need a tighter estimate. `Gauge` is
deterministic but currently **hangs in `FORCES_ONLY`** (known issue, integrator-safe) —
set `FORCES_SKIP_GAUGE=1` to skip it; its value (~6.88) is known from the integrator path.

---

## Phase 2 (deferred)

- **EO Schur ratio action** — replace the full-lattice ratio with an EO Schur ratio
  (~2× smaller CG volume); requires a new `TwoFlavourSchurCloverRatioAction`. Begin only
  after the tuned masses are confirmed on the full volume.
- **QUDA strange force** (`QUDA_FORCE=1`) — needs a QUDA-enabled Grid-TXQCD build.
- **Port `HASEN_LADDER` into the production binary** `gen_qcd_cfgs_2plus1.cc`, following
  the `gen_txqcd_cfgs_2plus1.cc` pattern (QcdDiag HDF5 force diagnostics, QUDA force, LW
  gauge action), for the actual production ensemble generation.
