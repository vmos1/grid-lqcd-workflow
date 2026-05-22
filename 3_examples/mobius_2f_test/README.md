# mobius_2f_test — 2-flavor Möbius DWF: exact CG vs EOFA

Independent validation of the EOFA implementation by comparing against exact 2-flavor
CG — a completely different code path that uses standard conjugate gradient rather than
a rational approximation.

Both codes represent **[det(M_phys) / det(M_PV)]²** — the same 2-flavor fermion
determinant via different algorithms. They must produce identical expectation values
if both implementations are correct.

| Binary | Algorithm | Pseudofermion action |
|---|---|---|
| `dw2f_cg_mobius` | Exact 2-flavor CG | `TwoFlavourEvenOddRatioPseudoFermionAction` |
| `dw2f_eofa_mobius` | 2×1-flavor EOFA | Two `ExactOneFlavourRatioPseudoFermionAction` |

**Why this is an independent test of EOFA:** the exact CG uses a completely different
code path — standard CG on M†M, no rational approximation. Any bug in the EOFA
pseudofermion construction would show up as a discrepancy against the exact CG result.

**Boundary conditions:** `{1, 1, 1, -1}` — periodic in x,y,z, antiperiodic in t.

---

## Grid API note

`TwoFlavourEvenOddRatioPseudoFermionAction(NumOp, DenOp, CG, CG)` uses counterintuitive
naming: the weight is **det(DenOp) / det(NumOp)**, so:
- `NumOp` = PV operator (mass=1.0) — appears in the *denominator* of the ratio
- `DenOp` = physical operator (mass=0.1) — appears in the *numerator* of the ratio

This matches the EOFA convention where each 1-flavor action contributes det(M_phys)/det(M_PV).

---

## Build

Built against `install-grid-gpu` (mainline Grid) — both classes exist in mainline Grid.

```bash
source /lustre2/nplqcd/vayyar/grid_qcd/env.sh
cd 3_examples/mobius_2f_test
bash build.sh
```

---

## Run (lq cluster)

**Important:** 2f-EOFA runs approximately twice as slowly as 2f-CG (~18.5 s/traj vs
~8.6 s/traj) because it performs two CG solves per trajectory (one per flavor).
Use separate trajectory counts per job to fit within the 30-min time limit.

```bash
# 2f-CG: 200 trajectories ≈ 29 min (fits in 30-min job)
# 2f-EOFA: 90 trajectories ≈ 28 min (fits in 30-min job; NoMetropolisUntil adds to total)

mkdir -p $BASE_DIR/runs/mobius_2f_cg/hmc   $BASE_DIR/runs/mobius_2f_cg/meas
mkdir -p $BASE_DIR/runs/mobius_2f_eofa/hmc $BASE_DIR/runs/mobius_2f_eofa/meas

cp inputs/ip_hmc_2f_test.xml $BASE_DIR/runs/mobius_2f_cg/hmc/input.xml
cp inputs/ip_hmc_2f_test.xml $BASE_DIR/runs/mobius_2f_eofa/hmc/input.xml
# Edit mobius_2f_eofa input.xml: set <Trajectories>90</Trajectories>

sbatch $HOME/projects/grid_qcd/jobs/run-2f-cg.sbatch
sbatch $HOME/projects/grid_qcd/jobs/run-2f-eofa.sbatch
```

---

## Extending a run

```bash
cd $BASE_DIR/grid-lqcd-workflow/4_analysis
# 2f-CG: 150 trajectories per extension (~21 min)
python hmc_extend $BASE_DIR/runs/mobius_2f_cg/hmc   --trajectories 150
# 2f-EOFA: 90 trajectories per extension (~28 min)
python hmc_extend $BASE_DIR/runs/mobius_2f_eofa/hmc --trajectories 90
sbatch $HOME/projects/grid_qcd/jobs/run-2f-cg.sbatch
sbatch $HOME/projects/grid_qcd/jobs/run-2f-eofa.sbatch
```

---

## Analysis and comparison

```bash
cd $BASE_DIR/grid-lqcd-workflow/4_analysis
module load mambaforge/23.1.0-4
conda activate /lustre2/nplqcd/vayyar/conda-envs/hmc-analysis

python hmc_compare $BASE_DIR/runs/mobius_2f_cg/hmc \
                   $BASE_DIR/runs/mobius_2f_eofa/hmc \
                   --label1 "2f-CG" --label2 "2f-EOFA" \
                   --info "2-flavor Mobius DWF beta=5.4 Ls=8 m=0.1"
```

For a good comparison both runs need similar numbers of equilibrated trajectories.
The 2f-EOFA run needs to catch up to the 2f-CG run due to slower per-trajectory cost.

---

## What constitutes a good comparison

- **Plaquette and Polyakov pull < 2σ** — primary agreement test
- **⟨exp(−dH)⟩ ≈ 1** for both — integrator sanity; exact CG has no rational approx
  error so any deviation signals an MD step-size problem
- ~300 equilibrated trajectories for 2f-CG, ~300 for 2f-EOFA (requires ~4 extensions
  of 2f-EOFA at 90 traj/job)

---

## Thermalization

2-flavor dynamics thermalize faster than 1-flavor (~traj 30-40 vs ~traj 100)
because two fermion flavors drive stronger gauge field evolution. τ_int for the
plaquette is correspondingly lower (~1.8 vs ~3.7 for 1-flavor on the same lattice).
