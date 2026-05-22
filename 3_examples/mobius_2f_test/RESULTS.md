# Results — 2-flavor Möbius DWF: exact CG vs EOFA

## Run parameters

| Parameter | Value |
|---|---|
| Lattice | 4³×8 |
| β | 5.4 |
| Ls | 8 |
| mass | 0.1 |
| M5 | 1.8 |
| b, c | 1.5, 0.5 |
| MD integrator | MinimumNorm2, MDsteps=8, trajL=1.0 |
| Boundary conditions | {1,1,1,−1} (periodic xyz, antiperiodic t) |
| Trajectories (total) | 2f-CG: 660,  2f-EOFA: 350 |
| Cluster | lq (Fermilab), NVIDIA A100 GPU |
| Date | May 2026 |

## Observable comparison

Post burn-in statistics: 2f-CG N_eq ≈ 596 (burn-in traj 65), 2f-EOFA N_eq ≈ 308 (burn-in traj 43).

| Observable | 2f-CG mean ± σ | 2f-EOFA mean ± σ | Pull |
|---|---|---|---|
| Plaquette | 0.4790 ± 0.0012 | 0.4764 ± 0.0020 | +1.12σ ✓ |
| Polyakov \|P\| | 0.03708 ± 0.00074 | 0.0399 ± 0.0012 | −1.99σ ✓ |
| Polyakov Re(P) | 0.0002 ± 0.0013 | 0.0010 ± 0.0019 | −0.34σ ✓ |
| ⟨dH⟩ | 0.0092 ± 0.0046 | 0.0006 ± 0.0011 | +1.80σ ✓ |
| ⟨exp(−dH)⟩ | 0.9970 ± 0.0046 | 0.9995 ± 0.0010 | −0.54σ ✓ |

All observables consistent within 2σ. **Both algorithms sample the same distribution.**

## Autocorrelation times

| Observable | 2f-CG τ_int | 2f-EOFA τ_int |
|---|---|---|
| Plaquette | 4.41 ± 0.99 | 5.16 ± 1.72 |
| Polyakov \|P\| | 0.48 ± 0.05 | 0.57 ± 0.08 |
| ⟨dH⟩ | 0.52 ± 0.05 | 0.48 ± 0.07 |

τ_int values are consistent between the two algorithms within errors — both explore
the gauge space at similar rates for the 2-flavor action on this lattice.

## Notes

- 2f-EOFA runs ~2× slower per trajectory than 2f-CG (~18.5 s vs ~8.6 s on A100)
  due to two independent CG solves per trajectory (one per flavor)
- Thermalization is faster for 2-flavor than 1-flavor (~traj 30-65 vs ~traj 100),
  consistent with stronger gauge field driving from two fermion flavors
- Accept rate ~99% on this small lattice; expected to decrease on production lattices
- The exact CG (2f-CG) has no rational approximation — agreement with 2f-EOFA
  directly validates the EOFA pseudofermion construction

## Verdict

✓ **2-flavor exact CG and 2-flavor EOFA implementations validated** — both sample
[det(M_phys)/det(M_PV)]² correctly on a 4³×8 test lattice. The EOFA implementation
is confirmed against an independent, approximation-free reference.
