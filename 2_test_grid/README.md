# 4_test — Grid tests and benchmarks

Two scripts to verify correctness and measure performance of an installed Grid library.
Run these after a fresh build to confirm everything is working.

---

## Correctness test

```bash
./4_test/test_grid.sh
```

Runs three executables and checks that numerical results match the reference
implementation (`norm diff ~ 0`). Reports PASS/FAIL for each:

| Test | Executable | What it checks |
|---|---|---|
| SU3 matrix operations | `Benchmark_su3` | Runs to completion without crash |
| Domain Wall Fermion | `Benchmark_dwf` | Dhop result matches reference (norm diff ~ 10⁻³¹) |
| Wilson Fermion | `Benchmark_wilson` | Dhop result matches reference (norm diff ~ 10⁻³¹) |

Expected output:
```
==> Grid correctness tests (profile: txqcd)

  SU3 matrix operations           PASS
  Domain Wall Fermion Dhop        PASS
  Wilson Fermion Dhop             PASS

================================
  3 passed   0 failed
================================
```

---

## Performance benchmark

```bash
./4_test/benchmark_grid.sh
```

Runs four benchmarks and prints a summary of peak performance numbers.
Use this to characterise a new machine or detect regressions after a rebuild.

| Benchmark | Executable | Key metric |
|---|---|---|
| SU3 matrix multiply | `Benchmark_su3` | GFlop/s |
| Memory bandwidth | `Benchmark_memory_bandwidth` | GB/s (AXPY, SCALE, READ) |
| Domain Wall Fermion | `Benchmark_dwf` | Mflop/s (Dhop, DhopEO) |
| Wilson Fermion | `Benchmark_wilson` | Mflop/s (Dhop, DhopEO) |

Reference numbers (Apple M-series, 1 MPI rank, `--grid 4.4.4.4`):

| Metric | Value |
|---|---|
| Memory bandwidth (SCALE) | ~67 GB/s |
| Memory bandwidth (AXPY)  | ~51 GB/s |
| DWF Dhop                 | ~5,800 Mflop/s |
| DWF DhopEO               | ~5,300 Mflop/s |

---

## Notes

- Both scripts use `GRID_PROFILE` from `config.sh` — switch profiles to benchmark mainline vs TXQCD
- Grid geometry is fixed at `4.4.4.4` with 1 MPI rank — suitable for a single-node test
- These scripts run executables directly (no run directories created in `runs/`)
