# perf_profile: timing tools for the pure-Grid HMC driver

Written during grid-upstream-rebase (2026-11-22 to 11-25), kept for grid-perf-profile.
They read our driver's log lines or Nsight Systems exports. Not part of the Grid fork.

## From an HMC log

| tool | reads | prints |
|---|---|---|
| `pupdate_overhead.awk` | `[i][j] P update elapsed time: X ms (force: Y ms)` and `[l] batched smeared P update ... (pullback+Ta: Y ms)` | per force piece: calls, update s, force s, the rest (update minus force) in s and in ms per call |
| `mg_solve_sum.awk` | `[GridMG PFn] solve ... iters N ... seconds S ... vcycle_applies fine F coarse C` | per rung: solves, iterations, seconds, fine and coarse applications, ms per fine application, coarse applications per iteration |

    awk -f grid-lqcd-workflow/common/perf_profile/mg_solve_sum.awk runs/<run>/hmc.log

Feed the first k lines (`grep 'P update elapsed' hmc.log | head -n k | awk -f ...`) to
compare two runs at equal work.

## From Nsight Systems exports (`nsys export --type sqlite`)

| tool | does |
|---|---|
| `mg_vcycle_compare.py OLD.sqlite NEW.sqlite` | multigrid solves found from a marker kernel (default `ProbeMG::blockProjectFused`); per V-cycle: wall, GPU busy and idle, kernels by name, copies, host CUDA calls |
| `mg_vcycle_host.py OLD.sqlite NEW.sqlite` | main-thread call-stack samples in idle GPU time inside those V-cycles: by module, leaf function, innermost program function and callee |
| `mg_printf_callers.py REPORT.sqlite` | which frames call printf inside the V-cycles |

    uv run --project grid-lqcd-workflow python grid-lqcd-workflow/common/perf_profile/mg_vcycle_compare.py OLD.sqlite NEW.sqlite

They import `common/nsys_force_split.py` and `common/nsys_host_idle.py`. The host tools need a
report recorded with CPU sampling and call stacks (`common/nsys_rank.sh`).
