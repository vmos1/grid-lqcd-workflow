# grid_qcd on Perlmutter: account, queues, paths, traps

Updated on 2026-09-24 11:22 CDT · perlmutter

Read by agents on every session run on Perlmutter, per rule 7 of the workspace
`AGENTS.md`. Shell environment is `perlmutter.sh` beside this file. General
NERSC rules live in the user's global instruction file; this file holds only
what this project needs on this machine. Machine-independent facts belong in
`__docs/`, not here.

## Account and queues

- GPU account **`m4599_g`**; CPU account `m4599`. Every job or allocation
  states one of them; there is no useful default.
- Default to the **interactive QOS** for anything under 4 nodes / 4 hours.
  Its priority (73320) outranks every `gpu_regular` job, NERSC auto-cancels an
  undeliverable request, and charging is on elapsed time, so ask for the full
  4 h and release early.
- Before requesting: `sinfo -s -p gpu_ss11`. If idle is single-digit or all
  idle nodes show `resv`/`plnd`/`drain`, nothing is grantable and the
  interactive request will time out; use `sbatch`. When the batch queue is
  jammed (hundreds pending ahead), interactive still wins; the idle count alone
  is the wrong signal.
- `sbatch` scripts always carry an explicit `--time`; a missing one has
  defaulted to 10 minutes. State the estimate before submitting. The global
  rule to ask before any `sbatch` or allocation still applies.

## Allocations go through wrappers only

```
bash ~/.claude/bin/alloc_gpu.sh <nodes 1-4> <hours 1-4>   # prints job id, returns at once
bash ~/.claude/bin/release_alloc.sh <jobid>               # always release; elapsed time is charged
```

The allocation is `--no-shell`; workloads attach with `SLURM_JOB_ID=<id> bash
<script>`; verify with `SLURM_JOB_ID=<id> srun -N 1 -n 1 hostname`. Never
raw `salloc` or `scancel`: `salloc ... bash -c '<anything>'` is arbitrary
execution the permission matcher cannot see, and `scancel`'s filter flags
cancel unbounded sets. The wrappers are write-denied to the agent, and only
the user adds their allow rule; an agent refusing to widen its own permissions
is correct. `salloc` can print `QOSMaxSubmitJobPerUserLimit` and still have
created the allocation: `squeue -u $USER` after any allocation error.

## Node binding

Binding is about task-to-NUMA-domain mapping, not the allocator. Fewer than 4
tasks per node mis-binds rank 0 onto domain 3 under both `salloc
--gpu-bind=none` and `sbatch --ntasks-per-node=1 --gpus-per-node=1`. A 1-GPU
probe owns the whole node (`--gpus-per-node=4 --cpus-per-task=128`). Gate,
do not print: `srun ... numactl -m 0 -N 0 true || exit 1`.

## Filesystems

- Workspace, installs, scripts, inputs, run output, docs: CFS at
  `/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd` (this root).
- **Source trees and build dirs go on `$PSCRATCH`.** The m4599 CFS quota was
  at 99% of its inode limit on 2026-08-17 at 56% of space; `df`/`du` do not
  show it (`cfsquota -h /global/cfs/cdirs/m4599`). A QUDA tree is ~38k files.
  Tar the source onto CFS afterwards as one inode; `$PSCRATCH` purges after
  ~8 weeks idle.
- Run output stays under `runs/` here, never inside either git tree. Run
  directory names are unpadded `runs/YYYY_M_D_<name>/`.

## Build and runtime environment

- `source grid-lqcd-workflow/machines/perlmutter.sh` before building or
  running: it pins `cray-mpich/9.0.1`, `cudatoolkit/12.9`, and sets
  `LD_LIBRARY_PATH` from `CRAY_LD_LIBRARY_PATH`. Without that last line the
  system symlink farm wins and every rank dies at ~10 s with
  `libcudart.so.13: cannot open shared object file`, exit 127. After any PE
  change: `ldd <binary> | grep -E 'gtl|cudart|not found'`, want zero
  "not found". A ~10 s death that names a shell variable and exits 1 is a
  missing `--export`, not the loader; read the last line first.
- Login nodes are shared: `configure` there, `make -j` inside an allocation.

## Tools on the login node

- `claude` is a shell function that routes accounts; `command claude` reaches
  the binary (needed for `--version` and headless probes). `codex` is installed.
- No `gh`; PR creation is through the GitHub web UI.
- Claude Code's permission classifier blocks executing a file named `*.sbatch`
  even inside an existing allocation; expect to ask, or name drivers `.sh`.
- The user's terminal has no bracketed paste: any command handed over to run
  by hand must fit on one unwrapped line. Use `git -C <deepest dir>` so
  arguments are bare filenames; put multi-line content in a file and pass it
  with `-F`/`--body-file`.

## Where the rest is

Campaign-grade operating history: `perlmutter-ops` row in `__docs/_TOC.md`,
its `_SUM.md` entry, `2026_08_20_perlmutter_pe_bump_cudart13_postmortem.md`,
and section 7 of `2026_09_23_memory_promoted_facts.md`.
