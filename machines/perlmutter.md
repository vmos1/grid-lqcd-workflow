# grid_qcd on Perlmutter: account, queues, paths, traps

Updated on 2026-10-06 14:17 CDT · perlmutter

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
During a Slurm controller stall `release_alloc.sh` can report "already gone"
while the job still runs (and is still charged). Re-check with `squeue -u $USER`
and retry the release.

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
  Exception: a single-TU build against prebuilt Grid and QUDA libs (the CG
  benchmark, `Benchmark_comms_host_device`) takes ~1 min on one login core and
  is fine there. Grid's own device TUs are the slow ones (~1.3 h per TU) and
  `Benchmark_dwf_fp32` is ~40 min: keep those on a compute node.

## Python environment

- The system `python3` is 3.6 (no `math.prod`, no 3.8+ syntax). Use the repo's
  own uv environment: `pyproject.toml`, `.python-version` (3.12) and `uv.lock`
  at the root of `grid-lqcd-workflow`. Run scripts from that root as
  `uv run python <script>`; add a package with `uv add <name>` and commit the
  changed `pyproject.toml` and `uv.lock`.
- `uv` itself is in `~/.local/bin` (installed 2026-10-06 with
  `python3.13 -m pip install --user uv` from the NERSC Python module's
  interpreter). `perlmutter.sh` puts it on `PATH`.
- All uv state is on `$PSCRATCH/uv/` (`cache`, `python`,
  `venvs/grid-lqcd-workflow`), set by `perlmutter.sh`: Python plus packages
  are ~8k files, which the CFS inode quota cannot take. The repo's `.venv` is
  a local symlink to the same environment (gitignored), so `uv run` without
  `perlmutter.sh` still uses scratch instead of creating ~8k files on CFS.
  Never delete that symlink while the repo is on CFS.
- After a scratch purge (~8 weeks idle), `uv sync` in the repo root rebuilds
  everything from `uv.lock` in about a minute.

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

## Gauge configurations on HPSS

Updated on 2026-10-20 11:00 CDT · perlmutter

- The NPLQCD clover ensembles (cl21 = 2+1 flavour, cl3 = 3 flavour, 16³ to
  96³) are on NERSC HPSS tape under `/home/w/wdetmold/clover-iso/`, one
  directory per stream. HPSS is not mounted: a job cannot read a config there.
  Pull the few you need onto `$PSCRATCH`; never mirror an ensemble (cl21 48³
  alone is 33 TB, 6.1 GB per config).
- **Access needs NERSC project m747.** `clover-iso` is `drwxr-s--- wdetmold
  m747`; on 2026-10-20 this account (groups `m4599`, `amsc011`) got
  `HPSS_EACCES`, and m747 membership was being requested. Check with
  `hsi -q ls -l <path>`, which lists without recalling from tape.
- **Do not list configs in docs; query the catalog.** Every file has a record
  (tape path, size, stream, trajectory, full action) in the AmSC MetaCat
  catalog at FNAL. Reads are anonymous, no token, only the query text leaves
  NERSC:
  ```
  bash grid-lqcd-workflow/common/find_cfgs.sh                                      # all ensembles, one line each
  bash grid-lqcd-workflow/common/find_cfgs.sh cl21_48_96_b6p3_m0p2416_m0p2050       # its streams + HPSS dirs
  bash grid-lqcd-workflow/common/find_cfgs.sh cl21_48_96_b6p3_m0p2416_m0p2050 djm-3 2000 2050   # files
  ```
  `KIND=hmc_input` lists the Chroma restart XMLs instead. Per-ensemble physics
  and stream layout: `usqcd_catalog/ensembles/<ensemble>.toml` (upstream clone
  at the workspace root, ~5 KB each). Its README covers MQL and the REST URLs.
- Trajectory numbering restarts in every stream, so a config is named by
  (ensemble, stream, trajectory). Our 48³×96 `data/...-djm-3_cfg_2000.lime` is
  stream djm-3, trajectory 2000 of `cl21_48_96_b6p3_m0p2416_m0p2050`; its
  size matches the catalog record byte for byte (6115385552).
- Some cl3 ensembles (`cl3_16_48`, `cl3_32_48`) store configs inside htar
  archives, so the catalog shows archive records and the per-config count can
  read 0. Extract members with `usqcd_catalog/bin/htar_extract.sh`.
- To pull (once m747 is granted):
  `hsi -q "get $PSCRATCH/cfgs/<file> : <hpss path>"`. For more than a handful,
  run on the `xfer` QOS and order the list by tape position
  (`usqcd_catalog/scripts/hpss_select.py`, `docs/data_movement.md`); random
  tape recalls are slow and NERSC asks for batching
  (https://docs.nersc.gov/filesystems/HPSS-best-practices/).

## Where the rest is

Campaign-grade operating history: `perlmutter-ops` row in `__docs/_TOC.md`,
its `_SUM.md` entry, `2026_08_20_perlmutter_pe_bump_cudart13_postmortem.md`,
and section 7 of `2026_09_23_memory_promoted_facts.md`.
