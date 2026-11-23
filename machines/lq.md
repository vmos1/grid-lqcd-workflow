# grid_qcd on lq (Fermilab): account, queues, paths, traps

Updated on 2026-11-23 10:07 CST · fnal_lq

Read by agents on every session run on lq, per rule 7 of the workspace
`AGENTS.md`. Shell environment is `lq.sh` beside this file. General lq rules
live in the user's global instruction file; this file holds only what this
project needs on this machine. Campaign facts and numbers belong in `__docs/`
(campaign `lq-port`, handoff `2026_11_18_handoff_lq_port.md` §7c–7e), not here.

## Targets and accounts

Updated on 2026-11-20 09:22 CST · fnal_lq

| `LQ_TARGET` | runs on | account | OS | built on | release |
|---|---|---|---|---|---|
| `gpu` (default) | `lq2_gpu`: 18 nodes × 4 A100-SXM4-80GB (sm_80) | `nplqcd.lq2_gpu` | EL8 | `lq1_cpu` (`nplqcd.lq1_cpu`, `--constraint=intel6248`) | `installs/<stage>/m10_gpu` |
| `cpu` | `lq3_cpu`: 23 nodes × 2 × 32-core EPYC 9355 (Zen 5), 1.5 TB, NDR IB `mlx5_0` | `nplqcd.lq3_cpu` | EL9 | `lq3_cpu`, only there | `installs/<stage>/m10_cpu` |

- `LQ_TARGET` selects the target in `lq.sh` and in every lq build and launch script.
- The EL9 toolchain runs neither on the EL8 login node nor on lq1/lq2: everything for the
  CPU target, configure included, happens inside a job on lq3.
- lq3 nodes are always exclusive (`OverSubscribe=EXCLUSIVE`). A job gets the whole node's
  memory unless it asks for less.

## Queues

Updated on 2026-11-20 09:22 CST · fnal_lq

- **Always pass `--qos`.** With none a job falls to `opp` (priority 10, 8 h) and waits at
  the back of the queue.
- `normal`: priority 250, 1 day, no per-job caps. Use it for anything beyond test's caps:
  48³, 4 GPUs, more than one node.
- `test`: priority 500, 30 min.
  - Per job: at most **2 GPUs**, 1000 CPUs, **500G** memory.
  - Group-wide (every nplqcd user together): 4 GPUs, 2 nodes.
  - Per user: 1 running, 3 submitted (`sacctmgr show qos`, 2026-11-20).
  - A full 4-GPU lq2 node is therefore never a test job: it pends forever.
  - On lq3 a test job must pass `--mem=500G` or less, or it is refused with
    `QOSMaxMemoryPerJob` (the default is the whole 1.5 TB).
- No interactive or debug partition. The agent runs `squeue` and `sinfo` freely, and
  `sbatch` and `scancel` only with the user's approval of the exact command. It never
  runs `salloc`. Submit, report the job id, hand back. Watch a job only when asked, with
  one `common/watch_job.sh JOBID LOG STALL_MIN POLL_S MAX_H` in the background.

## Launching (both machines)

Updated on 2026-11-23 10:07 CST · fnal_lq

- **Always `srun --mpi=pmix`, one step over all nodes, inside `sbatch`.** It is the recipe of
  both reference projects on lq (Grid-TXQCD on lq2,
  `/lustre2/nplqcd/Grid-TXQCD/production/MULTI_GPU_SRUN_NOTES.md`; Grid-CPU on lq3) and works
  on one node and across nodes.
  - `mpirun` cannot start ranks on other nodes ("An ORTE daemon has unexpectedly failed
    after launch") and Grid-TXQCD measured it 8× slower than `srun` on one node.
  - srun does not inherit the job's `--cpus-per-task` (Slurm ≥ 22.05): pass it again.
  - The launchers (`5_studies/hasenbusch_tune/lq/gate16_puregrid.sh`, `traj48_lq.sh`,
    `forces48_lq.sh`) run a binding probe through the same srun line first, and write a
    `LAUNCH` line beside the Perlmutter-format `ENV` line (stack, `libmpi`, layout, MCA and
    UCX settings). `gate16_puregrid.sh` keeps `LQ_LAUNCHER=mpirun` only to reproduce the
    11-19 runs.
- **Watch the run's own log.** The launchers send Grid's output to `runs/<run>/hmc.log`;
  the Slurm log falls silent once the run starts, so a watcher on it raises a false stall.
  A multigrid build prints nothing for many minutes (about 40 min on one lq3 node).

## Launching on lq2 (GPU)

Updated on 2026-11-23 10:07 CST · fnal_lq

- **gompi, one rank per GPU, whole nodes, QOS normal.** Each rank's GPU and that GPU's NUMA
  node are set by the launcher's `select_gpu` wrapper (`--cpu-bind=none`). 48³: 4 nodes ×
  4 ranks, `--mpi 1.2.2.4`, Perlmutter's layout, and Perlmutter's speed (+1.0%, §7e).
- **Grid's `Node communicator of size 1` under gompi is harmless.** gompi's OpenMPI 4.1.5
  (external PMIx 4.2.6) does not group a node's ranks (`MPI_COMM_TYPE_SHARED` gives groups
  of 1 under every launcher; probe 1363335), but the speed is unaffected (L222).
- **Only for 4 ranks on QOS test's 2 GPUs:** two ranks on one GPU time-slice it (6× slower)
  unless NVIDIA's MPS daemon runs (`GPU_MPS=1`, started and stopped by `gate16_puregrid.sh`).
  HPC-X 2.15 (`LQ_MPI_STACK=hpcx`, from
  `/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/2023/comm_libs/12.2/hpcx/hpcx-2.15`,
  loaded by `lq.sh` with `hpcx-init.sh` and `hpcx_load`) groups the ranks but cannot be
  started by `srun` (no munge plugin in its PMIx), so it needs `LQ_LAUNCHER=mpirun` and one
  node. Neither is needed on whole nodes.
- **Memory:** 48³ peaks at ~30.5 of 80 GB per GPU at `--device-mem 8000`; a larger cap is
  open (Perlmutter measured −8% at 16000).

## Launching on lq3 (CPU)

Updated on 2026-11-23 10:07 CST · fnal_lq

- **MPI:** Grid-CPU's OpenMPI 4.1.8 at `/lustre2/nplqcd/install/openmpi-4.1.8-ucx-gcc15`,
  built against the system UCX 1.21, the only UCX on lq3 with the mlx5 transports
  (`LQ_MPI_STACK=lq3-ompi418`). It groups a node's ranks correctly. `lq.sh` sets
  `OMPI_MCA_pml=ucx`, `OMPI_MCA_osc=ucx` and `UCX_NET_DEVICES=mlx5_0:1`; srun passes them to
  every node. Multi-node logs fill with harmless UCX warnings
  (`ibv_query_port_speed ... Protocol not supported`).
- **Binding:** `srun --cpus-per-task=T --cpu-bind=cores`,
  `OMP_NUM_THREADS=T OMP_PLACES=cores OMP_PROC_BIND=close`, Grid `--threads T --shm 2048`.
- **48³ rank counts:** with the default multigrid blocks (4.4.4.4, then 2.3.3.3) the
  decomposition must have x ∈ {1, 3}, y, z ∈ {1, 2}, t ∈ {1, 2, 4}: 16, 24 or 48 ranks, never
  32 or 64 (job 1364148 died after 12 min in `Cartesian_red_black.h`). `forces48_lq.sh` checks
  this before launching; `traj48_lq.sh` does not yet.
- **Layouts that ran (48³, §7e):** 1 node × 16 ranks × 4 cores `1.2.2.4` (~11 h per
  trajectory); 4 nodes × 12 × 5 `3.2.2.4` (~2.7 h); **6 nodes × 8 × 8 `3.2.2.4` (~1.75 h),
  the production layout** (user, 11-23). Scaling is linear to 6 nodes.
- **Not yet tried:** `--dslash-asm` (Grid-CPU: +20–30% on the Wilson hopping term), other
  routes for rung PF3, multigrid rebuild cadence. A/B them with `forces48_m10_cpu_job.sh`
  (`GRID_EXTRA_ARGS`, `HASEN_GRID_*`); baseline job 1364292.

## Building

Updated on 2026-11-20 09:22 CST · fnal_lq

- **Grid:** `1_build_grid/lq/build_grid_stock_hmc.sh` (`TARGET=gpu|cpu`). Job wrappers:
  `build_grid_stock_hmc_job.sh` (lq1, gpu) and `build_grid_stock_hmc_cpu_job.sh` (lq3,
  exclusive, `--mem=0`).
  - It stages a patched copy of `builds/external/Grid` at `3d3eff86` into
    `builds/grid-stock-<sha8>-p08_<target>/`.
  - It applies patches 01–08 plus `lq-01-perfcount-rdtsc-stubs-cuda.patch`. Stock
    `PerfCount.h` lacks CUDA stubs for the x86 cycle counter; the patch sits inside
    `#ifdef GRID_CUDA`, so it is inert on CPU.
- **Driver:** `5_studies/hasenbusch_tune/lq/build_driver_stock_job.sh` (gpu) and
  `build_driver_stock_cpu_job.sh` (cpu). Both run the portable
  `perlmutter/build_driver_stock.sh` and write `installs/dev/m10_<target>/PROVENANCE.txt`.
  `make -j` runs only inside these jobs.
- **gpu environment:**
  - Lmod modules `gompi/2023a`, `ucx_cuda/1.14.1_cuda_12.2.1`, `ucc_cuda/1.2.0_cuda_12.2.1`.
  - One CUDA only: the `cuda/12.2.1` that `ucx_cuda` loads. The 2026-05 build also put
    NVHPC's CUDA first on PATH, mixing two CUDA installs in one build.
  - `CUDA_ARCH=sm_80`.
- **cpu environment:**
  - spack `/srv/software/el9/spack/v1.2.2`, environment `gcc-15-toolset`: gcc 15.3, plus
    GMP, MPFR and FFTW from the same spack.
  - Configure: `--enable-simd=AVX512 --enable-shm=shmopen -march=znver5`.
  - **FFTW is required.** Stock Grid's CPU `FFT.h` does not compile without it (job
    1363360); GPU builds use cuFFT.
- **MPFR trap (gpu):** GMP and MPFR are complete in `/usr` and configure finds them with no
  flag. Never pass `--with-mpfr=/usr`: it adds `-I/usr/include`, which breaks libstdc++'s
  `#include_next`, and `-L/usr/lib`, EL8's 32-bit directory. On cpu, point
  `--with-gmp`/`--with-mpfr` at spack's prefixes, never at `/usr`.
- **Lmod and `set -u`:** Lmod is not initialised in non-login shells (sbatch, scripts).
  - `lq.sh` sources the profile.d files itself.
  - Those files read unset variables, so `lq.sh` pauses a caller's `set -u` around them
    (with it on: `MODULEPATH: unbound variable`).
- **cmake:** compute nodes have no system `cmake`. Load a module inside jobs.

## Paths

Updated on 2026-11-20 09:22 CST · fnal_lq

- **Workspace root:** `/lustre2/nplqcd/vayyar/grid_qcd` (Lustre, no user quota). Layout as
  in handoff §2:
  - our three repos;
  - `builds/` (here on Lustre, recreatable) and `installs/{external,dev,test,prod,retired}`;
  - `inputs/`, `runs/`, `logs/` (Slurm logs and gate summaries), `_archive/`.
  - `/home` is NFS: keep it small.
- **`inputs/`:** `cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime` (16³ gate),
  `cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime` (48³, a copy), and `perlmutter_ref/`
  (the 16³ m10 reference logs).
- **`installs/external/lime`** links to `/lustre2/nplqcd/dwf/c-lime/install` (`CLIME_ROOT`).
- **Archived:** `_archive/2026_05_lq_gpu_build/` keeps only the configure records of the
  2026-05 Grid and Grid-TXQCD GPU builds; the installs themselves are gone.
  `_archive/lq_launch_reference/` keeps the older single-node launchers.
  `3_examples/*/build.sh` and some READMEs still point at the old `install-*-gpu/` and
  `submit_scripts/`, which no longer exist here.
- **Shared group data under `/lustre2/nplqcd/`** is read-only for this project:
  - `cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3/*.lime` (Chroma SZINQIO, 6.1 GB each);
  - `dwf/c-lime/install/bin` (`lime_contents`, `lime_extract_record`);
  - `Grid-TXQCD/production/` and `Grid-CPU/` (reference builds and notes);
  - `install/openmpi-4.1.8-ucx-gcc15` (lq3 MPI);
  - the Chroma install `chroma/install_oct_2025/`.

## Analysis and tools

Updated on 2026-11-20 09:22 CST · fnal_lq

- **Python:** the repo's uv environment (uv in `~/.local/bin`, Python 3.12), e.g.
  `uv --directory grid-lqcd-workflow run python 5_studies/hasenbusch_tune/data_extraction_scripts/compare_gate16_xmachine.py REF NEW`.
- **Older analysis env:** `module load mambaforge/23.1.0-4`, then
  `conda activate /lustre2/nplqcd/vayyar/conda-envs/hmc-analysis`.
- **JupyterLab:** `~/.venvs/jupyter`, set up as in `__docs/2026_05_29_jupyterlab_setup.md`.
- **Login-node tools:**
  - `claude` is a shell function that routes accounts; `command claude` reaches the
    binary (needed for `--version` and headless probes).
  - `codex` is installed.
  - No `gh`: PRs are made through the GitHub web UI.

## Git branch on lq

Updated on 2026-11-22 12:11 CST · fnal_lq

- **lq commits go on `lq-port`** (since 2026-11-22); Perlmutter keeps
  `hasenbusch-tune-compact`. On 11-20 both machines pushed to the same branch: the lq push
  was rejected, `pull --rebase` rewrote the lq hashes, and a commit had to re-cite one.
- **Pick up Perlmutter's work by merge, never rebase:** `git fetch origin`, then
  `git merge origin/hasenbusch-tune-compact` on `lq-port`, so hashes cited in docs stay valid.
- **Hand back at milestones** (e.g. a promotion) by merging `lq-port` into
  `hasenbusch-tune-compact` or by a PR in the web UI, only with the user's approval.
- **The checkout is shared by every lq session:** never `git switch` it to another branch
  without telling the user, since the switch moves them all.

## Older notes (Grid-TXQCD era, 2026-05 to 09, not rechecked on stock Grid)

Updated on 2026-11-20 09:22 CST · fnal_lq

- `Test_general_stencil` failed (nvlink hugepages not configured), and so did
  `Test_innerproduct_norm` (single-precision GPU rounding). Neither was a regression.
- RHMC `OFRp.hi` was hardcoded to 100.0 in `dwrhmc_mobius.cc`
  (`3_examples/mobius_dwf_test/README.md`).
- Cold start plus stout smearing gave NaN, so cold smoke tests used `STOUT_NSMEAR=0`.
