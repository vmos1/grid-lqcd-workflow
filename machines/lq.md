# grid_qcd on lq (Fermilab): account, queues, paths, traps

Updated on 2026-11-20 09:22 CST · fnal_lq

Read by agents on every session run on lq, per rule 7 of the workspace
`AGENTS.md`. Shell environment is `lq.sh` beside this file. General lq rules
live in the user's global instruction file; this file holds only what this
project needs on this machine. Campaign facts and numbers belong in `__docs/`
(campaign `lq-port`, handoff `2026_11_18_handoff_lq_port.md` §7c–7d), not here.

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

## Launching on lq2 (GPU)

Updated on 2026-11-20 09:22 CST · fnal_lq

- **Use HPC-X, not gompi (`LQ_MPI_STACK=hpcx`).**
  - **The fault:** gompi/2023a's OpenMPI 4.1.5 (external PMIx 4.2.6) does not see ranks on
    one node as node-local. `MPI_COMM_TYPE_SHARED` gives groups of 1 under every `mpirun`
    and `srun --mpi=pmix*` form, and its shared-memory BTL (vader) refuses to start. Grid
    prints `Node communicator of size 1` and sends every halo through MPI (probe job
    1363335, `runs/2026_11_19_mpi_locality_probe/`). Grid-TXQCD's production logs show the
    same.
  - **The fix:** HPC-X 2.15 from the NVHPC 23.7 tree groups ranks correctly (probe
    1363336). It lives at
    `/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/2023/comm_libs/12.2/hpcx/hpcx-2.15`
    and is OpenMPI 4.1.5rc2 with internal PMIx 3, UCX 1.15, CUDA-aware.
  - **How `lq.sh` loads it:** the gompi modules first, then `source hpcx-init.sh` and
    `hpcx_load` (HPC-X's module file is broken). It also turns hcoll and ucc collectives off.
  - The gompi-built binary runs on HPC-X's `libmpi.so.40` unchanged: same ABI, and
    `LD_LIBRARY_PATH` wins over its RUNPATH.
  - **Check:** Grid's `Node communicator of size N` must equal the ranks per node.
- **More ranks than GPUs needs MPS (`GPU_MPS=1`).**
  - Without MPS, two processes on one GPU time-slice it: 7.9 ms per CG iteration against
    0.91 ms with NVIDIA's MPS daemon (handoff §7d, 333 s against 52.5 s per 16³ trajectory).
  - The gate launcher starts and stops the daemon inside the job, with per-run pipe and
    log directories under `/tmp`.
  - Under MPS a rank's GPU is its position in `CUDA_VISIBLE_DEVICES`.
  - This matters on QOS test (2 GPUs). On a full node use one rank per GPU and no MPS.
- **Single node:** `mpirun -np N --map-by ppr:N:node --bind-to none` inside `sbatch`, with
  a wrapper that sets each rank's GPU and binds it to that GPU's NUMA node
  (`5_studies/hasenbusch_tune/lq/gate16_puregrid.sh`, §10).
- **Multi-node: unsolved, and not yet run by this project.**
  - `mpirun` cannot start its daemons on other nodes ("An ORTE daemon has unexpectedly
    failed after launch"; no keyless ssh, no route back). Grid-TXQCD therefore launches
    with `srun --mpi=pmix`
    (`/lustre2/nplqcd/Grid-TXQCD/production/MULTI_GPU_SRUN_NOTES.md`).
  - But gompi under `srun` has the grouping fault above, and HPC-X cannot be launched by
    `srun` (its PMIx has no munge plugin).
  - Solve this before the first multi-node lq2 run.
- **Check before every run:** the gate launcher writes a `LAUNCH` line beside the
  Perlmutter-style `ENV` line. It records stack, `libmpi`, MPS, the GPU-to-NUMA map, MCA and
  UCX settings. It refuses to run when the binary's `libmpi` and `mpirun` are not both
  under the stack named by `LQ_MPI_STACK`.

## Launching on lq3 (CPU)

Updated on 2026-11-20 09:22 CST · fnal_lq

- **MPI:** Grid-CPU's OpenMPI 4.1.8 at `/lustre2/nplqcd/install/openmpi-4.1.8-ucx-gcc15`,
  built against the system UCX 1.21, the only UCX on lq3 with the mlx5 transports
  (`LQ_MPI_STACK=lq3-ompi418`, the only cpu stack).
  - It groups ranks correctly: 16 ranks gave groups of 16 (build job 1363360), and the gate
    printed node communicator 4 of 4.
  - `lq.sh` sets `OMPI_MCA_pml=ucx`, `OMPI_MCA_osc=ucx` and `UCX_NET_DEVICES=mlx5_0:1`.
- **Gate layout:** 4 ranks × 16 cores,
  `mpirun --map-by ppr:4:node:PE=16 --bind-to core`,
  `OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close`, Grid `--threads 16 --shm 2048`.
- **Not tried with our driver:** Grid-CPU's benchmarks favour 16 ranks × 4 threads per
  node and `--dslash-asm` for solvers and HMC
  (`/lustre2/nplqcd/Grid-CPU/AGENT_REFERENCE.md`). Worth testing at 48³.
- **Multi-node:** `mpirun` cannot start remote daemons here either.
  - Grid-CPU uses `srun --mpi=pmix`, or its shim `bench_lq3/mpirun_srun_shim/mpirun`.
  - Forward the three settings above, plus the `OMP_*` variables, `PATH` and
    `LD_LIBRARY_PATH`.
  - Not yet run by this project.

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

## Older notes (Grid-TXQCD era, 2026-05 to 09, not rechecked on stock Grid)

Updated on 2026-11-20 09:22 CST · fnal_lq

- `Test_general_stencil` failed (nvlink hugepages not configured), and so did
  `Test_innerproduct_norm` (single-precision GPU rounding). Neither was a regression.
- RHMC `OFRp.hi` was hardcoded to 100.0 in `dwrhmc_mobius.cc`
  (`3_examples/mobius_dwf_test/README.md`).
- Cold start plus stout smearing gave NaN, so cold smoke tests used `STOUT_NSMEAR=0`.
