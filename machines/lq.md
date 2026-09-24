# grid_qcd on lq (Fermilab): account, queues, paths, traps

Updated on 2026-09-24 12:14 CDT · fnal_lq

Read by agents on every session run on lq, per rule 7 of the workspace
`AGENTS.md`. Shell environment is `lq.sh` beside this file. General lq rules
live in the user's global instruction file; this file holds only what this
project needs on this machine. Machine-independent facts belong in `__docs/`,
not here.

## Account and queues

- GPU account **`nplqcd.lq2_gpu`**, partition `lq2_gpu` (18 nodes, 4× A100
  each, 1-day limit). CPU work: `nplqcd.lq1_cpu` on `lq1_cpu`.
- **Always pass `--qos=normal`** (priority 250, 1 day). With no `--qos` a job
  falls to the default `opp` (priority 10, preemptible, 8 h) and sits at the
  back of the queue. `--qos=test` (priority 500, 30 min) is capped at 4 GPUs
  **group-wide**, shared by every nplqcd user: fine for one 4-GPU node, an
  8-GPU `test` job pends forever on `QOSGrpGRES`. Some older scripts in
  `submit_scripts/` omit `--qos`; add it before submitting.
- No interactive or debug GPU partition. The agent runs `squeue` and `sinfo`
  freely; `sbatch` and `scancel` only with the user's approval of the exact
  command; never `salloc`. Submit, report the job id and how to check it, and
  hand back; do not poll.

## Launching

- Multi-node `mpirun` must run under `sbatch`. `salloc -N2 ... bash script`
  runs the script on the login node (`lq.fnal.gov`, no GPUs) and OpenMPI dies
  with "An ORTE daemon has unexpectedly failed after launch". `sbatch` runs it
  on the first compute node. Proven launcher: `tune-hasenbusch.sbatch`
  (mpirun + `CUDA_VISIBLE_DEVICES[$OMPI_COMM_WORLD_LOCAL_RANK]`).
- Most sbatch scripts send Slurm's own `-o`/`-e` to `/dev/null` and write
  `runs/<run>/hmc/slurm-<jobid>.log` themselves.

## Filesystems and paths

- Workspace root `/lustre2/nplqcd/vayyar/grid_qcd` (Lustre, no user quota).
  Build and run here; `/home` is NFS, keep it small.
- `install-grid-gpu/`, `install-txqcd-gpu/`: mainline Grid and Grid-TXQCD
  installs. `build-grid-gpu/`, `build-txqcd-gpu/`: build trees; test and
  bench binaries live there and are not installed.
- `submit_scripts/`: lq sbatch scripts, not in any repo. `runs/<ensemble>/hmc/`
  and `/meas/`: job output; `runs/_archive/` holds the pre-cleanup logs,
  including the original 1-flavour EOFA/RHMC logs (traj 0–210,
  `mobius-eofa-1278542.log`, `mobius-rhmc-1278943.log`). Run directories and
  script names are descriptive; any dated one keeps unpadded `YYYY_M_D_`.
- Shared group data under `/lustre2/nplqcd/` is read-only for this project:
  `cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3/*.lime` (Chroma SZINQIO, 6.1 GB
  each, incl. `cfg_2000.lime`), `dwf/c-lime/install/bin` (`lime_contents`,
  `lime_extract_record`), `Grid-TXQCD/production/`, the Chroma install under
  `chroma/install_oct_2025/`.

## Build and runtime environment

- `source grid-lqcd-workflow/config.sh` (auto-selects `lq.sh`): modules
  `gompi/2023a`, `gcc/12.3.0`, `ucx_cuda`/`ucc_cuda` for CUDA 12.2, HDF5; CUDA
  itself from NVHPC 23.7 under `/srv/software/el8/`, `sm_80`.
- Rebuild the examples after any Grid reinstall; they link the install's
  `libGrid.a`. 1f EOFA/RHMC: `3_examples/mobius_dwf_test/build.sh` (against
  `install-txqcd-gpu`); 2f: `3_examples/mobius_2f_test/build.sh` (against
  `install-grid-gpu`); Hasenbusch tune: `5_studies/hasenbusch_tune/lq/build.sh`.
- Extend an HMC run with `4_analysis/hmc_extend <run>/hmc --trajectories N`
  (archives input.xml, sets `NoMetropolisUntil=0`), then resubmit.
- Analysis env: `module load mambaforge/23.1.0-4 && conda activate
  /lustre2/nplqcd/vayyar/conda-envs/hmc-analysis`. JupyterLab:
  `~/.venvs/jupyter`, setup in `__docs/2026_05_29_jupyterlab_setup.md`.

## Known failures on lq

- `Test_general_stencil` always fails (nvlink hugepages not configured) and
  `Test_innerproduct_norm` always fails (single-precision GPU rounding). Not
  regressions; see `README.md`.
- RHMC `OFRp.hi` is hardcoded to 100.0 in `dwrhmc_mobius.cc`; recheck on
  larger lattices (`3_examples/mobius_dwf_test/README.md`).
- Cold start + stout smearing gives NaN; use `STOUT_NSMEAR=0` for cold smoke
  tests (`5_studies/hasenbusch_tune/README.md`).

## Tools on the login node

- `claude` is a shell function that routes accounts; `command claude` reaches
  the binary (needed for `--version` and headless probes). `codex` is installed.
- No `gh`; PR creation is through the GitHub web UI.

## Where the rest is

lq has no operations campaign in `__docs/`. The run history of the lq-era
campaigns (Mobius EOFA/RHMC, early Hasenbusch tune) is in `__docs/_TOC.md`
under `hasen-tune` and its dated docs.
