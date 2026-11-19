#!/bin/bash
#SBATCH --job-name=gate16-m10-gpu
#SBATCH --partition=lq2_gpu
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/gate16_m10_gpu_%j.log
#
# The 16^3 gate of release m10_gpu on one lq2 node against the Perlmutter reference
# (__docs/2026_11_18_handoff_lq_port.md section 7c). 4 ranks on 2 A100 (two per GPU): QOS test
# caps a job at 2 GPUs and 500 GB (MaxTRES, checked 2026-11-19; a 4-GPU test job pends forever
# on QOSMaxMemoryPerJob / the GPU cap). The decomposition is Perlmutter's, so the comparison
# holds; one GPU per rank is first exercised by the 48^3 runs on QOS normal.
# 16 cores per rank take all 64 cores of the node, so the binding probe never meets cores of
# another job. Three cases, one after the other, each a gate16_puregrid.sh run with its own cap:
#   t10  CASE=traj    one trajectory, CG routes, --mpi 1.1.1.4     (ref runs/2026_10_14_gate16s_t10)
#   f10  CASE=forces  FORCES_ONLY, one sample, same recipe         (ref runs/2026_10_14_gate16s_f10)
#   mg   CASE=traj    Grid MG on rungs 0,1 and their heatbaths, --mpi 1.1.2.2, GRID_MG_BLOCK2=2.2.1.3
#                     (optional; ref runs/2026_11_13_m8_gate16_mg, binary _stock_m8)
# Layout tests, by CASES only (not in the default set): with 2 ranks per GPU, job 1363337 ran
# 10x slower per CG iteration than Perlmutter even with the node communicator fixed.
#   t10r2   t10 on 2 ranks, one per GPU, --mpi 1.1.1.2 (decomposition differs from the reference)
#   t10mps  t10 on 4 ranks, 2 per GPU, with NVIDIA's MPS daemon running
# QOS test: priority 500, 30 min, one job per user; the reference t10 took 43 s per
# trajectory on Perlmutter. The comparison runs afterwards on the login node.
# Optional: BIN=<binary> (default the m10_gpu release), TAG=<suffix for the run names>,
# LQ_MPI_STACK=gompi|hpcx (default gompi; machines/lq.sh). Job 1363314 ran on gompi, whose
# OpenMPI gives Grid a node communicator of size 1; HPC-X groups the 4 ranks (probe 1363336).
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
LAUNCH=$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune/lq/gate16_puregrid.sh
BIN=${BIN:-$ROOT/installs/dev/m10_gpu/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
TAG=${TAG:-}
LQ_MPI_STACK=${LQ_MPI_STACK:-gompi}
CASES=${CASES:-t10 f10 mg}

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date), LQ_MPI_STACK=${LQ_MPI_STACK} TAG=${TAG} CASES=${CASES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
nvidia-smi topo -m
numactl -H || echo "numactl -H failed"

# No route, layout or mode knob may leak from the submitting shell into a case: every case
# passes its own through env (the launcher scrubs most families itself, but reads the routes,
# MPI_GEOM and the seed from its caller).
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(FORCES_|HASEN_|GRID_MG_|CUSTOM_|DIAG_|MPI_GEOM$|HMC_SEED_OFFSET$|CASE$|CASE_MIN$|RUN$|NRANKS$|GPU_MPS$)' || true)

# run_case <name> <traj|forces> <cap minutes> [VAR=value ...]
run_case() {
  local name=$1 mode=$2 cap=$3
  shift 3
  echo
  echo "---- case ${name}: CASE=${mode} cap ${cap} min $* ($(date +%T))"
  local t0=$SECONDS
  env CASE="$mode" CASE_MIN="$cap" BIN="$BIN" RUN="lq_gate16_${name}${TAG}" LQ_MPI_STACK="$LQ_MPI_STACK" "$@" bash "$LAUNCH"
  local rc=$?
  echo "case ${name}: exit=${rc}  $((SECONDS - t0)) s"
}

for c in $CASES; do
  case $c in
    t10)    run_case t10 traj 8 ;;
    f10)    run_case f10 forces 5 ;;
    mg)     run_case mg traj 8 MPI_GEOM=1.1.2.2 HASEN_GRID_MG_RUNGS=0,1 HASEN_GRID_MG_HEATBATH_RUNGS=0,1 GRID_MG_BLOCK2=2.2.1.3 ;;
    t10r2)  run_case t10r2 traj 8 NRANKS=2 MPI_GEOM=1.1.1.2 ;;
    t10mps) run_case t10mps traj 8 GPU_MPS=1 ;;
    *)      echo "unknown case '$c' (t10 f10 mg t10r2 t10mps)" ;;
  esac
done
echo
echo "=== done $(date)"
