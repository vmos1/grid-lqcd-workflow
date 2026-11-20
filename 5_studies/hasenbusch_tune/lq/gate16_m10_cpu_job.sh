#!/bin/bash
#SBATCH --job-name=gate16-m10-cpu
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/gate16_m10_cpu_%j.log
#
# The 16^3 gate of release m10_cpu on one lq3 node (EPYC 9355, 64 cores) against the Perlmutter
# reference (__docs/2026_11_18_handoff_lq_port.md section 7c). The CPU twin of
# gate16_m10_gpu_job.sh: gate16_puregrid.sh with LQ_TARGET=cpu, Perlmutter's decomposition
# (--mpi 1.1.1.4, 4 ranks), each rank on 16 cores with 16 OpenMP threads. Only Grid's GPU flags
# differ from the reference; the physics recipe, seed and tolerances are the same.
# QOS test on lq3: 30 min, one job, --mem required (400G here; per-job cap 500G). No CPU timing
# exists yet for this driver, so the first gate runs t10 and f10 only; mg can follow.
#   t10  CASE=traj    one trajectory, CG routes   (ref runs/2026_10_14_gate16s_t10)
#   f10  CASE=forces  FORCES_ONLY, one sample      (ref runs/2026_10_14_gate16s_f10)
# Optional: BIN=<binary> (default the m10_cpu release), TAG=<suffix> (default _cpu),
# CASES (default "t10 f10"; also mg).
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
LAUNCH=$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune/lq/gate16_puregrid.sh
BIN=${BIN:-$ROOT/installs/dev/m10_cpu/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
TAG=${TAG:-_cpu}
CASES=${CASES:-t10 f10}

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date), LQ_TARGET=cpu LQ_LAUNCHER=${LQ_LAUNCHER:-srun} TAG=${TAG} CASES=${CASES}"
lscpu | grep -E "^(Model name|Socket|Core|Thread|NUMA node)"
numactl -H | head -3 || echo "numactl -H failed"

# No route, layout or mode knob may leak from the submitting shell into a case.
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(FORCES_|HASEN_|GRID_MG_|CUSTOM_|DIAG_|MPI_GEOM$|HMC_SEED_OFFSET$|CASE$|CASE_MIN$|RUN$|NRANKS$|GPU_MPS$|LQ_MPI_STACK$)' || true)

# run_case <name> <traj|forces> <cap minutes> [VAR=value ...]
run_case() {
  local name=$1 mode=$2 cap=$3
  shift 3
  echo
  echo "---- case ${name}: CASE=${mode} cap ${cap} min $* ($(date +%T))"
  local t0=$SECONDS
  env LQ_TARGET=cpu CASE="$mode" CASE_MIN="$cap" BIN="$BIN" RUN="lq_gate16_${name}${TAG}" "$@" bash "$LAUNCH"
  local rc=$?
  echo "case ${name}: exit=${rc}  $((SECONDS - t0)) s"
}

for c in $CASES; do
  case $c in
    t10) run_case t10 traj 14 ;;
    f10) run_case f10 forces 8 ;;
    mg)  run_case mg traj 14 MPI_GEOM=1.1.2.2 HASEN_GRID_MG_RUNGS=0,1 HASEN_GRID_MG_HEATBATH_RUNGS=0,1 GRID_MG_BLOCK2=2.2.1.3 ;;
    *)   echo "unknown case '$c' (t10 f10 mg)" ;;
  esac
done
echo
echo "=== done $(date)"
