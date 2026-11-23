#!/bin/bash
#SBATCH --job-name=forces48-ab-cpu
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=5
#SBATCH --mem=0
#SBATCH --time=02:15:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/forces48_ab_cpu_%j.log
#
# lq3 speed A/B tests of release m10_cpu at 48^3: several forces48_lq.sh runs back to back on
# the same 4 nodes (12 ranks x 5 cores, --mpi 3.2.2.4), each one setting different from the
# baseline. Compare sample 1 (steady state) of each case; sample 0 carries the MG build.
#   base    trajectory routes, no extra flags (as job 1364292, rerun on these nodes)
#   asm     + Grid's AVX-512 assembly Wilson kernel (--dslash-asm; Grid-CPU: +20-30% on Dhop)
#   pf3mg   rung PF3 on the shared multigrid instead of mixed-precision CG
#   pf3cg   rung PF3 on fp64 CG instead of mixed-precision CG
# Physics is the same in every case (same action, tolerances); only solver path or kernel
# changes, so forces agree to solver tolerance. Optional: CASES (default all four), BIN.
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
LAUNCH=$ROOT/grid-lqcd-workflow/5_studies/hasenbusch_tune/lq/forces48_lq.sh
export BIN=${BIN:-$ROOT/installs/prod/m10_cpu/bin/gen_qcd_hasenbusch_tune_compact_schur_stock}
export LQ_TARGET=cpu LQ_MPI_STACK=lq3-ompi418 MPI_GEOM=3.2.2.4 TIMEOUT_MIN=30 FORCES_SAMPLES=2
CASES=${CASES:-base asm pf3mg pf3cg}
echo "=== job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, $(date), CASES=${CASES}"

# No route or flag may leak from the submitting shell into a case.
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(HASEN_GRID_|GRID_MG_|GRID_EXTRA_ARGS$|FO_ROUTES$|RUN$)' || true)

# run_case <name> [VAR=value ...]
run_case() {
  local name=$1
  shift
  echo
  echo "---- case ${name}: $* ($(date +%T))"
  local t0=$SECONDS
  env RUN="forces48_ab_${name}" "$@" bash "$LAUNCH"
  echo "case ${name}: exit=$?  $((SECONDS - t0)) s"
}

for c in $CASES; do
  case $c in
    base)  run_case base ;;
    asm)   run_case asm GRID_EXTRA_ARGS=--dslash-asm ;;
    pf3mg) run_case pf3mg HASEN_GRID_MG_RUNGS=0,1,2,3 HASEN_GRID_MIXED_CG_RUNGS= ;;
    pf3cg) run_case pf3cg HASEN_GRID_MIXED_CG_RUNGS= ;;
    *)     echo "unknown case '$c' (base asm pf3mg pf3cg)" ;;
  esac
done
echo
echo "=== summary: sample 1 force time per monomial (s), per case"
grep -a -h -E '^---- case|FORCES_ONLY level .* sample 1 ' "$ROOT/logs/forces48_ab_cpu_${SLURM_JOB_ID}.log" | sed -E 's/.*FORCES_ONLY level ([A-Za-z0-9]+) sample 1 .*refresh=([0-9.e+-]+) time=([0-9.e+-]+).*/  \1 refresh=\2 time=\3/'
echo "=== done $(date)"
