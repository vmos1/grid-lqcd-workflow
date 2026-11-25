#!/bin/bash
#SBATCH --job-name=chroma-cpu-lq3-debug
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/chroma_cpu_lq3_debug_%j.log
#
# One job to localise and fix the wrong mg_proto QPhiX solves of the lq3 CPU Chroma (job 1365191:
# rel resid ~1.5 against chroma's operator in every mg_proto QPhiX case; chroma's own QPhiX solver and
# native BiCGStab passed). Steps (STEPS, default "unit ref variants"):
#   unit      mgproto_unit_tests.sh: mg_proto's own QPhiX-vs-reference operator tests, built with and
#             without MG_QPHIX_COMPRESS12 (decisive for the compression hypothesis).
#   ref       the base build, case mgproto_ref_4r only: mg_proto WITHOUT QPhiX against chroma's operator.
#   variants  for each of VARIANTS (default "nocomp12 o2sky gcc11", cheapest and most likely first; see
#             build_chroma_cpu.sh): check, build into its own prefix installs/external/chroma-cpu-lq3-<v>,
#             then all test cases. Stops at the first variant whose tests all pass (STOP_AT_PASS=1;
#             STOP_AT_PASS=0 builds and tests every variant). The base install is never touched.
# Build timing for scale (job 1364686): whole stack 9 min, mg_proto 25 s, chroma 2.5 min; the test
# cases about 1 min each.
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
LQ=${ROOT}/grid-lqcd-workflow/1_build_chroma/lq
STEPS=${STEPS:-unit ref variants}
VARIANTS=${VARIANTS:-nocomp12 o2sky gcc11}
STOP_AT_PASS=${STOP_AT_PASS:-1}
declare -a SUMMARY=()

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date), steps: ${STEPS}; variants: ${VARIANTS}"
for step in ${STEPS}; do
  echo
  echo "==== ${step} ($(date +%T))"
  case ${step} in
    unit)
      JOBS=64 bash "${LQ}/mgproto_unit_tests.sh"
      SUMMARY+=("unit: exit $? (verdicts in runs/*_mgproto_unit_${SLURM_JOB_ID}/SUMMARY.txt)")
      ;;
    ref)
      CASES=mgproto_ref_4r bash "${LQ}/build_chroma_cpu.sh" test
      SUMMARY+=("ref (base, mg_proto without QPhiX): exit $?")
      ;;
    variants)
      for v in ${VARIANTS}; do
        echo
        echo "==== variant ${v}: check ($(date +%T))"
        if ! VARIANT=${v} bash "${LQ}/build_chroma_cpu.sh" check; then
          SUMMARY+=("variant ${v}: preflight FAILED"); continue
        fi
        echo "==== variant ${v}: build ($(date +%T))"
        if ! VARIANT=${v} JOBS=64 bash "${LQ}/build_chroma_cpu.sh" build; then
          SUMMARY+=("variant ${v}: build FAILED"); continue
        fi
        echo "==== variant ${v}: test ($(date +%T))"
        if VARIANT=${v} bash "${LQ}/build_chroma_cpu.sh" test; then
          SUMMARY+=("variant ${v}: ALL TESTS PASSED (installs/external/chroma-cpu-lq3-${v})")
          [[ "${STOP_AT_PASS}" == 1 ]] && break
        else
          SUMMARY+=("variant ${v}: tests FAILED (runs/*_chroma_cpu_lq3_${v}_smoke_${SLURM_JOB_ID}/RESULTS.txt)")
        fi
      done
      ;;
    *) SUMMARY+=("unknown step ${step}") ;;
  esac
done

echo
echo "=== summary ($(date +%T))"
printf '  %s\n' "${SUMMARY[@]}"
echo "=== done $(date)"
