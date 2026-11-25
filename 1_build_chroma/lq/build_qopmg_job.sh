#!/bin/bash
#SBATCH --job-name=chroma-qopmg-lq3
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_qopmg_lq3_%j.log
#
# Build and test the QOP-MG CPU Chroma (build_qopmg.sh beside this file) on one lq3 node: QLA, QDP/C,
# QOPQDP, then chroma with --enable-qop-mg into installs/external/chroma-cpu-lq3-qopmg (the base install
# is only read). The sources are fetched first, on the login node (done 2026-11-24):
#   bash 1_build_chroma/lq/build_qopmg.sh fetch
# STEPS picks the actions (default "check build test validate"); the job stops at the first that fails.
# A re-run skips every package already installed. Expected: build ~1 h, test ~15 min, validate ~1 h (the
# QOP-MG 16^3x48 HMC pair rebuilds its subspace on most MD::ratio force solves); the 5 h limit covers the
# per-run timeouts (test 4 x 15 min, validate 40 + 90 + 45 min). RESULTS.txt is written as each case ends.
set -euo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
SCRIPT=${ROOT}/grid-lqcd-workflow/1_build_chroma/lq/build_qopmg.sh
STEPS=${STEPS:-check build test validate}

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date), steps: ${STEPS}"
for step in ${STEPS}; do
  echo
  echo "==== ${step} ($(date +%T))"
  JOBS=64 bash "${SCRIPT}" "${step}"
done
echo "=== done $(date)"
