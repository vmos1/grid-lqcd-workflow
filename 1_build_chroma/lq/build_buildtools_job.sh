#!/bin/bash
#SBATCH --job-name=buildtools-lq3
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_buildtools_lq3_%j.log
#
# Build autoconf, automake, libtool, cmake and libxml2 into installs/external/buildtools-lq3 on one
# lq3 node (build_buildtools.sh beside this file); lq3 lacks them and they are EL9 builds. The
# tarballs are fetched and verified first, on the login node, with no build:
#   bash 1_build_chroma/lq/build_buildtools.sh fetch
# STEPS picks the actions (default "preflight build test"). A re-run skips tools already installed.
# The Chroma build (build_chroma_cpu_job.sh) can be queued behind this job with
# --dependency=afterok:<this job id>.
set -euo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
SCRIPT=${ROOT}/grid-lqcd-workflow/1_build_chroma/lq/build_buildtools.sh
STEPS=${STEPS:-preflight build test}

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date), steps: ${STEPS}"
for step in ${STEPS}; do
  echo
  echo "==== ${step} ($(date +%T))"
  JOBS=64 bash "${SCRIPT}" "${step}"
done
echo "=== done $(date)"
