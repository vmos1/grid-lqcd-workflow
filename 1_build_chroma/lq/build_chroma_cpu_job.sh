#!/bin/bash
#SBATCH --job-name=chroma-cpu-lq3-build
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_chroma_cpu_lq3_%j.log
#
# Build and smoke-test the lq3 CPU Chroma (build_chroma_cpu.sh beside this file) on one lq3 node:
# the EL9 toolchain runs only there, so configure runs here too. 64 cores, 1.5 TB: chromaform's
# make -j64. Staging runs first on the login node, with no network and no compiler:
#   bash 1_build_chroma/lq/build_chroma_cpu.sh prepare
# STEPS picks the actions (default "check build test"). check takes about a minute and stops the job
# before anything is built if cmake, autotools, xml2-config, jinja2 or a staged source is missing.
# A preflight-only job fits QOS test (on lq3 a test job must ask for 500G or less):
#   sbatch --qos=test --mem=500G --time=00:10:00 --export=ALL,STEPS=check build_chroma_cpu_job.sh
# A re-run after a failure resumes: chromaform skips every package already installed.
# Needs installs/external/buildtools-lq3 (cmake, autotools, libxml2; lq3 has none): build_buildtools_job.sh,
# so queue this job behind it with --dependency=afterok:<buildtools job id>.
set -euo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
SCRIPT=${ROOT}/grid-lqcd-workflow/1_build_chroma/lq/build_chroma_cpu.sh
STEPS=${STEPS:-check build test}

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date), steps: ${STEPS}"
for step in ${STEPS}; do
  echo
  echo "==== ${step} ($(date +%T))"
  JOBS=64 bash "${SCRIPT}" "${step}"
done
echo "=== done $(date)"
