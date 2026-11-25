#!/bin/bash
#SBATCH --job-name=chroma-gpu-lq2-build
#SBATCH --partition=lq1_cpu
#SBATCH --account=nplqcd.lq1_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=intel6248
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_chroma_gpu_lq2_%j.log
#
# Build the lq2 GPU Chroma (build_chroma_gpu.sh beside this file) on one lq1 node: LLVM 18.1.8, QMP,
# QDP-JIT, QUDA (multigrid, sm_80) and Chroma into installs/external/chroma-gpu-lq2, then the ldd
# acceptance. Why lq1: it runs EL8 with the same /srv/software/el8 module tree as lq2 (the lq2 Grid build
# is made there too), nvcc needs no GPU, and lq1 has idle nodes while lq2 is busy. Not lq3: EL9.
# The sources are fetched first, on the login node (no build):
#   bash 1_build_chroma/lq/build_chroma_gpu.sh fetch
# STEPS picks the actions (default "check build accept"); the job stops at the first that fails. check
# takes seconds and stops the job, with nothing built, on a non-EL8 node or a missing module or source.
# A re-run after a failure or a time-out resumes: chromaform skips every package already installed, but
# rebuilds the interrupted package from scratch (it empties its build folder first).
# 40 cores, 190 GB: LLVM at make -j40, the rest at -j24 (nvcc takes 2-5 GB per process).
# Expected (estimates, no lq timing on record): LLVM ~0.5 h, QMP + QDP-JIT ~0.3 h, QUDA ~1-2 h, Chroma
# ~1 h. The 8 h limit is about twice that: lq1 has idle nodes, so a long limit does not delay the start,
# and a time-out inside QUDA or Chroma would cost that whole package again.
set -euo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
SCRIPT=${ROOT}/grid-lqcd-workflow/1_build_chroma/lq/build_chroma_gpu.sh
STEPS=${STEPS:-check build accept}

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date), steps: ${STEPS}"
for step in ${STEPS}; do
  echo
  echo "==== ${step} ($(date +%T))"
  bash "${SCRIPT}" "${step}"
done
echo "=== done $(date)"
