#!/bin/bash
#SBATCH --job-name=m10-gpu-driver-build
#SBATCH --partition=lq1_cpu
#SBATCH --account=nplqcd.lq1_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=intel6248
#SBATCH --time=00:30:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_driver_m10_gpu_%j.log
#
# Build the compact-Schur HMC driver (mode 1: stock Grid 3d3eff86 + patches) against the lq
# GPU Grid build and make the release installs/dev/m10_gpu (bin/ + PROVENANCE.txt).
# The compile is ../perlmutter/build_driver_stock.sh, portable given GRID_CONFIG and BIN: it
# sources config.sh, which selects machines/lq.sh on lq. One nvcc process: ~3 min, 2.4 GB peak
# (job 1363179), so 4 cores, 16 GB. lq1, not lq2: EL8 like the lq2 GPU nodes, no GPU needed to
# compile. Never overwrites a release.
# lq1 has no NVIDIA driver, so the shared script's final ldd check always reports libcuda.so.1
# "not found" there and exits 1 (job 1363179; it resolves on lq2 and on the login node). This
# job accepts exactly that case, re-checking the rest itself; any other failure stays fatal.
set -euo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
WF=${ROOT}/grid-lqcd-workflow
HB=${WF}/5_studies/hasenbusch_tune
STAGE=${ROOT}/builds/grid-stock-3d3eff86-p08_gpu
REL=${ROOT}/installs/dev/m10_gpu
export GRID_CONFIG=${STAGE}/Grid/build/grid-config
export BIN=${REL}/bin/gen_qcd_hasenbusch_tune_compact_schur_stock

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date)"
if [[ -e "${BIN}" ]]; then
  echo "ERROR: ${BIN} exists; a release is never overwritten. Make a new one." >&2
  exit 1
fi
mkdir -p "${REL}/bin"

rc=0
bash "${HB}/perlmutter/build_driver_stock.sh" || rc=$?
ldd_note="0 unresolved on $(hostname)"
if (( rc != 0 )); then
  [[ -x "${BIN}" ]] || exit "${rc}"
  # Patch 04's header must be the one compiled (the shared script checks this before ldd).
  GCM=${STAGE}/Grid/Grid/algorithms/multigrid/GeneralCoarsenedMatrix.h
  used=$(tr ' \\' '\n\n' < "${BIN}.d" | grep 'multigrid/GeneralCoarsenedMatrix.h$' | head -1)
  if [[ "$(readlink -f "${used}")" != "$(readlink -f "${GCM}")" ]]; then
    echo "ERROR: compiled GeneralCoarsenedMatrix.h = '${used:-none}', expected ${GCM}" >&2
    exit "${rc}"
  fi
  # Only libcuda.so.1 may be unresolved (needs lq.sh for every other library).
  source "${WF}/machines/lq.sh"
  missing=$(ldd "${BIN}" | awk '/not found/ {print $1}' | tr '\n' ' ')
  if [[ "${missing}" != "libcuda.so.1 " ]]; then
    echo "ERROR: unresolved libraries on $(hostname): ${missing}" >&2
    exit "${rc}"
  fi
  ldd_note="only libcuda.so.1 unresolved on $(hostname) (no NVIDIA driver on lq1 CPU nodes; resolves on lq2)"
  echo "OK: ${ldd_note}; build accepted."
fi

dirty=""
[[ -n "$(git -C "${WF}" status --porcelain --untracked-files=no)" ]] && dirty=", dirty"
{
  echo "# Release m10_gpu: mode 1 (stock Grid + patches 01-08 + lq fixes, compact-Schur driver), lq2 A100"
  echo "built:      $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname), job ${SLURM_JOB_ID}"
  echo "workflow:   $(git -C "${WF}" rev-parse HEAD) ($(git -C "${WF}" rev-parse --abbrev-ref HEAD)${dirty})"
  echo "driver:     ${HB}/src/gen_qcd_hasenbusch_tune_compact_schur.cc  sha256=$(sha256sum "${HB}/src/gen_qcd_hasenbusch_tune_compact_schur.cc" | cut -d' ' -f1)"
  echo "build:      ${HB}/perlmutter/build_driver_stock.sh, GRID_CONFIG=${GRID_CONFIG}"
  echo "modules:    $(bash -c "source ${WF}/machines/lq.sh >/dev/null 2>&1 && module -t list 2>&1 | tr '\n' ' '")"
  echo "exe:        $(basename "${BIN}")  sha256=$(sha256sum "${BIN}" | cut -d' ' -f1)  bytes=$(stat -c %s "${BIN}")"
  echo "deps:       $(basename "${BIN}").d lists every header the compile read"
  echo "ldd:        ${ldd_note}"
  echo "--- Grid build: ${STAGE}/PROVENANCE.txt ---"
  cat "${STAGE}/PROVENANCE.txt"
} > "${REL}/PROVENANCE.txt"
echo "=== done $(date)"
