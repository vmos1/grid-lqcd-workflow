#!/bin/bash
#SBATCH --job-name=m10-gpu-driver-build
#SBATCH --partition=lq1_cpu
#SBATCH --account=nplqcd.lq1_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint=intel6248
#SBATCH --time=01:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_driver_m10_gpu_%j.log
#
# Build the compact-Schur HMC driver (mode 1: stock Grid 3d3eff86 + patches) against the lq
# GPU Grid build and make the release installs/dev/m10_gpu (bin/ + PROVENANCE.txt).
# The compile is ../perlmutter/build_driver_stock.sh, portable given GRID_CONFIG and BIN: it
# sources config.sh, which selects machines/lq.sh on lq. One nvcc process, so 4 cores, 64 GB.
# lq1, not lq2: EL8 like the lq2 GPU nodes, no GPU needed to compile. Never overwrites a release.
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

bash "${HB}/perlmutter/build_driver_stock.sh"

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
  echo "--- Grid build: ${STAGE}/PROVENANCE.txt ---"
  cat "${STAGE}/PROVENANCE.txt"
} > "${REL}/PROVENANCE.txt"
echo "=== done $(date)"
