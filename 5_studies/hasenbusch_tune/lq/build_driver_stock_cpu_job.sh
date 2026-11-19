#!/bin/bash
#SBATCH --job-name=m10-cpu-driver-build
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/build_driver_m10_cpu_%j.log
#
# Build the compact-Schur HMC driver (mode 1: stock Grid 3d3eff86 + patches) against the lq3
# CPU Grid build (1_build_grid/lq/build_grid_stock_hmc_cpu_job.sh) and make the release
# installs/dev/m10_cpu (bin/ + PROVENANCE.txt). The CPU twin of build_driver_stock_job.sh.
# The compile is ../perlmutter/build_driver_stock.sh, portable given GRID_CONFIG and BIN: it
# sources config.sh, which sources machines/lq.sh; LQ_TARGET=cpu selects the lq3 toolchain.
# One g++ process; on lq3 (EL9) because the toolchain is EL9-only. The CPU path of the driver
# has never been compiled before (handoff section 7: the cuBLAS log-det has a CPU fallback,
# QUDA is compiled out). Never overwrites a release.
set -euo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
WF=${ROOT}/grid-lqcd-workflow
HB=${WF}/5_studies/hasenbusch_tune
STAGE=${ROOT}/builds/grid-stock-3d3eff86-p08_cpu
REL=${ROOT}/installs/dev/m10_cpu
export LQ_TARGET=cpu
export GRID_CONFIG=${STAGE}/Grid/build/grid-config
export BIN=${REL}/bin/gen_qcd_hasenbusch_tune_compact_schur_stock

echo "=== job ${SLURM_JOB_ID} on $(hostname), $(date)"
if [[ -e "${BIN}" ]]; then
  echo "ERROR: ${BIN} exists; a release is never overwritten. Make a new one." >&2
  exit 1
fi
[[ -f "${STAGE}/install/lib/libGrid.a" ]] || { echo "ERROR: no installed CPU libGrid under ${STAGE}/install" >&2; exit 1; }
mkdir -p "${REL}/bin"

# The shared script ends with an ldd check (0 unresolved, no libquda); any failure is fatal here.
bash "${HB}/perlmutter/build_driver_stock.sh"

dirty=""
[[ -n "$(git -C "${WF}" status --porcelain --untracked-files=no)" ]] && dirty=", dirty"
{
  echo "# Release m10_cpu: mode 1 (stock Grid + patches 01-08 + lq fixes, compact-Schur driver), lq3 EPYC 9355"
  echo "built:      $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname), job ${SLURM_JOB_ID}"
  echo "workflow:   $(git -C "${WF}" rev-parse HEAD) ($(git -C "${WF}" rev-parse --abbrev-ref HEAD)${dirty})"
  echo "driver:     ${HB}/src/gen_qcd_hasenbusch_tune_compact_schur.cc  sha256=$(sha256sum "${HB}/src/gen_qcd_hasenbusch_tune_compact_schur.cc" | cut -d' ' -f1)"
  echo "build:      ${HB}/perlmutter/build_driver_stock.sh, GRID_CONFIG=${GRID_CONFIG}"
  echo "toolchain:  LQ_TARGET=cpu (machines/lq.sh): spack gcc-15-toolset, MPI ${LQ_CPU_MPI_ROOT:-/lustre2/nplqcd/install/openmpi-4.1.8-ucx-gcc15}"
  echo "exe:        $(basename "${BIN}")  sha256=$(sha256sum "${BIN}" | cut -d' ' -f1)  bytes=$(stat -c %s "${BIN}")"
  echo "deps:       $(basename "${BIN}").d lists every header the compile read"
  echo "ldd:        0 unresolved on $(hostname)"
  echo "--- Grid build: ${STAGE}/PROVENANCE.txt ---"
  cat "${STAGE}/PROVENANCE.txt"
} > "${REL}/PROVENANCE.txt"
echo "=== done $(date)"
