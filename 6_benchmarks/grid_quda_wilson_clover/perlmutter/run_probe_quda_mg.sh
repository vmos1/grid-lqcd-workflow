#!/usr/bin/env bash
# Run the Phase 0 QUDA-MG probe. Allocation-neutral: invoke inside an existing
# Slurm allocation, like run_benchmark_v2.sh and run_probe_grid_mg.sh.
#
# The comms/launch configuration is the v3 config, copied verbatim from
# run_benchmark_v2.sh rather than re-chosen, so this probe's numbers sit on the
# same footing as everything else in the campaign.
#
# Unlike the Grid probe this one needs QUDA_RESOURCE_PATH: QUDA autotunes on first
# use and writes a cache there. A cold cache inflates the first solve, which is why
# the probe runs a warm solve before any timed repeat.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(cd "${HERE}/.." && pwd)
WORKFLOW=$(cd "${BENCH_DIR}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)
source "${WORKFLOW}/machines/perlmutter.sh"

: "${PSCRATCH:?PSCRATCH must be set on Perlmutter}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'ERROR: run_probe_quda_mg.sh must be invoked inside an existing Slurm allocation\n' >&2
  exit 2
fi

GRID_SHA=${GRID_SHA:-$(git -C "${ROOT}/Grid" rev-parse HEAD)}
BIN=${BIN:-${PSCRATCH}/grid_quda_wilson_clover/benchmark/${GRID_SHA}/bin/probe_quda_mg_clover}
OUT=${OUT:-${ROOT}/runs/2026_9_11_quda_mg_probe}

LATT=${LATT:-16.16.16.48}
MPI=${MPI:-1.1.1.1}
MASS=${MASS:--0.2416}
CSW=${CSW:-1.20536588031793}
# wilson | clover. Wilson is doc §4's control -- MG is EXPECTED to lose on it.
ACTION=${ACTION:-clover}
CHECKERBOARD=${CHECKERBOARD:-even}
MG_MODE=${MG_MODE:-like_for_like}   # like_for_like | production
SLOPPY=${SLOPPY:-single}            # production runs SINGLE + RECONS_12
# Outer working precision. PRECISE=single makes the WHOLE QUDA solve fp32, the
# only configuration that matches a fully-fp32 Grid solve: QUDA's MG forces its
# sloppy precision to match the hierarchy, so it cannot be given an fp64 outer.
# ⚠️ fp32 cannot reach 1e-10 -- pair PRECISE=single with TOL=1e-6.
PRECISE=${PRECISE:-double}
# mixed | single. mixed = QUDA's production default: fp32 preconditioner but HALF
# null vectors and HALF coarse halos. single raises both to fp32, giving the ONE
# like-for-like precision comparison against Grid (which cannot do half at all).
# ⛔ single DENIES QUDA its half advantage ⇒ expect it slower than QUDA's best;
# it is a diagnostic, not QUDA's capability. double cannot be compiled.
MG_PRECISION=${MG_PRECISION:-mixed}
TOL=${TOL:-1e-10}
MAXITER=${MAXITER:-5000}
SOLVE_REPEATS=${SOLVE_REPEATS:-3}
# Stout smearing, applied through GRID's Smear_Stout -- the SAME code path as the
# Grid probe, deliberately, so the two sides cannot disagree on the smeared links.
STOUT_NSMEAR=${STOUT_NSMEAR:-0}
STOUT_RHO=${STOUT_RHO:-0.125}
RUN_CG=${RUN_CG:-1}
# ⛔ Set RUN_MG=0 to measure QUDA CG at SLOPPY=double. QUDA's MG setup requires the
# sloppy precision to match the hierarchy's and aborts otherwise ("Precisions 4 8 do
# not match"), and MG is built BEFORE the CG reference, so a fp64 CG row is only
# reachable with MG skipped.
RUN_MG=${RUN_MG:-1}
MG_VERIFY=${MG_VERIFY:-0}           # QUDA's own MG self-check; slow, use at bring-up
INPUT=${INPUT:-physical}
CFG=${CFG:-${ROOT}/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime}

ACCELERATOR_THREADS=${ACCELERATOR_THREADS:-8}
SHM_MB=${SHM_MB:-2048}
NTASKS=${NTASKS:-1}
NODES=${NODES:-1}
NTPN=${NTPN:-${NTASKS}}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
GPUS_PER_TASK=${GPUS_PER_TASK:-1}
RUN_ID=${RUN_ID:-$(date +%Y%m%dT%H%M%S)}
RUN_DIR=${RUN_DIR:-${OUT}/${RUN_ID}}
LOG=${LOG:-${RUN_DIR}/probe.log}
# ⛔ SHARED, not per-run. This used to default to ${RUN_DIR}/quda_resource, so every
# run started with a COLD tunecache and paid QUDA's autotune from scratch -- which
# lands in the first timed solve and, across a scan, adds a per-run cost that has
# nothing to do with the solver being measured. One persistent directory lets the
# tunecache accumulate and be reused.
QUDA_RESOURCE_PATH=${QUDA_RESOURCE_PATH:-${PSCRATCH}/grid_quda_wilson_clover/quda_tunecache}
export QUDA_RESOURCE_PATH

mkdir -p "${RUN_DIR}" "${QUDA_RESOURCE_PATH}"

case "${ACTION}" in wilson|clover) ;; *) printf 'ERROR: ACTION must be wilson|clover\n' >&2; exit 2 ;; esac
case "${CHECKERBOARD}" in even|odd) ;; *) printf 'ERROR: CHECKERBOARD must be even|odd\n' >&2; exit 2 ;; esac
case "${MG_MODE}" in like_for_like|production) ;; *) printf 'ERROR: MG_MODE must be like_for_like|production\n' >&2; exit 2 ;; esac
case "${SLOPPY}" in single|double) ;; *) printf 'ERROR: SLOPPY must be single|double\n' >&2; exit 2 ;; esac
case "${MG_PRECISION}" in mixed|single) ;; *) printf 'ERROR: MG_PRECISION must be mixed|single\n' >&2; exit 2 ;; esac
case "${INPUT}" in hot|physical) ;; *) printf 'ERROR: INPUT must be hot|physical\n' >&2; exit 2 ;; esac
if [[ "${INPUT}" == physical && ! -f "${CFG}" ]]; then
  printf 'ERROR: physical gauge configuration not found: %s\n' "${CFG}" >&2
  exit 1
fi

# --- v3 comms config, inherited verbatim from run_benchmark_v2.sh ---
export SLURM_CPU_BIND=${SLURM_CPU_BIND:-cores}
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export MPICH_RDMA_ENABLED_CUDA=${MPICH_RDMA_ENABLED_CUDA:-1}
export MPICH_GPU_IPC_ENABLED=${MPICH_GPU_IPC_ENABLED:-1}
export MPICH_GPU_EAGER_REGISTER_HOST_MEM=${MPICH_GPU_EAGER_REGISTER_HOST_MEM:-0}
export MPICH_GPU_NO_ASYNC_MEMCPY=${MPICH_GPU_NO_ASYNC_MEMCPY:-0}
export MPICH_OFI_NIC_POLICY=${MPICH_OFI_NIC_POLICY:-GPU}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export QUDA_ENABLE_MPS=${QUDA_ENABLE_MPS:-1}
export QUDA_ENABLE_P2P=${QUDA_ENABLE_P2P:-1}

# ⛔ --gpu-bind=none below is required: --gpu-bind=closest breaks CUDA IPC and hangs.
SELECT_GPU=${RUN_DIR}/select_gpu
cat > "${SELECT_GPU}" <<'EOF'
#!/bin/bash
export GPU=$SLURM_LOCALID
export NUMA=$SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=$GPU
exec numactl -m $NUMA -N $NUMA "$@"
EOF
chmod +x "${SELECT_GPU}"

[[ -x "${BIN}" ]] || {
  printf 'ERROR: probe binary not found: %s\n' "${BIN}" >&2
  printf 'Build it with: %s\n' "${HERE}/build_probe_quda_mg.sh" >&2
  exit 1
}

cache_state=empty
if find "${QUDA_RESOURCE_PATH}" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .; then
  cache_state=seeded
fi

{
  printf 'ENV DATE=%s\n' "$(date --iso-8601=seconds)"
  printf 'ENV HOST=%s SLURM_JOB_ID=%s\n' "$(hostname)" "${SLURM_JOB_ID:-none}"
  printf 'ENV LATT=%s MPI=%s MASS=%s CSW=%s INPUT=%s\n' "${LATT}" "${MPI}" "${MASS}" "${CSW}" "${INPUT}"
  printf 'ENV ACTION=%s CHECKERBOARD=%s MG_MODE=%s SLOPPY=%s MG_PRECISION=%s TOL=%s MAXITER=%s\n' \
    "${ACTION}" "${CHECKERBOARD}" "${MG_MODE}" "${SLOPPY}" "${MG_PRECISION}" "${TOL}" "${MAXITER}"
  printf 'ENV STOUT_NSMEAR=%s STOUT_RHO=%s QUDA_RESOURCE_PATH=%s\n' \
    "${STOUT_NSMEAR}" "${STOUT_RHO}" "${QUDA_RESOURCE_PATH}"
  printf 'ENV SOLVE_REPEATS=%s RUN_CG=%s MG_VERIFY=%s CACHE_STATE=%s\n' \
    "${SOLVE_REPEATS}" "${RUN_CG}" "${MG_VERIFY}" "${cache_state}"
  printf 'ENV NODES=%s NTASKS=%s NTPN=%s GPUS_PER_TASK=%s CPUS_PER_TASK=%s\n' \
    "${NODES}" "${NTASKS}" "${NTPN}" "${GPUS_PER_TASK}" "${CPUS_PER_TASK}"
  printf 'ENV BIN=%s\n' "${BIN}"
  printf 'ENV GRID_SHA=%s\n' "$(git -C "${ROOT}/Grid" rev-parse HEAD)"
  printf 'ENV QUDA_SHA=%s\n' "$(git -C "${ROOT}/quda" rev-parse HEAD)"
  printf 'ENV MODULES=%s\n' "$(module -t list 2>&1 | tr '\n' ',')"
} > "${RUN_DIR}/provenance.env"

args=(
  --grid "${LATT}"
  --mpi "${MPI}"
  --accelerator-threads "${ACCELERATOR_THREADS}"
  --shm "${SHM_MB}"
  --shm-mpi 0
  --comms-overlap
  --probe-action "${ACTION}"
  --probe-checkerboard "${CHECKERBOARD}"
  --probe-mg-mode "${MG_MODE}"
  --probe-sloppy "${SLOPPY}"
  --probe-precise "${PRECISE}"
  --probe-mg-precision "${MG_PRECISION}"
  --probe-mass "${MASS}"
  --probe-csw "${CSW}"
  --probe-tol "${TOL}"
  --probe-maxiter "${MAXITER}"
  --probe-solve-repeats "${SOLVE_REPEATS}"
  --probe-stout-nsmear "${STOUT_NSMEAR}"
  --probe-stout-rho "${STOUT_RHO}"
  --probe-run-cg "${RUN_CG}"
  --probe-run-mg "${RUN_MG}"
  --probe-mg-verify "${MG_VERIFY}"
)
if [[ "${INPUT}" == physical ]]; then
  args+=(--probe-cfg "${CFG}")
fi

srun_cmd=(
  srun
  -N "${NODES}"
  -n "${NTASKS}"
  --ntasks-per-node="${NTPN}"
  --gpus-per-task="${GPUS_PER_TASK}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --cpu-bind=cores
  --gpu-bind=none
  --chdir="${RUN_DIR}"
)

{
  printf '%s\n' '--- provenance ---'
  while IFS= read -r line; do printf '%s\n' "${line}"; done < "${RUN_DIR}/provenance.env"
  printf '%s\n' '--- command ---'
  printf '%q ' "${srun_cmd[@]}" "${SELECT_GPU}" "${BIN}" "${args[@]}"
  printf '\n--- output ---\n'
} > "${LOG}"

set +e
"${srun_cmd[@]}" "${SELECT_GPU}" "${BIN}" "${args[@]}" 2>&1 | tee -a "${LOG}"
status=${PIPESTATUS[0]}
set -e

printf 'probe exit status %s, log %s\n' "${status}" "${LOG}"
exit "${status}"
