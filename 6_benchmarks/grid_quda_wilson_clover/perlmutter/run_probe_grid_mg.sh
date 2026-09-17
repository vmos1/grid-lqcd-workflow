#!/usr/bin/env bash
# Run the Phase 0 Grid-MG probe. Allocation-neutral: invoke inside an existing
# Slurm allocation, exactly like run_benchmark_v2.sh.
#
# The comms/launch configuration below is the SAME v3 config run_benchmark_v2.sh
# carries, deliberately copied rather than re-chosen: the probe's timings are only
# useful as a first look at the MG setup/solve split if the environment matches the
# jobs those numbers will later be compared against. No QUDA variables appear here
# because the probe links no QUDA.
#
# Usage:
#   NODES=1 NTASKS=1 LATT=16.16.16.48 MPI=1.1.1.1 ./run_probe_grid_mg.sh
#   CHECKERBOARD=odd ./run_probe_grid_mg.sh     # expected to ABORT -- see doc §6

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(cd "${HERE}/.." && pwd)
WORKFLOW=$(cd "${BENCH_DIR}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)
source "${WORKFLOW}/machines/perlmutter.sh"

: "${PSCRATCH:?PSCRATCH must be set on Perlmutter}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'ERROR: run_probe_grid_mg.sh must be invoked inside an existing Slurm allocation\n' >&2
  exit 2
fi

GRID_SHA=${GRID_SHA:-$(git -C "${ROOT}/Grid" rev-parse HEAD)}
BIN=${BIN:-${PSCRATCH}/grid_quda_wilson_clover/benchmark/${GRID_SHA}/bin/probe_grid_mg_schur_clover}
OUT=${OUT:-${ROOT}/runs/2026_9_11_grid_mg_probe}

LATT=${LATT:-16.16.16.48}
MPI=${MPI:-1.1.1.1}
MASS=${MASS:--0.2416}
CSW=${CSW:-1.20536588031793}
# wilson | clover. Wilson is doc §4's control -- MG is EXPECTED to lose on it.
ACTION=${ACTION:-clover}
CHECKERBOARD=${CHECKERBOARD:-even}
BLOCK=${BLOCK:-4.4.4.4}
# Coarse stencil range in hops. 1=9 points, 2=33, 4=81. NOT a free knob: Mpc is a
# 2-hop operator, so hops=1 cannot represent its diagonal coarse-block couplings
# and the coarsening aliases silently. See the probe source.
STENCIL_HOPS=${STENCIL_HOPS:-2}
# Schur convention: one=symmetric (QUDA-comparable), mooee=asymmetric (what the CG
# rows use, but QUDA MG cannot coarsen it -- coarse_op.cuh:990).
SCHUR=${SCHUR:-one}
# Coarse-solver Krylov depth. mmax allocates 2*mmax coarse vectors per GCRnStep
# call, so it is a MEMORY knob: the 50 inherited from Grid's small-lattice test
# churned ~200 MB per coarse solve at C3 for a solve that converges in 2 steps.
COARSE_MMAX=${COARSE_MMAX:-8}
COARSE_NSTEP=${COARSE_NSTEP:-8}
# Post-smoother knobs. Same allocation pathology as COARSE_MMAX but on FINE
# vectors (~64 MB each at C3, so mmax=4 churns ~510 MB per application), and the
# smoother is 29-31% of the tuned solve. SMOOTHER_TOL 0.1 is Grid's test value;
# QUDA asks only 0.25, so we may be over-solving.
SMOOTHER_MMAX=${SMOOTHER_MMAX:-4}
SMOOTHER_NSTEP=${SMOOTHER_NSTEP:-4}
SMOOTHER_TOL=${SMOOTHER_TOL:-0.1}
SMOOTHER_MAXITER=${SMOOTHER_MAXITER:-1}
OUTER_MMAX=${OUTER_MMAX:-16}
OUTER_NSTEP=${OUTER_NSTEP:-16}
# double | single. Precision of the MG PRECONDITIONER only; the outer solver and
# every gate stay fp64 either way (fp32 cannot reach the 1e-10 gate). `single`
# is the setting that matches QUDA's own structure.
# ⛔ There is no `half`: Grid has no fp16 arithmetic (vComplexH is a uint16_t
# container, Tensor_traits.h:231). QUDA's half null vectors have no counterpart.
MG_PRECISION=${MG_PRECISION:-double}
# Stage 1 waste-removal switch. 0 = CONTROL (reproduces Grid's PrecGCRNonHermitian and
# blockProject exactly); 1 = cleaned path. The sub-switches default to FAST_MG and exist only
# to isolate one change without a rebuild. See __docs/2026_09_15_grid_mg_fix_plan.md §4.
FAST_MG=${FAST_MG:-0}
FAST_GCR=${FAST_GCR:-${FAST_MG}}
FAST_PROJECT=${FAST_PROJECT:-${FAST_MG}}
PERSISTENT_TEMPS=${PERSISTENT_TEMPS:-${FAST_MG}}
if [[ "${FAST_MG}" == "0" ]]; then VERIFY_RESIDUAL=${VERIFY_RESIDUAL:-1}; else VERIFY_RESIDUAL=${VERIFY_RESIDUAL:-0}; fi
# Stage 2 preconditioner-strength knobs. Defaults reproduce the values inherited from Grid's
# own test; QUDA's counterparts are 0.1 (coarse solver) and 5e-6 (setup).
COARSE_TOL=${COARSE_TOL:-0.2}
SUBSPACE_TOL=${SUBSPACE_TOL:-0.001}
SUBSPACE_ROUNDS=${SUBSPACE_ROUNDS:-3}
SUBSPACE_MMAX=${SUBSPACE_MMAX:-10}
SUBSPACE_MAXITER=${SUBSPACE_MAXITER:-30}
TOL=${TOL:-1e-10}
MAXITER=${MAXITER:-1000}
CG_MAXITER=${CG_MAXITER:-50000}
RUN_CG=${RUN_CG:-1}
INPUT=${INPUT:-hot}
CFG=${CFG:-${ROOT}/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime}

ACCELERATOR_THREADS=${ACCELERATOR_THREADS:-8}
SHM_MB=${SHM_MB:-2048}
# ⛔ Grid's device SOFTWARE CACHE for lattice fields, in MB. Empty = Grid's default,
# which is a large fraction of the card (33.9 GB per rank at C3/16 GPU).
#
# That default OOMs the MG path: GeneralCoarsenedMatrix::Mult goes through
# PaddedCell, whose halo buffers are raw acceleratorAllocDevice calls OUTSIDE the
# managed cache. With the cache holding 33.9 GB, a 226 MB PaddedCell allocation
# failed at C3 -- after the coarsening had completed, on the first coarse operator
# application (PaddedCell.h:183). Capping the cache leaves room for those raw
# allocations. This is an MG-specific need; the CG benchmark never touches
# PaddedCell and does not want it.
DEVICE_MEM_MB=${DEVICE_MEM_MB:-}
# Grid log streams. Add "Performance" to surface GeneralCoarsenedMatrix::Mult's
# internal breakdown (exch / mult / temps / copy) -- the only way to see where the
# coarse operator's time actually goes. Verbose: it prints per application.
# ⛔ Do NOT bypass this script to add flags by hand: launching the binary without
# sourcing machines/perlmutter.sh dies with `libmpfr.so.4` exit 127.
GRID_LOG=${GRID_LOG:-}
NTASKS=${NTASKS:-1}
NODES=${NODES:-1}
NTPN=${NTPN:-${NTASKS}}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
GPUS_PER_TASK=${GPUS_PER_TASK:-1}
RUN_ID=${RUN_ID:-$(date +%Y%m%dT%H%M%S)}
RUN_DIR=${RUN_DIR:-${OUT}/${RUN_ID}}
LOG=${LOG:-${RUN_DIR}/probe.log}

mkdir -p "${RUN_DIR}"

case "${ACTION}" in
  wilson|clover) ;;
  *) printf 'ERROR: ACTION must be wilson or clover\n' >&2; exit 2 ;;
esac
case "${CHECKERBOARD}" in
  even|odd) ;;
  *) printf 'ERROR: CHECKERBOARD must be even or odd\n' >&2; exit 2 ;;
esac
case "${INPUT}" in
  hot|physical) ;;
  *) printf 'ERROR: INPUT must be hot or physical\n' >&2; exit 2 ;;
esac
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

# select_gpu wrapper: 1:1 local-rank -> GPU + NUMA binding (paboyle model).
# NOTE --gpu-bind=none below is required: --gpu-bind=closest breaks CUDA IPC and
# hangs (memory: halo-mpi-rdma-bisection-grid-exonerated).
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
  printf 'Build it with: %s\n' "${HERE}/build_probe_grid_mg.sh" >&2
  exit 1
}

{
  printf 'ENV DATE=%s\n' "$(date --iso-8601=seconds)"
  printf 'ENV HOST=%s SLURM_JOB_ID=%s\n' "$(hostname)" "${SLURM_JOB_ID:-none}"
  printf 'ENV LATT=%s MPI=%s MASS=%s CSW=%s INPUT=%s\n' "${LATT}" "${MPI}" "${MASS}" "${CSW}" "${INPUT}"
  printf 'ENV ACTION=%s SCHUR=%s DEVICE_MEM_MB=%s\n' "${ACTION}" "${SCHUR}" "${DEVICE_MEM_MB:-default}"
  printf 'ENV CHECKERBOARD=%s BLOCK=%s STENCIL_HOPS=%s TOL=%s MAXITER=%s RUN_CG=%s\n' \
    "${CHECKERBOARD}" "${BLOCK}" "${STENCIL_HOPS}" "${TOL}" "${MAXITER}" "${RUN_CG}"
  printf 'ENV COARSE_MMAX=%s COARSE_NSTEP=%s\n' "${COARSE_MMAX}" "${COARSE_NSTEP}"
  printf 'ENV SMOOTHER_MMAX=%s SMOOTHER_NSTEP=%s SMOOTHER_TOL=%s SMOOTHER_MAXITER=%s\n' \
    "${SMOOTHER_MMAX}" "${SMOOTHER_NSTEP}" "${SMOOTHER_TOL}" "${SMOOTHER_MAXITER}"
  printf 'ENV OUTER_MMAX=%s OUTER_NSTEP=%s MG_PRECISION=%s\n' \
    "${OUTER_MMAX}" "${OUTER_NSTEP}" "${MG_PRECISION}"
  printf 'ENV FAST_MG=%s FAST_GCR=%s FAST_PROJECT=%s PERSISTENT_TEMPS=%s VERIFY_RESIDUAL=%s\n' \
    "${FAST_MG}" "${FAST_GCR}" "${FAST_PROJECT}" "${PERSISTENT_TEMPS}" "${VERIFY_RESIDUAL}"
  printf 'ENV COARSE_TOL=%s SUBSPACE_TOL=%s SUBSPACE_ROUNDS=%s SUBSPACE_MMAX=%s SUBSPACE_MAXITER=%s\n' \
    "${COARSE_TOL}" "${SUBSPACE_TOL}" "${SUBSPACE_ROUNDS}" "${SUBSPACE_MMAX}" "${SUBSPACE_MAXITER}"
  printf 'ENV NODES=%s NTASKS=%s NTPN=%s GPUS_PER_TASK=%s CPUS_PER_TASK=%s\n' \
    "${NODES}" "${NTASKS}" "${NTPN}" "${GPUS_PER_TASK}" "${CPUS_PER_TASK}"
  printf 'ENV BIN=%s\n' "${BIN}"
  printf 'ENV GRID_SHA=%s\n' "$(git -C "${ROOT}/Grid" rev-parse HEAD)"
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
  --probe-block "${BLOCK}"
  --probe-stencil-hops "${STENCIL_HOPS}"
  --probe-schur "${SCHUR}"
  --probe-coarse-mmax "${COARSE_MMAX}"
  --probe-coarse-nstep "${COARSE_NSTEP}"
  --probe-smoother-mmax "${SMOOTHER_MMAX}"
  --probe-smoother-nstep "${SMOOTHER_NSTEP}"
  --probe-smoother-tol "${SMOOTHER_TOL}"
  --probe-smoother-maxiter "${SMOOTHER_MAXITER}"
  --probe-outer-mmax "${OUTER_MMAX}"
  --probe-outer-nstep "${OUTER_NSTEP}"
  --probe-mg-precision "${MG_PRECISION}"
  --probe-fast-mg "${FAST_MG}"
  --probe-fast-gcr "${FAST_GCR}"
  --probe-fast-project "${FAST_PROJECT}"
  --probe-persistent-temps "${PERSISTENT_TEMPS}"
  --probe-verify-residual "${VERIFY_RESIDUAL}"
  --probe-coarse-tol "${COARSE_TOL}"
  --probe-subspace-tol "${SUBSPACE_TOL}"
  --probe-subspace-rounds "${SUBSPACE_ROUNDS}"
  --probe-subspace-mmax "${SUBSPACE_MMAX}"
  --probe-subspace-maxiter "${SUBSPACE_MAXITER}"
  --probe-mass "${MASS}"
  --probe-csw "${CSW}"
  --probe-tol "${TOL}"
  --probe-maxiter "${MAXITER}"
  --probe-cg-maxiter "${CG_MAXITER}"
  --probe-run-cg "${RUN_CG}"
)
if [[ -n "${DEVICE_MEM_MB}" ]]; then
  args+=(--device-mem "${DEVICE_MEM_MB}")
fi
if [[ -n "${GRID_LOG}" ]]; then
  args+=(--log "${GRID_LOG}")
fi
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
