#!/usr/bin/env bash
# Run the shared Grid/QUDA harness, or print a matched stock-QUDA dslash_test command.
# This script is allocation-neutral: invoke it inside an existing Slurm allocation.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(cd "${HERE}/.." && pwd)
WORKFLOW=$(cd "${BENCH_DIR}/../.." && pwd)
ROOT=$(cd "${WORKFLOW}/.." && pwd)
source "${WORKFLOW}/machines/perlmutter.sh"

: "${PSCRATCH:?PSCRATCH must be set on Perlmutter}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'ERROR: run_benchmark.sh must be invoked inside an existing Slurm allocation\n' >&2
  exit 2
fi
MODE=${MODE:-shared}
GRID_SHA=${GRID_SHA:-$(git -C "${ROOT}/Grid" rev-parse HEAD)}
BIN=${BIN:-${PSCRATCH}/grid_quda_wilson_clover/benchmark/${GRID_SHA}/bin/benchmark_grid_quda_wilson_clover}
QUDA_SHA=${QUDA_SHA:-$(git -C "${ROOT}/quda" rev-parse HEAD)}
DTEST=${DTEST:-${PSCRATCH}/grid_quda_wilson_clover/quda-dslash/${QUDA_SHA}/bin/dslash_test}
OUT=${OUT:-${ROOT}/runs/2026_8_25_grid_quda_wilson_clover}
ACTION=${ACTION:-both}
DTEST_OP=${DTEST_OP:-normal_pc}
INPUT=${INPUT:-hot}
CFG=${CFG:-${ROOT}/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime}
# Must match the CFG default above: cl21_48_96 is the production 48^3x96 ensemble
# (confirmed against Grid-TXQCD/production/run_b6p3_48_2node.sh's --grid 48.48.48.96).
LATT=${LATT:-48.48.48.96}
MPI=${MPI:-1.1.1.1}
MASS=${MASS:--0.2416}
CSW=${CSW:-1.20536588031793}
PRECISION=${PRECISION:-strict}
# Independent QUDA knobs. Empty means "take the value implied by PRECISION"
# (strict -> double/no, production -> single/12). Set them explicitly to
# decompose the speedup knob by knob rather than as a bundled preset.
#   SLOPPY_PRECISION:   double | single | half
#   SLOPPY_RECONSTRUCT: no     | 12     | 8
SLOPPY_PRECISION=${SLOPPY_PRECISION:-}
SLOPPY_RECONSTRUCT=${SLOPPY_RECONSTRUCT:-}
# GRID_SOLVER=mixed additionally times Grid's own double/single mixed-precision
# CG, which is the fair counterpart to QUDA's sloppy solve. Grid has no
# counterpart to gauge reconstruction.
GRID_SOLVER=${GRID_SOLVER:-double}
# Grid clover representation: standard (non-compact) | compact. Empty means "let
# the binary default" (standard), so this is backward-compatible. Ignored for
# ACTION=wilson.
CLOVER_IMPL=${CLOVER_IMPL:-}
# Number of right-hand sides for the OPTIONAL multi-RHS pass (Grid's
# CompactWilsonCloverFermion5D, batched, vs QUDA's single-RHS solver run N times).
# Empty or 1 = not passed / disabled, so this is backward-compatible and no
# existing run changes. Requires a clover ACTION.
NRHS=${NRHS:-}
SAMPLES=${SAMPLES:-7}
WARMUPS=${WARMUPS:-2}
REPETITIONS=${REPETITIONS:-20}
SOLVE_REPEATS=${SOLVE_REPEATS:-5}
TOL=${TOL:-1e-10}
MAXITER=${MAXITER:-50000}
ITERATION_TOLERANCE=${ITERATION_TOLERANCE:-0.02}
ACCELERATOR_THREADS=${ACCELERATOR_THREADS:-8}
SHM_MB=${SHM_MB:-2048}
NTASKS=${NTASKS:-1}
NODES=${NODES:-1}
NTPN=${NTPN:-${NTASKS}}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
GPUS_PER_TASK=${GPUS_PER_TASK:-1}
RUN_ID=${RUN_ID:-$(date +%Y%m%dT%H%M%S)}
RUN_DIR=${RUN_DIR:-${OUT}/${RUN_ID}}
RECORDS=${RECORDS:-${RUN_DIR}/records.jsonl}
LOG=${LOG:-${RUN_DIR}/${MODE}.log}
QUDA_RESOURCE_PATH=${QUDA_RESOURCE_PATH:-${RUN_DIR}/quda_resource}
export QUDA_RESOURCE_PATH

mkdir -p "${RUN_DIR}" "${QUDA_RESOURCE_PATH}"

# --- v2 CORRECTED COMMS CONFIG (per Grid dev P. Boyle, systems/Perlmutter/dwf4.slurm) ---
# RDMA + GPU-IPC + QUDA-P2P ENABLED; no OMP affinity pinning; select_gpu wrapper
# does device selection + NUMA binding (replaces --enable-setdevice reliance).
export SLURM_CPU_BIND=${SLURM_CPU_BIND:-cores}
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export MPICH_RDMA_ENABLED_CUDA=${MPICH_RDMA_ENABLED_CUDA:-1}
export MPICH_GPU_IPC_ENABLED=${MPICH_GPU_IPC_ENABLED:-1}
export MPICH_GPU_EAGER_REGISTER_HOST_MEM=${MPICH_GPU_EAGER_REGISTER_HOST_MEM:-0}
export MPICH_GPU_NO_ASYNC_MEMCPY=${MPICH_GPU_NO_ASYNC_MEMCPY:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export QUDA_ENABLE_MPS=${QUDA_ENABLE_MPS:-1}
export QUDA_ENABLE_P2P=${QUDA_ENABLE_P2P:-1}

# select_gpu wrapper: 1:1 local-rank -> GPU + numactl NUMA binding (paboyle model)
SELECT_GPU=${RUN_DIR}/select_gpu
cat > "${SELECT_GPU}" <<'EOF'
#!/bin/bash
export GPU=$SLURM_LOCALID
export NUMA=$SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=$GPU
exec numactl -m $NUMA -N $NUMA "$@"
EOF
chmod +x "${SELECT_GPU}"

case "${PRECISION}" in
  strict|production) ;;
  *) printf 'ERROR: PRECISION must be strict or production\n' >&2; exit 2 ;;
esac
case "${ACTION}" in
  wilson|clover|both) ;;
  *) printf 'ERROR: ACTION must be wilson, clover, or both\n' >&2; exit 2 ;;
esac
case "${DTEST_OP}" in
  dslash|mat|normal_pc) ;;
  *) printf 'ERROR: DTEST_OP must be dslash, mat, or normal_pc\n' >&2; exit 2 ;;
esac
case "${INPUT}" in
  hot|physical) ;;
  *) printf 'ERROR: INPUT must be hot or physical\n' >&2; exit 2 ;;
esac
if [[ "${INPUT}" == physical && ! -f "${CFG}" ]]; then
  printf 'ERROR: physical gauge configuration not found: %s\n' "${CFG}" >&2
  exit 1
fi

cache_state=empty
if find "${QUDA_RESOURCE_PATH}" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .; then
  cache_state=seeded
fi

write_provenance() {
  local target=$1
  {
    printf 'ENV DATE=%s\n' "$(date --iso-8601=seconds)"
    printf 'ENV HOST=%s SLURM_JOB_ID=%s\n' "$(hostname)" "${SLURM_JOB_ID:-none}"
    printf 'ENV MODE=%s ACTION=%s DTEST_OP=%s INPUT=%s PRECISION=%s\n' \
      "${MODE}" "${ACTION}" "${DTEST_OP}" "${INPUT}" "${PRECISION}"
    printf 'ENV LATT=%s MPI=%s MASS=%s CSW=%s\n' "${LATT}" "${MPI}" "${MASS}" "${CSW}"
    printf 'ENV SAMPLES=%s WARMUPS=%s REPETITIONS=%s SOLVE_REPEATS=%s TOL=%s MAXITER=%s\n' \
      "${SAMPLES}" "${WARMUPS}" "${REPETITIONS}" "${SOLVE_REPEATS}" "${TOL}" "${MAXITER}"
    printf 'ENV ITERATION_TOLERANCE=%s\n' "${ITERATION_TOLERANCE}"
    printf 'ENV SLOPPY_PRECISION=%s SLOPPY_RECONSTRUCT=%s GRID_SOLVER=%s\n' \
      "${SLOPPY_PRECISION:-from-preset}" "${SLOPPY_RECONSTRUCT:-from-preset}" "${GRID_SOLVER}"
    printf 'ENV CLOVER_IMPL=%s NRHS=%s\n' "${CLOVER_IMPL:-default-standard}" "${NRHS:-disabled}"
    printf 'ENV NODES=%s NTASKS=%s NTPN=%s GPUS_PER_TASK=%s CPUS_PER_TASK=%s\n' \
      "${NODES}" "${NTASKS}" "${NTPN}" "${GPUS_PER_TASK}" "${CPUS_PER_TASK}"
    printf 'ENV BIN=%s DTEST=%s CFG=%s\n' "${BIN}" "${DTEST}" "${CFG}"
    printf 'ENV RECORDS=%s QUDA_RESOURCE_PATH=%s CACHE_STATE=%s\n' \
      "${RECORDS}" "${QUDA_RESOURCE_PATH}" "${cache_state}"
    printf 'ENV GRID_SHA=%s QUDA_SHA=%s\n' \
      "$(git -C "${ROOT}/Grid" rev-parse HEAD)" "$(git -C "${ROOT}/quda" rev-parse HEAD)"
    printf 'ENV MODULES=%s\n' "$(module -t list 2>&1 | tr '\n' ',')"
  } > "${target}"
}

write_provenance "${RUN_DIR}/provenance.env"

common_srun=(
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

run_shared() {
  [[ -x "${BIN}" ]] || {
    printf 'ERROR: benchmark binary not found: %s\n' "${BIN}" >&2
    exit 1
  }
  local args=(
    --grid "${LATT}"
    --mpi "${MPI}"
    --accelerator-threads "${ACCELERATOR_THREADS}"
    --shm "${SHM_MB}"
    --shm-mpi 0
    --comms-overlap
    --benchmark-action "${ACTION}"
    --benchmark-input "${INPUT}"
    --benchmark-mass "${MASS}"
    --benchmark-csw "${CSW}"
    --benchmark-precision "${PRECISION}"
    --benchmark-samples "${SAMPLES}"
    --benchmark-warmups "${WARMUPS}"
    --benchmark-repetitions "${REPETITIONS}"
    --benchmark-solve-repeats "${SOLVE_REPEATS}"
    --benchmark-tol "${TOL}"
    --benchmark-maxiter "${MAXITER}"
    --benchmark-iteration-tolerance "${ITERATION_TOLERANCE}"
    --benchmark-grid-solver "${GRID_SOLVER}"
    --benchmark-output "${RECORDS}"
    --benchmark-cache-state "${cache_state}"
  )
  if [[ "${INPUT}" == physical ]]; then
    args+=(--benchmark-cfg "${CFG}")
  fi
  if [[ -n "${SLOPPY_PRECISION}" ]]; then
    args+=(--benchmark-sloppy-precision "${SLOPPY_PRECISION}")
  fi
  if [[ -n "${SLOPPY_RECONSTRUCT}" ]]; then
    args+=(--benchmark-sloppy-reconstruct "${SLOPPY_RECONSTRUCT}")
  fi
  if [[ -n "${CLOVER_IMPL}" ]]; then
    args+=(--benchmark-clover-impl "${CLOVER_IMPL}")
  fi
  if [[ -n "${NRHS}" ]]; then
    args+=(--benchmark-nrhs "${NRHS}")
  fi
  {
    printf '%s\n' '--- provenance ---'
    while IFS= read -r line; do printf '%s\n' "${line}"; done < "${RUN_DIR}/provenance.env"
    printf '%s\n' '--- command ---'
    printf '%q ' "${common_srun[@]}" "${SELECT_GPU}" "${BIN}" "${args[@]}"
    printf '\n--- output ---\n'
  } > "${LOG}"
  "${common_srun[@]}" "${SELECT_GPU}" "${BIN}" "${args[@]}" 2>&1 | tee -a "${LOG}"
}

run_dslash() {
  [[ -x "${DTEST}" ]] || {
    printf 'ERROR: stock QUDA dslash_test not found: %s\n' "${DTEST}" >&2
    printf 'Build the stock source against the completed QUDA test/install libraries; see the Build section of %s.\n' \
      "${BENCH_DIR}/2026_8_25_README.md"
    exit 1
  }
  if [[ "${INPUT}" != hot ]]; then
    printf 'ERROR: stock dslash_test companion supports its own generated gauge, not INPUT=physical\n' >&2
    exit 2
  fi
  if [[ "${ACTION}" == both ]]; then
    printf 'ERROR: MODE=dslash runs one stock-QUDA action at a time; set ACTION=wilson or ACTION=clover\n' >&2
    exit 2
  fi

  local dslash_type
  case "${ACTION}" in
    wilson) dslash_type=wilson ;;
    clover) dslash_type=clover ;;
  esac

  local dtest_type
  case "${DTEST_OP}" in
    dslash) dtest_type=Dslash ;;
    mat) dtest_type=Mat ;;
    normal_pc) dtest_type=MatPCDagMatPC ;;
  esac

  local quda_prec
  local quda_recon
  case "${PRECISION}" in
    strict)
      quda_prec=double
      quda_recon=18
      ;;
    production)
      quda_prec=single
      quda_recon=12
      ;;
  esac

  IFS=. read -r gx gy gz gt <<< "${LATT}"
  IFS=. read -r px py pz pt <<< "${MPI}"
  for extent in "${gx}" "${gy}" "${gz}" "${gt}" "${px}" "${py}" "${pz}" "${pt}"; do
    if [[ ! "${extent}" =~ ^[1-9][0-9]*$ ]]; then
      printf 'ERROR: LATT and MPI must each contain four positive dot-separated integers\n' >&2
      exit 2
    fi
  done
  if (( gx % px != 0 || gy % py != 0 || gz % pz != 0 || gt % pt != 0 )); then
    printf 'ERROR: each LATT extent must be divisible by its MPI extent\n' >&2
    exit 2
  fi
  if (( px * py * pz * pt != NTASKS )); then
    printf 'ERROR: MPI process-grid volume must equal NTASKS\n' >&2
    exit 2
  fi

  local lx=$((gx / px))
  local ly=$((gy / py))
  local lz=$((gz / pz))
  local lt=$((gt / pt))
  local args=(
    --dslash-type "${dslash_type}"
    --mass "${MASS}"
    --prec "${quda_prec}"
    --recon "${quda_recon}"
    --test "${dtest_type}"
    --xdim "${lx}"
    --ydim "${ly}"
    --zdim "${lz}"
    --tdim "${lt}"
    --gridsize "${px}" "${py}" "${pz}" "${pt}"
    --niter "${REPETITIONS}"
    --gtest_filter=DslashTest.benchmark:DslashTest.verify
  )
  if [[ "${ACTION}" == clover ]]; then
    args+=(--clover-csw "${CSW}")
    args+=(--compute-clover true)
  fi
  # Match the shared harness's action-dependent matpc enum. QUDA's
  # DiracWilsonPC::M rejects the asymmetric enums outright (only even-even and
  # odd-odd are valid for a scalar diagonal block), so Wilson must use plain
  # odd-odd while clover uses odd-odd-asym. Passed for every DTEST_OP so the
  # Dslash/Mat cases construct the same preconditioned Dirac object as the
  # normal-operator case; dslash_test's own default is even-even.
  local dtest_matpc
  case "${ACTION}" in
    wilson) dtest_matpc=odd-odd ;;
    clover) dtest_matpc=odd-odd-asym ;;
  esac
  args+=(--matpc "${dtest_matpc}")

  {
    printf '%s\n' '--- provenance ---'
    while IFS= read -r line; do printf '%s\n' "${line}"; done < "${RUN_DIR}/provenance.env"
    printf '%s\n' '--- timing scope ---'
    printf 'resident_device_quda; generated hot gauge; --niter=%s; one internal warm-up\n' "${REPETITIONS}"
    printf '%s\n' '--- command ---'
    printf '%q ' "${common_srun[@]}" "${DTEST}" "${args[@]}"
    printf '\n--- output ---\n'
  } > "${LOG}"
  "${common_srun[@]}" "${DTEST}" "${args[@]}" 2>&1 | tee -a "${LOG}"
}

case "${MODE}" in
  shared) run_shared ;;
  dslash) run_dslash ;;
  *) printf 'ERROR: MODE must be shared or dslash\n' >&2; exit 2 ;;
esac
