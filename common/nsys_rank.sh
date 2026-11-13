#!/bin/bash
# =============================================================================
# nsys_rank.sh -- run a command under Nsight Systems on chosen MPI ranks only, unchanged on
# every other rank. Put it between the GPU/NUMA binding wrapper and the binary (the
# LAUNCH_PREFIX hook of traj48_puregrid.sh and chroma_baseg_mds.sh), so the profiled process
# already has its device and binding; profiling the binding wrapper would trace numactl.
#
#   srun ... select_gpu nsys_rank.sh <binary> <args>
#
# Environment:
#   NSYS_OUT       required. Report path prefix; "_r<rank>" is appended. Must be on Lustre
#                  ($PSCRATCH), never CFS: there nsys writes the .qdstrm but the import to
#                  .nsys-rep fails with errno 524 (run_benchmark_v2.sh, 2026-09). A stranded
#                  .qdstrm imports with host-linux-x64/QdstrmImporter --input-file X.qdstrm.
#   NSYS_RANKS     comma list of SLURM_PROCIDs to profile (default 0), or "all"
#   NSYS_TRACE     default cuda,nvtx. Add mpi (nsys's own PMPI interposer, --mpi-impl=mpich) ONLY
#                  with NSYS_RANKS=all; any other combination is refused on every rank.
#                  2026-11-12, allocation 61175263: rank 0 alone with mpi tracing hung Chroma at
#                  start-up for the whole 3 h. Rank 0's report shows MPI_Init_thread done at 2.3 s,
#                  a communicator derived from MPI_COMM_WORLD at 2.399 s, and an MPI_Barrier on it
#                  16 us later that never returned: a collective only the profiled rank entered.
#   NSYS_DELAY     seconds from launch before collection starts (default 0)
#   NSYS_DURATION  seconds of collection (default: until the program exits)
#                  With a window, nsys writes the report when the window closes and then exits;
#                  the program keeps running as long as the other ranks hold the step open (with
#                  NSYS_RANKS=all the step ends when the last report is written).
#                  Sizes measured 2026-11-12, 48^3 on 16 GPUs, rank 0, cuda,nvtx:
#                    Grid   180 s window: 250 MB report, written 60 s after the window closed.
#                    Grid  1200 s window: 1.3 GB report (15 M kernels, 65 M events), written
#                          about 6 min after the window closed, and then the rank was OOM-killed
#                          (host memory, step 61186102.9), which ended the run. The report was
#                          complete. For a run that must survive, keep the window to ~600 s.
#                    Chroma 600 s window: 4.0 GB report; the run finished normally.
#                  In the sqlite export every event time counts from the start of collection
#                  (window = 0..duration); the program launch is the "Process was launched"
#                  row of DIAGNOSTIC_EVENT, at about -NSYS_DELAY.
#   NSYS           nsys binary (default the HPC SDK 25.5 one, as run_benchmark_v2.sh)
# CPU sampling and context-switch tracing stay off (they need perf_event permissions).
# --kill=none: the end of collection never signals the program, which runs to its normal end.
# =============================================================================
NSYS=${NSYS:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Systems/bin/nsys}
RANK=${SLURM_PROCID:-0}
RANKS=${NSYS_RANKS:-0}
TRACE=${NSYS_TRACE:-cuda,nvtx}
# Checked on every rank, before the unprofiled ones exec, so the whole step fails at once.
case ",$TRACE," in
  *",mpi,"*)
    if [ "$RANKS" != all ]; then
      echo "nsys_rank.sh: NSYS_TRACE=$TRACE with NSYS_RANKS=$RANKS: MPI tracing needs NSYS_RANKS=all (it deadlocks otherwise, see the header)" >&2
      exit 2
    fi ;;
esac
if [ "$RANKS" != all ]; then
  case ",$RANKS," in
    *",$RANK,"*) ;;
    *) exec "$@" ;;
  esac
fi
: "${NSYS_OUT:?nsys_rank.sh: set NSYS_OUT (a path prefix on PSCRATCH)}"
case "$NSYS_OUT" in
  /global/cfs/*) echo "nsys_rank.sh: NSYS_OUT is on CFS; the report import fails there, use PSCRATCH" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$NSYS_OUT")"
# nsys stages the raw .qdstrm in TMPDIR, which is /tmp (memory) on compute nodes: a long trace
# goes next to the report instead. The profiled program inherits this TMPDIR.
export TMPDIR=$(dirname "$NSYS_OUT")/tmp_r$RANK
mkdir -p "$TMPDIR"
OPTS=(--trace="$TRACE" --sample=none --cpuctxsw=none --kill=none
      --force-overwrite=false --output="${NSYS_OUT}_r${RANK}")
case ",$TRACE," in *",mpi,"*) OPTS+=(--mpi-impl=mpich) ;; esac
[ -n "${NSYS_DELAY:-}" ] && OPTS+=(--delay="$NSYS_DELAY")
[ -n "${NSYS_DURATION:-}" ] && OPTS+=(--duration="$NSYS_DURATION")
echo "nsys_rank.sh: rank $RANK profiled: $NSYS profile ${OPTS[*]} $*" >&2
exec "$NSYS" profile "${OPTS[@]}" "$@"
