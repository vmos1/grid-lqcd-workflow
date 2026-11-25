#!/bin/bash
#SBATCH --job-name=chroma_baseg
#SBATCH --partition=lq2_gpu
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/chroma_baseg_lq_%j.log
# =============================================================================================
# chroma_baseg_mds.sh (lq) -- Chroma hmc trajectories on lq2, the lq twin of
# ../perlmutter/chroma_baseg_mds.sh: the same inputs (../candidates/), the same decomposition
# (-geom 1 2 2 4 on 4 nodes x 4 A100, one rank per GPU), the same hmc command line and output
# names, so lq's timings compare with Perlmutter's. Runs inside the job it is submitted as; it
# never submits, cancels or deletes anything. Every submission is approved by the user, e.g.:
#
#   MDS=12 SUFFIX=_mgall20_s300 N_TRAJ=1 sbatch --time=02:30:00 --export=ALL \
#     /lustre2/nplqcd/vayyar/grid_qcd/grid-lqcd-workflow/5_studies/chroma_inputs/lq/chroma_baseg_mds.sh
#
# Settings (environment at submission; everything else is fixed):
#   MDS, SUFFIX        select ../candidates/hmc_baseg_mds${MDS}${SUFFIX}.ini.xml (its README maps
#                      each setup to MDS and SUFFIX)
#   N_TRAJ             trajectories of this run (rewrites <NUpdatesThisRun>); default the input's
#   RUN                run name, default chroma_baseg_mds${MDS}${SUFFIX}; run dir runs/<Y_M_D>_<RUN>
#   CFG_DIR            where the start configuration named by the input's <cfg_file> is found,
#                      by file name; default <workspace>/inputs (cfg_2000.lime is there)
#   CHROMA_EXTRA_ARGS  appended to hmc's arguments; default '-poolsize 20g' as Perlmutter's chains
#                      since 10-07 (L193: without it QDP-JIT takes 92% of the GPU and QUDA pages).
#                      Set it to the empty string to drop it.
#   SRC_TUNE           directory holding a QUDA tunecache.tsv of an earlier lq run of this binary,
#                      copied into this run's QUDA_RESOURCE_PATH; default none (cold). Perlmutter
#                      seeded every timed run: time only with a seeded cache (L083, L084).
#   CHROMA_STACK       group (default): the group's install on NVHPC 23.11's Open MPI 3.1.5 (TCP between
#                      nodes); gpu-lq2: ours, installs/external/chroma-gpu-lq2 on machines/lq.sh's gompi
#                      stack (Open MPI 4.1.5 + UCX, as our Grid). env_chroma_lq.sh; the ldd gate below
#                      requires each stack's own libmpi.so.40 and, for gpu-lq2, nothing from NVHPC
#   CHROMA_IBV         system (default) or bundled: inter-node transport of the group stack, env_chroma_lq.sh
#                      (gpu-lq2: na)
#   PMIX_PSEC          auto (default): start ranks as Slurm sets them up and, only if the probe
#                      then fails on PMIx security, again with PMIX_SECURITY_MODE=native;
#                      default: never override; native: always override
#   MIN_NET_GBS        stop before hmc if the probe's inter-node bandwidth (GB/s, 4 MiB messages)
#                      is below this; default 0 = report only
#   GEOM               default '1 2 2 4' (Perlmutter's); the product must equal the job's ranks
#   TIMEOUT_MIN        guard on hmc in minutes; default the job's time left minus 10
#   GPU_MON_MS         nvidia-smi memory sampling period on local rank 0 of each node, default
#                      2000; 0 = off
#
# What the job does; any failure before step 5 stops it with nothing run:
#   1. settings and job shape: one rank per GPU, GEOM's product = ranks;
#   2. environment (env_chroma_lq.sh, CHROMA_STACK), ldd of hmc on the compute node (no unresolved
#      library; libmpi = the stack's: NVHPC 23.11's Open MPI 3.1.5 for group, gompi's Open MPI 4.1.5
#      for gpu-lq2, which may load nothing from NVHPC's comm_libs; libcuda.so.1 = the driver, never a
#      toolkit stub), libdevice for QDP-JIT;
#   3. run dir and a run-local copy of the input in which only <cfg_file> (-> CFG_DIR), an
#      absolute <SavePrefix> (-> the run dir) and, with N_TRAJ, <NUpdatesThisRun> may differ:
#      the diff is checked and logged;
#   4. binding and the MPI gate. Every rank keeps all GPUs visible (QDP-JIT picks device = local
#      rank mod GPU count, and QUDA's intra-node peer-to-peer, on in Perlmutter's runs, needs the
#      peers visible), and numactl holds it to its GPU's NUMA domain (srun --cpu-bind=none, as
#      machines/lq.md). mpi_probe.c, compiled here with the MPI hmc loads and started by hmc's
#      srun line, checks: every rank starts (PMIx), block placement, each rank's QDP-JIT device
#      = its local id, all GPUs visible, and measures the inter-node bandwidth (group: openib or
#      TCP; gpu-lq2: UCX, which should give InfiniBand-class bandwidth; MIN_NET_GBS=6 makes a
#      TCP fallback (2.9 GB/s, job 1365190) stop the job before hmc);
#   5. hmc. Its output is appended to <stem>.log, whose first lines are Perlmutter's (ENV, INPUT=,
#      BIN=, outer/inner n_steps), then lq's INPUT_DIFF, LAUNCH, PROBE, NET and CMD lines;
#   6. summary: per trajectory update_no, deltaH, AccProb, AcceptP, seconds_for_trajectory (the
#      time per trajectory the Perlmutter docs quote), w_plaq, transport, GPU memory high-water,
#      failure lines, and the line to submit a warm-cache rerun.
#
# Output in runs/<Y_M_D>_<RUN>/ (stem = chroma_baseg_mds${MDS}${SUFFIX}, Perlmutter's names):
# <stem>.log, <stem>.out.xml (-o), <stem>.xmldat (-l), the input copy and input.diff,
# quda_resource/ (this run's QUDA tune cache, the next run's SRC_TUNE), plaquette/, eigen_mdagm/,
# bind_numa, gpu_mon, gpu_mem.*.csv, mpi_probe*, and the configuration and restart XML hmc saves
# when it ends (SavePrefix; 6.1 GB at 48^3). Refuses to reuse a run dir. QUDA's abort in teardown
# at exit is cosmetic once the last trajectory is written (Perlmutter); judge by the summary.
# =============================================================================================
set -uo pipefail

ROOT=/lustre2/nplqcd/vayyar/grid_qcd
WF=$ROOT/grid-lqcd-workflow
HERE=$WF/5_studies/chroma_inputs/lq
SELF=$HERE/chroma_baseg_mds.sh

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

[ -n "${SLURM_JOB_ID:-}" ] || die "runs only inside a Slurm job (sbatch ... $SELF); it never submits"

# Minutes left in this job, from squeue's %L ([D-]HH:MM:SS or MM:SS).
job_left_min() {
  local l p s=0
  l=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null) || return 1
  case $l in *-*) s=$(( ${l%%-*} * 86400 )); l=${l#*-} ;; esac
  IFS=: read -r -a p <<< "$l"
  case ${#p[@]} in
    3) s=$(( s + 10#${p[0]} * 3600 + 10#${p[1]} * 60 + 10#${p[2]} )) ;;
    2) s=$(( s + 10#${p[0]} * 60 + 10#${p[1]} )) ;;
    *) return 1 ;;
  esac
  echo $(( s / 60 ))
}

# ---- 1. Settings and job shape ------------------------------------------------------------------
MDS=${MDS:-}
[[ "$MDS" =~ ^[0-9]+$ ]] || die "set MDS (12, 10 or 8) and SUFFIX, as in ../candidates/README.md; got MDS='$MDS'"
SUFFIX=${SUFFIX:-}
STEM=chroma_baseg_mds${MDS}${SUFFIX}
N_TRAJ=${N_TRAJ:-}
RUN=${RUN:-$STEM}
CFG_DIR=${CFG_DIR:-$ROOT/inputs}
CHROMA_EXTRA_ARGS=${CHROMA_EXTRA_ARGS--poolsize 20g}
SRC_TUNE=${SRC_TUNE:-}
PMIX_PSEC=${PMIX_PSEC:-auto}
MIN_NET_GBS=${MIN_NET_GBS:-0}
GEOM=${GEOM:-1 2 2 4}
TIMEOUT_MIN=${TIMEOUT_MIN:-}
GPU_MON_MS=${GPU_MON_MS:-2000}

[ -z "$N_TRAJ" ] || [[ "$N_TRAJ" =~ ^[1-9][0-9]*$ ]] || die "N_TRAJ must be a positive integer, got '$N_TRAJ'"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN may hold only letters, digits and . _ -, got '$RUN'"
case $PMIX_PSEC in auto|default|native) ;; *) die "PMIX_PSEC must be auto, default or native, got '$PMIX_PSEC'" ;; esac
[[ "$MIN_NET_GBS" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MIN_NET_GBS must be a number (GB/s), got '$MIN_NET_GBS'"
[[ "$GPU_MON_MS" =~ ^[0-9]+$ ]] || die "GPU_MON_MS must be a non-negative integer, got '$GPU_MON_MS'"
[ -z "$SRC_TUNE" ] || [ -f "$SRC_TUNE/tunecache.tsv" ] || die "SRC_TUNE=$SRC_TUNE holds no tunecache.tsv"
read -r -a GEOM_A <<< "$GEOM"
[ "${#GEOM_A[@]}" -eq 4 ] || die "GEOM must be four integers, got '$GEOM'"
for d in "${GEOM_A[@]}"; do [[ "$d" =~ ^[1-9][0-9]*$ ]] || die "GEOM must be four positive integers, got '$GEOM'"; done

NODES=${SLURM_JOB_NUM_NODES:?SLURM_JOB_NUM_NODES unset}
NTASKS=${SLURM_NTASKS:?SLURM_NTASKS unset (give the job --ntasks-per-node)}
CPT=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK unset (give the job --cpus-per-task)}
PER_NODE=$(( NTASKS / NODES ))
[ $(( PER_NODE * NODES )) -eq "$NTASKS" ] || die "$NTASKS tasks do not split evenly over $NODES nodes"
[ $(( GEOM_A[0] * GEOM_A[1] * GEOM_A[2] * GEOM_A[3] )) -eq "$NTASKS" ] ||
  die "GEOM='$GEOM' needs $(( GEOM_A[0] * GEOM_A[1] * GEOM_A[2] * GEOM_A[3] )) ranks, the job has $NTASKS"
if [ -z "$TIMEOUT_MIN" ]; then
  LEFT=$(job_left_min) || die "cannot read the job's time left from squeue; set TIMEOUT_MIN"
  TIMEOUT_MIN=$(( LEFT - 10 ))
fi
[[ "$TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]] || die "TIMEOUT_MIN must be a positive number of minutes, got '$TIMEOUT_MIN' (job too short?)"

# ---- 2. Environment and loader checks -----------------------------------------------------------
# shellcheck source=env_chroma_lq.sh
source "$HERE/env_chroma_lq.sh" || die "sourcing $HERE/env_chroma_lq.sh failed"
set -u
LDD_OUT=$(ldd "$CHROMA_HMC" 2>&1) || die "ldd failed on $CHROMA_HMC: $LDD_OUT"
if grep -q 'not found' <<< "$LDD_OUT"; then
  grep 'not found' <<< "$LDD_OUT" >&2
  die "ldd reports unresolved libraries or versions for $CHROMA_HMC on $(hostname)"
fi
LIBMPI=$(grep -o -m1 '/[^ ]*libmpi\.so[^ ]*' <<< "$LDD_OUT" || true)
[ "$LIBMPI" = "$CHROMA_LIBMPI_WANT" ] ||
  die "hmc loads libmpi from '$LIBMPI'; CHROMA_STACK=$CHROMA_STACK expects $CHROMA_LIBMPI_WANT"
if [ "$CHROMA_STACK" = gpu-lq2 ] && grep -E 'nvhpc|comm_libs|openmpi-3\.1\.5' <<< "$LDD_OUT" >&2; then
  die "CHROMA_STACK=gpu-lq2: hmc loads the libraries above from NVHPC / Open MPI 3.1.5"
fi
LIBCUDA=$(awk '$1 == "libcuda.so.1" { print $3 }' <<< "$LDD_OUT")
case $LIBCUDA in
  */stubs/*|"${CUDA_PATH:-/nonexistent}"/*) die "hmc loads libcuda.so.1 from $LIBCUDA, a CUDA toolkit stub, not the driver" ;;
  /*) ;;
  *) die "hmc: libcuda.so.1 does not resolve on $(hostname) ('$LIBCUDA')" ;;
esac
[ -f "$CUDA_PATH/nvvm/libdevice/libdevice.10.bc" ] || die "QDP-JIT needs $CUDA_PATH/nvvm/libdevice/libdevice.10.bc"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: not a GPU node ($(hostname))"
command -v numactl >/dev/null 2>&1 || die "numactl not found on $(hostname)"
NGPU=$(nvidia-smi -L | grep -c '^GPU ')
[ "$NGPU" -eq "$PER_NODE" ] || die "one rank per GPU: $PER_NODE ranks per node, $NGPU GPUs on $(hostname)"
BIN_SHA256=$(sha256sum "$CHROMA_HMC" | cut -d' ' -f1)

# ---- 3. Run dir and the run-local input ---------------------------------------------------------
SRC_INPUT=$CHROMA_INPUTS/hmc_baseg_mds${MDS}${SUFFIX}.ini.xml
[ -f "$SRC_INPUT" ] || die "missing input $SRC_INPUT (see $CHROMA_INPUTS/README.md)"
CFG_NAME=$(sed -n 's#.*<cfg_file>\([^<]*\)</cfg_file>.*#\1#p' "$SRC_INPUT" | head -1)
[ -n "$CFG_NAME" ] || die "no <cfg_file> in $SRC_INPUT"
CFG=$CFG_DIR/${CFG_NAME##*/}
[ -f "$CFG" ] || die "start configuration $CFG not found (the input names $CFG_NAME)"
RUN_DIR=$ROOT/runs/$(date +%Y_%-m_%-d)_${RUN}
[ -e "$RUN_DIR" ] && die "$RUN_DIR exists; choose another RUN"
mkdir -p "$RUN_DIR/plaquette" "$RUN_DIR/eigen_mdagm" "$RUN_DIR/quda_resource" "$CUDA_CACHE_PATH" ||
  die "cannot create $RUN_DIR or $CUDA_CACHE_PATH"

INPUT=$RUN_DIR/${SRC_INPUT##*/}
SED_ARGS=(-e "s#<cfg_file>[^<]*</cfg_file>#<cfg_file>${CFG}</cfg_file>#"
          -e 's#<SavePrefix>/[^<]*/\([^/<]*\)</SavePrefix>#<SavePrefix>./\1</SavePrefix>#')
[ -n "$N_TRAJ" ] && SED_ARGS+=(-e "s#<NUpdatesThisRun>[0-9]*</NUpdatesThisRun>#<NUpdatesThisRun>${N_TRAJ}</NUpdatesThisRun>#")
sed "${SED_ARGS[@]}" "$SRC_INPUT" > "$INPUT" || die "cannot write $INPUT"
diff "$SRC_INPUT" "$INPUT" > "$RUN_DIR/input.diff"
BAD=$(grep -E '^[<>]' "$RUN_DIR/input.diff" | grep -v -E '<(cfg_file|SavePrefix|NUpdatesThisRun)>' || true)
[ -z "$BAD" ] || die "the input copy differs from $SRC_INPUT beyond <cfg_file>, <SavePrefix>, <NUpdatesThisRun>: $BAD"
grep -q -F "<cfg_file>${CFG}</cfg_file>" "$INPUT" || die "<cfg_file> was not rewritten in $INPUT"
NUPD=$(sed -n 's#.*<NUpdatesThisRun>\([0-9]*\)</NUpdatesThisRun>.*#\1#p' "$INPUT" | head -1)
[ -z "$N_TRAJ" ] || [ "$NUPD" = "$N_TRAJ" ] || die "<NUpdatesThisRun> is '$NUPD' in $INPUT, N_TRAJ=$N_TRAJ"
SAVE_PREFIX=$(sed -n 's#.*<SavePrefix>\([^<]*\)</SavePrefix>.*#\1#p' "$INPUT" | head -1)
case $SAVE_PREFIX in /*) die "<SavePrefix> $SAVE_PREFIX is still absolute in $INPUT" ;; esac

export QUDA_RESOURCE_PATH=$RUN_DIR/quda_resource
TUNE_SEEDED=no
if [ -n "$SRC_TUNE" ]; then
  cp "$SRC_TUNE/tunecache.tsv" "$QUDA_RESOURCE_PATH/" || die "cannot copy $SRC_TUNE/tunecache.tsv"
  TUNE_SEEDED=yes
fi
export OMP_NUM_THREADS=$CPT

# ---- 4. Binding and the MPI gate ----------------------------------------------------------------
# GPU table of this node (lq2 nodes are identical): nvidia-smi index (= PCI bus order, and CUDA's
# order under CUDA_DEVICE_ORDER=PCI_BUS_ID), bus id, NUMA domain from sysfs.
GPUS=() BUSES=() NUMAS=()
while IFS=, read -r idx bus; do
  idx=${idx// /}
  bus=$(tr 'A-F' 'a-f' <<< "${bus// /}")
  numa=$(cat "/sys/bus/pci/devices/${bus#0000}/numa_node") || die "no NUMA node in sysfs for GPU $idx ($bus)"
  [[ "$numa" =~ ^[0-9]+$ ]] || die "GPU $idx ($bus) has NUMA node '$numa'"
  GPUS+=("$idx") BUSES+=("$bus") NUMAS+=("$numa")
done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)
[ "${#GPUS[@]}" -eq "$PER_NODE" ] || die "GPU table has ${#GPUS[@]} entries, want $PER_NODE"

BIND=$RUN_DIR/bind_numa
write_bind() {   # $1: PMIx security mode forced in every rank, or empty
  {
    printf '%s\n' '#!/bin/bash' \
      '# srun local rank L -> NUMA domain of GPU L; every GPU stays visible (QDP-JIT takes device' \
      '# local rank mod count; QUDA peer-to-peer needs the peers). Written by chroma_baseg_mds.sh.' \
      "RANK_NUMA=(${NUMAS[*]})" \
      'L=${SLURM_LOCALID:?bind_numa: not started by srun}' \
      'export CUDA_DEVICE_ORDER=PCI_BUS_ID'
    [ -n "$1" ] && printf 'export PMIX_SECURITY_MODE=%s PMIX_MCA_psec=%s\n' "$1" "$1"
    printf '%s\n' 'exec numactl -N "${RANK_NUMA[$L]}" -m "${RANK_NUMA[$L]}" "$@"'
  } > "$BIND"
  chmod +x "$BIND"
}
WRAP=("$BIND")
if [ "$GPU_MON_MS" -gt 0 ]; then
  GPU_MON=$RUN_DIR/gpu_mon
  {
    printf '%s\n' '#!/bin/bash' '# nvidia-smi memory sampler on local rank 0 of each node (written by chroma_baseg_mds.sh)' \
      'if [ "${SLURM_LOCALID:-}" = 0 ]; then'
    printf '  timeout -k 10s %sm nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.total --format=csv,noheader,nounits -lms %s > "%s/gpu_mem.$(hostname -s).csv" 2>&1 < /dev/null &\n' \
      "$(( TIMEOUT_MIN + 5 ))" "$GPU_MON_MS" "$RUN_DIR"
    printf '%s\n' 'fi' 'exec "$@"'
  } > "$GPU_MON"
  chmod +x "$GPU_MON"
  WRAP=("$GPU_MON" "$BIND")
fi
SRUN=(srun --mpi=pmix --nodes="$NODES" --ntasks="$NTASKS" --ntasks-per-node="$PER_NODE" --cpus-per-task="$CPT"
      --gpus-per-node="$PER_NODE" --cpu-bind=none --kill-on-bad-exit=1 --chdir="$RUN_DIR")

OMPI_CC=gcc mpicc -std=gnu11 -O1 -o "$RUN_DIR/mpi_probe" "$HERE/mpi_probe.c" > "$RUN_DIR/mpi_probe.build" 2>&1 ||
  { cat "$RUN_DIR/mpi_probe.build" >&2; die "mpi_probe.c did not build with $(command -v mpicc)"; }
run_probe() {   # $1: forced PMIx security mode or empty; $2: output file
  write_bind "$1"
  timeout -k 10s 5m "${SRUN[@]}" --label "$BIND" "$RUN_DIR/mpi_probe" > "$2" 2>&1
}
PSEC_USED="" PROBE_OUT=$RUN_DIR/mpi_probe.out
case $PMIX_PSEC in
  default) run_probe "" "$PROBE_OUT" && PSEC_USED=slurm ;;
  native)  run_probe native "$PROBE_OUT" && PSEC_USED=native ;;
  auto)
    if run_probe "" "$PROBE_OUT"; then
      PSEC_USED=slurm
    elif grep -q -a -i -E 'psec|munge|pmix_init|PMIx_Init' "$PROBE_OUT"; then
      warn "MPI probe failed on PMIx security ($PROBE_OUT); trying again with PMIX_SECURITY_MODE=native"
      PROBE_OUT=$RUN_DIR/mpi_probe.native.out
      run_probe native "$PROBE_OUT" && PSEC_USED=native
    fi ;;
esac
if [ -z "$PSEC_USED" ]; then
  tail -40 "$PROBE_OUT" >&2
  die "MPI probe failed: srun could not start the ranks ($PROBE_OUT). hmc was not run (run dir $RUN_DIR)."
fi
write_bind "$([ "$PSEC_USED" = native ] && echo native)"   # hmc gets the wrapper that worked

# Placement checks, from one PROBE line per rank: block placement (local id = rank mod ranks per
# node), QDP-JIT's device (first set of PMI_RANK, OMPI_COMM_WORLD_LOCAL_RANK, SLURM_LOCALID, mod
# the GPU count) = local id = the GPU whose NUMA domain bind_numa chose, all GPUs visible.
CHECK=$(awk -v n="$NTASKS" -v per="$PER_NODE" '
  { for (i = 1; i <= NF; i++) if ($i == "PROBE" && $(i + 1) == "rank") break
    if (i > NF) next
    split($(i + 2), rr, "/"); r = rr[1] + 0; seen++
    pmi = $(i + 6); ompi = $(i + 8); lid = $(i + 10); cvd = $(i + 12)
    dev = (pmi != "unset") ? pmi : ((ompi != "unset") ? ompi : lid)
    if (lid + 0 != r % per) printf "rank %d: SLURM_LOCALID %s, block placement gives %d; ", r, lid, r % per
    if ((dev + 0) % per != lid + 0) printf "rank %d: QDP-JIT device %d, but bound to the NUMA domain of GPU %s; ", r, (dev + 0) % per, lid
    if (cvd != "unset" && split(cvd, g, ",") != per) printf "rank %d: CUDA_VISIBLE_DEVICES=%s, want all %d GPUs; ", r, cvd, per }
  END { if (seen != n) printf "%d of %d ranks reported; ", seen, n }' "$PROBE_OUT")
[ -z "$CHECK" ] || die "MPI probe placement: $CHECK($PROBE_OUT)"
SUMMARY=$(grep -a -m1 -o 'PROBE_SUMMARY.*' "$PROBE_OUT")
[ -n "$SUMMARY" ] || die "no PROBE_SUMMARY line in $PROBE_OUT"
INTER_GBS=$(awk '{ for (i = 1; i <= NF; i++) if ($i == "inter_GBs") print $(i + 1) }' <<< "$SUMMARY")
OPENIB_OFF=$(grep -a -c 'unable to find any relevant network interfaces' "$PROBE_OUT" || true)
NET_LINE="NET probe=${PROBE_OUT##*/} chroma_stack=$CHROMA_STACK pmix_psec=$PSEC_USED chroma_ibv=$CHROMA_IBV rdmav_drivers=${RDMAV_DRIVERS:-unset} openib_no_device_warnings=$OPENIB_OFF ${SUMMARY#PROBE_SUMMARY }"
if awk -v a="$INTER_GBS" -v m="$MIN_NET_GBS" 'BEGIN { exit !(m > 0 && (a == "na" || a + 0 < m)) }'; then
  die "inter-node bandwidth ${INTER_GBS} GB/s is below MIN_NET_GBS=$MIN_NET_GBS. hmc was not run. $NET_LINE"
fi
[ "$OPENIB_OFF" -eq 0 ] || warn "openib found no usable HCA: inter-node messages go over TCP. $NET_LINE"

# ---- 5. Provenance lines, then hmc ------------------------------------------------------------
LOG=$RUN_DIR/${STEM}.log
OUT_XML=$RUN_DIR/${STEM}.out.xml
XMLDAT=$RUN_DIR/${STEM}.xmldat
read -r -a EXTRA_WORDS <<< "$CHROMA_EXTRA_ARGS"
HMC_ARGS=(-i "$INPUT" -o "$OUT_XML" -l "$XMLDAT" -geom "${GEOM_A[@]}" "${EXTRA_WORDS[@]}")
{
  # Perlmutter's first lines, same keys
  echo "ENV MDSTEPS=$MDS QMP_GEOM='$GEOM' NTASKS=$NTASKS NODES=$NODES CHROMA_EXTRA_ARGS='$CHROMA_EXTRA_ARGS'"
  echo "INPUT=$INPUT"
  echo "BIN=$CHROMA_HMC"
  echo "outer n_steps: $(grep -m1 -o '<n_steps>[0-9]*</n_steps>' "$INPUT")"
  echo "inner n_steps: $(grep -o '<n_steps>[0-9]*</n_steps>' "$INPUT" | sed -n 2p)"
  # lq's
  echo "INPUT_SRC=$SRC_INPUT"
  sed 's/^/INPUT_DIFF /' "$RUN_DIR/input.diff"
  printf 'LAUNCH host=%s jobid=%s launcher=srun nodes=%s ranks_per_node=%s cpus_per_task=%s nodelist=%s' \
    "$(hostname)" "$SLURM_JOB_ID" "$NODES" "$PER_NODE" "$CPT" "${SLURM_JOB_NODELIST:-?}"
  printf ' chroma_stack=%s bin_sha256=%s libmpi=%s libcuda=%s ompi="%s" pmix_psec=%s chroma_ibv=%s rdmav_drivers=%s' \
    "$CHROMA_STACK" "$BIN_SHA256" "$LIBMPI" "$LIBCUDA" "$(mpirun --version 2>&1 | head -1)" "$PSEC_USED" "$CHROMA_IBV" \
    "${RDMAV_DRIVERS:-unset}"
  printf ' mca="%s" ucx="%s"' \
    "$(compgen -e | grep '^OMPI_MCA_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
    "$(compgen -e | grep '^UCX_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)"
  printf ' modules="%s" cuda_path=%s gpus=%s buses=%s gpu_numa=%s gpu="%s"' \
    "$(module -t list 2>&1 | tr '\n' ' ')" "$CUDA_PATH" "$(IFS=,; echo "${GPUS[*]}")" "$(IFS=,; echo "${BUSES[*]}")" \
    "$(IFS=,; echo "${NUMAS[*]}")" "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader -i "${GPUS[0]}")"
  printf ' binding="numactl -N/-m = NUMA of GPU <local rank>, all GPUs visible, CUDA_DEVICE_ORDER=PCI_BUS_ID" omp=%s' "$OMP_NUM_THREADS"
  printf ' quda_env="%s" quda_resource=%s tune_seeded=%s src_tune=%s cuda_cache=%s' \
    "$(compgen -e | grep '^QUDA_' | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)" \
    "$QUDA_RESOURCE_PATH" "$TUNE_SEEDED" "${SRC_TUNE:-none}" "$CUDA_CACHE_PATH"
  printf ' n_updates=%s save_prefix=%s timeout_min=%s\n' "$NUPD" "$SAVE_PREFIX" "$TIMEOUT_MIN"
  grep -a -o 'PROBE rank.*' "$PROBE_OUT"
  echo "$NET_LINE"
  printf 'CMD '
  printf '%q ' timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "${WRAP[@]}" "$CHROMA_HMC" "${HMC_ARGS[@]}"
  printf '\n'
} > "$LOG"

echo "running hmc $STEM, $NUPD trajectories, on $NODES nodes x $PER_NODE ranks (geom $GEOM) -> $LOG"
cd "$RUN_DIR" || die "cannot cd to $RUN_DIR"
timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "${WRAP[@]}" "$CHROMA_HMC" "${HMC_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 6. Summary -----------------------------------------------------------------------------------
echo "==== $RUN: exit=$rc  log: $LOG"
[ "$rc" -eq 124 ] && echo "exit 124 = TIMEOUT_MIN ($TIMEOUT_MIN min) reached: the trajectory in flight is lost"
grep -a -E '^(ENV |INPUT=|BIN=|outer n_steps|inner n_steps|INPUT_DIFF |LAUNCH |NET )' "$LOG"
SRC_XML=$OUT_XML
grep -q -a '<seconds_for_trajectory>' "$OUT_XML" 2>/dev/null || SRC_XML=$XMLDAT
echo "---- per trajectory, from ${SRC_XML##*/} (seconds_for_trajectory = the time per trajectory Perlmutter quotes) ----"
if [ -f "$SRC_XML" ]; then
  grep -a -o -E '<(update_no|deltaH|AccProb|AcceptP|seconds_for_trajectory|seconds_for_measurements)>[^<]*' "$SRC_XML" |
    awk -F'>' '
      function pr() { printf "update %s  deltaH %s  AccProb %s  AcceptP %s  seconds_for_trajectory %s  seconds_for_measurements %s\n", u, d, p, a, s, m }
      { k = substr($1, 2); v = $2
        if (k == "update_no") { if (u != "") pr(); u = v; d = p = a = s = m = "-"; next }
        if (k == "deltaH") d = v; else if (k == "AccProb") p = v; else if (k == "AcceptP") a = v
        else if (k == "seconds_for_trajectory") s = v; else if (k == "seconds_for_measurements") m = v }
      END { if (u != "") pr() }'
else
  echo "(neither $OUT_XML nor $XMLDAT exists)"
fi
echo "---- w_plaq per measurement (plaquette/thin_plaquette_dat.xml) ----"
grep -a -o '<w_plaq>[^<]*' "$RUN_DIR/plaquette/thin_plaquette_dat.xml" 2>/dev/null | sed 's/<w_plaq>/w_plaq /' || echo "(none)"
echo "---- GPU and transport lines from hmc ----"
echo "QUDA peer-to-peer lines: $(grep -a -c 'Peer-to-peer enabled' "$LOG" || true); openib no-device warnings: $(grep -a -c 'unable to find any relevant network interfaces' "$LOG" || true)"
grep -a -m4 -E 'GPU-Direct|Using device|CUDA device|pool' "$LOG"
# Standing rule for cross-code timing (2026_08_21_grid_vs_chroma_timing_breakdown.md section 6):
# record whether QUDA's tune cache was loaded; a cold cache inflates every trajectory ~3.8x.
grep -a -m3 -E 'sets of cached parameters|Cache file not found|does not match current QUDA version' "$LOG" ||
  echo "(no QUDA tune cache line in the log)"
if compgen -G "$RUN_DIR/gpu_mem.*.csv" > /dev/null; then
  echo "---- nvidia-smi high-water, MiB ----"
  awk -F', *' '$4 ~ /^[0-9]+$/ { f = FILENAME; sub(/.*\//, "", f); k = f " gpu" $2
                if (!(k in m) || $4 + 0 > m[k]) m[k] = $4 + 0; t[k] = $5 }
               END { for (k in m) printf "%s peak_used=%d total=%s\n", k, m[k], t[k] }' \
    "$RUN_DIR"/gpu_mem.*.csv | sort | tee "$RUN_DIR/gpu_mem_peak.txt"
fi
echo "---- saved configuration / restart files ----"
ls -l "$RUN_DIR" | grep -E 'lime|restart|\.ini\.xml' || echo "(none)"
echo "---- failure lines (QUDA teardown aborts after the last trajectory are cosmetic) ----"
grep -a -E 'QUDA_ERROR|ERROR|Error|[Aa]bort|out of memory|cudaMalloc|[Ss]egmentation|SIGSEGV|[Bb]us error|FAILED|srun: error' "$LOG" | head -40
echo "---- warm-cache rerun (time it; seeded from this run's QUDA tune cache) ----"
echo "CHROMA_STACK=$CHROMA_STACK SRC_TUNE=$QUDA_RESOURCE_PATH MDS=$MDS SUFFIX=$SUFFIX N_TRAJ=${N_TRAJ:-$NUPD} RUN=${RUN}_warm sbatch --time=<t> --export=ALL $SELF"
