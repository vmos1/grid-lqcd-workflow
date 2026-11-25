#!/bin/bash
# =============================================================================================
# chroma_baseg_cpu.sh (lq3) -- Chroma hmc trajectories on lq3 CPU nodes, the CPU twin of
# chroma_baseg_mds.sh (lq2) and ../perlmutter/chroma_baseg_mds.sh: Perlmutter's chain-1 input with
# only the solver machinery changed (make_cpu_input.py; hmc_baseg_mds12_qphixcg_s300_cpu.ini.xml),
# the lq3 CPU Chroma (installs/external/chroma-cpu-lq3: QDP++ + QPhiX AVX-512, gcc 15.3, Grid-CPU's
# OpenMPI 4.1.8), srun --mpi=pmix as every lq3 job. Runs inside the job chroma_baseg_cpu_job.sh
# submits; it never submits, cancels or deletes anything.
#
# Settings (environment at submission; everything else is fixed):
#   INPUT_SRC   the input; default hmc_baseg_mds12_qphixcg_s300_cpu.ini.xml beside this script
#   STEM        output names; default chroma_baseg_mds12_qphixcg_s300_cpu
#   N_TRAJ      trajectories of this run (rewrites <NUpdatesThisRun>); default the input's
#   RUN         run name, default $STEM; run dir runs/<Y_M_D>_<RUN>
#   CFG_DIR     where the start configuration named by <cfg_file> is found, by file name;
#               default <workspace>/inputs (cfg_2000.lime is there)
#   GEOM        QMP/QDP geometry, default '2 2 2 6' (48 ranks: 6 nodes x 8); product = ranks
#   QPHIX_BY, QPHIX_BZ, QPHIX_PXY, QPHIX_PXYZ, QPHIX_MINCT   QPhiX's options (4 4 1 0 1); -c is the
#               threads per rank (--cpus-per-task), -sy -sz 1. Every QPhiX-enabled chroma needs them.
#   CHROMA_PREFIX  default <workspace>/installs/external/chroma-cpu-lq3
#   CHROMA_EXTRA_ARGS appended to hmc's arguments; default none
#   TIMEOUT_MIN guard on hmc in minutes; default the job's time left minus 10
#
# What the job does; any failure before step 4 stops it with nothing run:
#   1. settings and job shape: GEOM's product = ranks; each rank's local volume fits QPhiX (local X
#      a multiple of 8: soalen 4 on the checkerboarded X; local Y, Z multiples of BY, BZ; local T even);
#   2. environment (machines/lq.sh LQ_TARGET=cpu), ldd of hmc on the compute node (no unresolved
#      library, libmpi = Grid-CPU's OpenMPI 4.1.8);
#   3. run dir and a run-local copy of the input in which only <cfg_file> (-> CFG_DIR), <SavePrefix>
#      (-> ./$STEM) and, with N_TRAJ, <NUpdatesThisRun> may differ: the diff is checked and logged;
#   4. hmc. Its output goes to <stem>.log, whose first lines are Perlmutter's (ENV, INPUT=, BIN=,
#      outer/inner n_steps), then INPUT_SRC, INPUT_DIFF, LAUNCH and CMD lines;
#   5. summary: per trajectory update_no, deltaH, AccProb, AcceptP, seconds_for_trajectory, w_plaq,
#      Perlmutter's trajectory 2001 for comparison, solver statistics (QPhiX CG and multishift solves,
#      iterations, seconds, GFLOPS: the numbers to extrapolate a cut-off run from), failure lines.
#
# Output in runs/<Y_M_D>_<RUN>/: <stem>.log, <stem>.out.xml (-o), <stem>.xmldat (-l), the input copy,
# input.diff, plaquette/, and the configuration and restart XML hmc saves when it ends (SavePrefix;
# 6.1 GB at 48^3). Refuses to reuse a run dir.
# =============================================================================================
set -uo pipefail

ROOT=/lustre2/nplqcd/vayyar/grid_qcd
WF=$ROOT/grid-lqcd-workflow
HERE=$WF/5_studies/chroma_inputs/lq
SELF=$HERE/chroma_baseg_cpu.sh

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

[ -n "${SLURM_JOB_ID:-}" ] || die "runs only inside a Slurm job (sbatch $HERE/chroma_baseg_cpu_job.sh); it never submits"

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

# ---- 1. Settings, job shape, local volume ------------------------------------------------------
INPUT_SRC=${INPUT_SRC:-$HERE/hmc_baseg_mds12_qphixcg_s300_cpu.ini.xml}
STEM=${STEM:-chroma_baseg_mds12_qphixcg_s300_cpu}
N_TRAJ=${N_TRAJ:-}
RUN=${RUN:-$STEM}
CFG_DIR=${CFG_DIR:-$ROOT/inputs}
GEOM=${GEOM:-2 2 2 6}
QPHIX_BY=${QPHIX_BY:-4} QPHIX_BZ=${QPHIX_BZ:-4} QPHIX_PXY=${QPHIX_PXY:-1} QPHIX_PXYZ=${QPHIX_PXYZ:-0} QPHIX_MINCT=${QPHIX_MINCT:-1}
CHROMA_PREFIX=${CHROMA_PREFIX:-$ROOT/installs/external/chroma-cpu-lq3}
CHROMA_EXTRA_ARGS=${CHROMA_EXTRA_ARGS:-}
TIMEOUT_MIN=${TIMEOUT_MIN:-}

[ -f "$INPUT_SRC" ] || die "missing input $INPUT_SRC"
[ -z "$N_TRAJ" ] || [[ "$N_TRAJ" =~ ^[1-9][0-9]*$ ]] || die "N_TRAJ must be a positive integer, got '$N_TRAJ'"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN may hold only letters, digits and . _ -, got '$RUN'"
[[ "$STEM" =~ ^[A-Za-z0-9._-]+$ ]] || die "STEM may hold only letters, digits and . _ -, got '$STEM'"
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

read -r -a NROW <<< "$(sed -n 's#.*<nrow>\([^<]*\)</nrow>.*#\1#p' "$INPUT_SRC" | head -1)"
[ "${#NROW[@]}" -eq 4 ] || die "no 4-number <nrow> in $INPUT_SRC"
LOCAL=()
for d in 0 1 2 3; do
  [ $(( NROW[d] % GEOM_A[d] )) -eq 0 ] || die "lattice ${NROW[*]} does not split by GEOM $GEOM in direction $d"
  LOCAL+=( $(( NROW[d] / GEOM_A[d] )) )
done
[ $(( LOCAL[0] % 8 )) -eq 0 ] || die "local X ${LOCAL[0]}: QPhiX (soalen 4 on the checkerboarded X) needs a multiple of 8"
[ $(( LOCAL[1] % QPHIX_BY )) -eq 0 ] || die "local Y ${LOCAL[1]} is not a multiple of QPHIX_BY=$QPHIX_BY"
[ $(( LOCAL[2] % QPHIX_BZ )) -eq 0 ] || die "local Z ${LOCAL[2]} is not a multiple of QPHIX_BZ=$QPHIX_BZ"
[ $(( LOCAL[3] % 2 )) -eq 0 ] || die "local T ${LOCAL[3]} must be even"

if [ -z "$TIMEOUT_MIN" ]; then
  LEFT=$(job_left_min) || die "cannot read the job's time left from squeue; set TIMEOUT_MIN"
  TIMEOUT_MIN=$(( LEFT - 10 ))
fi
[[ "$TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]] || die "TIMEOUT_MIN must be a positive number of minutes, got '$TIMEOUT_MIN' (job too short?)"

# ---- 2. Environment and loader checks -----------------------------------------------------------
LQ_TARGET=cpu source "$WF/machines/lq.sh" || die "sourcing machines/lq.sh (LQ_TARGET=cpu) failed"
set -u
CHROMA_HMC=$CHROMA_PREFIX/bin/hmc
[ -x "$CHROMA_HMC" ] || die "no executable hmc at $CHROMA_HMC"
LDD_OUT=$(ldd "$CHROMA_HMC" 2>&1) || die "ldd failed on $CHROMA_HMC: $LDD_OUT"
if grep -q 'not found' <<< "$LDD_OUT"; then
  grep 'not found' <<< "$LDD_OUT" >&2
  die "ldd reports unresolved libraries or versions for $CHROMA_HMC on $(hostname)"
fi
LIBMPI=$(grep -o -m1 '/[^ ]*libmpi\.so[^ ]*' <<< "$LDD_OUT" || true)
case $LIBMPI in
  "$LQ_CPU_MPI_ROOT"/lib/*) ;;
  *) die "hmc loads libmpi from '$LIBMPI'; expected Grid-CPU's OpenMPI at $LQ_CPU_MPI_ROOT" ;;
esac
BIN_SHA256=$(sha256sum "$CHROMA_HMC" | cut -d' ' -f1)

# ---- 3. Run dir and the run-local input ---------------------------------------------------------
CFG_NAME=$(sed -n 's#.*<cfg_file>\([^<]*\)</cfg_file>.*#\1#p' "$INPUT_SRC" | head -1)
[ -n "$CFG_NAME" ] || die "no <cfg_file> in $INPUT_SRC"
CFG=$CFG_DIR/${CFG_NAME##*/}
[ -f "$CFG" ] || die "start configuration $CFG not found (the input names $CFG_NAME)"
RUN_DIR=$ROOT/runs/$(date +%Y_%-m_%-d)_${RUN}
[ -e "$RUN_DIR" ] && die "$RUN_DIR exists; choose another RUN"
mkdir -p "$RUN_DIR/plaquette" || die "cannot create $RUN_DIR"

INPUT=$RUN_DIR/${INPUT_SRC##*/}
SED_ARGS=(-e "s#<cfg_file>[^<]*</cfg_file>#<cfg_file>${CFG}</cfg_file>#"
          -e "s#<SavePrefix>[^<]*</SavePrefix>#<SavePrefix>./${STEM}</SavePrefix>#")
[ -n "$N_TRAJ" ] && SED_ARGS+=(-e "s#<NUpdatesThisRun>[0-9]*</NUpdatesThisRun>#<NUpdatesThisRun>${N_TRAJ}</NUpdatesThisRun>#")
sed "${SED_ARGS[@]}" "$INPUT_SRC" > "$INPUT" || die "cannot write $INPUT"
diff "$INPUT_SRC" "$INPUT" > "$RUN_DIR/input.diff"
BAD=$(grep -E '^[<>]' "$RUN_DIR/input.diff" | grep -v -E '<(cfg_file|SavePrefix|NUpdatesThisRun)>' || true)
[ -z "$BAD" ] || die "the input copy differs from $INPUT_SRC beyond <cfg_file>, <SavePrefix>, <NUpdatesThisRun>: $BAD"
grep -q -F "<cfg_file>${CFG}</cfg_file>" "$INPUT" || die "<cfg_file> was not rewritten in $INPUT"
NUPD=$(sed -n 's#.*<NUpdatesThisRun>\([0-9]*\)</NUpdatesThisRun>.*#\1#p' "$INPUT" | head -1)
[ -z "$N_TRAJ" ] || [ "$NUPD" = "$N_TRAJ" ] || die "<NUpdatesThisRun> is '$NUPD' in $INPUT, N_TRAJ=$N_TRAJ"
SAVE_PREFIX=$(sed -n 's#.*<SavePrefix>\([^<]*\)</SavePrefix>.*#\1#p' "$INPUT" | head -1)

# ---- 4. Provenance lines, then hmc --------------------------------------------------------------
export OMP_NUM_THREADS=$CPT OMP_PLACES=cores OMP_PROC_BIND=close
LOG=$RUN_DIR/${STEM}.log
OUT_XML=$RUN_DIR/${STEM}.out.xml
XMLDAT=$RUN_DIR/${STEM}.xmldat
QPHIX_ARGS=(-by "$QPHIX_BY" -bz "$QPHIX_BZ" -pxy "$QPHIX_PXY" -pxyz "$QPHIX_PXYZ" -minct "$QPHIX_MINCT" -c "$CPT" -sy 1 -sz 1)
read -r -a EXTRA_WORDS <<< "$CHROMA_EXTRA_ARGS"
HMC_ARGS=(-i "$INPUT" -o "$OUT_XML" -l "$XMLDAT" -geom "${GEOM_A[@]}" "${QPHIX_ARGS[@]}" "${EXTRA_WORDS[@]}")
SRUN=(srun --mpi=pmix --nodes="$NODES" --ntasks="$NTASKS" --ntasks-per-node="$PER_NODE" --cpus-per-task="$CPT"
      --cpu-bind=cores --kill-on-bad-exit=1 --chdir="$RUN_DIR")
{
  # Perlmutter's first lines, same keys
  echo "ENV MDSTEPS=$(grep -m1 -o '<n_steps>[0-9]*</n_steps>' "$INPUT" | tr -dc 0-9) QMP_GEOM='$GEOM' NTASKS=$NTASKS NODES=$NODES CHROMA_EXTRA_ARGS='$CHROMA_EXTRA_ARGS'"
  echo "INPUT=$INPUT"
  echo "BIN=$CHROMA_HMC"
  echo "outer n_steps: $(grep -m1 -o '<n_steps>[0-9]*</n_steps>' "$INPUT")"
  echo "inner n_steps: $(grep -o '<n_steps>[0-9]*</n_steps>' "$INPUT" | sed -n 2p)"
  # lq3's
  echo "INPUT_SRC=$INPUT_SRC"
  sed 's/^/INPUT_DIFF /' "$RUN_DIR/input.diff"
  printf 'LAUNCH host=%s jobid=%s launcher=srun nodes=%s ranks_per_node=%s cpus_per_task=%s nodelist=%s' \
    "$(hostname)" "$SLURM_JOB_ID" "$NODES" "$PER_NODE" "$CPT" "${SLURM_JOB_NODELIST:-?}"
  printf ' bin_sha256=%s libmpi=%s ompi="%s" pml=%s osc=%s ucx_net=%s' \
    "$BIN_SHA256" "$LIBMPI" "$(mpirun --version 2>&1 | head -1)" "${OMPI_MCA_pml:-unset}" "${OMPI_MCA_osc:-unset}" "${UCX_NET_DEVICES:-unset}"
  printf ' local_volume=%s qphix="%s" omp=%s/%s/%s binding=cpu-bind=cores' \
    "$(IFS=.; echo "${LOCAL[*]}")" "${QPHIX_ARGS[*]}" "$OMP_NUM_THREADS" "$OMP_PLACES" "$OMP_PROC_BIND"
  printf ' n_updates=%s save_prefix=%s timeout_min=%s\n' "$NUPD" "$SAVE_PREFIX" "$TIMEOUT_MIN"
  printf 'CMD '
  printf '%q ' timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "$CHROMA_HMC" "${HMC_ARGS[@]}"
  printf '\n'
} > "$LOG"

echo "running hmc $STEM, $NUPD trajectories, on $NODES nodes x $PER_NODE ranks x $CPT threads (geom $GEOM, local ${LOCAL[*]}) -> $LOG"
cd "$RUN_DIR" || die "cannot cd to $RUN_DIR"
timeout -k 60s "${TIMEOUT_MIN}m" "${SRUN[@]}" "$CHROMA_HMC" "${HMC_ARGS[@]}" >> "$LOG" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> "$LOG"

# ---- 5. Summary -----------------------------------------------------------------------------------
echo "==== $RUN: exit=$rc  log: $LOG"
[ "$rc" -eq 124 ] && echo "exit 124 = TIMEOUT_MIN ($TIMEOUT_MIN min) reached: the trajectory in flight is lost"
grep -a -E '^(ENV |INPUT=|BIN=|outer n_steps|inner n_steps|INPUT_DIFF |LAUNCH )' "$LOG"
SRC_XML=$OUT_XML
grep -q -a '<seconds_for_trajectory>' "$OUT_XML" 2>/dev/null || SRC_XML=$XMLDAT
echo "---- per trajectory, from ${SRC_XML##*/} ----"
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
echo "Perlmutter chain 1 (seeds 300..3300, QUDA MG, 16 A100): update 2001  deltaH 0.176973486226416  AccProb 0.837801997520988  AcceptP false  seconds_for_trajectory 1651.885493"
echo "---- w_plaq per measurement (plaquette/thin_plaquette_dat.xml) ----"
grep -a -o '<w_plaq>[^<]*' "$RUN_DIR/plaquette/thin_plaquette_dat.xml" 2>/dev/null | sed 's/<w_plaq>/w_plaq /' || echo "(none)"
echo "---- solver statistics (QPhiX; per-iteration seconds = time / iterations) ----"
awk '
  /QPHIX_CLOVER_CG_SOLVER: [0-9]+ iters/ { n_cg++; for (i = 1; i <= NF; i++) if ($i == "iters,") it_cg += $(i - 1) }
  /QPHIX_CLOVER_CG_SOLVER: Solver Time=/ { split($0, a, "Time="); t_cg += a[2] + 0 }
  /QPHIX_CLOVER_MULTI_SHIFT_CG_MDAGM_SOLVER: Iters=/ { n_ms++; split($0, a, "Iters="); it_ms += a[2] + 0
                                                        split($0, b, "Solver Time="); t_ms += b[2] + 0 }
  END {
    printf "CG (two-flavour monomials): %d solves, %d iterations, %.1f s", n_cg, it_cg, t_cg
    if (it_cg > 0) printf ", %.2f ms per iteration, %.0f iterations per solve", 1000 * t_cg / it_cg, it_cg / n_cg
    printf "\nmultishift CG (strange RHMC): %d solves, %d iterations, %.1f s", n_ms, it_ms, t_ms
    if (it_ms > 0) printf ", %.2f ms per iteration", 1000 * t_ms / it_ms
    printf "\n" }' "$LOG"
grep -a -c 'SOLVE FAILED' "$LOG" | sed 's/^/SOLVE FAILED lines: /'
echo "---- saved configuration / restart files ----"
ls -l "$RUN_DIR" | grep -E 'lime|restart|\.ini\.xml' || echo "(none)"
echo "---- failure lines ----"
grep -a -E 'SOLVE FAILED|ERROR|Error|[Aa]bort|out of memory|[Ss]egmentation|SIGSEGV|[Bb]us error|srun: error' "$LOG" | head -40
