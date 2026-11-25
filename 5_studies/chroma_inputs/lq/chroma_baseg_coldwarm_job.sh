#!/bin/bash
#SBATCH --job-name=chroma_coldwarm
#SBATCH --partition=lq2_gpu
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/chroma_coldwarm_lq_%j.log
# =============================================================================================
# chroma_baseg_coldwarm_job.sh (lq) -- two chroma_baseg_mds.sh runs in one lq2 allocation: first a
# cold-cache run (QUDA tunes from nothing; untimed), then the warm rerun seeded from the first run's
# QUDA tune cache (SRC_TUNE), the one to time. One allocation instead of two saves an lq2 queue wait
# (job 1365190 waited 16.5 h for 4 nodes). Every submission is approved by the user, e.g.:
#
#   CHROMA_STACK=gpu-lq2 MDS=12 SUFFIX=_mgall20_s300 N_TRAJ=1 MIN_NET_GBS=6 \
#   RUN=chroma_baseg_mds12_mgall20_s300_t1_gpulq2 sbatch --time=04:00:00 --export=ALL \
#     /lustre2/nplqcd/vayyar/grid_qcd/grid-lqcd-workflow/5_studies/chroma_inputs/lq/chroma_baseg_coldwarm_job.sh
#
# Settings: every chroma_baseg_mds.sh setting (CHROMA_STACK, MDS, SUFFIX, N_TRAJ, MIN_NET_GBS, ...)
# applies to both runs, except:
#   RUN               base name; the runs are RUN_cold and RUN_warm (run dirs runs/<Y_M_D>_RUN_cold, _warm).
#                     Default chroma_baseg_mds${MDS}${SUFFIX}
#   COLD_TIMEOUT_MIN  guard on the cold hmc, minutes (default 150); the warm run gets the job's time left
#                     minus 10 min (the launcher's default)
#   SRC_TUNE          ignored: the cold run never gets one, the warm run gets the cold run's
#   TIMEOUT_MIN       ignored (see COLD_TIMEOUT_MIN)
# The warm run starts only if the cold one exited 0 and left a quda_resource/tunecache.tsv.
# Each run's summary (deltaH, seconds_for_trajectory, NET line, ...) is in this job's log.
# =============================================================================================
set -uo pipefail
ROOT=/lustre2/nplqcd/vayyar/grid_qcd
LAUNCHER=$ROOT/grid-lqcd-workflow/5_studies/chroma_inputs/lq/chroma_baseg_mds.sh
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -n "${SLURM_JOB_ID:-}" ] || die "runs only inside a Slurm job (sbatch ... $0); it never submits"
[ -n "${MDS:-}" ] || die "set MDS and SUFFIX as for chroma_baseg_mds.sh"
BASE=${RUN:-chroma_baseg_mds${MDS}${SUFFIX:-}}
COLD_TIMEOUT_MIN=${COLD_TIMEOUT_MIN:-150}
[[ "$COLD_TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]] || die "COLD_TIMEOUT_MIN must be a positive number of minutes"
unset TIMEOUT_MIN

echo "=== job $SLURM_JOB_ID on $(hostname), $(date): cold then warm, base run name $BASE, CHROMA_STACK=${CHROMA_STACK:-group}"
D0=$(date +%Y_%-m_%-d)
RUN=${BASE}_cold SRC_TUNE= TIMEOUT_MIN=$COLD_TIMEOUT_MIN bash "$LAUNCHER"
rc=$?
D1=$(date +%Y_%-m_%-d)
COLD_DIR=""
for d in "$D0" "$D1"; do
  [ -d "$ROOT/runs/${d}_${BASE}_cold" ] && COLD_DIR=$ROOT/runs/${d}_${BASE}_cold && break
done
echo "=== cold run: exit $rc, run dir ${COLD_DIR:-not found} ($(date))"
[ "$rc" -eq 0 ] || die "the cold run failed (exit $rc); no warm run"
[ -n "$COLD_DIR" ] && [ -f "$COLD_DIR/quda_resource/tunecache.tsv" ] ||
  die "no tunecache.tsv under ${COLD_DIR:-runs/*_${BASE}_cold}/quda_resource; no warm run"

RUN=${BASE}_warm SRC_TUNE=$COLD_DIR/quda_resource bash "$LAUNCHER"
rc=$?
echo "=== warm run: exit $rc ($(date))"
exit "$rc"
