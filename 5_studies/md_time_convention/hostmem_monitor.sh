#!/bin/bash
# Host-memory sampler, one task per node: every DT seconds appends
#   epoch  MemTotal  MemAvailable  /dev/shm-used  summed-RSS-of-PATTERN  n-processes   (MiB)
# to <out_dir>/hostmem_<node>.log, until DURATION seconds pass or the allocation ends.
# Written after the 10-06 host OOM of the MDSTEPS 8 scan (job 59387301, nid001536): it tells a
# node that starts short of memory (MemAvailable low before the run) from a run whose RSS grows.
#
# Run it as its own overlapping step with a tiny footprint; the main step must also allow
# sharing (traj48_puregrid.sh: SRUN_EXTRA=--overlap), or one of the two waits for CPUs:
#   SLURM_JOB_ID=<id> srun --jobid=<id> --overlap -N <nodes> --ntasks-per-node=1 \
#     --cpus-per-task=1 --mem=200M bash hostmem_monitor.sh <out_dir> [pattern] [dt_s] [duration_s]
# Defaults: pattern gen_qcd (ps truncates names to 15 characters), dt 2 s, duration 14400 s.

OUT=${1:?usage: hostmem_monitor.sh <out_dir> [pattern] [dt_s] [duration_s]}
PAT=${2:-gen_qcd}
DT=${3:-2}
DUR=${4:-14400}
NODE=${SLURMD_NODENAME:-$(hostname)}
mkdir -p "$OUT"
LOG=$OUT/hostmem_$NODE.log
echo "# epoch MemTotal_MiB MemAvailable_MiB shm_used_MiB rss_${PAT}_MiB nproc_${PAT}" > "$LOG"
END=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$END" ]; do
  mem=$(awk '/^MemTotal:/ {t = $2} /^MemAvailable:/ {a = $2} END {printf "%d %d", t / 1024, a / 1024}' /proc/meminfo)
  shm=$(df -m --output=used /dev/shm | tail -1 | tr -d ' ')
  rss=$(ps -eo rss=,comm= | awk -v p="$PAT" '$2 ~ p {s += $1; n++} END {printf "%d %d", s / 1024, n}')
  echo "$(date +%s) $mem $shm $rss" >> "$LOG"
  sleep "$DT"
done
