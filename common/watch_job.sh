#!/bin/bash
# Watch one Slurm job and exit once, when it needs attention: the job reached a final state,
# or it is RUNNING but its log has not grown for STALL_MIN minutes (a hang). Prints nothing
# while waiting; on exit prints the reason, the sacct line and the log tail.
#
# For agents: start it in the background (Claude Code: Bash run_in_background), which costs no
# tokens while it waits and notifies the agent when it exits. It dies with the session. Only
# when the user asks for a job to be watched (__docs/agent/preferences.md, jobs and scheduler).
#
# Usage: watch_job.sh JOBID [LOG] [STALL_MIN=30] [POLL_S=120] [MAX_H=26]
#   LOG        the job's own log; needed for the stall check and the tail (optional)
#   STALL_MIN  minutes without log growth while RUNNING that count as a stall (0 = off)
#   POLL_S     seconds between sacct queries (keep >= 60: slurmdbd is shared)
#   MAX_H      give up after this many hours (the watcher exits and says so)
set -uo pipefail

JOB=${1:?usage: watch_job.sh JOBID [LOG] [STALL_MIN] [POLL_S] [MAX_H]}
LOG=${2:-}
STALL_MIN=${3:-30}
POLL_S=${4:-120}
MAX_H=${5:-26}
START=$(date +%s)

report() {
  echo "=== watch_job ${JOB}: $1 ($(date '+%Y-%m-%d %H:%M %Z'))"
  sacct -X -j "${JOB}" -o JobID,JobName%30,Partition,State,ExitCode,Elapsed,NodeList 2>&1
  if [[ -n "${LOG}" && -f "${LOG}" ]]; then
    echo "--- last 15 lines of ${LOG}"
    tail -n 15 "${LOG}"
  fi
}

while true; do
  # sacct, not squeue: squeue forgets a job soon after it ends. Empty output (job just
  # submitted, or a transient sacct failure) counts as still waiting.
  state=$(sacct -X -n -P -j "${JOB}" -o State 2>/dev/null | head -1 | awk '{print $1}')
  case "${state}" in
    ""|PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|CONFIGURING|COMPLETING) ;;
    *) report "job ended: ${state}"; exit 0 ;;
  esac

  if [[ "${state}" == RUNNING && -n "${LOG}" && -f "${LOG}" && "${STALL_MIN}" -gt 0 ]]; then
    idle_min=$(( ( $(date +%s) - $(stat -c %Y "${LOG}") ) / 60 ))
    if (( idle_min >= STALL_MIN )); then
      report "RUNNING but log unchanged for ${idle_min} min: possible hang"
      exit 0
    fi
  fi

  if (( $(date +%s) - START >= MAX_H * 3600 )); then
    report "watcher gave up after ${MAX_H} h, job still ${state:-unknown}"
    exit 0
  fi
  sleep "${POLL_S}"
done
