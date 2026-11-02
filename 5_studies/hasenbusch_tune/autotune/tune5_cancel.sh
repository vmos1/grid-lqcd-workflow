#!/usr/bin/env bash
# Cancel ONE of the tune 5 session's own batch jobs: numeric id, owned by $USER, named tune5_*, not
# an interactive window (those go through tune5_release.sh). Installed in ~/.claude/bin/.
# Usage: bash tune5_cancel.sh <jobid>
set -uo pipefail
[[ $# -eq 1 && $1 =~ ^[0-9]+$ ]] || { echo "usage: tune5_cancel.sh <jobid>" >&2; exit 2; }
info=$(squeue -h -j "$1" -o '%u %j %q' 2>/dev/null)
rc=$?
[[ $rc -eq 0 ]] || { echo "squeue failed (exit $rc): cannot check job $1; retry, the job may still be running" >&2; exit 3; }
[[ -n $info ]] || { echo "job $1 is not in the queue (already ended)"; exit 0; }
read -r owner name qos <<< "$info"
[[ $owner == "$USER" ]] || { echo "job $1 belongs to $owner" >&2; exit 1; }
[[ $name == tune5_* && $name != tune5_window* ]] || { echo "job $1 is '$name', not a tune5 batch job" >&2; exit 1; }
[[ $qos != *interactive* ]] || { echo "job $1 is interactive: use tune5_release.sh" >&2; exit 1; }
echo "cancelling $1 ($name)"
scancel "$1"
