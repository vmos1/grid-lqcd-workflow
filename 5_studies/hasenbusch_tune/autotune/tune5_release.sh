#!/usr/bin/env bash
# Release ONE tune 5 interactive window: numeric id, owned by $USER, named tune5_window_*, on an
# interactive QOS. Never touches any other allocation. Installed in ~/.claude/bin/.
# Usage: bash tune5_release.sh <jobid>
set -uo pipefail
[[ $# -eq 1 && $1 =~ ^[0-9]+$ ]] || { echo "usage: tune5_release.sh <jobid>" >&2; exit 2; }
info=$(squeue -h -j "$1" -o '%u %j %q' 2>/dev/null)
rc=$?
# A Slurm controller stall makes squeue fail: say so, never "not in the queue" (it may still be charging).
[[ $rc -eq 0 ]] || { echo "squeue failed (exit $rc): cannot check job $1; retry, the window may still be running" >&2; exit 3; }
[[ -n $info ]] || { echo "job $1 is not in the queue (already ended)"; exit 0; }
read -r owner name qos <<< "$info"
[[ $owner == "$USER" ]] || { echo "job $1 belongs to $owner" >&2; exit 1; }
[[ $name == tune5_window_fo || $name == tune5_window_screen ]] || { echo "job $1 is '$name', not a tune5 window" >&2; exit 1; }
[[ $qos == *interactive* ]] || { echo "job $1 has qos $qos, not interactive" >&2; exit 1; }
echo "releasing $1 ($name, $qos)"
scancel "$1"
