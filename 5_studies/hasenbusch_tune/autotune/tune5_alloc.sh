#!/usr/bin/env bash
# Interactive 40 GB GPU window for the tune 5 session, named tune5_window_fo or tune5_window_screen
# so the session can tell it from every other allocation and run only the matching launcher on it.
# Same fixed shape as alloc_gpu.sh: a kind and two integers, nothing else. Gives up if Slurm has not
# granted it within 5 min (--immediate), so a busy queue never leaves a pending request behind.
# Installed in ~/.claude/bin/ (write-denied to agents).
# Usage: bash tune5_alloc.sh <fo|screen> <nodes 1-4> <hours 1-4>     Release: bash tune5_release.sh <jobid>
set -euo pipefail
[[ $# -eq 3 && $1 =~ ^(fo|screen)$ && $2 =~ ^[1-4]$ && $3 =~ ^[1-4]$ ]] || { echo "usage: tune5_alloc.sh <fo|screen> <nodes 1-4> <hours 1-4>" >&2; exit 2; }
set +e
salloc --no-shell --immediate=300 --job-name="tune5_window_$1" --nodes="$2" --constraint='gpu&hbm40g' \
  --gpus-per-node=4 --qos=interactive --time="$3:00:00" --account=m4599_g
rc=$?
set -e
echo "--- salloc exit=$rc; this user's tune5 jobs ---"
squeue -h -u "$USER" -o '%i %j %q %T %M %l %D %N' | grep -E '^[0-9]+ tune5_' || true
exit "$rc"
