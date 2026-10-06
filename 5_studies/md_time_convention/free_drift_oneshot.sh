#!/bin/bash
# One command for the whole free-drift test, for the user to run from the workspace root as
#   ! bash grid-lqcd-workflow/5_studies/md_time_convention/free_drift_oneshot.sh
# Allocates one interactive GPU node for 1 h through the sanctioned wrapper, runs
# run_free_drift.sh on it (about 5-10 min), releases the allocation (also on Ctrl-C or error),
# and prints the summary. Log: runs/2026_10_5_free_drift/run.log.

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
OUT=${OUT:-$(cd "$HERE/../../.." && pwd)/runs/2026_10_5_free_drift}
mkdir -p "$OUT"

ALLOC=$(bash ~/.claude/bin/alloc_gpu.sh 1 1 2>&1)
echo "$ALLOC"
JOB=$(echo "$ALLOC" | grep -o 'Granted job allocation [0-9]*' | grep -o '[0-9]*$')
[ -n "$JOB" ] || { echo "ERROR: no allocation granted; check squeue -u $USER" >&2; exit 1; }
trap 'bash ~/.claude/bin/release_alloc.sh "$JOB"' EXIT

echo "running free drift on allocation $JOB (log $OUT/run.log)"
SLURM_JOB_ID=$JOB OUT=$OUT bash "$HERE/run_free_drift.sh" > "$OUT/run.log" 2>&1
sed -n '/=== summary/,$p' "$OUT/run.log"
