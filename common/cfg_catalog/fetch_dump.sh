#!/bin/bash
# fetch_dump.sh — save the raw MetaCat records of one ensemble and artifact kind.
#
#   fetch_dump.sh <ensemble> <out.json> [kind]     kind: gauge_config (default), hmc_input
#
# The output has one JSON record per line (size, stream, trajectory, tape path,
# full action metadata). The other scripts in this folder read it. Anonymous
# read; only the query text leaves the machine.

set -euo pipefail

ENS=$1
OUT=$2
KIND=${3:-gauge_config}
URL=https://metacat.fnal.gov:9443/amsc_meta_prod/app/data/query

curl -sf --max-time 300 -G -o "$OUT" \
    --data-urlencode 'with_meta=yes' \
    --data-urlencode "query=files from lqcd:$ENS where lqcd.artifact.kind = \"$KIND\"" \
    "$URL"
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
