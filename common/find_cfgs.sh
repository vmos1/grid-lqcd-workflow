#!/bin/bash
# find_cfgs.sh — look up gauge configurations in the USQCD/NPLQCD MetaCat catalog.
#
# The catalog (AmSC Fermi Data Platform, metacat.fnal.gov) holds one record per
# stored file: HPSS path, size, stream, trajectory and the ensemble's physics.
# Reads are anonymous; only the query text leaves the machine.
#
#   find_cfgs.sh                                   one line per lqcd ensemble
#   find_cfgs.sh <ensemble>                        streams of that ensemble
#   find_cfgs.sh <ensemble> <stream> [tmin] [tmax] files, with HPSS paths
#
# Example (our 48^3 x 96 config is stream djm-3, trajectory 2000):
#   find_cfgs.sh cl21_48_96_b6p3_m0p2416_m0p2050 djm-3 2000 2050
#
# Set KIND to list a different artifact kind (default gauge_config; also
# hmc_input, hmc_log_archive). Files are on NERSC HPSS (tape): pull them with
# `hsi get` onto $PSCRATCH before use; see machines/perlmutter.md.

set -euo pipefail

URL=https://metacat.fnal.gov:9443/amsc_meta_prod/app/data
KIND=${KIND:-gauge_config}
ENS=${1:-}
STREAM=${2:-}
TMIN=${3:-}
TMAX=${4:-}

PY_LIST='
import json, sys
rows = [d for d in json.load(sys.stdin) if d["namespace"] == "lqcd"]
print("%-50s %-17s %5s %8s %8s %7s %5s" % ("ensemble", "volume", "beta", "m_l", "m_s", "configs", "strms"))
for d in sorted(rows, key=lambda d: d["name"]):
    m = d["metadata"]
    if "lqcd.parent.ensemble" in m or "lqcd.volume" not in m:
        continue
    vol = "x".join(str(v) for v in m["lqcd.volume"])
    print("%-50s %-17s %5s %8s %8s %7s %5s" % (d["name"], vol, m.get("lqcd.beta"),
          m.get("lqcd.mass.light"), m.get("lqcd.mass.strange"),
          m.get("lqcd.n_configs", "?"), m.get("lqcd.n_streams", "?")))
'

# The query endpoint returns one JSON record per line, separated by characters
# that only str.splitlines() treats as line breaks.
PY_FILES='
import json, sys, collections
recs = []
for line in sys.stdin.read().splitlines():
    try:
        recs.append(json.loads(line))
    except ValueError:
        pass
if not recs:
    sys.exit("no records matched")
mode = sys.argv[1]
if mode == "streams":
    by = collections.defaultdict(list)
    size = collections.Counter()
    where = {}
    for r in recs:
        m = r["metadata"]
        s = m.get("lqcd.stream")
        by[s].append(m.get("lqcd.trajectory", -1))
        size[s] += r.get("size", 0)
        where[s] = m.get("lqcd.storage.path", "").rsplit("/", 1)[0]
    print("%-8s %6s %13s %8s  %s" % ("stream", "files", "trajectories", "TB", "HPSS directory"))
    for s in sorted(by):
        t = sorted(by[s])
        print("%-8s %6d %6d..%-6d %8.2f  %s" % (s, len(t), t[0], t[-1], size[s] / 1e12, where[s]))
else:
    recs.sort(key=lambda r: r["metadata"].get("lqcd.trajectory", -1))
    for r in recs:
        m = r["metadata"]
        print("%-8s %6s %6.2f GB  %s" % (m.get("lqcd.stream"), m.get("lqcd.trajectory", "-"),
              r.get("size", 0) / 1e9, m.get("lqcd.storage.path")))
'

if [ -z "$ENS" ]; then
    curl -sf --max-time 120 "$URL/datasets" | python3 -I -c "$PY_LIST"
    exit 0
fi

Q="files from lqcd:$ENS where lqcd.artifact.kind = \"$KIND\""
[ -n "$STREAM" ] && Q="$Q and lqcd.stream = \"$STREAM\""
[ -n "$TMIN" ] && Q="$Q and lqcd.trajectory >= $TMIN"
[ -n "$TMAX" ] && Q="$Q and lqcd.trajectory <= $TMAX"
echo "# MQL: $Q" >&2

MODE=files
[ -z "$STREAM" ] && MODE=streams
curl -sf --max-time 300 -G --data-urlencode 'with_meta=yes' --data-urlencode "query=$Q" "$URL/query" \
    | python3 -I -c "$PY_FILES" "$MODE"
