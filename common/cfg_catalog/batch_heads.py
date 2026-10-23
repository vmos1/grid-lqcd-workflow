"""Read many config headers from HPSS and classify each one against a reference input.

usage: python3 batch_heads.py <list file> <out root> <reference xml> [stream:traj ...]

<list file> has `stream trajectory size path` per line (noxml_paths.py); with
stream:traj arguments only those entries are read. Each header lands in
<out root>/<stream>_<traj>/ (lime_head.py). One line per config: setup (A if
the input has `rat_3strange`, B if `rat_strange`), MG first-level blocking,
and the categories that differ from the reference (fingerprint.py; run
control ignored). For the cl21 48^3x96 study the reference was our
cfg_2000's input: `lime_head.py data/...-djm-3_cfg_2000.lime <dir>`, then
`<dir>/03_scidac-record-xml.xml`. Check `cache_status.py` first: TAPE
entries stage 6 GB each.
"""
import os, subprocess, sys
import xml.etree.ElementTree as ET

LIB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIB)
from fingerprint import fingerprints

lst, out, refxml = sys.argv[1], sys.argv[2], sys.argv[3]
want = set(sys.argv[4:])
ref = fingerprints(refxml)

for line in open(lst):
    stream, traj, size, path = line.split()
    if want and f"{stream}:{traj}" not in want:
        continue
    d = os.path.join(out, f"{stream}_{traj}")
    r = subprocess.run([sys.executable, os.path.join(LIB, "lime_head.py"), "hpss:" + path, d],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    fn = os.path.join(d, "03_scidac-record-xml.xml")
    if not os.path.exists(fn):
        err = [l.strip() for l in r.stdout.splitlines() if "hsi:" in l or "***" in l]
        print(f"{stream:8s} {traj:>5s}  NOT READ  {err[0] if err else r.stdout.strip()[-80:]}")
        continue
    root = ET.parse(fn).getroot()
    ids = [e.findtext("NamedObject/monomial_id") for e in root.find("HMCTrj/Monomials")]
    setup = "A" if "HMC::rat_3strange" in ids else ("B" if "HMC::rat_strange" in ids else "?")
    blk = sorted({b.findall("elem")[0].text for b in root.iter("Blocking") if b.findall("elem")})
    fp = fingerprints(fn)
    diff = [c.split(" (")[0] for c in ref if c != "run control" and ref[c] != fp.get(c)]
    print(f"{stream:8s} {traj:>5s}  setup {setup}  MG blocking {blk or 'none'}  "
          f"differs from reference in: {diff if diff else 'nothing (besides run control)'}")
