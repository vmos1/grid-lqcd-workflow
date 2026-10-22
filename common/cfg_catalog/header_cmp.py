"""Compare HMC input XMLs against a reference, per fingerprint category.

usage: python3 header_cmp.py <reference xml> <xml> [<xml> ...]

Prints `same` or the differing lines per category (see fingerprint.py).
Use it on lime_head.py output (`03_scidac-record-xml.xml`) or restart XMLs.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint import fingerprints, show_diff

ref = fingerprints(sys.argv[1])
for fn in sys.argv[2:]:
    print(f"\n### {fn}")
    fp = fingerprints(fn)
    for cat in ref:
        a, b = ref[cat], fp.get(cat, [])
        if a == b:
            print(f"  {cat}: same")
        else:
            print(f"  {cat}: DIFFERENT ({len(a)} vs {len(b)} lines)")
            show_diff(a, b, limit=15)
