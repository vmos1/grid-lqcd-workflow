"""Group Chroma HMC input XMLs by settings fingerprints; print variants and diffs.

usage: python3 fingerprint.py <xml root with one subdir per stream>

Each file (restart XML or a config's scidac-record-xml) is split into
categories: monomials (masses, csw, smearing, rational approx; solver blocks
excluded), non-MG solver settings, MG settings, chrono predictor, integrator +
Hamiltonian, run control, inline measurements. Seed, start config and
trajectory counter are ignored. For each category the script prints every
distinct variant with the streams and trajectory ranges that use it, and the
lines that differ from the most common variant.
"""
import collections, glob, hashlib, os, re, sys
import xml.etree.ElementTree as ET

SKIP_IN_MONOMIAL = {"InvertParam", "ChronologicalPredictor"}


def flat(node, path="", skip=()):
    """Flatten an element into 'a/b[i]/c=value' lines."""
    out = []
    counts = collections.Counter()
    for ch in node:
        if ch.tag in skip:
            continue
        i = counts[ch.tag]
        counts[ch.tag] += 1
        p = f"{path}/{ch.tag}[{i}]" if ch.tag == "elem" else f"{path}/{ch.tag}"
        if len(ch):
            out += flat(ch, p, skip)
        else:
            out.append(f"{p}={(ch.text or '').strip()}")
    return out


def fingerprints(fn):
    r = ET.parse(fn).getroot()
    mon, sol, mg, chrono = [], [], [], []
    for m in r.find("HMCTrj/Monomials"):
        mid = m.findtext("NamedObject/monomial_id")
        mon += [f"{mid}{x}" for x in flat(m, skip=SKIP_IN_MONOMIAL)]
        for ip in m.iter("InvertParam"):
            sol += [f"{mid}{x}" for x in flat(ip, skip={"MULTIGRIDParams"})]
            for g in ip.iter("MULTIGRIDParams"):
                mg += [f"{mid}{x}" for x in flat(g)]
        for c in m.iter("ChronologicalPredictor"):
            chrono += [f"{mid}{x}" for x in flat(c)]
    mc = r.find("MCControl")
    return {
        "monomials (masses, csw, smearing, rational)": mon,
        "solver settings (non-MG)": sol,
        "multigrid settings": mg,
        "chrono predictor": chrono,
        "integrator + hamiltonian": flat(r.find("HMCTrj/MDIntegrator")) + flat(r.find("HMCTrj/Hamiltonian")),
        "run control": [
            f"reunit={mc.findtext('Cfg/reunit')}",
            f"SaveInterval={mc.findtext('SaveInterval')}",
            f"NUpdatesThisRun={mc.findtext('NUpdatesThisRun')}",
            f"NWarmUpUpdates={mc.findtext('NWarmUpUpdates')}",
        ],
        "inline measurements": flat(mc.find("InlineMeasurements")),
    }


def show_diff(a, b, indent="      ", limit=12):
    sa, sb = set(a), set(b)
    for x in sorted(sa - sb)[:limit]:
        print(f"{indent}- {x}")
    for x in sorted(sb - sa)[:limit]:
        print(f"{indent}+ {x}")
    if len(sa - sb) > limit or len(sb - sa) > limit:
        print(f"{indent}... {len(sa - sb)} removed / {len(sb - sa)} added lines in total")


def ranges(ts):
    ts = sorted(ts)
    return f"{ts[0]}-{ts[-1]} (n={len(ts)})"


def main(root):
    groups = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    content = collections.defaultdict(dict)
    nfiles = 0
    for sdir in sorted(glob.glob(os.path.join(root, "*"))):
        stream = os.path.basename(sdir)
        for fn in glob.glob(os.path.join(sdir, "*_restart_*.xml")):
            traj = int(re.search(r"_restart_(\d+)\.xml$", fn).group(1))
            for cat, lines in fingerprints(fn).items():
                h = hashlib.md5("\n".join(lines).encode()).hexdigest()[:8]
                groups[cat][h][stream].append(traj)
                content[cat][h] = lines
            nfiles += 1
    print(f"files: {nfiles}")
    for cat, variants in groups.items():
        print(f"\n== {cat}: {len(variants)} variant(s)")
        order = sorted(variants, key=lambda h: -sum(len(v) for v in variants[h].values()))
        for h in order:
            tot = sum(len(v) for v in variants[h].values())
            where = ", ".join(f"{s} {ranges(t)}" for s, t in sorted(variants[h].items()))
            print(f"  [{h}] {tot} files: {where}")
            if h != order[0]:
                show_diff(content[cat][order[0]], content[cat][h])


if __name__ == "__main__":
    main(sys.argv[1])
