#!/usr/bin/env python3
"""Chroma HMC runs -> the per-trajectory CSV of `compare_runs_multiwindow.py --csv`, so Chroma and
Grid runs share one downstream path (tables, `obs_autocorr_compare.py`, the 2x2 figure).

Generalises the one-off `runs/2026_10_7_baseG_trajl1_seed300_20/compare_vs_chroma.py` (2026-10-27):
same sources, any number of runs, each one or more run directories stitched in order.
  <rundir>/*.xmldat                          update_no, deltaH, AcceptP, seconds_for_trajectory
  <rundir>/plaquette/thin_plaquette_dat.xml.<n>               w_plaq (thin)
  <rundir>/plaquette/plaquette_sm1_rho0.125_dat.xml.<n>       w_plaq (stout-smeared, if measured)
Chroma has no MD-only interval, so wall_s and cycle_s are both seconds_for_trajectory (full
trajectory); columns Chroma does not measure (Polyakov loop, force time, kicks) are left empty.
The first trajectory of each directory is flagged cold / resume, as in the Grid CSV.

Usage:
  chroma_runs_csv.py --run 'LABEL=rundir[,rundir...]' [--run ...] --csv out.csv
"""
import csv
import math
import re
import sys
from pathlib import Path

COLS = ["run", "traj", "traj_abs", "accepted", "cold", "resume", "dH", "exp_dH",
        "plaquette", "plaquette_smeared", "poly_re", "poly_im",
        "wall_s", "sigma_force_s", "limiter_kick", "cycle_s"]


def w_plaq(path):
    if not path.exists():
        return None
    return float(re.search(r"<w_plaq>(\S+?)</w_plaq>", path.read_text(errors="replace")).group(1))


def run_rows(label, dirs):
    rows = []
    for k, d in enumerate(dirs):
        d = Path(d)
        x = next(d.glob("*.xmldat")).read_text(errors="replace")
        ups = [int(u) for u in re.findall(r"<update_no>(\d+)", x)]
        dhs = [float(v) for v in re.findall(r"<deltaH>(\S+?)</deltaH>", x)]
        acs = [v == "true" for v in re.findall(r"<AcceptP>(\w+)</AcceptP>", x)]
        sec = [float(v) for v in re.findall(r"<seconds_for_trajectory>(\S+?)</seconds_for_trajectory>", x)]
        n = min(len(ups), len(dhs), len(acs), len(sec))
        for j in range(n):
            u = ups[j]
            thin = w_plaq(d / "plaquette" / ("thin_plaquette_dat.xml.%d" % u))
            sm = w_plaq(d / "plaquette" / ("plaquette_sm1_rho0.125_dat.xml.%d" % u))
            rows.append({
                "run": label, "traj": len(rows) + 1, "traj_abs": u, "accepted": int(acs[j]),
                "cold": 1 if (k == 0 and j == 0) else 0, "resume": 1 if (k > 0 and j == 0) else 0,
                "dH": "%.6f" % dhs[j], "exp_dH": "%.6f" % math.exp(-dhs[j]),
                "plaquette": "%.8f" % thin, "plaquette_smeared": "%.8f" % sm if sm is not None else "",
                "poly_re": "", "poly_im": "", "wall_s": "%.1f" % sec[j], "sigma_force_s": "",
                "limiter_kick": "", "cycle_s": "%.1f" % sec[j]})
    return rows


def main(argv):
    runs, out = [], None
    i = 0
    while i < len(argv):
        if argv[i] == "--run":
            label, _, dirs = argv[i + 1].partition("=")
            runs.append((label, dirs.split(",")))
            i += 2
        elif argv[i] == "--csv":
            out = argv[i + 1]
            i += 2
        else:
            sys.exit("unknown argument %s\n%s" % (argv[i], __doc__))
    if not runs or not out:
        sys.exit(__doc__)
    rows = [r for label, dirs in runs for r in run_rows(label, dirs)]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    sys.stderr.write("wrote %s  (%d rows)\n" % (out, len(rows)))


if __name__ == "__main__":
    main(sys.argv[1:])
