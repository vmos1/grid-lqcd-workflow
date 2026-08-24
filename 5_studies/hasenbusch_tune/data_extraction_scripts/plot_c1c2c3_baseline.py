#!/usr/bin/env python3
"""2x2 per-trajectory comparison figure for the base+G / C1 / C2 / C3 doc.

(C3 was added 2026-08-24, and this file, the doc and the figure were all renamed
to match on the same day -- see __docs/2026_8_20_c1_c2_c3_vs_baseg_comparison.md
and this directory's README.)

Thin driver over `plot_accept_runs.py`: it only rebinds that module's PALETTE /
DISPLAY / TITLE for this run set and calls its `main()`. Nothing about the
plotting itself is reimplemented, so the panels, colour rules and accept/reject
marker convention stay identical to every other figure in __docs/figs/.

Two things are deliberately different from the default title text:
  * "matched draw" is dropped. The four runs share seed 300 and start from the
    same cfg_2000, but they are NOT the same draw sequence -- C1's ladder has
    one fewer pseudofermion field than the baseline's, so it consumes the RNG
    stream differently and the chains diverge after the first trajectory. They
    are independent chains from a common start, which is what the observable
    comparison in the doc actually assumes.
  * the trajectory count and lattice are named, since the figure is used
    standalone in talks.

Usage:
  plot_c1c2c3_baseline.py <in.csv> <out.png>

  # CSV comes from regen_c1c2c3_comparison.sh, which is the run manifest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accept_runs as P

# Same Okabe-Ito hues and marker shapes as the parent module, re-keyed to this
# doc's run labels (which come from compare_runs_multiwindow.py's --run labels).
P.PALETTE = [
    ("base+G",        "#0072B2", "o"),
    ("C1 (u1a)",      "#E69F00", "s"),
    ("C2 (w3@5)",     "#009E73", "^"),
    ("C3 (w3@6+mid)", "#D55E00", "D"),
    ("_extra1",       "#CC79A7", "v"),
]
P.DISPLAY = {
    "base+G":        "base+G — baseline (2-level, MDSTEPS=12)",
    "C1 (u1a)":      "C1 = u1a (2-level, MDSTEPS=12)",
    "C2 (w3@5)":     "C2 = w3 (3-level, MDSTEPS=5)",
    "C3 (w3@6+mid)": "C3 = w3 + tail on strange level (3-level, MDSTEPS=6)",
}
P.TITLE = ("48³ HMC: C1 / C2 / C3 vs base+G baseline — 20 trajectories each, "
           "seed 300, from cfg_2000")
# Linear dH axis: none of these four runs blows up (every |dH| < 0.6), and the
# default symlog's linthresh=1 would squash all 80 points into the linear stub
# around zero, hiding exactly the spread the panel is meant to show.
P.DH_SCALE = "linear"
# Reference band = the baseline's OWN largest kick over its 20 trajectories
# (PF1, 0.1240). The nominal 0.10 default would put all four runs above the
# band and read as "all unhealthy"; the question this figure has to answer is
# narrower -- how far past the already-demonstrated-safe baseline kick do the
# candidates go.
#
# Note C3 goes a long way past it: its tail chain-max is 0.4203, 3.4x the band,
# and it still ran 20/20 ACC. That is not a defect of the band -- it is the
# figure's main finding, and the reason 0.4203 is now the campaign's ceiling to
# argue future candidates against. Panel D's y-range widens accordingly.
P.KICK_BAND = 0.1240
P.KICK_BAND_LABEL = "base+G max kick (0.124)"

if __name__ == "__main__":
    P.main(sys.argv[1] if len(sys.argv) > 1 else "accept_runs.csv",
           sys.argv[2] if len(sys.argv) > 2 else "c1c2_baseline_2x2.png")
