"""Chroma input for B2 at tail 24 (2026-10-26): the B2 input (runs/2026_10_20_chroma_b2/make_b2_input.py,
tail +0.1000, seeds 300 1300 2300 3300, 20 trajectories, multigrid at every mass) with the tail on
its own middle level: light rungs, strange and log-dets 12 / tail 24 / gauge 48 (LEVEL_SPECS "t24",
added to make_inputs.py the same day). Matches the Grid runs B2_tail24_* (CUSTOM_LEVEL_* 0/0/1/2).
Usage: python3 make_b2t24_input.py
"""
import sys

sys.path.insert(0, "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/inputs")
import make_inputs as mi  # noqa: E402

# Named hmc_baseg_mds12<SUFFIX> so submit_scripts/chroma_validation/chroma_baseg_mds.sh finds it
# with MDS=12 SUFFIX=_b2t24_20_s300.
out = "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/inputs/hmc_baseg_mds12_b2t24_20_s300.ini.xml"
mi.emit(out, 20, 12, "baseg_mds12_b2t24_20_s300",
        ladder=[-0.2416, -0.2400, -0.2320, -0.2180, 0.1000],
        levels="t24",
        seed=[300, 1300, 2300, 3300])
