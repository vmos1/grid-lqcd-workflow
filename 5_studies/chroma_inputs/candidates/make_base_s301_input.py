"""Chroma baseline, second chain (2026-10-26): the 10-08 baseline chain's input (hmc_baseg_mds12_mgall20_s300,
base+G at 12/48, multigrid at every mass, 20 trajectories) with NEW seeds 301 1301 2301 3301, for an
independent 20 (the 10-08 chain stopped at 16 on a QUDA double-free). Same recipe as the B1/B2 inputs
(runs/2026_10_22_chroma_b1/make_b1_input.py): make_inputs.emit() with the baseline ladder written out.
Usage: python3 make_base_s301_input.py
"""
import sys

sys.path.insert(0, "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/inputs")
import make_inputs as mi  # noqa: E402

# Named hmc_baseg_mds12<SUFFIX> so submit_scripts/chroma_validation/chroma_baseg_mds.sh finds it
# with MDS=12 SUFFIX=_base_20_s301.
out = "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/inputs/hmc_baseg_mds12_base_20_s301.ini.xml"
# Regenerated 2026-10-26 before the job started: also the stout-smeared plaquette (the fermion
# action's smearing) and every trajectory's configuration on $PSCRATCH, for offline Wilson flow.
mi.emit(out, 20, 12, "baseg_mds12_base_20_s301",
        ladder=[-0.2416, -0.2400, -0.2320, -0.2180, -0.1870],
        seed=[301, 1301, 2301, 3301],
        keep_stout_plaq=True,
        save_dir="/pscratch/sd/v/vpa/chroma_cfgs/baseg_mds12_base_20_s301")
