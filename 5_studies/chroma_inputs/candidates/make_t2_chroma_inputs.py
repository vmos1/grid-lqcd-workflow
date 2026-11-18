"""Chroma inputs for the tune 2 20-trajectory chains (2026-11-02): the baseline ladder and the
Chroma baseline chain's settings (seeds 300 1300 2300 3300, 20 trajectories, multigrid at every
mass, as runs/2026_10_8_chroma_baseg_mds12_mgall20_s300 and the B2 chains), with the integrator
nesting replaced by the tune 2 layouts (LEVEL_SPECS added to make_inputs.py 2026-11-02):
  finalist  8/16/32/128  spec l8s16t32g128, outer n_steps 8   -> hmc_baseg_mds8_t2fin_20_s300
  hedge     10/40/80     spec l10t40g80,    outer n_steps 10  -> hmc_baseg_mds10_t2hedge_20_s300
Named hmc_baseg_mds<MDS><SUFFIX> so submit_scripts/chroma_validation/chroma_baseg_mds.sh finds
them with MDS=8 SUFFIX=_t2fin_20_s300 and MDS=10 SUFFIX=_t2hedge_20_s300.
Usage: python3 make_t2_chroma_inputs.py
"""
import sys

sys.path.insert(0, "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/inputs")
import make_inputs as mi  # noqa: E402

INPUTS = "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/inputs"
SEEDS = [300, 1300, 2300, 3300]

print("== finalist 8/16/32/128")
mi.emit(INPUTS + "/hmc_baseg_mds8_t2fin_20_s300.ini.xml", 20, 8, "baseg_mds8_t2fin_20_s300",
        levels="l8s16t32g128", seed=SEEDS)
print("== hedge 10/40/80")
mi.emit(INPUTS + "/hmc_baseg_mds10_t2hedge_20_s300.ini.xml", 20, 10, "baseg_mds10_t2hedge_20_s300",
        levels="l10t40g80", seed=SEEDS)
