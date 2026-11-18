# Chroma inputs for the baseline and the HMC candidates

The Chroma `hmc` inputs of the 20-trajectory chains from cfg_2000 (48³×96, 4 nodes × 4 A100,
`-geom 1.2.2.4`), and the scripts that wrote them. Copied 2026-11-18 from `chroma/inputs/` and
the `runs/` folders where they were made on Perlmutter; the XML files are the exact inputs
those runs read. Results: `__docs/2026_11_16_hmc_candidates_benchmark.md` (private docs repo).

| setup | input | made by | launcher settings |
|---|---|---|---|
| baseline (seed 300) | `hmc_baseg_mds12_mgall20_s300.ini.xml` | `make_inputs.py` (10-08 chain) | `MDS=12 SUFFIX=_mgall20_s300` |
| baseline (seed 301, saves every configuration) | `hmc_baseg_mds12_base_20_s301.ini.xml` | `make_base_s301_input.py` | `MDS=12 SUFFIX=_base_20_s301` |
| B2_t24 | `hmc_baseg_mds12_b2t24_20_s300.ini.xml` | `make_b2t24_input.py` | `MDS=12 SUFFIX=_b2t24_20_s300` |
| D_8_16_32_64 (gauge level written as 128 here; 128 and 64 agree) | `hmc_baseg_mds8_t2fin_20_s300.ini.xml` | `make_t2_chroma_inputs.py` | `MDS=8 SUFFIX=_t2fin_20_s300` |
| D_10_40_80 | `hmc_baseg_mds10_t2hedge_20_s300.ini.xml` | `make_t2_chroma_inputs.py` | `MDS=10 SUFFIX=_t2hedge_20_s300` |

`make_exc_inputs.py` writes inputs that restart from a stream's saved configuration with the
seed of that save's restart XML (the baseline recipe, from the template
`hmc_baseg_mds12_mgall20_s300.ini.xml`): the way to continue a production stream.

The launcher is `../perlmutter/chroma_baseg_mds.sh` (Perlmutter: it sources
`../perlmutter/env_chroma_pm.sh` and finds `hmc_baseg_mds${MDS}${SUFFIX}.ini.xml` in
`$CHROMA_INPUTS`). Another machine writes its own launcher under `../<machine>/`.

Paths are Perlmutter's and absolute: each maker's `sys.path` line (where `make_inputs.py`
lives; it is `..` in this repo) and its output path, and in each XML `<cfg_file>` (cfg_2000)
and, for seed 301, the configuration save directory. Change those, nothing else.
