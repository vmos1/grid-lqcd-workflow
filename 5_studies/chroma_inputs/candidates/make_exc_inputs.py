"""Phase 1 Chroma inputs for the exceptional-configs campaign (2026-11-10).

Each input is the validated template chroma/inputs/hmc_baseg_mds12_mgall20_s300.ini.xml (the
production recipe transcribed from cfg_2000's metadata, run on our build in tune 1, job 59499342)
with five fields replaced:

  cfg_file          the start configuration on $PSCRATCH/cfgs48/
  RNG Seed          the seed of the start's own restart XML (the original next job's seed),
                    or an explicit --seed where no restart XML exists (stream 2420)
  StartUpdateNum    the start trajectory
  NUpdatesThisRun   --ntraj (default 10)
  SavePrefix        ./exc_<tag> (SaveInterval stays 1000000: no configurations written)

Physics: diffed 11-10 against djm-1's restart XML 2810 -- identical monomials, integrator,
solvers and MG; the template only drops the in-run measurements and adds <Precision>SINGLE</>
twice (our build's validated setting).

usage: python3 make_exc_inputs.py <starts.tsv> <out dir> [--ntraj N] [--save-dir DIR]
       [--save-interval K]   (added 11-16: hmc saves config + restart XML whenever
       cur_update % K == 0, hmc.cc:637; K=1 saves every trajectory; default keeps 1000000)
starts.tsv columns: tag  cfg_path  start_traj  restart_xml_or_-  [seed4 as a.b.c.d]
Writes <out dir>/hmc_baseg_mds12_exc_<tag>.ini.xml; refuses to overwrite.
"""
import os
import re
import sys

TEMPLATE = "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/inputs/hmc_baseg_mds12_mgall20_s300.ini.xml"


def sub_once(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    if n != 1:
        sys.exit(f"template field not found: {what}")
    return new


def restart_seed(path):
    with open(path) as f:
        xml = f.read()
    m = re.search(r"<RNG>\s*<Seed>(.*?)</Seed>", xml, flags=re.S)
    if not m:
        sys.exit(f"no RNG Seed in {path}")
    elems = re.findall(r"<elem>\s*(\d+)\s*</elem>", m.group(1))
    if len(elems) != 4:
        sys.exit(f"seed in {path} has {len(elems)} elements, expected 4")
    return elems


def main():
    args = sys.argv[1:]
    ntraj = 10
    if "--ntraj" in args:
        i = args.index("--ntraj")
        ntraj = int(args[i + 1])
        del args[i:i + 2]
    # --save-dir (added 11-12): directory of the end-of-run save. hmc always writes the final
    # configuration and restart XML when a run finishes (hmc.cc: saveState after the update
    # loop, whatever SaveInterval is), so a finished run can be continued from them.
    save_dir = "."
    if "--save-dir" in args:
        i = args.index("--save-dir")
        save_dir = args[i + 1].rstrip("/")
        if not os.path.isdir(save_dir):
            sys.exit(f"--save-dir is not a directory: {save_dir}")
        del args[i:i + 2]
    save_interval = None
    if "--save-interval" in args:
        i = args.index("--save-interval")
        save_interval = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    starts, outdir = args
    with open(TEMPLATE) as f:
        tmpl = f.read()
    for line in open(starts):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        tag, cfg, traj, rxml = cols[:4]
        if not os.path.isfile(cfg):
            sys.exit(f"{tag}: configuration not found: {cfg}")
        if rxml != "-":
            seed = restart_seed(rxml)
            src = os.path.basename(rxml)
        else:
            if len(cols) < 5:
                sys.exit(f"{tag}: no restart XML and no explicit seed")
            seed = cols[4].split(".")
            src = "explicit"
        x = tmpl
        x = sub_once(r"<cfg_file>.*?</cfg_file>", f"<cfg_file>{cfg}</cfg_file>", x, "cfg_file")
        x = sub_once(r"(<RNG>\s*<Seed>).*?(</Seed>)",
                     lambda m: m.group(1) + "".join(f"\n        <elem>{s}</elem>" for s in seed)
                     + "\n      " + m.group(2), x, "RNG Seed")
        x = sub_once(r"<StartUpdateNum>\d+</StartUpdateNum>",
                     f"<StartUpdateNum>{int(traj)}</StartUpdateNum>", x, "StartUpdateNum")
        x = sub_once(r"<NUpdatesThisRun>\d+</NUpdatesThisRun>",
                     f"<NUpdatesThisRun>{ntraj}</NUpdatesThisRun>", x, "NUpdatesThisRun")
        x = sub_once(r"<SavePrefix>.*?</SavePrefix>",
                     f"<SavePrefix>{save_dir}/exc_{tag}</SavePrefix>", x, "SavePrefix")
        if save_interval is not None:
            x = sub_once(r"<SaveInterval>\d+</SaveInterval>",
                         f"<SaveInterval>{save_interval}</SaveInterval>", x, "SaveInterval")
        out = os.path.join(outdir, f"hmc_baseg_mds12_exc_{tag}.ini.xml")
        if os.path.exists(out):
            sys.exit(f"refusing to overwrite {out}")
        with open(out, "w") as f:
            f.write(x)
        print(f"{tag}\ttraj {traj}\tntraj {ntraj}\tsave {save_interval or 'end'}\tseed {' '.join(seed)} ({src})\t-> {out}")


if __name__ == "__main__":
    main()
