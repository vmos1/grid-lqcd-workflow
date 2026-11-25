#!/usr/bin/env python3
"""Chroma HMC inputs for autotune 2 (2026-11-23).

Any light ladder (2 to 10 masses), any nested layout L/S/T/G, Grid's solver rule, a save after
every trajectory, and the resume of a run from its restart file. Built on make_inputs.py
(grid-lqcd-workflow/5_studies/chroma_inputs/, linked from chroma/inputs/, used unchanged): the
production template from cfg_2000's metadata, set_ladder, set_levels, set_seed, set_mccontrol,
trim_measurements, set_cg_heavy. What this file adds:

  - ladders longer than the template's 4 rungs. Rungs has1, has2, has3 are identical apart from
    their masses (HMC and MD copies alike), so an extra rung is a copy of has1, appended at the
    heavy end (ids hasx1, hasx2, ...), with its copyList entry and Hamiltonian reference;
  - a level spec from a layout L/S/T/G: the steps of the light rungs (with the two clover
    log-dets), the strange, the tail and the gauge. Distinct step counts sorted are the levels,
    outermost first; a nested level runs 2 x k x its parent's steps (k = its n_steps);
  - Grid's routing rule (tune5.py and at2_run.sh, MG_BELOW): multigrid for every mass below
    -0.226, CG at and above (set_cg_heavy with that crossover). --routing mgall keeps the
    template's multigrid at every mass (Chroma's own baseline), for checks against old inputs;
  - a decode of the written file (ladder, levels, solvers, seed, trajectories, saves), compared
    with the request: any mismatch exits 3 and the file must not be run.

Usage:
  at2_chroma_inputs.py new    --out F --tag T --ladder m0,m1,... --layout L/S/T/G --ntraj N
                              --seed S|template [--save-dir D] [--routing grid|mgall]
  at2_chroma_inputs.py resume --out F --restart R --ntraj N
  at2_chroma_inputs.py show   F
Seed S means the four QDP++ seed words S, S+1000, S+2000, S+3000 (the Chroma chains so far:
300 = 300 1300 2300 3300); "template" keeps the seed in cfg_2000's metadata.
"""
import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/inputs")
import make_inputs as mi  # noqa: E402

MG_BELOW = -0.226
GROUP_ORDER = ["logdets", "rungs", "strange", "tail", "gauge"]
EXTRA_SOURCE = "has1"
MAX_MASSES = 10


def die(msg, code=2):
    sys.stderr.write("at2_chroma_inputs: %s\n" % msg)
    sys.exit(code)


# ---------------------------------------------------------------------------------------------
# Layout -> level spec


def layout_spec(layout):
    """L/S/T/G -> (outer steps, spec for make_inputs.set_levels, steps per group)."""
    try:
        parts = [int(p) for p in layout.split("/")]
    except ValueError:
        parts = []
    if len(parts) != 4 or min(parts) < 1:
        die("layout must be L/S/T/G, four positive whole numbers: %s" % layout)
    steps = dict(zip(["light", "strange", "tail", "gauge"], parts))
    distinct = sorted(set(parts))
    mults = []
    for a, b in zip(distinct, distinct[1:]):
        if b % (2 * a):
            die("steps %d and %d do not nest: a nested level runs 2 x k x its parent's steps" % (a, b))
        mults.append(b // (2 * a))
    level = {"logdets": distinct.index(steps["light"]), "rungs": distinct.index(steps["light"]),
             "strange": distinct.index(steps["strange"]), "tail": distinct.index(steps["tail"]),
             "gauge": distinct.index(steps["gauge"])}
    spec = []
    for i in range(len(distinct)):
        names = [g for g in GROUP_ORDER if level[g] == i]
        spec.append((None if i == 0 else mults[i - 1], names))
    return distinct[0], spec, steps


# ---------------------------------------------------------------------------------------------
# Ladder of any length


def _mon_elem(root, mid):
    for elem in root.find(".//Monomials"):
        if (elem.findtext(".//monomial_id") or "").strip() == mid:
            return elem
    die("monomial %s not found in the template" % mid)


def set_ladder_any(root, ladder):
    """Retarget the ladder; returns the rung ids in ladder order (lightest first)."""
    order, old = mi.read_ladder(root)
    n_new, n_old = len(ladder) - 1, len(order)
    if n_new <= n_old:
        if n_new == n_old and all(abs(a - b) < 1e-9 for a, b in zip(ladder, old)):
            print("  ladder: the template's, %s" % ",".join("%.4f" % m for m in old))
            return order
        return mi.set_ladder(root, ladder)

    # The template's rungs take the first n_old + 1 masses; the tail sits at ladder[n_old] for now.
    keep = mi.set_ladder(root, ladder[:n_old + 1])
    src = order.index(EXTRA_SOURCE)
    src_num, src_den = ladder[src], ladder[src + 1]
    mons = root.find(".//Monomials")
    ham = root.find(".//Hamiltonian/monomial_ids")
    cl = root.find(".//MDIntegrator/copyList")
    last_mon = _mon_elem(root, "MD::%s" % keep[-1])
    last_ham = [e for e in ham if (e.text or "").strip() == "HMC::%s" % keep[-1]][0]
    last_cl = [e for e in cl if (e.findtext("copyFrom") or "").strip() == "HMC::%s" % keep[-1]][0]

    new_ids = []
    for j in range(n_old, n_new):
        rid = "hasx%d" % (j - n_old + 1)
        for kind in ("HMC", "MD"):
            elem = copy.deepcopy(_mon_elem(root, "%s::%s" % (kind, EXTRA_SOURCE)))
            elem.find(".//monomial_id").text = "%s::%s" % (kind, rid)
            mi._rewrite_masses(elem.find("Action"), src_num, ladder[j], "%s::%s <Action>" % (kind, rid))
            mi._rewrite_masses(elem.find("PrecAction"), src_den, ladder[j + 1],
                               "%s::%s <PrecAction>" % (kind, rid))
            pos = list(mons).index(last_mon) + 1
            mons.insert(pos, elem)
            last_mon = elem
        h = ET.Element("elem")
        h.text = "HMC::%s" % rid
        h.tail = last_ham.tail
        ham.insert(list(ham).index(last_ham) + 1, h)
        last_ham = h
        c = copy.deepcopy(last_cl)
        c.find("copyFrom").text = "HMC::%s" % rid
        c.find("copyTo").text = "MD::%s" % rid
        cl.insert(list(cl).index(last_cl) + 1, c)
        last_cl = c
        new_ids.append(rid)

    for _, elem in mi._monomials(root, mi.TAIL_MONOMIAL):
        mi._rewrite_masses(elem, ladder[n_old], ladder[-1], "tail")

    got_order, got = mi.read_ladder(root)
    if len(got) != len(ladder) or any(abs(a - b) > 1e-9 for a, b in zip(got, ladder)):
        die("grown ladder reads back as %s, not %s" % (got, ladder), 3)
    print("  grown : %d rungs copied from %s: %s" % (len(new_ids), EXTRA_SOURCE, ", ".join(new_ids)))
    return got_order


# ---------------------------------------------------------------------------------------------
# Decode a written input


def decode(path):
    root = ET.parse(path).getroot()
    order, ladder = mi.read_ladder(root)
    levels = []
    lvl = root.find(".//MDIntegrator/Integrator")
    parent = None
    while lvl is not None:
        n = int(lvl.findtext("n_steps"))
        steps = n if parent is None else 2 * n * parent
        ids = [(e.text or "").strip() for e in lvl.find("monomial_ids")]
        levels.append(dict(name=(lvl.findtext("Name") or "").strip(), n_steps=n, steps=steps, ids=ids))
        parent = steps
        lvl = lvl.find("SubIntegrator")
    where = {}
    for i, l in enumerate(levels):
        for mid in l["ids"]:
            where[mid] = l["steps"]
    groups = {"light": where.get("MD::%s" % order[0]), "strange": where.get("HMC::rat_strange"),
              "tail": where.get("MD::has_cancel_2flav"), "gauge": where.get("HMC::lw_tree_gauge"),
              "logdets": where.get("HMC::logdet_2flav")}
    rung_steps = sorted(set(where.get("MD::%s" % r) for r in order))
    solvers = {}
    for ip in root.iter("InvertParam"):
        m = ip.findtext("CloverParams/Mass")
        inv = (ip.findtext("invType") or "").strip()
        if m is None or not inv:
            continue
        kind = {"QUDA_MULTIGRID_CLOVER_INVERTER": "MG", "QUDA_CLOVER_INVERTER": "CG"}.get(inv, inv)
        solvers.setdefault(round(float(m), 6), set()).add(kind)
    seed = [e.text.strip() for e in root.find(".//MCControl/RNG/Seed").findall("elem")]
    mc = root.find(".//MCControl")
    return dict(order=order, ladder=ladder, levels=levels, groups=groups, rung_steps=rung_steps,
                solvers=solvers, seed=seed,
                start=mc.findtext("StartUpdateNum").strip(),
                ntraj=int(mc.findtext("NUpdatesThisRun")),
                save_interval=int(mc.findtext("SaveInterval")),
                save_prefix=mc.findtext("SavePrefix").strip(),
                cfg=mc.findtext("Cfg/cfg_file").strip())


def print_decode(d, path):
    print("decode %s" % path)
    print("  ladder : %s (%d masses, rungs %s)" % (",".join("%.4f" % m for m in d["ladder"]),
                                                   len(d["ladder"]), ",".join(d["order"])))
    print("  layout : light %s / strange %s / tail %s / gauge %s (log-dets %s; rung steps %s)"
          % (d["groups"]["light"], d["groups"]["strange"], d["groups"]["tail"], d["groups"]["gauge"],
             d["groups"]["logdets"], d["rung_steps"]))
    for i, l in enumerate(d["levels"]):
        print("    L%d %s n_steps=%d steps=%d: %s" % (i, l["name"], l["n_steps"], l["steps"], " ".join(l["ids"])))
    print("  solvers: %s" % ", ".join("%.4f %s" % (m, "+".join(sorted(k)))
                                      for m, k in sorted(d["solvers"].items())))
    print("  seed %s  start %s  trajectories %d  save every %d to %s"
          % (" ".join(d["seed"]), d["start"], d["ntraj"], d["save_interval"], d["save_prefix"]))
    print("  start configuration %s" % d["cfg"])


def check(d, ladder, steps, routing, seed_words, ntraj, save_dir):
    bad = []
    if len(d["ladder"]) != len(ladder) or any(abs(a - b) > 1e-9 for a, b in zip(d["ladder"], ladder)):
        bad.append("ladder %s" % d["ladder"])
    for g in ("light", "strange", "tail", "gauge"):
        if d["groups"][g] != steps[g]:
            bad.append("%s steps %s, asked %s" % (g, d["groups"][g], steps[g]))
    if d["groups"]["logdets"] != steps["light"] or d["rung_steps"] != [steps["light"]]:
        bad.append("log-dets or rungs off the light level")
    if any(l["name"] != mi.INTEGRATOR_NAME for l in d["levels"]):
        bad.append("integrator not %s on every level" % mi.INTEGRATOR_NAME)
    for m in ladder:
        kinds = d["solvers"].get(round(m, 6), set())
        want = "MG" if (routing == "mgall" or m < MG_BELOW) else "CG"
        if kinds != {want}:
            bad.append("mass %.4f solvers %s, want %s" % (m, sorted(kinds), want))
    if seed_words is not None and d["seed"] != [str(s) for s in seed_words]:
        bad.append("seed %s" % d["seed"])
    if d["ntraj"] != ntraj:
        bad.append("trajectories %d" % d["ntraj"])
    if save_dir is not None and (d["save_interval"] != 1 or not d["save_prefix"].startswith(save_dir)):
        bad.append("saves %d %s" % (d["save_interval"], d["save_prefix"]))
    if bad:
        die("the written input does not match the request: " + "; ".join(bad), 3)
    print("  check  : matches the request")


# ---------------------------------------------------------------------------------------------


def cmd_new(a):
    try:
        ladder = [float(x) for x in a.ladder.split(",")]
    except ValueError:
        die("ladder must be comma-separated masses: %s" % a.ladder)
    if not 2 <= len(ladder) <= MAX_MASSES:
        die("ladder must have 2 to %d masses" % MAX_MASSES)
    if any(b <= x for x, b in zip(ladder, ladder[1:])):
        die("ladder masses must increase: %s" % a.ladder)
    if abs(ladder[0] - (-0.2416)) > 1e-9:
        die("the ladder must start at the light mass -0.2416")
    mds, spec, steps = layout_spec(a.layout)
    seed_words = None if a.seed == "template" else [int(a.seed) + 1000 * k for k in range(4)]
    if a.save_dir is not None and not os.path.isdir(a.save_dir):
        die("save dir %s does not exist (Chroma will not create it)" % a.save_dir)
    if os.path.exists(a.out):
        die("%s exists: never overwrite an input" % a.out)

    tree = ET.parse(mi.TEMPLATE)
    root = tree.getroot()
    rung_ids = set_ladder_any(root, ladder)
    name = "at2_" + a.layout.replace("/", "_")
    mi.LEVEL_SPECS[name] = spec
    mi.set_levels(root, name, mds, rung_ids)
    if seed_words is not None:
        mi.set_seed(root, seed_words)
    mi.set_mccontrol(root, a.tag, a.ntraj, save_dir=a.save_dir)
    mi.trim_measurements(root, keep_mg_erase=True)
    if a.routing == "grid":
        cg = [m for m in ladder if m >= MG_BELOW]
        if cg:
            mi.set_cg_heavy(root, ladder, crossover=MG_BELOW)
        else:
            print("  routing: every mass below %.3f, multigrid throughout" % MG_BELOW)
    tree.write(a.out, encoding="UTF-8", xml_declaration=True)
    print("wrote %s" % a.out)
    d = decode(a.out)
    print_decode(d, a.out)
    check(d, ladder, steps, a.routing, seed_words, a.ntraj, a.save_dir)


def cmd_resume(a):
    if os.path.exists(a.out):
        die("%s exists: never overwrite an input" % a.out)
    tree = ET.parse(a.restart)
    root = tree.getroot()
    mc = root.find(".//MCControl")
    cfg = mc.find("Cfg/cfg_file")
    path = cfg.text.strip()
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(a.restart)), path))
        cfg.text = path
    if not os.path.exists(path):
        die("the restart's configuration %s does not exist" % path)
    mc.find("NUpdatesThisRun").text = str(a.ntraj)
    tree.write(a.out, encoding="UTF-8", xml_declaration=True)
    print("wrote %s (resume of %s)" % (a.out, a.restart))
    d = decode(a.out)
    print_decode(d, a.out)
    if d["ntraj"] != a.ntraj or d["cfg"] != path:
        die("the written resume input does not match the request", 3)
    print("  check  : matches the request")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    n = sub.add_parser("new")
    n.add_argument("--out", required=True)
    n.add_argument("--tag", required=True)
    n.add_argument("--ladder", required=True)
    n.add_argument("--layout", required=True)
    n.add_argument("--ntraj", type=int, required=True)
    n.add_argument("--seed", required=True)
    n.add_argument("--save-dir")
    n.add_argument("--routing", choices=["grid", "mgall"], default="grid")
    r = sub.add_parser("resume")
    r.add_argument("--out", required=True)
    r.add_argument("--restart", required=True)
    r.add_argument("--ntraj", type=int, required=True)
    s = sub.add_parser("show")
    s.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "new":
        if a.seed != "template" and not a.seed.isdigit():
            die("seed must be a whole number or 'template'")
        cmd_new(a)
    elif a.cmd == "resume":
        cmd_resume(a)
    elif a.cmd == "show":
        print_decode(decode(a.file), a.file)
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
