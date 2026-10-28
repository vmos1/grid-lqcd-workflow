#!/usr/bin/env python3
"""
Emit Chroma HMC inputs from the baseline recipe embedded in cfg_2000.lime's
LIME metadata.

Run extract_baseline.sh first; it produces _baseline_template.xml, the verbatim
<Params> block.  This script then makes the edits our runs need:

  1. Outer <n_steps> -> 12 / 10 / 8.  The inner SubIntegrator <n_steps> stays
     at 2, which is 48 inner steps after the factor-of-2 drift-slot recursion,
     NOT 24.  That pitfall has bitten this project twice; see
     __docs/2026_8_10_integrator_factor_of_2_explained.md.

  2. Trim <InlineMeasurements>.  The production recipe measures a Wilson flow
     (nstep=200) and a Ritz eigenvalue bound EVERY trajectory -- neither is
     something the Grid runs do, and both would land squarely in the wall-clock
     numbers we are trying to compare.  Kept:

       PLAQUETTE (thin)                 cheap, and our correctness anchor
       ERASE_QUDA_MULTIGRID_SUBSPACE    KEPT DELIBERATELY -- see note below

     Dropped: PLAQUETTE (stout-smeared), EIGBNDSMDAGM, WILSON_FLOW,
     ERASE_NAMED_OBJECT(wflow).

NOTE ON MG CADENCE.
The baseline does NOT simply rebuild multigrid from scratch every trajectory --
an earlier version of this note claimed it did, which was wrong.  The embedded
recipe uses one shared <SubspaceID>mg_subspace</SubspaceID> with an adaptive
iteration-count policy: <ThresholdCount>70</ThresholdCount> on the
near-critical Mass=-0.2416 solver and 500 on the other 17.  That 70 is exactly
the threshold our Grid setup was matched to.

ERASE_QUDA_MULTIGRID_SUBSPACE (Frequency 1) additionally clears the subspace
*between* trajectories -- on top of the adaptive refresh, not instead of it.
It is kept because the goal is to reproduce the baseline as it ran.
--no-mg-erase drops it, which is a deliberate DEVIATION from the baseline;
use it only to isolate the cost of that erase, not for the headline number.

--cg-heavy / --cg-from are likewise deliberate deviations: they swap Chroma's
solver routing for Grid's.  See set_cg_heavy() below.

--candidate {baseg,c1,c2,c3} emits a whole campaign configuration -- ladder,
integrator levels, MDSTEPS and solver routing together.  See CANDIDATES and
__docs/2026_8_24_chroma_c1_c2_c3_comparison.md.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

# This file and _baseline_template.xml live in grid-lqcd-workflow/5_studies/chroma_inputs/ (moved
# 2026-10-28) and are symlinked from <workspace>/chroma/inputs/, where the generated inputs and
# every caller are. Load it through that path: abspath keeps the symlink's directory, so HERE is
# chroma/inputs and GRID_QCD the workspace root; loaded from the repo path, GRID_QCD would be wrong.
HERE = os.path.dirname(os.path.abspath(__file__))
GRID_QCD = os.path.abspath(os.path.join(HERE, "..", ".."))
TEMPLATE = os.path.join(HERE, "_baseline_template.xml")
CFG = os.path.join(
    GRID_QCD, "data", "cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime"
)

KEEP_MEASUREMENTS = {"PLAQUETTE", "ERASE_QUDA_MULTIGRID_SUBSPACE"}

# The 9 monomials of the production baseline.  Candidates with a thinner ladder
# drop rungs from this list; check_monomials() is told what to expect.
EXPECTED_MONOMIALS = [
    "HMC::logdet_2flav", "HMC::logdet_1flav", "HMC::rat_strange",
    "HMC::has3", "HMC::has2", "HMC::has1", "HMC::has0",
    "HMC::has_cancel_2flav", "HMC::lw_tree_gauge",
]

RATIO_MONOMIAL = "TWO_FLAVOR_EOPREC_CONSTDET_RATIO_CONV_CONV_FERM_MONOMIAL"
TAIL_MONOMIAL = "TWO_FLAVOR_EOPREC_CONSTDET_FERM_MONOMIAL"
INTEGRATOR_NAME = "LCM_STS_FORCE_GRAD"

CG_MAXITER = 20000
# Grid's routing, and the default: multigrid on rungs 0-2, CG at -0.2180 and
# heavier.  Measured 2026-08-24 to be worth 565 s/traj over Chroma's own
# two-subspace baseline at identical physics (plaquette agrees to 1e-13).
# The crossover is a property of the implementation pair, not of the physics --
# Chroma updates its subspace on nearly every solve where Grid shares one donor
# subspace across rungs, so Chroma's crossover should sit further left.  Hence
# a threshold rather than a fixed list.
CG_CROSSOVER_DEFAULT = -0.2180

# ---------------------------------------------------------------------------
# Campaign configurations.
#
# A ladder [m0 .. mk] means: rung i is the ratio (m_i, m_{i+1}), and the tail is
# a plain two-flavour determinant at m_k.  Chroma stores a rung's numerator in
# <Action> and its denominator in <PrecAction>; the NUMERATOR is the mass the MD
# force solves at, the denominator serves the heatbath.
LADDERS = {
    "baseg": [-0.2416, -0.2400, -0.2320, -0.2180, -0.1870],
    "u1a":   [-0.2416, -0.2250, -0.1870,  0.3044],
    "w3":    [-0.2416, -0.2380, -0.2340, -0.2180, -0.1870],
}

# Integrator levels, outermost first.  n_steps is the XML value, NOT the step
# count: the force-gradient two-drift-slot recursion means
#     steps(level) = 2 * n_steps * steps(parent)
# so base+G's inner <n_steps>2</n_steps> under an outer 12 is 48 real steps, not
# 24.  set_levels() prints the resolved counts so this stays checkable.
LEVEL_SPECS = {
    # base+G: 2-level, tail rides with the gauge field on the inner level
    "baseg": [(None, ["logdets", "rungs", "strange"]), (2, ["tail", "gauge"])],
    # C1/u1a: 2-level, tail moved UP to the light level; gauge alone below.
    # The tail determinant sits at +0.3044, heavy enough that its force no
    # longer needs the gauge sector's 48 steps.  This is u1's whole design.
    "u1a":   [(None, ["logdets", "rungs", "strange", "tail"]), (2, ["gauge"])],
    # C2: 3-level, strange gets its own level, tail stays with gauge
    "c2":    [(None, ["logdets", "rungs"]), (1, ["strange"]), (2, ["tail", "gauge"])],
    # C3: 3-level, tail comes off gauge but only as far as the strange level
    "c3":    [(None, ["logdets", "rungs"]), (1, ["strange", "tail"]), (2, ["gauge"])],
    # Structural probe, not a physics candidate.  C2's layout with ONE extra
    # monomial added to the middle level -- the heaviest ladder rung, whose force
    # is ~56x smaller than the tail's (0.0244 vs 1.3616 in Grid units).
    #
    # Every configuration that has run successfully so far carries at most ONE
    # monomial on a nested MIDDLE level (C2's is strange alone).  Multi-monomial
    # levels are proven only at the root (C1: seven) and at the innermost level
    # (base+G, C2: tail+gauge).  C3 is the first to put two on a middle level,
    # and it fails -- so "two monomials on a middle level is mis-built" is a live
    # hypothesis that the C1/C3 force-threshold comparison does NOT rule out,
    # because C1 is 2-level and its tail rides the ROOT.
    #
    # This isolates it: start from a configuration known to work, change only the
    # monomial count on the middle level, and keep the added force negligible.
    # Breaks => construction bug.  Runs => multi-monomial middle levels are sound
    # and C3's failure is about the tail's force, not the wiring.
    "c2rungmid": [(None, ["logdets", "rungs_light"]),
                  (1, ["strange", "rung_heavy"]),
                  (2, ["tail", "gauge"])],
    # T24 (2026-10-22, tune1-perturbations B2 at tail 24): base+G with the tail
    # alone on a middle level, 12 / 24 / 48. One monomial on the middle level, the
    # pattern C2 ran with (its strange alone). Grid: CUSTOM_LEVEL_* 0/0/1/2.
    "t24":   [(None, ["logdets", "rungs", "strange"]), (1, ["tail"]), (1, ["gauge"])],
}

STATIC_GROUPS = {
    "logdets": ["HMC::logdet_2flav", "HMC::logdet_1flav"],
    "strange": ["HMC::rat_strange"],
    "tail":    ["MD::has_cancel_2flav"],
    "gauge":   ["HMC::lw_tree_gauge"],
}

CANDIDATES = {
    "baseg": dict(ladder="baseg", levels="baseg", mdsteps=12, cg_from=-0.2180),
    "c1":    dict(ladder="u1a",   levels="u1a",   mdsteps=12, cg_from=-0.2250),
    "c2":    dict(ladder="w3",    levels="c2",    mdsteps=5,  cg_from=-0.2180),
    "c3":    dict(ladder="w3",    levels="c3",    mdsteps=6,  cg_from=-0.2180),
}


def trim_measurements(root, keep_mg_erase, keep_stout_plaq=False):
    """Drop the expensive per-trajectory diagnostics; keep the thin plaquette.

    keep_stout_plaq (2026-10-26): also keep the stout-smeared plaquette (rho 0.125,
    one step, all directions: the fermion action's smearing, the field Grid's
    PlaquetteLogger prints as its second line). Seconds per trajectory."""
    inline = root.find(".//InlineMeasurements")
    if inline is None:
        print("  (no InlineMeasurements block)")
        return
    dropped, kept = [], []
    for elem in list(inline):
        name_node = elem.find("Name")
        name = name_node.text.strip() if name_node is not None and name_node.text else "?"

        keep = name in KEEP_MEASUREMENTS
        if name == "ERASE_QUDA_MULTIGRID_SUBSPACE" and not keep_mg_erase:
            keep = False
        # Of the two PLAQUETTE entries keep only the thin (unsmeared) one,
        # unless keep_stout_plaq asks for the smeared one too.
        if name == "PLAQUETTE":
            gs = elem.find(".//GaugeState/Name")
            if gs is not None and gs.text and "STOUT" in gs.text and not keep_stout_plaq:
                keep = False

        if keep:
            kept.append(name)
        else:
            dropped.append(name)
            inline.remove(elem)
    print("  kept measurements   : %s" % (", ".join(kept) or "(none)"))
    print("  dropped measurements: %s" % (", ".join(dropped) or "(none)"))


def set_outer_nsteps(root, mds):
    """Set ONLY the top-level Integrator's n_steps; leave SubIntegrator alone."""
    integ = root.find(".//MDIntegrator/Integrator")
    if integ is None:
        sys.exit("ERROR: no <MDIntegrator>/<Integrator> found")
    n = integ.find("n_steps")
    if n is None:
        sys.exit("ERROR: outer Integrator has no <n_steps>")
    n.text = str(mds)

    sub = integ.find("SubIntegrator/n_steps")
    inner = sub.text.strip() if sub is not None and sub.text else "?"
    if inner != "2":
        sys.exit("ERROR: inner n_steps is %s, expected 2 (see factor-of-2 doc)" % inner)
    return inner


def set_seed(root, seed):
    """Replace the QDP++ RNG seed (4 words) under <RNG><Seed>.

    Every candidate so far has shared one seed, which is why C1/C2/C3 give
    bit-identical heatbath draws and are perfectly controlled against each other.
    Changing it is how you tell a reproducible result from a one-off event.
    """
    node = root.find(".//RNG/Seed")
    if node is None:
        sys.exit("ERROR: no <RNG><Seed> block found in the template")
    elems = node.findall("elem")
    if len(elems) != len(seed):
        sys.exit("ERROR: seed has %d words, template expects %d"
                 % (len(seed), len(elems)))
    old = [e.text for e in elems]
    for e, v in zip(elems, seed):
        e.text = str(v)
    print("  seed: %s -> %s" % (",".join(old), ",".join(str(v) for v in seed)))


def set_mccontrol(root, tag, ntraj, repro=False, reverse=False, save_dir=None):
    def put(name, value):
        node = root.find(".//%s" % name)
        if node is not None:
            node.text = value

    put("cfg_file", CFG)
    put("NUpdatesThisRun", str(ntraj))
    if save_dir is None:
        put("SaveInterval", "1000000")      # do not write configs during validation
        put("SavePrefix", "./chroma_%s" % tag)
    else:
        # 2026-10-26: every trajectory's configuration (6.1 GB each), for offline
        # observables (Wilson flow). save_dir must exist and should be on $PSCRATCH.
        put("SaveInterval", "1")
        put("SavePrefix", os.path.join(save_dir, "chroma_%s" % tag))
    put("MonitorForces", "true")            # needed for the per-monomial cross-check

    # ReproCheckP and ReverseCheckP are DIFFERENT checks -- see hmc.cc:79-107.
    #
    #   ReproCheckP   re-runs the same trajectory forward a second time from the
    #                 saved config and RNG seed and compares.  It detects
    #                 non-determinism (races, uninitialised state), NOT
    #                 integrator error.
    #
    #   ReverseCheckP flips the momenta at the end of the trajectory, integrates
    #                 back, and reports DeltaDeltaH / DeltaQPerSite /
    #                 DeltaPPerSite (abs_hmc.h:199-261).  This is the one that
    #                 separates "the integrator is correct but inaccurate"
    #                 (reversible, violation at solver precision) from "the MD
    #                 map is not the symplectic map we think it is"
    #                 (irreversible), which is what a large Delta H at a step
    #                 size another code handles fine demands.
    #
    # Both roughly double trajectory cost, so both stay off for timing runs.
    # Note Chroma's own defaults are true/frequency-10 for BOTH; writing the
    # tags explicitly (as the template does) is what suppresses them.
    put("ReproCheckP", "true" if repro else "false")
    put("ReverseCheckP", "true" if reverse else "false")

    if reverse:
        # The frequency gate is `cur_update % frequency == 0` (hmc.cc:445).
        # Chroma's default of 10 never fires on a short run starting at update
        # 2001, so the frequency tag has to be present and set to 1.  The
        # template does not carry it, so insert it after <ReverseCheckP>.
        parent = root.find(".//ReverseCheckP/..")
        node = parent.find("ReverseCheckFrequency")
        if node is None:
            node = ET.Element("ReverseCheckFrequency")
            parent.insert(list(parent).index(parent.find("ReverseCheckP")) + 1,
                          node)
        node.text = "1"


# ---------------------------------------------------------------------------
# Ladder surgery


def _monomials(root, kind):
    """Every monomial <elem> of the given <Name>, as (base_id, elem) pairs.

    <Name> is read as a direct child on purpose -- the subtree is full of other
    <Name> tags (STOUT_FERM_STATE, SIMPLE_FERMBC, ...) that a .// search would
    hit.  <monomial_id> conversely sits one level down inside <NamedObject>.
    """
    out = []
    for elem in root.iter("elem"):
        name = elem.findtext("Name")
        mid = elem.findtext(".//monomial_id")
        if name and mid and name.strip() == kind:
            out.append((mid.strip().split("::")[-1], elem))
    return out


def _rewrite_masses(subtree, old, new, where):
    """Retarget every <Mass> in a subtree that currently reads `old`.

    Scoping to the subtree is what makes this unambiguous: a rung's <Action>
    holds exactly two masses (the FermionAction and its InvertParam's
    CloverParams), both the numerator, so a value-keyed rewrite cannot stray
    into the denominator even when two rungs share a mass.
    """
    hits = [m for m in subtree.iter("Mass") if abs(float(m.text) - old) < 1e-9]
    if len(hits) != 2:
        sys.exit("ERROR: %s holds %d masses at %.4f, expected 2 -- the "
                 "template's monomial layout has changed" % (where, len(hits), old))
    for m in hits:
        m.text = "%.4f" % new


def read_ladder(root):
    """Recover the ladder from the template by chaining rungs off the tail.

    Returns (rung_ids_outermost_first, ladder_masses).  Chaining rather than
    trusting the has3/has2/... names means a renamed or reordered template is
    caught rather than silently mis-mapped.
    """
    rungs = {}
    for base, elem in _monomials(root, RATIO_MONOMIAL):
        num = float(elem.find("Action/FermionAction/Mass").text)
        den = float(elem.find("PrecAction/FermionAction/Mass").text)
        if base in rungs and rungs[base] != (num, den):
            sys.exit("ERROR: rung %s has inconsistent HMC/MD masses" % base)
        rungs[base] = (num, den)

    tails = _monomials(root, TAIL_MONOMIAL)
    if not tails:
        sys.exit("ERROR: no tail (%s) monomial found" % TAIL_MONOMIAL)
    tail_mass = float(tails[0][1].find("FermionAction/Mass").text)

    by_den = {}
    for base, (num, den) in rungs.items():
        by_den[round(den, 6)] = base

    order, cur = [], tail_mass
    while round(cur, 6) in by_den:
        base = by_den.pop(round(cur, 6))
        order.append(base)
        cur = rungs[base][0]
    order.reverse()

    if len(order) != len(rungs):
        sys.exit("ERROR: ladder does not form a single chain (found %d of %d "
                 "rungs: %s)" % (len(order), len(rungs), order))

    ladder = [rungs[b][0] for b in order] + [tail_mass]
    return order, ladder


def _drop_rungs(root, bases):
    """Remove rung monomials entirely: the HMC and MD <elem>s, the copyList
    entry that links them, and the Hamiltonian's monomial_ids reference.  The
    integrator's own list is rebuilt from scratch by set_levels()."""
    mons = root.find(".//Monomials")
    for elem in list(mons):
        mid = elem.findtext(".//monomial_id")
        if mid and mid.strip().split("::")[-1] in bases:
            mons.remove(elem)

    cl = root.find(".//MDIntegrator/copyList")
    if cl is not None:
        for elem in list(cl):
            cf = (elem.findtext("copyFrom") or "").strip().split("::")[-1]
            if cf in bases:
                cl.remove(elem)

    ham = root.find(".//Hamiltonian/monomial_ids")
    if ham is not None:
        for elem in list(ham):
            if (elem.text or "").strip().split("::")[-1] in bases:
                ham.remove(elem)


def set_ladder(root, new_ladder):
    """Retarget the Hasenbusch ladder, dropping surplus rungs.

    A ladder of k+1 masses defines k ratio rungs plus a tail determinant at the
    last mass.  Rungs are matched to the template positionally from the light
    end, so a shorter ladder drops rungs nearest the tail -- which is what the
    thin-ladder (u1) candidates want.
    """
    if len(set(new_ladder)) != len(new_ladder):
        sys.exit("ERROR: ladder has repeated masses: %s" % (new_ladder,))

    order, old_ladder = read_ladder(root)
    n_new, n_old = len(new_ladder) - 1, len(order)
    if n_new > n_old:
        sys.exit("ERROR: ladder needs %d rungs, template has %d -- the template "
                 "cannot grow rungs, only lose them" % (n_new, n_old))

    keep, drop = order[:n_new], order[n_new:]

    for i, base in enumerate(keep):
        for _, elem in [(b, e) for b, e in _monomials(root, RATIO_MONOMIAL) if b == base]:
            _rewrite_masses(elem.find("Action"), old_ladder[i], new_ladder[i],
                            "%s <Action>" % base)
            _rewrite_masses(elem.find("PrecAction"), old_ladder[i + 1],
                            new_ladder[i + 1], "%s <PrecAction>" % base)

    for _, elem in _monomials(root, TAIL_MONOMIAL):
        _rewrite_masses(elem, old_ladder[-1], new_ladder[-1], "tail")

    if drop:
        _drop_rungs(root, set(drop))

    print("  ladder: %s -> %s"
          % (",".join("%.4f" % m for m in old_ladder),
             ",".join("%.4f" % m for m in new_ladder)))
    print("  rungs : %s%s"
          % (", ".join(keep), "  (dropped %s)" % ", ".join(drop) if drop else ""))
    return keep


# ---------------------------------------------------------------------------
# Integrator levels


# Dropping the force-gradient sub-step on one level, for the C3 diagnosis.
#
# LCM_STS_MIN_NORM_2 has the SAME recursion as LCM_STS_FORCE_GRAD -- it calls
# subIntegrator(s, dtauby2) twice per step and takes dtau = traj_length/n_steps
# identically -- with fg_update(s, xi_dtdt, one_minus_2lambda_dt) replaced by a
# plain expSdt(s, one_minus_2lambda_dt).  So swapping it in preserves the level
# structure and every step count exactly and removes ONLY the displaced-field
# evaluation.  That makes it a single-variable test of whether the force-gradient
# displacement is what breaks C3 in Chroma.
#
# lambda MUST be written explicitly: the force-gradient integrator hardcodes
# lambda = 1/6, while LCM_STS_MIN_NORM_2 defaults to the Omelyan value
# 0.1931833275037836.  Leaving it out would change two things at once.
NO_FG_NAME = "LCM_STS_MIN_NORM_2"
NO_FG_LAMBDA = "0.16666666666666666"


def set_levels(root, spec_name, mdsteps, rung_ids, fg_off=()):
    """Rebuild the <Integrator> nesting from a LEVEL_SPECS entry.

    Prints the resolved per-level step counts, because <n_steps> is not the step
    count -- the force-gradient recursion multiplies by 2 at every nesting
    level.  Reading those printed numbers is the check that has twice been
    skipped to this project's cost.
    """
    spec = LEVEL_SPECS[spec_name]
    md = root.find(".//MDIntegrator")
    if md is None:
        sys.exit("ERROR: no <MDIntegrator> found")
    old = md.find("Integrator")
    if old is not None:
        md.remove(old)

    groups = dict(STATIC_GROUPS)
    groups["rungs"] = ["MD::%s" % b for b in rung_ids]
    # Split for the c2rungmid structural probe.  rung_ids is in ladder order, so
    # the last entry is the heaviest rung (has0 = the (-0.2180, -0.1870) ratio),
    # the one whose force is smallest relative to the tail.
    groups["rungs_light"] = groups["rungs"][:-1]
    groups["rung_heavy"] = groups["rungs"][-1:]

    top, parent, totals = None, None, []
    for depth, (n_steps, names) in enumerate(spec):
        lvl = ET.Element("Integrator" if depth == 0 else "SubIntegrator")
        if depth in fg_off:
            ET.SubElement(lvl, "Name").text = NO_FG_NAME
        else:
            ET.SubElement(lvl, "Name").text = INTEGRATOR_NAME
        steps = mdsteps if depth == 0 else n_steps
        ET.SubElement(lvl, "n_steps").text = str(steps)
        if depth in fg_off:
            ET.SubElement(lvl, "lambda").text = NO_FG_LAMBDA
        mids = ET.SubElement(lvl, "monomial_ids")
        for g in names:
            for mid in groups[g]:
                ET.SubElement(mids, "elem").text = mid

        totals.append(mdsteps if depth == 0 else 2 * n_steps * totals[-1])
        if parent is None:
            top = lvl
        else:
            parent.append(lvl)
        parent = lvl

    md.append(top)

    print("  levels: %s" % spec_name)
    for depth, (n_steps, names) in enumerate(spec):
        print("    L%d n_steps=%-2s steps/traj=%-3d  %-28s %s"
              % (depth, mdsteps if depth == 0 else n_steps, totals[depth],
                 ", ".join(names),
                 "%s lambda=%s" % (NO_FG_NAME, NO_FG_LAMBDA)
                 if depth in fg_off else INTEGRATOR_NAME))


# ---------------------------------------------------------------------------
# Solver routing


def mass_block_counts(ladder):
    """How many <InvertParam> blocks carry each ladder mass.

    Every rung holds two (numerator and denominator) and exists as an HMC copy
    and an MD copy, so an interior mass appears 4 times: twice in the rung below
    it, twice in the rung above.  The head mass has no rung above it, hence 2.
    The tail mass reaches 4 via the rung below plus the tail determinant's own
    two copies.  Summed over the selected masses this is asserted against the
    conversion count, so a template change cannot silently half-apply.
    """
    return {m: (2 if i == 0 else 4) for i, m in enumerate(ladder)}


def set_cg_heavy(root, ladder, crossover=CG_CROSSOVER_DEFAULT, maxiter=CG_MAXITER):
    """Adopt Grid's solver routing: multigrid on the light rungs, plain CG past
    the crossover.

    Chroma's baseline runs multigrid at every mass, on two subspaces --
    mg_subspace over the ladder and subspace_mg for the tail -- and so pays
    ~2.3 s of subspace_update_time on every force evaluation, 751.7 s/traj in
    total (39% of its own trajectory).  That per-call cost is flat in mass while
    the solve it accelerates roughly halves toward the heavy end, so there is a
    crossover; Grid puts it after rung 2 and runs QUDA CG at -0.2180 and
    -0.1870.  This converts the matching <InvertParam> blocks in place.

    Selection is by MASS, which catches a rung's <PrecAction> (heatbath) block
    as well as its <Action> (force) one.  Grid runs CG for the heatbath on ALL
    rungs, so this leaves a residual mismatch of two heatbath solves per
    trajectory -- ~0.4% of the trajectory, documented rather than chased.

    Only mg_subspace survives, which is also the one ERASE_QUDA_MULTIGRID_-
    SUBSPACE names -- subspace_mg is used at the tail mass only, so it
    disappears entirely and nothing is left referring to it.

    <MaxIter> is raised: 500 is a GCR *outer* iteration count, and CG at these
    masses and RsdTarget=1e-12 needs a couple of thousand.
    """
    counts = mass_block_counts(ladder)
    targets = sorted(m for m in counts if m >= crossover - 1e-9)
    if not targets:
        sys.exit("ERROR: crossover %s selects no mass (ladder is %s)"
                 % (crossover, sorted(counts)))
    expected = sum(counts[m] for m in targets)

    converted = []
    for ip in root.iter("InvertParam"):
        inv = ip.find("invType")
        mass = ip.find("CloverParams/Mass")
        if inv is None or mass is None:
            continue
        if (inv.text or "").strip() != "QUDA_MULTIGRID_CLOVER_INVERTER":
            continue
        m = float(mass.text)
        if not any(abs(m - target) < 1e-9 for target in targets):
            continue

        inv.text = "QUDA_CLOVER_INVERTER"
        ip.find("SolverType").text = "CG"
        ip.find("MaxIter").text = str(maxiter)
        for tag in ("MULTIGRIDParams", "SubspaceID"):
            node = ip.find(tag)
            if node is not None:
                ip.remove(node)
        converted.append(m)

    if len(converted) != expected:
        sys.exit("ERROR: converted %d InvertParam blocks to CG, expected %d "
                 "(masses seen: %s) -- the template's solver layout has changed"
                 % (len(converted), expected, sorted(set(converted))))

    left = sum(1 for ip in root.iter("InvertParam")
               if (ip.findtext("invType") or "").strip()
               == "QUDA_MULTIGRID_CLOVER_INVERTER")
    print("  cg-heavy: %d blocks -> QUDA_CLOVER_INVERTER/CG at %s; %d MG blocks left"
          % (len(converted), ", ".join("%.4f" % m for m in sorted(set(converted))), left))


def check_monomials(path, expected):
    with open(path) as f:
        text = f.read()
    missing = [m for m in expected if m not in text]
    if missing:
        sys.exit("ERROR: %s is missing monomials: %s" % (path, missing))


# ---------------------------------------------------------------------------


def emit(out, ntraj, mdsteps, tag, ladder=None, levels=None, cg_from=None,
         keep_mg_erase=True, repro=False, reverse=False, fg_off=(), seed=None,
         keep_stout_plaq=False, save_dir=None):
    tree = ET.parse(TEMPLATE)
    root = tree.getroot()

    expected = list(EXPECTED_MONOMIALS)
    rung_ids = None

    if ladder is not None:
        rung_ids = set_ladder(root, ladder)
        expected = [m for m in expected
                    if not m.startswith("HMC::has")
                    or m == "HMC::has_cancel_2flav"
                    or m.split("::")[-1] in rung_ids]

    if levels is not None:
        if rung_ids is None:
            rung_ids = read_ladder(root)[0]
        set_levels(root, levels, mdsteps, rung_ids, fg_off=fg_off)
    elif fg_off:
        sys.exit("ERROR: --fg-off needs an explicit level spec (--levels/--candidate)")
    else:
        inner = set_outer_nsteps(root, mdsteps)
        print("  levels: baseline (outer n_steps=%d, inner n_steps=%s -> %d steps)"
              % (mdsteps, inner, 2 * int(inner) * mdsteps))

    if seed is not None:
        set_seed(root, seed)
    if save_dir is not None and not os.path.isdir(save_dir):
        sys.exit("ERROR: save_dir %s does not exist (Chroma will not create it)" % save_dir)
    set_mccontrol(root, tag, ntraj, repro=repro, reverse=reverse, save_dir=save_dir)
    trim_measurements(root, keep_mg_erase=keep_mg_erase, keep_stout_plaq=keep_stout_plaq)

    if cg_from is not None:
        set_cg_heavy(root, ladder or read_ladder(root)[1], crossover=cg_from)

    tree.write(out, encoding="UTF-8", xml_declaration=True)
    check_monomials(out, expected)
    print("wrote %s  mdsteps=%d ntraj=%d%s%s"
          % (os.path.basename(out), mdsteps, ntraj,
             "  ReproCheckP=true" if repro else "",
             "  ReverseCheckP=true (freq 1)" if reverse else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdsteps", type=int, nargs="+", default=None)
    ap.add_argument("--ntraj", type=int, default=10)
    ap.add_argument("--candidate", choices=sorted(CANDIDATES),
                    help="emit a whole campaign configuration: ladder, levels, "
                         "MDSTEPS and solver routing together (see CANDIDATES)")
    ap.add_argument("--levels", choices=sorted(LEVEL_SPECS),
                    help="override the candidate's level layout, keeping its "
                         "ladder and routing.  With --mdsteps this isolates one "
                         "edit at a time, e.g. --candidate c3 --levels c2 "
                         "--mdsteps 6 is C3's step count with C2's tail "
                         "placement.")
    ap.add_argument("--repro", action="store_true",
                    help="turn on Chroma's ReproCheckP: re-runs the SAME "
                         "trajectory forward from the saved RNG seed and "
                         "compares.  Detects non-determinism, NOT integrator "
                         "error.  Roughly doubles the cost.")
    ap.add_argument("--reverse", action="store_true",
                    help="turn on Chroma's ReverseCheckP (frequency 1): flips "
                         "the momenta, integrates back, and reports "
                         "DeltaDeltaH / DeltaQPerSite / DeltaPPerSite.  This "
                         "is the check that separates an inaccurate-but-valid "
                         "integrator from a broken MD map.  Roughly doubles "
                         "the cost.")
    ap.add_argument("--seed", type=int, nargs=4, metavar="W",
                    help="override the QDP++ RNG seed (4 words). Use to test "
                         "whether a result reproduces or was a one-off.")
    ap.add_argument("--fg-off", type=int, nargs="+", metavar="LEVEL",
                    default=[],
                    help="run the given integrator level(s) with "
                         "LCM_STS_MIN_NORM_2 (lambda=1/6) instead of "
                         "LCM_STS_FORCE_GRAD, dropping the force-gradient "
                         "displacement while keeping every step count "
                         "identical.  0 = outermost.  See NO_FG_NAME.")
    ap.add_argument("--no-mg-erase", action="store_true",
                    help="drop ERASE_QUDA_MULTIGRID_SUBSPACE (see module docstring)")
    ap.add_argument("--cg-heavy", action="store_true",
                    help="Grid-style routing: MG on the light rungs, plain QUDA "
                         "CG at -0.2180 and -0.1870 (see set_cg_heavy)")
    ap.add_argument("--cg-from", type=float, metavar="MASS",
                    help="as --cg-heavy but with the multigrid->CG crossover at "
                         "MASS: every ladder mass >= MASS goes to CG.  Use to "
                         "scan the crossover, e.g. --cg-from -0.2320")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    if not os.path.exists(TEMPLATE):
        sys.exit("missing %s -- run extract_baseline.sh first" % TEMPLATE)
    if not os.path.exists(CFG):
        sys.exit("missing gauge config %s" % CFG)

    if args.candidate:
        c = CANDIDATES[args.candidate]
        # An explicit --mdsteps/--levels overrides the candidate's design point,
        # so one edit can be probed at a time without inventing a new CANDIDATES
        # entry for every diagnostic variant.
        mdsteps = args.mdsteps[0] if args.mdsteps else c["mdsteps"]
        levels = args.levels or c["levels"]
        tag = "%s%s" % (args.candidate, args.suffix)
        emit(os.path.join(HERE, "hmc_%s.ini.xml" % tag),
             args.ntraj, mdsteps, tag,
             ladder=LADDERS[c["ladder"]], levels=levels,
             cg_from=c["cg_from"], keep_mg_erase=not args.no_mg_erase,
             repro=args.repro, reverse=args.reverse,
             fg_off=tuple(args.fg_off), seed=args.seed)
        return

    cg_from = args.cg_from
    if cg_from is None and args.cg_heavy:
        cg_from = CG_CROSSOVER_DEFAULT

    for mds in args.mdsteps or [12, 10, 8]:
        tag = "baseg_mds%d%s" % (mds, args.suffix)
        emit(os.path.join(HERE, "hmc_%s.ini.xml" % tag),
             args.ntraj, mds, tag, cg_from=cg_from,
             keep_mg_erase=not args.no_mg_erase, repro=args.repro,
             reverse=args.reverse, seed=args.seed)


if __name__ == "__main__":
    main()
