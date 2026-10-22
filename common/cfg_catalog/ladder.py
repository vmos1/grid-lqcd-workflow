"""Compact readout of a Chroma HMC input: action parameters, monomials, integrator.

usage: python3 ladder.py <xml> [<xml> ...]

One line per monomial: id, type, Action mass, PrecAction (Hasenbusch) mass,
rational power/degree and pseudofermion count; then the integrator tree.
"""
import sys
import xml.etree.ElementTree as ET


def vals(node, tag):
    return sorted({(x.text or "").strip() for x in node.iter(tag)})


def walk(node, depth):
    if node is None:
        return
    ids = [e.text for e in node.findall("monomial_ids/elem")]
    print(f"  {'  ' * depth}integrator {node.findtext('Name')} n_steps {node.findtext('n_steps')}: {ids}")
    walk(node.find("SubIntegrator"), depth + 1)


for fn in sys.argv[1:]:
    h = ET.parse(fn).getroot().find("HMCTrj")
    print(f"\n### {fn}")
    print("  beta:", vals(h, "beta"), " u0:", vals(h, "u0"), " csw:", vals(h, "clovCoeff"),
          " rho:", vals(h, "rho"), " n_smear:", vals(h, "n_smear"), " bc:", vals(h, "boundary"))
    for m in h.find("Monomials"):
        mid = m.findtext("NamedObject/monomial_id")
        act = [x.text.strip() for x in m.findall("Action/FermionAction/Mass")] or \
              [x.text.strip() for x in m.findall("FermionAction/Mass")]
        prec = [x.text.strip() for x in m.findall("PrecAction/FermionAction/Mass")]
        ai = m.find(".//ApproxInfo")
        extra = f" power {ai.findtext('numPower')}/{ai.findtext('denPower')} degree {ai.findtext('degree')}" if ai is not None else ""
        nf = m.findtext("num_flavors") or ""
        npf = m.findtext("num_pf") or ""
        print(f"  {mid:24s} {m.findtext('Name'):52s} mass {act} prec {prec}{extra}"
              f"{' nf=' + nf if nf else ''}{' npf=' + npf if npf else ''}")
    integ = h.find("MDIntegrator")
    print("  tau0:", integ.findtext("tau0"))
    walk(integ.find("Integrator"), 0)
