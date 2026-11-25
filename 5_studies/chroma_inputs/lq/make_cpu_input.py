#!/usr/bin/env python3
"""Make the lq3 CPU twin of a Perlmutter/lq2 Chroma HMC input: only the solver machinery changes.

    python3 -I make_cpu_input.py SRC.ini.xml OUT.ini.xml

The lq3 CPU Chroma (installs/external/chroma-cpu-lq3, chroma a761e7e, QDP++ + QPhiX + mg_proto, no QUDA)
has no multigrid usable inside an HMC monomial: the monomials' <InvertParam> is created by the MdagM
solver factory (quarkprop4_w.cc, WilsonTypeFermAct::invMdagM), and chroma registers mg_proto only as
LinOp (propagator) solvers. So every solve becomes a QPhiX solve (chroma's QPhiX solvers, built without
link compression, passed the native-vs-QPhiX cross-check, job 1365191, and check || r || / || b ||
against chroma's own operator, aborting above RsdTarget x RsdToleranceFactor):

  QUDA_MULTIGRID_CLOVER_INVERTER  -> QPHIX_CLOVER_INVERTER, SolverType CG (MdagM CG on the even-odd
                                     operator); CloverParams, RsdTarget, RsdToleranceFactor and
                                     AntiPeriodicT kept; MaxIter 100000 (plain CG at -0.2416 needs
                                     > 20,000 iterations, __docs/_SUM.md exceptional-configs phase 0)
  MULTI_CG_QUDA_CLOVER_INVERTER   -> MULTI_CG_QPHIX_CLOVER_INVERTER; CloverParams, RsdTarget (one per
                                     pole), MaxIter, RsdToleranceFactor and AntiPeriodicT kept;
                                     MaxShifts = number of poles
  QUDA_4D_PREDICTOR (MaxChrono n) -> MINIMAL_RESIDUAL_EXTRAPOLATION_4D_PREDICTOR (MaxChrono n): the same
                                     minimal-residual extrapolation of the initial guess QUDA's chrono
                                     does; QUDA_4D_PREDICTOR exists only in QUDA builds
  ERASE_QUDA_MULTIGRID_SUBSPACE   -> removed (inline measurement of the QUDA MG subspace; QUDA builds only)

Everything else is copied byte for byte. The script proves it: with every <InvertParam> and
<ChronologicalPredictor> block and the ERASE_QUDA measurement cut from both files, the rest must be
identical. It also refuses to write a file still naming QUDA, Cuda or MULTIGRID.
"""
import re
import sys

MAXITER_CG = 100000

INV_RE = re.compile(r'(?P<indent>[ \t]*)<InvertParam>\n(?P<body>.*?)\n(?P=indent)</InvertParam>', re.S)
PRED_RE = re.compile(r'(?P<indent>[ \t]*)<ChronologicalPredictor>\n(?P<body>.*?)\n(?P=indent)</ChronologicalPredictor>', re.S)
ERASE_RE = re.compile(r'[ \t]*<elem>\n[ \t]*<Name>ERASE_QUDA_MULTIGRID_SUBSPACE</Name>\n.*?\n[ \t]*</elem>\n', re.S)


def tag(body, name, required=True):
    m = re.search(r'<%s>(.*?)</%s>' % (name, name), body, re.S)
    if m is None:
        if required:
            sys.exit('ERROR: <%s> missing in an InvertParam block' % name)
        return None
    return m.group(1)


def clover_params(body):
    m = re.search(r'([ \t]*)<CloverParams>\n.*?\n\1</CloverParams>', body, re.S)
    if m is None:
        sys.exit('ERROR: no <CloverParams> block in an InvertParam block')
    return m.group(0)


def reindent(block, old, new):
    return '\n'.join(new + line[len(old):] if line.startswith(old) else line for line in block.split('\n'))


counts = {'mg': 0, 'multi': 0, 'pred': 0, 'erase': 0}


def new_invert(m):
    ind, body = m.group('indent'), m.group('body')
    inner = ind + '  '
    inv = tag(body, 'invType')
    cp = clover_params(body)
    cp_ind = re.match(r'[ \t]*', cp).group(0)
    cp = reindent(cp, cp_ind, inner)
    if inv == 'QUDA_MULTIGRID_CLOVER_INVERTER':
        counts['mg'] += 1
        lines = ['<invType>QPHIX_CLOVER_INVERTER</invType>', None,
                 '<RsdTarget>%s</RsdTarget>' % tag(body, 'RsdTarget'),
                 '<MaxIter>%d</MaxIter>' % MAXITER_CG,
                 '<RsdToleranceFactor>%s</RsdToleranceFactor>' % tag(body, 'RsdToleranceFactor'),
                 '<AntiPeriodicT>%s</AntiPeriodicT>' % tag(body, 'AntiPeriodicT'),
                 '<SolverType>CG</SolverType>',
                 '<Verbose>false</Verbose>']
    elif inv == 'MULTI_CG_QUDA_CLOVER_INVERTER':
        counts['multi'] += 1
        rsd = tag(body, 'RsdTarget')
        lines = ['<invType>MULTI_CG_QPHIX_CLOVER_INVERTER</invType>', None,
                 '<RsdTarget>%s</RsdTarget>' % rsd,
                 '<MaxIter>%s</MaxIter>' % tag(body, 'MaxIter'),
                 '<MaxShifts>%d</MaxShifts>' % len(rsd.split()),
                 '<RsdToleranceFactor>%s</RsdToleranceFactor>' % tag(body, 'RsdToleranceFactor'),
                 '<AntiPeriodicT>%s</AntiPeriodicT>' % tag(body, 'AntiPeriodicT'),
                 '<Verbose>false</Verbose>']
    else:
        return m.group(0)          # any other solver: untouched
    out = [ind + '<InvertParam>']
    for l in lines:
        out.append(cp if l is None else inner + l)
    out.append(ind + '</InvertParam>')
    return '\n'.join(out)


def new_pred(m):
    ind, body = m.group('indent'), m.group('body')
    if tag(body, 'Name') != 'QUDA_4D_PREDICTOR':
        return m.group(0)
    counts['pred'] += 1
    inner = ind + '  '
    return '\n'.join([ind + '<ChronologicalPredictor>',
                      inner + '<Name>MINIMAL_RESIDUAL_EXTRAPOLATION_4D_PREDICTOR</Name>',
                      inner + '<MaxChrono>%s</MaxChrono>' % tag(body, 'MaxChrono'),
                      ind + '</ChronologicalPredictor>'])


def strip_solvers(text):
    text = ERASE_RE.sub('', text)
    text = INV_RE.sub('<InvertParam/>', text)
    return PRED_RE.sub('<ChronologicalPredictor/>', text)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        old = f.read()
    new, n = ERASE_RE.subn('', old)
    counts['erase'] = n
    new = INV_RE.sub(new_invert, new)
    new = PRED_RE.sub(new_pred, new)
    if strip_solvers(old) != strip_solvers(new):
        sys.exit('ERROR: the files differ outside the solver blocks; nothing written')
    left = re.findall(r'.*(?:QUDA|Cuda|MULTIGRID).*', new)
    if left:
        sys.exit('ERROR: QUDA items left, nothing written:\n' + '\n'.join(left[:10]))
    with open(dst, 'w') as f:
        f.write(new)
    print('wrote %s from %s' % (dst, src))
    print('  QUDA_MULTIGRID_CLOVER_INVERTER -> QPHIX_CLOVER_INVERTER (CG, MaxIter %d): %d blocks'
          % (MAXITER_CG, counts['mg']))
    print('  MULTI_CG_QUDA_CLOVER_INVERTER -> MULTI_CG_QPHIX_CLOVER_INVERTER: %d blocks' % counts['multi'])
    print('  QUDA_4D_PREDICTOR -> MINIMAL_RESIDUAL_EXTRAPOLATION_4D_PREDICTOR: %d blocks' % counts['pred'])
    print('  ERASE_QUDA_MULTIGRID_SUBSPACE measurements removed: %d' % counts['erase'])
    print('  outside these blocks the files are identical (checked)')


if __name__ == '__main__':
    main()
