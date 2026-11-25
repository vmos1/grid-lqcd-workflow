#!/usr/bin/env python3
"""Make the lq3 CPU QOP-MG twin of a Perlmutter/lq2 Chroma HMC input: only the solver machinery changes.

    python3 -I make_cpu_input_qopmg.py SRC.ini.xml OUT.ini.xml [options]

Companion of make_cpu_input.py (QPhiX CG everywhere). Here the QUDA multigrid solves go to USQCD's QOPQDP
Wilson-clover multigrid (chroma's QOP_CLOVER_MULTIGRID_INVERTER, lib/actions/ferm/invert/qop_mg/, built with
BUILD_QOP_MG), the only CPU multigrid chroma accepts as an MdagM (HMC) solver; the rest is as make_cpu_input.py.

  QUDA_MULTIGRID_CLOVER_INVERTER  -> QOP_CLOVER_MULTIGRID_INVERTER (MdagM: two full-lattice D solves with the
                                     gamma5 trick, odd sites kept; chroma checks || chi - M^dag M psi || / || chi ||
                                     with its own operator and aborts above Residual x RsdToleranceFactor)
       Mass, clovCoeff            -> Mass, Clover (CloverT = Clover, AnisoXi = AnisoNu = 1; anisotropic refused)
       RsdTarget, RsdToleranceFactor -> Residual, RsdToleranceFactor (same meaning: final M^dag M residual)
       MaxIter (QUDA GCR, per solve) -> MaxIter = 2 x (chroma gives each of the two D solves MaxIter/2)
       ThresholdCount             -> RefreshThreshold (iterations of one D solve at which the subspace is erased
                                     and rebuilt from scratch; a failed solve is retried once on the new space)
       SubspaceID                 -> ExternalSubspace true, SubspaceId (one setup shared by every block with it)
       (none)                     -> MassCrit = lightest Mass among the blocks sharing the SubspaceId: null
                                     vectors and coarse operators are built at that mass, whichever solve
                                     creates the space (QOP cannot re-coarsen at another mass)
       Blocking, NullVectors      -> Levels = number of coarse levels, Blocking, NumNullVecs (chirally split)
       Pre/Post-SmootherApplications -> NumPreHits, NumPostHits (QOP smoother: GCR on the even-odd operator)
       RelaxationOmegaMG          -> Underrelax
       CoarseResidual, MaxCoarseIterations (QUDA levels 1..) -> CoarseResidual, CoarseMaxIter
       PrecondGCRNKrylov, bottom MaxCoarseIterations -> CoarseNumGCRVecs (bottom: the CA-GCR basis = its iters)
       OuterGCRNKrylov            -> NumGCRVecs
       MaxIterSubspaceCreate      -> NullMaxIter (cap of each BiCGStab pass of QOP's inverse-iteration setup)
       (none)                     -> NullResidual 0.4, NullConvergence 0.5, NumExtraVecs 0 (QOP defaults; QUDA's
                                     RsdTargetSubspaceCreate 5e-6 is a CG tolerance with no QOP equivalent)
       dropped (no QOP equivalent): Precision, Reconstruct, Cuda*, Delta, Pipeline, SolverType, AsymmetricLinop,
                                     SmootherType/Tol, CoarseSolverType, CycleType, SchwarzType, SetupOnGPU,
                                     SubspaceSolver, RsdTargetSubspaceCreate, MaxIterSubspaceRefresh,
                                     GenerateNullspace, GenerateAllLevels, CheckMultigridSetup, SolutionCheckP,
                                     AxialGaugeFix, AutotuneDslash, RelaxationOmegaOuter
  QUDA_4D_PREDICTOR (MaxChrono n) -> on a monomial whose force solve is QOP-MG: ZERO_GUESS_4D_PREDICTOR (default,
                                     --predictor). The QOP MdagM solver aborts unless the predictor is two-step;
                                     its MRE path projects through Lunprec on the odd subset only (the clover
                                     block, not the Schur operator) and its Y guess is gamma5 times the stored
                                     solution, so MRE/LAST give no better guesses than zero (LAST helps X only).
                                     Elsewhere: MINIMAL_RESIDUAL_EXTRAPOLATION_4D_PREDICTOR (MaxChrono n), as
                                     make_cpu_input.py.
  ERASE_QUDA_MULTIGRID_SUBSPACE   -> ERASE_MG_SUBSPACE for every QOP SubspaceId (same Frequency): QOP builds its
                                     coarse operators once at setup and never re-coarsens when the links change
                                     (QUDA does), so every space is rebuilt at least once per trajectory.
  MULTI_CG_QUDA_CLOVER_INVERTER   -> MULTI_CG_QPHIX_CLOVER_INVERTER, exactly as make_cpu_input.py.
  --cg-mass M                     -> the QUDA MG blocks at Mass M become QPhiX CG blocks exactly as
                                     make_cpu_input.py (MaxIter 100000), e.g. heavy rungs if CG wins there.

Everything else is copied byte for byte. The script proves it: with every <InvertParam> and
<ChronologicalPredictor> block and the ERASE_* measurements cut from both files, the rest must be identical.
It refuses to write a file still naming QUDA, Cuda or MULTIGRIDParams, blocks sharing a SubspaceId with
different multigrid settings, or a RefreshThreshold above MaxIter/2 (a failed solve would then abort
instead of being retried on a new subspace).

Options:
  --predictor zero|last|mre   predictor of monomials whose force solve is QOP-MG (default zero)
  --refresh OLD=NEW           RefreshThreshold NEW where the source has ThresholdCount OLD (repeatable)
  --split-subspaces           one SubspaceId per (SubspaceID, Mass): every mass gets its own setup at its
                              own mass, instead of sharing one built at the lightest
  --cg-mass M                 route the QUDA MG blocks at Mass M to QPhiX CG (repeatable)
  --set NAME=VALUE            override a QOP parameter in every QOP block, e.g. --set 'NullResidual=0.2 0.2'
                              or --set Verbose=1 (repeatable; not Mass, MassCrit, SubspaceId, Blocking)
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET

MAXITER_CG = 100000          # QPhiX CG blocks, as make_cpu_input.py
NULL_RESIDUAL = '0.4'        # QOP defaults (wilsonmg.c / QOPQDP init_level)
NULL_CONVERGENCE = '0.5'
NUM_EXTRA_VECS = '0'
QOP_VERBOSE = '0'

INV_RE = re.compile(r'(?P<indent>[ \t]*)<InvertParam>\n(?P<body>.*?)\n(?P=indent)</InvertParam>', re.S)
PRED_RE = re.compile(r'(?P<indent>[ \t]*)<ChronologicalPredictor>\n(?P<body>.*?)\n(?P=indent)</ChronologicalPredictor>', re.S)
ERASE_RE = re.compile(r'(?P<indent>[ \t]*)<elem>\n[ \t]*<Name>(?P<name>ERASE_QUDA_MULTIGRID_SUBSPACE|ERASE_MG_SUBSPACE)'
                      r'</Name>\n(?P<body>.*?)\n(?P=indent)</elem>\n', re.S)
MONO_RE = re.compile(r'(?P<indent>[ \t]*)<elem>\n(?P=indent)  <Name>(?P<name>\w+_MONOMIAL)</Name>\n(?P<body>.*?)\n'
                     r'(?P=indent)</elem>', re.S)
PREDICTORS = {'zero': 'ZERO_GUESS_4D_PREDICTOR', 'last': 'LAST_SOLUTION_4D_PREDICTOR',
              'mre': 'MINIMAL_RESIDUAL_EXTRAPOLATION_4D_PREDICTOR'}
SETTABLE = ('Clover', 'Residual', 'MaxIter', 'RefreshThreshold', 'RsdToleranceFactor', 'TerminateOnFail',
            'NumGCRVecs', 'Verbose', 'NumNullVecs', 'NullMaxIter', 'NullResidual', 'NullConvergence',
            'NumExtraVecs', 'Underrelax', 'NumPreHits', 'NumPostHits', 'CoarseNumGCRVecs', 'CoarseMaxIter',
            'CoarseResidual')
# Settings that must agree between blocks sharing one subspace (QOP builds the space from the first solve's
# values and re-applies the current solve's values to the shared structure).
SHARED = ('Clover', 'MassCrit', 'Levels', 'Blocking', 'NumNullVecs', 'NullMaxIter', 'NullResidual',
          'NullConvergence', 'NumExtraVecs', 'Underrelax', 'NumPreHits', 'NumPostHits', 'CoarseNumGCRVecs',
          'CoarseMaxIter', 'CoarseResidual', 'NumGCRVecs')


def die(msg):
    sys.exit('ERROR: ' + msg + '; nothing written')


def tag(body, name, required=True):
    m = re.search(r'<%s>(.*?)</%s>' % (name, name), body, re.S)
    if m is None:
        if required:
            die('<%s> missing in an InvertParam block' % name)
        return None
    return m.group(1)


def clover_params(body):
    m = re.search(r'([ \t]*)<CloverParams>\n.*?\n\1</CloverParams>', body, re.S)
    if m is None:
        die('no <CloverParams> block in an InvertParam block')
    return m.group(0)


def reindent(block, old, new):
    return '\n'.join(new + line[len(old):] if line.startswith(old) else line for line in block.split('\n'))


def words(node, name):
    e = node.find(name)
    if e is None or e.text is None:
        die('<%s> missing in a MULTIGRIDParams block' % name)
    return e.text.split()


# ---- the QPhiX blocks: byte for byte what make_cpu_input.py writes ----------------------------------------
def qphix_block(ind, body, inv):
    inner = ind + '  '
    cp = clover_params(body)
    cp = reindent(cp, re.match(r'[ \t]*', cp).group(0), inner)
    if inv == 'QUDA_MULTIGRID_CLOVER_INVERTER':
        lines = ['<invType>QPHIX_CLOVER_INVERTER</invType>', None,
                 '<RsdTarget>%s</RsdTarget>' % tag(body, 'RsdTarget'),
                 '<MaxIter>%d</MaxIter>' % MAXITER_CG,
                 '<RsdToleranceFactor>%s</RsdToleranceFactor>' % tag(body, 'RsdToleranceFactor'),
                 '<AntiPeriodicT>%s</AntiPeriodicT>' % tag(body, 'AntiPeriodicT'),
                 '<SolverType>CG</SolverType>',
                 '<Verbose>false</Verbose>']
    else:
        rsd = tag(body, 'RsdTarget')
        lines = ['<invType>MULTI_CG_QPHIX_CLOVER_INVERTER</invType>', None,
                 '<RsdTarget>%s</RsdTarget>' % rsd,
                 '<MaxIter>%s</MaxIter>' % tag(body, 'MaxIter'),
                 '<MaxShifts>%d</MaxShifts>' % len(rsd.split()),
                 '<RsdToleranceFactor>%s</RsdToleranceFactor>' % tag(body, 'RsdToleranceFactor'),
                 '<AntiPeriodicT>%s</AntiPeriodicT>' % tag(body, 'AntiPeriodicT'),
                 '<Verbose>false</Verbose>']
    out = [ind + '<InvertParam>']
    for l in lines:
        out.append(cp if l is None else inner + l)
    out.append(ind + '</InvertParam>')
    return '\n'.join(out)


# ---- the QOP-MG blocks ---------------------------------------------------------------------------------------
def qop_params(body):
    """QOP-MG parameters (strings, in output order) mapped from one QUDA MG InvertParam body."""
    root = ET.fromstring('<InvertParam>\n' + body + '\n</InvertParam>')
    cp = root.find('CloverParams')
    if cp is None or cp.find('Mass') is None or cp.find('clovCoeff') is None:
        die('QUDA MG block without CloverParams/Mass or clovCoeff')
    aniso = cp.find('AnisoParam/anisoP')
    if aniso is not None and aniso.text.strip() != 'false':
        die('anisotropic CloverParams: QOP-MG anisotropy conventions not mapped')
    mg = root.find('MULTIGRIDParams')
    if mg is None:
        die('QUDA_MULTIGRID_CLOVER_INVERTER block without MULTIGRIDParams')
    blocking = [' '.join(e.text.split()) for e in mg.find('Blocking').findall('elem')]
    nlev = len(blocking)                              # coarse levels; QUDA levels = nlev + 1
    nvec = words(mg, 'NullVectors')
    npre = words(mg, 'Pre-SmootherApplications')
    npost = words(mg, 'Post-SmootherApplications')
    omega = words(mg, 'RelaxationOmegaMG')
    cres = words(mg, 'CoarseResidual')               # QUDA per level, [0] = fine (unused as a coarse solver)
    cmax = words(mg, 'MaxCoarseIterations')
    nkp = mg.find('PrecondGCRNKrylov').text.strip()
    nko = mg.find('OuterGCRNKrylov').text.strip()
    nullmax = words(mg, 'MaxIterSubspaceCreate')
    for name, v, n in (('NullVectors', nvec, nlev), ('Pre-SmootherApplications', npre, nlev),
                       ('Post-SmootherApplications', npost, nlev), ('RelaxationOmegaMG', omega, nlev),
                       ('CoarseResidual', cres, nlev + 1), ('MaxCoarseIterations', cmax, nlev + 1),
                       ('MaxIterSubspaceCreate', nullmax, nlev)):
        if len(v) < n:
            die('%s has %d values, %d levels need %d' % (name, len(v), nlev + 1, n))
    subspace = tag(body, 'SubspaceID')
    thr = tag(body, 'ThresholdCount')
    p = [('Mass', root.find('CloverParams/Mass').text.strip()),
         ('MassCrit', None),                                         # filled in per subspace
         ('Clover', root.find('CloverParams/clovCoeff').text.strip()),
         ('AnisoXi', '1.0'), ('AnisoNu', '1.0'),
         ('Residual', tag(body, 'RsdTarget')),
         ('MaxIter', str(2 * int(tag(body, 'MaxIter')))),
         ('RefreshThreshold', thr),
         ('RsdToleranceFactor', tag(body, 'RsdToleranceFactor')),
         ('TerminateOnFail', 'true'),
         ('NumGCRVecs', nko),
         ('Verbose', QOP_VERBOSE),
         ('Levels', str(nlev)),
         ('Blocking', blocking),
         ('NumNullVecs', ' '.join(nvec[:nlev])),
         ('NullMaxIter', ' '.join(nullmax[:nlev])),
         ('NullResidual', ' '.join([NULL_RESIDUAL] * nlev)),
         ('NullConvergence', ' '.join([NULL_CONVERGENCE] * nlev)),
         ('NumExtraVecs', ' '.join([NUM_EXTRA_VECS] * nlev)),
         ('Underrelax', ' '.join(omega[:nlev])),
         ('NumPreHits', ' '.join(npre[:nlev])),
         ('NumPostHits', ' '.join(npost[:nlev])),
         ('CoarseNumGCRVecs', ' '.join([nkp] * (nlev - 1) + [cmax[nlev]])),
         ('CoarseMaxIter', ' '.join(cmax[1:nlev + 1])),
         ('CoarseResidual', ' '.join(cres[1:nlev + 1])),
         ('ExternalSubspace', 'true'),
         ('SubspaceId', subspace)]
    return p, thr


def qop_block(ind, p):
    i1, i2 = ind + '  ', ind + '    '
    out = [ind + '<InvertParam>', i1 + '<invType>QOP_CLOVER_MULTIGRID_INVERTER</invType>']
    for k, v in p:
        if k == 'Blocking':
            out.append(i1 + '<Blocking>')
            out += [i2 + '<elem>%s</elem>' % b for b in v]
            out.append(i1 + '</Blocking>')
        else:
            out.append(i1 + '<%s>%s</%s>' % (k, v, k))
    out.append(ind + '</InvertParam>')
    return '\n'.join(out)


def parent_tag(text, pos, ind):
    """Opening tag enclosing the block that starts at pos (pretty-printed XML, 2-space steps)."""
    want = len(ind) - 2
    for line in reversed(text[:pos].split('\n')):
        s = line.lstrip(' ')
        if len(line) - len(s) == want and re.match(r'<\w+>$', s):
            return s[1:-1]
    return 'monomial'


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--predictor', choices=sorted(PREDICTORS), default='zero')
    ap.add_argument('--refresh', action='append', default=[])
    ap.add_argument('--split-subspaces', action='store_true')
    ap.add_argument('--cg-mass', action='append', default=[])
    ap.add_argument('--set', action='append', default=[])
    if len(sys.argv) < 3 or sys.argv[1] in ('-h', '--help'):
        sys.exit(__doc__)
    a = ap.parse_args()
    refresh = {}
    for r in a.refresh:
        old, _, new = r.partition('=')
        if not (old.isdigit() and new.isdigit()):
            die('--refresh wants OLD=NEW integers, got %r' % r)
        refresh[old] = new
    sets = []
    for s in a.set:
        k, _, v = s.partition('=')
        if k not in SETTABLE or not v.strip():
            die('--set %r: NAME must be one of %s' % (s, ', '.join(SETTABLE)))
        sets.append((k, ' '.join(v.split())))
    cg_masses = set(float(m) for m in a.cg_mass)

    with open(a.src) as f:
        old = f.read()

    # Pass 1: every QUDA MG block routed to QOP, its subspace and the lightest mass of each subspace.
    def route(body):
        if tag(body, 'invType') != 'QUDA_MULTIGRID_CLOVER_INVERTER':
            return None
        return 'cg' if float(tag(body, 'Mass')) in cg_masses else 'qop'

    def space_id(p):
        d = dict(p)
        if a.split_subspaces:
            return '%s_m%s' % (d['SubspaceId'], d['Mass'].lstrip('-').replace('.', 'p'))
        return d['SubspaceId']

    def finish(p):
        p = [(k, (dict(sets)[k] if k in dict(sets) else v)) for k, v in p]
        sid = space_id(p)
        return [(k, sid if k == 'SubspaceId' else v) for k, v in p], sid

    masscrit = {}
    for m in INV_RE.finditer(old):
        if route(m.group('body')) == 'qop':
            p, _ = finish(qop_params(m.group('body'))[0])
            sid, mass = dict(p)['SubspaceId'], dict(p)['Mass']
            if sid not in masscrit or float(mass) < float(masscrit[sid]):
                masscrit[sid] = mass
    if not masscrit:
        die('no QUDA_MULTIGRID_CLOVER_INVERTER block to convert')

    # Pass 2: rewrite monomial by monomial (the predictor follows the monomial's first, i.e. force, solve).
    rows, shared, counts = [], {}, {'qop': 0, 'cg': 0, 'multi': 0, 'pred': 0}

    def new_invert(m, mono_id, mono_body, first):
        ind, body = m.group('indent'), m.group('body')
        inv = tag(body, 'invType')
        where = parent_tag(mono_body, m.start(), ind)
        kind = route(body)
        if inv == 'MULTI_CG_QUDA_CLOVER_INVERTER':
            counts['multi'] += 1
            rows.append((mono_id, where, tag(body, 'Mass'), 'MULTI_CG_QPHIX', '', '', tag(body, 'RsdTarget').split()[0], '', ''))
            out = qphix_block(ind, body, inv)
        elif kind == 'cg':
            counts['cg'] += 1
            rows.append((mono_id, where, tag(body, 'Mass'), 'QPHIX_CG', '', '', tag(body, 'RsdTarget'), str(MAXITER_CG), ''))
            out = qphix_block(ind, body, inv)
        elif kind == 'qop':
            counts['qop'] += 1
            p, thr = qop_params(body)
            if thr in refresh:
                p = [(k, refresh[thr] if k == 'RefreshThreshold' else v) for k, v in p]
            p, sid = finish(p)
            p = [(k, masscrit[sid] if k == 'MassCrit' else v) for k, v in p]
            d = dict(p)
            if int(d['RefreshThreshold']) > int(d['MaxIter']) // 2:
                die('%s %s: RefreshThreshold %s > MaxIter/2 = %d (a failed solve would abort, not retry)'
                    % (mono_id, where, d['RefreshThreshold'], int(d['MaxIter']) // 2))
            key = tuple((k, str(d[k])) for k in SHARED)
            if shared.setdefault(sid, key) != key:
                die('blocks sharing SubspaceId %s differ in multigrid settings' % sid)
            rows.append((mono_id, where, d['Mass'], 'QOP_MG', sid, d['MassCrit'], d['Residual'], d['MaxIter'],
                         d['RefreshThreshold']))
            out = qop_block(ind, p)
        else:
            return m.group(0)                        # any other solver: untouched
        if not first:
            first.append(kind or 'multi')
        return out

    def new_pred(m, first):
        ind, body = m.group('indent'), m.group('body')
        if tag(body, 'Name') != 'QUDA_4D_PREDICTOR':
            return m.group(0)
        counts['pred'] += 1
        inner = ind + '  '
        if first and first[0] == 'qop':
            name = PREDICTORS[a.predictor]
        else:
            name = PREDICTORS['mre']
        lines = [ind + '<ChronologicalPredictor>', inner + '<Name>%s</Name>' % name]
        if name == PREDICTORS['mre']:
            lines.append(inner + '<MaxChrono>%s</MaxChrono>' % tag(body, 'MaxChrono'))
        lines.append(ind + '</ChronologicalPredictor>')
        return '\n'.join(lines)

    def new_monomial(m):
        ind, body = m.group('indent'), m.group('body')
        mono_id = tag(body, 'monomial_id', required=False) or m.group('name')
        first = []
        body = INV_RE.sub(lambda mm: new_invert(mm, mono_id, m.group('body'), first), body)
        body = PRED_RE.sub(lambda mm: new_pred(mm, first), body)
        return '%s<elem>\n%s  <Name>%s</Name>\n%s\n%s</elem>' % (ind, ind, m.group('name'), body, ind)

    new = MONO_RE.sub(new_monomial, old)

    # ERASE: one ERASE_MG_SUBSPACE per QOP subspace, where the QUDA one was (or at the end of the list).
    erase_src = [e for e in ERASE_RE.finditer(new) if e.group('name') == 'ERASE_QUDA_MULTIGRID_SUBSPACE']
    if len(erase_src) > 1:
        die('more than one ERASE_QUDA_MULTIGRID_SUBSPACE measurement')
    if erase_src:
        e = erase_src[0]
        ind, freq = e.group('indent'), tag(e.group('body'), 'Frequency')
    else:
        mm = re.search(r'\n([ \t]*)</InlineMeasurements>', new)
        if mm is None:
            die('no <InlineMeasurements> to add ERASE_MG_SUBSPACE to')
        ind, freq = mm.group(1) + '  ', '1'
    erase = ''.join('%s<elem>\n%s  <Name>ERASE_MG_SUBSPACE</Name>\n%s  <Frequency>%s</Frequency>\n'
                    '%s  <NamedObject>\n%s    <object_id>%s</object_id>\n%s  </NamedObject>\n%s</elem>\n'
                    % (ind, ind, ind, freq, ind, ind, sid, ind, ind) for sid in sorted(masscrit))
    if erase_src:
        new = new[:erase_src[0].start()] + erase + new[erase_src[0].end():]
    else:
        new = new[:mm.start() + 1] + erase + new[mm.start() + 1:]

    # Proof: identical outside the solver blocks and the ERASE measurements.
    def strip_solvers(text):
        text = ERASE_RE.sub('', text)
        text = INV_RE.sub('<InvertParam/>', text)
        return PRED_RE.sub('<ChronologicalPredictor/>', text)

    if strip_solvers(old) != strip_solvers(new):
        die('the files differ outside the solver blocks')
    left = re.findall(r'.*(?:QUDA|Cuda|MULTIGRIDParams).*', new)
    if left:
        die('QUDA items left:\n' + '\n'.join(left[:10]))
    n_src = len([1 for m in INV_RE.finditer(old) if tag(m.group('body'), 'invType') == 'QUDA_MULTIGRID_CLOVER_INVERTER'])
    if counts['qop'] + counts['cg'] != n_src:
        die('%d QUDA MG blocks in the source, %d converted (one outside a monomial?)' % (n_src, counts['qop'] + counts['cg']))
    try:
        ET.fromstring(new.split('?>', 1)[1] if new.startswith('<?xml') else new)
    except ET.ParseError as err:
        die('output is not well-formed XML: %s' % err)

    with open(a.dst, 'w') as f:
        f.write(new)
    print('wrote %s from %s' % (a.dst, a.src))
    print('  QUDA_MULTIGRID_CLOVER_INVERTER -> QOP_CLOVER_MULTIGRID_INVERTER: %d blocks, -> QPhiX CG: %d blocks'
          % (counts['qop'], counts['cg']))
    print('  MULTI_CG_QUDA_CLOVER_INVERTER -> MULTI_CG_QPHIX_CLOVER_INVERTER: %d blocks' % counts['multi'])
    print('  QUDA_4D_PREDICTOR -> %s on QOP-MG force solves, MRE elsewhere: %d blocks'
          % (PREDICTORS[a.predictor], counts['pred']))
    print('  ERASE_QUDA_MULTIGRID_SUBSPACE -> ERASE_MG_SUBSPACE (Frequency %s): %s' % (freq, ' '.join(sorted(masscrit))))
    print('  subspaces (MassCrit): %s' % ', '.join('%s (%s)' % (k, masscrit[k]) for k in sorted(masscrit)))
    if sets:
        print('  --set: %s' % ', '.join('%s=%s' % kv for kv in sets))
    print('  outside these blocks the files are identical (checked)')
    print('  %-24s %-10s %-8s %-15s %-14s %-8s %-9s %-7s %s' % ('monomial', 'block', 'Mass', 'solver', 'SubspaceId',
                                                                 'MassCrit', 'Residual', 'MaxIter', 'Refresh'))
    for r in rows:
        print('  %-24s %-10s %-8s %-15s %-14s %-8s %-9s %-7s %s' % r)


if __name__ == '__main__':
    main()
