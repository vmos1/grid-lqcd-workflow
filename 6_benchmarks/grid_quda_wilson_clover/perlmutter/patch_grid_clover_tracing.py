#!/usr/bin/env python3
"""Add the missing GRID_TRACE annotations to Grid's compact Wilson-clover operator.

Grid's shipped tracing (--enable-tracing=nvtx) annotates the Wilson hopping term
thoroughly -- HaloExchange, Gather, Merge, DhopInterior, DhopExterior -- and the
CG and vector algebra.  It annotates the CLOVER term not at all.  Since Wilson is
at parity between Grid and QUDA and the whole measured gap is the clover term,
the shipped tracing covers everything except the part that matters.

This adds four one-line macros at the four distinct entry points:

    Meooe     -> CloverMeooe        hopping term, as seen by the clover operator
    MeooeDag  -> CloverMeooeDag
    Mooee     -> CloverMooee         the clover term itself
    MooeeInv  -> CloverMooeeInv      and its inverse

MooeeDag and MooeeInvDag are deliberately NOT annotated: they simply delegate to
Mooee and MooeeInv (the blocks are hermitian), so annotating them would nest a
range inside an identical one.  MooeeInternal is likewise left alone -- all four
Mooee variants funnel through it, so a range there would nest inside these and
double-count when regions are summed.

With these, an even-odd Schur solve decomposes cleanly: each M_pc application is
two CloverMeooe plus one CloverMooeeInv plus one CloverMooee.

It ALSO fixes Grid/perfmon/Tracing.h, whose `#include <nvToolsExt.h>` does not
resolve on CUDA >= 12.9: that header now ships only under nvtx3/.  The include
must be corrected rather than worked around with -I<cuda>/include/nvtx3 -- that
puts the whole nvtx3 directory on the include path, and Thrust pulls in
<nvtx3/nvtx3.hpp> for its own ranges; the C++ wrapper then finds the C API
already included outside its versioning context and the build dies in nvtx3.hpp
with "nvtxColorType_t is undefined" and similar.  (Learned the hard way.)

The third break on CUDA >= 12.9 -- configure.ac appending a -lnvToolsExt that no
longer exists -- is handled by build_grid_mainline.sh's `patch-tracing` action,
since it lives in the generated configure rather than in a source file.

Idempotent: re-running on an already-patched tree is a no-op.

Usage:  patch_grid_clover_tracing.py <path to Grid source tree>
"""

import io
import os
import sys

REL = 'Grid/qcd/action/fermion/implementation/CompactWilsonCloverFermionImplementation.h'
TRACING = 'Grid/perfmon/Tracing.h'
ILDG = 'Grid/parallelIO/IldgIO.h'

# (function signature fragment, trace label)
SITES = [
    ('CompactWilsonCloverFermion<Impl, CloverHelpers>::Meooe(const FermionField& in, FermionField& out) {',
     'CloverMeooe'),
    ('CompactWilsonCloverFermion<Impl, CloverHelpers>::MeooeDag(const FermionField& in, FermionField& out) {',
     'CloverMeooeDag'),
    ('CompactWilsonCloverFermion<Impl, CloverHelpers>::Mooee(const FermionField& in, FermionField& out) {',
     'CloverMooee'),
    ('CompactWilsonCloverFermion<Impl, CloverHelpers>::MooeeInv(const FermionField& in, FermionField& out) {',
     'CloverMooeeInv'),
]


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: %s <grid source tree>' % os.path.basename(sys.argv[0]))
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        sys.exit('ERROR: not found: %s' % path)

    lines = io.open(path, encoding='utf-8').read().split('\n')
    added, already = 0, 0

    for sig, label in SITES:
        hits = [i for i, l in enumerate(lines) if sig in l]
        if len(hits) != 1:
            sys.exit('ERROR: expected exactly one match for %r, found %d' % (label, len(hits)))
        i = hits[0]
        if 'GRID_TRACE' in lines[i + 1]:
            already += 1
            continue
        lines.insert(i + 1, '  GRID_TRACE("%s");' % label)
        added += 1

    if added:
        io.open(path, 'w', encoding='utf-8').write('\n'.join(lines))
    print('%s: %d annotation(s) added, %d already present' % (REL, added, already))

    fix_tracing_header(os.path.join(sys.argv[1], TRACING))
    drop_ildg_lfn_assert(os.path.join(sys.argv[1], ILDG))


def drop_ildg_lfn_assert(path):
    """Remove IldgIO.h's hard assert on the ildg-LFN record.

    ⚠️ NOT part of the tracing work and NOT for upstream -- this is a
    pre-existing deviation the project already carries on its other Grid build.
    Stock Grid insists an ILDG file carry an ildg-LFN record; the Chroma/SciDAC
    configurations this campaign reads do not have one, so every run aborts at
    load with `GRID_ASSERT failure: ... found_ildgLFN`. Grid-TXQCD's fork has
    already dropped the same assert.

    Applied here so a freshly staged tree can read the campaign's gauge fields.
    """
    if not os.path.isfile(path):
        sys.exit('ERROR: not found: %s' % path)
    text = io.open(path, encoding='utf-8').read()
    needle = '    GRID_ASSERT(found_ildgLFN);\n'
    if needle not in text:
        print('%s: ildg-LFN assert already absent' % ILDG)
        return
    io.open(path, 'w', encoding='utf-8').write(text.replace(needle, '', 1))
    print('%s: dropped the ildg-LFN assert' % ILDG)


def fix_tracing_header(tpath):
    """Two fixes to Grid/perfmon/Tracing.h, both needed on CUDA >= 12.9.

    1. The header is at <nvtx3/nvToolsExt.h> now, not at the include root.
    2. ⛔ The include sits INSIDE `NAMESPACE_BEGIN(Grid)`, so the NVTX C API is
       declared as Grid::nvtxRangePushA and friends. That is the root cause of
       two otherwise baffling failures:
         - any code outside namespace Grid that includes the same header gets
           nothing (the include guard is already set) and fails with
           "nvtxRangePushA is undefined";
         - CUB's cub/detail/nvtx.cuh includes <nvtx3/nvtx3.hpp>, whose C++
           wrapper then cannot find the C types at global scope, producing ~40
           errors inside nvtx3.hpp that name neither Grid nor CUB.
       Hoisting the include above NAMESPACE_BEGIN fixes both.
    """
    if not os.path.isfile(tpath):
        sys.exit('ERROR: not found: %s' % tpath)

    text = io.open(tpath, encoding='utf-8').read()
    guarded = '#ifdef GRID_TRACING_NVTX\n#include <nvtx3/nvToolsExt.h>\n#endif\n\nNAMESPACE_BEGIN(Grid);'

    if guarded in text:
        print('%s: already fixed' % TRACING)
        return

    # Drop the include from wherever it currently sits, in either spelling.
    before = text
    for spelling in ('#include <nvtx3/nvToolsExt.h>\n', '#include <nvToolsExt.h>\n'):
        text = text.replace(spelling, '')
    if text == before:
        sys.exit('ERROR: no nvToolsExt include found in %s' % TRACING)

    if 'NAMESPACE_BEGIN(Grid);' not in text:
        sys.exit('ERROR: NAMESPACE_BEGIN(Grid) not found in %s' % TRACING)
    text = text.replace('NAMESPACE_BEGIN(Grid);', guarded, 1)

    io.open(tpath, 'w', encoding='utf-8').write(text)
    print('%s: include corrected to nvtx3/ and hoisted out of namespace Grid' % TRACING)


if __name__ == '__main__':
    main()
