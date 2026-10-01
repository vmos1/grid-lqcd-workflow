#pragma once
// gauge_fingerprint.h  (pure-Grid HMC, M3)
//
// A cheap identity test for a gauge-like field, so a solver can skip a redundant fp32
// ImportGauge (MixedPrecCGRungSolver, mixed_cg_rung_solver.h) or decide whether the Grid
// multigrid hierarchy already carries the current gauge field (GridMGHierarchy::SetGaugeIfNew,
// grid_mg_hierarchy.h).
//
// Fingerprint = sum_x tr_colour U_mu(x), one complex number per Lorentz component: 4 for a
// LatticeGaugeField, 8 for a fermion operator's DoubledGaugeField (forward links with the
// boundary phases, then the backward adjoints). One TraceIndex + one global sum.
// Two fields compare EQUAL only if every component matches bit for bit. The reduction is
// deterministic (fixed GPU block order, fixed MPI communicator), so the same field always
// gives the same fingerprint; two different gauge fields of an HMC trajectory agreeing in all
// 4 (or 8) complex trace sums to the last bit does not happen in practice. Equality therefore
// means "the same field" and inequality "a different field" (or, harmlessly, a re-import).

#include <Grid/Grid.h>

#include <vector>

namespace Grid {

typedef std::vector<ComplexD> GaugeFingerprint;

namespace detail_fingerprint {
template <class T, int N>
constexpr int LorentzExtent(const iVector<T, N> &) { return N; }
}  // namespace detail_fingerprint

// vobj = iVector<iScalar<iMatrix<vComplex, Nc>>, N> (a gauge field or a doubled gauge field,
// either precision).
template <class vobj>
inline GaugeFingerprint GaugeTraceFingerprint(const Lattice<vobj> &U)
{
  auto trU = TraceIndex<ColourIndex>(U);  // iVector<iScalar<iScalar<vComplex>>, N> per site
  auto s = sum(trU);                      // global, scalar object
  const int n = detail_fingerprint::LorentzExtent(s);
  GaugeFingerprint fp(n);
  for (int mu = 0; mu < n; ++mu) {
    const auto z = s(mu)()();
    fp[mu] = ComplexD((RealD)z.real(), (RealD)z.imag());
  }
  return fp;
}

}  // namespace Grid
