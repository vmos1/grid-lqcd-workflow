#pragma once
// fused_clover_force.h  (pure-Grid HMC, M5 fix 2)
//
// Fused clover-force assembly: ONE accumulated outer-product field and ONE clover-derivative
// pass per force call, replacing the per-call MooDeriv/MeeDeriv of the carried Schur actions.
// Spec: __docs/2026_09_30_pure_grid_force_cost_analysis.md sections 3, 5 (fix 2) and 6.
// Opt-in: HASEN_GRID_FUSED_CLOVER_FORCE=1, parsed once by the driver into
// FusedCloverForceEnabled(); default off, and then nothing here is constructed or run.
//
// What one patch-01 body computes. Line numbers refer to the staged, patched stock tree,
// $PSCRATCH/grid_pure_hmc/stock-grid/3d3eff86.../Grid/Grid/qcd/action/fermion/implementation/
// CompactWilsonCloverFermionImplementation.h (patch 01-clover-eo-force-MooDeriv-MeeDeriv.patch):
// MooDeriv lines 278-324, MeeDeriv 329-375, identical apart from the parity they assert.
// For half-grid spinors U, V of one parity p (the `dag` argument is ignored):
//   283-286  Uf, Vf = zero-filled full-grid copies of U, V           (p-sites only)
//   291-292  Ulinks[mu] = forward links of this->Umu (the doubled field: -0.5 * U with the
//            boundary phases, WilsonFermionImplementation.h:88-99; see Finish below)
//   294      Lambda = outerProduct(Uf, Vf), i.e. Lambda(x) = Uf(x) Vf(x)^dag   (spin x colour)
//   310-323  mat = 0; for mu, for nu != mu, in row-major (mu,nu) order, count = 0..11:
//   316        factor = 2 csw_r   (the test is `nu==4 || mu==4`, never true, so csw_t is never
//                                  used; harmless because the driver builds csw_r == csw_t)
//   317-318    lambda = TraceSpin( sigma[count] Lambda ),  sigma[count] = sigma_{mu nu}
//                (sigma table 296-308: SigmaXY.. for mu<nu, MinusSigma(nu,mu) for mu>nu)
//   319        force_mu -= factor * Cmunu(Ulinks, lambda, mu, nu)
//   322      mat_mu = Ulinks[mu] * force_mu   (pokeLorentz)
// csw_r is the operator member, already 0.5 * the constructor's csw (constructor lines 64-65).
//
// Why one pass is exact. Every step after the outer product is real-linear in Lambda:
// Gamma.Lambda, the spin trace and Cmunu (which uses lambda and adj(lambda),
// WilsonCloverHelpers.h:42-79) and the final link product. So for REAL coefficients c_k and
// the same links and csw,  sum_k c_k Body[U_k, V_k] = Body'[ sum_k c_k U_k (x) V_k^dag ],
// exact up to summation order. Moo inputs live on odd sites, Mee inputs on even sites, and a
// body's Lambda is exactly zero on the other parity (lines 283-286), so one Lambda holding the
// odd-site sum on odd sites and the even-site sum on even sites serves both kinds at once.
// The callers must pass spinors and coefficients exactly as the per-call code adds the
// individual results (see the carried action headers), and all accumulated calls must come
// from operators with the same Umu and csw (true in the actions: every operator imports the
// same U in the same deriv, and the driver builds them all with the same csw and phases).
//
// Arithmetic kept as close to the body as Grid allows:
//   * The outer product, Gamma.Lambda and the spin trace run on the half (red-black) grid, site
//     by site the same arithmetic as the body's full-grid kernels (all three are site-local and
//     lane-wise); setCheckerboard then assembles each full-grid lambda. This is the carried
//     QCDLogDetCompactCloverEOAction::deriv_gpu pattern ("Bit-identical to deriv_cpu").
//   * 6 sigma traces instead of 12: sigma[count] for mu>nu is MinusSigma(nu,mu), whose Gamma
//     kernel is the exact negation of Sigma(nu,mu)'s (Gamma.h multMinusSigmaXY vs multSigmaXY:
//     timesI <-> timesMinusI; the other pairs permute with +-1, +-i), and the trace is a sum, so
//     the body's lambda_{mu nu} for mu>nu is bit-for-bit -lambda_{nu mu}. As in the logdet,
//     the sign is folded into the factor: (-f)*Cmunu(l) == f*Cmunu(-l) exactly, because
//     Cmunu is a sum of products each linear in l and IEEE negation is exact.
//   * The Cmunu calls, their (mu,nu) order, the factor, `force_mu -=` and the link product
//     are the body's statements. A single pair with coefficient 1 therefore reproduces one
//     body call exactly (test_fused_clover_force.cc checks this).
//
// Memory per instance: two half-grid PropagatorFields (= one full-grid one, the size of one
// body's Lambda); Finish adds six full-grid colour-matrix lambdas and a half-grid Gamma.Lambda
// temporary, which is less than one body's Lambda + Slambda peak.
//
// Item D (clover_cmunu.h, both default off, both bit-identical): HASEN_GRID_CLOVER_STAPLE_CACHE=1
// runs the Cmunu loop with the lambda-independent link products cached per gauge field (372 ->
// 204 Cshifts per pass on a cache hit); HASEN_GRID_CLOVER_STENCIL=1 replaces the whole loop by
// one PaddedCell exchange per field and site-local kernels (takes precedence over the cache);
// HASEN_GRID_DEVICE_CB=1 (patch 06's switch) also moves the 12 lambda setCheckerboard copies of
// FinishLinks to the device.

#include <Grid/Grid.h>

#include "clover_force/clover_cmunu.h"

#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace Grid {

// Process-wide switch, set once by the driver from HASEN_GRID_FUSED_CLOVER_FORCE (default off).
// A function-local static in an inline function: one instance across translation units.
inline bool &FusedCloverForceEnabled()
{
  static bool on = false;
  return on;
}

template <class Impl>
class FusedCloverForce {
 public:
  INHERIT_IMPL_TYPES(Impl);

  // rbgrid: the operator's FermionRedBlackGrid (for 4D Wilson the same object as its
  // GaugeRedBlackGrid), i.e. the grid of the half-grid spinors passed to Accumulate*.
  explicit FusedCloverForce(GridBase *rbgrid) : LambdaOdd(rbgrid), LambdaEven(rbgrid) { Reset(); }

  void Reset()
  {
    LambdaOdd = Zero();
    LambdaOdd.Checkerboard() = Odd;
    LambdaEven = Zero();
    LambdaEven.Checkerboard() = Even;
    n_odd_ = 0;
    n_even_ = 0;
  }

  // Lambda += coeff * X (x) Y^dag on the odd sites: stands for `coeff * MooDeriv(X, Y)`.
  void AccumulateMoo(RealD coeff, const FermionField &X, const FermionField &Y)
  {
    GRID_ASSERT(X.Checkerboard() == Odd);  // MooDeriv lines 280-281
    GRID_ASSERT(Y.Checkerboard() == Odd);
    Accumulate(LambdaOdd, coeff, X, Y);
    n_odd_++;
  }

  // Lambda += coeff * X (x) Y^dag on the even sites: stands for `coeff * MeeDeriv(X, Y)`.
  void AccumulateMee(RealD coeff, const FermionField &X, const FermionField &Y)
  {
    GRID_ASSERT(X.Checkerboard() == Even);  // MeeDeriv lines 331-332
    GRID_ASSERT(Y.Checkerboard() == Even);
    Accumulate(LambdaEven, coeff, X, Y);
    n_even_++;
  }

  // The clover pass on the accumulated Lambda with the operator's own links and csw, exactly
  // as a body reads them (this->Umu, csw_r). Used by the actions. Lambda is left untouched.
  // `tag` names the caller in a one-time log line (first Finish per tag per process).
  //
  // The operator's Umu is the ONLY supported link source. It is not the imported gauge
  // field: WilsonFermion::ImportGauge stores Umu = DoubleStore(-0.5 * U) with the boundary
  // phases (WilsonFermionImplementation.h:88-99), and the body's force is in that
  // normalisation. Every force term has four links, so links taken from a plain U would give
  // 16 times the body's force (a plain-link variant, removed 2026-10-01, measured exactly
  // rel = 15 at 16^3). Do not add one back without reproducing ImportGauge.
  template <class FermOp>
  void Finish(FermOp &op, GaugeField &force_out, const std::string &tag = "")
  {
    // Cmunu below is WilsonCloverHelpers<Impl>::Cmunu, which is what both standard clover
    // helpers forward to (CloverHelpers.h:97-99 and 224-226). Exp-clover helpers have their
    // own Cmunu and are refused here.
    static_assert(std::is_same<FermOp, CompactWilsonCloverFermion<Impl, CompactCloverHelpers<Impl>>>::value ||
                      std::is_same<FermOp, WilsonCloverFermion<Impl, CloverHelpers<Impl>>>::value,
                  "FusedCloverForce: only the standard (compact) Wilson-clover operators");
    // Patch line 316 always takes 2*csw_r; the fused pass reproduces that, so it is only
    // claimed for the isotropic case the driver builds.
    GRID_ASSERT(op.csw_r == op.csw_t);
    conformable(force_out.Grid(), op.GaugeGrid());
    std::vector<GaugeLinkField> Ulinks(Nd, force_out.Grid());
    for (int mu = 0; mu < Nd; mu++)  // WilsonImpl::extractLinkField body (WilsonImpl.h:187-191)
      Ulinks[mu] = PeekIndex<LorentzIndex>(op.Umu, mu);
    Announce(tag);
    FinishLinks(Ulinks, op.csw_r, force_out);
  }

  int NumMoo() const { return n_odd_; }
  int NumMee() const { return n_even_; }

 private:
  PropagatorField LambdaOdd;   // sum_k c_k X_k (x) Y_k^dag over the Moo pairs, odd half grid
  PropagatorField LambdaEven;  // the same over the Mee pairs, even half grid
  int n_odd_ = 0;
  int n_even_ = 0;

  static void Accumulate(PropagatorField &L, RealD coeff, const FermionField &X, const FermionField &Y)
  {
    conformable(X.Grid(), L.Grid());
    conformable(Y.Grid(), L.Grid());
    // The body's `Impl::outerProductImpl(Lambda, Uf, Vf)` (patch line 294), whose body is
    // `mat = outerProduct(B, A)` (WilsonImpl.h:179-181), on the half grid. Grid's lattice
    // outerProduct returns a field with the default checkerboard; set it before the axpy.
    PropagatorField XY(L.Grid());
    XY = outerProduct(X, Y);
    XY.Checkerboard() = L.Checkerboard();
    L = L + coeff * XY;
  }

  void Announce(const std::string &tag) const
  {
    if (tag.empty()) return;
    static std::set<std::string> seen;
    if (!seen.insert(tag).second) return;
    std::cout << GridLogMessage << "[FusedCloverForce] " << tag << ": " << n_odd_ << " Moo + "
              << n_even_ << " Mee outer products -> one clover pass (HASEN_GRID_FUSED_CLOVER_FORCE=1)"
              << std::endl;
  }

  // Patch lines 310-323 on the accumulated Lambda, with 6 sigma traces (see file header).
  void FinishLinks(std::vector<GaugeLinkField> &Ulinks, RealD csw_r, GaugeField &force_out) const
  {
    GridBase *fgrid = force_out.Grid();
    GridBase *rbgrid = LambdaOdd.Grid();
    conformable(Ulinks[0].Grid(), fgrid);

    // sigma_{mu nu} for mu < nu: the body's sigma[count] at the six mu<nu positions.
    const Gamma::Algebra positive_sigma[6] = {
        Gamma::Algebra::SigmaXY,   // (0,1)
        Gamma::Algebra::SigmaXZ,   // (0,2)
        Gamma::Algebra::SigmaXT,   // (0,3)
        Gamma::Algebra::SigmaYZ,   // (1,2)
        Gamma::Algebra::SigmaYT,   // (1,3)
        Gamma::Algebra::SigmaZT};  // (2,3)
    const int sigma_idx[4][4] = {{-1, 0, 1, 2},
                                 { 0,-1, 3, 4},
                                 { 1, 3,-1, 5},
                                 { 2, 4, 5,-1}};

    // lambda_k = TraceSpin(sigma_k Lambda) on both parities (patch lines 317-318).
    std::vector<GaugeLinkField> lambda(6, fgrid);
    {
      PropagatorField Slambda(rbgrid);
      GaugeLinkField lambda_h(rbgrid);
      const PropagatorField *halves[2] = {&LambdaEven, &LambdaOdd};
      for (int k = 0; k < 6; k++) {
        for (const PropagatorField *L : halves) {
          Slambda = Gamma(positive_sigma[k]) * (*L);
          lambda_h = TraceIndex<SpinIndex>(Slambda);  // WilsonImpl::TraceSpinImpl (WilsonImpl.h:183-185)
          lambda_h.Checkerboard() = L->Checkerboard();  // TraceIndex returns the default cb
          // Host setCheckerboard, or the device-side copy under HASEN_GRID_DEVICE_CB=1
          // (clover_cmunu.h; patch 06's switch). Copy only: bit-identical.
          CloverSetCheckerboard(lambda[k], lambda_h);
        }
      }
    }

    // HASEN_GRID_CLOVER_STENCIL=1: the whole loop below as one PaddedCell exchange per field
    // and site-local kernels (clover_cmunu.h, stage 2), then the same link product (line 322).
    if (CloverStencilEnabled()) {
      const RealD factor = 2.0 * csw_r;  // line 316, signed as below
      RealD sf[4][4];
      for (int mu = 0; mu < 4; mu++)
        for (int nu = 0; nu < 4; nu++) sf[mu][nu] = (mu == nu) ? 0.0 : ((mu < nu) ? factor : -factor);
      const GaugeLinkField *lam[6] = {&lambda[0], &lambda[1], &lambda[2], &lambda[3], &lambda[4], &lambda[5]};
      std::vector<GaugeLinkField> force_mu(Nd, fgrid);
      CloverCmunuStencil<Impl>::Instance().ForceMu(Ulinks, lam, sigma_idx, sf, force_mu);
      force_out = Zero();
      for (int mu = 0; mu < 4; mu++) pokeLorentz(force_out, Ulinks[mu] * force_mu[mu], mu);
      return;
    }

    // HASEN_GRID_CLOVER_STAPLE_CACHE=1: the same Cmunu with its lambda-independent link
    // products cached per gauge field (clover_cmunu.h; bit-identical). Off: the stock call.
    CloverStapleCache<Impl> *cache = nullptr;
    double t_loop0 = 0;
    if (CloverStapleCacheEnabled()) {
      cache = &CloverStapleCache<Impl>::Instance();
      cache->Prepare(Ulinks);
      t_loop0 = usecond();
    }

    // Patch lines 310-323: same statements, same (mu,nu) order.
    GaugeLinkField force_mu(fgrid);
    force_out = Zero();
    for (int mu = 0; mu < 4; mu++) {
      force_mu = Zero();
      for (int nu = 0; nu < 4; nu++) {
        if (mu == nu) continue;
        const RealD factor = 2.0 * csw_r;  // line 316 (its csw_t branch is unreachable)
        // sigma_{mu nu} = -sigma_{nu mu}: the sign goes into the factor (file header).
        const RealD signed_factor = (mu < nu) ? factor : -factor;
        if (cache)
          force_mu -= signed_factor * cache->Cmunu(Ulinks, lambda[sigma_idx[mu][nu]], mu, nu);
        else
          force_mu -= signed_factor * WilsonCloverHelpers<Impl>::Cmunu(Ulinks, lambda[sigma_idx[mu][nu]], mu, nu);
      }
      pokeLorentz(force_out, Ulinks[mu] * force_mu, mu);  // line 322
    }
    if (cache) cache->EndPass(usecond() - t_loop0);
  }
};

}  // namespace Grid
