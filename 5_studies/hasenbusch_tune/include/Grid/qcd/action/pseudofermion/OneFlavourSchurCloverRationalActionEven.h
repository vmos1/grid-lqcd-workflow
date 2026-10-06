#pragma once
// EVEN-parity variant of OneFlavourSchurCloverRationalAction.
//
// NOT the identical action on its own (corrected 2026-10-02): det(M_pc_oo) = det M / det M_ee
// and det(M_pc_ee) = det M / det M_oo, and with a clover term det M_ee != det M_oo. Sampling
// PhiEven ~ exp[-φ† (M_pc_ee†M_pc_ee)^(-1/2) φ] instead of PhiOdd therefore requires the
// partner log-det monomial on the OTHER block: -ln|det M_oo| (QCDLogDetCompactCloverEOAction
// with parity Odd), not the even block it was paired with from 2026-06-24 to 2026-10-02
// (that sampled |det M| det M_ee / det M_oo; grid_qcd docs
// 2026_10_02_strange_logdet_parity_mismatch.md, L189). The driver pairs the two automatically.
//
// Reason for an even-parity variant: QUDA's `computeCloverForceQuda`
// hardcodes EVEN_EVEN_ASYMMETRIC matpc and expects the X_k solutions on
// the EVEN sublattice — see lib/interface_quda.cpp:4488 in agrebe's
// QUDA-5.  Using the even convention end-to-end is the cleanest path to
// plug QUDA's force routine in.
//
// The action otherwise mirrors OneFlavourSchurCloverRationalAction line
// for line, with PhiOdd→PhiEven, pickCheckerboard(Odd→Even), and the
// odd-vs-even asymmetric pieces of the deriv chain swapped (MooDeriv ↔
// MeeDeriv at the diagonal-clover step; everything off-diagonal is
// already parity-agnostic in SchurDifferentiableOperator).

#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
// A3: every ImportGauge into FermOp goes through the exact-skip guard (HASEN_GRID_IMPORT_SKIP=1,
// default off = plain ImportGauge). The strange operator is also imported by the MP deriv and by
// the strange log-det action, all guarded, so the guard sees ALL imports into it (hazard 1).
#include "gauge_import/import_guard.h"
// OneFlavourSchurRationalExtras and the bounds check (separate force degree/tolerance, L191).
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h>

NAMESPACE_BEGIN(Grid);

template <class Impl,
          class FermionOp = WilsonCloverFermion<Impl, Grid::CloverHelpers<Impl>>>
class OneFlavourSchurCloverRationalActionEven
    : public Action<typename Impl::GaugeField> {
public:
  INHERIT_IMPL_TYPES(Impl);

  typedef OneFlavourRationalParams Params;
  typedef FermionOp FermionOperator;

  Params param;
  OneFlavourSchurRationalExtras extras;
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  MultiShiftFunction PowerNegHalf;  // the force (MD) approximation: extras.md_degree / md_tolerance
  RealD RemezErrorAction = 0.0;     // AlgRemez errors of x^(1/4) and of the force's x^(1/2)
  RealD RemezErrorMD = 0.0;

protected:
  FermionOperator &FermOp;
  FermionField PhiEven;
  long refreshes = 0;

public:
  OneFlavourSchurCloverRationalActionEven(FermionOperator &Op, Params &p,
                                          const OneFlavourSchurRationalExtras &x =
                                              OneFlavourSchurRationalExtras())
      : FermOp(Op), PhiEven(Op.FermionRedBlackGrid()), param(p), extras(x) {
    AlgRemez remez(param.lo, param.hi, param.precision);
    const int md_degree = extras.md_degree > 0 ? extras.md_degree : param.degree;
    const RealD md_tol = extras.md_tolerance > 0.0 ? extras.md_tolerance : param.tolerance;

    std::cout << GridLogMessage << "Generating degree " << param.degree
              << " for x^(1/4)" << std::endl;
    RemezErrorAction = remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);

    std::cout << GridLogMessage << "Generating degree " << md_degree
              << " for x^(1/2)" << std::endl;
    RemezErrorMD = remez.generateApprox(md_degree, 1, 2);
    PowerNegHalf.Init(remez, md_tol, true);
  }

  std::string action_name() override {
    return "OneFlavourSchurCloverRationalActionEven";
  }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] lo=" << param.lo
       << " hi=" << param.hi << " degree=" << param.degree << std::endl;
    return os.str();
  }

  void refresh(const GaugeField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    RealD scale = std::sqrt(0.5);

    FermionField eta(FermOp.FermionGrid());
    FermionField etaEven(FermOp.FermionRedBlackGrid());

    gaussian(pRNG, eta);
    eta = eta * scale;
    pickCheckerboard(Even, etaEven, eta);

    GuardedImportGauge(FermOp, U);

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter, PowerQuarter);
    msCG(Mpc, etaEven, PhiEven);

    const long n = ++refreshes;
    if (extras.bounds_check_freq > 0 && (n - 1) % extras.bounds_check_freq == 0)
      OneFlavourSchurRationalBoundsCheck<FermionField>(Mpc, etaEven, PowerNegQuarter, PowerNegHalf,
                                                       RemezErrorAction, RemezErrorMD, param,
                                                       extras, action_name(), n);
  }

  RealD S(const GaugeField &U) override {
    GuardedImportGauge(FermOp, U);

    FermionField Y(FermOp.FermionRedBlackGrid());

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter,
                                                   PowerNegQuarter);
    msCG(Mpc, PhiEven, Y);

    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    const int Npole = PowerNegHalf.poles.size();

    std::vector<FermionField> MPhi_k(Npole, FermOp.FermionRedBlackGrid());
    FermionField X(FermOp.FermionRedBlackGrid());
    FermionField Y(FermOp.FermionRedBlackGrid());
    GaugeField tmp(FermOp.GaugeGrid());
    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();

    GuardedImportGauge(FermOp, U);

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter, PowerNegHalf);
    msCG(Mpc, PhiEven, MPhi_k);

    dSdU = Zero();
    for (int k = 0; k < Npole; k++) {
      RealD ak = PowerNegHalf.residues[k];

      X = MPhi_k[k];
      Mpc.Mpc(X, Y);

      // 1. Hopping forces (parity-agnostic in SchurDifferentiableOperator).
      Mpc.MpcDeriv(tmp, Y, X);
      dSdU = dSdU + ak * tmp;
      Mpc.MpcDagDeriv(tmp, X, Y);
      dSdU = dSdU + ak * tmp;

      // 2. Even-site clover force (M_ee derivative — was MooDeriv on the
      //    odd-parity action; here our diagonal block is M_ee).
      FermOp.MeeDeriv(tmp, Y, X, DaggerNo);
      dSdU = dSdU + ak * tmp;
      FermOp.MeeDeriv(tmp, X, Y, DaggerYes);
      dSdU = dSdU + ak * tmp;

      // 3. Odd-site clover force (chain rule through Moo^{-1}).  Mirror of
      //    the odd-parity version: project X, Y onto Odd via Meooe and
      //    MooeeInv, then call MooDeriv on those projected fields.
      FermionField W_o(fcbgrid), Z_o(fcbgrid), tmp1(fcbgrid);

      FermOp.Meooe(X, tmp1);          // even → odd
      FermOp.MooeeInv(tmp1, W_o);     // odd → odd

      FermOp.MeooeDag(Y, tmp1);       // even → odd
      FermOp.MooeeInvDag(tmp1, Z_o);  // odd → odd

      FermOp.MooDeriv(tmp, Z_o, W_o, DaggerNo);
      dSdU = dSdU + ak * tmp;
      FermOp.MooDeriv(tmp, W_o, Z_o, DaggerYes);
      dSdU = dSdU + ak * tmp;
    }
  }
};

NAMESPACE_END(Grid);
