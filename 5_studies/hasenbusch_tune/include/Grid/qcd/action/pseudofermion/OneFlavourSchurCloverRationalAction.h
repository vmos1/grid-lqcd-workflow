#pragma once
// One-flavour Schur-complement RHMC for Wilson-Clover.
//
// det(M†M)^{1/2} = det(Mee†Mee)^{1/2} × det(Mpc†Mpc)^{1/2}
//
// This action handles: det(Mpc†Mpc)^{1/2} via rational approximation.
// Paired with QCDLogDetCloverEOAction(nf=1) for the even-even block.
//
// Unlike OneFlavourEvenOddRationalPseudoFermionAction, this does NOT assume ConstEE.
// The force includes clover corrections (MooDeriv, MeeDeriv chain rule).

#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

NAMESPACE_BEGIN(Grid);

template <class Impl,
          class FermionOp = WilsonCloverFermion<Impl, Grid::CloverHelpers<Impl>>>
class OneFlavourSchurCloverRationalAction
    : public Action<typename Impl::GaugeField> {
public:
  INHERIT_IMPL_TYPES(Impl);

  typedef OneFlavourRationalParams Params;
  typedef FermionOp FermionOperator;

  Params param;
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  MultiShiftFunction PowerNegHalf;

protected:
  FermionOperator &FermOp;
  FermionField PhiOdd;

public:
  OneFlavourSchurCloverRationalAction(FermionOperator &Op, Params &p)
      : FermOp(Op), PhiOdd(Op.FermionRedBlackGrid()), param(p) {
    AlgRemez remez(param.lo, param.hi, param.precision);

    std::cout << GridLogMessage << "Generating degree " << param.degree
              << " for x^(1/4)" << std::endl;
    remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);

    std::cout << GridLogMessage << "Generating degree " << param.degree
              << " for x^(1/2)" << std::endl;
    remez.generateApprox(param.degree, 1, 2);
    PowerNegHalf.Init(remez, param.tolerance, true);
  }

  std::string action_name() override {
    return "OneFlavourSchurCloverRationalAction";
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
    FermionField etaOdd(FermOp.FermionRedBlackGrid());

    gaussian(pRNG, eta);
    eta = eta * scale;
    pickCheckerboard(Odd, etaOdd, eta);

    FermOp.ImportGauge(U);

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter, PowerQuarter);
    msCG(Mpc, etaOdd, PhiOdd);
  }

  RealD S(const GaugeField &U) override {
    FermOp.ImportGauge(U);

    FermionField Y(FermOp.FermionRedBlackGrid());

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter,
                                                   PowerNegQuarter);
    msCG(Mpc, PhiOdd, Y);

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

    FermOp.ImportGauge(U);

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter, PowerNegHalf);
    msCG(Mpc, PhiOdd, MPhi_k);

    dSdU = Zero();
    for (int k = 0; k < Npole; k++) {
      RealD ak = PowerNegHalf.residues[k];

      X = MPhi_k[k];
      Mpc.Mpc(X, Y);

      // 1. Hopping forces
      Mpc.MpcDeriv(tmp, Y, X);
      dSdU = dSdU + ak * tmp;
      Mpc.MpcDagDeriv(tmp, X, Y);
      dSdU = dSdU + ak * tmp;

      // 2. Odd-site clover force
      FermOp.MooDeriv(tmp, Y, X, DaggerNo);
      dSdU = dSdU + ak * tmp;
      FermOp.MooDeriv(tmp, X, Y, DaggerYes);
      dSdU = dSdU + ak * tmp;

      // 3. Even-site clover force (chain rule through Mee^{-1})
      FermionField W_e(fcbgrid), Z_e(fcbgrid), tmp1(fcbgrid);

      FermOp.Meooe(X, tmp1);
      FermOp.MooeeInv(tmp1, W_e);

      FermOp.MeooeDag(Y, tmp1);
      FermOp.MooeeInvDag(tmp1, Z_e);

      FermOp.MeeDeriv(tmp, Z_e, W_e, DaggerNo);
      dSdU = dSdU + ak * tmp;
      FermOp.MeeDeriv(tmp, W_e, Z_e, DaggerYes);
      dSdU = dSdU + ak * tmp;
    }
  }
};

NAMESPACE_END(Grid);
