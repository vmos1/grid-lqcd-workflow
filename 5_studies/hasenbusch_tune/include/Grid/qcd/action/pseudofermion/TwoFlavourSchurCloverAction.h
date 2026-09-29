#pragma once
// Two-flavour Schur-complement pseudofermion action for Wilson-Clover.
//
// Paired with QCDLogDetCloverEOAction for the even-even block:
//   det(M†M) = det(Mee†Mee) × det(Mpc†Mpc)
//
// This action handles:  S_pf = Phi† (Mpc†Mpc)^{-1} Phi   [Phi on odd sublattice]
//
// Unlike TwoFlavourEvenOddPseudoFermionAction, this does NOT assume ConstEE.
// The force includes clover contributions:
//   1. Hopping derivatives (dMoe/dU, dMeo/dU) via SchurDifferentiableOperator
//   2. Odd-site clover derivative: dMoo/dU
//   3. Even-site chain rule: Moe Mee^{-1} dMee/dU Mee^{-1} Meo

#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

NAMESPACE_BEGIN(Grid);

template <class Impl,
          class FermionOp = WilsonCloverFermion<Impl, Grid::CloverHelpers<Impl>>>
class TwoFlavourSchurCloverAction
    : public Action<typename Impl::GaugeField> {
public:
  INHERIT_IMPL_TYPES(Impl);

  typedef FermionOp FermionOperator;

protected:
  FermionOperator &FermOp;
  OperatorFunction<FermionField> &DerivativeSolver;
  OperatorFunction<FermionField> &ActionSolver;
  FermionField PhiOdd;

public:
  TwoFlavourSchurCloverAction(FermionOperator &Op,
                               OperatorFunction<FermionField> &DS,
                               OperatorFunction<FermionField> &AS)
      : FermOp(Op), DerivativeSolver(DS), ActionSolver(AS),
        PhiOdd(Op.FermionRedBlackGrid()) {}

  std::string action_name() override { return "TwoFlavourSchurCloverAction"; }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] has no parameters" << std::endl;
    return os.str();
  }

  void refresh(const GaugeField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    RealD scale = std::sqrt(0.5);

    FermionField eta(FermOp.FermionGrid());
    FermionField etaOdd(FermOp.FermionRedBlackGrid());

    gaussian(pRNG, eta);
    pickCheckerboard(Odd, etaOdd, eta);

    FermOp.ImportGauge(U);
    SchurDifferentiableOperator<Impl> PCop(FermOp);

    PCop.MpcDag(etaOdd, PhiOdd);
    PhiOdd = PhiOdd * scale;
  }

  RealD S(const GaugeField &U) override {
    FermOp.ImportGauge(U);

    FermionField X(FermOp.FermionRedBlackGrid());
    FermionField Y(FermOp.FermionRedBlackGrid());

    SchurDifferentiableOperator<Impl> PCop(FermOp);

    X = Zero();
    ActionSolver(PCop, PhiOdd, X);
    PCop.Op(X, Y);
    RealD action = norm2(Y);

    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action << std::endl;
    return action;
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    FermOp.ImportGauge(U);

    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();
    FermionField X(fcbgrid);
    FermionField Y(fcbgrid);
    GaugeField tmp(FermOp.GaugeGrid());

    SchurDifferentiableOperator<Impl> Mpc(FermOp);

    // Solve for X = (Mpc†Mpc)^{-1} Phi, Y = Mpc X
    X = Zero();
    DerivativeSolver(Mpc, PhiOdd, X);
    Mpc.Mpc(X, Y);

    // 1. Standard hopping-term forces (MoeDeriv, MeoDeriv through SchurDeriv)
    Mpc.MpcDeriv(tmp, Y, X);
    dSdU = tmp;
    Mpc.MpcDagDeriv(tmp, X, Y);
    dSdU = dSdU + tmp;

    // 2. Odd-site clover force from dMoo/dU
    FermOp.MooDeriv(tmp, Y, X, DaggerNo);
    dSdU = dSdU + tmp;
    FermOp.MooDeriv(tmp, X, Y, DaggerYes);
    dSdU = dSdU + tmp;

    // 3. Even-site clover force (chain rule through Mee^{-1})
    //    W_e = Mee^{-1} Meo X,  Z_e = Mee^{-†} Moe† Y
    FermionField W_e(fcbgrid), Z_e(fcbgrid), tmp1(fcbgrid);

    FermOp.Meooe(X, tmp1);
    FermOp.MooeeInv(tmp1, W_e);

    FermOp.MeooeDag(Y, tmp1);
    FermOp.MooeeInvDag(tmp1, Z_e);

    FermOp.MeeDeriv(tmp, Z_e, W_e, DaggerNo);
    dSdU = dSdU + tmp;
    FermOp.MeeDeriv(tmp, W_e, Z_e, DaggerYes);
    dSdU = dSdU + tmp;
  }
};

NAMESPACE_END(Grid);
