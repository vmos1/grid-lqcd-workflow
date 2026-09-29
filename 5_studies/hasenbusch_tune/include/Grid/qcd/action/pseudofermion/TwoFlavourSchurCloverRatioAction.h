#pragma once
// Two-flavour Schur-complement Hasenbusch RATIO action for Wilson-Clover.
//
// det(NumOp)^2 / det(DenOp)^2, restricted to the odd-parity Schur complement
// ONLY -- no even-even (clover diagonal) term, no ConstEE assumption.
//
// Additive class: does NOT touch TwoFlavourEvenOddRatio.h (Wilson-only ratio,
// which assumes ConstEE==1 and therefore cannot be used with a clover operator
// -- that assumption is false for clover, where M_ee depends on U).  Instead
// this mirrors TwoFlavourSchurCloverAction.h (which already handles clover
// correctly by leaving M_ee to a SEPARATE deterministic log-det monomial,
// QCDLogDetCloverEOAction / QCDLogDetCompactCloverEOAction) and extends it to
// a NumOp/DenOp ratio the same way TwoFlavourRatio.h extends TwoFlavour.h.
//
// Physics: in a Hasenbusch chain built from N ladder masses m_0 < m_1 < ... <
// m_{N-1}, det(M(m_0))^2 telescopes as
//   det(M(m_0))^2 = det(M_ee(m_0))^2 * [ Prod_k det(Schur(m_k))^2/det(Schur(m_{k+1}))^2 ]
//                    * det(Schur(m_{N-1}))^2
// The M_ee(m_0) factor is supplied ONCE, deterministically, by
// QCDLogDetCloverEOAction at the lightest mass -- it is exact algebra, not a
// new approximation, so no EE term or force is needed here.  Each ratio rung
// in the product above is one instance of this class (NumOp = heavier mass,
// DenOp = lighter mass).
//
// This action handles:  S = phi^dag Vpc (Mpc^dag Mpc)^-1 Vpc^dag phi
//   NumOp == V (heavier)     DenOp == M (lighter)
// which is exactly TwoFlavourEvenOddRatioPseudoFermionAction's odd-sector
// math (TwoFlavourEvenOddRatio.h) with every PhiEven / even-sector line
// deleted, and the GRID_ASSERT(ConstEE()==1) removed (there is no EE term
// left to guard).

#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

NAMESPACE_BEGIN(Grid);

template <class Impl,
          class FermionOp = WilsonCloverFermion<Impl, Grid::CloverHelpers<Impl>>>
class TwoFlavourSchurCloverRatioAction
    : public Action<typename Impl::GaugeField> {
public:
  INHERIT_IMPL_TYPES(Impl);

  typedef FermionOp FermionOperator;

protected:
  FermionOperator &NumOp;  // heavier mass
  FermionOperator &DenOp;  // lighter mass

  OperatorFunction<FermionField> &DerivativeSolver;
  OperatorFunction<FermionField> &ActionSolver;
  OperatorFunction<FermionField> &HeatbathSolver;

  FermionField PhiOdd;

public:
  TwoFlavourSchurCloverRatioAction(FermionOperator &_NumOp,
                                    FermionOperator &_DenOp,
                                    OperatorFunction<FermionField> &DS,
                                    OperatorFunction<FermionField> &AS)
      : TwoFlavourSchurCloverRatioAction(_NumOp, _DenOp, DS, AS, AS) {}

  TwoFlavourSchurCloverRatioAction(FermionOperator &_NumOp,
                                    FermionOperator &_DenOp,
                                    OperatorFunction<FermionField> &DS,
                                    OperatorFunction<FermionField> &AS,
                                    OperatorFunction<FermionField> &HS)
      : NumOp(_NumOp), DenOp(_DenOp),
        DerivativeSolver(DS), ActionSolver(AS), HeatbathSolver(HS),
        PhiOdd(_NumOp.FermionRedBlackGrid()) {
    conformable(_NumOp.FermionGrid(), _DenOp.FermionGrid());
    conformable(_NumOp.FermionRedBlackGrid(), _DenOp.FermionRedBlackGrid());
    conformable(_NumOp.GaugeGrid(), _DenOp.GaugeGrid());
    conformable(_NumOp.GaugeRedBlackGrid(), _DenOp.GaugeRedBlackGrid());
  }

  std::string action_name() override {
    std::stringstream sstream;
    sstream << "TwoFlavourSchurCloverRatioAction det(" << DenOp.Mass()
            << ") / det(" << NumOp.Mass() << ")";
    return sstream.str();
  }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] has no further parameters" << std::endl;
    return os.str();
  }

  const FermionField &getPhiOdd() const { return PhiOdd; }

  void refresh(const GaugeField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    RealD scale = std::sqrt(0.5);

    FermionField eta(NumOp.FermionGrid());
    gaussian(pRNG, eta);
    eta = eta * scale;

    FermionField etaOdd(NumOp.FermionRedBlackGrid());
    pickCheckerboard(Odd, etaOdd, eta);

    NumOp.ImportGauge(U);
    DenOp.ImportGauge(U);

    SchurDifferentiableOperator<Impl> Mpc(DenOp);
    SchurDifferentiableOperator<Impl> Vpc(NumOp);

    // P(phi) = e^{- phi^dag Vpc (Mpc^dag Mpc)^-1 Vpc^dag phi}
    // Take phi = Vpc^dag^{-1} Mpc^dag eta ; sampled via one heatbath solve.
    FermionField tmp(NumOp.FermionRedBlackGrid());
    Mpc.MpcDag(etaOdd, PhiOdd);
    tmp = Zero();
    HeatbathSolver(Vpc, PhiOdd, tmp);
    Vpc.Mpc(tmp, PhiOdd);
  }

  //////////////////////////////////////////////////////
  // S = phi^dag Vpc (Mpc^dag Mpc)^-1 Vpc^dag phi
  //////////////////////////////////////////////////////
  RealD S(const GaugeField &U) override {
    NumOp.ImportGauge(U);
    DenOp.ImportGauge(U);

    SchurDifferentiableOperator<Impl> Mpc(DenOp);
    SchurDifferentiableOperator<Impl> Vpc(NumOp);

    FermionField X(NumOp.FermionRedBlackGrid());
    FermionField Y(NumOp.FermionRedBlackGrid());

    Vpc.MpcDag(PhiOdd, Y);   // Y = Vpc^dag phi
    X = Zero();
    ActionSolver(Mpc, Y, X); // X = (Mpc^dag Mpc)^-1 Vpc^dag phi

    RealD action = real(innerProduct(Y, X));

    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action << std::endl;
    return action;
  }

  //////////////////////////////////////////////////////
  // dS/dU = phi^dag dVpc (Mpc^dag Mpc)^-1 Vpc^dag  phi
  //       - phi^dag Vpc (Mpc^dag Mpc)^-1 [ Mpc^dag dMpc + dMpc^dag Mpc ] (Mpc^dag Mpc)^-1 Vpc^dag phi
  //       + phi^dag Vpc (Mpc^dag Mpc)^-1 dVpc^dag  phi
  //////////////////////////////////////////////////////
  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    NumOp.ImportGauge(U);
    DenOp.ImportGauge(U);

    GridBase *fcbgrid = NumOp.FermionRedBlackGrid();
    SchurDifferentiableOperator<Impl> Mpc(DenOp);
    SchurDifferentiableOperator<Impl> Vpc(NumOp);

    FermionField X(fcbgrid);
    FermionField Y(fcbgrid);
    GaugeField force(dSdU.Grid());

    Vpc.MpcDag(PhiOdd, Y);          // Y = Vpc^dag phi
    X = Zero();
    DerivativeSolver(Mpc, Y, X);    // X = (Mpc^dag Mpc)^-1 Vpc^dag phi
    Mpc.Mpc(X, Y);                  // Y = Mpc^dag^{-1} Vpc^dag phi

    // NumOp/Vpc hopping-derivative terms (Moe/Meo, per SchurDifferentiableOperator).
    Vpc.MpcDagDeriv(force, X, PhiOdd);  dSdU = force;
    Vpc.MpcDeriv(force, PhiOdd, X);     dSdU = dSdU + force;

    // DenOp/Mpc hopping-derivative terms.
    Mpc.MpcDeriv(force, Y, X);          dSdU = dSdU - force;
    Mpc.MpcDagDeriv(force, X, Y);       dSdU = dSdU - force;

    // SchurDifferentiableOperator::MpcDeriv/MpcDagDeriv only differentiate the
    // off-diagonal (Moe/Meo) hopping terms and explicitly "assume Mee is indept
    // of U" (see EvenOddSchurDifferentiable.h) -- true for Wilson (ConstEE==1,
    // which is what TwoFlavourEvenOddRatio.h relies on) but FALSE for clover,
    // where BOTH the odd-odd diagonal block (Moo) and the even-even block (Mee,
    // via the Meo Mee^-1 Moe chain inside Mpc) depend on U.  TwoFlavourSchurCloverAction
    // supplies exactly these two missing pieces for a single operator; here they
    // are needed for BOTH NumOp (Vpc) and DenOp (Mpc), using each operator's own
    // (X, PhiOdd) / (Y, X) argument pairing established above.
    GaugeField tmp(dSdU.Grid());

    // DenOp odd-diagonal (Moo) force -- same (Y,X) pairing as the Mpc hopping terms.
    DenOp.MooDeriv(tmp, Y, X, DaggerNo);   dSdU = dSdU - tmp;
    DenOp.MooDeriv(tmp, X, Y, DaggerYes);  dSdU = dSdU - tmp;

    // DenOp even-diagonal (Mee) chain-rule force:
    //   W_e = Mee^{-1} Meo X,  Z_e = Mee^{-dag} Moe^dag Y
    FermionField W_e(fcbgrid), Z_e(fcbgrid), tmp1(fcbgrid);
    DenOp.Meooe(X, tmp1);         DenOp.MooeeInv(tmp1, W_e);
    DenOp.MeooeDag(Y, tmp1);      DenOp.MooeeInvDag(tmp1, Z_e);
    DenOp.MeeDeriv(tmp, Z_e, W_e, DaggerNo);   dSdU = dSdU - tmp;
    DenOp.MeeDeriv(tmp, W_e, Z_e, DaggerYes);  dSdU = dSdU - tmp;

    // NumOp odd-diagonal (Moo) force -- same (PhiOdd,X) pairing as the Vpc
    // hopping terms (PhiOdd plays DenOp-side "Y"'s role here).
    NumOp.MooDeriv(tmp, PhiOdd, X, DaggerNo);   dSdU = dSdU + tmp;
    NumOp.MooDeriv(tmp, X, PhiOdd, DaggerYes);  dSdU = dSdU + tmp;

    // NumOp even-diagonal (Mee) chain-rule force:
    //   W_v = Mee^{-1} Meo X,  Z_v = Mee^{-dag} Moe^dag PhiOdd
    FermionField W_v(fcbgrid), Z_v(fcbgrid);
    NumOp.Meooe(X, tmp1);         NumOp.MooeeInv(tmp1, W_v);
    NumOp.MeooeDag(PhiOdd, tmp1); NumOp.MooeeInvDag(tmp1, Z_v);
    NumOp.MeeDeriv(tmp, Z_v, W_v, DaggerNo);   dSdU = dSdU + tmp;
    NumOp.MeeDeriv(tmp, W_v, Z_v, DaggerYes);  dSdU = dSdU + tmp;

    // No separate EvenEven stochastic term: this class only ever represents
    // the odd-parity Schur complement.  The clover M_ee DETERMINANT (as
    // opposed to the M_ee dependence of Mpc/Vpc handled above) is supplied
    // separately and exactly by QCDLogDetCloverEOAction /
    // QCDLogDetCompactCloverEOAction at the lightest ladder mass -- see the
    // telescoping argument in the file header comment.
    dSdU = -dSdU;
  }
};

NAMESPACE_END(Grid);
