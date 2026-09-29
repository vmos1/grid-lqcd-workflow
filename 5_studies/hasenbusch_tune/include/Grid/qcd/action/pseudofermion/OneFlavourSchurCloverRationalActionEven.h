#pragma once
// EVEN-parity variant of OneFlavourSchurCloverRationalAction.
//
// Identical action — the partition function is unchanged whether we sample
// PhiOdd ~ exp[-φ† (M_pc_oo†M_pc_oo)^(-1/2) φ] or
// PhiEven ~ exp[-φ† (M_pc_ee†M_pc_ee)^(-1/2) φ], because det(M_pc_oo) =
// det(M_pc_ee) by the Schur identity.
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
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  MultiShiftFunction PowerNegHalf;

protected:
  FermionOperator &FermOp;
  FermionField PhiEven;

public:
  OneFlavourSchurCloverRationalActionEven(FermionOperator &Op, Params &p)
      : FermOp(Op), PhiEven(Op.FermionRedBlackGrid()), param(p) {
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

    FermOp.ImportGauge(U);

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter, PowerQuarter);
    msCG(Mpc, etaEven, PhiEven);
  }

  RealD S(const GaugeField &U) override {
    FermOp.ImportGauge(U);

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

    FermOp.ImportGauge(U);

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
