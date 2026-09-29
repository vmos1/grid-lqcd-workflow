#pragma once
// Mixed-precision variant of OneFlavourSchurCloverRationalAction.
//
// Inherits refresh() and S() from the base (run multishift CG in double once
// per trajectory; keeping them full-DP avoids any accept/reject bias).
// Overrides deriv() to use ConjugateGradientMultiShiftMixedPrec for the
// MD force evaluation — the dominant cost.
//
// Used by both gen_qcd_cfgs.cc (Nf=2+1 reference QCD) and gen_txqcd_cfgs.cc
// (Nf=2 TXQCD light + Nf=1 QCD strange) to keep the strange-quark force
// computation consistent across drivers.

#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h>

namespace Grid {

template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class OneFlavourSchurCloverRationalActionMP
    : public OneFlavourSchurCloverRationalAction<ImplD, FermOpD_> {
 public:
  typedef OneFlavourSchurCloverRationalAction<ImplD, FermOpD_> Base;
  typedef typename Base::FermionField FermionField;
  typedef FermOpD_ FermOpD;
  typedef FermOpF_ FermOpF;
  typedef typename ImplD::GaugeField GaugeField;

  OneFlavourSchurCloverRationalActionMP(FermOpD &opD, FermOpF &opF,
                                        GridBase *sp_rbgrid,
                                        OneFlavourRationalParams &p,
                                        int reliable_update_freq = 50)
      : Base(opD, p), opF_(opF), sp_rbgrid_(sp_rbgrid),
        reliable_freq_(reliable_update_freq) {}

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    auto &FermOp   = this->FermOp;
    auto &PhiOdd   = this->PhiOdd;
    auto &PowerNegHalf = this->PowerNegHalf;
    const int Npole = PowerNegHalf.poles.size();

    std::vector<FermionField> MPhi_k(Npole, FermOp.FermionRedBlackGrid());
    FermionField X(FermOp.FermionRedBlackGrid());
    FermionField Y(FermOp.FermionRedBlackGrid());
    GaugeField tmp(FermOp.GaugeGrid());
    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();

    FermOp.ImportGauge(U);

    // Refresh the single-precision operator from the RAW gauge field U.
    // (Not FermOp.Umu, which is the internally-doubled -0.5*U storage and
    // would get re-doubled if fed to opF_.ImportGauge.)
    typename ImplF::GaugeField UmuF(opF_.GaugeGrid());
    {
      typename ImplD::GaugeLinkField U_d(U.Grid());
      typename ImplF::GaugeLinkField U_f(opF_.GaugeGrid());
      for (int mu = 0; mu < Nd; ++mu) {
        U_d = PeekIndex<LorentzIndex>(U, mu);
        precisionChange(U_f, U_d);
        PokeIndex<LorentzIndex>(UmuF, U_f, mu);
      }
    }
    opF_.ImportGauge(UmuF);

    SchurDifferentiableOperator<ImplD> Mpc(FermOp);
    SchurDifferentiableOperator<ImplF> Mpc_f(opF_);

    ConjugateGradientMultiShiftMixedPrec<FermionField,
                                         typename ImplF::FermionField>
        msCG(this->param.MaxIter, PowerNegHalf, sp_rbgrid_, Mpc_f, reliable_freq_);
    msCG(Mpc, PhiOdd, MPhi_k);

    dSdU = Zero();
    for (int k = 0; k < Npole; k++) {
      RealD ak = PowerNegHalf.residues[k];
      X = MPhi_k[k];
      Mpc.Mpc(X, Y);

      Mpc.MpcDeriv(tmp, Y, X);          dSdU = dSdU + ak * tmp;
      Mpc.MpcDagDeriv(tmp, X, Y);       dSdU = dSdU + ak * tmp;
      FermOp.MooDeriv(tmp, Y, X, DaggerNo);  dSdU = dSdU + ak * tmp;
      FermOp.MooDeriv(tmp, X, Y, DaggerYes); dSdU = dSdU + ak * tmp;

      FermionField W_e(fcbgrid), Z_e(fcbgrid), tmp1(fcbgrid);
      FermOp.Meooe(X, tmp1);      FermOp.MooeeInv(tmp1, W_e);
      FermOp.MeooeDag(Y, tmp1);   FermOp.MooeeInvDag(tmp1, Z_e);
      FermOp.MeeDeriv(tmp, Z_e, W_e, DaggerNo);  dSdU = dSdU + ak * tmp;
      FermOp.MeeDeriv(tmp, W_e, Z_e, DaggerYes); dSdU = dSdU + ak * tmp;
    }
  }

 private:
  FermOpF &opF_;
  GridBase *sp_rbgrid_;
  int reliable_freq_;
};

}  // namespace Grid
