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
// M5 fused clover force (opt-in, HASEN_GRID_FUSED_CLOVER_FORCE=1): resolved through -I$HB/src
// (build_driver_stock.sh, build_test_*.sh). Local edit, not in the Grid-TXQCD fork copy.
#include "clover_force/fused_clover_force.h"
#include "gauge_import/import_guard.h"

namespace Grid {

// A2 phase timers (HASEN_GRID_STRANGE_TIMERS=1, read once, default OFF): deriv() brackets each
// phase with accelerator_barrier() + usecond() and prints one summary line per call on the
// boss rank. Pure measurement: no field is touched, so numerics are unchanged; the barriers
// serialise the stream, so the sum of phases is slightly above an untimed deriv.
inline bool StrangeTimersEnabled() {
  static const bool on = ImportSkipEnv("HASEN_GRID_STRANGE_TIMERS");
  return on;
}
struct StrangePhaseClock {
  bool on;
  double t;
  explicit StrangePhaseClock(bool o) : on(o), t(0.0) { mark(); }
  void mark() {
    if (!on) return;
    accelerator_barrier();
    t = usecond();
  }
  // add the time since the last mark/lap to acc (microseconds) and restart the clock
  void lap(double &acc) {
    if (!on) return;
    accelerator_barrier();
    const double n = usecond();
    acc += n - t;
    t = n;
  }
};

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

  // x: separate force degree/tolerance and the refresh-time bounds check (base header); the force
  // multishift below runs on PowerNegHalf, i.e. at x.md_degree and x.md_tolerance.
  OneFlavourSchurCloverRationalActionMP(FermOpD &opD, FermOpF &opF,
                                        GridBase *sp_rbgrid,
                                        OneFlavourRationalParams &p,
                                        int reliable_update_freq = 50,
                                        const OneFlavourSchurRationalExtras &x =
                                            OneFlavourSchurRationalExtras())
      : Base(opD, p, x), opF_(opF), sp_rbgrid_(sp_rbgrid),
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

    // A2 timers (microseconds); all clock calls are no-ops with the gate off.
    const bool tm = StrangeTimersEnabled();
    StrangePhaseClock clk(tm);
    double t_setup = 0, t_solve = 0, t_mpc = 0, t_mpcd = 0, t_mpcdag = 0, t_axpy = 0;
    double t_moo = 0, t_chain = 0, t_mee = 0, t_finish = 0;
    double t_impD = 0, t_prec = 0, t_impF = 0;

    GuardedImportGauge(FermOp, U);
    clk.lap(t_impD);

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
    clk.lap(t_prec);
    GuardedImportGauge(opF_, UmuF);
    clk.lap(t_impF);

    SchurDifferentiableOperator<ImplD> Mpc(FermOp);
    SchurDifferentiableOperator<ImplF> Mpc_f(opF_);

    ConjugateGradientMultiShiftMixedPrec<FermionField,
                                         typename ImplF::FermionField>
        msCG(this->param.MaxIter, PowerNegHalf, sp_rbgrid_, Mpc_f, reliable_freq_);
    clk.lap(t_setup);
    msCG(Mpc, PhiOdd, MPhi_k);
    clk.lap(t_solve);

    dSdU = Zero();
    clk.lap(t_axpy);
    if (FusedCloverForceEnabled()) {
      // M5 fused clover force (HASEN_GRID_FUSED_CLOVER_FORCE=1, src/clover_force/
      // fused_clover_force.h): per pole the hopping derivatives are unchanged; the four
      // MooDeriv/MeeDeriv calls of the loop below become four outer products with the pole's
      // residue a_k (the loop adds each result as dSdU + ak * tmp), accumulated over all poles,
      // and ONE clover pass after the loop (4*Npole calls -> 1).
      FusedCloverForce<ImplD> fused(fcbgrid);
      for (int k = 0; k < Npole; k++) {
        RealD ak = PowerNegHalf.residues[k];
        clk.mark();
        X = MPhi_k[k];
        Mpc.Mpc(X, Y);
        clk.lap(t_mpc);

        Mpc.MpcDeriv(tmp, Y, X);          clk.lap(t_mpcd);
        dSdU = dSdU + ak * tmp;           clk.lap(t_axpy);
        Mpc.MpcDagDeriv(tmp, X, Y);       clk.lap(t_mpcdag);
        dSdU = dSdU + ak * tmp;           clk.lap(t_axpy);
        fused.AccumulateMoo(ak, Y, X);    // FermOp.MooDeriv(Y, X) -> dSdU + ak * tmp
        fused.AccumulateMoo(ak, X, Y);    // FermOp.MooDeriv(X, Y) -> dSdU + ak * tmp
        clk.lap(t_moo);

        FermionField W_e(fcbgrid), Z_e(fcbgrid), tmp1(fcbgrid);
        FermOp.Meooe(X, tmp1);      FermOp.MooeeInv(tmp1, W_e);
        FermOp.MeooeDag(Y, tmp1);   FermOp.MooeeInvDag(tmp1, Z_e);
        clk.lap(t_chain);
        fused.AccumulateMee(ak, Z_e, W_e);  // FermOp.MeeDeriv(Z_e, W_e) -> dSdU + ak * tmp
        fused.AccumulateMee(ak, W_e, Z_e);  // FermOp.MeeDeriv(W_e, Z_e) -> dSdU + ak * tmp
        clk.lap(t_mee);
      }
      clk.mark();
      fused.Finish(FermOp, tmp, "OneFlavourSchurCloverRationalActionMP");
      dSdU = dSdU + tmp;
      clk.lap(t_finish);
      if (tm) StrangeTimersReport(U, Npole, t_setup, t_solve, t_mpc, t_mpcd, t_mpcdag, t_axpy,
                                  t_moo, t_chain, t_mee, t_finish, t_impD, t_prec, t_impF,
                                  tmp, Y, X);
      return;
    }
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
  // A2: print one summary line (seconds, boss rank) for the call just finished, plus a probe of
  // where one MpcDeriv spends its time. The probe re-runs the body of the stock
  // SchurDifferentiableOperator::MpcDeriv (public pieces only) once on the last pole's Y, X, into
  // the scratch field `scr`, with a clock between its steps; it changes no result.
  void StrangeTimersReport(const GaugeField &U, int Npole, double t_setup, double t_solve,
                           double t_mpc, double t_mpcd, double t_mpcdag, double t_axpy,
                           double t_moo, double t_chain, double t_mee, double t_finish,
                           double t_impD, double t_prec, double t_impF, GaugeField &scr, const FermionField &Yp, const FermionField &Xp) {
    auto &FermOp = this->FermOp;
    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();
    double p_grid = 0, p_chain = 0, p_hop = 0, p_asm = 0, p_neg = 0;
    StrangePhaseClock pc(true);
    GridRedBlackCartesian *forcecb = new GridRedBlackCartesian(scr.Grid());
    GaugeField ForceO(forcecb);
    GaugeField ForceE(forcecb);
    FermionField tmp1(fcbgrid);
    FermionField tmp2(fcbgrid);
    pc.lap(p_grid);
    const bool input_odd = (Yp.Checkerboard() == Odd);
    FermOp.Meooe(Xp, tmp1);
    FermOp.MooeeInv(tmp1, tmp2);
    pc.lap(p_chain);
    if (input_odd) FermOp.MoeDeriv(ForceO, Yp, tmp2, DaggerNo);
    else           FermOp.MeoDeriv(ForceE, Yp, tmp2, DaggerNo);
    pc.lap(p_hop);
    FermOp.MeooeDag(Yp, tmp1);
    FermOp.MooeeInvDag(tmp1, tmp2);
    pc.lap(p_chain);
    if (input_odd) FermOp.MeoDeriv(ForceE, tmp2, Xp, DaggerNo);
    else           FermOp.MoeDeriv(ForceO, tmp2, Xp, DaggerNo);
    pc.lap(p_hop);
    SchurDifferentiableOperator<ImplD>::AssembleForce(scr, ForceE, ForceO);
    pc.lap(p_asm);
    scr = -scr;
    pc.lap(p_neg);
    delete forcecb;
    if (U.Grid()->IsBoss()) {
      const double s = 1.0e-6;
      const double n = Npole;
      // std::fixed/setprecision are sticky on std::cout: restore them below so the log format of
      // every later line is exactly that of a run with the timers off.
      const std::ios_base::fmtflags f0 = std::cout.flags();
      const std::streamsize p0 = std::cout.precision();
      std::cout << GridLogMessage << std::fixed << std::setprecision(4)
                << "[StrangeTimers] deriv s: importD " << t_impD * s << " precisionChange "
                << t_prec * s << " importF " << t_impF * s << " setup(rest) " << t_setup * s
                << " solve " << t_solve * s
                << " | " << Npole << " poles: Mpc " << t_mpc * s << " MpcDeriv " << t_mpcd * s
                << " MpcDagDeriv " << t_mpcdag * s << " axpy(2/pole) " << t_axpy * s
                << " Moo(fused,2/pole) " << t_moo * s << " MeeChain " << t_chain * s
                << " Mee(fused,2/pole) " << t_mee * s << " | Finish+add " << t_finish * s
                << " || sum20 " << (t_mpcd + t_mpcdag) * s << " per-call "
                << (t_mpcd + t_mpcdag) * s / (2.0 * n) << std::endl;
      std::cout << GridLogMessage << std::fixed << std::setprecision(5)
                << "[StrangeTimers] probe, one MpcDeriv-equivalent s: forcecb+allocs "
                << p_grid * s << " Meooe/MooeeInv chains " << p_chain * s
                << " MoeDeriv+MeoDeriv " << p_hop * s << " AssembleForce " << p_asm * s
                << " negate " << p_neg * s << std::endl;
      std::cout.flags(f0);
      std::cout.precision(p0);
    }
  }

  FermOpF &opF_;
  GridBase *sp_rbgrid_;
  int reliable_freq_;
};

}  // namespace Grid
