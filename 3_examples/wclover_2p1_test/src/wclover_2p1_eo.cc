// 2+1 flavor Wilson Clover HMC — EO-preconditioned version.
//
// Fermion determinant: [det(M_l)/det(M_PV)]^2 * [det(M_h)/det(M_PV)]
//
// EO decomposition: det(M) = det(Mee) * det(Mpc)
//
//   Light sector (Nf=2):
//     QCDLogDetCloverEOAction      — S = -2 ln|det(Mee)|, gauge force via σ-loop
//     TwoFlavourSchurCloverAction  — S = Phi†(Mpc†Mpc)^{-1}Phi on odd sublattice
//     Solver: mixed-precision CG (SP inner + DP correction)
//
//   Heavy/strange sector (Nf=1):
//     QCDLogDetCloverEOAction      — S = -ln|det(Mee)|
//     OneFlavourSchurCloverRationalActionMP — RHMC on odd sublattice, MP multishift
//
// Build against install-txqcd-gpu (QCDLogDetCloverEOAction and
// TwoFlavourSchurCloverAction are TXQCD-fork additions, not in mainline Grid).
// The binary does plain QCD — no TXQCD auxiliary fields.

#include <Grid/Grid.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>
#include <Grid/algorithms/iterative/ConjugateGradientMixedPrec.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h>

namespace Grid {

// ---------------------------------------------------------------------------
// Mixed-precision CG wrapper satisfying OperatorFunction<FieldD>.
// The SP Schur operator must be refreshed from the current gauge field before
// each call (done by TwoFlavourSchurCloverActionMP::deriv below).
// ---------------------------------------------------------------------------
template <class FieldD, class FieldF, class SchurOpD, class SchurOpF>
class MixedPrecCGWrapper : public OperatorFunction<FieldD> {
 public:
  using OperatorFunction<FieldD>::operator();

  MixedPrecCGWrapper(RealD tol, int max_inner, int max_outer,
                     GridBase *rbgrid_f,
                     SchurOpD &schur_d, SchurOpF &schur_f)
      : tol_(tol), max_inner_(max_inner), max_outer_(max_outer),
        rbgrid_f_(rbgrid_f), schur_d_(schur_d), schur_f_(schur_f) {}

  void operator()(LinearOperatorBase<FieldD> &, const FieldD &src,
                  FieldD &sol) override {
    MixedPrecisionConjugateGradient<FieldD, FieldF> MPCG(
        tol_, max_inner_, max_outer_, rbgrid_f_, schur_f_, schur_d_);
    MPCG(src, sol);
  }

 private:
  RealD tol_;
  int max_inner_, max_outer_;
  GridBase *rbgrid_f_;
  SchurOpD &schur_d_;
  SchurOpF &schur_f_;
};

// ---------------------------------------------------------------------------
// TwoFlavourSchurCloverAction subclass that keeps the SP operator in sync
// with the current gauge field before each deriv() call.
// ---------------------------------------------------------------------------
template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class TwoFlavourSchurCloverActionMP
    : public TwoFlavourSchurCloverAction<ImplD, FermOpD_> {
 public:
  typedef TwoFlavourSchurCloverAction<ImplD, FermOpD_> Base;
  typedef typename ImplD::GaugeField GaugeField;

  TwoFlavourSchurCloverActionMP(typename Base::FermionOperator &opD,
                                FermOpF_ &opF,
                                OperatorFunction<typename Base::FermionField> &DS,
                                OperatorFunction<typename Base::FermionField> &AS)
      : Base(opD, DS, AS), opF_(opF) {}

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    typename ImplF::GaugeField UmuF(opF_.GaugeGrid());
    typename ImplD::GaugeLinkField U_d(U.Grid());
    typename ImplF::GaugeLinkField U_f(opF_.GaugeGrid());
    for (int mu = 0; mu < Nd; ++mu) {
      U_d = PeekIndex<LorentzIndex>(U, mu);
      precisionChange(U_f, U_d);
      PokeIndex<LorentzIndex>(UmuF, U_f, mu);
    }
    opF_.ImportGauge(UmuF);
    Base::deriv(U, dSdU);
  }

 private:
  FermOpF_ &opF_;
};

}  // namespace Grid

using namespace Grid;

// ---------------------------------------------------------------------------
// XML parameter structs
// ---------------------------------------------------------------------------
struct CloverFermionParameters : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(CloverFermionParameters,
                                  double, mass,
                                  double, csw_r,
                                  double, csw_t,
                                  double, StoppingCondition,
                                  double, MDStoppingCondition,
                                  int,    MaxCGIterations);
};

struct RHMCParameters : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(RHMCParameters,
                                  double, lo,
                                  double, hi,
                                  int,    degree,
                                  int,    MaxIter,
                                  double, tolerance,
                                  double, mdtolerance,
                                  int,    precision,
                                  int,    BoundsCheckFreq);
};

struct WilsonCloverEOHMCParameters : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(WilsonCloverEOHMCParameters,
                                  double,                  gauge_beta,
                                  bool,                    ApplySmearing,
                                  CloverFermionParameters, Light,
                                  CloverFermionParameters, Strange,
                                  RHMCParameters,          StrangeRHMC)

  template <class ReaderClass>
  WilsonCloverEOHMCParameters(Reader<ReaderClass> &Reader) {
    read(Reader, "Action", *this);
  }
};

struct SmearingParam : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(SmearingParam,
                                  double,  rho,
                                  Integer, Nsmear)

  template <class ReaderClass>
  SmearingParam(Reader<ReaderClass> &Reader) {
    read(Reader, "StoutSmearing", *this);
  }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid is setup to use "
            << GridThread::GetThreads() << " threads" << std::endl;

  typedef GenericHMCRunner<MinimumNorm2>       HMCWrapper;
  typedef WilsonImplR                          FermionImplD;
  typedef WilsonImplF                          FermionImplF;
  typedef WilsonCloverFermion<FermionImplD, CloverHelpers<FermionImplD>> WCF;
  typedef WilsonCloverFermion<FermionImplF, CloverHelpers<FermionImplF>> WCF_f;
  typedef Grid::XmlReader                      Serialiser;

  HMCWrapper TheHMC;
  TheHMC.ReadCommandLine(argc, argv);

  if (TheHMC.ParameterFile.empty()) {
    std::cout << "Input file not specified. Use --ParameterFile.\nAborting" << std::endl;
    exit(1);
  }
  Serialiser Reader(TheHMC.ParameterFile);
  WilsonCloverEOHMCParameters MyParams(Reader);
  bool ApplySmearing = MyParams.ApplySmearing;

  TheHMC.Resources.AddFourDimGrid("gauge");

  CheckpointerParameters CPparams(Reader);
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);
  RNGModuleParameters RNGpar(Reader);
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  typedef PolyakovMod<HMCWrapper::ImplPolicy>  PolyakovObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  TheHMC.Resources.AddObservable<PolyakovObs>();

  // ── Gauge action ──────────────────────────────────────────────────────────
  WilsonGaugeActionR Waction(MyParams.gauge_beta);

  // ── Grids ─────────────────────────────────────────────────────────────────
  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();
  Coordinate latt = GridPtr->FullDimensions();
  Coordinate mpi  = GridDefaultMpi();

  // Single-precision grids for mixed-precision CG
  GridCartesian        GridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
  GridRedBlackCartesian RBGridF(&GridF);

  LatticeGaugeField U(GridPtr);

  // Antiperiodic BC in time
  std::vector<Complex> boundary = {1, 1, 1, -1};
  WilsonImplParams FermParamsD, FermParamsF;
  FermParamsD.boundary_phases = boundary;
  FermParamsF.boundary_phases = boundary;

  // ── Fermion operators ─────────────────────────────────────────────────────
  Real m_l = MyParams.Light.mass;
  Real m_s = MyParams.Strange.mass;
  Real pv  = 1.0;
  WilsonAnisotropyCoefficients anis;

  // Light (double + single precision)
  WCF   LightOp (U, *GridPtr, *GridRBPtr, m_l, MyParams.Light.csw_r, MyParams.Light.csw_t, anis, FermParamsD);
  WCF   LightPV (U, *GridPtr, *GridRBPtr, pv,  MyParams.Light.csw_r, MyParams.Light.csw_t, anis, FermParamsD);

  LatticeGaugeFieldF UF(&GridF);
  {
    LatticeColourMatrix  Umu_d(GridPtr);
    LatticeColourMatrixF Umu_f(&GridF);
    for (int mu = 0; mu < Nd; ++mu) {
      Umu_d = PeekIndex<LorentzIndex>(U, mu);
      precisionChange(Umu_f, Umu_d);
      PokeIndex<LorentzIndex>(UF, Umu_f, mu);
    }
  }
  WCF_f LightOpF(UF, GridF, RBGridF, m_l, MyParams.Light.csw_r, MyParams.Light.csw_t, anis, FermParamsF);

  // Strange (double + single precision)
  WCF   StrangeOp (U, *GridPtr, *GridRBPtr, m_s, MyParams.Strange.csw_r, MyParams.Strange.csw_t, anis, FermParamsD);
  WCF_f StrangeOpF(UF, GridF, RBGridF,      m_s, MyParams.Strange.csw_r, MyParams.Strange.csw_t, anis, FermParamsF);

  // ── Solvers ───────────────────────────────────────────────────────────────
  // Action solver: tight DP CG for accept/reject
  ConjugateGradient<LatticeFermion> CG_action(MyParams.Light.StoppingCondition,
                                              MyParams.Light.MaxCGIterations);

  // Derivative solver: mixed-precision CG for MD force
  SchurDifferentiableOperator<FermionImplD> SchurOpD(LightOp);
  SchurDifferentiableOperator<FermionImplF> SchurOpF(LightOpF);
  MixedPrecCGWrapper<LatticeFermion, LatticeFermionF,
                     SchurDifferentiableOperator<FermionImplD>,
                     SchurDifferentiableOperator<FermionImplF>>
      CG_md(MyParams.Light.MDStoppingCondition, MyParams.Light.MaxCGIterations,
            50, &RBGridF, SchurOpD, SchurOpF);

  // ── RHMC rational approx for the strange single flavor ────────────────────
  OneFlavourRationalParams OFRp;
  OFRp.lo             = MyParams.StrangeRHMC.lo;
  OFRp.hi             = MyParams.StrangeRHMC.hi;
  OFRp.MaxIter        = MyParams.StrangeRHMC.MaxIter;
  OFRp.tolerance      = MyParams.StrangeRHMC.tolerance;
  OFRp.mdtolerance    = MyParams.StrangeRHMC.mdtolerance;
  OFRp.degree         = MyParams.StrangeRHMC.degree;
  OFRp.precision      = MyParams.StrangeRHMC.precision;
  OFRp.BoundsCheckFreq = MyParams.StrangeRHMC.BoundsCheckFreq;

  // ── EO pseudofermion actions ──────────────────────────────────────────────
  //
  // Light Nf=2:
  //   LightLogDet  — contributes -2 ln|det(Mee_l)| + gauge force
  //   LightSchurPF — pseudofermion on odd sublattice, MP CG
  //
  // Strange Nf=1:
  //   StrangeLogDet  — contributes -ln|det(Mee_s)| + gauge force
  //   StrangeSchurPF — rational PF on odd sublattice, MP multishift CG

  QCDLogDetCloverEOAction<FermionImplD> LightLogDet(LightOp, 2);
  LightLogDet.is_smeared = ApplySmearing;

  TwoFlavourSchurCloverActionMP<FermionImplD, FermionImplF> LightSchurPF(
      LightOp, LightOpF, CG_md, CG_action);
  LightSchurPF.is_smeared = ApplySmearing;

  QCDLogDetCloverEOAction<FermionImplD> StrangeLogDet(StrangeOp, 1);
  StrangeLogDet.is_smeared = ApplySmearing;

  OneFlavourSchurCloverRationalActionMP<FermionImplD, FermionImplF> StrangeSchurPF(
      StrangeOp, StrangeOpF, &RBGridF, OFRp, 50);
  StrangeSchurPF.is_smeared = ApplySmearing;

  // ── Action levels — 3-level nested integrator ─────────────────────────────
  // Level 1 (outermost): light fermion — most expensive, fewest steps
  // Level 2: strange fermion — intermediate
  // Level 3 (innermost): gauge — cheapest, most steps
  ActionLevel<HMCWrapper::Field> Level1(1);
  Level1.push_back(&LightLogDet);
  Level1.push_back(&LightSchurPF);

  ActionLevel<HMCWrapper::Field> Level2(2);
  Level2.push_back(&StrangeLogDet);
  Level2.push_back(&StrangeSchurPF);

  ActionLevel<HMCWrapper::Field> Level3(8);
  Level3.push_back(&Waction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);
  TheHMC.TheAction.push_back(Level3);

  TheHMC.Parameters.initialize(Reader);

  if (ApplySmearing) {
    SmearingParam SmPar(Reader);
    Smear_Stout<HMCWrapper::ImplPolicy> Stout(SmPar.rho);
    SmearedConfiguration<HMCWrapper::ImplPolicy> SmearingPolicy(GridPtr, SmPar.Nsmear, Stout);
    TheHMC.Run(SmearingPolicy);
  } else {
    TheHMC.Run();
  }

  Grid_finalize();
}
