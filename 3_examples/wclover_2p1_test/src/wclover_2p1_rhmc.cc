// 2+1 flavor Wilson Clover HMC.
//
// Fermion determinant: [det(M_l)/det(M_PV)]^2 * [det(M_h)/det(M_PV)]
//
// Light sector (2 degenerate flavors):
//   TwoFlavourEvenOddRatioPseudoFermionAction — exact CG, no rational approx.
//
// Heavy sector (1 flavor):
//   OneFlavourEvenOddRatioRationalPseudoFermionAction — RHMC rational approx
//   for [det(M_h)/det(M_PV)]^{1/2}.
//
// Note: EOFA is not available for Wilson Clover — it requires CayleyFermion5D
// (Möbius/DWF only). RHMC is the standard approach for an odd fermion flavor.
//
// Wilson Clover is 4D: no Ls, FGrid, b, c, M5.
// Boundary conditions: periodic in x,y,z; antiperiodic in t.

#include <Grid/Grid.h>

namespace Grid {

  struct CloverFermionParameters : Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(CloverFermionParameters,
                                    double, mass,
                                    double, csw_r,
                                    double, csw_t,
                                    double, StoppingCondition,
                                    double, MDStoppingCondition,
                                    int,    MaxCGIterations);
  };

  // Rational approximation bounds for the heavy single-flavor RHMC.
  // lo/hi bound the eigenvalues of (M_h^dag M_h)_eo.
  // Measure with a short run; lo ~ m_h^2, hi ~ O(10).
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

  struct WilsonCloverHMCParameters : Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(WilsonCloverHMCParameters,
                                    double,                  gauge_beta,
                                    bool,                    ApplySmearing,
                                    CloverFermionParameters, Light,
                                    CloverFermionParameters, Heavy,
                                    RHMCParameters,          HeavyRHMC)

    template <class ReaderClass>
    WilsonCloverHMCParameters(Reader<ReaderClass> &Reader) {
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

}

int main(int argc, char **argv) {
  using namespace Grid;

  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid is setup to use "
            << GridThread::GetThreads() << " threads" << std::endl;

  typedef GenericHMCRunner<MinimumNorm2>           HMCWrapper;
  typedef WilsonImplR                              FermionImplPolicy;
  typedef WilsonCloverFermionD                     FermionAction;
  typedef typename FermionAction::FermionField     FermionField;
  typedef Grid::XmlReader                          Serialiser;

  HMCWrapper TheHMC;
  TheHMC.ReadCommandLine(argc, argv);

  if (TheHMC.ParameterFile.empty()) {
    std::cout << "Input file not specified. "
              << "Use --ParameterFile option.\nAborting" << std::endl;
    exit(1);
  }
  Serialiser Reader(TheHMC.ParameterFile);
  WilsonCloverHMCParameters MyParams(Reader);
  bool ApplySmearing = MyParams.ApplySmearing;

  TheHMC.Resources.AddFourDimGrid("gauge");

  CheckpointerParameters CPparams(Reader);
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);
  RNGModuleParameters RNGpar(Reader);
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<HMCWrapper::ImplPolicy>  PlaqObs;
  typedef PolyakovMod<HMCWrapper::ImplPolicy>   PolyakovObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  TheHMC.Resources.AddObservable<PolyakovObs>();

  WilsonGaugeActionR Waction(MyParams.gauge_beta);

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();

  LatticeGaugeField U(GridPtr);

  // Periodic xyz, antiperiodic t
  std::cout << GridLogMessage << "boundary condition {1,1,1,-1}" << std::endl;
  std::vector<Complex> boundary = {1, 1, 1, -1};
  FermionAction::ImplParams FermParams(boundary);

  // ── Fermion operators ────────────────────────────────────────────────────
  // Convention for ratio actions: constructor(NumOp, DenOp) → det(DenOp)/det(NumOp)
  // NumOp = Pauli-Villars (mass=1.0) — denominator of det ratio
  // DenOp = physical quark            — numerator of det ratio
  Real m_l = MyParams.Light.mass;
  Real m_h = MyParams.Heavy.mass;
  Real pv  = 1.0;

  WilsonAnisotropyCoefficients anis;  // isotropic: xi_0=1, nu=1, no anisotropy

  FermionAction LightOp(U, *GridPtr, *GridRBPtr, m_l, MyParams.Light.csw_r, MyParams.Light.csw_t, anis, FermParams);
  FermionAction LightPV(U, *GridPtr, *GridRBPtr, pv,  MyParams.Light.csw_r, MyParams.Light.csw_t, anis, FermParams);

  FermionAction HeavyOp(U, *GridPtr, *GridRBPtr, m_h, MyParams.Heavy.csw_r, MyParams.Heavy.csw_t, anis, FermParams);
  FermionAction HeavyPV(U, *GridPtr, *GridRBPtr, pv,  MyParams.Heavy.csw_r, MyParams.Heavy.csw_t, anis, FermParams);

  // ── Solvers ──────────────────────────────────────────────────────────────
  ConjugateGradient<FermionField> CG_l(MyParams.Light.StoppingCondition,
                                       MyParams.Light.MaxCGIterations);
  ConjugateGradient<FermionField> CG_h(MyParams.Heavy.StoppingCondition,
                                       MyParams.Heavy.MaxCGIterations);

  // ── RHMC rational approx for the heavy single flavor ─────────────────────
  // Approximates [det(M_h)/det(M_PV)]^{1/2} on eigenvalue range [lo, hi]
  // of (M_h^dag M_h)_eo. Verify lo < lambda_min and hi > lambda_max before
  // production runs.
  OneFlavourRationalParams OFRp;
  OFRp.lo             = MyParams.HeavyRHMC.lo;
  OFRp.hi             = MyParams.HeavyRHMC.hi;
  OFRp.MaxIter        = MyParams.HeavyRHMC.MaxIter;
  OFRp.tolerance      = MyParams.HeavyRHMC.tolerance;
  OFRp.mdtolerance    = MyParams.HeavyRHMC.mdtolerance;
  OFRp.degree         = MyParams.HeavyRHMC.degree;
  OFRp.precision      = MyParams.HeavyRHMC.precision;
  OFRp.BoundsCheckFreq = MyParams.HeavyRHMC.BoundsCheckFreq;

  // ── Pseudofermion actions ─────────────────────────────────────────────────

  // Non-EO variants required throughout: WilsonClover has ConstEE()=0 because
  // the clover term makes the even-even block gauge-field dependent. Both
  // TwoFlavourEvenOddRatio and OneFlavourEvenOddRatioRational assert ConstEE==1.

  // 2-flavor light: exact CG, [det(M_l)/det(M_PV)]^2
  TwoFlavourRatioPseudoFermionAction<FermionImplPolicy>
      Nf2(LightPV, LightOp, CG_l, CG_l);
  Nf2.is_smeared = ApplySmearing;

  // 1-flavor heavy: RHMC, [det(M_h)/det(M_PV)]
  OneFlavourRatioRationalPseudoFermionAction<FermionImplPolicy>
      Nf1(HeavyPV, HeavyOp, OFRp);
  Nf1.is_smeared = ApplySmearing;

  // ── Action levels — 3-level nested integrator ────────────────────────────
  // Level step multipliers are relative to the outermost level.
  // Light fermion force is largest; heavy is intermediate; gauge is cheapest.
  // Tune ratios (1:2:8 here) by measuring force norms from a short run.
  ActionLevel<HMCWrapper::Field> Level1(1);   // light fermion
  Level1.push_back(&Nf2);

  ActionLevel<HMCWrapper::Field> Level2(2);   // heavy fermion
  Level2.push_back(&Nf1);

  ActionLevel<HMCWrapper::Field> Level3(8);   // gauge
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
