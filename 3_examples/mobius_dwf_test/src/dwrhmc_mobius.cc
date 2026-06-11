// Möbius DWF HMC with 1-flavor RHMC pseudofermion action.
// Companion to dweofa_mobius.cc — same parameters, different algorithm.
// Use identical XML input for a direct EOFA vs RHMC comparison.
//
// Algorithm difference vs EOFA:
//   EOFA: ExactOneFlavourRatioPseudoFermionAction (MobiusEOFAFermionD)
//   RHMC: OneFlavourEvenOddRatioRationalPseudoFermionAction (MobiusFermionD)
//         rational approx of [det(M_phys)/det(M_PV)]^{1/2}
//
// Based on dweofa_mobius_1flavor.cc (Sungwoo / Peter Boyle / Guido Cossu)

/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Copyright (C) 2015-2016
Licensed under GPL v2 or later.
*************************************************************************************/

#include <Grid/Grid.h>

namespace Grid {

  struct FermionParameters: Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(FermionParameters,
                                    int,    Ls,
                                    double, mass,
                                    double, M5,
                                    double, b,
                                    double, c,
                                    double, StoppingCondition,
                                    int,    MaxCGIterations,
                                    bool,   ApplySmearing);
  };

  struct MobiusHMCParameters: Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(MobiusHMCParameters,
                                    double,           gauge_beta,
                                    FermionParameters, Mobius)

    template <class ReaderClass>
    MobiusHMCParameters(Reader<ReaderClass>& Reader) {
      read(Reader, "Action", *this);
    }
  };

  struct SmearingParam: Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(SmearingParam,
                                    double,  rho,
                                    Integer, Nsmear)

    template <class ReaderClass>
    SmearingParam(Reader<ReaderClass>& Reader) {
      read(Reader, "StoutSmearing", *this);
    }
  };

}


int main(int argc, char **argv) {
  using namespace Grid;

  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid is setup to use "
            << GridThread::GetThreads() << " threads" << std::endl;

  typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
  typedef WilsonImplR                   FermionImplPolicy;
  typedef MobiusFermionD                FermionAction;   // standard Möbius, no EOFA shifts
  typedef typename FermionAction::FermionField FermionField;
  typedef Grid::XmlReader               Serialiser;

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  HMCWrapper TheHMC;
  TheHMC.ReadCommandLine(argc, argv);

  if (TheHMC.ParameterFile.empty()) {
    std::cout << "Input file not specified. "
              << "Use --ParameterFile option.\nAborting" << std::endl;
    exit(1);
  }
  Serialiser Reader(TheHMC.ParameterFile);
  MobiusHMCParameters MyParams(Reader);
  bool ApplySmearing = MyParams.Mobius.ApplySmearing;

  TheHMC.Resources.AddFourDimGrid("gauge");

  // ── RHMC rational approximation parameters ──────────────────────────────────
  // Bounds on the eigenvalues of M_phys†M_phys (even-odd preconditioned).
  // lo < min eigenvalue, hi > max eigenvalue.
  // For mass=0.1 Möbius DWF: lo~0.01 (≈mass^2), hi~30.
  // Increase degree for higher-mass quarks or finer lattices if accept rate drops.
  OneFlavourRationalParams OFRp;
  OFRp.lo        = 0.005;   // conservative lower bound
  OFRp.hi        = 100.0;   // measured lambda_max ~87 on 4^3x8; must exceed hi
  OFRp.MaxIter   = 10000;
  OFRp.tolerance = 1.0e-7;
  OFRp.degree    = 14;
  OFRp.precision = 50;

  // Checkpointer and RNG (same XML tags as EOFA code)
  CheckpointerParameters CPparams(Reader);
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);
  RNGModuleParameters RNGpar(Reader);
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  // Observables: Plaquette and Polyakov loop (same as EOFA)
  typedef PlaquetteMod<HMCWrapper::ImplPolicy>  PlaqObs;
  typedef PolyakovMod<HMCWrapper::ImplPolicy>   PolyakovObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  TheHMC.Resources.AddObservable<PolyakovObs>();

  // ── Gauge and fermion grid setup ─────────────────────────────────────────────
  WilsonGaugeActionR Waction(MyParams.gauge_beta);

  const int Ls    = MyParams.Mobius.Ls;
  auto GridPtr    = TheHMC.Resources.GetCartesian();
  auto GridRBPtr  = TheHMC.Resources.GetRBCartesian();
  auto FGrid      = SpaceTimeGrid::makeFiveDimGrid(Ls, GridPtr);
  auto FrbGrid    = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, GridPtr);

  LatticeGaugeField U(GridPtr);

  Real   mass = MyParams.Mobius.mass;
  Real   pv   = 1.0;
  RealD  M5   = MyParams.Mobius.M5;
  RealD  b    = MyParams.Mobius.b;
  RealD  c    = MyParams.Mobius.c;

  std::cout << GridLogMessage << "boundary condition {1,1,1,-1}" << std::endl;
  std::vector<Complex> boundary = {1,1,1,-1};
  FermionAction::ImplParams Params(boundary);

  ConjugateGradient<FermionField> CG(MyParams.Mobius.StoppingCondition,
                                     MyParams.Mobius.MaxCGIterations);

  // ── 1-flavor RHMC pseudofermion action ───────────────────────────────────────
  // Computes [det(M_phys) / det(M_PV)]^{1/2} via rational approximation.
  // NumOp = Pauli-Villars (mass=1), DenOp = physical (mass=0.1).
  FermionAction FermOp(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr, mass, M5, b, c, Params);
  FermionAction PVOp  (U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr, pv,   M5, b, c, Params);

  OneFlavourEvenOddRatioRationalPseudoFermionAction<FermionImplPolicy>
      RHMC(PVOp, FermOp, OFRp);

  RHMC.is_smeared = ApplySmearing;

  // ── Action levels (same structure as EOFA) ───────────────────────────────────
  ActionLevel<HMCWrapper::Field> Level1(1);  // fermion
  Level1.push_back(&RHMC);

  ActionLevel<HMCWrapper::Field> Level2(4);  // gauge
  Level2.push_back(&Waction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);

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
