// 2-flavor Möbius DWF HMC via EOFA (two independent 1-flavor EOFA actions).
//
// Fermion determinant: [det(M_phys) / det(M_PV)]^2
//
// Uses two ExactOneFlavourRatioPseudoFermionAction instances, each
// contributing det(M_phys)/det(M_PV) via the EOFA algebraic identity.
// Independent pseudofermion fields for each flavor.
//
// This should produce an identical ensemble to dw2f_cg_mobius.cc —
// both represent the same fermion determinant, via different algorithms.
//
// Boundary conditions: periodic in x,y,z; antiperiodic in t.

#include <Grid/Grid.h>

namespace Grid {

  struct FermionParameters : Serializable {
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

  struct MobiusHMCParameters : Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(MobiusHMCParameters,
                                    double,            gauge_beta,
                                    FermionParameters, Mobius)

    template <class ReaderClass>
    MobiusHMCParameters(Reader<ReaderClass> &Reader) {
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

  typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
  typedef WilsonImplR                   FermionImplPolicy;
  typedef MobiusFermionD                FermionAction;
  typedef typename FermionAction::FermionField FermionField;
  typedef Grid::XmlReader               Serialiser;

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

  CheckpointerParameters CPparams(Reader);
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);
  RNGModuleParameters RNGpar(Reader);
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<HMCWrapper::ImplPolicy>  PlaqObs;
  typedef PolyakovMod<HMCWrapper::ImplPolicy>   PolyakovObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  TheHMC.Resources.AddObservable<PolyakovObs>();

  WilsonGaugeActionR Waction(MyParams.gauge_beta);

  const int Ls   = MyParams.Mobius.Ls;
  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();
  auto FGrid     = SpaceTimeGrid::makeFiveDimGrid(Ls, GridPtr);
  auto FrbGrid   = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, GridPtr);

  LatticeGaugeField U(GridPtr);

  Real  mass = MyParams.Mobius.mass;
  Real  pv   = 1.0;
  RealD M5   = MyParams.Mobius.M5;
  RealD b    = MyParams.Mobius.b;
  RealD c    = MyParams.Mobius.c;

  std::cout << GridLogMessage << "boundary condition {1,1,1,-1}" << std::endl;
  std::vector<Complex> boundary = {1, 1, 1, -1};
  FermionAction::ImplParams Params(boundary);

  // EOFA rational approximation parameters (same as 1-flavor test)
  OneFlavourRationalParams OFRp;
  OFRp.lo        = 0.98;
  OFRp.hi        = 25.0;
  OFRp.MaxIter   = 10000;
  OFRp.tolerance = 1.0e-7;
  OFRp.degree    = 10;
  OFRp.precision = 40;

  ConjugateGradient<FermionField> CG(MyParams.Mobius.StoppingCondition,
                                     MyParams.Mobius.MaxCGIterations);

  // ── Flavor 1 EOFA operators ──────────────────────────────────────────────
  // Each pair (Op_L, Op_R) carries an independent pseudofermion field.
  // Each action contributes det(M_phys)/det(M_PV) via the EOFA identity.
  MobiusEOFAFermionD Op_L1(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr,
                            mass, mass, pv,   0.0, -1, M5, b, c, Params);
  MobiusEOFAFermionD Op_R1(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr,
                            pv,   mass, pv,  -1.0,  1, M5, b, c, Params);
  ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> EOFA1(Op_L1, Op_R1, CG, OFRp, true);
  EOFA1.is_smeared = ApplySmearing;

  // ── Flavor 2 EOFA operators ──────────────────────────────────────────────
  // Identical parameters to flavor 1; independent phi field gives the
  // second power of the determinant.
  MobiusEOFAFermionD Op_L2(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr,
                            mass, mass, pv,   0.0, -1, M5, b, c, Params);
  MobiusEOFAFermionD Op_R2(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr,
                            pv,   mass, pv,  -1.0,  1, M5, b, c, Params);
  ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> EOFA2(Op_L2, Op_R2, CG, OFRp, true);
  EOFA2.is_smeared = ApplySmearing;

  // ── Action levels ────────────────────────────────────────────────────────
  // Both EOFA actions on the fermion level; gauge on its own level.
  ActionLevel<HMCWrapper::Field> Level1(1);  // fermions
  Level1.push_back(&EOFA1);
  Level1.push_back(&EOFA2);

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
