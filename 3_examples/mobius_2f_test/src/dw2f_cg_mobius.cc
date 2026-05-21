// 2-flavor Möbius DWF HMC via exact CG (ratio pseudofermion).
//
// Fermion determinant: [det(M_phys) / det(M_PV)]^2
//
// Uses TwoFlavourEvenOddRatioPseudoFermionAction — standard CG on M†M,
// no rational approximation. This is the reference implementation for
// comparison against dw2f_eofa_mobius.cc.
//
// Grid naming convention (counterintuitive):
//   constructor(NumOp, DenOp)  -->  weight = det(DenOp) / det(NumOp)
//   NumOp = V = PV operator (mass=1.0)    --> denominator of ratio
//   DenOp = M = physical operator (mass)  --> numerator of ratio
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

  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  typedef PolyakovMod<HMCWrapper::ImplPolicy>  PolyakovObs;
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

  // ── Fermion operators ────────────────────────────────────────────────────
  // NumOp (V): PV regulator, mass=1.0 — appears in denominator of det ratio
  // DenOp (M): physical quark, mass=0.1 — appears in numerator of det ratio
  // Weight contribution: [det(M_phys) / det(M_PV)]^2
  FermionAction NumOp(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr, pv,   M5, b, c, Params);
  FermionAction DenOp(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr, mass, M5, b, c, Params);

  ConjugateGradient<FermionField> CG(MyParams.Mobius.StoppingCondition,
                                     MyParams.Mobius.MaxCGIterations);

  // Exact 2-flavor pseudofermion: no rational approximation
  TwoFlavourEvenOddRatioPseudoFermionAction<FermionImplPolicy> Nf2(NumOp, DenOp, CG, CG);
  Nf2.is_smeared = ApplySmearing;

  // ── Action levels ────────────────────────────────────────────────────────
  ActionLevel<HMCWrapper::Field> Level1(1);  // fermion
  Level1.push_back(&Nf2);

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
