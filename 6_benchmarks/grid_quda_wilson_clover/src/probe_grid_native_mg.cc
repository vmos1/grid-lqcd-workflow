// probe_grid_native_mg.cc
//
// GRID'S OWN Wilson multigrid, run on a REAL configuration.
//
// WHY THIS EXISTS.
//
// Every Grid MG number in this campaign came from probe_grid_mg_schur_clover.cc,
// which was assembled by hand from tests/debug/Test_general_coarse_wilson.cc --
// the DWF/HDCG *general-coarsening* stack. Grid ALSO ships a purpose-built
// Wilson/clover multigrid framework, tests/solver/Test_multigrid_common.h, with
// a factory (createMGInstance), a K-cycle, an FGMRES smoother, runChecks() and
// reportTimings(). We had never run it on a real configuration: the one attempt
// (2026-09-17) used a HOT gauge field, where aggregation has no smooth low modes
// to work with and the result is meaningless -- the same mistake the smearing and
// mass-scan work already caught twice.
//
// This file is a VERBATIM FORK of tests/solver/Test_wilson_mg.cc. Only the inputs
// change:
//   - the gauge field is read from ILDG and optionally stout-smeared, instead of
//     SU<Nc>::HotConfiguration
//   - mass is a command-line argument instead of the hardcoded -0.25
//   - the solver tolerance is a command-line argument
// The multigrid itself -- createMGInstance, the K-cycle, the smoother, the
// coarsening -- is Grid's, included unmodified from Grid's tests/solver. Nothing
// in Grid/ is edited, per the project's fork-don't-edit rule.
//
// WHAT IT ANSWERS. The stock test runs THREE solvers on the SAME system:
// CG, FGMRES with a trivial preconditioner, and FGMRES + MG. So it answers
// "is Grid's own multigrid faster than Grid's own CG" using nothing but Grid's
// machinery, with each solver as the other's control.
//
// ⚠️ IT PRECONDITIONS MdagMLinearOperator -- the FULL M^dag M, not the even-odd
// Schur operator our other probe and QUDA both use. The comparison here is
// therefore Grid-vs-Grid and internally consistent; it is NOT comparable to a
// QUDA MG number without separate care. See
// [[grid-quda-16x-is-preconditioning-not-backend]] for what happens when full-op
// and EO-preconditioned numbers get compared by accident.
//
// ⚠️ Needs an mg_params.xml in the working directory. Grid's XmlWriter flushes on
// DESTRUCTION and the missing-input abort happens in the same scope, so the
// mg_params_template.xml it claims to write never appears. Bools must be spelled
// `true`, not `1`.

#include <Grid/Grid.h>
#include <Test_multigrid_common.h>

using namespace std;
using namespace Grid;

namespace {
std::string read_string(int argc, char **argv, const std::string &option, const std::string &fallback)
{
  if (GridCmdOptionExists(argv, argv + argc, option)) return GridCmdOptionPayload(argv, argv + argc, option);
  return fallback;
}
double read_double(int argc, char **argv, const std::string &option, double fallback)
{
  const std::string payload = read_string(argc, argv, option, "");
  return payload.empty() ? fallback : std::stod(payload);
}
int read_int(int argc, char **argv, const std::string &option, int fallback)
{
  const std::string payload = read_string(argc, argv, option, "");
  return payload.empty() ? fallback : std::stoi(payload);
}
} // namespace

int main(int argc, char **argv) {

  Grid_init(&argc, &argv);

  GridCartesian *        FGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid);

  std::vector<int> fSeeds({1, 2, 3, 4});
  GridParallelRNG  fPRNG(FGrid);
  fPRNG.SeedFixedIntegers(fSeeds);

  LatticeFermion    src(FGrid); gaussian(fPRNG, src);
  LatticeFermion result(FGrid); result = Zero();
  LatticeGaugeField Umu(FGrid);

  // ---- inputs (the only departure from the stock test) --------------------
  const std::string cfg = read_string(argc, argv, "--probe-cfg", "");
  const RealD mass = read_double(argc, argv, "--probe-mass", -0.25);
  const RealD tol = read_double(argc, argv, "--probe-tol", 1.0e-12);
  const int maxiter = read_int(argc, argv, "--probe-maxiter", 50000);
  const int stout_nsmear = read_int(argc, argv, "--probe-stout-nsmear", 0);
  const RealD stout_rho = read_double(argc, argv, "--probe-stout-rho", 0.125);

  if (cfg.empty()) {
    SU<Nc>::HotConfiguration(fPRNG, Umu);
    std::cout << GridLogMessage << "gauge HOT (no --probe-cfg given)" << std::endl;
  } else {
    FieldMetaData header;
    IldgReader reader;
    reader.open(cfg);
    reader.readConfiguration(Umu, header);
    reader.close();
    std::cout << GridLogMessage << "gauge " << cfg << " plaquette "
              << WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu) << std::endl;
  }

  if (stout_nsmear > 0) {
    Smear_Stout<PeriodicGimplD> stout(stout_rho);
    LatticeGaugeField Usmear(FGrid);
    for (int n = 0; n < stout_nsmear; ++n) {
      stout.smear(Usmear, Umu);
      Umu = Usmear;
    }
    std::cout << GridLogMessage << "stout smearing rho " << stout_rho << " n_smear " << stout_nsmear
              << " -> smeared plaquette " << WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu)
              << std::endl;
  }
  std::cout << GridLogMessage << "mass " << mass << ", tol " << tol << std::endl;

  MultiGridParams mgParams;
  std::string     inputXml{"./mg_params.xml"};

  if(GridCmdOptionExists(argv, argv + argc, "--inputxml")) {
    inputXml = GridCmdOptionPayload(argv, argv + argc, "--inputxml");
    GRID_ASSERT(inputXml.length() != 0);
  }

  {
    XmlWriter writer("mg_params_template.xml");
    write(writer, "Params", mgParams);
    std::cout << GridLogMessage << "Written mg_params_template.xml" << std::endl;

    XmlReader reader(inputXml);
    read(reader, "Params", mgParams);
    std::cout << GridLogMessage << "Read in " << inputXml << std::endl;
  }

  checkParameterValidity(mgParams);
  std::cout << mgParams << std::endl;

  LevelInfo levelInfo(FGrid, mgParams);

  // Note: We do chiral doubling, so actually only nbasis/2 full basis vectors are used
  const int nbasis = 40;

  WilsonFermionD Dw(Umu, *FGrid, *FrbGrid, mass);

  MdagMLinearOperator<WilsonFermionD, LatticeFermion> MdagMOpDw(Dw);

  std::cout << GridLogMessage << "**************************************************" << std::endl;
  std::cout << GridLogMessage << "Testing Multigrid for Wilson" << std::endl;
  std::cout << GridLogMessage << "**************************************************" << std::endl;

  TrivialPrecon<LatticeFermion> TrivialPrecon;
  auto MGPreconDw = createMGInstance<vSpinColourVector, vTComplex, nbasis, WilsonFermionD>(mgParams, levelInfo, Dw, Dw);

  GridStopWatch setupTimer;
  setupTimer.Start();
  MGPreconDw->setup();
  setupTimer.Stop();
  std::cout << GridLogMessage << "MG setup " << setupTimer.Elapsed() << std::endl;

  if(GridCmdOptionExists(argv, argv + argc, "--runchecks")) {
    RealD toleranceForMGChecks = (getPrecision<LatticeFermion>::value == 1) ? 1e-6 : 1e-13;
    MGPreconDw->runChecks(toleranceForMGChecks);
  }

  std::vector<std::unique_ptr<OperatorFunction<LatticeFermion>>> solversDw;
  std::vector<std::string> solverNames;

  solversDw.emplace_back(new ConjugateGradient<LatticeFermion>(tol, maxiter, false));
  solverNames.push_back("CG");
  solversDw.emplace_back(new FlexibleGeneralisedMinimalResidual<LatticeFermion>(tol, maxiter, TrivialPrecon, 100, false));
  solverNames.push_back("FGMRES (trivial precon)");
  solversDw.emplace_back(new FlexibleGeneralisedMinimalResidual<LatticeFermion>(tol, maxiter, *MGPreconDw, 100, false));
  solverNames.push_back("FGMRES + MG");

  // One warm pass per solver before the timed one, so neither is charged for
  // warming the machine on the other's behalf -- matching the treatment the
  // other probes now use.
  for(size_t i = 0; i < solversDw.size(); ++i) {
    std::cout << std::endl << "Starting with a new solver: " << solverNames[i] << std::endl;
    result = Zero();
    (*solversDw[i])(MdagMOpDw, src, result);

    result = Zero();
    accelerator_barrier();
    FGrid->Barrier();
    const double t0 = usecond();
    (*solversDw[i])(MdagMOpDw, src, result);
    accelerator_barrier();
    FGrid->Barrier();
    const double secs = (usecond() - t0) / 1.0e6;

    // Independent residual on the same operator, outside the timed region.
    LatticeFermion check(FGrid);
    MdagMOpDw.Op(result, check);
    check = check - src;
    const RealD rel = std::sqrt(norm2(check) / norm2(src));
    std::cout << GridLogMessage << "SOLVER RESULT  " << solverNames[i] << "  " << secs
              << " s, independent residual " << rel << std::endl;
  }

  MGPreconDw->reportTimings();

  Grid_finalize();
}
