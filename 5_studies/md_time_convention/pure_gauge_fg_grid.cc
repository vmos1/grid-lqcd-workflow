// Pure-gauge HMC on the production configuration: how the gauge monomial ALONE responds to the MD
// step count at the production trajectory length. Luscher-Weisz gauge action only (no fermions),
// single level, Metropolis on, started from an imported ILDG configuration (cfg_2000, 48^3x96).
// The gauge level of production base+G runs the same ForceGradient recursion at h = TRAJL/48
// (12 outer x 2 x GAUGE_INNER_MULT 2), so PG_MDSTEPS=48 here is the gauge level of base+G.
//
// Derived from pure_gauge_hmc_grid.cc (same directory, unchanged). Differences:
//   1. Integrator chosen at run time: GenericHMCRunner<ForceGradient> (default) or <LeapFrog>.
//   2. Start from an external ILDG .lime (PG_IMPORT_CFG) with fresh RNG seeds from PG_SEED.
//      GenericHMCRunner's start types cannot take an externally loaded field without an RNG
//      checkpoint file (CheckpointStartReseed restores an RNG file first), and its Runner() is
//      private, so main() runs the body of HMCWrapperTemplate::Runner() itself on the runner's own
//      Resources / TheAction / Parameters: AddRNGs + SeedFixedIntegers (the ColdStart seeding),
//      the field read with IldgReader exactly as the production driver's IMPORT_CFG path
//      (gen_qcd_hasenbusch_tune_compact_schur.cc), then the integrator, HybridMonteCarlo, evolve.
//   3. Gauge action fixed to the production form: PlaqPlusRectangleActionR(beta, -beta/(20 u0^2)),
//      beta 6.3, u0 0.84570646270714 by default (driver "LW gauge action" block).
//   4. PerformRandomShift = false, as the production driver (Grid's HMCparameters default is true).
//   A NERSC checkpointer is loaded only because Resources.GetObservables() requires one; its save
//   interval exceeds any trajectory number, so nothing is ever written.
//
// Env:
//   PG_IMPORT_CFG   ILDG .lime gauge configuration (required)
//   PG_INTEGRATOR   forcegradient (default) | leapfrog
//   PG_MDSTEPS      default 48;  PG_TRAJL default 1.0;  PG_NTRAJ default 1 (all with Metropolis)
//   PG_SEED         added to every RNG seed element (default 0); same seed = same momentum draws
//   PG_BETA         default 6.3; PG_U0 default 0.84570646270714
// Usage: pure_gauge_fg_grid --grid 48.48.48.96 --mpi 1.1.2.2 --accelerator-threads 8 ...

#include <Grid/Grid.h>
#include <cstdlib>
#include <iomanip>
#include <string>

using namespace Grid;

static double env_real(const char *name, double def)
{
  const char *s = std::getenv(name);
  return (s && *s) ? std::atof(s) : def;
}

static int env_int(const char *name, int def)
{
  const char *s = std::getenv(name);
  return (s && *s) ? std::atoi(s) : def;
}

static std::string env_str(const char *name, const std::string &def)
{
  const char *s = std::getenv(name);
  return (s && *s) ? std::string(s) : def;
}

template <template <typename, typename, typename> class Integ>
static void run_hmc(int argc, char **argv, const std::string &integ_name)
{
  typedef GenericHMCRunner<Integ> HMCWrapper;
  typedef typename HMCWrapper::ImplPolicy Impl;
  HMCWrapper TheHMC;

  TheHMC.Resources.AddFourDimGrid("gauge");

  CheckpointerParameters CPparams;          // never saves: interval > any trajectory number
  CPparams.config_prefix = "ckpoint_lat";
  CPparams.rng_prefix    = "ckpoint_rng";
  CPparams.saveInterval  = 1000000000;
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  const int seed = env_int("PG_SEED", 0);
  RNGModuleParameters RNGpar;
  RNGpar.serial_seeds   = std::to_string(1 + seed) + " " + std::to_string(2 + seed) + " " +
                          std::to_string(3 + seed) + " " + std::to_string(4 + seed) + " " +
                          std::to_string(5 + seed);
  RNGpar.parallel_seeds = std::to_string(6 + seed) + " " + std::to_string(7 + seed) + " " +
                          std::to_string(8 + seed) + " " + std::to_string(9 + seed) + " " +
                          std::to_string(10 + seed);
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<Impl> PlaqObs;
  TheHMC.Resources.template AddObservable<PlaqObs>();

  const RealD beta  = env_real("PG_BETA", 6.3);
  const RealD u0    = env_real("PG_U0", 0.84570646270714);
  const RealD c_rect = -beta / (20.0 * u0 * u0);
  PlaqPlusRectangleActionR LW(beta, c_rect);
  ActionLevel<typename HMCWrapper::Field> Level1(1);
  Level1.push_back(&LW);
  TheHMC.TheAction.push_back(Level1);

  TheHMC.Parameters.MD.name           = integ_name;
  TheHMC.Parameters.MD.MDsteps        = env_int("PG_MDSTEPS", 48);
  TheHMC.Parameters.MD.trajL          = env_real("PG_TRAJL", 1.0);
  TheHMC.Parameters.Trajectories      = env_int("PG_NTRAJ", 1);
  TheHMC.Parameters.NoMetropolisUntil = 0;
  TheHMC.Parameters.MetropolisTest    = true;
  TheHMC.Parameters.PerformRandomShift = false;
  TheHMC.Parameters.StartTrajectory   = 0;
  TheHMC.Parameters.StartingType      = "ImportILDG";   // informational; Runner() is not used

  const std::string cfg = env_str("PG_IMPORT_CFG", "");
  if (cfg.empty()) {
    std::cout << GridLogError << "PG_IMPORT_CFG must name an ILDG .lime configuration" << std::endl;
    exit(1);
  }

  std::cout << GridLogMessage << std::setprecision(15) << "PURE_GAUGE_FG integrator=" << integ_name
            << " action=lw beta=" << beta << " u0=" << u0 << " c_rect=" << c_rect
            << " trajL=" << TheHMC.Parameters.MD.trajL
            << " MDsteps=" << TheHMC.Parameters.MD.MDsteps
            << " h=" << TheHMC.Parameters.MD.trajL / TheHMC.Parameters.MD.MDsteps
            << " ntraj=" << TheHMC.Parameters.Trajectories << " seed=" << seed
            << " random_shift=no import=" << cfg << std::setprecision(6) << std::endl;

  // ---- body of HMCWrapperTemplate::Runner(), with the start replaced by the ILDG import ----
  GridCartesian *UGrid = TheHMC.Resources.GetCartesian();
  LatticeGaugeField U(UGrid);

  TheHMC.Resources.AddRNGs();
  TheHMC.Resources.SeedFixedIntegers();     // as ColdStart / CheckpointStartReseed: from PG_SEED

  {
    FieldMetaData header;
    IldgReader IR;
    IR.open(cfg);
    IR.readConfiguration(U, header);
    IR.close();
  }
  std::cout << GridLogMessage << "Initial plaquette = " << std::setprecision(10)
            << WilsonLoops<Impl>::avgPlaquette(U) << std::setprecision(6) << std::endl;

  NoSmearing<Impl> Smearing;
  typedef typename HMCWrapper::template IntegratorType<NoSmearing<Impl>> TheIntegrator;
  TheIntegrator MDynamics(UGrid, TheHMC.Parameters.MD, TheHMC.TheAction, Smearing);
  MDynamics.setMomentumFilter(*(TheHMC.Resources.GetMomentumFilter()));
  Smearing.set_Field(U);

  HybridMonteCarlo<TheIntegrator> HMC(TheHMC.Parameters, MDynamics,
                                      TheHMC.Resources.GetSerialRNG(),
                                      TheHMC.Resources.GetParallelRNG(),
                                      TheHMC.Resources.GetObservables(), U);
  HMC.evolve();

  std::cout << GridLogMessage << "Final plaquette = " << std::setprecision(10)
            << WilsonLoops<Impl>::avgPlaquette(U) << std::setprecision(6) << std::endl;
}

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  GridLogLayout();

  const std::string integ = env_str("PG_INTEGRATOR", "forcegradient");
  if (integ == "forcegradient") {
    run_hmc<ForceGradient>(argc, argv, "ForceGradient");
  } else if (integ == "leapfrog") {
    run_hmc<LeapFrog>(argc, argv, "LeapFrog");
  } else {
    std::cout << GridLogError << "PG_INTEGRATOR must be forcegradient or leapfrog, got " << integ
              << std::endl;
    exit(1);
  }

  Grid_finalize();
}
