// Pure-gauge HMC for the Grid vs Chroma MD-time test with forces on
// (__docs/2026_10_05_grid_chroma_md_time_handoff.md §6). Single-level leapfrog, Metropolis on.
// Chroma twin: pure_gauge_chroma.ini.xml.in. The step-size dependence of dH, compared with Chroma's
// at matched step counts, tests the momentum normalisation AND the force kick together; the
// equilibrium plaquette tests the gauge-action normalisation.
//
// Derived from stock tests/hmc/Test_hmc_WilsonGauge.cc (3d3eff86); only the parameters differ.
//
// Env:
//   PG_ACTION     wilson (default) | lw  (lw = PlaqPlusRectangle, c_rect = -beta/(20 u0^2), the
//                 production driver's Luscher-Weisz form)
//   PG_BETA       default 6.0;  PG_U0 default 1.0 (lw only)
//   PG_TRAJL      default 1.0;  PG_MDSTEPS default 10
//   PG_NTRAJ      trajectories with Metropolis (default 100)
//   PG_NOMETROP   leading trajectories without the Metropolis test (default 0; thermalization)
//   PG_START      cold (default) | ckpt | reseed  (ckpt: restore gauge + RNG of trajectory
//                 PG_START_TRAJ from PG_CKPT_DIR; reseed: restore the gauge field, then seed the
//                 RNGs from PG_SEED, so replicas from one start are independent chains)
//   PG_CKPT_DIR   checkpoint directory prefix (default "."); PG_SAVE save interval (default 1000000)
//   PG_SEED       added to every RNG seed element (default 0; cold and reseed starts)
// Usage: pure_gauge_hmc_grid --grid 8.8.8.8 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <cstdlib>
#include <string>

using namespace Grid;

static double env_real(const char *name, double def)
{
  const char *s = std::getenv(name);
  return s ? std::atof(s) : def;
}

static int env_int(const char *name, int def)
{
  const char *s = std::getenv(name);
  return s ? std::atoi(s) : def;
}

static std::string env_str(const char *name, const std::string &def)
{
  const char *s = std::getenv(name);
  return s ? std::string(s) : def;
}

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  GridLogLayout();

  typedef GenericHMCRunner<LeapFrog> HMCWrapper;
  HMCWrapper TheHMC;

  TheHMC.Resources.AddFourDimGrid("gauge");

  const std::string ckpt_dir = env_str("PG_CKPT_DIR", ".");
  CheckpointerParameters CPparams;
  CPparams.config_prefix = ckpt_dir + "/ckpoint_lat";
  CPparams.rng_prefix    = ckpt_dir + "/ckpoint_rng";
  CPparams.saveInterval  = env_int("PG_SAVE", 1000000);
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

  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();

  const std::string action = env_str("PG_ACTION", "wilson");
  const RealD beta = env_real("PG_BETA", 6.0);
  const RealD u0   = env_real("PG_U0", 1.0);
  WilsonGaugeActionR Wilson(beta);
  PlaqPlusRectangleActionR LW(beta, -beta / (20.0 * u0 * u0));

  ActionLevel<HMCWrapper::Field> Level1(1);
  if (action == "wilson") {
    Level1.push_back(&Wilson);
  } else if (action == "lw") {
    Level1.push_back(&LW);
  } else {
    std::cout << GridLogError << "PG_ACTION must be wilson or lw, got " << action << std::endl;
    exit(1);
  }
  TheHMC.TheAction.push_back(Level1);

  const std::string start = env_str("PG_START", "cold");
  TheHMC.Parameters.MD.MDsteps        = env_int("PG_MDSTEPS", 10);
  TheHMC.Parameters.MD.trajL          = env_real("PG_TRAJL", 1.0);
  TheHMC.Parameters.Trajectories      = env_int("PG_NTRAJ", 100);
  TheHMC.Parameters.NoMetropolisUntil = env_int("PG_NOMETROP", 0);
  TheHMC.Parameters.MetropolisTest    = true;
  if (start == "cold") {
    TheHMC.Parameters.StartingType    = "ColdStart";
    TheHMC.Parameters.StartTrajectory = 0;
  } else if (start == "ckpt") {
    TheHMC.Parameters.StartingType    = "CheckpointStart";
    TheHMC.Parameters.StartTrajectory = env_int("PG_START_TRAJ", 0);
  } else if (start == "reseed") {           // restore the gauge field, then reseed from PG_SEED
    TheHMC.Parameters.StartingType    = "CheckpointStartReseed";
    TheHMC.Parameters.StartTrajectory = env_int("PG_START_TRAJ", 0);
  } else {
    std::cout << GridLogError << "PG_START must be cold, ckpt or reseed, got " << start << std::endl;
    exit(1);
  }

  TheHMC.ReadCommandLine(argc, argv);

  std::cout << GridLogMessage << "PURE_GAUGE action=" << action << " beta=" << beta << " u0=" << u0
            << " trajL=" << TheHMC.Parameters.MD.trajL
            << " MDsteps=" << TheHMC.Parameters.MD.MDsteps
            << " ntraj=" << TheHMC.Parameters.Trajectories
            << " nometrop=" << TheHMC.Parameters.NoMetropolisUntil
            << " start=" << start << " start_traj=" << TheHMC.Parameters.StartTrajectory
            << " seed=" << seed << std::endl;

  TheHMC.Run();

  Grid_finalize();
}
