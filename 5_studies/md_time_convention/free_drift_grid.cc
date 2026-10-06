// Free-drift test of Grid's MD time unit against Chroma's
// (__docs/2026_10_05_grid_chroma_md_time_handoff.md §5).
//
// Cold start, Wilson gauge action at FD_BETA (default 0: the force is exactly zero), one
// trajectory of length FD_TRAJL in FD_MDSTEPS leapfrog steps, Metropolis off. With zero force
// the momenta never change, so every link ends at exp(TRAJL * P) and the final plaquette measures
// how far Grid moves the links per unit of its MD time. The Chroma twin is
// free_drift_chroma.ini.xml.in; the trajectory lengths at which the two plaquettes agree give the
// time conversion. With FD_BETA = 0 the action is zero, so "Total H before trajectory" is the
// kinetic energy alone.
//
// Derived from stock tests/hmc/Test_hmc_WilsonGauge.cc (3d3eff86); only the parameters differ.
//
// Env: FD_TRAJL (default 1.0), FD_MDSTEPS (default 10), FD_BETA (default 0.0),
//      FD_SEED (default 0, added to every RNG seed element).
// Usage: free_drift_grid --grid 8.8.8.8 --mpi 1.1.1.1

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

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  GridLogLayout();

  typedef GenericHMCRunner<LeapFrog> HMCWrapper;
  HMCWrapper TheHMC;

  TheHMC.Resources.AddFourDimGrid("gauge");

  // Never reached for one trajectory; the runner expects a checkpointer.
  CheckpointerParameters CPparams;
  CPparams.config_prefix = "ckpoint_lat";
  CPparams.rng_prefix    = "ckpoint_rng";
  CPparams.saveInterval  = 1000000;
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  const int seed = env_int("FD_SEED", 0);
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

  const RealD beta = env_real("FD_BETA", 0.0);
  WilsonGaugeActionR Waction(beta);

  ActionLevel<HMCWrapper::Field> Level1(1);
  Level1.push_back(&Waction);
  TheHMC.TheAction.push_back(Level1);

  TheHMC.Parameters.MD.MDsteps      = env_int("FD_MDSTEPS", 10);
  TheHMC.Parameters.MD.trajL        = env_real("FD_TRAJL", 1.0);
  TheHMC.Parameters.StartingType    = "ColdStart";
  TheHMC.Parameters.StartTrajectory = 0;
  TheHMC.Parameters.Trajectories    = 1;
  TheHMC.Parameters.NoMetropolisUntil = 0;
  TheHMC.Parameters.MetropolisTest  = false;

  TheHMC.ReadCommandLine(argc, argv);

  std::cout << GridLogMessage << "FREE_DRIFT beta=" << beta
            << " trajL=" << TheHMC.Parameters.MD.trajL
            << " MDsteps=" << TheHMC.Parameters.MD.MDsteps
            << " seed=" << seed << std::endl;

  TheHMC.Run();

  Grid_finalize();
}
