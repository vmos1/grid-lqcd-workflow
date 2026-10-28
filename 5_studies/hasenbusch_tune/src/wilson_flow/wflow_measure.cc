// wflow_measure.cc -- offline Wilson (gradient) flow on saved gauge configurations, for the
// Grid-vs-Chroma and candidate-vs-baseline physics checks of tune1-perturbations (2026-10-26).
//
// Grid's own WilsonFlow (Luscher's third-order integrator on the Wilson-action flow) and its
// energy densities; nothing here reimplements them. For each configuration: the thin plaquette
// at t = 0 (it must equal the HMC log's value for that trajectory, which identifies the file),
// then t^2 E(t) in the clover and plaquette definitions every WF_MEAS steps up to WF_TMAX, and
// the clover topological charge at WF_TMAX. The scales t0 (t^2 E = 0.3) and w0
// (t d/dt t^2 E = 0.3) are read off the printed curve offline, so the program stays a pure
// measurement and the analysis can change without a rebuild.
//
// One code for both codes' configurations: Grid checkpoints (ckpoint_lat.<n>, ILDG) and Chroma
// files (SZINQIO/ILDG LIME) go through the same reader, so a Grid-vs-Chroma difference in t0
// can only come from the configurations.
//
// Env:
//   WF_CONFIGS  file listing configuration paths, one per line ('#' lines skipped)
//   WF_EPS      flow step (default 0.01)
//   WF_TMAX     flow time to reach (default 8.0)
//   WF_MEAS     measure every WF_MEAS steps (default 1)
//   WF_STOUT_RHO  stout rho for the smeared plaquette, one step, all directions (default 0.125,
//               the HMC driver's fermion smearing; added 2026-10-28)
// Output (boss rank, after Grid's log prefix):
//   WFLOW <cfg> plaq0 <thin plaquette>
//   WFLOW <cfg> plaq0_smeared <stout-smeared plaquette, = Grid's second "Plaquette:" log line>
//   WFLOW <cfg> flow <step> <t> <t^2 E clover> <t^2 E plaquette>
//   WFLOW <cfg> topq <Q clover> t <tmax>
//   WFLOW <cfg> seconds <read + flow wall time>
//
// Usage: inside an allocation, e.g. through perlmutter/run_test_grid_mg.sh with BIN set to this
// binary, LATT=48.48.48.96, MPI_GEOM=1.1.2.2 (one node, 4 ranks x 1 GPU).
#include <Grid/Grid.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace Grid;

namespace {

double env_double(const char *name, double def)
{
  const char *v = std::getenv(name);
  return (v && *v) ? std::atof(v) : def;
}

int env_int(const char *name, int def)
{
  const char *v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : def;
}

// The driver's IMPORT_CFG reader (gen_qcd_hasenbusch_tune_compact_schur.cc, "Gauge field"),
// as in logdet_parity/test_logdet_parity.cc: NERSC header -> NerscIO, else IldgReader.
void read_config(const std::string &cfg, LatticeGaugeField &Umu)
{
  FILE *fp = std::fopen(cfg.c_str(), "rb");
  GRID_ASSERT(fp != nullptr);
  char magic[16] = {0};
  const size_t nread = std::fread(magic, 1, sizeof(magic), fp);
  std::fclose(fp);
  FieldMetaData header;
  if (nread >= 12 && std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
    NerscIO::readConfiguration<GaugeStatistics<PeriodicGimplR>>(Umu, header, cfg);
  } else {
    IldgReader IR;
    IR.open(cfg);
    IR.readConfiguration(Umu, header);
    IR.close();
  }
}

}  // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  const char *list = std::getenv("WF_CONFIGS");
  GRID_ASSERT(list != nullptr && *list != '\0');  // WF_CONFIGS=<file of configuration paths>
  std::vector<std::string> cfgs;
  {
    std::ifstream in(list);
    GRID_ASSERT(in.good());
    std::string line;
    while (std::getline(in, line))
      if (!line.empty() && line[0] != '#') cfgs.push_back(line);
  }
  const double eps = env_double("WF_EPS", 0.01);
  const double tmax = env_double("WF_TMAX", 8.0);
  const int meas = env_int("WF_MEAS", 1);
  const double stout_rho = env_double("WF_STOUT_RHO", 0.125);
  const int nstep = int(std::lround(tmax / eps));
  GRID_ASSERT(eps > 0.0 && nstep > 0 && meas > 0);

  GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  LatticeGaugeField U(grid), V(grid);

  std::cout << std::setprecision(12);
  std::cout << GridLogMessage << "WFLOW settings configs " << cfgs.size() << " eps " << eps
            << " tmax " << tmax << " nstep " << nstep << " meas " << meas << std::endl;

  for (const std::string &cfg : cfgs) {
    const double t_start = usecond();
    read_config(cfg, U);
    std::cout << GridLogMessage << "WFLOW " << cfg << " plaq0 "
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(U) << std::endl;
    // Stout-smeared plaquette, one step at WF_STOUT_RHO in all directions: the HMC driver's
    // fermion smearing (params.h STOUT_RHO 0.125, STOUT_NSMEAR 1) and Grid's second
    // "Plaquette:" line. V is free until the flow overwrites it.
    Smear_Stout<PeriodicGimplR> stout(stout_rho);
    stout.smear(V, U);
    std::cout << GridLogMessage << "WFLOW " << cfg << " plaq0_smeared "
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(V) << std::endl;

    WilsonFlow<PeriodicGimplR> WF(eps, nstep, meas);
    WF.resetActions();
    WF.addMeasurement(meas, [&cfg](int step, RealD t, const LatticeGaugeField &W) {
      const RealD ec = WilsonFlowBase<PeriodicGimplR>::energyDensityCloverleaf(t, W);
      const RealD ep = WilsonFlowBase<PeriodicGimplR>::energyDensityPlaquette(t, W);
      std::cout << GridLogMessage << "WFLOW " << cfg << " flow " << step << " " << t << " " << ec
                << " " << ep << std::endl;
    });
    WF.smear(V, U);

    std::cout << GridLogMessage << "WFLOW " << cfg << " topq "
              << WilsonLoops<PeriodicGimplR>::TopologicalCharge(V) << " t " << nstep * eps
              << std::endl;
    std::cout << GridLogMessage << "WFLOW " << cfg << " seconds "
              << (usecond() - t_start) / 1.0e6 << std::endl;
  }

  Grid_finalize();
  return 0;
}
