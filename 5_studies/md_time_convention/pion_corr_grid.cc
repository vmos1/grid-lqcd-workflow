// Point-source "pion" correlator C(t) = sum_x sum_{spin,colour src} |M^-1 delta_0|^2 (x,t) on a
// configuration, with the production clover operator: the Grid half of the hopping-term comparison
// with Chroma's PROPAGATOR prop_corr (__docs/2026_10_05_grid_chroma_md_time_handoff.md §8). C(t) is
// basis independent (summed over source and sink spin), so Grid's and Chroma's gamma conventions
// drop out; a mismatch in mass, csw, smearing, boundary or operator normalisation does not.
//
// Operator and reader exactly as the strange spectrum probe (eig_probe_strange_smeared.cc) and the
// driver: CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers>, csw_r = csw_t, cF = 1,
// default anisotropy, boundary phases (1,1,1,-1), imported on stout-smeared links
// (Smear_Stout(rho), SmearedConfiguration nsmear, get_U(true)); IMPORT_CFG reader (ILDG or NERSC).
// Solve: SchurRedBlackDiagMooeeSolve + double CG, then the unpreconditioned residual |M x - b|/|b|.
//
// Usage: pion_corr_grid --grid 48.48.48.96 --mpi 1.2.2.2 --config <path> [--mass -0.2050]
//          [--csw 1.20536588031793] [--stout-rho 0.125] [--stout-nsmear 1] [--cg-tol 1e-11]
//          [--cg-maxit 50000] [--links smeared|thin]
// Output (boss rank): GAUGE, SOLVE (per spin/colour: iterations, true residual), PROPCORR t C(t).

#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/fermion/CompactWilsonCloverFermion.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Grid;

namespace {

typedef CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<WilsonImplR>> WCF;

std::string arg_string(int argc, char **argv, const std::string &opt, const std::string &fallback)
{
  if (!GridCmdOptionExists(argv, argv + argc, opt)) return fallback;
  return GridCmdOptionPayload(argv, argv + argc, opt);
}
double arg_double(int argc, char **argv, const std::string &opt, double fallback)
{
  const std::string s = arg_string(argc, argv, opt, "");
  return s.empty() ? fallback : std::stod(s);
}
int arg_int(int argc, char **argv, const std::string &opt, int fallback)
{
  const std::string s = arg_string(argc, argv, opt, "");
  return s.empty() ? fallback : std::stoi(s);
}

// The driver's IMPORT_CFG reader.
bool read_config(const std::string &cfg, LatticeGaugeField &Umu)
{
  FILE *fp = std::fopen(cfg.c_str(), "rb");
  if (!fp) return false;
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
  return true;
}

}  // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  {
    const std::string cfg = arg_string(argc, argv, "--config", "");
    const double mass = arg_double(argc, argv, "--mass", -0.2050);
    const double csw = arg_double(argc, argv, "--csw", 1.20536588031793);
    const double stout_rho = arg_double(argc, argv, "--stout-rho", 0.125);
    const int stout_nsmear = arg_int(argc, argv, "--stout-nsmear", 1);
    const double cg_tol = arg_double(argc, argv, "--cg-tol", 1e-11);
    const int cg_maxit = arg_int(argc, argv, "--cg-maxit", 50000);
    const std::string links = arg_string(argc, argv, "--links", "smeared");

    const Coordinate latt = GridDefaultLatt();
    GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian RBGrid_(&Grid_);
    const bool boss = Grid_.IsBoss();

    if (boss)
      std::cout << "RUN config " << cfg << " mass " << std::setprecision(15) << mass << " csw " << csw
                << " stout rho " << stout_rho << " nsmear " << stout_nsmear << " cg_tol " << cg_tol
                << " links " << links << std::endl;

    LatticeGaugeField Umu(&Grid_);
    if (cfg.empty() || !read_config(cfg, Umu)) {
      if (boss) std::cout << "ERROR cannot read --config " << cfg << std::endl;
      Grid_finalize();
      return 1;
    }
    Smear_Stout<PeriodicGimplR> Stout(stout_rho);
    SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear, Stout);
    Smear.set_Field(Umu);
    LatticeGaugeField &Usm = Smear.get_U(true);
    // Collectives on every rank, print on the boss only.
    const RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
    const RealD plaq_sm = WilsonLoops<PeriodicGimplR>::avgPlaquette(Usm);
    if (boss)
      std::cout << "GAUGE plaquette=" << std::setprecision(16) << plaq << " smeared=" << plaq_sm
                << std::endl;

    WilsonImplParams impl_p;
    impl_p.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
    WilsonAnisotropyCoefficients anis;
    WCF Op(Umu, Grid_, RBGrid_, mass, csw, csw, /*cF=*/1.0, anis, impl_p);
    Op.ImportGauge(links == "thin" ? Umu : Usm);

    ConjugateGradient<LatticeFermion> CG(cg_tol, cg_maxit);
    SchurRedBlackDiagMooeeSolve<LatticeFermion> Solver(CG);

    const int Nt = latt[Nd - 1];
    std::vector<RealD> corr(Nt, 0.0);
    LatticeFermion src(&Grid_), sol(&Grid_), res(&Grid_);
    LatticeComplex dens(&Grid_);
    const Coordinate origin({0, 0, 0, 0});

    for (int s = 0; s < Ns; ++s) {
      for (int c = 0; c < Nc; ++c) {
        const double t0 = usecond();
        src = Zero();
        SpinColourVector sv = Zero();
        sv()(s)(c) = Complex(1.0, 0.0);
        pokeSite(sv, src, origin);
        sol = Zero();
        Solver(Op, src, sol);
        Op.M(sol, res);
        res = res - src;
        const RealD rel = std::sqrt(norm2(res) / norm2(src));
        dens = localNorm2(sol);
        std::vector<TComplex> ct;
        sliceSum(dens, ct, Nd - 1);
        for (int t = 0; t < Nt; ++t) corr[t] += real(TensorRemove(ct[t]));
        if (boss)
          std::cout << "SOLVE spin=" << s << " colour=" << c << " cg_iters=" << CG.IterationsToComplete
                    << std::scientific << std::setprecision(4) << " true_rel_res=" << rel
                    << std::fixed << std::setprecision(1) << " t_s=" << (usecond() - t0) * 1e-6
                    << std::endl;
      }
    }
    if (boss)
      for (int t = 0; t < Nt; ++t)
        std::cout << "PROPCORR " << t << " " << std::scientific << std::setprecision(15) << corr[t]
                  << std::endl;
  }
  Grid_finalize();
  return 0;
}
