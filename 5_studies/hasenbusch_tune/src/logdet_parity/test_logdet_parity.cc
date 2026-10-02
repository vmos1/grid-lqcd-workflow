// test_logdet_parity.cc  (pure-Grid HMC, task P: parity pairing of the strange log-det)
//
// Measures, per gauge configuration, both clover diagonal-block log-dets of the strange operator
// exactly as the driver (gen_qcd_hasenbusch_tune_compact_schur.cc) builds and evaluates its
// StrangeLogDet monomial, with N_f = 1:
//   S_even = -ln|det Mee|   (QCDLogDetCompactCloverEOAction, parity Even, the driver's monomial)
//   S_odd  = -ln|det Moo|   (same class, parity Odd)
//   D      = S_even - S_odd
// on the stout-smeared links the monomial sees (is_smeared = true: SmearedConfiguration over
// Smear_Stout(rho), get_U(true)) and on the thin links.  Why: det M = det Mee det S_oo
// = det Moo det S_ee.  An Odd strange pseudofermion (S_oo) pairs with S_even, an Even one (S_ee)
// with S_odd.  An Even pseudofermion with the even log-det samples |det M| e^{-D} instead of
// |det M|; for a correctly weighted ensemble <D> = 0 by translation symmetry.
//
// Exactness test (first configuration, thin links): U'(x) = U(x + x^) (Cshift +1 in direction 0)
// swaps even and odd sites and changes nothing else (boundary phases are in t only, the clover
// term is built from the raw links), so S_odd(U) == S_even(U') and S_even(U) == S_odd(U') up to
// summation order.  With --force-test 1 (default) also the derivative:
// F_odd(U) == Cshift(F_even(U'), 0, -1), and the converse; printed as relative L2 norm and the
// largest per-site |diff|, with the even-vs-odd force difference on U as the negative control.
//
// Reader: the driver's IMPORT_CFG branch (NERSC header -> NerscIO, else IldgReader; patch 02
// makes the LFN record optional, so Chroma lime files and Grid ckpoint_lat.NNNN both read).
// Operator: the driver's StrangeOp (CompactWilsonCloverFermion<WilsonImplR,
// CompactCloverHelpers>, csw_r = csw_t = --csw, cF = 1, default anisotropy, boundary phases
// (1,1,1,-1)).  Defaults are the 48^3x96 production values (traj48_puregrid.sh).
//
// Usage (inside an allocation; run_test_grid_mg.sh with BIN= and LATT=/MPI_GEOM= works):
//   test_logdet_parity --grid 48.48.48.96 --mpi 1.1.2.2 --device-mem 20000
//       --config <path> [--config <path> ...] [--config-list <file, one path per line>]
//       [--mass -0.2050] [--csw 1.20536588031793] [--stout-rho 0.125] [--stout-nsmear 1]
//       [--shift-test 1] [--force-test 1]
// Output lines (boss rank): "PARITY cfg=..." per configuration, "SHIFT ..." and "FORCE ..." for
// the exactness test, "MEM ..." device/host high-water marks, "TEST RESULT: PASS|FAIL"
// (PASS needs every shift rel <= 1e-12, no NaN, and the negative control O(1)).

#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCompactCloverEOAction.h>

// Not used here; included so build_test_fused_clover.sh's dependency check (it verifies that
// this tree's copy of the fused header is the one compiled) passes unchanged.
#include "clover_force/fused_clover_force.h"

#include <sys/resource.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Grid;

namespace {

typedef CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<WilsonImplR>> WCF;
typedef QCDLogDetCompactCloverEOAction<WilsonImplR> LogDetAction;

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

// Every "--config <path>" in order, then every non-empty line of every "--config-list <file>".
std::vector<std::string> config_paths(int argc, char **argv)
{
  std::vector<std::string> out;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--config") out.push_back(argv[i + 1]);
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) != "--config-list") continue;
    std::ifstream f(argv[i + 1]);
    std::string line;
    while (std::getline(f, line))
      if (!line.empty() && line[0] != '#') out.push_back(line);
  }
  return out;
}

// The driver's IMPORT_CFG reader (gen_qcd_hasenbusch_tune_compact_schur.cc, "Gauge field").
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

// Device bytes in use (all of this process's allocations, Grid's software cache included) and
// the host peak RSS; high-water marks are kept across calls.
double dev_used_gb_max = 0.0;
void mem_mark(GridBase *grid, const std::string &where)
{
  double dev_used = 0.0;
#ifdef GRID_CUDA
  size_t fr = 0, tot = 0;
  cudaMemGetInfo(&fr, &tot);
  dev_used = double(tot - fr) / 1e9;
#endif
  RealD d = dev_used;
  grid->GlobalMax(d);
  if (d > dev_used_gb_max) dev_used_gb_max = d;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  RealD rss = double(ru.ru_maxrss) / 1e6;  // kB -> GB
  grid->GlobalMax(rss);
  if (grid->IsBoss())
    std::cout << "MEM " << where << " device_used_GB(max over ranks)=" << std::setprecision(4) << d
              << " device_high_water_GB=" << dev_used_gb_max << " host_peak_rss_GB(max rank)=" << rss
              << std::endl;
}

// Largest per-site |a - b| (sqrt of the site norm2), over all ranks.
RealD max_site_diff(const LatticeGaugeField &a, const LatticeGaugeField &b)
{
  LatticeGaugeField d(a.Grid());
  d = a - b;
  auto ln = localNorm2(d);
  typedef decltype(ln)::vector_object::scalar_object sobj;
  std::vector<sobj> v;
  unvectorizeToLexOrdArray(v, ln);
  RealD m = 0.0;
  for (auto &s : v) {
    const RealD r = real(TensorRemove(s));
    if (r > m) m = r;
  }
  a.Grid()->GlobalMax(m);
  return std::sqrt(m);
}

RealD rel_norm(const LatticeGaugeField &a, const LatticeGaugeField &ref)
{
  LatticeGaugeField d(ref.Grid());
  d = a - ref;
  return std::sqrt(norm2(d) / norm2(ref));
}

}  // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  bool pass = true;
  {
    const std::vector<std::string> cfgs = config_paths(argc, argv);
    const double mass = arg_double(argc, argv, "--mass", -0.2050);
    const double csw = arg_double(argc, argv, "--csw", 1.20536588031793);
    const double stout_rho = arg_double(argc, argv, "--stout-rho", 0.125);
    const int stout_nsmear = arg_int(argc, argv, "--stout-nsmear", 1);
    const int shift_test = arg_int(argc, argv, "--shift-test", 1);
    const int force_test = arg_int(argc, argv, "--force-test", 1);
    const double threshold = 1.0e-12;

    // ---- grids: as the driver builds them ----
    const Coordinate latt = GridDefaultLatt();
    const Coordinate mpi = GridDefaultMpi();
    GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), mpi);
    GridRedBlackCartesian RBGrid_(&Grid_);
    const bool boss = Grid_.IsBoss();
    if (boss) {
      std::cout << "RUN mass " << std::setprecision(15) << mass << " csw " << csw << " stout rho "
                << stout_rho << " nsmear " << stout_nsmear << " nconfigs " << cfgs.size()
                << " shift_test " << shift_test << " force_test " << force_test << std::endl;
      for (auto &c : cfgs) std::cout << "RUN config " << c << std::endl;
    }
    if (cfgs.empty()) {
      if (boss) std::cout << "no --config given" << std::endl << "TEST RESULT: FAIL" << std::endl;
      Grid_finalize();
      return 1;
    }
    mem_mark(&Grid_, "start");

    LatticeGaugeField Umu(&Grid_);
    // Smearing: as the driver ("Stout smearing" block); the monomials see get_U(true).
    Smear_Stout<PeriodicGimplR> Stout(stout_rho);
    SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear, Stout);

    // The operator is built once, on the first configuration's thin links (the constructor
    // imports them); every S()/deriv() below re-imports its argument (GuardedImportGauge), as
    // the driver's monomial does.
    std::unique_ptr<WCF> StrangeOp;
    std::unique_ptr<LogDetAction> LdEven, LdOdd;

    for (size_t ic = 0; ic < cfgs.size(); ++ic) {
      const std::string &cfg = cfgs[ic];
      const double t0 = usecond();
      if (!read_config(cfg, Umu)) {
        if (boss) std::cout << "cannot open " << cfg << std::endl;
        pass = false;
        continue;
      }
      const double t_read = usecond();
      const RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
      Smear.set_Field(Umu);
      LatticeGaugeField &Usm = Smear.get_U(true);
      const RealD plaq_sm = WilsonLoops<PeriodicGimplR>::avgPlaquette(Usm);
      if (boss)
        std::cout << "GAUGE cfg=" << cfg << " plaquette=" << std::setprecision(16) << plaq
                  << " smeared=" << plaq_sm << std::endl;
      mem_mark(&Grid_, "after read+smear");

      if (!StrangeOp) {
        // ---- operator: as the driver's StrangeOp (antiperiodic T, cF = 1) ----
        WilsonImplParams impl_p;
        impl_p.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
        WilsonAnisotropyCoefficients anis;
        StrangeOp = std::make_unique<WCF>(Umu, Grid_, RBGrid_, mass, csw, csw, /*cF=*/1.0, anis,
                                          impl_p);
        LdEven = std::make_unique<LogDetAction>(*StrangeOp, 1);       // the driver's monomial
        LdOdd = std::make_unique<LogDetAction>(*StrangeOp, 1, Odd);
        if (boss)
          std::cout << "OP " << LdEven->action_name() << " / " << LdOdd->action_name()
                    << std::endl << LdEven->LogParameters();
        mem_mark(&Grid_, "after operator");
      }

      const double t1 = usecond();
      const RealD se_sm = LdEven->S(Usm);
      const RealD so_sm = LdOdd->S(Usm);
      const double t2 = usecond();
      const RealD se_th = LdEven->S(Umu);
      const RealD so_th = LdOdd->S(Umu);
      const double t3 = usecond();
      mem_mark(&Grid_, "after S");
      if (boss)
        std::cout << "PARITY cfg=" << cfg << std::setprecision(15) << " plaq=" << plaq
                  << " plaq_sm=" << plaq_sm << " S_even_sm=" << se_sm << " S_odd_sm=" << so_sm
                  << " D_sm=" << (se_sm - so_sm) << " S_even_thin=" << se_th
                  << " S_odd_thin=" << so_th << " D_thin=" << (se_th - so_th)
                  << std::setprecision(4) << " t_read_s=" << (t_read - t0) * 1e-6
                  << " t_smeared_pair_s=" << (t2 - t1) * 1e-6
                  << " t_thin_pair_s=" << (t3 - t2) * 1e-6 << std::endl;
      if (!std::isfinite(se_sm) || !std::isfinite(so_sm) || !std::isfinite(se_th) ||
          !std::isfinite(so_th))
        pass = false;

      if (ic == 0 && shift_test) {
        // ---- exactness: one-site shift in x swaps the checkerboards ----
        LatticeGaugeField Ush(&Grid_);
        Ush = Cshift(Umu, 0, 1);
        const RealD plaq_sh = WilsonLoops<PeriodicGimplR>::avgPlaquette(Ush);
        const double ts0 = usecond();
        const RealD se_sh = LdEven->S(Ush);
        const RealD so_sh = LdOdd->S(Ush);
        const double ts1 = usecond();
        const RealD r1 = std::fabs(se_sh - so_th) / std::fabs(so_th);
        const RealD r2 = std::fabs(so_sh - se_th) / std::fabs(se_th);
        if (boss)
          std::cout << "SHIFT thin plaq(U')=" << std::setprecision(16) << plaq_sh
                    << std::setprecision(15) << " S_even(U')=" << se_sh << " S_odd(U)=" << so_th
                    << " rel=" << std::setprecision(3) << r1 << std::setprecision(15)
                    << " | S_odd(U')=" << so_sh << " S_even(U)=" << se_th
                    << " rel=" << std::setprecision(3) << r2 << " t_pair_s=" << (ts1 - ts0) * 1e-6
                    << std::endl;
        if (!(r1 <= threshold) || !(r2 <= threshold)) pass = false;

        if (force_test) {
          // ---- the derivative, same way (thin links; the action is a function of its input) ----
          LatticeGaugeField Fo(&Grid_), Fe(&Grid_), Fe_sh(&Grid_), Fo_sh(&Grid_), back(&Grid_);
          const double tf0 = usecond();
          LdOdd->deriv(Umu, Fo);
          const double tf1 = usecond();
          LdEven->deriv(Umu, Fe);
          const double tf2 = usecond();
          LdEven->deriv(Ush, Fe_sh);
          LdOdd->deriv(Ush, Fo_sh);
          mem_mark(&Grid_, "after deriv");
          back = Cshift(Fe_sh, 0, -1);
          const RealD fr1 = rel_norm(back, Fo);
          const RealD fm1 = max_site_diff(back, Fo);
          back = Cshift(Fo_sh, 0, -1);
          const RealD fr2 = rel_norm(back, Fe);
          const RealD fm2 = max_site_diff(back, Fe);
          const RealD ctrl = rel_norm(Fe, Fo);
          const RealD nFo = std::sqrt(norm2(Fo)), nFe = std::sqrt(norm2(Fe));
          if (boss)
            std::cout << "FORCE |F_odd(U)|=" << std::setprecision(15) << nFo
                      << " |F_even(U)|=" << nFe << std::setprecision(3)
                      << " | F_odd(U) vs shiftback F_even(U'): rel=" << fr1 << " max_site=" << fm1
                      << " | F_even(U) vs shiftback F_odd(U'): rel=" << fr2 << " max_site=" << fm2
                      << " | control F_even(U) vs F_odd(U): rel=" << ctrl
                      << " | t_deriv_odd_s=" << (tf1 - tf0) * 1e-6
                      << " t_deriv_even_s=" << (tf2 - tf1) * 1e-6 << std::endl;
          if (!(fr1 <= threshold) || !(fr2 <= threshold) || !(ctrl > 0.1) || !(nFo > 0.0))
            pass = false;
        }
      }
      if (boss)
        std::cout << "TIME cfg=" << cfg << " total_s=" << std::setprecision(4)
                  << (usecond() - t0) * 1e-6 << std::endl;
    }
    mem_mark(&Grid_, "end");
  }
  std::cout << GridLogMessage << "TEST RESULT: " << (pass ? "PASS" : "FAIL") << std::endl;
  Grid_finalize();
  return pass ? 0 : 1;
}
