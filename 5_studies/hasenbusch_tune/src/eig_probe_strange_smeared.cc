// eig_probe_strange_smeared.cc
//
// Spectrum of the strange Schur normal operator Q = Mpc^dag Mpc EXACTLY as the strange RHMC
// (OneFlavourSchurCloverRationalActionMP, pseudofermion on the ODD checkerboard) uses it, on
// BOTH the stout-smeared links the action sees (SmearedConfiguration, get_U(true)) and the thin
// links; plus the production rational approximations and their relative errors.
//
// Derived from eig_probe_strange.cc (thin links, WilsonCloverFermion, EVEN checkerboard, one
// config).  Differences, all to match the driver (gen_qcd_hasenbusch_tune_compact_schur.cc):
//   operator   CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers>, csw_r = csw_t = csw,
//              cF = 1, default anisotropy, boundary phases (1,1,1,-1)  [driver line ~439]
//   Schur      SchurDifferentiableOperator<WilsonImplR> on the ODD checkerboard
//   links      the operator is imported with Usm (stout rho, nsmear) or Umu (thin), plain
//              ImportGauge as GuardedImportGauge does with the switches off
//   reader     the driver's IMPORT_CFG branch (as test_logdet_parity.cc); several --config
// lambda_max: own power iteration (history, residual stop); lambda_min: inverse iteration with
// a tight inner CG (default 1e-10), history of Rayleigh quotient and eigen-residual
// |Q v - lambda v| / lambda per step.  Rayleigh quotients are variational: lambda_max estimate
// is a lower bound, lambda_min estimate an upper bound, on the true extremes.
//
// Rational part: AlgRemez(lo, hi, 64), degree d: generateApprox(d,1,4) -> r_+1/4 (getPFE) and
// r_-1/4 (getIPFE); generateApprox(d,1,2) -> r_-1/2 (getIPFE): the carried action's order
// (OneFlavourSchurCloverRationalAction constructor).  Relative errors |r_p(x)/x^p - 1| on a log
// grid and at each measured eigenvalue, and |r_-1/4(x)^2 / r_-1/2(x) - 1| (force/action
// consistency: S uses r_-1/4, the force uses r_-1/2).
//
// Usage:
//   eig_probe_strange_smeared --grid 48.48.48.96 --mpi 1.1.2.2 --device-mem 20000
//      --config <path> [--config <path> ...] [--mass -0.2050] [--csw 1.20536588031793]
//      [--stout-rho 0.125] [--stout-nsmear 1] [--rat-lo 0.4] [--rat-hi 35] [--rat-degree 20]
//      [--cg-tol 1e-10] [--cg-maxit 20000] [--pm-iters 1000] [--pm-res 1e-6]
//      [--inv-iters 60] [--inv-res 1e-6] [--nstarts 2] [--links both|smeared|thin]
// Output lines for the boss rank: GAUGE, PM / INV (histories), SPEC (one per config and links),
// RATGRID, RATAT, RATSUM.

#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/fermion/CompactWilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/EvenOddSchurDifferentiable.h>

// Not used here; included so build_test_fused_clover.sh's dependency check passes unchanged.
#include "clover_force/fused_clover_force.h"

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

std::vector<std::string> config_paths(int argc, char **argv)
{
  std::vector<std::string> out;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--config") out.push_back(argv[i + 1]);
  return out;
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

struct Extremes {
  RealD lmax = 0, lmax_res = 0, lmin = 0, lmin_res = 0;
  int pm_iters = 0, inv_iters = 0;
  RealD lmax_all_spread = 0, lmin_all_spread = 0;  // max-min over starts, relative
};

// Rational approximations as the carried action builds them.
struct Rat {
  MultiShiftFunction PowerQuarter, PowerNegQuarter, PowerNegHalf;
  Rat(RealD lo, RealD hi, int degree, int precision, RealD tol)
  {
    AlgRemez remez(lo, hi, precision);
    std::cout << GridLogMessage << "Generating degree " << degree << " for x^(1/4)" << std::endl;
    remez.generateApprox(degree, 1, 4);
    PowerQuarter.Init(remez, tol, false);
    PowerNegQuarter.Init(remez, tol, true);
    std::cout << GridLogMessage << "Generating degree " << degree << " for x^(1/2)" << std::endl;
    remez.generateApprox(degree, 1, 2);
    PowerNegHalf.Init(remez, tol, true);
  }
  // relative errors at x
  void errs(double x, double &e_p14, double &e_m14, double &e_m12, double &e_cons)
  {
    const double rp = PowerQuarter.approx(x), rm = PowerNegQuarter.approx(x),
                 rh = PowerNegHalf.approx(x);
    e_p14 = std::fabs(rp / std::pow(x, 0.25) - 1.0);
    e_m14 = std::fabs(rm / std::pow(x, -0.25) - 1.0);
    e_m12 = std::fabs(rh / std::pow(x, -0.5) - 1.0);
    e_cons = std::fabs(rm * rm / rh - 1.0);
  }
  void print(const std::string &tag, const std::string &label, double x)
  {
    double a, b, c, d;
    errs(x, a, b, c, d);
    std::cout << tag << " " << label << " x=" << std::setprecision(8) << std::scientific << x
              << " err_p14=" << a << " err_m14=" << b << " err_m12=" << c << " cons=" << d
              << std::endl;
  }
};

}  // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  {
    const std::vector<std::string> cfgs = config_paths(argc, argv);
    const double mass = arg_double(argc, argv, "--mass", -0.2050);
    const double csw = arg_double(argc, argv, "--csw", 1.20536588031793);
    const double stout_rho = arg_double(argc, argv, "--stout-rho", 0.125);
    const int stout_nsmear = arg_int(argc, argv, "--stout-nsmear", 1);
    const double rat_lo = arg_double(argc, argv, "--rat-lo", 0.4);
    const double rat_hi = arg_double(argc, argv, "--rat-hi", 35.0);
    const int rat_degree = arg_int(argc, argv, "--rat-degree", 20);
    const double cg_tol = arg_double(argc, argv, "--cg-tol", 1e-10);
    const int cg_maxit = arg_int(argc, argv, "--cg-maxit", 20000);
    const int pm_iters = arg_int(argc, argv, "--pm-iters", 1000);
    const double pm_res = arg_double(argc, argv, "--pm-res", 1e-6);
    const int inv_iters = arg_int(argc, argv, "--inv-iters", 60);
    const double inv_res = arg_double(argc, argv, "--inv-res", 1e-6);
    const int nstarts = arg_int(argc, argv, "--nstarts", 2);
    const std::string links = arg_string(argc, argv, "--links", "both");

    const Coordinate latt = GridDefaultLatt();
    const Coordinate mpi = GridDefaultMpi();
    GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), mpi);
    GridRedBlackCartesian RBGrid_(&Grid_);
    const bool boss = Grid_.IsBoss();
    GridParallelRNG pRNG(&Grid_);
    pRNG.SeedFixedIntegers({1, 2, 3, 4});

    if (boss) {
      std::cout << "RUN mass " << std::setprecision(15) << mass << " csw " << csw << " stout rho "
                << stout_rho << " nsmear " << stout_nsmear << " rat " << rat_lo << " " << rat_hi
                << " deg " << rat_degree << " cg_tol " << cg_tol << " pm_iters " << pm_iters
                << " inv_iters " << inv_iters << " nstarts " << nstarts << " links " << links
                << " nconfigs " << cfgs.size() << std::endl;
      for (auto &c : cfgs) std::cout << "RUN config " << c << std::endl;
    }
    if (cfgs.empty()) {
      if (boss) std::cout << "no --config given" << std::endl;
      Grid_finalize();
      return 1;
    }

    // ---- production rational approximations (same call order and precision as the action) ----
    Rat rat(rat_lo, rat_hi, rat_degree, 64, 1e-8);
    if (boss) {
      for (int k = 0; k <= 50; ++k) {
        const double x = std::pow(10.0, -3.0 + 5.0 * k / 50.0);
        rat.print("RATGRID", "", x);
      }
      // worst case inside the design interval
      double w1 = 0, w2 = 0, w3 = 0, w4 = 0;
      for (int k = 0; k <= 4000; ++k) {
        const double x = rat_lo * std::pow(rat_hi / rat_lo, k / 4000.0);
        double a, b, c, d;
        rat.errs(x, a, b, c, d);
        w1 = std::max(w1, a); w2 = std::max(w2, b); w3 = std::max(w3, c); w4 = std::max(w4, d);
      }
      std::cout << "RATSUM in [" << rat_lo << "," << rat_hi << "] max err_p14=" << std::scientific
                << std::setprecision(4) << w1 << " err_m14=" << w2 << " err_m12=" << w3
                << " cons=" << w4 << std::endl;
    }

    LatticeGaugeField Umu(&Grid_);
    Smear_Stout<PeriodicGimplR> Stout(stout_rho);
    SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear, Stout);

    std::unique_ptr<WCF> StrangeOp;

    for (size_t ic = 0; ic < cfgs.size(); ++ic) {
      const std::string &cfg = cfgs[ic];
      if (!read_config(cfg, Umu)) {
        if (boss) std::cout << "cannot open " << cfg << std::endl;
        continue;
      }
      const RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
      Smear.set_Field(Umu);
      LatticeGaugeField &Usm = Smear.get_U(true);
      const RealD plaq_sm = WilsonLoops<PeriodicGimplR>::avgPlaquette(Usm);
      if (boss)
        std::cout << "GAUGE cfg=" << cfg << " plaquette=" << std::setprecision(16) << plaq
                  << " smeared=" << plaq_sm << std::endl;

      if (!StrangeOp) {
        WilsonImplParams impl_p;
        impl_p.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
        WilsonAnisotropyCoefficients anis;
        StrangeOp = std::make_unique<WCF>(Umu, Grid_, RBGrid_, mass, csw, csw, /*cF=*/1.0, anis,
                                          impl_p);
      }

      for (int pass = 0; pass < 2; ++pass) {
        const bool smeared = (pass == 0);
        if (smeared && links == "thin") continue;
        if (!smeared && links == "smeared") continue;
        const std::string lname = smeared ? "smeared" : "thin";
        const double t0 = usecond();
        StrangeOp->ImportGauge(smeared ? Usm : Umu);
        SchurDifferentiableOperator<WilsonImplR> Mpc(*StrangeOp);  // HermOp = Mpc^dag Mpc

        LatticeFermion x(&RBGrid_), v(&RBGrid_), Av(&RBGrid_), y(&RBGrid_), r(&RBGrid_);
        LatticeFermion src_full(&Grid_);
        std::vector<RealD> lmaxs, lmins, lmax_ress, lmin_ress;

        for (int st = 0; st < nstarts; ++st) {
          // ---- lambda_max: power iteration, ODD checkerboard ----
          random(pRNG, src_full);
          pickCheckerboard(Odd, v, src_full);
          RealD lam = 0, res = 1;
          int it = 0;
          for (it = 0; it < pm_iters; ++it) {
            RealD n = std::sqrt(norm2(v));
            v = v * (1.0 / n);
            Mpc.HermOp(v, Av);
            lam = real(innerProduct(v, Av));
            r = Av - lam * v;
            res = std::sqrt(norm2(r)) / lam;
            if (boss && (it % 20 == 0 || res < pm_res))
              std::cout << "PM cfg=" << ic << " links=" << lname << " start=" << st
                        << " iter=" << it << std::setprecision(10) << std::scientific
                        << " lambda=" << lam << " res=" << res << std::endl;
            if (res < pm_res) break;
            v = Av;
          }
          if (boss)
            std::cout << "PMEND cfg=" << ic << " links=" << lname << " start=" << st
                      << " iters=" << it << std::setprecision(12) << std::scientific
                      << " lambda_max=" << lam << " res=" << res << std::endl;
          lmaxs.push_back(lam);
          lmax_ress.push_back(res);

          // ---- lambda_min: inverse iteration, tight CG ----
          random(pRNG, src_full);
          pickCheckerboard(Odd, x, src_full);
          { RealD n = std::sqrt(norm2(x)); x = x * (1.0 / n); }
          ConjugateGradient<LatticeFermion> CG(cg_tol, cg_maxit);
          RealD lmn = 0, rmn = 1;
          int k = 0;
          for (k = 0; k < inv_iters; ++k) {
            y = Zero();
            y.Checkerboard() = Odd;
            CG(Mpc, x, y);  // (Mpc^dag Mpc) y = x
            { RealD n = std::sqrt(norm2(y)); y = y * (1.0 / n); }
            x = y;
            Mpc.HermOp(x, Av);
            lmn = real(innerProduct(x, Av)) / norm2(x);
            r = Av - lmn * x;
            rmn = std::sqrt(norm2(r)) / lmn;
            if (boss)
              std::cout << "INV cfg=" << ic << " links=" << lname << " start=" << st
                        << " iter=" << k << std::setprecision(10) << std::scientific
                        << " lambda=" << lmn << " res=" << rmn << std::endl;
            if (rmn < inv_res) break;
          }
          lmins.push_back(lmn);
          lmin_ress.push_back(rmn);
          if (boss)
            std::cout << "INVEND cfg=" << ic << " links=" << lname << " start=" << st
                      << " iters=" << (k + 1) << std::setprecision(12) << std::scientific
                      << " lambda_min=" << lmn << " res=" << rmn << std::endl;
        }

        // best: largest lmax (lower bound), smallest lmin (upper bound)
        size_t imax = 0, imin = 0;
        for (size_t s = 1; s < lmaxs.size(); ++s) {
          if (lmaxs[s] > lmaxs[imax]) imax = s;
          if (lmins[s] < lmins[imin]) imin = s;
        }
        RealD sp_max = 0, sp_min = 0;
        for (size_t s = 0; s < lmaxs.size(); ++s) {
          sp_max = std::max(sp_max, std::fabs(lmaxs[s] - lmaxs[imax]) / lmaxs[imax]);
          sp_min = std::max(sp_min, std::fabs(lmins[s] - lmins[imin]) / lmins[imin]);
        }
        const RealD lmax = lmaxs[imax], lmin = lmins[imin];
        if (boss) {
          std::cout << "SPEC cfg=" << cfg << " links=" << lname << std::setprecision(8)
                    << std::scientific << " plaq=" << (smeared ? plaq_sm : plaq)
                    << " lambda_min=" << lmin << " min_res=" << lmin_ress[imin]
                    << " min_start_spread=" << sp_min << " lambda_max=" << lmax
                    << " max_res=" << lmax_ress[imax] << " max_start_spread=" << sp_max
                    << " cond=" << lmax / lmin
                    << " inside[" << rat_lo << "," << rat_hi << "]="
                    << ((lmin >= rat_lo && lmax <= rat_hi) ? "YES" : "NO")
                    << " t_s=" << (usecond() - t0) * 1e-6 << std::endl;
          rat.print("RATAT", "cfg=" + cfg + " links=" + lname + " lambda_min", lmin);
          rat.print("RATAT", "cfg=" + cfg + " links=" + lname + " lambda_max", lmax);
        }
      }
    }
  }
  Grid_finalize();
  return 0;
}
