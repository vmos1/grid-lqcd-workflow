// test_fused_clover_force.cc  (pure-Grid HMC, M5 fix 2 acceptance test)
//
// Standalone check of src/clover_force/fused_clover_force.h against the patch-01 bodies it
// replaces, before any HMC run. On the 16^3 cl3 config, stout-smeared exactly as the driver
// (and test_grid_mg_odd.cc) does, with the compact clover operator at --mass:
//   1. FUSED CHECK: N random pairs (X_k, Y_k) on Odd with random real coefficients a_k and N on
//      Even with b_k. Reference = sum_k a_k MooDeriv(X_k, Y_k) + sum_k b_k MeeDeriv(X'_k, Y'_k)
//      (2N body calls); fused = AccumulateMoo/AccumulateMee with the same pairs and
//      coefficients + one Finish(op). rel = |fused - ref| / |ref| (expect ~1e-14, summation
//      order only). Both timings printed.
//   2. SINGLE PAIR: one pair, coefficient 1, against one MooDeriv and one MeeDeriv call. The
//      fused arithmetic equals the body's for one pair (header notes), so expect rel = 0.
// PASS needs every rel <= 1e-12, |ref| > 0 and no NaN. (A third check, a plain-gauge-field
// Finish(U, csw), was removed 2026-10-01 together with that overload: the operator stores
// -0.5 * U, so it was 16x off, rel = 15 in runs/2026_10_1_fusedclover16; see the header.)
//
// Usage (inside an allocation, 4 GPUs, one rank per GPU; run_test_grid_mg.sh with BIN= works):
//   test_fused_clover_force --grid 16.16.16.48 --mpi 1.1.2.2 [--cfg <lime>] [--mass -0.245]
//                           [--csw 1.24930970916466] [--npairs 5] [--stout-rho 0.125]
//                           [--stout-nsmear 1]

#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>

#include "clover_force/fused_clover_force.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
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

double rel_diff(const LatticeGaugeField &a, const LatticeGaugeField &ref)
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
    const std::string cfg = arg_string(
        argc, argv, "--cfg",
        "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/data/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime");
    const double mass = arg_double(argc, argv, "--mass", -0.245);
    const double csw = arg_double(argc, argv, "--csw", 1.24930970916466);
    const int npairs = arg_int(argc, argv, "--npairs", 5);
    const double stout_rho = arg_double(argc, argv, "--stout-rho", 0.125);
    const int stout_nsmear = arg_int(argc, argv, "--stout-nsmear", 1);
    const double threshold = 1.0e-12;

    // ---- grids: as the driver builds them ----
    const Coordinate latt = GridDefaultLatt();
    const Coordinate mpi = GridDefaultMpi();
    GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), mpi);
    GridRedBlackCartesian RBGrid_(&Grid_);
    const bool boss = Grid_.IsBoss();
    if (boss)
      std::cout << "RUN cfg " << cfg << " mass " << mass << " csw " << csw << " npairs " << npairs
                << " stout rho " << stout_rho << " nsmear " << stout_nsmear << std::endl;

    // ---- gauge field: read, as the driver's IMPORT_CFG branch (test_grid_mg_odd.cc) ----
    LatticeGaugeField Umu(&Grid_);
    {
      FILE *fp = std::fopen(cfg.c_str(), "rb");
      if (!fp) {
        if (boss) std::cout << "cannot open --cfg " << cfg << std::endl << "TEST RESULT: FAIL" << std::endl;
        Grid_finalize();
        return 1;
      }
      char magic[16] = {0};
      const size_t nread = std::fread(magic, 1, sizeof(magic), fp);
      std::fclose(fp);
      FieldMetaData header;
      if (nread >= 12 && std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
        NerscIO::readConfiguration<GaugeStatistics<PeriodicGimplR>>(Umu, header, cfg);
      } else {
#ifdef HAVE_LIME
        IldgReader IR;
        IR.open(cfg);
        IR.readConfiguration(Umu, header);
        IR.close();
#else
        if (boss) std::cout << "this Grid has no LIME; cannot read " << cfg << std::endl;
        GRID_ASSERT(0);
#endif
      }
    }
    const double plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);

    // ---- stout smearing: as the driver (SmearedConfiguration over Smear_Stout) ----
    Smear_Stout<PeriodicGimplR> Stout(stout_rho);
    SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear, Stout);
    Smear.set_Field(Umu);
    LatticeGaugeField &Usm = (stout_nsmear > 0) ? Smear.get_SmearedU() : Umu;
    const double plaq_sm = WilsonLoops<PeriodicGimplR>::avgPlaquette(Usm);
    if (boss) std::cout << "GAUGE plaquette " << plaq << " smeared " << plaq_sm << std::endl;

    // ---- operator: as the driver's LightOps (antiperiodic T, cF = 1) ----
    WilsonImplParams impl_p;
    impl_p.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
    WilsonAnisotropyCoefficients anis;
    WCF op(Usm, Grid_, RBGrid_, mass, csw, csw, /*cF=*/1.0, anis, impl_p);

    // ---- random pairs and coefficients (fixed seeds; coefficients identical on all ranks) ----
    GridParallelRNG pRNG(&Grid_);
    pRNG.SeedFixedIntegers(std::vector<int>({31, 41, 59, 26}));
    std::mt19937_64 gen(20261001);
    std::uniform_real_distribution<double> coeff_dist(-1.0, 1.0);
    LatticeFermion full(&Grid_);
    auto draw = [&](int cb, std::vector<LatticeFermion> &v) {
      for (auto &f : v) {
        gaussian(pRNG, full);
        pickCheckerboard(cb, f, full);
      }
    };
    std::vector<LatticeFermion> Xo(npairs, &RBGrid_), Yo(npairs, &RBGrid_);
    std::vector<LatticeFermion> Xe(npairs, &RBGrid_), Ye(npairs, &RBGrid_);
    draw(Odd, Xo);
    draw(Odd, Yo);
    draw(Even, Xe);
    draw(Even, Ye);
    std::vector<RealD> ao(npairs), ae(npairs);
    for (auto &a : ao) a = coeff_dist(gen);
    for (auto &a : ae) a = coeff_dist(gen);

    LatticeGaugeField tmp(&Grid_), ref(&Grid_), fused(&Grid_);

    // Warm-up (untimed): first-call costs land outside both timings.
    op.MooDeriv(tmp, Xo[0], Yo[0], DaggerNo);

    // ---- 1. reference: 2N individual body calls, summed with the coefficients ----
    double t0 = usecond();
    ref = Zero();
    for (int k = 0; k < npairs; k++) {
      op.MooDeriv(tmp, Xo[k], Yo[k], DaggerNo);
      ref = ref + ao[k] * tmp;
    }
    for (int k = 0; k < npairs; k++) {
      op.MeeDeriv(tmp, Xe[k], Ye[k], DaggerNo);
      ref = ref + ae[k] * tmp;
    }
    const double nref = norm2(ref);  // also a sync point
    const double t_ref = (usecond() - t0) / 1.0e6;

    // ---- 1. fused: one accumulated Lambda, one clover pass ----
    t0 = usecond();
    FusedCloverForce<WilsonImplR> F(&RBGrid_);
    for (int k = 0; k < npairs; k++) F.AccumulateMoo(ao[k], Xo[k], Yo[k]);
    for (int k = 0; k < npairs; k++) F.AccumulateMee(ae[k], Xe[k], Ye[k]);
    F.Finish(op, fused);
    const double nfused = norm2(fused);
    const double t_fused = (usecond() - t0) / 1.0e6;
    const double rel = rel_diff(fused, ref);
    const bool ok1 = (nref > 0.0) && std::isfinite(nfused) && (rel <= threshold);

    // ---- 2. single pair, coefficient 1 ----
    LatticeGaugeField one_ref(&Grid_), one_fused(&Grid_);
    F.Reset();
    F.AccumulateMoo(1.0, Xo[0], Yo[0]);
    F.Finish(op, one_fused);
    op.MooDeriv(one_ref, Xo[0], Yo[0], DaggerNo);
    const double rel_moo1 = rel_diff(one_fused, one_ref);
    F.Reset();
    F.AccumulateMee(1.0, Xe[0], Ye[0]);
    F.Finish(op, one_fused);
    op.MeeDeriv(one_ref, Xe[0], Ye[0], DaggerNo);
    const double rel_mee1 = rel_diff(one_fused, one_ref);
    const bool ok2 = std::isfinite(rel_moo1) && std::isfinite(rel_mee1) &&
                     (rel_moo1 <= threshold) && (rel_mee1 <= threshold);

    pass = ok1 && ok2;
    if (boss) {
      std::cout << std::scientific << std::setprecision(6);
      std::cout << "FUSED CHECK rel=" << rel << " |ref|^2=" << nref << " |fused|^2=" << nfused
                << " pairs=" << npairs << "+" << npairs << " t_individual_s=" << t_ref
                << " t_fused_s=" << t_fused << " speedup=" << (t_fused > 0 ? t_ref / t_fused : 0.0)
                << (ok1 ? " ok" : " BAD") << std::endl;
      std::cout << "SINGLE PAIR CHECK moo rel=" << rel_moo1 << " mee rel=" << rel_mee1
                << ((rel_moo1 == 0.0 && rel_mee1 == 0.0) ? " (exact)" : "") << (ok2 ? " ok" : " BAD")
                << std::endl;
      std::cout << "TEST RESULT: " << (pass ? "PASS" : "FAIL") << std::endl;
    }
  }
  Grid_finalize();
  return pass ? 0 : 1;
}
