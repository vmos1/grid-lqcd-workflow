// test_fast_import.cc  (pure-Grid HMC, C2 acceptance test)
//
// Standalone check of src/gauge_import/fast_import.h against the stock
// CompactWilsonCloverFermion::ImportGauge, before any HMC run. On the 16^3 cl3 config, stout-smeared
// exactly as the driver does, with the driver's compact clover operators (antiperiodic T, cF = 1),
// fp64 at --mass and --mass2 and fp32 at --mass (U precision-changed per direction as the driver's
// strange MP action does):
//   1. EXACTNESS. Reference = the operator's fields after a stock op.ImportGauge(U). For each case
//      the operator's 12 clover fields (Diagonal, Triangle, their Inv, Even, Odd variants) and the
//      doubled links (Umu, UmuEven, UmuOdd) are zeroed, FastImportGaugeWith(op, U, share, gpuinv)
//      runs, and every field is compared with the reference: max |diff| over all real words and
//      rel = |diff| / |ref|. Cases: (share 0, gpu 0) = the phase-by-phase reproduction; share 1 on
//      a cold cache (miss), again (hit), and a second fp64 operator at --mass2 fed from the first
//      one's cache (hit across masses: the C2b situation); then gpu 1 with share 0 and 1. Expect 0
//      on every field without gpu, and 0 on the 6 non-inverse fields with gpu.
//   2. INVERSION QUALITY (gpu cases). Per site and 6x6 block: max |entry diff| and relative
//      Frobenius norm |inv_gpu - inv_eigen|_F / |inv_eigen|_F against the stock Eigen inverse, the
//      residual |A inv - 1|_F of both, and the block's 2-norm condition number (Eigen
//      SelfAdjointEigenSolver on the rebuilt Hermitian block, which includes diag_mass).
//   3. TIMING. reps calls each of stock ImportGauge (once with GridLogDebug on, which prints the
//      stock phase breakdown) and of the fast path per combination with phase timers
//      (HASEN_GRID_IMPORT_TIMERS-style "ImportTimer" lines); "miss" alternates two gauge fields so
//      every call recomputes the field strength.
// PASS: exact cases max |diff| == 0 on all 15 fields; gpu cases 0 on the 9 non-inverse fields and
// inverse rel <= 1e-13 (fp64) / 1e-6 (fp32, both round the fp64 inverse to fp32).
//
// Usage (inside an allocation, 4 GPUs, one rank per GPU; run_test_grid_mg.sh with BIN= works):
//   test_fast_import --grid 16.16.16.48 --mpi 1.1.2.2 [--cfg <lime>] [--mass -0.245]
//                    [--mass2 -0.2] [--csw 1.24930970916466] [--stout-rho 0.125]
//                    [--stout-nsmear 1] [--reps 5]

#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>

#include "clover_force/fused_clover_force.h"  // not used; build_test_fused_clover.sh checks it compiled
#include "gauge_import/fast_import.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Grid;

namespace {

typedef CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<WilsonImplR>> WCF;
typedef CompactWilsonCloverFermion<WilsonImplF, CompactCloverHelpers<WilsonImplF>> WCF_f;

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

// max |word| over all real words of a field, global over ranks.
template <class F>
double max_abs_words(const F &f)
{
  typedef typename F::scalar_type S;
  typedef typename S::value_type R;
  autoView(v, f, CpuRead);
  const R *p = (const R *)&v[0];
  const size_t n = (size_t)f.Grid()->oSites() * sizeof(typename F::vector_object) / sizeof(R);
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(p[i]));
  f.Grid()->GlobalMax(m);
  return m;
}

struct Cmp {
  double maxabs = 0.0;
  double rel = 0.0;
};
template <class F>
Cmp compare(const F &a, const F &ref)
{
  F d(ref.Grid());
  d.Checkerboard() = ref.Checkerboard();
  d = a - ref;
  Cmp c;
  c.maxabs = max_abs_words(d);
  const double nr = norm2(ref);
  c.rel = nr > 0 ? std::sqrt(norm2(d) / nr) : std::sqrt(norm2(d));
  return c;
}

// The 15 fields an import writes, as a copy.
template <class Op>
struct Snapshot {
  decltype(Op::Umu) Umu, UmuEven, UmuOdd;
  decltype(Op::Diagonal) D, De, Do, Di, Die, Dio;
  decltype(Op::Triangle) T, Te, To, Ti, Tie, Tio;
  explicit Snapshot(Op &op)
      : Umu(op.Umu.Grid()), UmuEven(op.UmuEven.Grid()), UmuOdd(op.UmuOdd.Grid()),
        D(op.Diagonal.Grid()), De(op.DiagonalEven.Grid()), Do(op.DiagonalOdd.Grid()),
        Di(op.DiagonalInv.Grid()), Die(op.DiagonalInvEven.Grid()), Dio(op.DiagonalInvOdd.Grid()),
        T(op.Triangle.Grid()), Te(op.TriangleEven.Grid()), To(op.TriangleOdd.Grid()),
        Ti(op.TriangleInv.Grid()), Tie(op.TriangleInvEven.Grid()), Tio(op.TriangleInvOdd.Grid()) {
    Umu = op.Umu; UmuEven = op.UmuEven; UmuOdd = op.UmuOdd;
    D = op.Diagonal; De = op.DiagonalEven; Do = op.DiagonalOdd;
    Di = op.DiagonalInv; Die = op.DiagonalInvEven; Dio = op.DiagonalInvOdd;
    T = op.Triangle; Te = op.TriangleEven; To = op.TriangleOdd;
    Ti = op.TriangleInv; Tie = op.TriangleInvEven; Tio = op.TriangleInvOdd;
  }
};

template <class Op>
void scramble(Op &op)
{
  op.Umu = Zero(); op.UmuEven = Zero(); op.UmuOdd = Zero();
  op.Diagonal = Zero(); op.DiagonalEven = Zero(); op.DiagonalOdd = Zero();
  op.DiagonalInv = Zero(); op.DiagonalInvEven = Zero(); op.DiagonalInvOdd = Zero();
  op.Triangle = Zero(); op.TriangleEven = Zero(); op.TriangleOdd = Zero();
  op.TriangleInv = Zero(); op.TriangleInvEven = Zero(); op.TriangleInvOdd = Zero();
}

// Compare all 15 fields; returns max |diff| over the non-inverse and over the inverse fields.
template <class Op>
void compare_all(const char *label, Op &op, const Snapshot<Op> &r, bool boss, double &max_noninv,
                 double &max_inv, double &rel_inv)
{
  struct Row { const char *name; Cmp c; bool inv; };
  std::vector<Row> rows = {
      {"Umu", compare(op.Umu, r.Umu), false},
      {"UmuEven", compare(op.UmuEven, r.UmuEven), false},
      {"UmuOdd", compare(op.UmuOdd, r.UmuOdd), false},
      {"Diagonal", compare(op.Diagonal, r.D), false},
      {"DiagonalEven", compare(op.DiagonalEven, r.De), false},
      {"DiagonalOdd", compare(op.DiagonalOdd, r.Do), false},
      {"Triangle", compare(op.Triangle, r.T), false},
      {"TriangleEven", compare(op.TriangleEven, r.Te), false},
      {"TriangleOdd", compare(op.TriangleOdd, r.To), false},
      {"DiagonalInv", compare(op.DiagonalInv, r.Di), true},
      {"DiagonalInvEven", compare(op.DiagonalInvEven, r.Die), true},
      {"DiagonalInvOdd", compare(op.DiagonalInvOdd, r.Dio), true},
      {"TriangleInv", compare(op.TriangleInv, r.Ti), true},
      {"TriangleInvEven", compare(op.TriangleInvEven, r.Tie), true},
      {"TriangleInvOdd", compare(op.TriangleInvOdd, r.Tio), true},
  };
  max_noninv = max_inv = rel_inv = 0.0;
  for (auto &w : rows) {
    if (w.inv) { max_inv = std::max(max_inv, w.c.maxabs); rel_inv = std::max(rel_inv, w.c.rel); }
    else max_noninv = std::max(max_noninv, w.c.maxabs);
  }
  if (boss) {
    std::cout << std::scientific << std::setprecision(3) << "CMP " << label;
    for (auto &w : rows) std::cout << " " << w.name << "=" << w.c.maxabs << "/" << w.c.rel;
    std::cout << std::endl;
    std::cout << "CASE " << label << " max_abs_noninv=" << max_noninv << " max_abs_inv=" << max_inv
              << " rel_inv=" << rel_inv << std::endl;
  }
}

// Per-site, per-block quality of the GPU inverse against the stock Eigen inverse.
template <class Impl, class Op>
void inverse_quality(const char *label, Op &op, const Snapshot<Op> &r, bool boss)
{
  typedef CompactWilsonCloverHelpers<Impl> CH;
  typedef Eigen::Matrix<std::complex<double>, 6, 6> M6;
  GridBase *grid = op.Diagonal.Grid();
  autoView(d_v, op.Diagonal, CpuRead);
  autoView(t_v, op.Triangle, CpuRead);
  autoView(dg_v, op.DiagonalInv, CpuRead);
  autoView(tg_v, op.TriangleInv, CpuRead);
  autoView(dr_v, r.Di, CpuRead);
  autoView(tr_v, r.Ti, CpuRead);
  const int Nsimd = Impl::Simd::Nsimd();
  double max_entry = 0, max_relF = 0, max_res_gpu = 0, max_res_eig = 0, max_cond = 0, neg_min_cond = -1e300;
  double neg_min_lmin = -1e300, max_lmax = 0;
  auto build = [&](const auto &dd, const auto &tt, int b) {
    M6 m;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) {
        if (i == j) m(i, j) = std::complex<double>(dd()(b)(i).real(), dd()(b)(i).imag());
        else {
          auto z = tt()(b)(CH::triangle_index(i, j));
          m(i, j) = (i < j) ? std::complex<double>(z.real(), z.imag()) : std::complex<double>(z.real(), -z.imag());
        }
      }
    return m;
  };
  for (int64_t ss = 0; ss < (int64_t)grid->oSites(); ++ss) {
    for (int l = 0; l < Nsimd; ++l) {
      auto ds = extractLane(l, d_v[ss]);
      auto ts = extractLane(l, t_v[ss]);
      auto dgs = extractLane(l, dg_v[ss]);
      auto tgs = extractLane(l, tg_v[ss]);
      auto drs = extractLane(l, dr_v[ss]);
      auto trs = extractLane(l, tr_v[ss]);
      for (int b = 0; b < 2; ++b) {
        const M6 A = build(ds, ts, b), G = build(dgs, tgs, b), E = build(drs, trs, b);
        max_entry = std::max(max_entry, (G - E).cwiseAbs().maxCoeff());
        max_relF = std::max(max_relF, (G - E).norm() / E.norm());
        max_res_gpu = std::max(max_res_gpu, (A * G - M6::Identity()).norm());
        max_res_eig = std::max(max_res_eig, (A * E - M6::Identity()).norm());
        Eigen::SelfAdjointEigenSolver<M6> es(A, Eigen::EigenvaluesOnly);
        const double lmin = es.eigenvalues()(0), lmax = es.eigenvalues()(5);
        const double cond = std::fabs(lmax / lmin);
        max_cond = std::max(max_cond, cond);
        neg_min_cond = std::max(neg_min_cond, -cond);
        neg_min_lmin = std::max(neg_min_lmin, -lmin);
        max_lmax = std::max(max_lmax, lmax);
      }
    }
  }
  grid->GlobalMax(max_entry);
  grid->GlobalMax(max_relF);
  grid->GlobalMax(max_res_gpu);
  grid->GlobalMax(max_res_eig);
  grid->GlobalMax(max_cond);
  grid->GlobalMax(neg_min_cond);
  grid->GlobalMax(neg_min_lmin);
  grid->GlobalMax(max_lmax);
  if (boss)
    std::cout << std::scientific << std::setprecision(3) << "INVQ " << label
              << " max_entry_diff=" << max_entry << " max_block_relFrob=" << max_relF
              << " max_resid_gpu=" << max_res_gpu << " max_resid_eigen=" << max_res_eig
              << " cond_min=" << -neg_min_cond << " cond_max=" << max_cond
              << " lambda_min=" << -neg_min_lmin << " lambda_max=" << max_lmax << std::endl;
}

template <class Op, class GF>
double time_stock(Op &op, const GF &U, int reps)
{
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    accelerator_barrier();
    const double t0 = usecond();
    op.ImportGauge(U);
    accelerator_barrier();
    best = std::min(best, (usecond() - t0) / 1e6);
  }
  return best;
}
template <class Op, class GF>
double time_fast(Op &op, const GF &U, const GF *U2, bool share, bool gpu, int reps)
{
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    const GF &Uc = (U2 && (r & 1)) ? *U2 : U;
    accelerator_barrier();
    const double t0 = usecond();
    FastImportGaugeWith(op, Uc, share, gpu, true);
    accelerator_barrier();
    best = std::min(best, (usecond() - t0) / 1e6);
  }
  return best;
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
    const double mass2 = arg_double(argc, argv, "--mass2", -0.2);
    const double csw = arg_double(argc, argv, "--csw", 1.24930970916466);
    const double stout_rho = arg_double(argc, argv, "--stout-rho", 0.125);
    const int stout_nsmear = arg_int(argc, argv, "--stout-nsmear", 1);
    const int reps = arg_int(argc, argv, "--reps", 5);

    // ---- grids: as the driver builds them ----
    const Coordinate latt = GridDefaultLatt();
    const Coordinate mpi = GridDefaultMpi();
    GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), mpi);
    GridRedBlackCartesian RBGrid_(&Grid_);
    GridCartesian GridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
    GridRedBlackCartesian RBGridF(&GridF);
    const bool boss = Grid_.IsBoss();
    if (boss)
      std::cout << "RUN cfg " << cfg << " mass " << mass << " mass2 " << mass2 << " csw " << csw
                << " stout rho " << stout_rho << " nsmear " << stout_nsmear << " reps " << reps << std::endl;

    // ---- gauge field: read, as the driver's IMPORT_CFG branch ----
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

    // ---- stout smearing: as the driver ----
    Smear_Stout<PeriodicGimplR> Stout(stout_rho);
    SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear, Stout);
    Smear.set_Field(Umu);
    LatticeGaugeField &Usm = (stout_nsmear > 0) ? Smear.get_SmearedU() : Umu;
    const double plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);    // collective: all ranks
    const double plaq_sm = WilsonLoops<PeriodicGimplR>::avgPlaquette(Usm);
    if (boss) std::cout << "GAUGE plaquette " << plaq << " smeared " << plaq_sm << std::endl;

    // fp32 copies, per direction as OneFlavourSchurCloverRationalActionMP::deriv does.
    auto to_single = [&](const LatticeGaugeField &Ud) {
      LatticeGaugeFieldF Uf(&GridF);
      LatticeColourMatrix ud(&Grid_);
      LatticeColourMatrixF uf(&GridF);
      for (int mu = 0; mu < Nd; ++mu) {
        ud = PeekIndex<LorentzIndex>(Ud, mu);
        precisionChange(uf, ud);
        PokeIndex<LorentzIndex>(Uf, uf, mu);
      }
      return Uf;
    };
    LatticeGaugeFieldF UsmF = to_single(Usm);
    LatticeGaugeFieldF UmuF = to_single(Umu);

    // ---- operators: as the driver's LightOps / StrangeOpF ----
    WilsonImplParams impl_p;
    impl_p.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
    WilsonAnisotropyCoefficients anis;
    WCF op(Usm, Grid_, RBGrid_, mass, csw, csw, /*cF=*/1.0, anis, impl_p);
    WCF op2(Usm, Grid_, RBGrid_, mass2, csw, csw, /*cF=*/1.0, anis, impl_p);
    WCF_f opF(UsmF, GridF, RBGridF, mass, csw, csw, /*cF=*/1.0, anis, impl_p);

    // ---- references: stock import ----
    op.ImportGauge(Usm);
    op2.ImportGauge(Usm);
    opF.ImportGauge(UsmF);
    Snapshot<WCF> ref(op), ref2(op2);
    Snapshot<WCF_f> refF(opF);

    const double tol_inv_d = 1e-13, tol_inv_f = 1e-6;
    double mn, mi, ri;
    auto exact_case = [&](bool ok, const char *label) {
      if (!ok) pass = false;
      if (boss) std::cout << "RESULT " << label << (ok ? " ok" : " BAD") << std::endl;
    };

    // ---- 1. exactness, fp64 ----
    scramble(op); FastImportGaugeWith(op, Usm, false, false, false);
    compare_all("D_share0_gpu0", op, ref, boss, mn, mi, ri);
    exact_case(mn == 0.0 && mi == 0.0, "D_share0_gpu0 exact");

    scramble(op); FastImportGaugeWith(op, Usm, true, false, false);   // cold cache: miss
    compare_all("D_share1_miss", op, ref, boss, mn, mi, ri);
    exact_case(mn == 0.0 && mi == 0.0 && FieldStrengthCacheFor<WilsonImplD>().misses == 1, "D_share1_miss exact");

    scramble(op); FastImportGaugeWith(op, Usm, true, false, false);   // hit
    compare_all("D_share1_hit", op, ref, boss, mn, mi, ri);
    exact_case(mn == 0.0 && mi == 0.0 && FieldStrengthCacheFor<WilsonImplD>().hits == 1, "D_share1_hit exact");

    scramble(op2); FastImportGaugeWith(op2, Usm, true, false, false); // hit across masses
    compare_all("D_mass2_share1_hit", op2, ref2, boss, mn, mi, ri);
    exact_case(mn == 0.0 && mi == 0.0 && FieldStrengthCacheFor<WilsonImplD>().hits == 2, "D_mass2_share1_hit exact");

    scramble(op); FastImportGaugeWith(op, Usm, false, true, false);
    compare_all("D_share0_gpu1", op, ref, boss, mn, mi, ri);
    inverse_quality<WilsonImplD>("D_share0_gpu1", op, ref, boss);
    exact_case(mn == 0.0 && ri <= tol_inv_d, "D_share0_gpu1 noninv exact, inverse roundoff");

    scramble(op); FastImportGaugeWith(op, Usm, true, true, false);
    compare_all("D_share1_gpu1", op, ref, boss, mn, mi, ri);
    exact_case(mn == 0.0 && ri <= tol_inv_d, "D_share1_gpu1 noninv exact, inverse roundoff");

    scramble(op2); FastImportGaugeWith(op2, Usm, true, true, false);
    compare_all("D_mass2_share1_gpu1", op2, ref2, boss, mn, mi, ri);
    inverse_quality<WilsonImplD>("D_mass2_share1_gpu1", op2, ref2, boss);
    exact_case(mn == 0.0 && ri <= tol_inv_d, "D_mass2_share1_gpu1 noninv exact, inverse roundoff");

    // ---- 1. exactness, fp32 ----
    scramble(opF); FastImportGaugeWith(opF, UsmF, false, false, false);
    compare_all("F_share0_gpu0", opF, refF, boss, mn, mi, ri);
    exact_case(mn == 0.0 && mi == 0.0, "F_share0_gpu0 exact");

    scramble(opF); FastImportGaugeWith(opF, UsmF, true, false, false);
    compare_all("F_share1_miss", opF, refF, boss, mn, mi, ri);
    exact_case(mn == 0.0 && mi == 0.0 && FieldStrengthCacheFor<WilsonImplF>().misses == 1, "F_share1_miss exact");

    scramble(opF); FastImportGaugeWith(opF, UsmF, true, false, false);
    compare_all("F_share1_hit", opF, refF, boss, mn, mi, ri);
    exact_case(mn == 0.0 && mi == 0.0 && FieldStrengthCacheFor<WilsonImplF>().hits == 1, "F_share1_hit exact");

    scramble(opF); FastImportGaugeWith(opF, UsmF, true, true, false);
    compare_all("F_share1_gpu1", opF, refF, boss, mn, mi, ri);
    inverse_quality<WilsonImplF>("F_share1_gpu1", opF, refF, boss);
    exact_case(mn == 0.0 && ri <= tol_inv_f, "F_share1_gpu1 noninv exact, inverse roundoff");

    // ---- 2b. a second, rougher field: the thin links (no smearing), fp64, both on ----
    {
      WCF opT(Umu, Grid_, RBGrid_, mass, csw, csw, /*cF=*/1.0, anis, impl_p);
      opT.ImportGauge(Umu);
      Snapshot<WCF> refT(opT);
      scramble(opT); FastImportGaugeWith(opT, Umu, true, true, false);   // miss: new U
      compare_all("D_thin_share1_gpu1", opT, refT, boss, mn, mi, ri);
      inverse_quality<WilsonImplD>("D_thin_share1_gpu1", opT, refT, boss);
      exact_case(mn == 0.0 && ri <= tol_inv_d, "D_thin_share1_gpu1 noninv exact, inverse roundoff");
    }

    // ---- 3. timing (best of reps) ----
    if (boss) std::cout << "TIMING stock ImportGauge phase breakdown (GridLogDebug, one call):" << std::endl;
    GridLogDebug.Active(1);
    op.ImportGauge(Usm);
    opF.ImportGauge(UsmF);
    GridLogDebug.Active(0);
    const double ts_d = time_stock(op, Usm, reps);
    const double ts_f = time_stock(opF, UsmF, reps);
    if (boss) std::cout << "TIMING fast-path ImportTimer lines follow (fp64 then fp32)" << std::endl;
    const double tf_d00 = time_fast(op, Usm, (const LatticeGaugeField *)nullptr, false, false, reps);
    const double tf_d10m = time_fast(op, Usm, &Umu, true, false, reps);
    const double tf_d10h = time_fast(op, Usm, (const LatticeGaugeField *)nullptr, true, false, reps);
    const double tf_d01 = time_fast(op, Usm, (const LatticeGaugeField *)nullptr, false, true, reps);
    const double tf_d11h = time_fast(op, Usm, (const LatticeGaugeField *)nullptr, true, true, reps);
    const double tf_f00 = time_fast(opF, UsmF, (const LatticeGaugeFieldF *)nullptr, false, false, reps);
    const double tf_f11h = time_fast(opF, UsmF, (const LatticeGaugeFieldF *)nullptr, true, true, reps);
    if (boss) {
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "TIMING fp64 best_s stock=" << ts_d << " fast_share0_gpu0=" << tf_d00
                << " share1_miss=" << tf_d10m << " share1_hit=" << tf_d10h << " share0_gpu1=" << tf_d01
                << " share1_hit_gpu1=" << tf_d11h << std::endl;
      std::cout << "TIMING fp32 best_s stock=" << ts_f << " fast_share0_gpu0=" << tf_f00
                << " share1_hit_gpu1=" << tf_f11h << std::endl;
      std::cout << "TEST RESULT: " << (pass ? "PASS" : "FAIL") << std::endl;
    }
  }
  Grid_finalize();
  return pass ? 0 : 1;
}
