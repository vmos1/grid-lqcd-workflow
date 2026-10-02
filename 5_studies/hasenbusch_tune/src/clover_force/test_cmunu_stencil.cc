// test_cmunu_stencil.cc  (pure-Grid HMC, item D acceptance test)
//
// Standalone check of src/clover_force/clover_cmunu.h against the stock clover insertion
// WilsonCloverHelpers<WilsonImplR>::Cmunu (the 31-Cshift chain) it replaces, before any HMC run.
// Random SU(3) links (scaled by -0.5, the operator's doubled-link normalisation; irrelevant to
// the algebra) and six random gaussian colour-matrix lambdas, one per unordered pair as the
// callers build them. For all 12 ordered (mu,nu):
//   1. STAGE 1 full (HASEN_GRID_CLOVER_STAPLE_CACHE=1): cache.Cmunu vs the chain, first on a
//      cache miss (Prepare builds), then on a hit. max |diff| and |diff|/|ref| per pair.
//   2. STAGE 1 lean (HASEN_GRID_CLOVER_STAPLE_CACHE_LEAN=1): the same.
//   3. The callers' whole pass, force_mu = - sum_nu f_{mu nu} Cmunu(...), chain vs each variant.
//   4. STAGE 2 (HASEN_GRID_CLOVER_STENCIL=1): CloverCmunuStencil::ForceMu, the same pass as one
//      PaddedCell exchange per field + site-local kernels; max |diff| and rel per force_mu.
// Expect max |diff| = 0 for both stages (bit-identical; stage 2 through its Opaque barriers).
// PASS needs stage 1 exact, |ref| > 0, no NaN, and stage 2 within 1e-14 relative on every
// force_mu (the brief's bound if not exact; the line says "(exact)" when max |diff| = 0).
// Timings (s per 12-pair pass, mean of --reps): chain, full miss, full hit, lean miss, lean hit,
// stage 2 with a link miss and a link hit.
// Comms measurement for stage 2 (the brief asks for it whatever is decided): one Cshift of a
// colour-matrix field per direction and sign, vs one PaddedCell Exchange / ExchangePeriodic at
// depth 1 and 2, and the GeneralLocalStencil table build for the 27-point depth-1 footprint.
//
// Usage (inside an allocation, 4 GPUs, one rank per GPU; run_test_grid_mg.sh with BIN= works):
//   test_cmunu --grid 16.16.16.48 --mpi 1.1.2.2 [--reps 3]

#include <Grid/Grid.h>

#include "clover_force/fused_clover_force.h"  // build_test_fused_clover.sh checks this header
#include "clover_force/clover_cmunu.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Grid;

namespace {

typedef WilsonImplR Impl;
typedef typename Impl::GaugeLinkField GLF;

int arg_int(int argc, char **argv, const std::string &opt, int fallback)
{
  if (!GridCmdOptionExists(argv, argv + argc, opt)) return fallback;
  return std::stoi(GridCmdOptionPayload(argv, argv + argc, opt));
}

// max over all sites and colour entries of |a - b|, global.
double max_abs_diff(const GLF &a, const GLF &b)
{
  GLF d(a.Grid());
  d = a - b;
  std::vector<typename GLF::scalar_object> host(a.Grid()->lSites());
  unvectorizeToLexOrdArray(host, d);
  double m = 0.0;
  for (const auto &s : host)
    for (int i = 0; i < Nc; i++)
      for (int j = 0; j < Nc; j++) {
        const double re = s()()(i, j).real(), im = s()()(i, j).imag();
        m = std::max(m, std::sqrt(re * re + im * im));
      }
  a.Grid()->GlobalMax(m);
  return m;
}

double rel_diff(const GLF &a, const GLF &ref)
{
  GLF d(ref.Grid());
  d = a - ref;
  return std::sqrt(norm2(d) / norm2(ref));
}

const int sigma_idx[4][4] = {{-1, 0, 1, 2}, {0, -1, 3, 4}, {1, 3, -1, 5}, {2, 4, 5, -1}};

// The callers' loop (fused_clover_force.h FinishLinks / log-det deriv_gpu) for one Cmunu flavour.
template <class CmunuFn>
void force_pass(std::vector<GLF> &force_mu, std::vector<GLF> &U, std::vector<GLF> &lambda, CmunuFn cmunu)
{
  const RealD factor = 2.0 * 0.62465485233233;  // 2 csw_r, any real number serves
  for (int mu = 0; mu < Nd; mu++) {
    force_mu[mu] = Zero();
    for (int nu = 0; nu < Nd; nu++) {
      if (mu == nu) continue;
      const RealD signed_factor = (mu < nu) ? factor : -factor;
      force_mu[mu] -= signed_factor * cmunu(U, lambda[sigma_idx[mu][nu]], mu, nu);
    }
  }
}

}  // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  bool pass = true;
  {
    const int reps = arg_int(argc, argv, "--reps", 3);
    const Coordinate latt = GridDefaultLatt();
    const Coordinate mpi = GridDefaultMpi();
    GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), mpi);
    const bool boss = Grid_.IsBoss();
    if (boss) std::cout << "RUN test_cmunu latt " << latt << " mpi " << mpi << " reps " << reps << std::endl;

    GridParallelRNG pRNG(&Grid_);
    pRNG.SeedFixedIntegers(std::vector<int>({27, 18, 28, 18}));
    LatticeGaugeField Umu(&Grid_);
    SU<Nc>::HotConfiguration(pRNG, Umu);
    std::vector<GLF> U(Nd, &Grid_);
    for (int mu = 0; mu < Nd; mu++) U[mu] = -0.5 * PeekIndex<LorentzIndex>(Umu, mu);
    std::vector<GLF> lambda(6, &Grid_);
    for (auto &l : lambda) gaussian(pRNG, l);

    auto chain = [](std::vector<GLF> &Uv, GLF &l, int mu, int nu) {
      return WilsonCloverHelpers<Impl>::Cmunu(Uv, l, mu, nu);
    };

    std::cout << std::scientific << std::setprecision(6);

    // ---- 1./2. per pair, full and lean, miss then hit ----
    for (int lean = 0; lean < 2; lean++) {
      CloverStapleCache<Impl> cache(lean == 1);
      for (int round = 0; round < 2; round++) {
        const bool hit = cache.Prepare(U);
        double worst_abs = 0, worst_rel = 0;
        for (int mu = 0; mu < Nd; mu++)
          for (int nu = 0; nu < Nd; nu++) {
            if (mu == nu) continue;
            GLF &l = lambda[sigma_idx[mu][nu]];
            GLF ref = chain(U, l, mu, nu);
            GLF got = cache.Cmunu(U, l, mu, nu);
            const double mad = max_abs_diff(got, ref), rel = rel_diff(got, ref);
            const double nref = norm2(ref);
            if (!(nref > 0) || !std::isfinite(rel) || mad != 0.0) pass = false;
            worst_abs = std::max(worst_abs, mad);
            worst_rel = std::max(worst_rel, rel);
            if (boss)
              std::cout << "PAIR " << (lean ? "lean" : "full") << (hit ? " hit " : " miss") << " mu=" << mu
                        << " nu=" << nu << " max_abs_diff=" << mad << " rel=" << rel << " |ref|^2=" << nref
                        << std::endl;
          }
        if (boss)
          std::cout << "STAGE1 " << (lean ? "lean" : "full") << (hit ? " hit " : " miss") << " worst max_abs_diff="
                    << worst_abs << " worst rel=" << worst_rel << (worst_abs == 0.0 ? " (exact)" : " BAD") << std::endl;
      }
    }

    // ---- 3. the callers' pass, and timings ----
    std::vector<GLF> f_ref(Nd, &Grid_), f_got(Nd, &Grid_);
    force_pass(f_ref, U, lambda, chain);  // warm-up
    double t0 = usecond();
    for (int r = 0; r < reps; r++) force_pass(f_ref, U, lambda, chain);
    const double t_chain = (usecond() - t0) / 1e6 / reps;

    std::vector<GLF> Ualt(Nd, &Grid_);  // a second link set, to force misses
    for (int mu = 0; mu < Nd; mu++) Ualt[mu] = 0.5 * U[mu];
    for (int lean = 0; lean < 2; lean++) {
      CloverStapleCache<Impl> cache(lean == 1);
      auto cached = [&cache](std::vector<GLF> &Uv, GLF &l, int mu, int nu) { return cache.Cmunu(Uv, l, mu, nu); };
      cache.Prepare(Ualt);
      cache.Prepare(U);
      force_pass(f_got, U, lambda, cached);  // warm-up
      double t_miss = 0, t_hit = 0;
      for (int r = 0; r < reps; r++) {
        cache.Prepare(Ualt);  // untimed: leaves the cache on the other field
        t0 = usecond();
        cache.Prepare(U);     // miss: rebuild
        force_pass(f_got, U, lambda, cached);
        t_miss += usecond() - t0;
        t0 = usecond();
        cache.Prepare(U);     // hit
        force_pass(f_got, U, lambda, cached);
        t_hit += usecond() - t0;
      }
      double worst = 0;
      for (int mu = 0; mu < Nd; mu++) worst = std::max(worst, max_abs_diff(f_got[mu], f_ref[mu]));
      if (worst != 0.0) pass = false;
      if (boss)
        std::cout << "PASS-LEVEL " << (lean ? "lean" : "full") << " force_mu max_abs_diff=" << worst
                  << (worst == 0.0 ? " (exact)" : " BAD") << " t_chain_s=" << t_chain
                  << " t_miss_s=" << t_miss / 1e6 / reps << " t_hit_s=" << t_hit / 1e6 / reps
                  << " speedup_hit=" << t_chain / (t_hit / 1e6 / reps) << " fields=" << cache.NumFields()
                  << " hits=" << cache.Hits() << " misses=" << cache.Misses() << std::endl;
    }

    // ---- 4. stage 2 (HASEN_GRID_CLOVER_STENCIL=1): the callers' whole pass, vs the chain ----
    {
      CloverCmunuStencil<Impl> &S2 = CloverCmunuStencil<Impl>::Instance();
      const GLF *lam[6] = {&lambda[0], &lambda[1], &lambda[2], &lambda[3], &lambda[4], &lambda[5]};
      const RealD factor = 2.0 * 0.62465485233233;  // as force_pass
      RealD sf[4][4];
      for (int mu = 0; mu < Nd; mu++)
        for (int nu = 0; nu < Nd; nu++) sf[mu][nu] = (mu == nu) ? 0.0 : ((mu < nu) ? factor : -factor);
      S2.ForceMu(U, lam, sigma_idx, sf, f_got);  // first call: setup + link exchange
      double worst_abs = 0, worst_rel = 0;
      for (int mu = 0; mu < Nd; mu++) {
        const double mad = max_abs_diff(f_got[mu], f_ref[mu]), rel = rel_diff(f_got[mu], f_ref[mu]);
        const double nref = norm2(f_ref[mu]);  // collective: outside the boss-only print
        worst_abs = std::max(worst_abs, mad);
        worst_rel = std::max(worst_rel, rel);
        if (boss)
          std::cout << "STAGE2 force_mu mu=" << mu << " max_abs_diff=" << mad << " rel=" << rel
                    << " |ref|^2=" << nref << std::endl;
      }
      if (!std::isfinite(worst_rel) || worst_rel > 1.0e-14) pass = false;
      double t_miss = 0, t_hit = 0;
      for (int r = 0; r < reps; r++) {
        S2.ForceMu(Ualt, lam, sigma_idx, sf, f_got);  // untimed: leaves the links on the other field
        t0 = usecond();
        S2.ForceMu(U, lam, sigma_idx, sf, f_got);     // link miss
        t_miss += usecond() - t0;
        t0 = usecond();
        S2.ForceMu(U, lam, sigma_idx, sf, f_got);     // link hit
        t_hit += usecond() - t0;
      }
      double worst_again = 0;
      for (int mu = 0; mu < Nd; mu++) worst_again = std::max(worst_again, max_abs_diff(f_got[mu], f_ref[mu]));
      if (worst_again != worst_abs) pass = false;  // deterministic: hit == first call
      if (boss)
        std::cout << "STAGE2 worst max_abs_diff=" << worst_abs << " worst rel=" << worst_rel
                  << (worst_abs == 0.0 ? " (exact)" : "") << " t_chain_s=" << t_chain
                  << " t_miss_s=" << t_miss / 1e6 / reps << " t_hit_s=" << t_hit / 1e6 / reps
                  << " speedup_hit=" << t_chain / (t_hit / 1e6 / reps) << " link_hits=" << S2.Hits()
                  << " link_misses=" << S2.Misses() << std::endl;
    }

    // ---- comms measurement for stage 2 ----
    {
      GLF F(&Grid_), S(&Grid_);
      F = lambda[0];
      const int nrep = 10;
      for (int dim = 0; dim < Nd; dim++)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          S = Cshift(F, dim, sgn);  // warm-up
          t0 = usecond();
          for (int r = 0; r < nrep; r++) S = Cshift(F, dim, sgn);
          const double t = (usecond() - t0) / nrep;
          if (boss) std::cout << "COMMS Cshift dim=" << dim << " shift=" << sgn << " ms=" << t / 1e3 << std::endl;
        }
      for (int depth = 1; depth <= 2; depth++) {
        PaddedCell Ghost(depth, &Grid_);
        GLF P = Ghost.Exchange(F);  // warm-up
        t0 = usecond();
        for (int r = 0; r < nrep; r++) P = Ghost.Exchange(F);
        const double t_ex = (usecond() - t0) / nrep;
        P = Ghost.ExchangePeriodic(F);
        t0 = usecond();
        for (int r = 0; r < nrep; r++) P = Ghost.ExchangePeriodic(F);
        const double t_exp = (usecond() - t0) / nrep;
        GLF E = Ghost.Extract(P);
        const double ex_ok = max_abs_diff(E, F);
        if (ex_ok != 0.0) pass = false;
        if (boss)
          std::cout << "COMMS PaddedCell depth=" << depth << " padded_local=" << P.Grid()->LocalDimensions()
                    << " Exchange_ms=" << t_ex / 1e3 << " ExchangePeriodic_ms=" << t_exp / 1e3
                    << " extract_roundtrip_max_abs_diff=" << ex_ok << std::endl;
        if (depth == 1) {
          std::vector<Coordinate> shifts;
          for (int i = 0; i < 27; i++) shifts.push_back(Coordinate(Nd, 0));
          t0 = usecond();
          GeneralLocalStencil st(P.Grid(), shifts);
          const double t_st = (usecond() - t0) / 1e3;
          if (boss) std::cout << "COMMS GeneralLocalStencil 27 points depth 1 build_ms=" << t_st << std::endl;
        }
      }
    }

    if (boss) std::cout << "TEST RESULT: " << (pass ? "PASS" : "FAIL") << std::endl;
  }
  Grid_finalize();
  return pass ? 0 : 1;
}
