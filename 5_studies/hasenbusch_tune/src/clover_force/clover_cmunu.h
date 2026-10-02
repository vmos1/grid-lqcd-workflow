#pragma once
// clover_cmunu.h  (pure-Grid HMC, item D: the clover derivative without the 372-Cshift chain)
//
// The clover insertion C_{mu nu}[lambda] of the clover force, WilsonCloverHelpers<Impl>::Cmunu
// (staged stock tree Grid/qcd/action/fermion/WilsonCloverHelpers.h:42-79; both standard clover
// helpers forward to it, CloverHelpers.h:97-99 and 224-226), is called only from project code:
// FusedCloverForce::FinishLinks (fused_clover_force.h) and the carried
// QCDLogDetCompactCloverEOAction::deriv_cpu / deriv_gpu, 12 ordered (mu,nu) pairs per pass.
// Spec: __docs/2026_09_30_pure_grid_force_cost_analysis.md sections 2, 6 and 7.6 (item D).
//
// The chain, per (mu,nu), with Cshift(f,mu,+1)(x) = f(x+mu) and the PeriodicGaugeImpl shifts
// (CovariantCshift.h:41-73: CovShiftForward(L,mu,f) = L*Cshift(f,mu,+1), CovShiftBackward(L,mu,f)
// = Cshift(adj(L)*f,mu,-1), CovShiftIdentityBackward(L,mu) = Cshift(adj(L),mu,-1),
// ShiftStaple(L,mu) = Cshift(L,mu,+1)) has 31 Cshifts, 372 per pass. 14 of the 31 do not
// involve lambda; they are products of links only:
//   a[nu]    = CovShiftIdentityBackward(U[nu], nu)                     used by C1+ C2+ C4+
//   b[mu,nu] = CovShiftBackward(U[mu], mu, a[nu])                      used by C1+ C4+
//   d[mu,nu] = ShiftStaple(CovShiftForward(U[nu], nu, b[mu,nu]), mu)   = the whole of C4+ but lambda
//   e[mu,nu] = CovShiftBackward(U[mu], mu, U[nu])                      used by C1- C2- C4-
//   g[mu,nu] = ShiftStaple(CovShiftBackward(U[nu], nu, e[mu,nu]), mu)  = C1- and C4- but lambda
// With them cached a pass needs 17 lambda-dependent Cshifts per pair, 204 per pass (1.82x fewer).
//
// Stage 1, HASEN_GRID_CLOVER_STAPLE_CACHE=1 (read once, on iff the value starts with '1';
// default off, and then nothing here is constructed and the callers run the stock Cmunu):
// one process-wide cache per Impl holding a, b, d, e, g for the LAST link set seen, keyed
// exactly as import_guard.h keys its skip: a shadow copy of the four link fields and
// norm2(U[mu] - shadow[mu]) == 0 for every mu (device-side; reuse only on exact equality,
// otherwise rebuild). The callers pass the operator's own links (PeekIndex of FermOp.Umu, the
// doubled -0.5*U with the boundary phases), which every fermion monomial and log-det of one
// integrator level step shares, so the cache is rebuilt once per distinct gauge field and
// reused by the other passes at that field (seven per light-level step at base+G).
//
// Numerics: BIT-IDENTICAL to the chain. Each cached field is computed by the same Impl call on
// the same arguments as the sub-expression it replaces, and Cached::Cmunu below is the stock
// body statement by statement with those sub-expressions substituted (same products, same
// association, same `out = C1+; += C2+; += C3+; += C4+; -= C1-; -= C2-; -= C3-; -= C4-`).
// Checked by test_cmunu_stencil.cc (max |diff| = 0 for all 12 pairs) and the 16^3 smokes.
//
// Memory (colour-matrix fields of the full grid, 96 MB each per GPU at 48^3x96 on 16 GPUs):
// full mode 4 a + 12 x (b, d, e, g) = 52 fields + 4 shadows = 56 fields (~5.3 GB per GPU);
// HASEN_GRID_CLOVER_STAPLE_CACHE_LEAN=1 (read only with the cache on) keeps a, d, g only and
// recomputes b and e in each call (2 more Cshifts per pair, 228 per pass): 28 + 4 = 32 fields
// (~3.1 GB per GPU). Both modes are bit-identical to the chain.
//
// Also here: CloverSetCheckerboard, the host setCheckerboard of the clover-force assembly
// replaced by Grid's device-side acceleratorSetCheckerboard (Lattice_transfer.h) under
// HASEN_GRID_DEVICE_CB=1, read exactly as patch 06 reads it in EvenOddSchurDifferentiable.h
// (AssembleForce), so that one switch means one thing. Copy only: bit-identical.

#include <Grid/Grid.h>

#include <cstdlib>
#include <vector>

namespace Grid {

inline bool CloverCmunuEnvFlag(const char *name)
{
  const char *e = std::getenv(name);
  return e && e[0] == '1';
}
inline bool CloverStapleCacheEnabled()
{
  static const bool on = CloverCmunuEnvFlag("HASEN_GRID_CLOVER_STAPLE_CACHE");
  return on;
}
inline bool CloverStapleCacheLean()
{
  static const bool on = CloverCmunuEnvFlag("HASEN_GRID_CLOVER_STAPLE_CACHE_LEAN");
  return on;
}

// setCheckerboard(full, half), or its device-side twin under HASEN_GRID_DEVICE_CB=1 (patch 06).
template <class vobj>
inline void CloverSetCheckerboard(Lattice<vobj> &full, const Lattice<vobj> &half)
{
  static const bool device_cb = [](){ const char *e = getenv("HASEN_GRID_DEVICE_CB"); return e && e[0]=='1'; }();
  if (device_cb) acceleratorSetCheckerboard(full, half);
  else setCheckerboard(full, half);
}

template <class Impl>
class CloverStapleCache {
 public:
  INHERIT_IMPL_TYPES(Impl);

  // One instance per Impl per process (function-local static in an inline member: one
  // instance across translation units). Lives until exit, like import_guard.h's registry.
  static CloverStapleCache &Instance()
  {
    static CloverStapleCache c(CloverStapleCacheLean());
    return c;
  }

  explicit CloverStapleCache(bool lean) : lean_(lean) {}

  // Make the cache hold the staples of links U: exact-equality key on a shadow copy, rebuild
  // on any difference (or a new grid). Returns true on a hit. Call once per pass, before the
  // (mu,nu) loop; every Cmunu of that pass must use the same U.
  bool Prepare(const std::vector<GaugeLinkField> &U)
  {
    GRID_ASSERT(U.size() == Nd);
    const double t0 = usecond();
    GridBase *grid = U[0].Grid();
    bool same = (grid_ == grid);
    if (same) {
      GaugeLinkField diff(grid);
      for (int mu = 0; mu < Nd && same; mu++) {
        diff = U[mu] - shadow_[mu];
        if (norm2(diff) != 0.0) same = false;
      }
    }
    if (same) {
      hits_++;
    } else {
      if (grid_ != grid) Allocate(grid);
      for (int mu = 0; mu < Nd; mu++) shadow_[mu] = U[mu];
      Build(U);
      misses_++;
    }
    prepare_us_ += usecond() - t0;
    return same;
  }

  // WilsonCloverHelpers<Impl>::Cmunu(U, lambda, mu, nu) with the cached link products: the
  // stock statements, the lambda-independent sub-expressions read from the cache.
  GaugeLinkField Cmunu(const std::vector<GaugeLinkField> &U, const GaugeLinkField &lambda, int mu, int nu) const
  {
    conformable(lambda.Grid(), U[0].Grid());
    GRID_ASSERT(grid_ == U[0].Grid());
    const int p = P(mu, nu);
    if (lean_) {
      // Lean mode: b and e recomputed here by the same calls Build uses.
      GaugeLinkField b(lambda.Grid()), e(lambda.Grid());
      b = Impl::CovShiftBackward(U[mu], mu, a_[nu]);
      e = Impl::CovShiftBackward(U[mu], mu, U[nu]);
      return Body(U, lambda, mu, nu, a_[nu], b, d_[p], e, g_[p]);
    }
    return Body(U, lambda, mu, nu, a_[nu], b_[p], d_[p], e_[p], g_[p]);
  }

 private:
  // WilsonCloverHelpers.h:44-78 with a, b, d, e, g substituted (header).
  static GaugeLinkField Body(const std::vector<GaugeLinkField> &U, const GaugeLinkField &lambda, int mu, int nu,
                             const GaugeLinkField &a, const GaugeLinkField &b, const GaugeLinkField &d,
                             const GaugeLinkField &e, const GaugeLinkField &g)
  {
    GaugeLinkField out(lambda.Grid()), tmp(lambda.Grid());

    // C1+
    tmp = lambda * U[nu];
    out = Impl::ShiftStaple(Impl::CovShiftForward(tmp, nu, b), mu);

    // C2+
    tmp = U[mu] * Impl::ShiftStaple(adj(lambda), mu);
    out += Impl::ShiftStaple(Impl::CovShiftForward(U[nu], nu, Impl::CovShiftBackward(tmp, mu, a)), mu);

    // C3+ (all five shifts involve lambda: unchanged)
    tmp = U[nu] * Impl::ShiftStaple(adj(lambda), nu);
    out += Impl::ShiftStaple(Impl::CovShiftForward(U[nu], nu, Impl::CovShiftBackward(U[mu], mu, Impl::CovShiftIdentityBackward(tmp, nu))), mu);

    // C4+
    out += d * lambda;

    // C1-
    out -= Impl::ShiftStaple(lambda, mu) * g;

    // C2-
    tmp = adj(lambda) * U[nu];
    out -= Impl::ShiftStaple(Impl::CovShiftBackward(tmp, nu, e), mu);

    // C3- (unchanged)
    tmp = lambda * U[nu];
    out -= Impl::ShiftStaple(Impl::CovShiftBackward(U[nu], nu, Impl::CovShiftBackward(U[mu], mu, tmp)), mu);

    // C4-
    out -= g * lambda;

    return out;
  }

 public:
  // Cumulative statistics, one line on the boss rank at pass 1, 2, 4, ... and every 256th.
  void EndPass(double loop_us)
  {
    loop_us_ += loop_us;
    const long n = hits_ + misses_;
    if (grid_ && grid_->IsBoss() && (((n & (n - 1)) == 0) || (n % 256 == 0))) {
      std::cout << GridLogMessage << "[CloverStapleCache] passes=" << n << " hits=" << hits_
                << " misses=" << misses_ << " mode=" << (lean_ ? "lean" : "full")
                << " fields=" << NumFields() << " prepare_s=" << prepare_us_ * 1e-6
                << " cmunu_loop_s=" << loop_us_ * 1e-6
                << " per_pass_ms=" << (prepare_us_ + loop_us_) * 1e-3 / n
                << " (HASEN_GRID_CLOVER_STAPLE_CACHE=1)" << std::endl;
    }
  }

  int NumFields() const { return lean_ ? (Nd + 2 * 12 + Nd) : (Nd + 4 * 12 + Nd); }  // incl. shadows
  long Hits() const { return hits_; }
  long Misses() const { return misses_; }

 private:
  bool lean_;
  GridBase *grid_ = nullptr;
  std::vector<GaugeLinkField> shadow_, a_, b_, d_, e_, g_;
  long hits_ = 0, misses_ = 0;
  double prepare_us_ = 0, loop_us_ = 0;

  // Ordered pair (mu,nu), nu != mu, in the callers' row-major order: 0..11.
  static int P(int mu, int nu) { return 3 * mu + (nu < mu ? nu : nu - 1); }

  void Allocate(GridBase *grid)
  {
    grid_ = grid;
    shadow_.assign(Nd, GaugeLinkField(grid));
    a_.assign(Nd, GaugeLinkField(grid));
    d_.assign(12, GaugeLinkField(grid));
    g_.assign(12, GaugeLinkField(grid));
    if (!lean_) {
      b_.assign(12, GaugeLinkField(grid));
      e_.assign(12, GaugeLinkField(grid));
    }
    if (grid->IsBoss()) {
      const double mb = double(grid->lSites()) * sizeof(typename GaugeLinkField::scalar_object) / 1048576.0;
      std::cout << GridLogMessage << "[CloverStapleCache] allocated " << NumFields()
                << " colour-matrix fields (" << (lean_ ? "lean: a,d,g" : "full: a,b,d,e,g") << " + 4 shadows) = "
                << NumFields() * mb << " MB per rank" << std::endl;
    }
  }

  // The 14 lambda-independent shifts of each pair, by the stock calls (header).
  void Build(const std::vector<GaugeLinkField> &U)
  {
    for (int nu = 0; nu < Nd; nu++) a_[nu] = Impl::CovShiftIdentityBackward(U[nu], nu);
    // Lean mode: b, e go through two transients (b_[0], e_[0] when not lean are real slots).
    std::vector<GaugeLinkField> be_tmp(lean_ ? 2 : 0, GaugeLinkField(U[0].Grid()));
    for (int mu = 0; mu < Nd; mu++) {
      for (int nu = 0; nu < Nd; nu++) {
        if (mu == nu) continue;
        const int p = P(mu, nu);
        GaugeLinkField &b = lean_ ? be_tmp[0] : b_[p];
        GaugeLinkField &e = lean_ ? be_tmp[1] : e_[p];
        b = Impl::CovShiftBackward(U[mu], mu, a_[nu]);
        d_[p] = Impl::ShiftStaple(Impl::CovShiftForward(U[nu], nu, b), mu);
        e = Impl::CovShiftBackward(U[mu], mu, U[nu]);
        g_[p] = Impl::ShiftStaple(Impl::CovShiftBackward(U[nu], nu, e), mu);
      }
    }
  }
};

// ---------------------------------------------------------------------------------------------
// Stage 2, HASEN_GRID_CLOVER_STENCIL=1 (read once; default off): the callers' whole clover loop
//   force_mu[mu] = 0;  for nu != mu (ascending):  force_mu[mu] -= sf[mu][nu] * Cmunu(U, lambda_k, mu, nu)
// as ONE halo exchange per input field and site-local kernels, with Grid's own machinery for
// this (what the gauge action uses for its rectangle staples, WilsonLoops.h:1123-1330):
// PaddedCell + GeneralLocalStencil + accelerator_for.
//   * Footprint: writing every Cshift out (Cshift(f,mu,+1)(x) = f(x+mu)), the eight terms of
//     pair (mu,nu) read U_nu at x, x+mu, x-nu, x+mu-nu, U_mu at x+nu, x-nu, and lambda at x,
//     x+mu, x+nu, x+mu+nu, x-nu, x+mu-nu: all within one step per direction, so depth 1
//     suffices; the union over the 12 pairs is 27 points (0, +-mu, +mu+nu for mu<nu, +mu-nu).
//   * Exchanges: PaddedCell(1).ExchangePeriodic (face exchange, no Cshift; plain periodic, as
//     Cmunu's Cshift is, the boundary phases being inside the operator's links) of the four
//     links, cached per gauge field with the exact shadow-copy key of stage 1, and of the six
//     lambdas every pass: 6 exchanges per pass on a hit, 10 on a miss (vs 372 Cshifts).
//   * Kernel: one accelerator_for per mu over the padded sites; the eight terms with the chain's
//     products in the chain's association, e.g. C1+ = (lambda(x+mu) U_nu(x+mu)) (U_mu(x+nu)^dag
//     U_nu(x)^dag), C2+ = U_nu(x+mu) ((U_mu(x+nu) lambda(x+mu+nu)^dag)^dag U_nu(x)^dag), and the
//     chain's accumulation order and caller update (f = f - sf * out). Then Extract.
// Numerics: BIT-IDENTICAL to the chain. The same IEEE operations on the same values, in
// registers instead of through memory, with an optimisation barrier (Opaque) at each of the
// chain's kernel boundaries. Without the barriers the fused kernel differed from the chain by
// 1.2e-16 relative on every force_mu (nvcc folded adj() negations into the products and paired
// the multiply-adds differently); with them max |diff| = 0 (test_cmunu_stencil.cc, runs
// d_cmunu16s3 vs d_cmunu16s4) at unchanged speed.
// Memory per GPU at 48^3x96 on 16 GPUs (padded local 48.26.26.26, 121 MB per padded colour
// matrix): persistent 4 padded links + 4 shadows + the 27-point table (16 B per point per padded
// outer site, 91 MB) = 0.96 GB; transient per pass 6 padded lambdas + 1 padded output + the
// exchange temporaries, ~1.1 GB.
// Callers: FusedCloverForce::FinishLinks and the log-det deriv_gpu (deriv_cpu, the non-default
// path, has 12 lambdas built inside its loop and keeps the chain / stage 1). With both gates on,
// the stencil path is used and the staple cache is not touched.
inline bool CloverStencilEnabled()
{
  static const bool on = CloverCmunuEnvFlag("HASEN_GRID_CLOVER_STENCIL");
  return on;
}

template <class Impl>
class CloverCmunuStencil {
 public:
  INHERIT_IMPL_TYPES(Impl);
  typedef typename GaugeLinkField::vector_object vobj;
  typedef LatticeView<vobj> View;

  // Per-mu kernel arguments, captured by value.
  struct MuArgs {
    int mu;
    int nu[3];
    int lk[3];       // field index of lambda_k in the view table (4 + k)
    RealD sf[3];     // the caller's signed factor
    int pt[3][6];    // stencil points: 0, +mu, +nu, -nu, +mu+nu, +mu-nu
  };

  static CloverCmunuStencil &Instance()
  {
    static CloverCmunuStencil s;  // PaddedCell and stencil intentionally never freed (exit order)
    return s;
  }

  // Optimisation barrier on every real number of a colour matrix: afterwards the compiler must
  // treat the value as unknown, exactly as a value the chain writes to a field and reads back in
  // its next kernel. Without it the fused kernel let nvcc fold adj() negations and pair the
  // multiply-adds (FMA contraction, nvcc's default) differently from the chain's one-product
  // kernels: measured 1.2e-16 relative on every force_mu at 16^3 (run d_cmunu16s3). Device
  // code only (empty PTX asm with a read-write .f64 operand); a no-op on the host pass.
  template <class M>
  static accelerator_inline M Opaque(const M &in)
  {
    M m = in;
#ifdef __CUDA_ARCH__
    for (int i = 0; i < Nc; i++)
      for (int j = 0; j < Nc; j++) {
        double re = m()()(i, j).real();
        double im = m()()(i, j).imag();
        asm volatile("" : "+d"(re));
        asm volatile("" : "+d"(im));
        m()()(i, j) = typename std::decay<decltype(m()()(i, j))>::type(re, im);
      }
#endif
    return m;
  }

  // force_mu[mu] = - sum_{nu != mu, ascending} sf[mu][nu] * Cmunu(U, *lam[sig[mu][nu]], mu, nu).
  void ForceMu(const std::vector<GaugeLinkField> &U, const GaugeLinkField *const lam[6], const int sig[4][4],
               const RealD sf[4][4], std::vector<GaugeLinkField> &force_mu)
  {
    GRID_ASSERT(U.size() == Nd && force_mu.size() == Nd);
    GridBase *grid = U[0].Grid();
    for (int k = 0; k < 6; k++) conformable(lam[k]->Grid(), grid);
    if (grid_ != grid) Setup(grid);
    double t0 = usecond();
    const bool hit = PrepareLinks(U);
    double t1 = usecond();
    t_links_us_ += t1 - t0;

    std::vector<GaugeLinkField> Lp;
    Lp.reserve(6);
    for (int k = 0; k < 6; k++) Lp.emplace_back(ghost_->ExchangePeriodic(*lam[k]));
    double t2 = usecond();
    t_lambda_us_ += t2 - t1;

    GridBase *pgrid = ghost_->grids.back();
    const int nf = Nd + 6;
    View *v_host = (View *)malloc(nf * sizeof(View));
    for (int i = 0; i < Nd; i++) v_host[i] = Up_[i].View(AcceleratorRead);
    for (int k = 0; k < 6; k++) v_host[Nd + k] = Lp[k].View(AcceleratorRead);
    View *v = (View *)acceleratorAllocDevice(nf * sizeof(View));
    acceleratorCopyToDevice(v_host, v, nf * sizeof(View));

    GaugeLinkField Fp(pgrid);
    double t_kern = 0, t_ext = 0;
    for (int mu = 0; mu < Nd; mu++) {
      MuArgs A;
      A.mu = mu;
      int j = 0;
      for (int nu = 0; nu < Nd; nu++) {
        if (nu == mu) continue;
        A.nu[j] = nu;
        A.lk[j] = Nd + sig[mu][nu];
        A.sf[j] = sf[mu][nu];
        for (int q = 0; q < 6; q++) A.pt[j][q] = pt_[mu][nu][q];
        j++;
      }
      const double tk = usecond();
      {
        autoView(F_v, Fp, AcceleratorWrite);
        auto st_v = stencil_->View(AcceleratorRead);
        // Neighbour read of field `fld` at stencil point q of the current pair (lane-permuted).
#define CLV_RD(fld, q) coalescedReadGeneralPermute(v[fld][st_v.GetEntry(A.pt[jj][q], ss)->_offset], st_v.GetEntry(A.pt[jj][q], ss)->_permute, Nd)
        accelerator_for(ss, pgrid->oSites(), (size_t)pgrid->Nsimd(), {
          decltype(coalescedRead(v[0][0])) f, out, t, gx;
          f = Zero();
          f = Opaque(f);  // as the chain's force_mu = Zero(), read back from memory
          for (int jj = 0; jj < 3; jj++) {
            const int nu = A.nu[jj];
            const int lk = A.lk[jj];
            const int mu_ = A.mu;
            auto Unu0 = CLV_RD(nu, 0);     // U_nu(x)
            auto UnuM = CLV_RD(nu, 1);     // U_nu(x+mu)
            auto UnuN_ = CLV_RD(nu, 3);    // U_nu(x-nu)
            auto UnuMN_ = CLV_RD(nu, 5);   // U_nu(x+mu-nu)
            auto UmuN = CLV_RD(mu_, 2);    // U_mu(x+nu)
            auto UmuN_ = CLV_RD(mu_, 3);   // U_mu(x-nu)
            auto L0 = CLV_RD(lk, 0);       // lambda(x)
            auto LM = CLV_RD(lk, 1);       // lambda(x+mu)
            auto LN = CLV_RD(lk, 2);       // lambda(x+nu)
            auto LN_ = CLV_RD(lk, 3);      // lambda(x-nu)
            auto LMN = CLV_RD(lk, 4);      // lambda(x+mu+nu)
            auto LMN_ = CLV_RD(lk, 5);     // lambda(x+mu-nu)
            // One statement per chain kernel; Opaque() wherever the chain writes a field and
            // reads it back (each Cshift input/output, each closure of adj(.)), so every
            // product sees its operands in the same form as in the chain (see Opaque).
            // C1+ = tmp(x+mu) b(x+mu+nu), tmp = lambda U_nu, b(y) = U_mu(y-mu)^dag a(y-mu),
            //       a(y) = U_nu(y-nu)^dag (CovShiftIdentityBackward: adj closed into a field)
            auto t1 = Opaque(LM * UnuM);
            auto a = Opaque(adj(Unu0));
            auto b = Opaque(adj(UmuN) * a);
            out = Opaque(t1 * b);
            // C2+ = U_nu(x+mu) (tmp(x+nu)^dag a(x+nu)), tmp(y) = U_mu(y) adj(lambda)(y+mu)
            auto s = Opaque(adj(LMN));
            t = Opaque(UmuN * s);
            auto bp = Opaque(adj(t) * a);
            auto cf = Opaque(UnuM * bp);
            out = Opaque(out + cf);
            // C3+ = U_nu(x+mu) (U_mu(x+nu)^dag adj(tmp)(x)), tmp(y) = U_nu(y) adj(lambda)(y+nu)
            s = Opaque(adj(LN));
            t = Opaque(Unu0 * s);
            auto ap = Opaque(adj(t));
            auto bpp = Opaque(adj(UmuN) * ap);
            cf = Opaque(UnuM * bpp);
            out = Opaque(out + cf);
            // C4+ = d(x) lambda(x), d(x) = U_nu(x+mu) b(x+mu+nu)   (out += d * lambda: one kernel)
            auto d = Opaque(UnuM * b);
            out = Opaque(out + d * L0);
            // C1- = lambda(x+mu) g(x), g(x) = U_nu(x+mu-nu)^dag e(x+mu-nu), e(y) = U_mu(y-mu)^dag U_nu(y-mu)
            auto e = Opaque(adj(UmuN_) * UnuN_);
            gx = Opaque(adj(UnuMN_) * e);
            out = Opaque(out - LM * gx);
            // C2- = tmp(x+mu-nu)^dag e(x+mu-nu), tmp = adj(lambda) U_nu (adj inside the kernel)
            t = Opaque(adj(LMN_) * UnuMN_);
            auto g2 = Opaque(adj(t) * e);
            out = Opaque(out - g2);
            // C3- = U_nu(x+mu-nu)^dag (U_mu(x-nu)^dag tmp(x-nu)), tmp = lambda U_nu
            t = Opaque(LN_ * UnuN_);
            auto e3 = Opaque(adj(UmuN_) * t);
            auto g3 = Opaque(adj(UnuMN_) * e3);
            out = Opaque(out - g3);
            // C4- = g(x) lambda(x)
            out = Opaque(out - gx * L0);
            // the caller: force_mu -= signed_factor * Cmunu   (one kernel)
            f = Opaque(f - A.sf[jj] * out);
          }
          coalescedWrite(F_v[ss], f);
        });
#undef CLV_RD
      }
      const double te = usecond();
      force_mu[mu] = ghost_->Extract(Fp);
      t_kern += te - tk;
      t_ext += usecond() - te;
    }
    for (int i = 0; i < nf; i++) v_host[i].ViewClose();
    free(v_host);
    acceleratorFreeDevice(v);
    t_kernel_us_ += t_kern;
    t_extract_us_ += t_ext;
    if (hit) hits_++; else misses_++;
    Report();
  }

  long Hits() const { return hits_; }
  long Misses() const { return misses_; }

 private:
  GridBase *grid_ = nullptr;
  PaddedCell *ghost_ = nullptr;
  GeneralLocalStencil *stencil_ = nullptr;
  std::vector<GaugeLinkField> shadow_, Up_;
  bool links_valid_ = false;  // shadow_/Up_ hold a link set (reset by Setup)
  int pt_[4][4][6];
  long hits_ = 0, misses_ = 0;
  double t_links_us_ = 0, t_lambda_us_ = 0, t_kernel_us_ = 0, t_extract_us_ = 0, t_setup_us_ = 0;

  void Setup(GridBase *grid)
  {
    const double t0 = usecond();
    GridCartesian *cgrid = dynamic_cast<GridCartesian *>(grid);
    GRID_ASSERT(cgrid != nullptr);  // the links live on the full (non-checkerboarded) grid
    ghost_ = new PaddedCell(1, cgrid);  // a previous one (other grid) is leaked on purpose
    GridBase *pgrid = ghost_->grids.back();
    std::vector<Coordinate> shifts;
    auto unit = [](int a, int sa, int b, int sb) {
      Coordinate c(Nd, 0);
      if (a >= 0) c[a] += sa;
      if (b >= 0) c[b] += sb;
      return c;
    };
    shifts.push_back(unit(-1, 0, -1, 0));
    for (int mu = 0; mu < Nd; mu++) shifts.push_back(unit(mu, +1, -1, 0));
    for (int mu = 0; mu < Nd; mu++) shifts.push_back(unit(mu, -1, -1, 0));
    for (int mu = 0; mu < Nd; mu++)
      for (int nu = mu + 1; nu < Nd; nu++) shifts.push_back(unit(mu, +1, nu, +1));
    for (int mu = 0; mu < Nd; mu++)
      for (int nu = 0; nu < Nd; nu++)
        if (nu != mu) shifts.push_back(unit(mu, +1, nu, -1));
    GRID_ASSERT(shifts.size() == 27);
    auto find = [&shifts](const Coordinate &c) {
      for (int i = 0; i < (int)shifts.size(); i++) {
        bool eq = true;
        for (int d = 0; d < Nd; d++) eq = eq && (shifts[i][d] == c[d]);
        if (eq) return i;
      }
      GRID_ASSERT(0);
      return -1;
    };
    for (int mu = 0; mu < Nd; mu++)
      for (int nu = 0; nu < Nd; nu++) {
        if (mu == nu) continue;
        pt_[mu][nu][0] = find(unit(-1, 0, -1, 0));
        pt_[mu][nu][1] = find(unit(mu, +1, -1, 0));
        pt_[mu][nu][2] = find(unit(nu, +1, -1, 0));
        pt_[mu][nu][3] = find(unit(nu, -1, -1, 0));
        pt_[mu][nu][4] = find(unit(mu, +1, nu, +1));
        pt_[mu][nu][5] = find(unit(mu, +1, nu, -1));
      }
    stencil_ = new GeneralLocalStencil(pgrid, shifts);
    shadow_.assign(Nd, GaugeLinkField(grid));
    Up_.assign(Nd, GaugeLinkField(pgrid));
    links_valid_ = false;
    grid_ = grid;
    t_setup_us_ += usecond() - t0;
    if (grid->IsBoss()) {
      const double mb = double(pgrid->lSites()) * sizeof(typename GaugeLinkField::scalar_object) / 1048576.0;
      std::cout << GridLogMessage << "[CloverCmunuStencil] padded local " << pgrid->LocalDimensions()
                << ", 27-point depth-1 stencil built in " << t_setup_us_ * 1e-3 << " ms; persistent 4 padded links ("
                << 4 * mb << " MB) + 4 shadows + table per rank (HASEN_GRID_CLOVER_STENCIL=1)" << std::endl;
    }
  }

  // Exact key as stage 1 (and import_guard.h); refresh the padded links on any difference.
  bool PrepareLinks(const std::vector<GaugeLinkField> &U)
  {
    bool same = links_valid_;
    if (same) {
      GaugeLinkField diff(grid_);
      for (int mu = 0; mu < Nd && same; mu++) {
        diff = U[mu] - shadow_[mu];
        if (norm2(diff) != 0.0) same = false;
      }
    }
    if (!same) {
      for (int mu = 0; mu < Nd; mu++) {
        shadow_[mu] = U[mu];
        Up_[mu] = ghost_->ExchangePeriodic(U[mu]);
      }
      links_valid_ = true;
    }
    return same;
  }

  void Report() const
  {
    const long n = hits_ + misses_;
    if (grid_->IsBoss() && (((n & (n - 1)) == 0) || (n % 256 == 0))) {
      const double tot = t_links_us_ + t_lambda_us_ + t_kernel_us_ + t_extract_us_;
      std::cout << GridLogMessage << "[CloverCmunuStencil] passes=" << n << " link_hits=" << hits_
                << " link_misses=" << misses_ << " links_s=" << t_links_us_ * 1e-6
                << " lambda_exchange_s=" << t_lambda_us_ * 1e-6 << " kernel_s=" << t_kernel_us_ * 1e-6
                << " extract_s=" << t_extract_us_ * 1e-6 << " per_pass_ms=" << tot * 1e-3 / n
                << " (HASEN_GRID_CLOVER_STENCIL=1)" << std::endl;
    }
  }
};

}  // namespace Grid
