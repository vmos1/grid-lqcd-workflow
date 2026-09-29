#pragma once
// Log-determinant action for the even-site diagonal block in the
// EO-preconditioned Wilson-Clover operator -- COMPACT-CLOVER variant.
//
// This is a faithful copy of QCDLogDetCloverEOAction specialised for
// CompactWilsonCloverFermion.  The compact operator stores the clover term in a
// lean "diagonal + triangle" layout instead of the ~6 full-volume CloverField
// members of the non-compact operator, so it does NOT expose CloverTermEven /
// CloverTermInvEven.  The only difference here is that the full even-parity
// clover block (and its inverse) is reconstructed ON DEMAND into a transient
// scratch CloverField via CompactWilsonCloverHelpers::ConvertLayout, and that
// scratch is handed to the identical, already-tested logdet / Cmunu math.
//
// The scratch lives only for the duration of S() / deriv() (each runs ~twice
// per trajectory), so steady-state memory stays lean -- the whole point of the
// compact port.  See hasenbusch_tests_perlmutter.md and
// wclover_eo_compact_a_vs_b_comparison.md for context.
//
// det(M) = det(Mee) * det(Mpc)
// For Nf=2: S_logdet = -ln det(Mee†Mee) = -2 Σ_{x∈even} ln|det(Mee(x))|

#include <Grid/qcd/action/fermion/CompactWilsonCloverFermion.h>
#include <Grid/algorithms/blas/BatchedBlas.h>
#ifdef GRID_CUDA
#include <cublas_v2.h>
#endif

NAMESPACE_BEGIN(Grid);

// NB: the default CloverHelpers is CompactCloverHelpers (the compact operator is
// only instantiated in libGrid with Compact{,Exp}CloverHelpers), unlike the
// non-compact action which defaults to CloverHelpers.  CompactCloverHelpers
// provides the same Cmunu gauge-staple used by deriv().
template <class Impl, class CloverHelpers = Grid::CompactCloverHelpers<Impl>>
class QCDLogDetCompactCloverEOAction : public Action<typename Impl::GaugeField> {
public:
  INHERIT_IMPL_TYPES(Impl);
  INHERIT_CLOVER_TYPES(Impl);
  INHERIT_COMPACT_CLOVER_TYPES(Impl);

  typedef CompactWilsonCloverFermion<Impl, CloverHelpers> FermionOperator;

  QCDLogDetCompactCloverEOAction(FermionOperator &Op, int nf = 2)
      : FermOp(Op), Nf(nf) {}

  ~QCDLogDetCompactCloverEOAction() {
    if (n_deriv_ > 0) {
      auto fmt = [](uint64_t us) { return double(us) * 1e-6; };
      std::cout << GridLogMessage
                << "[QCDLogDetCompact timers/destructor] n_deriv=" << n_deriv_
                << "  total=" << fmt(t_total_us_) << " s"
                << "  importU=" << fmt(t_import_us_) << " s"
                << "  setCb=" << fmt(t_setcb_us_) << " s"
                << "  links=" << fmt(t_links_us_) << " s"
                << "  cmunu=" << fmt(t_cmunu_us_) << " s"
                << "  per-deriv=" << fmt(t_total_us_) / n_deriv_ * 1e3 << " ms"
                << std::endl;
    }
  }

  std::string action_name() override { return "QCDLogDetCompactCloverEOAction"; }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << FermOp.diag_mass - 4.0
       << " csw_r=" << 2.0 * FermOp.csw_r << " csw_t=" << 2.0 * FermOp.csw_t << std::endl;
    return os.str();
  }

  void refresh(const GaugeField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
  }

  // Reconstruct the full even-parity clover block Mee from the compact
  // Diagonal/Triangle storage into a transient scratch CloverField on the
  // RB (even) grid.  ConvertLayout sets the scratch checkerboard to match.
  void ReconstructEven(CloverField &out) {
    CompactWilsonCloverHelpers<Impl>::ConvertLayout(
        FermOp.DiagonalEven, FermOp.TriangleEven, out);
  }
  // Same for Mee^{-1} from the inverse Diagonal/Triangle storage.
  void ReconstructInvEven(CloverField &out) {
    CompactWilsonCloverHelpers<Impl>::ConvertLayout(
        FermOp.DiagonalInvEven, FermOp.TriangleInvEven, out);
  }

  // GPU log|det| using cuBLAS getrfBatched: after LU factorisation, the
  // diagonal of U gives log|det| as Σ log|U_kk|.  Same machine-precision
  // result as Eigen.determinant() but avoids the per-site CPU dispatch.
  // Even-parity only (matches CPU path which uses the reconstructed Mee).
#if defined(GRID_CUDA)
  RealD compute_logdet_gpu(CloverField &CT) {
    constexpr int N  = Ns * Impl::Dimension;     // 12 for Wilson SU(3)
    constexpr int N2 = N * N;                    // 144
    int lvol = CT.Grid()->lSites();

    typedef typename SiteClover::scalar_object SCsObj;
    std::vector<SCsObj> ct_lex(lvol);
    unvectorizeToLexOrdArray(ct_lex, CT);

    std::vector<ComplexD> M_host(lvol * N2);
    thread_for(site, lvol, {
      const SCsObj &Qx = ct_lex[site];
      for (int j = 0; j < Ns; ++j)
        for (int k = 0; k < Ns; ++k)
          for (int a = 0; a < Impl::Dimension; ++a)
            for (int b = 0; b < Impl::Dimension; ++b) {
              int row = a + j * Impl::Dimension;
              int col = b + k * Impl::Dimension;
              M_host[site * N2 + row + col * N] =
                  ComplexD(Qx()(j, k)(a, b));
            }
    });

    // Function-static persistent device buffers (parallel to InstantiateGPU).
    static deviceVector<ComplexD>  M_dev;
    static deviceVector<int>       pivot, info;
    static deviceVector<ComplexD*> Aptrs;
    static deviceVector<RealD>     logdet_per_site;
    static int cached_lvol = -1;
    if ((int)M_dev.size()           < lvol * N2) M_dev.resize(lvol * N2);
    if ((int)pivot.size()           < lvol * N)  pivot.resize(lvol * N);
    if ((int)info.size()            < lvol)      info.resize(lvol);
    if ((int)Aptrs.size()           < lvol)      Aptrs.resize(lvol);
    if ((int)logdet_per_site.size() < lvol)      logdet_per_site.resize(lvol);
    bool rebuild_ptrs = (cached_lvol != lvol);
    cached_lvol = lvol;

    acceleratorCopyToDevice(M_host.data(), &M_dev[0],
                            lvol * N2 * sizeof(ComplexD));
    if (rebuild_ptrs) {
      ComplexD *MA = &M_dev[0];
      ComplexD **Ap = &Aptrs[0];
      accelerator_for(i, lvol, 1, { Ap[i] = &MA[i * N2]; });
    }

    GridBLAS::Init();
    cublasHandle_t handle = GridBLAS::gridblasHandle;
    cublasStatus_t st = cublasZgetrfBatched(
        handle, N,
        reinterpret_cast<cuDoubleComplex **>(&Aptrs[0]),
        N, &pivot[0], &info[0], lvol);
    if (st != 0) {
      std::cout << GridLogError
                << "[QCDLogDetCompact::compute_logdet_gpu] cublasZgetrfBatched status=" << st
                << std::endl;
      abort();
    }

    // log|det(A)| = Σ_k log|U_kk|.  Sign of det doesn't matter for log|·|.
    ComplexD *MA = &M_dev[0];
    RealD    *LD = &logdet_per_site[0];
    accelerator_for(i, lvol, 1, {
      RealD s = 0.0;
      for (int k = 0; k < N; ++k) {
        ComplexD u_kk = MA[i * N2 + k + k * N];
        // |z|² = re² + im² then 0.5·log to avoid std::abs() device unavailability
        RealD mod2 = u_kk.real() * u_kk.real() + u_kk.imag() * u_kk.imag();
        s += 0.5 * ::log(mod2);
      }
      LD[i] = s;
    });

    std::vector<RealD> ld_host(lvol);
    acceleratorCopyFromDevice(&logdet_per_site[0], ld_host.data(),
                              lvol * sizeof(RealD));
    RealD logdet = 0.0;
    for (int i = 0; i < lvol; ++i) logdet += ld_host[i];

    // Same WCF_BUFFERS_TRANSIENT=1 contract as InstantiateGPU.  S() runs only
    // twice per traj (initial+final H) so transient cost is negligible.
    static int transient = []() {
      const char *e = std::getenv("WCF_BUFFERS_TRANSIENT");
      return (e && *e && std::atoi(e)) ? 1 : 0;
    }();
    if (transient) {
      M_dev.resize(0);  pivot.resize(0);  info.resize(0);
      Aptrs.resize(0);  logdet_per_site.resize(0);
      cached_lvol = -1;
    }
    return logdet;
  }
#endif

  RealD S(const GaugeField &U) override {
    FermOp.ImportGauge(U);

    // Reconstruct the full even-parity clover block on demand from the compact
    // Diagonal/Triangle storage.  Transient: lives only for this call.
    CloverField CTe(FermOp.DiagonalEven.Grid());
    ReconstructEven(CTe);

#if defined(GRID_CUDA)
    static int use_gpu = []() {
      const char *e = std::getenv("WCF_LOGDET_GPU");
      if (!e || !*e) return 1;  // default ON, bit-exact 2026-05-26
      return std::atoi(e);
    }();
    if (use_gpu) {
      RealD logdet = compute_logdet_gpu(CTe);
      FermOp.GaugeGrid()->GlobalSum(logdet);
      RealD action = -RealD(Nf) * logdet;
      std::cout << GridLogMessage << "[" << action_name()
                << "] S = " << action << " (GPU)" << std::endl;
      return action;
    }
#endif

    int DimRep = Impl::Dimension;
    int lvol = CTe.Grid()->lSites();

    std::vector<typename SiteClover::scalar_object> ct_lex(lvol);
    unvectorizeToLexOrdArray(ct_lex, CTe);

    // OMP-parallel over sites.  Each thread keeps its own EigenM scratch and
    // partial logdet sum.  Equivalent at machine precision since logdet is a
    // sum of independent per-site terms.
    RealD logdet = 0.0;
    std::vector<RealD> partial(thread_max(0), 0.0);
    thread_for(site, lvol, {
      Eigen::MatrixXcd EigenM = Eigen::MatrixXcd::Zero(Ns * DimRep, Ns * DimRep);
      for (int j = 0; j < Ns; j++)
        for (int k = 0; k < Ns; k++)
          for (int a = 0; a < DimRep; a++)
            for (int b = 0; b < DimRep; b++)
              EigenM(a + j * DimRep, b + k * DimRep) =
                  std::complex<double>(ct_lex[site]()(j, k)(a, b));
      partial[thread_num(0)] += std::log(std::abs(EigenM.determinant()));
    });
    for (auto &p : partial) logdet += p;

    FermOp.GaugeGrid()->GlobalSum(logdet);
    RealD action = -RealD(Nf) * logdet;
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action << std::endl;
    return action;
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    static int use_gpu = []() {
      const char *e = std::getenv("WCF_LOGDET_DERIV_GPU");
      if (!e || !*e) return 1;  // default ON, exact zero diff vs CPU 2026-05-26
      return std::atoi(e);
    }();
    if (use_gpu) { deriv_gpu(U, dSdU); return; }
    deriv_cpu(U, dSdU);
  }

  void deriv_cpu(const GaugeField &U, GaugeField &dSdU) {
    auto t_total0 = usecond();
    auto t_imp0 = usecond();
    FermOp.ImportGauge(U);
    t_import_us_ += usecond() - t_imp0;

    GridBase *fgrid = FermOp.GaugeGrid();

    // Reconstruct Mee^{-1} on even sites from compact storage (transient).
    CloverField CTInvEven(FermOp.DiagonalInvEven.Grid());
    ReconstructInvEven(CTInvEven);

    // Mee^{-1} on even sites, zero on odd — acts as the "propagator" Lambda.
    auto t_cb0 = usecond();
    CloverField Lambda(fgrid);
    Lambda = Zero();
    setCheckerboard(Lambda, CTInvEven);
    t_setcb_us_ += usecond() - t_cb0;

    auto t_lnk0 = usecond();
    std::vector<GaugeLinkField> Ulinks(Nd, fgrid);
    for (int mu = 0; mu < Nd; ++mu)
      Ulinks[mu] = PeekIndex<LorentzIndex>(FermOp.Umu, mu);
    t_links_us_ += usecond() - t_lnk0;

    // Sigma loop: same structure as WilsonCloverFermion::MDeriv.
    // Uses 12-element sigma array (including MinusSigma for nu<mu pairs).
    Gamma::Algebra sigma[] = {
        Gamma::Algebra::SigmaXY,
        Gamma::Algebra::SigmaXZ,
        Gamma::Algebra::SigmaXT,
        Gamma::Algebra::MinusSigmaXY,
        Gamma::Algebra::SigmaYZ,
        Gamma::Algebra::SigmaYT,
        Gamma::Algebra::MinusSigmaXZ,
        Gamma::Algebra::MinusSigmaYZ,
        Gamma::Algebra::SigmaZT,
        Gamma::Algebra::MinusSigmaXT,
        Gamma::Algebra::MinusSigmaYT,
        Gamma::Algebra::MinusSigmaZT};

    auto t_cmn0 = usecond();
    GaugeLinkField force_mu(fgrid), lambda(fgrid);
    GaugeField clover_force(fgrid);
    int count = 0;
    clover_force = Zero();
    for (int mu = 0; mu < 4; mu++) {
      force_mu = Zero();
      for (int nu = 0; nu < 4; nu++) {
        if (mu == nu) continue;

        RealD factor = (nu == 3 || mu == 3) ? 2.0 * FermOp.csw_t
                                             : 2.0 * FermOp.csw_r;
        CloverField Slambda = Gamma(sigma[count]) * Lambda;
        lambda = TraceIndex<SpinIndex>(Slambda);
        force_mu -= factor * CloverHelpers::Cmunu(Ulinks, lambda, mu, nu);
        count++;
      }
      pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
    }

    // S = -Nf * ln|det Mee|.
    // clover_force = Tr(Mee^{-1} dMee/dU) in UdSdU convention.
    // Grid pseudofermion convention: UdSdU = -dS/dU (force opposes gradient).
    // dS/dU = -Nf * clover_force, so UdSdU = +Nf * clover_force.
    dSdU = RealD(Nf) * clover_force;
    t_cmunu_us_ += usecond() - t_cmn0;
    t_total_us_ += usecond() - t_total0;
    n_deriv_++;
  }

  // GPU/RB-Even path.  Two optimisations:
  //   (a) Run Gamma·X and TraceIndex on the RB-Even reconstructed Mee^{-1}
  //       directly — half the lattice traffic vs the CPU full-grid path.
  //   (b) Exploit σ_νμ = -σ_μν: only 6 unique sigmas instead of 12.  The
  //       Cmunu loop applies the sign at consumption (nu<mu → negate).
  // Bit-identical to deriv_cpu (Grid expression templates fuse Gamma+Trace,
  // and the per-site math is the same arithmetic in the same order).
  void deriv_gpu(const GaugeField &U, GaugeField &dSdU) {
    auto t_total0 = usecond();
    auto t_imp0 = usecond();
    FermOp.ImportGauge(U);
    t_import_us_ += usecond() - t_imp0;

    GridBase *fgrid  = FermOp.GaugeGrid();
    GridBase *rbgrid = FermOp.DiagonalInvEven.Grid();

    // Reconstruct Mee^{-1} on the RB (even) grid from compact storage (transient).
    CloverField CTInvEven(rbgrid);
    ReconstructInvEven(CTInvEven);

    auto t_lnk0 = usecond();
    std::vector<GaugeLinkField> Ulinks(Nd, fgrid);
    for (int mu = 0; mu < Nd; ++mu)
      Ulinks[mu] = PeekIndex<LorentzIndex>(FermOp.Umu, mu);
    t_links_us_ += usecond() - t_lnk0;

    // 6 unique +sigma_{μν} for μ<ν.  Index k = sigma_idx[mu][nu] gives the
    // entry; sign is +1 if μ<ν, -1 if μ>ν (since σ_νμ = -σ_μν).
    const Gamma::Algebra positive_sigma[6] = {
        Gamma::Algebra::SigmaXY,   // (0,1)
        Gamma::Algebra::SigmaXZ,   // (0,2)
        Gamma::Algebra::SigmaXT,   // (0,3)
        Gamma::Algebra::SigmaYZ,   // (1,2)
        Gamma::Algebra::SigmaYT,   // (1,3)
        Gamma::Algebra::SigmaZT};  // (2,3)
    const int sigma_idx[4][4] = {{-1, 0, 1, 2},
                                  { 0,-1, 3, 4},
                                  { 1, 3,-1, 5},
                                  { 2, 4, 5,-1}};

    // Build 6 RB-Even sigma-trace ColourMatrices via Grid Lattice expressions.
    // Each is one fused kernel: Gamma(σ_k) · CTInvEven → TraceIndex<SpinIndex>.
    auto t_cb0 = usecond();
    std::array<GaugeLinkField, 6> lambda_e{
        GaugeLinkField(rbgrid), GaugeLinkField(rbgrid), GaugeLinkField(rbgrid),
        GaugeLinkField(rbgrid), GaugeLinkField(rbgrid), GaugeLinkField(rbgrid)};
    for (int k = 0; k < 6; ++k) {
      CloverField Slambda_e =
          Gamma(positive_sigma[k]) * CTInvEven;
      lambda_e[k] = TraceIndex<SpinIndex>(Slambda_e);
    }

    // Push to full-grid Lattice<ColourMatrix> (only Even populated, Odd = 0)
    // for Cmunu input.  Mirrors CPU path's implicit zero-on-odd.
    std::array<GaugeLinkField, 6> lambda_full{
        GaugeLinkField(fgrid), GaugeLinkField(fgrid), GaugeLinkField(fgrid),
        GaugeLinkField(fgrid), GaugeLinkField(fgrid), GaugeLinkField(fgrid)};
    for (int k = 0; k < 6; ++k) {
      lambda_full[k] = Zero();
      setCheckerboard(lambda_full[k], lambda_e[k]);
    }
    t_setcb_us_ += usecond() - t_cb0;

    auto t_cmn0 = usecond();
    GaugeLinkField force_mu(fgrid);
    GaugeField clover_force(fgrid);
    clover_force = Zero();
    for (int mu = 0; mu < 4; mu++) {
      force_mu = Zero();
      for (int nu = 0; nu < 4; nu++) {
        if (mu == nu) continue;

        RealD factor = (nu == 3 || mu == 3) ? 2.0 * FermOp.csw_t
                                             : 2.0 * FermOp.csw_r;
        // σ_νμ = -σ_μν → fold the sign into `factor` so we don't need a second
        // (signed) lambda copy.
        RealD signed_factor = (mu < nu) ? factor : -factor;
        int k = sigma_idx[mu][nu];
        force_mu -= signed_factor *
                    CloverHelpers::Cmunu(Ulinks, lambda_full[k], mu, nu);
      }
      pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
    }
    dSdU = RealD(Nf) * clover_force;
    t_cmunu_us_ += usecond() - t_cmn0;
    t_total_us_ += usecond() - t_total0;
    n_deriv_++;
  }

private:
  FermionOperator &FermOp;
  int Nf;

  // Per-component timers (accumulate across deriv calls; printed on destruct).
  uint64_t n_deriv_ = 0;
  uint64_t t_total_us_  = 0;
  uint64_t t_import_us_ = 0;
  uint64_t t_setcb_us_  = 0;
  uint64_t t_links_us_  = 0;
  uint64_t t_cmunu_us_  = 0;
};

NAMESPACE_END(Grid);
