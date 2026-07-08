#include "params.h"
#include "eig_diag.h"
#include <cstring>
#include <cstdio>
#include <chrono>
#include <set>
#include <sstream>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CompactWilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/qcd/observables/plaquette.h>
#include <Grid/qcd/observables/polyakov_loop.h>
#ifdef HAVE_HDF5
#include <Grid/serialisation/Hdf5IO.h>
#endif
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavour.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverQudaForceRationalActionMP.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverQudaForceActionMP.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioActionQuda.h>
#include <Grid/algorithms/iterative/QudaMGSchurSolver.h>
#include <Grid/algorithms/iterative/QudaCGSchurSolver.h>
#endif
#include <Grid/algorithms/iterative/ConjugateGradientMixedPrec.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h>

using namespace TXQCDProduction;

// Mixed-precision CG wrapper that satisfies the OperatorFunction<FieldD>
// interface expected by TwoFlavourSchurCloverAction as its derivative solver.
// Assumes the single-precision Schur operator has already been updated (by
// TwoFlavourSchurCloverActionMP::deriv() below) to match the current gauge.
namespace Grid {
template <class FieldD, class FieldF, class SchurOpD, class SchurOpF>
class MixedPrecCGWrapper : public OperatorFunction<FieldD> {
 public:
  using OperatorFunction<FieldD>::operator();

  MixedPrecCGWrapper(RealD tol, int max_inner, int max_outer,
                     GridBase *rbgrid_f,
                     SchurOpD &schur_d, SchurOpF &schur_f)
      : tol_(tol), max_inner_(max_inner), max_outer_(max_outer),
        rbgrid_f_(rbgrid_f), schur_d_(schur_d), schur_f_(schur_f) {}

  void operator()(LinearOperatorBase<FieldD> &LinOp_unused,
                  const FieldD &src, FieldD &sol) override {
    MixedPrecisionConjugateGradient<FieldD, FieldF> MPCG(
        tol_, max_inner_, max_outer_, rbgrid_f_, schur_f_, schur_d_);
    MPCG(src, sol);
  }

 private:
  RealD tol_;
  int max_inner_, max_outer_;
  GridBase *rbgrid_f_;
  SchurOpD &schur_d_;
  SchurOpF &schur_f_;
};

// Subclass of TwoFlavourSchurCloverAction that keeps the SP operator in sync
// with the raw gauge field before each deriv().  Passes the CG work to the
// mixed-precision wrapper installed as the DerivativeSolver.
template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class TwoFlavourSchurCloverActionMP
    : public TwoFlavourSchurCloverAction<ImplD, FermOpD_> {
 public:
  typedef TwoFlavourSchurCloverAction<ImplD, FermOpD_> Base;
  typedef FermOpD_ FermOpD;
  typedef FermOpF_ FermOpF;
  typedef typename ImplD::GaugeField GaugeField;

  TwoFlavourSchurCloverActionMP(typename Base::FermionOperator &opD,
                                FermOpF &opF,
                                OperatorFunction<typename Base::FermionField> &DS,
                                OperatorFunction<typename Base::FermionField> &AS)
      : Base(opD, DS, AS), opF_(opF) {}

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    // Refresh SP operator from the raw gauge field U (per-mu precisionChange).
    typename ImplF::GaugeField UmuF(opF_.GaugeGrid());
    typename ImplD::GaugeLinkField U_d(U.Grid());
    typename ImplF::GaugeLinkField U_f(opF_.GaugeGrid());
    for (int mu = 0; mu < Nd; ++mu) {
      U_d = PeekIndex<LorentzIndex>(U, mu);
      precisionChange(U_f, U_d);
      PokeIndex<LorentzIndex>(UmuF, U_f, mu);
    }
    opF_.ImportGauge(UmuF);
    // Delegate to base; base's DerivativeSolver is MixedPrecCGWrapper which
    // will invoke MPCG against the now-synced SP Schur operator.
    Base::deriv(U, dSdU);
  }

 private:
  FermOpF &opF_;
};
}  // namespace Grid

#ifdef GRID_HAVE_QUDA
// TwoFlavourSchurCloverRatioActionQuda now lives in its own additive
// Grid-TXQCD header (shared with gen_qcd_hasenbusch_tune_compact_schur.cc) --
// see TwoFlavourSchurCloverRatioActionQuda.h for the rationale.
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioActionQuda.h>
#endif

// OneFlavourSchurCloverRationalActionMP — mixed-precision rational action
// shared with gen_txqcd_cfgs.cc.
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>
// EVEN-parity Grid-native Schur monomial — same parity as the QUDA force path,
// opt-in via STRANGE_EVEN=1 for same-parity grid-vs-quda force validation.
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h>

struct QcdDiag : public HmcObservable<LatticeGaugeField> {
  struct ActionRef { std::string name; Action<LatticeGaugeField> *action; };

  std::string prefix_;
  int interval_;
  std::vector<ActionRef> actions_;
  SmearedConfiguration<PeriodicGimplR> &smear_;
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  GridParallelRNG &prng_;

  std::vector<int>    traj_;
  std::vector<RealD>  plaq_, vev_trminv_;
  std::vector<std::vector<RealD>> force_avg_, force_max_, fdt_avg_, fdt_max_;
  // Per-traj eigenvalue diagnostic (γ5·M signed Rayleigh quotients).  Enabled
  // by EIG_DIAG=1 env; empty inner vectors when disabled.
  std::vector<std::vector<RealD>> eig_M2_, eig_g5M_;
  // Sign-problem order parameters (basis-independent): smallest |γ5·M|
  // and #modes with |γ5·M|<zero_eps.  These are the sharp det-sign
  // diagnostics — immune to ± near-degenerate-pair relabeling.
  std::vector<RealD> eig_minabs_, eig_nnear_;

  QcdDiag(const std::string &prefix, int interval,
          std::vector<ActionRef> actions,
          SmearedConfiguration<PeriodicGimplR> &smear,
          GridCartesian &grid, GridRedBlackCartesian &rbgrid,
          GridParallelRNG &prng)
      : prefix_(prefix), interval_(interval), actions_(std::move(actions)),
        smear_(smear), grid_(grid), rbgrid_(rbgrid), prng_(prng) {}

  void TrajectoryComplete(int traj, LatticeGaugeField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    traj_.push_back(traj);
    RealD pl = WilsonLoops<PeriodicGimplR>::avgPlaquette(U);
    plaq_.push_back(pl);
    std::cout << GridLogMessage << "[QcdDiag] traj=" << traj << " plaq=" << pl
              << std::endl;

    int na = (int)actions_.size();
    std::vector<RealD> fa(na), fm(na), fdta(na), fdtm(na);
    for (int i = 0; i < na; ++i) {
      fa[i]   = actions_[i].action->deriv_norm_average();
      fm[i]   = actions_[i].action->deriv_max_average();
      fdta[i] = actions_[i].action->Fdt_norm_average();
      fdtm[i] = actions_[i].action->Fdt_max_average();
    }
    force_avg_.push_back(fa); force_max_.push_back(fm);
    fdt_avg_.push_back(fdta); fdt_max_.push_back(fdtm);

    smear_.set_Field(U);
    LatticeGaugeField Usm = smear_.get_SmearedU();
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    // Antiperiodic BC in time (direction Nd-1=3) to match chroma's
    // <boundary>1 1 1 -1</boundary>.  Grid's default is all-periodic.
    WilsonImplParams impl_p;
    impl_p.boundary_phases.resize(Nd, 1.0);
    impl_p.boundary_phases[Nd - 1] = -1.0;  // antiperiodic time
    WCF Dw(Usm, grid_, rbgrid_, mass_light, csw, csw,
           WilsonAnisotropyCoefficients(), impl_p);
    MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    RealD V = (RealD)grid_.gSites();
    RealD acc = 0.0;
    for (int h = 0; h < n_vev_noise; ++h) {
      LatticeFermion eta(&grid_), b(&grid_), x(&grid_);
      gaussian(prng_, eta);
      Dw.Mdag(eta, b);
      x = Zero();
      CG(HermOp, b, x);
      acc += innerProduct(eta, x).real() / (2.0 * V);
    }
    vev_trminv_.push_back(acc / n_vev_noise);

    // ---- γ5·M signed eigenvalue diagnostic (opt-in via EIG_DIAG=1) --------
    std::vector<RealD> eig_M2, eig_g5M;
    if (eig_diag_enabled()) {
      EigDiagParams ep = eig_diag_params_from_env();
      RunEigDiagQcd(Dw, &grid_, prng_, ep, eig_M2, eig_g5M);
      std::cout << GridLogMessage << "[QcdDiag] traj=" << traj
                << " γ5·M signed lowest |·|:";
      for (auto e : eig_g5M) std::cout << " " << e;
      std::cout << std::endl;
    }
    eig_M2_.push_back(eig_M2);
    eig_g5M_.push_back(eig_g5M);
    {
      RealD eig_ma; int eig_nn;
      eig_order_params(eig_g5M, eig_diag_params_from_env().zero_eps,
                       eig_ma, eig_nn);
      eig_minabs_.push_back(eig_ma);
      eig_nnear_.push_back((RealD)eig_nn);
    }

#ifdef HAVE_HDF5
    if (traj % interval_ == 0) {
      std::string fname = prefix_ + "." + std::to_string(traj) + ".h5";
      Hdf5Writer wr(fname);
      write(wr, "traj", traj_);
      write(wr, "plaq", plaq_);
      write(wr, "vev_trminv", vev_trminv_);
      write(wr, "force_avg", force_avg_);
      write(wr, "force_max", force_max_);
      write(wr, "fdt_avg", fdt_avg_);
      write(wr, "fdt_max", fdt_max_);
      write(wr, "eig_M2", eig_M2_);
      write(wr, "eig_g5M", eig_g5M_);
      write(wr, "eig_min_abs_g5M", eig_minabs_);
      write(wr, "eig_n_near_zero", eig_nnear_);
      std::vector<std::string> names;
      for (auto &a : actions_) names.push_back(a.name);
      write(wr, "action_names", names);
      traj_.clear(); plaq_.clear(); vev_trminv_.clear();
      force_avg_.clear(); force_max_.clear();
      fdt_avg_.clear(); fdt_max_.clear();
      eig_M2_.clear(); eig_g5M_.clear();
      eig_minabs_.clear(); eig_nnear_.clear();
      std::cout << GridLogMessage << "Diagnostics written to " << fname << std::endl;
    }
#endif
  }
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // Multi-stream support: if the STREAM_ID env var is set (0, 1, 2, ...),
  // write to a stream-indexed cfg dir and offset RNG seeds so each stream is
  // an independent Markov chain.  STREAM_ID unset => default behaviour
  // (cfgs/qcd, original seed) for backwards compatibility.
  int stream_id = -1;
  if (const char *sid = std::getenv("STREAM_ID"); sid && *sid) stream_id = std::atoi(sid);
  std::string cfg_dir = (stream_id < 0)
      ? qcd_cfg_dir()
      : (qcd_cfg_dir() + "_s" + std::to_string(stream_id));
  // Optional suffix so action-variant / tuning variants don't clobber the
  // main stream dirs (e.g. SUFFIX=_lwfix → cfgs/qcd_s0_lwfix).
  if (const char *sfx = std::getenv("SUFFIX"); sfx && *sfx) cfg_dir += sfx;
  std::cout << GridLogMessage << "STREAM_ID=" << stream_id
            << " cfg_dir=" << cfg_dir << std::endl;

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm + n_prod;
  if (const char *nt = std::getenv("N_TRAJ"); nt && *nt) total_traj = std::atoi(nt);
  std::cout << GridLogMessage << "total_traj=" << total_traj << std::endl;
  mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int start_traj = 0;
  int latest = -1;
  for (int t = meas_skip; t <= total_traj; t += meas_skip) {
    if (file_exists(cfg_dir + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(cfg_dir + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }

  LatticeGaugeField Umu(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj " << latest << std::endl;
    std::string cf = cfg_dir + "/ckpoint_lat." + std::to_string(latest);
    std::string rf = cfg_dir + "/ckpoint_rng." + std::to_string(latest);
    FieldMetaData header;
    NerscIO::readRNGState(sRNG, pRNG, header, rf);
    typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
    NerscIO::readConfiguration<GaugeStats>(Umu, header, cf);
    start_traj = latest;
  } else if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    // Import an external thermalized config (chroma LIME or NERSC) as the
    // starting gauge field — skips tepid-start thermalization entirely.
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic << std::endl;
    // Auto-detect NERSC vs LIME by magic bytes (same logic as compute_plaq).
    FILE *f = std::fopen(ic, "rb");
    char magic[16] = {0};
    if (f) { std::fread(magic, 1, sizeof(magic), f); std::fclose(f); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      NerscIO::readConfiguration<GaugeStats>(Umu, header, std::string(ic));
    } else {
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(Umu, header);
      IR.close();
    }
    // Still need RNG seeds — use stream-id offset as default.
    int sid = std::max(0, stream_id);
    sRNG.SeedFixedIntegers({11 + 100*sid, 12 + 100*sid, 13 + 100*sid,
                            14 + 100*sid, 15 + 100*sid});
    pRNG.SeedFixedIntegers({16 + 100*sid, 17 + 100*sid, 18 + 100*sid,
                            19 + 100*sid, 20 + 100*sid});
  } else {
    // Offset RNG seeds by 100 × stream_id so streams are independent.
    int sid = std::max(0, stream_id);
    sRNG.SeedFixedIntegers({11 + 100*sid, 12 + 100*sid, 13 + 100*sid,
                            14 + 100*sid, 15 + 100*sid});
    pRNG.SeedFixedIntegers({16 + 100*sid, 17 + 100*sid, 18 + 100*sid,
                            19 + 100*sid, 20 + 100*sid});
    std::string start_type = "tepid";
    if (const char *st = std::getenv("START_TYPE"); st && *st) start_type = st;
    // WEAK_FIELD_SCALE: U_mu = exp(i * scale * Σ_a c_a t_a) per link via
    // LieRandomize.  Grid's default TepidConfiguration uses scale=0.01 which
    // gives plaq ≈ 0.99996; chroma's WEAK_FIELD uses ≈0.1 giving plaq ≈ 0.997.
    // Default raised to 0.1 to match chroma's convention.
    double wf_scale = 0.1;
    if (const char *ws = std::getenv("WEAK_FIELD_SCALE"); ws && *ws) wf_scale = std::atof(ws);
    std::cout << GridLogMessage << "START_TYPE=" << start_type
              << "  WEAK_FIELD_SCALE=" << wf_scale << std::endl;
    if (start_type == "hot") {
      SU<Nc>::HotConfiguration(pRNG, Umu);
    } else if (start_type == "cold") {
      SU<Nc>::ColdConfiguration(pRNG, Umu);
    } else {
      // Scaled tepid / weak-field: LieRandomize per link with caller-chosen
      // scale, bypassing Grid's hard-coded 0.01.
      LatticeColourMatrix Ulink(Umu.Grid());
      for (int mu = 0; mu < Nd; ++mu) {
        SU<Nc>::LieRandomize(pRNG, Ulink, wf_scale);
        PokeIndex<LorentzIndex>(Umu, Ulink, mu);
      }
    }
  }

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

  // ── Hasenbusch ladder ─────────────────────────────────────────────────────
  // HASEN_LADDER: comma-separated masses light→heavy, e.g.
  //   "-0.2416,-0.2400,-0.2320,-0.2180,-0.1870"
  // Lightest entry must equal mass_light.  Heaviest entry is the bare-det tail
  // mass (no Pauli-Villars appended) -- matches the Chroma cfg_2000 action and
  // the already-validated compact_schur.cc construction.  If unset: single-
  // level (no Hasenbusch) -- byte-for-byte today's behavior.
  std::vector<RealD> ladder;
  if (const char *hl = std::getenv("HASEN_LADDER"); hl && *hl) {
    std::string s(hl); size_t pos = 0, c;
    while ((c = s.find(',', pos)) != std::string::npos) {
      ladder.push_back(std::atof(s.substr(pos, c - pos).c_str())); pos = c + 1;
    }
    if (pos < s.size()) ladder.push_back(std::atof(s.substr(pos).c_str()));
    if (ladder.size() < 2) { std::cerr << "HASEN_LADDER needs >=2 masses.\n"; exit(1); }
    for (size_t i = 1; i < ladder.size(); ++i)
      if (!(ladder[i] > ladder[i-1])) {
        std::cerr << "HASEN_LADDER must be strictly increasing.\n"; exit(1);
      }
  } else {
    ladder = { mass_light };
    std::cout << GridLogMessage << "HASEN_LADDER not set -- single-level baseline." << std::endl;
  }
  std::cout << GridLogMessage << "Hasenbusch chain (" << (ladder.size()-1)
            << " ratios + bare-det tail at " << ladder.back() << "):";
  for (auto m : ladder) std::cout << " " << m;
  std::cout << std::endl;

  // Single-precision grid + gauge field + operator for mixed-precision CG.
  GridCartesian        GridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
  GridRedBlackCartesian RBGridF(&GridF);
  LatticeGaugeFieldF UmuF(&GridF);
  {
    LatticeColourMatrix  U_d(&Grid);
    LatticeColourMatrixF U_f(&GridF);
    for (int mu = 0; mu < Nd; ++mu) {
      U_d = PeekIndex<LorentzIndex>(Umu, mu);
      precisionChange(U_f, U_d);
      PokeIndex<LorentzIndex>(UmuF, U_f, mu);
    }
  }
  typedef WilsonCloverFermion<WilsonImplF, CloverHelpers<WilsonImplF>> WCF_f;
  // Antiperiodic BC in time to match chroma's <boundary>1 1 1 -1</boundary>.
  // Grid default is all-periodic → wrong det(M) → equilibrium plaq shift.
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;  // antiperiodic time
  WilsonImplParams impl_pF;
  impl_pF.boundary_phases.resize(Nd, 1.0);
  impl_pF.boundary_phases[Nd - 1] = -1.0;
  // SP tail operator -- bound to the HEAVIEST ladder mass (== mass_light when
  // HASEN_LADDER is unset, identical to today).
  WCF_f FermOpF(UmuF, GridF, RBGridF, ladder.back(), csw, csw,
                WilsonAnisotropyCoefficients(), impl_pF);

  // Light quarks (Nf=2): EO-preconditioned LogDet + Hasenbusch ratio chain +
  // Schur tail.  One DP operator per ladder mass; small Δm between rungs ->
  // well-conditioned -> DP CG suffices for the ratio rungs (mirrors
  // gen_qcd_hasenbusch_tune_compact_schur.cc's rationale).
  int n_ops = (int)ladder.size();
  std::vector<std::unique_ptr<WCF>> LightOps;
  for (int i = 0; i < n_ops; ++i)
    LightOps.emplace_back(std::make_unique<WCF>(
        Umu, Grid, RBGrid, ladder[i], csw, csw,
        WilsonAnisotropyCoefficients(), impl_p));
  std::cout << GridLogMessage << "Built " << n_ops << " DP light operators." << std::endl;
  // FermOp = the tail operator (heaviest ladder mass).  All existing
  // FermOp-based code below (mixed-precision tail action, TEST_SIGMA_LOOP,
  // TEST_LAMBDA, FD_NONEO, QUDA_FORCE_LIGHT) is unchanged and now transparently
  // targets the tail -- identical to today when HASEN_LADDER is unset.
  WCF &FermOp = *LightOps.back();

  // Action solver (accept/reject): tight DP CG at cg_tol=1e-8.
  // Derivative solver (MD force): mixed-precision CG (SP inner + DP correction),
  // outer tolerance 1e-6 (mdtol bias cancels on Metropolis accept/reject).
  RealD cg_action_tol = cg_tol;
  if (const char *t = std::getenv("CG_TOL"); t && *t) cg_action_tol = std::atof(t);
  std::cout << GridLogMessage << "CG_TOL=" << cg_action_tol << std::endl;
  ConjugateGradient<LatticeFermion> CG_action(cg_action_tol, cg_max);
  SchurDifferentiableOperator<WilsonImplR> SchurOpD(FermOp);
  SchurDifferentiableOperator<WilsonImplF> SchurOpF(FermOpF);
  // CG_MD_TOL env var: MD force CG tolerance.  Default 1e-6 (chroma-style),
  // but force-FD test showed 0.2% force-action mismatch in TwoFlavourSchurMP
  // at this tol → tightening here lets us probe whether the residual
  // mismatch is just sloppy CG or something structural.
  RealD cg_md_tol = 1e-6;
  if (const char *t = std::getenv("CG_MD_TOL"); t && *t) cg_md_tol = std::atof(t);
  std::cout << GridLogMessage << "CG_MD_TOL=" << cg_md_tol << std::endl;
  MixedPrecCGWrapper<LatticeFermion, LatticeFermionF,
                     SchurDifferentiableOperator<WilsonImplR>,
                     SchurDifferentiableOperator<WilsonImplF>>
      CG_md(cg_md_tol, cg_max, 50, &RBGridF, SchurOpD, SchurOpF);
  // LogDet is at the LIGHTEST ladder mass -- the EE determinants telescope
  // exactly across the chain (see TwoFlavourSchurCloverRatioAction.h header).
  QCDLogDetCloverEOAction<WilsonImplR> LightLogDet(*LightOps[0], 2);
  LightLogDet.is_smeared = true;
  // SOLVER_DEBUG=1 forces DP CG_action also for the derivative — used to
  // disentangle MP-CG bug from structural deriv vs S inconsistency.
  bool solver_debug = false;
  if (const char *t = std::getenv("SOLVER_DEBUG"); t && std::atoi(t)) solver_debug = true;
  // QUDA_FORCE=1 (Phase D) routes the Nf=2 light through computeCloverForceQuda
  // with PyQUDA's dagger=YES convention (cos=1.0 vs Path A on hot 4⁴).
  std::unique_ptr<TwoFlavourSchurCloverActionMP<WilsonImplR, WilsonImplF>>
      LightSchurPF_baseHolder;
  Action<LatticeGaugeField> *LightSchurPFptr = nullptr;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<TwoFlavourSchurCloverQudaForceActionMP<WilsonImplR, WilsonImplF>>
      LightSchurPF_qudaHolder;
  // QUDA_FORCE_LIGHT=1 opts in to the (currently buggy — cos=0.25 vs Path A
  // due to ODD_ODD_ASYMMETRIC parity convention) TwoFlavour QUDA force class.
  // Default off so QUDA_FORCE=1 only routes the strange RHMC through QUDA.
  // Fix: write TwoFlavourSchurCloverActionEven (mirrors OneFlavourSchur...Even)
  // and adapt the QUDA force class to use EVEN parity + matpc=EE_ASYMMETRIC.
  if (std::getenv("QUDA_FORCE_LIGHT") != nullptr) {
    QudaCloverParams qp_l;
    qp_l.mass = mass_light;
    qp_l.csw  = csw;
    qp_l.anti_periodic_t = true;
    qp_l.tol = cg_tol;
    qp_l.max_iter = cg_max;
    qp_l.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    LightSchurPF_qudaHolder = std::make_unique<
        TwoFlavourSchurCloverQudaForceActionMP<WilsonImplR, WilsonImplF>>(
        FermOp, FermOpF,
        solver_debug ? (OperatorFunction<LatticeFermion>&)CG_action
                     : (OperatorFunction<LatticeFermion>&)CG_md,
        CG_action,
        qp_l);
    LightSchurPF_qudaHolder->is_smeared = true;
    LightSchurPFptr = LightSchurPF_qudaHolder.get();
    std::cout << GridLogMessage
              << "[Light Nf=2] QUDA_FORCE active — full QUDA force kernel"
              << std::endl;
  } else
#endif
  {
    LightSchurPF_baseHolder = std::make_unique<
        TwoFlavourSchurCloverActionMP<WilsonImplR, WilsonImplF>>(
        FermOp, FermOpF,
        solver_debug ? (OperatorFunction<LatticeFermion>&)CG_action
                     : (OperatorFunction<LatticeFermion>&)CG_md,
        CG_action);
    LightSchurPF_baseHolder->is_smeared = true;
    LightSchurPFptr = LightSchurPF_baseHolder.get();
  }
  Action<LatticeGaugeField> &LightSchurPF = *LightSchurPFptr;

  // ── Hasenbusch ratio rungs (EO-Schur, clover-correct) ─────────────────────
  // n_pf = n_ops - 1 ratios det(Schur(ladder[k]))^2 / det(Schur(ladder[k+1]))^2,
  // NumOp = heavier (ladder[k+1]), DenOp = lighter (ladder[k]).  Well-conditioned
  // by construction (small mass splitting) -> plain DP CG suffices, except any
  // rung listed in HASEN_MG_RUNG (QUDA multigrid, QudaMGSchurSolver.h) or
  // HASEN_QUDA_CG_RUNGS (plain QUDA CG, QudaCGSchurSolver.h -- half-volume
  // direct solve with matpc_type=ODD_ODD_ASYMMETRIC, the parity/form Grid's
  // own SchurDiagMooeeOperator uses; QudaMGSchurSolver must NOT be reused for
  // plain CG -- its inner QudaCloverInverter hardcodes matpc_type=EVEN_EVEN,
  // correct-but-irrelevant for MG's QUDA_DIRECT_SOLVE, wrong for CG's
  // QUDA_NORMOP_PC_SOLVE; see gen_qcd_hasenbusch_tune_compact_schur.cc, where
  // this was found and fixed first). Both env vars accept a comma-separated
  // list of rung indices; HASEN_MG_RUNG takes precedence if a rung is listed
  // in both. n_pf==0 when HASEN_LADDER is unset -> no rungs built, identical
  // to today.
  ConjugateGradient<LatticeFermion> CG_ladder_deriv(cg_md_tol, cg_max);
  int n_pf = n_ops - 1;
  std::vector<std::unique_ptr<Action<LatticeGaugeField>>> RatioPF;
#ifdef GRID_HAVE_QUDA
  std::vector<std::unique_ptr<QudaRungSolverBase>> QudaRungSolver(n_pf);
  // Parallel, INDEPENDENT solver vector for the heatbath (refresh()) step --
  // built at NumOp's mass (ladder[k+1]), not DenOp's (ladder[k]); see
  // TwoFlavourSchurCloverRatioActionQuda.h and
  // gen_qcd_hasenbusch_tune_compact_schur.cc (where this was added first).
  std::vector<std::unique_ptr<QudaCGSchurSolver>> HeatbathQudaSolver(n_pf);
#endif
  auto parse_rung_list = [](const char *env) {
    std::set<int> out;
    if (const char *v = std::getenv(env); v && *v) {
      std::stringstream ss(v);
      std::string tok;
      while (std::getline(ss, tok, ',')) if (!tok.empty()) out.insert(std::atoi(tok.c_str()));
    }
    return out;
  };
  std::set<int> mg_rungs      = parse_rung_list("HASEN_MG_RUNG");
  std::set<int> quda_cg_rungs = parse_rung_list("HASEN_QUDA_CG_RUNGS");
  std::set<int> quda_cg_heatbath_rungs = parse_rung_list("HASEN_QUDA_CG_HEATBATH_RUNGS");
  for (int k = 0; k < n_pf; ++k) {
    bool want_mg = mg_rungs.count(k);
    bool want_quda_cg = !want_mg && quda_cg_rungs.count(k);
    if (want_mg || want_quda_cg) {
#ifdef GRID_HAVE_QUDA
      QudaCloverParams qp_mg;
      qp_mg.mass = ladder[k];  // DenOp mass -- the rung being QUDA-accelerated
      qp_mg.csw  = csw;
      qp_mg.anti_periodic_t = true;
      qp_mg.tol = cg_md_tol;
      qp_mg.max_iter = cg_max;
      qp_mg.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
      qp_mg.use_multigrid = want_mg;
      if (want_mg) {
        // Same HMC_MG_* knobs as the existing USE_HMC_MG single-mass path
        // (TwoFlavourSchurCloverQudaForceActionMP.h) -- reused verbatim.
        const char *nlv = std::getenv("HMC_MG_NLEVEL");
        qp_mg.mg.n_level = nlv ? std::atoi(nlv) : 2;
        auto parse_block = [](const char *s, std::array<int,4> dflt) {
          if (!s || !*s) return dflt;
          std::array<int,4> b = dflt;
          std::sscanf(s, "%d %d %d %d", &b[0], &b[1], &b[2], &b[3]);
          return b;
        };
        std::array<int,4> b0 = parse_block(std::getenv("HMC_MG_BLOCK_L0"), {4,4,4,4});
        std::array<int,4> b1 = parse_block(std::getenv("HMC_MG_BLOCK_L1"), {2,2,2,2});
        qp_mg.mg.geo_block_size = (qp_mg.mg.n_level >= 3)
            ? std::vector<std::array<int,4>>{b0, b1}
            : std::vector<std::array<int,4>>{b0};
        if (const char *nv = std::getenv("HMC_MG_NVEC")) {
          int a=0,b=0,c=0; int n = std::sscanf(nv, "%d %d %d", &a,&b,&c);
          std::vector<int> v; if(n>=1)v.push_back(a); if(n>=2)v.push_back(b); if(n>=3)v.push_back(c);
          if (!v.empty()) { qp_mg.mg.n_vec_levels = v; qp_mg.mg.n_vec = v[0]; }
        }
        if (const char *cm = std::getenv("HMC_MG_COARSE_MAXITER"))
          qp_mg.mg.coarse_solver_maxiter = std::atoi(cm);
        if (const char *r = std::getenv("HMC_MG_REFRESH"))
          qp_mg.mg.setup_maxiter_refresh = std::atoi(r);
        if (const char *r = std::getenv("HMC_MG_REBUILD_EVERY"))
          qp_mg.mg.rebuild_every = std::atoi(r);
        // HMC_MG_REFRESH_EVERY / HMC_MG_THRESHOLD_COUNT / HMC_MG_RSD_TOL_FACTOR
        applyHmcMgCadenceEnv(qp_mg.mg);
        QudaRungSolver[k] = std::make_unique<QudaMGSchurSolver>(LightOps[k]->GaugeGrid(), qp_mg, Odd);
      } else {
        QudaRungSolver[k] = std::make_unique<QudaCGSchurSolver>(LightOps[k]->GaugeGrid(), qp_mg, Odd);
      }
      // Heatbath (refresh()) acceleration is INDEPENDENT of the deriv/action
      // choice above -- separate env var, separate solver instance, built at
      // NumOp's mass (ladder[k+1]).  Only offered on rungs that already have
      // a QUDA deriv/action solver (this loop branch).
      OperatorFunction<LatticeFermion> *heatbath_solver_ptr = &CG_action;
      if (quda_cg_heatbath_rungs.count(k)) {
        QudaCloverParams qp_hb;
        qp_hb.mass = ladder[k+1];  // NumOp mass -- refresh solves M_1^dag M_1
        qp_hb.csw  = csw;
        qp_hb.anti_periodic_t = true;
        qp_hb.tol = cg_action_tol;  // heatbath sets phi for the whole trajectory -- match CG_action's tol
        qp_hb.max_iter = cg_max;
        qp_hb.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
        qp_hb.use_multigrid = false;
        if (std::getenv("QUDA_FORCE_RECON_NO") != nullptr)
          qp_hb.recon_sloppy = QUDA_RECONSTRUCT_NO;
        HeatbathQudaSolver[k] = std::make_unique<QudaCGSchurSolver>(LightOps[k+1]->GaugeGrid(), qp_hb, Odd);
        heatbath_solver_ptr = HeatbathQudaSolver[k].get();
        std::cout << GridLogMessage << "[Ladder] rung " << k
                  << " HeatbathSolver = QUDA CG, mass=" << ladder[k+1] << std::endl;
      }
      RatioPF.emplace_back(std::make_unique<TwoFlavourSchurCloverRatioActionQuda<WilsonImplR>>(
          *LightOps[k+1], *LightOps[k], *QudaRungSolver[k], *heatbath_solver_ptr));
      std::cout << GridLogMessage << "[Ladder] rung " << k
                << " DerivativeSolver/ActionSolver = "
                << (want_mg ? ("QUDA MG (" + std::to_string(qp_mg.mg.n_level) + " levels)") : std::string("QUDA CG"))
                << ", mass=" << ladder[k] << std::endl;
#else
      std::cerr << "HASEN_MG_RUNG/HASEN_QUDA_CG_RUNGS require a QUDA build.\n"; exit(1);
#endif
    } else {
      RatioPF.emplace_back(std::make_unique<TwoFlavourSchurCloverRatioAction<WilsonImplR>>(
          *LightOps[k+1], *LightOps[k], CG_ladder_deriv, CG_action));
    }
    RatioPF.back()->is_smeared = true;
  }
  std::cout << GridLogMessage << "Built " << n_pf << " Schur ratio PF levels." << std::endl;

  // Strange quark (Nf=1): EO-preconditioned LogDet + Schur RHMC.
  // MD force uses mixed-precision multishift CG (SP inner + DP reliable).
  WCF StrangeFermOp(Umu, Grid, RBGrid, mass_strange, csw, csw,
                    WilsonAnisotropyCoefficients(), impl_p);
  WCF_f StrangeFermOpF(UmuF, GridF, RBGridF, mass_strange, csw, csw,
                       WilsonAnisotropyCoefficients(), impl_pF);
  // Chroma-matched bounds for the rat_3strange monomial on this ensemble:
  // <lowerMin>0.0001</lowerMin> <upperMax>32</upperMax>, force <degree>13</degree>.
  // Was (1e-4, 200, 10) — hi=200 wasted Remez fit on a region the spectrum
  // doesn't reach (real top is ~24-30); degree=10 was less accurate than
  // chroma's 13.  Force eval cost grows ~30% from extra poles; trade is
  // fewer cleanup steps + tighter bound on |dH|.
  // Env-overridable (RAT_LO/RAT_HI/RAT_DEGREE), mirroring the compact driver:
  // for spectra where lo=1e-4 sits below lambda_min the Remez fit plants phantom
  // poles below the spectrum that QUDA's multishift cannot invert; e.g. the
  // 48^3 b6.3 thermalized config (lambda_min~0.43) needs RAT_LO~0.4 RAT_HI~35.
  const RealD rat_lo = TXQCDProduction::detail::env_real("RAT_LO", 1e-4);
  const RealD rat_hi = TXQCDProduction::detail::env_real("RAT_HI", 100.0);
  int rat_degree = 20;
  if (const char *rd = std::getenv("RAT_DEGREE"); rd && *rd) rat_degree = std::atoi(rd);
  std::cout << GridLogMessage << "Strange rational: lo=" << rat_lo << " hi=" << rat_hi
            << " degree=" << rat_degree << std::endl;
  OneFlavourRationalParams strange_rat(rat_lo, rat_hi, cg_max, cg_tol, rat_degree, 64,
                                       100, 1e-6, 1e-4);
  QCDLogDetCloverEOAction<WilsonImplR> StrangeLogDet(StrangeFermOp, 1);
  StrangeLogDet.is_smeared = true;
  // QUDA_FORCE=1 (Phase D) routes the strange Nf=1 RHMC deriv through
  // computeCloverForceQuda (PyQUDA dagger=YES convention, cos=1.0 vs PathA).
  std::unique_ptr<OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>>
      StrangeSchurPF_baseHolder;
  // Same-parity validation holder: Grid-native EVEN-parity Schur monomial,
  // opt-in via STRANGE_EVEN=1 (default off -> stock ODD-parity behavior).
  std::unique_ptr<OneFlavourSchurCloverRationalActionEven<WilsonImplR>>
      StrangeSchurPF_evenHolder;
  Action<LatticeGaugeField> *StrangeSchurPFptr = nullptr;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<OneFlavourSchurCloverQudaForceRationalActionMP<WilsonImplR, WilsonImplF>>
      StrangeSchurPF_qudaHolder;
  if (std::getenv("QUDA_FORCE") != nullptr) {
    QudaCloverParams qp;
    qp.mass = mass_strange;
    qp.csw  = csw;
    qp.anti_periodic_t = true;
    qp.tol = cg_tol;
    qp.max_iter = cg_max;
    qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    // QUDA_FORCE_RECON_NO=1 → reconstruct_sloppy = NO (full links). REQUIRED at
    // 48^3: the antiperiodic-t phase is baked in by negating the last-timeslice
    // time links (det=-1, not SU(3)); RECONSTRUCT_12 (the default) mis-rebuilds
    // the 3rd row of those links (it assumes det=+1), corrupting the sloppy
    // solve. Ported from gen_qcd_hasenbusch_tune_compact_schur.cc (memory
    // quda-strange-recon-fix) -- this driver's strange QUDA path never had it.
    if (std::getenv("QUDA_FORCE_RECON_NO") != nullptr)
      qp.recon_sloppy = QUDA_RECONSTRUCT_NO;
    StrangeSchurPF_qudaHolder = std::make_unique<
        OneFlavourSchurCloverQudaForceRationalActionMP<WilsonImplR, WilsonImplF>>(
        StrangeFermOp, StrangeFermOpF, &RBGridF, strange_rat, qp, 50);
    StrangeSchurPF_qudaHolder->is_smeared = true;
    StrangeSchurPFptr = StrangeSchurPF_qudaHolder.get();
    std::cout << GridLogMessage
              << "[Strange Nf=1] QUDA_FORCE active — full QUDA force kernel"
              << std::endl;
  } else
#endif
  {
    // STRANGE_EVEN=1 -> Grid's native EVEN-parity Schur monomial (same parity as
    // the QUDA force path) so a pure-grid vs grid-quda force comparison is
    // same-pseudofermion (single-precision floor) instead of cross-parity
    // (Odd-vs-Even -> different phi -> the ~0.175% structural offset).
    // Default (unset) -> stock ODD-parity mixed-precision monomial.
    if (std::getenv("STRANGE_EVEN") != nullptr) {
      StrangeSchurPF_evenHolder = std::make_unique<
          OneFlavourSchurCloverRationalActionEven<WilsonImplR>>(
          StrangeFermOp, strange_rat);
      StrangeSchurPF_evenHolder->is_smeared = true;
      StrangeSchurPFptr = StrangeSchurPF_evenHolder.get();
      std::cout << GridLogMessage
                << "[Strange Nf=1] STRANGE_EVEN active — Grid native EVEN-parity Schur"
                << std::endl;
    } else {
      StrangeSchurPF_baseHolder = std::make_unique<
          OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>>(
          StrangeFermOp, StrangeFermOpF, &RBGridF, strange_rat, 50);
      StrangeSchurPF_baseHolder->is_smeared = true;
      StrangeSchurPFptr = StrangeSchurPF_baseHolder.get();
    }
  }
  Action<LatticeGaugeField> &StrangeSchurPF = *StrangeSchurPFptr;

  // Grid's SymanzikGaugeAction(β,u0) uses the RBC/Iwasaki convention
  // (c_plaq = β·(1−8c1) ≈ 1.96β, c_rect = −β/(12·u0²)) — NOT the Lüscher-
  // Weisz tree-level action chroma's LW_TREE_GAUGEACT uses.  Chroma's source
  // literally sets  c0 = β  (the 5/3 is absorbed into β_XML) and
  //                 c1 = −c0/(20·u0²) = −β/(20·u0²).
  // So to match the chroma reference ensemble cl3_16_48_b6p1 at XML β=6.1
  // we construct PlaqPlusRectangleAction directly with chroma's coefficients.
  typedef PlaqPlusRectangleAction<PeriodicGimplR> PlaqRectR;
  PlaqRectR GaugeAction(beta, -beta / (20.0 * u0 * u0));
  // Chroma's LW_TREE_GAUGEACT operates on the THIN (unsmeared) gauge links;
  // stout only wraps the fermion via STOUT_FERM_STATE.  Setting
  // is_smeared=false here makes the gauge action see thin U directly —
  // matching chroma.  is_smeared=true was the cause of the Δplaq=0.12
  // equilibrium shift vs chroma's 0.5135.
  GaugeAction.is_smeared = false;

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  // Outer (fermion) level: 7 steps/trajectory.  Inner (gauge) level: 4 gauge
  // sub-steps per fermion step -> 28 gauge force evaluations per trajectory.
  // Mirrors chroma's Min-Norm 7 / 7x4 convention on this ensemble.
  // GAUGE_INNER_MULT env var lets us probe whether multi-rate is the source
  // of FG instability at MDs=7 — set to 1 to put gauge force at outer level.
  int gauge_inner_mult = 4;
  if (const char *m = std::getenv("GAUGE_INNER_MULT"); m && *m) gauge_inner_mult = std::atoi(m);
  std::cout << GridLogMessage << "GAUGE_INNER_MULT=" << gauge_inner_mult << std::endl;
  ActionLevel<LatticeGaugeField, Reps> L1(1);
  ActionLevel<LatticeGaugeField, Reps> L2(std::max(1, gauge_inner_mult));
  L1.push_back(&LightLogDet);
  for (auto &pf : RatioPF) L1.push_back(pf.get());
  L1.push_back(&LightSchurPF);
  L1.push_back(&StrangeLogDet);
  L1.push_back(&StrangeSchurPF);
  ActionSet<LatticeGaugeField, Reps> Aset;
  if (gauge_inner_mult <= 1) {
    L1.push_back(&GaugeAction);
    Aset.push_back(L1);
  } else {
    L2.push_back(&GaugeAction);
    Aset.push_back(L1);
    Aset.push_back(L2);
  }

  // INTEGRATOR env var: "MinimumNorm2" (default) or "ForceGradient".  Added
  // to allow direct comparison against TXQCD runs using the same integrator.
  std::string integrator_name = "MinimumNorm2";
  if (const char *env = std::getenv("INTEGRATOR"); env && *env)
    integrator_name = env;

  IntegratorParameters MD;
  MD.name = integrator_name;
  MD.MDsteps = 7;
  if (const char *ms = std::getenv("MDSTEPS"); ms && *ms) MD.MDsteps = std::atoi(ms);
  MD.trajL = sqrt(2.0);
  if (const char *tl = std::getenv("TRAJL"); tl && *tl) MD.trajL = std::atof(tl);
  std::cout << GridLogMessage << "INTEGRATOR=" << integrator_name
            << "  MDsteps=" << MD.MDsteps
            << "  trajL=" << MD.trajL << std::endl;

  // NoMetropolisUntil: default 10 - start_traj ≥ 0; override via NO_METROP env
  // var (0 = chroma-style Metropolis-from-trajectory-0).  With thermal aux /
  // weak-field gauge the initial configuration is close enough to equilibrium
  // that skipping Metropolis for the first few trajs risks settling into a
  // pure-MD equilibrium that differs from the Metropolis one.
  int no_metrop = std::max(0, 10 - start_traj);
  if (const char *nm = std::getenv("NO_METROP"); nm && *nm) no_metrop = std::atoi(nm);
  std::cout << GridLogMessage << "NoMetropolisUntil=" << no_metrop << std::endl;
  HMCparameters HMCp;
  HMCp.StartTrajectory     = start_traj;
  HMCp.Trajectories        = total_traj - no_metrop - start_traj;
  HMCp.NoMetropolisUntil   = no_metrop;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> Smear(&Grid, stout_nsmear_inv, Stout);
  Smear.set_Field(Umu);

  // ── Forces-only mode (force compute, no HMC) ──────────────────────────────
  // FORCES_ONLY=1: skip the integrator entirely.  Heatbath each action and
  // evaluate its MD force once on the (smeared) imported config -- exactly as
  // the integrator's first P-update does -- then print the per-term avg/max
  // force norm (the Hasenbusch tuning signal) and the deriv() wall time,
  // averaged over FORCES_SAMPLES draws.  Mirrors the FORCES_ONLY block in
  // gen_qcd_hasenbusch_tune_compact.cc so the two drivers' forces are directly
  // comparable (sizes for mass tuning; times for the 48^3 trajectory estimate).
  // FORCES_SKIP_STRANGE=1 omits the expensive strange RHMC force;
  // FORCES_SKIP_GAUGE=1 omits the gauge force.
  if (std::getenv("FORCES_ONLY") != nullptr) {
    int nsamp = 1;
    if (const char *ns = std::getenv("FORCES_SAMPLES"); ns && *ns) nsamp = std::atoi(ns);
    const bool skip_strange = std::getenv("FORCES_SKIP_STRANGE") != nullptr;
    const bool skip_gauge   = std::getenv("FORCES_SKIP_GAUGE")   != nullptr;
    const bool skip_light   = std::getenv("FORCES_SKIP_LIGHT")   != nullptr;  // LogDet + Schur PF
    struct Ref { std::string name; Action<LatticeGaugeField> *act; };
    std::vector<Ref> refs = {{"LightLogDet", &LightLogDet}};
    for (int k = 0; k < n_pf; ++k)
      refs.push_back({"PF" + std::to_string(k), RatioPF[k].get()});
    refs.push_back({"LightSchurPF",   &LightSchurPF});
    refs.push_back({"StrangeLogDet",  &StrangeLogDet});
    refs.push_back({"StrangeSchurPF", &StrangeSchurPF});
    refs.push_back({"Gauge",          &GaugeAction});
    auto skip = [&](const std::string &nm) {
      return (skip_strange && (nm == "StrangeSchurPF" || nm == "StrangeLogDet")) ||
             (skip_gauge && nm == "Gauge") ||
             (skip_light && (nm == "LightLogDet" || nm == "LightSchurPF" || nm.rfind("PF", 0) == 0));
    };
    LatticeGaugeField force(&Grid);
    const int nref = (int)refs.size();
    std::vector<double> acc_avg(nref, 0.0), acc_max(nref, 0.0), acc_time(nref, 0.0);
    Smear.set_Field(Umu);   // smear the fixed config once (deterministic)
    for (int s = 0; s < nsamp; ++s)
      for (int i = 0; i < nref; ++i) {
        if (skip(refs[i].name)) continue;
        Action<LatticeGaugeField> *act = refs[i].act;
        double ftime;
        if (act->is_smeared) {
          // Smeared fermion / log-det terms: heatbath, then evaluate the force
          // through the smearing chain rule -- exactly the integrator's first
          // P-update.  (The timer excludes the heatbath, as in HMC.)
          act->refresh(Smear, sRNG, pRNG);
          auto t0 = std::chrono::steady_clock::now();
          act->deriv(Smear, force);
          ftime = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } else {
          // Unsmeared gauge action: evaluate on the raw links Umu directly --
          // the SmearedConfiguration wrapper path reproducibly hangs on the
          // standalone gauge eval after the preceding smeared-fermion evals.
          auto t0 = std::chrono::steady_clock::now();
          act->deriv(Umu, force);
          ftime = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        force = PeriodicGimplR::projectForce(force);   // same projection the integrator uses
        double favg = std::sqrt(norm2(force) / force.Grid()->gSites());
        double fmax = std::sqrt(maxLocalNorm2(force));
        acc_avg[i]  += favg;
        acc_max[i]  += fmax;
        acc_time[i] += ftime;
        // Stream each level immediately so a later-level stall can't swallow
        // the numbers already computed.
        std::cout << GridLogMessage << "FORCES_ONLY level " << refs[i].name
                  << " sample " << s << " avg=" << favg << " max=" << fmax
                  << " time=" << ftime << std::endl;
      }
    auto emit = [&](const char *tag, std::vector<double> &acc) {
      std::cout << GridLogMessage << "FORCES_ONLY samples=" << nsamp << " " << tag << ":";
      for (int i = 0; i < nref; ++i)
        if (!skip(refs[i].name))
          std::cout << " " << refs[i].name << "=" << acc[i] / nsamp;
      std::cout << std::endl;
    };
    emit("avg", acc_avg);
    emit("max", acc_max);
    emit("time", acc_time);
    // NOTE: QUDA runs abort here at Grid_finalize() with a double-free of the
    // QUDA color-spinor buffer (action destructor + QUDA finalize both free it).
    // All forces are already printed, so data is complete. A proper fix belongs
    // in the Grid-TXQCD QUDA action/wrapper memory ownership, not here.
    Grid_finalize();
    return 0;
  }

  struct Ckpt : public HmcObservable<LatticeGaugeField> {
    std::string cfg_prefix, rng_prefix;
    int interval;
    void TrajectoryComplete(int t, LatticeGaugeField &U, GridSerialRNG &sR,
                            GridParallelRNG &pR) override {
      if (t % interval != 0) return;
      typedef GaugeStatistics<PeriodicGimplR> GS;
      NerscIO::writeRNGState(sR, pR, rng_prefix + "." + std::to_string(t));
      NerscIO::writeConfiguration<GS>(U, cfg_prefix + "." + std::to_string(t), 0, 1);
    }
  };
  Ckpt ckpt;
  ckpt.cfg_prefix = cfg_dir + "/ckpoint_lat";
  ckpt.rng_prefix = cfg_dir + "/ckpoint_rng";
  ckpt.interval   = meas_skip;

  std::vector<QcdDiag::ActionRef> diagActions = {{"LightLogDet", &LightLogDet}};
  for (int k = 0; k < n_pf; ++k)
    diagActions.push_back({"PF" + std::to_string(k), RatioPF[k].get()});
  diagActions.push_back({"LightSchurPF", &LightSchurPF});
  diagActions.push_back({"StrangeLogDet", &StrangeLogDet});
  diagActions.push_back({"StrangeSchurPF", &StrangeSchurPF});
  diagActions.push_back({"Gauge", &GaugeAction});
  QcdDiag diag(cfg_dir + "/hmc_diagnostics", meas_skip, diagActions,
               Smear, Grid, RBGrid, pRNG);

  // Pure-HMC timing build: QcdDiag does per-trajectory stochastic VEV solves
  // (+ optional eig + HDF5 dump) — non-MD work that pollutes wall/traj.
  // Attach only on RUN_DIAG=1.
  // Per-trajectory gauge observables (plaquette + Polyakov loop) — Grid's stock
  // loggers. Read-only gauge reductions: draw no RNG, mutate nothing, ~ms cost.
  // They fire after the trajectory's force/H/dH/Metropolis are finalized, so the
  // HMC sequence is bit-identical with or without them.
  PlaquetteLogger<PeriodicGimplR> plaqLog;
  PolyakovLogger<PeriodicGimplR>  polyLog;
  std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &plaqLog, &polyLog};
  if (std::getenv("RUN_DIAG") != nullptr) Obs.push_back(&diag);

  // TEST_SIGMA_LOOP: σ-loop replay test.  Runs Wilson::MeeDeriv (the trusted
  // reference) and our QCDLogDetCloverEOAction-style σ-loop with the SAME Λ =
  // outerProduct(Xf, Yf) on the smeared gauge field, comparing the resulting
  // gauge-force tensors element-by-element.  Both σ-loops should be bit-
  // identical (they are literally the same code, modulo the Nf prefactor and
  // a harmless mu==4 vs mu==3 typo that's invisible at csw_t==csw_r).  If
  // they don't match, the bug is in the σ-loop itself.  If they do match,
  // the bug must be in either (a) the Λ construction in deriv()
  // [setCheckerboard(Lambda, CloverTermInvEven)] or (b) the smearing chain-
  // rule integration in SmearedConfiguration::smeared_force.
  if (const char *t = std::getenv("TEST_SIGMA_LOOP"); t && std::atoi(t)) {
    Smear.set_Field(Umu);
    LatticeGaugeField Usm = Smear.get_SmearedU();
    FermOp.ImportGauge(Usm);

    // Random Gaussian fermions η, ξ on full grid, then take even-parity halves.
    LatticeFermion eta_full(&Grid), xi_full(&Grid);
    gaussian(pRNG, eta_full);
    gaussian(pRNG, xi_full);
    LatticeFermion X_e(&RBGrid), Y_e(&RBGrid);
    pickCheckerboard(Even, X_e, eta_full);
    pickCheckerboard(Even, Y_e, xi_full);

    // (1) Wilson::MeeDeriv (reference): builds Λ = outerProduct(Xf, Yf) with
    // Xf, Yf zero-padded onto full grid and runs its σ-loop.
    LatticeGaugeField force_W(&Grid);
    FermOp.MeeDeriv(force_W, X_e, Y_e, DaggerNo);

    // (2) Our σ-loop verbatim from QCDLogDetCloverEOAction::deriv, but driven
    // with the same Λ Wilson built (rather than setCheckerboard(_, CloverTermInvEven)).
    typedef WilsonImplR Impl;
    typedef CloverHelpers<Impl> CH;
    typedef typename Impl::PropagatorField PropagatorField;
    typedef typename Impl::GaugeLinkField  GaugeLinkField;

    LatticeFermion Xf(&Grid), Yf(&Grid);
    Xf = Zero(); Yf = Zero();
    setCheckerboard(Xf, X_e);
    setCheckerboard(Yf, Y_e);
    PropagatorField Lambda(&Grid);
    FermOp.outerProductImpl(Lambda, Xf, Yf);

    std::vector<GaugeLinkField> Ulinks(Nd, &Grid);
    FermOp.extractLinkField(Ulinks, FermOp.Umu);

    Gamma::Algebra sigma[] = {
        Gamma::Algebra::SigmaXY,      Gamma::Algebra::SigmaXZ,
        Gamma::Algebra::SigmaXT,      Gamma::Algebra::MinusSigmaXY,
        Gamma::Algebra::SigmaYZ,      Gamma::Algebra::SigmaYT,
        Gamma::Algebra::MinusSigmaXZ, Gamma::Algebra::MinusSigmaYZ,
        Gamma::Algebra::SigmaZT,      Gamma::Algebra::MinusSigmaXT,
        Gamma::Algebra::MinusSigmaYT, Gamma::Algebra::MinusSigmaZT};

    GaugeLinkField force_mu(&Grid), lambda(&Grid);
    LatticeGaugeField force_Q(&Grid);
    force_Q = Zero();
    int count = 0;
    for (int mu = 0; mu < 4; mu++) {
      force_mu = Zero();
      for (int nu = 0; nu < 4; nu++) {
        if (mu == nu) continue;
        // QCDLogDetCloverEOAction's convention: nu==3||mu==3 -> csw_t.
        RealD factor = (nu == 3 || mu == 3) ? 2.0 * FermOp.csw_t
                                             : 2.0 * FermOp.csw_r;
        PropagatorField Slambda = Gamma(sigma[count]) * Lambda;
        FermOp.TraceSpinImpl(lambda, Slambda);
        force_mu -= factor * CH::Cmunu(Ulinks, lambda, mu, nu);
        count++;
      }
      pokeLorentz(force_Q, Ulinks[mu] * force_mu, mu);
    }

    // Compare.
    LatticeGaugeField diff_F(&Grid);
    diff_F = force_W - force_Q;
    RealD nW = norm2(force_W);
    RealD nQ = norm2(force_Q);
    RealD nD = norm2(diff_F);
    std::cout << GridLogMessage << "[SIGMA] |force_W|^2 = " << std::setprecision(15) << nW << std::endl;
    std::cout << GridLogMessage << "[SIGMA] |force_Q|^2 = " << std::setprecision(15) << nQ << std::endl;
    std::cout << GridLogMessage << "[SIGMA] |force_W - force_Q|^2 = " << std::setprecision(15) << nD << std::endl;
    std::cout << GridLogMessage << "[SIGMA] relative diff = "
              << std::sqrt(nD / nW) << std::endl;

    // Per-direction breakdown — helps localise if only one mu is off.
    for (int mu = 0; mu < Nd; ++mu) {
      auto FW_mu = PeekIndex<LorentzIndex>(force_W, mu);
      auto FQ_mu = PeekIndex<LorentzIndex>(force_Q, mu);
      auto Dmu = FW_mu - FQ_mu;
      std::cout << GridLogMessage << "[SIGMA] mu=" << mu
                << " |W|^2=" << norm2(FW_mu)
                << " |Q|^2=" << norm2(FQ_mu)
                << " |W-Q|^2=" << norm2(Dmu) << std::endl;
    }

    Grid_finalize();
    return 0;
  }

  // TEST_LAMBDA: Hutchinson Λ-construction test.  Compares
  //   Λ_det   = setCheckerboard(0, CloverTermInvEven)         [QCDLogDet's Λ]
  // vs
  //   <Λ_stoch> = E_η[outerProduct(η_full, M_ee^{-1} η_full)] [Wilson-style]
  // sample-by-sample averaged.  If Mee is Hermitian, <Λ_stoch>_{αβ,ab} → Mee^{-1}_{αβ,ab}
  // = CloverTermInvEven_{αβ,ab} on even sites.  Disagreement (beyond 1/√N noise)
  // means a transpose/dagger convention mismatch between how
  // setCheckerboard(_, CloverTermInvEven) lays out indices and how
  // outerProduct(_, _) does — which would explain why our QCDLogDet σ-loop
  // produces a wrong force despite the σ-loop machinery being bit-identical.
  // Number of noise samples set by N_LAMBDA_SAMPLES (default 32).
  if (const char *t = std::getenv("TEST_LAMBDA"); t && std::atoi(t)) {
    Smear.set_Field(Umu);
    LatticeGaugeField Usm = Smear.get_SmearedU();
    FermOp.ImportGauge(Usm);

    int N_samples = 32;
    if (const char *n = std::getenv("N_LAMBDA_SAMPLES"); n && *n) N_samples = std::atoi(n);
    std::cout << GridLogMessage << "[LAMBDA] N_samples=" << N_samples << std::endl;

    typedef WilsonImplR Impl;
    typedef CloverHelpers<Impl> CH;
    typedef typename Impl::PropagatorField PropagatorField;
    typedef typename Impl::GaugeLinkField  GaugeLinkField;

    // Λ_det: deterministic, the actual one QCDLogDet::deriv uses.
    PropagatorField Lambda_det(&Grid);
    Lambda_det = Zero();
    setCheckerboard(Lambda_det, FermOp.CloverTermInvEven);

    // Λ_stoch_avg: noise-averaged outerProduct(η, Mee^{-1}η), even support.
    PropagatorField Lambda_stoch_avg(&Grid);
    Lambda_stoch_avg = Zero();
    for (int n = 0; n < N_samples; n++) {
      LatticeFermion eta_full(&Grid), z_full(&Grid);
      LatticeFermion eta_e(&RBGrid), z_e(&RBGrid);
      gaussian(pRNG, eta_full);
      pickCheckerboard(Even, eta_e, eta_full);
      eta_full = Zero();
      setCheckerboard(eta_full, eta_e);
      // z_e = Mee^{-1} eta_e (on-site clover inverse on even checkerboard).
      FermOp.MooeeInv(eta_e, z_e);
      z_full = Zero();
      setCheckerboard(z_full, z_e);

      PropagatorField stoch(&Grid);
      FermOp.outerProductImpl(stoch, eta_full, z_full);
      Lambda_stoch_avg = Lambda_stoch_avg + stoch;
    }
    Lambda_stoch_avg = Lambda_stoch_avg * (1.0 / N_samples);

    PropagatorField Lambda_diff(&Grid);
    Lambda_diff = Lambda_det - Lambda_stoch_avg;
    RealD nDet  = norm2(Lambda_det);
    RealD nSto  = norm2(Lambda_stoch_avg);
    RealD nDiff = norm2(Lambda_diff);
    std::cout << GridLogMessage << "[LAMBDA] |Λ_det|^2 = " << std::setprecision(15) << nDet << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] |<Λ_stoch>|^2 = " << std::setprecision(15) << nSto << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] |Λ_det − <Λ_stoch>|^2 = " << std::setprecision(15) << nDiff << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] rel diff = " << std::sqrt(nDiff / nDet) << std::endl;

    // Also try Λ_stoch_T = outerProduct(z_full, eta_full) — i.e. swap the
    // roles, since outerProduct convention could matter.  <outerProduct(z, η)>
    // = (Mee^{-1})^T_{αβ,ab} on even sites.
    PropagatorField Lambda_stoch_T_avg(&Grid);
    Lambda_stoch_T_avg = Zero();
    for (int n = 0; n < N_samples; n++) {
      LatticeFermion eta_full(&Grid), z_full(&Grid);
      LatticeFermion eta_e(&RBGrid), z_e(&RBGrid);
      gaussian(pRNG, eta_full);
      pickCheckerboard(Even, eta_e, eta_full);
      eta_full = Zero();
      setCheckerboard(eta_full, eta_e);
      FermOp.MooeeInv(eta_e, z_e);
      z_full = Zero();
      setCheckerboard(z_full, z_e);

      PropagatorField stoch(&Grid);
      FermOp.outerProductImpl(stoch, z_full, eta_full);
      Lambda_stoch_T_avg = Lambda_stoch_T_avg + stoch;
    }
    Lambda_stoch_T_avg = Lambda_stoch_T_avg * (1.0 / N_samples);

    PropagatorField Lambda_diff_T(&Grid);
    Lambda_diff_T = Lambda_det - Lambda_stoch_T_avg;
    std::cout << GridLogMessage << "[LAMBDA] (swapped order) |<Λ_stoch_T>|^2 = "
              << std::setprecision(15) << norm2(Lambda_stoch_T_avg) << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] (swapped order) |Λ_det − <Λ_stoch_T>|^2 = "
              << std::setprecision(15) << norm2(Lambda_diff_T) << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] (swapped order) rel diff = "
              << std::sqrt(norm2(Lambda_diff_T) / nDet) << std::endl;

    // Now apply the σ-loop to all three Λs and compare resulting forces.
    auto sigma_loop = [&](const PropagatorField& Lam, LatticeGaugeField& force) {
      Gamma::Algebra sigma_arr[] = {
          Gamma::Algebra::SigmaXY,      Gamma::Algebra::SigmaXZ,
          Gamma::Algebra::SigmaXT,      Gamma::Algebra::MinusSigmaXY,
          Gamma::Algebra::SigmaYZ,      Gamma::Algebra::SigmaYT,
          Gamma::Algebra::MinusSigmaXZ, Gamma::Algebra::MinusSigmaYZ,
          Gamma::Algebra::SigmaZT,      Gamma::Algebra::MinusSigmaXT,
          Gamma::Algebra::MinusSigmaYT, Gamma::Algebra::MinusSigmaZT};
      std::vector<GaugeLinkField> Ulinks(Nd, &Grid);
      FermOp.extractLinkField(Ulinks, FermOp.Umu);
      GaugeLinkField fmu(&Grid), lam(&Grid);
      force = Zero();
      int cc = 0;
      for (int mu = 0; mu < 4; mu++) {
        fmu = Zero();
        for (int nu = 0; nu < 4; nu++) {
          if (mu == nu) continue;
          RealD factor = (nu == 3 || mu == 3) ? 2.0 * FermOp.csw_t
                                               : 2.0 * FermOp.csw_r;
          PropagatorField Slam = Gamma(sigma_arr[cc]) * Lam;
          FermOp.TraceSpinImpl(lam, Slam);
          fmu -= factor * CH::Cmunu(Ulinks, lam, mu, nu);
          cc++;
        }
        pokeLorentz(force, Ulinks[mu] * fmu, mu);
      }
    };

    LatticeGaugeField force_det(&Grid), force_sto(&Grid), force_stoT(&Grid);
    sigma_loop(Lambda_det, force_det);
    sigma_loop(Lambda_stoch_avg, force_sto);
    sigma_loop(Lambda_stoch_T_avg, force_stoT);

    LatticeGaugeField fdiff = force_det - force_sto;
    LatticeGaugeField fdiffT = force_det - force_stoT;
    std::cout << GridLogMessage << "[LAMBDA] σ-loop |force_det|^2 = "
              << std::setprecision(15) << norm2(force_det) << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] σ-loop |force_sto|^2 = "
              << std::setprecision(15) << norm2(force_sto) << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] σ-loop |force_det − force_sto|^2 = "
              << std::setprecision(15) << norm2(fdiff) << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] σ-loop rel diff (det vs sto) = "
              << std::sqrt(norm2(fdiff) / norm2(force_det)) << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] σ-loop |force_det − force_stoT|^2 = "
              << std::setprecision(15) << norm2(fdiffT) << std::endl;
    std::cout << GridLogMessage << "[LAMBDA] σ-loop rel diff (det vs stoT) = "
              << std::sqrt(norm2(fdiffT) / norm2(force_det)) << std::endl;

    Grid_finalize();
    return 0;
  }

  // TEST_FORCE_FD: skip HMC, run force-FD consistency check on each action
  // through the full Smearer wiring (so we test the smeared_force chain rule
  // for is_smeared=true actions).  Compares dS_actual = S(U_+ε) − S(U_−ε)
  // against analytic dS_pred = −2ε Σ Re tr(p · U·∂S/∂U).  Ratio → 1 means
  // deriv() is consistent with S() at this U.  Diverging ratio at small ε →
  // analytic-vs-numeric force inconsistency.
  if (const char *t = std::getenv("TEST_FORCE_FD"); t && std::atoi(t)) {
    // FD_NONEO=1: also include non-EO TwoFlavourPseudoFermionAction for the
    // light quark, computing S = Phi†(M†M)⁻¹Phi on the FULL Wilson-Clover
    // operator (no EO splitting → no separate LogDet term → no
    // QCDLogDetCloverEOAction smearing-chain-rule path).  Diagnostic only:
    // if this passes FD with smearing while the EO version fails, the bug
    // is localized to the EO LogDet's chain rule integration.
    std::unique_ptr<TwoFlavourPseudoFermionAction<WilsonImplR>> LightTwoFlNonEO;
    bool fd_noneo = false;
    if (const char *fn = std::getenv("FD_NONEO"); fn && std::atoi(fn)) {
      fd_noneo = true;
      LightTwoFlNonEO.reset(new TwoFlavourPseudoFermionAction<WilsonImplR>(
          FermOp, CG_action, CG_action));
      LightTwoFlNonEO->is_smeared = true;
      std::cout << GridLogMessage
                << "[FD] FD_NONEO=1 — adding non-EO TwoFlavourPseudoFermion "
                   "for full M operator FD test"
                << std::endl;
    }

    std::vector<std::pair<std::string, Action<LatticeGaugeField>*>> actions = {
        {"PlaqRect",      &GaugeAction},
        {"LightLogDet",   &LightLogDet}};
    for (int k = 0; k < n_pf; ++k)
      actions.push_back({"PF" + std::to_string(k), RatioPF[k].get()});
    actions.push_back({"LightSchurPF",  &LightSchurPF});
    actions.push_back({"StrangeLogDet", &StrangeLogDet});
    actions.push_back({"StrangeSchurPF", &StrangeSchurPF});
    if (fd_noneo) actions.push_back({"LightTwoFlNonEO", LightTwoFlNonEO.get()});

    // FD_NO_SMEAR=1 disables stout smearing on all fermion actions for the
    // test — isolates whether bug is in plain Schur+clover deriv() or in the
    // SmearedConfiguration chain-rule plumbing.
    if (const char *fns = std::getenv("FD_NO_SMEAR"); fns && std::atoi(fns)) {
      LightLogDet.is_smeared = false;
      LightSchurPF.is_smeared = false;
      StrangeLogDet.is_smeared = false;
      StrangeSchurPF.is_smeared = false;
      for (auto &pf : RatioPF) pf->is_smeared = false;
      std::cout << GridLogMessage << "[FD] FD_NO_SMEAR=1 — fermion is_smeared=false" << std::endl;
    }

    // Refresh pseudofermions against the loaded U (smeared, where applicable).
    Smear.set_Field(Umu);
    for (auto &p : actions) p.second->refresh(Smear, sRNG, pRNG);

    LatticeGaugeField mom(&Grid);
    {
      LatticeColourMatrix mommu(&Grid);
      for (int mu = 0; mu < Nd; ++mu) {
        SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, mommu);
        PokeIndex<LorentzIndex>(mom, mommu, mu);
      }
    }

    LatticeGaugeField Umu_save(&Grid); Umu_save = Umu;
    LatticeGaugeField UdSdU(&Grid);
    for (auto &[name, act] : actions) {
      // Restore U + smear chain to the reference cfg.
      Umu = Umu_save;
      Smear.set_Field(Umu);
      // Compute analytic derivative through Smearer (respects is_smeared +
      // applies smeared_force chain rule when needed).
      act->deriv(Smear, UdSdU);
      // Apply the same projection the integrator does after deriv().  Only the
      // Lie-algebra (Ta) part of dSdU is physically meaningful — anything else
      // is dropped before the momentum update.
      UdSdU = PeriodicGimplR::projectForce(UdSdU);
      ComplexD dSpred(0.0, 0.0);
      for (int mu = 0; mu < Nd; ++mu) {
        auto UdSdUmu = PeekIndex<LorentzIndex>(UdSdU, mu);
        auto pmu     = PeekIndex<LorentzIndex>(mom, mu);
        LatticeComplex dS_mu = -2.0 * trace(pmu * UdSdUmu);
        dSpred += TensorRemove(sum(dS_mu));
      }
      RealD S0 = act->S(Smear);
      std::cout << GridLogMessage << "[FD][" << name << "] S(U)=" << std::setprecision(15) << S0
                << "  dS_pred=" << dSpred << std::endl;
      for (RealD eps : {1e-2, 1e-3, 1e-4, 1e-5}) {
        // U_± = (1 ± ε p) U with the smear chain refreshed via set_Field.
        LatticeGaugeField Up(&Grid), Um(&Grid);
        {
          autoView(Up_v, Up, CpuWrite);
          autoView(Um_v, Um, CpuWrite);
          autoView(U_v,  Umu_save, CpuRead);
          autoView(p_v,  mom, CpuRead);
          thread_foreach(i, p_v, {
            for (int mu = 0; mu < Nd; ++mu) {
              Up_v[i](mu) = U_v[i](mu) + p_v[i](mu) * U_v[i](mu) * eps;
              Um_v[i](mu) = U_v[i](mu) - p_v[i](mu) * U_v[i](mu) * eps;
            }
          });
        }
        Umu = Up; Smear.set_Field(Umu);
        RealD Sp = act->S(Smear);
        Umu = Um; Smear.set_Field(Umu);
        RealD Sm = act->S(Smear);
        RealD dS_actual = Sp - Sm;
        ComplexD ratio = dS_actual / (2.0 * eps * dSpred);
        std::cout << GridLogMessage << "[FD][" << name << "]"
                  << " ε=" << std::scientific << std::setprecision(2) << eps
                  << " dS_act/2ε=" << std::setprecision(8) << dS_actual / (2.0 * eps)
                  << " dS_pred=" << dSpred << " ratio=" << ratio << std::endl;
      }
    }
    Grid_finalize();
    return 0;
  }

  // Branch on integrator.  Both types compiled, chosen at runtime.
  if (integrator_name == "ForceGradient") {
    typedef ForceGradient<PeriodicGimplR,
                          SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  } else {
    typedef MinimumNorm2<PeriodicGimplR,
                         SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "QCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
