// gen_qcd_hasenbusch_tune_compact_schur.cc
//
// LEAN + FAST variant: compact-clover storage (CompactWilsonCloverFermion, ~6x
// smaller host footprint → fits 48^3x96 on 2 nodes) AND an EO-Schur light sector
// (vs the unpreconditioned full-operator "Tail" in gen_qcd_hasenbusch_tune_compact.cc).
// The two axes are orthogonal — see docs/2026_6_21_compact_vs_eoschur.md.  The light
// det(M_l)^2 is represented in the SAME EO-Schur form the production 2+1 driver uses
// (TwoFlavourSchurCloverAction monomial + QCDLogDetCompactCloverEOAction, plus
// TwoFlavourEvenOddRatio levels when HASEN_LADDER is set).  Forked from
// gen_qcd_hasenbusch_tune_compact.cc; only the LIGHT sector differs.
//
// Standalone Hasenbusch mass tuning for 2+1f Wilson-clover QCD.
// Intended for the cl21_48_96_b6p3_m0p2416_m0p2050 ensemble (or any ensemble
// with env-var overrides).  Built with the TXQCD production Makefile so QUDA
// is available for the strange RHMC force.
//
// Physics: represents det(M_l)^2 as a Hasenbusch ratio chain (HASEN_LADDER)
//          capped by a bare-det tail at the heaviest ladder mass, matching the
//          Chroma cfg_2000 action (no Pauli-Villars).
// Prints a "FORCES traj=N ..." line per trajectory for easy grep + analysis.
// No HDF5 output, no checkpoint management — designed for short tuning runs.
//
// Key env vars:
//   HASEN_LADDER   Comma-separated masses light→heavy, e.g.
//                  "-0.2416,-0.2400,-0.2320,-0.2180,-0.1870"
//                  Lightest must equal MASS_LIGHT. Heaviest entry is the
//                  bare-det tail mass (no Pauli-Villars is appended).
//   N_TRAJ         Number of trajectories (default 5)
//   MDSTEPS        MD steps per trajectory (default 20)
//   IMPORT_CFG     Path to starting gauge config (NERSC or Chroma LIME)
//   QUDA_FORCE     Set to 1 to use QUDA for strange RHMC force
//   MASS_LIGHT     Light quark mass (default -0.2416 for this ensemble)
//   MASS_STRANGE   Strange quark mass (default -0.2050)
//   CSW            Clover coefficient (default 1.20537)
//   BETA           Gauge coupling (default 6.3)
//   U0             Mean-field u0 for LW rect coefficient (default 1.0 = tree-level)
//   GAUGE_INNER_MULT  Gauge sub-steps per fermion step (default 4)
//
// Compile: listed in production/Makefile PROGRAMS — built by default.
// Run:     see ~/projects/grid_qcd/jobs/tune-hasenbusch.sbatch

#include "params.h"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCompactCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourEvenOddRatio.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>
// EVEN-parity Grid-native Schur monomial — same parity as the QUDA force path,
// opt-in via STRANGE_EVEN=1 for same-parity grid-vs-quda force validation.
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/algorithms/iterative/ConjugateGradientMixedPrec.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverQudaForceRationalActionMP.h>
#endif

using namespace Grid;
using namespace TXQCDProduction;

// ---------------------------------------------------------------------------
// Subclass of TwoFlavourSchurCloverAction that re-syncs the SP operator from the
// raw gauge field before each deriv(), delegating the CG work to the MP wrapper
// installed as the DerivativeSolver.  Copied verbatim from the production 2+1
// driver (gen_qcd_cfgs_2plus1.cc) so the light Schur tail can run a mixed-precision
// force on the compact operator.  Templated on the operator types → binds to
// CompactWilsonCloverFermion (the FermOpD_/FermOpF_ args below).
// ---------------------------------------------------------------------------
namespace Grid {
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
    Base::deriv(U, dSdU);
  }

 private:
  FermOpF &opF_;
};
}  // namespace Grid

// ---------------------------------------------------------------------------
// Mixed-precision CG wrapper — used for the strange RHMC force solver.
// ---------------------------------------------------------------------------
template <class FieldD, class FieldF, class SchurOpD, class SchurOpF>
class MixedPrecCGWrapper : public OperatorFunction<FieldD> {
 public:
  using OperatorFunction<FieldD>::operator();

  MixedPrecCGWrapper(RealD tol, int max_inner, int max_outer,
                     GridBase *rbgrid_f, SchurOpD &schur_d, SchurOpF &schur_f)
      : tol_(tol), max_inner_(max_inner), max_outer_(max_outer),
        rbgrid_f_(rbgrid_f), schur_d_(schur_d), schur_f_(schur_f) {}

  void operator()(LinearOperatorBase<FieldD> &, const FieldD &src,
                  FieldD &sol) override {
    MixedPrecisionConjugateGradient<FieldD, FieldF> MPCG(
        tol_, max_inner_, max_outer_, rbgrid_f_, schur_f_, schur_d_);
    MPCG(src, sol);
  }

 private:
  RealD tol_; int max_inner_, max_outer_;
  GridBase *rbgrid_f_;
  SchurOpD &schur_d_; SchurOpF &schur_f_;
};

// ---------------------------------------------------------------------------
// ForceNormObserver: prints one grep-able FORCES line per trajectory.
// Format: FORCES traj=N PF0=X PF1=X ... Tail=X Strange=X Gauge=X
// ---------------------------------------------------------------------------
struct ForceNormObserver : public HmcObservable<LatticeGaugeField> {
  struct Ref { std::string name; Action<LatticeGaugeField> *act; };
  std::vector<Ref> refs;

  void TrajectoryComplete(int traj, LatticeGaugeField &,
                          GridSerialRNG &, GridParallelRNG &) override {
    std::cout << GridLogMessage << "FORCES traj=" << traj;
    for (auto &r : refs)
      std::cout << " " << r.name << "=" << r.act->deriv_norm_average();
    std::cout << std::endl;
  }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // ── Grid setup ────────────────────────────────────────────────────────────
  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid_(&Grid_);

  // Single-precision grids for mixed-precision strange RHMC.
  GridCartesian         GridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
  GridRedBlackCartesian RBGridF(&GridF);

  // ── Run parameters ────────────────────────────────────────────────────────
  int n_traj  = 5;
  int mdsteps = 20;
  if (const char *nt = std::getenv("N_TRAJ");  nt && *nt) n_traj  = std::atoi(nt);
  if (const char *ms = std::getenv("MDSTEPS"); ms && *ms) mdsteps = std::atoi(ms);
  std::cout << GridLogMessage << "N_TRAJ=" << n_traj
            << "  MDSTEPS=" << mdsteps << std::endl;

  // ── TRAJL is MANDATORY for HMC (no compiled-in default) ─────────────────────
  // A silent trajL default lets this driver disagree with another driver's
  // trajectory length (the 2+1 driver defaults to sqrt(2); this one used to
  // default to 1.0), making HMC timings non-comparable.  Require TRAJL explicitly
  // and fail fast HERE — before the (5.7 GB at 48^3) config load.  Forces-only
  // runs don't use the integrator, so they are exempt.
  if (std::getenv("FORCES_ONLY") == nullptr) {
    const char *tl = std::getenv("TRAJL");
    if (!tl || !*tl) {
      std::cout << GridLogMessage
                << "FATAL: TRAJL is not set.  This driver has no default trajectory "
                   "length for HMC — set it explicitly, e.g. "
                   "TRAJL=0.35355339059327379 (sqrt(2)/4, the benchmark value)."
                << std::endl;
      Grid_finalize();
      return 1;
    }
  }

  // ── Hasenbusch ladder ─────────────────────────────────────────────────────
  // HASEN_LADDER: comma-separated masses, strictly increasing (light→heavy).
  // Lightest entry must equal MASS_LIGHT.  Heaviest entry is the bare-det Tail
  // mass (no Pauli-Villars is appended — see note below).
  // If unset: single-level (no Hasenbusch) — use for baseline timing.
  std::vector<RealD> ladder;
  if (const char *hl = std::getenv("HASEN_LADDER"); hl && *hl) {
    std::string s(hl); size_t pos = 0, c;
    while ((c = s.find(',', pos)) != std::string::npos) {
      ladder.push_back(std::atof(s.substr(pos, c - pos).c_str())); pos = c + 1;
    }
    if (pos < s.size()) ladder.push_back(std::atof(s.substr(pos).c_str()));
    if (ladder.size() < 2) { std::cerr << "HASEN_LADDER needs ≥2 masses.\n"; exit(1); }
    for (size_t i = 1; i < ladder.size(); ++i)
      if (!(ladder[i] > ladder[i-1])) {
        std::cerr << "HASEN_LADDER must be strictly increasing.\n"; exit(1);
      }
  } else {
    ladder = { mass_light };
    std::cout << GridLogMessage << "HASEN_LADDER not set — single-level baseline." << std::endl;
  }
  // No Pauli-Villars: the chain is capped by a bare-det tail on the heaviest
  // ladder mass (ladder.back()), matching the Chroma cfg_2000 action
  // (TWO_FLAVOR_EOPREC_CONSTDET_FERM_MONOMIAL at -0.1870), so the telescoping
  // recovers det(M_l)^2 exactly with no leftover PV factor.

  std::cout << GridLogMessage << "Hasenbusch chain (" << (ladder.size()-1)
            << " ratios + bare-det tail at " << ladder.back() << "):";
  for (auto m : ladder) std::cout << " " << m;
  std::cout << std::endl;

  // ── RNG ───────────────────────────────────────────────────────────────────
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid_);
  sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
  pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});

  // ── Gauge field ───────────────────────────────────────────────────────────
  LatticeGaugeField Umu(&Grid_);
  if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic << std::endl;
    FILE *fp = std::fopen(ic, "rb"); char magic[16] = {0};
    if (fp) { std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      NerscIO::readConfiguration<GaugeStatistics<PeriodicGimplR>>(Umu, header, std::string(ic));
    } else {
      IldgReader IR; IR.open(std::string(ic));
      IR.readConfiguration(Umu, header); IR.close();
    }
  } else {
    std::cout << GridLogMessage << "No IMPORT_CFG — cold start." << std::endl;
    SU<Nc>::ColdConfiguration(pRNG, Umu);
  }

  // Plaquette of the loaded/initial gauge field — validates that IMPORT_CFG was
  // read correctly (compare to the ensemble reference, e.g. cfg_2000 ≈ 0.54351).
  std::cout << GridLogMessage << "Initial plaquette = "
            << std::setprecision(10)
            << WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu) << std::endl;

  // ── Fermion operator setup ────────────────────────────────────────────────
  // Compact operator must use CompactCloverHelpers — that is what libGrid
  // explicitly instantiates (CloverHelpers would link-fail).
  typedef CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<WilsonImplR>> WCF;
  typedef CompactWilsonCloverFermion<WilsonImplF, CompactCloverHelpers<WilsonImplF>> WCF_f;

  // SP gauge field for mixed-precision strange.
  LatticeGaugeFieldF UmuF(&GridF);
  {
    LatticeColourMatrix  Ud(&Grid_); LatticeColourMatrixF Uf(&GridF);
    for (int mu = 0; mu < Nd; ++mu) {
      Ud = PeekIndex<LorentzIndex>(Umu, mu);
      precisionChange(Uf, Ud);
      PokeIndex<LorentzIndex>(UmuF, Uf, mu);
    }
  }

  // Antiperiodic BC in time (matches Chroma <boundary>1 1 1 -1</boundary>).
  WilsonImplParams impl_p, impl_pF;
  impl_p.boundary_phases  = std::vector<Complex>({1., 1., 1., -1.});
  impl_pF.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
  WilsonAnisotropyCoefficients anis;

  // One DP light operator per ladder mass (heaviest = bare-det tail mass; no PV).
  // Hasenbusch ratios have small Δm → CG is well-conditioned → DP only needed.
  int n_ops = (int)ladder.size();
  std::vector<std::unique_ptr<WCF>> LightOps;
  for (int i = 0; i < n_ops; ++i)
    LightOps.emplace_back(std::make_unique<WCF>(
        Umu, Grid_, RBGrid_, ladder[i], csw, csw, /*cF=*/1.0, anis, impl_p));
  std::cout << GridLogMessage << "Built " << n_ops << " DP light operators." << std::endl;

  // SP light operators (one per ladder mass) for the mixed-precision light Schur
  // MD force — mirrors the strange DP+SP pattern below.
  std::vector<std::unique_ptr<WCF_f>> LightOpsF;
  for (int i = 0; i < n_ops; ++i)
    LightOpsF.emplace_back(std::make_unique<WCF_f>(
        UmuF, GridF, RBGridF, ladder[i], csw, csw, /*cF=*/1.0, anis, impl_pF));

  // Strange: DP + SP for mixed-precision multishift RHMC.
  WCF   StrangeOp (Umu,  Grid_,  RBGrid_,  mass_strange, csw, csw, /*cF=*/1.0, anis, impl_p);
  WCF_f StrangeOpF(UmuF, GridF, RBGridF,  mass_strange, csw, csw, /*cF=*/1.0, anis, impl_pF);

  // ── Solvers ───────────────────────────────────────────────────────────────
  // CG tolerances are env-overridable for fast Hasenbusch mass tuning — the force
  // norms used for tuning are insensitive to solve precision.  Defaults reproduce
  // production (action/heatbath 1e-8, force 1e-6); e.g. TUNE_CG_TOL_ACTION=1e-4
  // TUNE_CG_TOL_DERIV=1e-4 cuts the dominant light-quark solves substantially.
  const RealD cg_tol_act = TXQCDProduction::detail::env_real("TUNE_CG_TOL_ACTION", cg_tol);
  const RealD cg_tol_drv = TXQCDProduction::detail::env_real("TUNE_CG_TOL_DERIV", 1e-6);
  // TUNE_CG_TOL_STRANGE loosens the strange RHMC's multishift CG (otherwise
  // pinned to the production cg_tol regardless of TUNE_CG_TOL_*) — only
  // matters when the strange sector is actually evaluated (full HMC, or
  // FORCES_ONLY without FORCES_SKIP_STRANGE).
  const RealD cg_tol_strange = TXQCDProduction::detail::env_real("TUNE_CG_TOL_STRANGE", cg_tol);
  std::cout << GridLogMessage << "CG tol: action/heatbath=" << cg_tol_act
            << " deriv=" << cg_tol_drv << " strange=" << cg_tol_strange << std::endl;
  ConjugateGradient<LatticeFermion> CG_action(cg_tol_act, cg_max);  // ratio S() + heatbath
  ConjugateGradient<LatticeFermion> CG_deriv(cg_tol_drv,  cg_max);  // ratio deriv()

  SchurDifferentiableOperator<WilsonImplR> StrangeSchurOpD(StrangeOp);
  SchurDifferentiableOperator<WilsonImplF> StrangeSchurOpF(StrangeOpF);
  MixedPrecCGWrapper<LatticeFermion, LatticeFermionF,
                     SchurDifferentiableOperator<WilsonImplR>,
                     SchurDifferentiableOperator<WilsonImplF>>
      CG_strange_md(1e-6, cg_max, 50, &RBGridF, StrangeSchurOpD, StrangeSchurOpF);

  // ── Actions ───────────────────────────────────────────────────────────────
  // EO-SCHUR light sector (vs the full-operator "Tail" in the compact driver).
  //   det(M_l)^2 = [Π_k det(Schur(m_k))^2 / det(Schur(m_{k+1}))^2] · det(Schur(m_last))^2
  //                × det(M_ee(m_0))^2
  // The Schur ratios are EO ratio actions; the Schur tail is an EO-Schur monomial; the
  // clover-block (M_ee) determinants telescope across the ladder to a SINGLE log-det at the
  // LIGHTEST mass (Nf=2).  Single-level baseline (no HASEN_LADDER) → one Schur monomial + one
  // light log-det, exactly the production 2+1 driver's light sector.  (The ladder's M_ee
  // bookkeeping is validated by ΔH — Target B.)

  // Light EO log-det: det(M_ee(mass_light))^2  (Nf=2; lightest mass = LightOps[0]).
  QCDLogDetCompactCloverEOAction<WilsonImplR> LightLogDet(*LightOps[0], 2);
  LightLogDet.is_smeared = true;

  // Hasenbusch ratio levels (EO/Schur-preconditioned).
  //   det(Schur(ladder[k]))^2 / det(Schur(ladder[k+1]))^2  via
  //   TwoFlavourEvenOddRatioPseudoFermionAction(NumOp=heavy, DenOp=light, DS, AS).
  // Small Δm → well-conditioned → DP CG (CG_deriv/CG_action) suffices (as in the full-op driver).
  int n_pf = n_ops - 1;
  std::vector<std::unique_ptr<TwoFlavourEvenOddRatioPseudoFermionAction<WilsonImplR>>> RatioPF;
  for (int k = 0; k < n_pf; ++k) {
    RatioPF.emplace_back(
        std::make_unique<TwoFlavourEvenOddRatioPseudoFermionAction<WilsonImplR>>(
            *LightOps[k+1],  // NumOp = heavier mass
            *LightOps[k],    // DenOp = lighter mass
            CG_deriv, CG_action));
    RatioPF.back()->is_smeared = true;
  }
  std::cout << GridLogMessage << "Built " << n_pf << " EO ratio PF levels." << std::endl;

  // Schur tail: det(Schur(ladder.back()))^2 as a single EO-Schur monomial, capping the chain.
  // Mixed-precision MD force (SP inner + DP reliable), mirroring the 2+1 driver's LightSchurPF.
  int itail = n_ops - 1;
  SchurDifferentiableOperator<WilsonImplR> LightTailSchurD(*LightOps[itail]);
  SchurDifferentiableOperator<WilsonImplF> LightTailSchurF(*LightOpsF[itail]);
  MixedPrecCGWrapper<LatticeFermion, LatticeFermionF,
                     SchurDifferentiableOperator<WilsonImplR>,
                     SchurDifferentiableOperator<WilsonImplF>>
      CG_light_md(cg_tol_drv, cg_max, 50, &RBGridF, LightTailSchurD, LightTailSchurF);
  TwoFlavourSchurCloverActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f> LightTailSchur(
      *LightOps[itail], *LightOpsF[itail], CG_light_md, CG_action);
  LightTailSchur.is_smeared = true;

  // Strange EO log-det: -ln|det(M_ee)| at mass_strange.
  QCDLogDetCompactCloverEOAction<WilsonImplR> StrangeLogDet(StrangeOp, 1);
  StrangeLogDet.is_smeared = true;

  // Strange RHMC. Bounds and degree matched to Chroma's rat_3strange monomial.
  // Env-overridable for spectra where lo=1e-4 sits below lambda_min (phantom poles
  // that QUDA's multishift cannot invert); e.g. 48^3 b6.3 needs RAT_LO~0.4 RAT_HI~35.
  const RealD rat_lo = TXQCDProduction::detail::env_real("RAT_LO", 1e-4);
  const RealD rat_hi = TXQCDProduction::detail::env_real("RAT_HI", 100.0);
  int rat_degree = 20;
  if (const char *rd = std::getenv("RAT_DEGREE"); rd && *rd) rat_degree = std::atoi(rd);
  std::cout << GridLogMessage << "Strange rational: lo=" << rat_lo << " hi=" << rat_hi
            << " degree=" << rat_degree << std::endl;
  OneFlavourRationalParams strange_rat(rat_lo, rat_hi, cg_max, cg_tol_strange, rat_degree, 64, 100, 1e-6, 1e-4);
  // FermOp template args must be the compact operator types (defaults are the
  // non-compact WilsonCloverFermion).  The action body is operator-generic.
  std::unique_ptr<OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>>
      StrangeBase;
  // Same-parity validation holder: Grid-native EVEN-parity Schur monomial,
  // opt-in via STRANGE_EVEN=1 (default off -> stock ODD-parity behavior).
  std::unique_ptr<OneFlavourSchurCloverRationalActionEven<WilsonImplR, WCF>>
      StrangeEven;
  Action<LatticeGaugeField> *StrangePtr = nullptr;
#ifdef GRID_HAVE_QUDA
  // FermOp template args must be the compact operator types (WCF/WCF_f), same as
  // the non-QUDA branch below; the action's defaults are the non-compact
  // WilsonCloverFermion, which won't bind the compact StrangeOp/StrangeOpF.
  std::unique_ptr<OneFlavourSchurCloverQudaForceRationalActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>>
      StrangeQuda;
  if (std::getenv("QUDA_FORCE") != nullptr) {
    QudaCloverParams qp;
    qp.mass = mass_strange; qp.csw = csw; qp.anti_periodic_t = true;
    qp.tol = cg_tol_strange; qp.max_iter = cg_max;
    qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    // QUDA_FORCE_SLOPPY_DP=1 → run the strange multishift force solve fully DOUBLE
    // (default cuda_prec_sloppy=SINGLE).  IMPORTANT: this does NOT fix the 48^3 b6.3
    // wrong-force bug.  Test 4 (job 54678995) ran with this flag ON and the QUDA
    // multishift still diverged ("too many true residual norm increases" → wrong
    // force ‖F‖avg 18 / max 1857) even in full double.  So the 48^3 QUDA-strange
    // failure is precision-INDEPENDENT (pure-Grid converges with the SAME shifts);
    // the cause is in the QUDA operator/shift setup (kappa-form normalization of the
    // Remez shifts / matpc), not sloppy precision.  See
    // docs/2026_6_19_quda_single_precision_issues_multishift_cg_strange.md.
    if (std::getenv("QUDA_FORCE_SLOPPY_DP") != nullptr)
      qp.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
    StrangeQuda = std::make_unique<
        OneFlavourSchurCloverQudaForceRationalActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>>(
        StrangeOp, StrangeOpF, &RBGridF, strange_rat, qp, 50);
    StrangeQuda->is_smeared = true;
    StrangePtr = StrangeQuda.get();
    std::cout << GridLogMessage << "[Strange] QUDA_FORCE active." << std::endl;
  } else
#endif
  if (std::getenv("STRANGE_EVEN") != nullptr) {
    // EVEN-parity Schur (same parity as the QUDA force path) so a pure-grid vs
    // grid-quda comparison is same-pseudofermion; default unset -> stock ODD.
    StrangeEven = std::make_unique<
        OneFlavourSchurCloverRationalActionEven<WilsonImplR, WCF>>(
        StrangeOp, strange_rat);
    StrangeEven->is_smeared = true;
    StrangePtr = StrangeEven.get();
    std::cout << GridLogMessage
              << "[Strange] STRANGE_EVEN active — Grid native EVEN-parity Schur"
              << std::endl;
  } else {
    StrangeBase = std::make_unique<
        OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>>(
        StrangeOp, StrangeOpF, &RBGridF, strange_rat, 50);
    StrangeBase->is_smeared = true;
    StrangePtr = StrangeBase.get();
  }
  Action<LatticeGaugeField> &StrangeSchurPF = *StrangePtr;

  // LW gauge action matching Chroma's LW_TREE_GAUGEACT with u0=1 (tree-level).
  // params.h defaults u0=0.8326 for the collaborator's ensemble; override via U0=1.0.
  const RealD u0_ens = TXQCDProduction::detail::env_real("U0", 1.0);
  PlaqPlusRectangleAction<PeriodicGimplR> GaugeAction(beta, -beta / (20.0 * u0_ens * u0_ens));
  GaugeAction.is_smeared = false;

  // ── Integrator: 3-level strange / light / gauge, env-configurable ────────
  // Tuned-ladder force table (RMS / max), mass-independent except PF*/Tail:
  //   Gauge   6.88 / 10.4  (stiff, cheap)         -> finest, sub-stepped
  //   Tail    0.835 / 2.5  PF0-3 0.2-0.43         -> light fermion band
  //   Strange 0.62 / 2.2   (soft, but 373s solve) -> 2.4x the light solve cost
  //
  // One general 3-level structure (coarsest=strange, then light, finest=gauge)
  // with two integer knobs:
  //   LIGHT_INNER_MULT (=1): light sub-steps per strange step.
  //       =1 -> strange co-stepped with light (forces are ~equal, so this is
  //             the force-balanced choice; same step counts/cost as a flat
  //             2-level fermion/gauge integrator).
  //       =2 -> strange evaluated half as often: its force allows ~its own
  //             rate, and its 373s solve is 2.4x the light's, so coarsening it
  //             can be a net win despite a small acceptance hit.  Decide by
  //             measured time/traj / acceptance, not by the proxy.
  //   GAUGE_INNER_MULT (=4): gauge sub-steps per light step.  Only gauge is
  //       well separated in force (sqrt(6.88/0.835) ~ 3).  =1 collapses gauge
  //       to the light rate (toward single-timescale) for an A/B baseline.
  //
  // Correctness: each action term is in exactly one level; the Metropolis test
  // keeps the sampled distribution exact for any multipliers -- only the
  // acceptance/efficiency changes.
  int gauge_mult = 4;
  if (const char *gm = std::getenv("GAUGE_INNER_MULT"); gm && *gm) gauge_mult = std::atoi(gm);
  int light_mult = 1;
  if (const char *lm = std::getenv("LIGHT_INNER_MULT"); lm && *lm) light_mult = std::atoi(lm);
  std::cout << GridLogMessage
            << "Integrator: 3-level strange/light/gauge  LIGHT_INNER_MULT="
            << light_mult << " GAUGE_INNER_MULT=" << gauge_mult << std::endl;

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  ActionLevel<LatticeGaugeField, Reps> Lstrange(1);          // coarsest
  ActionLevel<LatticeGaugeField, Reps> Llight(light_mult);
  ActionLevel<LatticeGaugeField, Reps> Lgauge(gauge_mult);   // finest

  Lstrange.push_back(&StrangeSchurPF);
  Lstrange.push_back(&StrangeLogDet);
  Llight.push_back(&LightLogDet);
  for (auto &pf : RatioPF) Llight.push_back(pf.get());
  Llight.push_back(&LightTailSchur);
  Lgauge.push_back(&GaugeAction);

  ActionSet<LatticeGaugeField, Reps> Aset;
  Aset.push_back(Lstrange);   // coarsest pushed first
  Aset.push_back(Llight);
  Aset.push_back(Lgauge);

  // ── Stout smearing ────────────────────────────────────────────────────────
  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear_inv, Stout);
  Smear.set_Field(Umu);

  // ── Force norm observer ───────────────────────────────────────────────────
  ForceNormObserver ForceObs;
  ForceObs.refs.push_back({"LightLogDet", &LightLogDet});
  for (int k = 0; k < n_pf; ++k)
    ForceObs.refs.push_back({"PF" + std::to_string(k), RatioPF[k].get()});
  ForceObs.refs.push_back({"LightSchurPF", &LightTailSchur});
  ForceObs.refs.push_back({"Strange",       &StrangeSchurPF});
  ForceObs.refs.push_back({"StrangeLogDet", &StrangeLogDet});
  ForceObs.refs.push_back({"Gauge",         &GaugeAction});

  // ── Forces-only mode (fast Hasenbusch mass tuning) ────────────────────────
  // FORCES_ONLY=1: skip the integrator entirely.  Heatbath each action and
  // evaluate its MD force once on the (smeared) imported config — exactly as the
  // integrator's first P-update does — then print the per-level avg/max force
  // norm (the Hasenbusch tuning signal), averaged over FORCES_SAMPLES draws.
  // FORCES_SKIP_STRANGE=1 omits the expensive strange RHMC force.  Combine with
  // TUNE_CG_TOL_* for fast tuning iterations (no full trajectory needed).
  if (std::getenv("FORCES_ONLY") != nullptr) {
    int nsamp = 1;
    if (const char *ns = std::getenv("FORCES_SAMPLES"); ns && *ns) nsamp = std::atoi(ns);
    const bool skip_strange = std::getenv("FORCES_SKIP_STRANGE") != nullptr;
    const bool skip_gauge   = std::getenv("FORCES_SKIP_GAUGE")   != nullptr;
    const bool skip_light   = std::getenv("FORCES_SKIP_LIGHT")   != nullptr;  // LogDet + PF ladder + Schur PF
    auto skip = [&](const std::string &nm) {
      return (skip_strange && (nm == "Strange" || nm == "StrangeLogDet")) ||
             (skip_gauge && nm == "Gauge") ||
             (skip_light && (nm == "LightLogDet" || nm == "LightSchurPF" || nm.rfind("PF", 0) == 0));
    };
    LatticeGaugeField force(&Grid_);
    const int nref = (int)ForceObs.refs.size();
    std::vector<double> acc_avg(nref, 0.0), acc_max(nref, 0.0), acc_time(nref, 0.0);
    Smear.set_Field(Umu);   // smear the fixed config once (deterministic)
    for (int s = 0; s < nsamp; ++s)
      for (int i = 0; i < nref; ++i) {
        if (skip(ForceObs.refs[i].name)) continue;
        Action<LatticeGaugeField> *act = ForceObs.refs[i].act;
        double ftime;
        if (act->is_smeared) {
          // Smeared pseudofermion / log-det levels: heatbath, then evaluate the
          // force through the smearing chain rule -- exactly as the integrator's
          // first P-update does.
          act->refresh(Smear, sRNG, pRNG);               // heatbath pseudofermions
          auto t0 = std::chrono::steady_clock::now();
          act->deriv(Smear, force);                      // force incl. smearing chain rule
          ftime = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } else {
          // Unsmeared gauge action: its force depends only on the raw (thin)
          // links, so evaluate it directly on Umu -- bypassing the heatbath (a
          // no-op for a gauge action) and the SmearedConfiguration wrapper.
          // The wrapper path reproducibly HANGS on the standalone gauge eval
          // after the preceding smeared-fermion evals (leftover smearing/comm
          // state); the raw call does not stall and returns the same number the
          // integrator computes (it also feeds get_U(false)==Umu to deriv).
          auto t0 = std::chrono::steady_clock::now();
          act->deriv(Umu, force);
          ftime = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        force = PeriodicGimplR::projectForce(force);     // same projection the integrator uses
        double favg = std::sqrt(norm2(force) / force.Grid()->gSites());
        double fmax = std::sqrt(maxLocalNorm2(force));
        acc_avg[i] += favg;
        acc_max[i] += fmax;
        acc_time[i] += ftime;
        // Stream each level immediately so a later-level stall can't swallow the
        // numbers already computed (the end-of-loop summary is otherwise gated
        // behind every ref, including the FORCES_ONLY gauge eval).
        std::cout << GridLogMessage << "FORCES_ONLY level " << ForceObs.refs[i].name
                  << " sample " << s << " avg=" << favg << " max=" << fmax
                  << " time=" << ftime << std::endl;
      }
    auto emit = [&](const char *tag, std::vector<double> &acc) {
      std::cout << GridLogMessage << "FORCES_ONLY samples=" << nsamp << " " << tag << ":";
      for (int i = 0; i < nref; ++i)
        if (!skip(ForceObs.refs[i].name))
          std::cout << " " << ForceObs.refs[i].name << "=" << acc[i] / nsamp;
      std::cout << std::endl;
    };
    emit("avg", acc_avg);
    emit("max", acc_max);
    emit("time", acc_time);
    Grid_finalize();
    return 0;
  }

  // ── HMC ──────────────────────────────────────────────────────────────────
  IntegratorParameters MD;
  MD.name = "MinimumNorm2"; MD.MDsteps = mdsteps;
  // TRAJL has no default and is validated at startup (see the mandatory check
  // above), so it is guaranteed present here.
  MD.trajL = std::atof(std::getenv("TRAJL"));

  HMCparameters HMCp;
  HMCp.StartTrajectory    = 0;
  HMCp.Trajectories       = n_traj;
  HMCp.NoMetropolisUntil  = 0;
  HMCp.MetropolisTest     = true;
  HMCp.PerformRandomShift = false;
  HMCp.StartingType       = "ColdStart";
  HMCp.MD                 = MD;

  std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ForceObs};

  typedef MinimumNorm2<PeriodicGimplR,
                       SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
  IntT MDyn(&Grid_, MD, Aset, Smear);
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
  HMC.evolve();

  std::cout << GridLogMessage << "Hasenbusch tuning run complete." << std::endl;
  Grid_finalize();
  return 0;
}
