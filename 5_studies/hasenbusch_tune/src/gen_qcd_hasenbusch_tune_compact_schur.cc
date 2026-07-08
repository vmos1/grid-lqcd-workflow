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
// Checkpointing (2026-07-08, roadmap D): opt-in via CKPT_DIR — per-trajectory
// ILDG .lime config + RNG state, with CKPT_RESUME_TRAJ stream resume.  Unset =
// the original no-output tuning behavior.  See the Checkpointer block in main.
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
#include <set>
#include <sstream>
#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCompactCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>
// EVEN-parity Grid-native Schur monomial — same parity as the QUDA force path,
// opt-in via STRANGE_EVEN=1 for same-parity grid-vs-quda force validation.
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/observables/plaquette.h>
#include <Grid/qcd/observables/polyakov_loop.h>
#include <Grid/algorithms/iterative/ConjugateGradientMixedPrec.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverQudaForceRationalActionMP.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioActionQuda.h>
#include <Grid/algorithms/iterative/QudaMGSchurSolver.h>
#include <Grid/algorithms/iterative/QudaCGSchurSolver.h>
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
  // HMC_SEED_OFFSET=k shifts every seed integer by k, giving an independent
  // momentum/pseudofermion stream. Unset/0 = the historical fixed seeds, so
  // all previous runs stay bit-reproducible.
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid_);
  int seed_off = 0;
  if (const char *so = std::getenv("HMC_SEED_OFFSET")) seed_off = std::atoi(so);
  if (seed_off)
    std::cout << GridLogMessage << "HMC_SEED_OFFSET=" << seed_off << std::endl;
  sRNG.SeedFixedIntegers({11 + seed_off, 12 + seed_off, 13 + seed_off,
                          14 + seed_off, 15 + seed_off});
  pRNG.SeedFixedIntegers({16 + seed_off, 17 + seed_off, 18 + seed_off,
                          19 + seed_off, 20 + seed_off});

  // ── Checkpointer (roadmap D: config I/O) ──────────────────────────────────
  // CKPT_DIR=<dir> enables per-trajectory checkpointing via Grid's stock
  // ILDGHmcCheckpointer: gauge -> <dir>/ckpoint_lat.<traj> (ILDG .lime,
  // readable back through this driver's own IMPORT_CFG path and by Chroma
  // tooling), full RNG state (serial+parallel) -> <dir>/ckpoint_rng.<traj>.
  // Unset = no checkpointing — the old behavior (accepted configs are lost).
  //   CKPT_INTERVAL=<n>     save every n-th trajectory (default 1).
  //   CKPT_START_TRAJ=<n>   trajectory-numbering offset for a FRESH start from
  //                         IMPORT_CFG (e.g. 2000 -> first save ckpoint_lat.2001).
  //   CKPT_RESUME_TRAJ=<n>  resume a stream: restore gauge + BOTH RNGs from
  //                         <dir>/ckpoint_{lat,rng}.<n> (IMPORT_CFG ignored, the
  //                         fixed seeds above overwritten) and continue at n+1.
  // Checkpoints are written after EVERY trajectory incl. rejects (the RNG state
  // must advance for stream continuity; a rejected trajectory re-saves the
  // reverted gauge field with the post-reject RNG).
  std::unique_ptr<ILDGHmcCheckpointer<PeriodicGimplR>> Ckpt;
  int ckpt_start = 0;
  if (const char *cd = std::getenv("CKPT_DIR"); cd && *cd) {
    CheckpointerParameters cp;
    cp.config_prefix  = std::string(cd) + "/ckpoint_lat";
    cp.smeared_prefix = std::string(cd) + "/ckpoint_lat_smr";
    cp.rng_prefix     = std::string(cd) + "/ckpoint_rng";
    cp.saveInterval   = 1;
    if (const char *ci = std::getenv("CKPT_INTERVAL"); ci && *ci)
      cp.saveInterval = std::atoi(ci);
    cp.saveSmeared    = false;
    cp.format         = "IEEE64BIG";
    Ckpt = std::make_unique<ILDGHmcCheckpointer<PeriodicGimplR>>(cp);
    std::cout << GridLogMessage << "[Ckpt] CKPT_DIR=" << cd
              << " interval=" << cp.saveInterval << std::endl;
  }
  if (const char *st = std::getenv("CKPT_START_TRAJ"); st && *st)
    ckpt_start = std::atoi(st);
  int ckpt_resume = -1;
  if (const char *rt = std::getenv("CKPT_RESUME_TRAJ"); rt && *rt)
    ckpt_resume = std::atoi(rt);
  if (ckpt_resume >= 0 && !Ckpt) {
    std::cout << GridLogMessage
              << "FATAL: CKPT_RESUME_TRAJ requires CKPT_DIR." << std::endl;
    Grid_finalize();
    return 1;
  }

  // ── Gauge field ───────────────────────────────────────────────────────────
  LatticeGaugeField Umu(&Grid_);
  if (ckpt_resume >= 0) {
    std::cout << GridLogMessage << "[Ckpt] CKPT_RESUME_TRAJ=" << ckpt_resume
              << " — restoring gauge + RNG state (IMPORT_CFG ignored)" << std::endl;
    Ckpt->CheckpointRestore(ckpt_resume, Umu, sRNG, pRNG);
    ckpt_start = ckpt_resume;
  } else if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
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

  // Hasenbusch ratio levels (EO/Schur-preconditioned, clover-correct).
  //   det(Schur(ladder[k]))^2 / det(Schur(ladder[k+1]))^2  via
  //   TwoFlavourSchurCloverRatioAction(NumOp=heavy, DenOp=light, DS, AS) -- the
  //   Schur-only ratio class (no ConstEE assumption, unlike Mike's
  //   TwoFlavourEvenOddRatio.h, which crashes here for clover).  The EE
  //   determinant is supplied separately by LightLogDet above (telescoping).
  // Small Δm → well-conditioned → DP CG (CG_deriv/CG_action) suffices (as in the full-op driver).
  // Two independent QUDA hooks, same QudaMGSchurSolver.h wrapper (2-solve gamma5 trick is agnostic
  // to which inner solver QudaCloverInverter runs):
  //   HASEN_MG_RUNG=<k>          -- rung k's Mpc(DenOp) solve via QUDA MULTIGRID (use_multigrid=true).
  //   HASEN_QUDA_CG_RUNGS=<csv>  -- these rungs' Mpc(DenOp) solve via plain QUDA CG (still GPU-side /
  //                                 faster than Grid's CG, no MG preconditioner setup cost) -- useful
  //                                 for the heavier, already-well-conditioned rungs where MG's extra
  //                                 setup isn't worth it but QUDA's CG throughput still helps.
  // A rung may only be routed one way; HASEN_MG_RUNG takes precedence if both list the same rung.
  // Both env vars accept a comma-separated list of rung indices (HASEN_MG_RUNG=0 or HASEN_MG_RUNG=0,1,2
  // both work) so a single run can put MG on every rung at once, for a same-seed timing/force
  // comparison against the Grid-CG and QUDA-CG passes.
  int n_pf = n_ops - 1;
  std::vector<std::unique_ptr<Action<LatticeGaugeField>>> RatioPF;
#ifdef GRID_HAVE_QUDA
  std::vector<std::unique_ptr<QudaRungSolverBase>> QudaRungSolver(n_pf);
  // Parallel, INDEPENDENT solver vector for the heatbath (refresh()) step --
  // built at NumOp's mass (ladder[k+1]), not DenOp's (ladder[k]), since
  // refresh's normal-equation problem is M_1^dag M_1, not M_0's.  Only
  // populated for rungs listed in HASEN_QUDA_CG_HEATBATH_RUNGS -- see
  // TwoFlavourSchurCloverRatioActionQuda.h for why QudaCGSchurSolver (no
  // extra code) is the right tool: it already solves exactly the
  // M^dag M x = b, then apply M problem HeatbathSolver poses.
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
      qp_mg.tol = cg_tol_drv;
      qp_mg.max_iter = cg_max;
      qp_mg.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
      qp_mg.use_multigrid = want_mg;
      // Diagnostic: QudaCloverParams defaults cuda_prec_sloppy=SINGLE (mixed-
      // precision inner iterations). QUDA_SLOPPY_DOUBLE=1 forces pure double
      // precision throughout, to test whether a plain-CG rung's force
      // discrepancy vs Grid-CG (seen at 48^3, NOT at 16^3) is caused by
      // single-precision roundoff accumulating over the many (~1000s at 48^3
      // near-critical rungs) CG iterations needed to hit a fixed residual
      // tolerance -- MG needs far fewer iterations for the same tolerance, so
      // if this hypothesis is right, only plain CG should be sensitive to it.
      if (std::getenv("QUDA_SLOPPY_DOUBLE") != nullptr) {
        qp_mg.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
      }
      // 48^3 antiperiodic-t: the boundary time links carry the baked-in -1
      // (det=-1); RECONSTRUCT_12 (the default) mis-rebuilds them, assuming
      // det=+1. Same fix as the strange/light paths in
      // gen_qcd_hasenbusch_tune_compact.cc (QUDA_FORCE_RECON_NO), ported here
      // because compact_schur's plain QUDA-CG rungs never had it -- 16^3 is
      // well-conditioned enough to tolerate the mis-reconstruction, 48^3 is not.
      if (std::getenv("QUDA_FORCE_RECON_NO") != nullptr) {
        qp_mg.recon_sloppy = QUDA_RECONSTRUCT_NO;
      }
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
      }
      if (want_mg) {
        QudaRungSolver[k] = std::make_unique<QudaMGSchurSolver>(LightOps[k]->GaugeGrid(), qp_mg, Odd);
      } else {
        // Plain QUDA-CG: QudaMGSchurSolver's zero-pad/gamma5 trick was only
        // ever validated for use_multigrid=true (MG's QUDA_DIRECT_SOLVE,
        // where matpc_type is irrelevant). Its inner QudaCloverInverter
        // hardcodes matpc_type=EVEN_EVEN, which is the WRONG parity/form for
        // our Odd-checkerboard asymmetric Schur system under CG's
        // QUDA_NORMOP_PC_SOLVE -- see QudaCGSchurSolver.h for the fix
        // (direct half-volume solve, ODD_ODD_ASYMMETRIC, same convention
        // already proven in OneFlavourSchurCloverQudaRationalActionMP.h).
        QudaRungSolver[k] = std::make_unique<QudaCGSchurSolver>(LightOps[k]->GaugeGrid(), qp_mg, Odd);
      }
      // Heatbath (refresh()) acceleration is INDEPENDENT of the deriv/action
      // choice above -- separate env var, separate solver instance, built at
      // NumOp's mass (ladder[k+1]).  Only offered on rungs that already have
      // a QUDA deriv/action solver (this loop branch) -- no infrastructure
      // exists yet to attach a QUDA heatbath solver to a plain-Grid-CG rung.
      OperatorFunction<LatticeFermion> *heatbath_solver_ptr = &CG_action;
      if (quda_cg_heatbath_rungs.count(k)) {
        QudaCloverParams qp_hb;
        qp_hb.mass = ladder[k+1];  // NumOp mass -- refresh solves M_1^dag M_1
        qp_hb.csw  = csw;
        qp_hb.anti_periodic_t = true;
        qp_hb.tol = cg_tol_act;  // heatbath sets phi for the whole trajectory -- match CG_action's tol
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
      RatioPF.emplace_back(std::make_unique<TwoFlavourSchurCloverRatioActionQuda<WilsonImplR, WCF>>(
          *LightOps[k+1], *LightOps[k], *QudaRungSolver[k], *heatbath_solver_ptr));
      std::cout << GridLogMessage << "[Ladder] rung " << k
                << " DerivativeSolver/ActionSolver = "
                << (want_mg ? ("QUDA MG (" + std::to_string(qp_mg.mg.n_level) + " levels)") : std::string("QUDA CG"))
                << ", mass=" << ladder[k] << std::endl;
#else
      std::cerr << "HASEN_MG_RUNG/HASEN_QUDA_CG_RUNGS require a QUDA build.\n"; exit(1);
#endif
    } else {
      RatioPF.emplace_back(
          std::make_unique<TwoFlavourSchurCloverRatioAction<WilsonImplR, WCF>>(
              *LightOps[k+1],  // NumOp = heavier mass
              *LightOps[k],    // DenOp = lighter mass
              CG_deriv, CG_action));
    }
    RatioPF.back()->is_smeared = true;
  }
  std::cout << GridLogMessage << "Built " << n_pf << " Schur ratio PF levels." << std::endl;

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
    // QUDA_FORCE_RECON_NO=1 → reconstruct_sloppy = NO (full links). REQUIRED at
    // 48^3: the antiperiodic-t phase is baked in by negating the last-timeslice
    // time links (det=-1, not SU(3)); RECONSTRUCT_12 (the default) mis-rebuilds
    // the 3rd row of those links (it assumes det=+1), corrupting the sloppy
    // solve. Ported from gen_qcd_hasenbusch_tune_compact.cc (memory
    // quda-strange-recon-fix) -- this driver's strange QUDA path never had it.
    if (std::getenv("QUDA_FORCE_RECON_NO") != nullptr)
      qp.recon_sloppy = QUDA_RECONSTRUCT_NO;
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

  // LW gauge action matching Chroma's LW_TREE_GAUGEACT: rect coeff
  // -beta/(20 u0^2) with u0 = the tadpole factor the ENSEMBLE was generated
  // with (embedded in each config's .lime XML) -- cl21 48^3 b6.3:
  // 0.84570646270714; cl3 16^3 b6.1: 0.832605301399891.  U0 must be set per
  // ensemble: a mismatch leaves the imported config off-shell and the gauge
  // relaxation drift drives the near-critical light sector singular
  // (tau~=0.15 blow-up, root-caused 2026-07-07, confirmed job 55645079).
  // NOTE this local default (1.0) shadows params.h's u0 default (0.8326) --
  // never rely on either; scripts set U0 explicitly.
  const RealD u0_ens = TXQCDProduction::detail::env_real("U0", 1.0);
  PlaqPlusRectangleAction<PeriodicGimplR> GaugeAction(beta, -beta / (20.0 * u0_ens * u0_ens));
  GaugeAction.is_smeared = false;
  std::cout << GridLogMessage << std::setprecision(15)
            << "[Action] BETA=" << beta << " U0=" << u0_ens
            << " rect_coeff=" << -beta / (20.0 * u0_ens * u0_ens)
            << " CSW=" << csw
            << " MASS_LIGHT=" << mass_light
            << " MASS_STRANGE=" << mass_strange
            << std::setprecision(6) << std::endl;

  // ── Integrator: 2-level fermions / gauge, matching Chroma ────────────────
  // ALL fermion monomials (strange RHMC + both logdets + light ladder + tail)
  // share the top level -- Chroma's production grouping (rat_strange + has0-3
  // + logdets at n_steps; only cancel+gauge finer).  In Grid's nested scheme
  // a child level always runs 2 x multiplier x its parent's steps (each
  // parent step recurses into the child once per drift slot -- same
  // convention as Chroma's lcm_sts_*_recursive), so
  //   gauge steps/traj = MDSTEPS x 2 x GAUGE_INNER_MULT.
  //   GAUGE_INNER_MULT=2 (default) at MDSTEPS=12 reproduces Chroma's
  //   12x2x2=48-step inner level exactly.
  // The earlier 3-level strange/light/gauge layout was dropped 2026-07-06:
  // it silently ran the light ladder 2x finer than strange (the recursion
  // factor -- no multiplier setting avoids it), i.e. 2x the light-sector
  // force solves for nothing Chroma needs.  Re-add later, if profiling ever
  // justifies it, by giving the light actions their own ActionLevel again.
  // See docs/2026_7_6_integrator_step_accounting_summary.md.
  //
  // Correctness: each action term is in exactly one level; the Metropolis
  // test keeps the sampled distribution exact for any multipliers -- only
  // acceptance/efficiency changes.
  int gauge_mult = 2;
  if (const char *gm = std::getenv("GAUGE_INNER_MULT"); gm && *gm) gauge_mult = std::atoi(gm);
  std::cout << GridLogMessage
            << "Integrator: 2-level (strange+light)/gauge  GAUGE_INNER_MULT="
            << gauge_mult << std::endl;

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  ActionLevel<LatticeGaugeField, Reps> Lferm(1);             // top: all fermions
  ActionLevel<LatticeGaugeField, Reps> Lgauge(gauge_mult);   // finest: gauge

  Lferm.push_back(&StrangeSchurPF);
  Lferm.push_back(&StrangeLogDet);
  Lferm.push_back(&LightLogDet);
  for (auto &pf : RatioPF) Lferm.push_back(pf.get());
  Lferm.push_back(&LightTailSchur);
  Lgauge.push_back(&GaugeAction);

  ActionSet<LatticeGaugeField, Reps> Aset;
  Aset.push_back(Lferm);      // coarsest pushed first
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
    std::vector<double> acc_avg(nref, 0.0), acc_max(nref, 0.0), acc_time(nref, 0.0), acc_refresh(nref, 0.0);
    Smear.set_Field(Umu);   // smear the fixed config once (deterministic)
    for (int s = 0; s < nsamp; ++s)
      for (int i = 0; i < nref; ++i) {
        if (skip(ForceObs.refs[i].name)) continue;
        Action<LatticeGaugeField> *act = ForceObs.refs[i].act;
        double ftime, rtime = 0.0;
        if (act->is_smeared) {
          // Smeared pseudofermion / log-det levels: heatbath, then evaluate the
          // force through the smearing chain rule -- exactly as the integrator's
          // first P-update does.  refresh() is the heatbath solve -- for a
          // ratio rung this is the expensive step on a near-critical mass (it
          // is NOT accelerated by HASEN_MG_RUNG, which only speeds up
          // deriv()/S(); see QudaMGSchurSolver notes), so time it separately
          // from deriv() rather than lump both into one number.
          auto tr0 = std::chrono::steady_clock::now();
          act->refresh(Smear, sRNG, pRNG);               // heatbath pseudofermions
          rtime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tr0).count();
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
        acc_refresh[i] += rtime;
        // Stream each level immediately so a later-level stall can't swallow the
        // numbers already computed (the end-of-loop summary is otherwise gated
        // behind every ref, including the FORCES_ONLY gauge eval).
        std::cout << GridLogMessage << "FORCES_ONLY level " << ForceObs.refs[i].name
                  << " sample " << s << " avg=" << favg << " max=" << fmax
                  << " refresh=" << rtime << " time=" << ftime << std::endl;
#ifdef GRID_HAVE_QUDA
        // FORCES_ONLY is a ONE-SHOT diagnostic (single heatbath+force per rung,
        // no MD loop) -- unlike the real HMC path, nothing revisits this rung's
        // solver again after its last sample, so its GPU memory (gauge/clover
        // copies, and for MG the coarse-grid + null-vector state) can be freed
        // immediately rather than held for the rest of the program. This caps
        // peak GPU memory at ~1 QUDA-routed rung's worth instead of accumulating
        // one per rung -- avoids the OOM seen routing all 4 rungs through
        // QUDA/MG simultaneously at 48^3. Real HMC trajectories must NOT do
        // this (they reuse the same solver every MD step, which is the whole
        // point of amortizing MG's setup cost) -- this code path only runs
        // inside the FORCES_ONLY block, never in the trajectory/integrator path.
        if (s == nsamp - 1 && ForceObs.refs[i].name.rfind("PF", 0) == 0) {
          int k = std::atoi(ForceObs.refs[i].name.c_str() + 2);
          if (k >= 0 && k < (int)QudaRungSolver.size() && QudaRungSolver[k]) {
            QudaRungSolver[k].reset();
            std::cout << GridLogMessage << "[FORCES_ONLY] freed QUDA solver for rung " << k << std::endl;
          }
        }
#endif
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
    emit("refresh", acc_refresh);
    emit("time", acc_time);
    Grid_finalize();
    return 0;
  }

  // ── HMC ──────────────────────────────────────────────────────────────────
  // INTEGRATOR env var: "MinimumNorm2" (default, 2nd order) or "ForceGradient"
  // (4th order -- the equivalent of Chroma's LCM_STS_FORCE_GRAD, tolerating
  // ~2-3x larger steps at the same acceptance).  Same convention as
  // gen_qcd_cfgs_2plus1.cc.
  std::string integrator_name = "MinimumNorm2";
  if (const char *env = std::getenv("INTEGRATOR"); env && *env)
    integrator_name = env;
  std::cout << GridLogMessage << "INTEGRATOR=" << integrator_name << std::endl;
  IntegratorParameters MD;
  MD.name = integrator_name; MD.MDsteps = mdsteps;
  // TRAJL has no default and is validated at startup (see the mandatory check
  // above), so it is guaranteed present here.
  MD.trajL = std::atof(std::getenv("TRAJL"));

  // NoMetropolisUntil: default 0 (chroma-style Metropolis-from-trajectory-0);
  // override via NO_METROP env var to skip accept/reject for the first N
  // trajectories -- useful for inspecting raw dH/force behaviour without the
  // gauge reverting on a reject (same convention as gen_qcd_cfgs_2plus1.cc).
  int no_metrop = 0;
  if (const char *nm = std::getenv("NO_METROP"); nm && *nm) no_metrop = std::atoi(nm);
  std::cout << GridLogMessage << "NoMetropolisUntil=" << no_metrop << std::endl;

  HMCparameters HMCp;
  // StartTrajectory sets the numbering the checkpointer saves under (evolve
  // saves traj+1): 0 unless CKPT_START_TRAJ (fresh numbered stream) or
  // CKPT_RESUME_TRAJ (continue at n+1) was given.
  HMCp.StartTrajectory    = ckpt_start;
  HMCp.Trajectories       = n_traj;
  HMCp.NoMetropolisUntil  = no_metrop;
  HMCp.MetropolisTest     = true;
  HMCp.PerformRandomShift = false;
  HMCp.StartingType       = "ColdStart";
  HMCp.MD                 = MD;

  // Per-trajectory gauge observables (plaquette + Polyakov loop) — Grid's stock
  // loggers. Read-only gauge reductions: draw no RNG, mutate nothing, ~ms cost.
  PlaquetteLogger<PeriodicGimplR> plaqLog;
  PolyakovLogger<PeriodicGimplR>  polyLog;
  std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ForceObs, &plaqLog, &polyLog};
  // Checkpointer runs last so the per-trajectory measurement lines land in the
  // log before the "Written ILDG Configuration" lines.
  if (Ckpt) Obs.push_back(Ckpt.get());

  // Branch on integrator.  Both types compiled, chosen at runtime (mirrors
  // gen_qcd_cfgs_2plus1.cc).
  if (integrator_name == "ForceGradient") {
    typedef ForceGradient<PeriodicGimplR,
                          SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid_, MD, Aset, Smear);
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  } else {
    typedef MinimumNorm2<PeriodicGimplR,
                         SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid_, MD, Aset, Smear);
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "Hasenbusch tuning run complete." << std::endl;
  Grid_finalize();
  return 0;
}
