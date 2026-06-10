// gen_qcd_hasenbusch_tune.cc
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
#include <cstring>
#include <iomanip>
#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourRatio.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavour.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/algorithms/iterative/ConjugateGradientMixedPrec.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverQudaForceRationalActionMP.h>
#endif

using namespace Grid;
using namespace TXQCDProduction;

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
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  typedef WilsonCloverFermion<WilsonImplF, CloverHelpers<WilsonImplF>> WCF_f;

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
        Umu, Grid_, RBGrid_, ladder[i], csw, csw, anis, impl_p));
  std::cout << GridLogMessage << "Built " << n_ops << " DP light operators." << std::endl;

  // Strange: DP + SP for mixed-precision multishift RHMC.
  WCF   StrangeOp (Umu,  Grid_,  RBGrid_,  mass_strange, csw, csw, anis, impl_p);
  WCF_f StrangeOpF(UmuF, GridF, RBGridF,  mass_strange, csw, csw, anis, impl_pF);

  // ── Solvers ───────────────────────────────────────────────────────────────
  // CG tolerances are env-overridable for fast Hasenbusch mass tuning — the force
  // norms used for tuning are insensitive to solve precision.  Defaults reproduce
  // production (action/heatbath 1e-8, force 1e-6); e.g. TUNE_CG_TOL_ACTION=1e-4
  // TUNE_CG_TOL_DERIV=1e-4 cuts the dominant light-quark solves substantially.
  const RealD cg_tol_act = TXQCDProduction::detail::env_real("TUNE_CG_TOL_ACTION", cg_tol);
  const RealD cg_tol_drv = TXQCDProduction::detail::env_real("TUNE_CG_TOL_DERIV", 1e-6);
  std::cout << GridLogMessage << "CG tol: action/heatbath=" << cg_tol_act
            << " deriv=" << cg_tol_drv << std::endl;
  ConjugateGradient<LatticeFermion> CG_action(cg_tol_act, cg_max);  // ratio S() + heatbath
  ConjugateGradient<LatticeFermion> CG_deriv(cg_tol_drv,  cg_max);  // ratio deriv()

  SchurDifferentiableOperator<WilsonImplR> StrangeSchurOpD(StrangeOp);
  SchurDifferentiableOperator<WilsonImplF> StrangeSchurOpF(StrangeOpF);
  MixedPrecCGWrapper<LatticeFermion, LatticeFermionF,
                     SchurDifferentiableOperator<WilsonImplR>,
                     SchurDifferentiableOperator<WilsonImplF>>
      CG_strange_md(1e-6, cg_max, 50, &RBGridF, StrangeSchurOpD, StrangeSchurOpF);

  // ── Actions ───────────────────────────────────────────────────────────────
  // NOTE: there is deliberately NO light det(M_ee) log-det action.  The
  // full-lattice ratio chain + bare-det tail below already represent the
  // COMPLETE det(M_l)^2 (both M_ee and M_oo); an EO log-det here (which is meant
  // to pair with a Schur action, as the strange sector does) would double-count
  // det(M_ee).  See the action-bookkeeping note in hasenbusch_tests_perlmutter.md.

  // Hasenbusch ratio levels.
  // Level k: det(M(ladder[k])) / det(M(ladder[k+1]))
  //   constructor: TwoFlavourRatioPseudoFermionAction(NumOp=heavy, DenOp=light, ...)
  //   represents det(DenOp=light)^2 / det(NumOp=heavy)^2
  int n_pf = n_ops - 1;
  std::vector<std::unique_ptr<TwoFlavourRatioPseudoFermionAction<WilsonImplR>>> RatioPF;
  for (int k = 0; k < n_pf; ++k) {
    RatioPF.emplace_back(
        std::make_unique<TwoFlavourRatioPseudoFermionAction<WilsonImplR>>(
            *LightOps[k+1],  // NumOp = heavier mass
            *LightOps[k],    // DenOp = lighter mass
            CG_deriv, CG_action));
    RatioPF.back()->is_smeared = true;
  }
  std::cout << GridLogMessage << "Built " << n_pf << " ratio PF levels." << std::endl;

  // Bare-det tail: det(M(ladder.back()))^2 as a single two-flavour pseudofermion,
  // capping the telescoping ratio chain so the light sector is exactly det(M_l)^2.
  // Replaces the old hand-inserted Pauli-Villars ratio (matches Chroma's
  // TWO_FLAVOR_EOPREC_CONSTDET_FERM_MONOMIAL tail at the heaviest mass).
  TwoFlavourPseudoFermionAction<WilsonImplR> LightTail(
      *LightOps[n_ops-1], CG_deriv, CG_action);
  LightTail.is_smeared = true;

  // Strange EO log-det: -ln|det(M_ee)| at mass_strange.
  QCDLogDetCloverEOAction<WilsonImplR> StrangeLogDet(StrangeOp, 1);
  StrangeLogDet.is_smeared = true;

  // Strange RHMC. Bounds and degree matched to Chroma's rat_3strange monomial.
  OneFlavourRationalParams strange_rat(1e-4, 100.0, cg_max, cg_tol, 20, 64, 100, 1e-6, 1e-4);
  std::unique_ptr<OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>>
      StrangeBase;
  Action<LatticeGaugeField> *StrangePtr = nullptr;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<OneFlavourSchurCloverQudaForceRationalActionMP<WilsonImplR, WilsonImplF>>
      StrangeQuda;
  if (std::getenv("QUDA_FORCE") != nullptr) {
    QudaCloverParams qp;
    qp.mass = mass_strange; qp.csw = csw; qp.anti_periodic_t = true;
    qp.tol = cg_tol; qp.max_iter = cg_max;
    qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    StrangeQuda = std::make_unique<
        OneFlavourSchurCloverQudaForceRationalActionMP<WilsonImplR, WilsonImplF>>(
        StrangeOp, StrangeOpF, &RBGridF, strange_rat, qp, 50);
    StrangeQuda->is_smeared = true;
    StrangePtr = StrangeQuda.get();
    std::cout << GridLogMessage << "[Strange] QUDA_FORCE active." << std::endl;
  } else
#endif
  {
    StrangeBase = std::make_unique<
        OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>>(
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

  // ── Integrator: flat 2-level (fermion L1 + gauge L2) ─────────────────────
  // Flat structure is intentional for tuning: all PF levels at the same MD
  // rate gives a clean per-level force comparison.
  int gauge_mult = 4;
  if (const char *gm = std::getenv("GAUGE_INNER_MULT"); gm && *gm) gauge_mult = std::atoi(gm);
  std::cout << GridLogMessage << "GAUGE_INNER_MULT=" << gauge_mult << std::endl;

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  ActionLevel<LatticeGaugeField, Reps> L1(1);
  ActionLevel<LatticeGaugeField, Reps> L2(gauge_mult);

  for (auto &pf : RatioPF) L1.push_back(pf.get());
  L1.push_back(&LightTail);
  L1.push_back(&StrangeLogDet);
  L1.push_back(&StrangeSchurPF);
  L2.push_back(&GaugeAction);

  ActionSet<LatticeGaugeField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);

  // ── Stout smearing ────────────────────────────────────────────────────────
  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear_inv, Stout);
  Smear.set_Field(Umu);

  // ── Force norm observer ───────────────────────────────────────────────────
  ForceNormObserver ForceObs;
  for (int k = 0; k < n_pf; ++k)
    ForceObs.refs.push_back({"PF" + std::to_string(k), RatioPF[k].get()});
  ForceObs.refs.push_back({"Tail", &LightTail});
  ForceObs.refs.push_back({"Strange", &StrangeSchurPF});
  ForceObs.refs.push_back({"Gauge",   &GaugeAction});

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
    auto skip = [&](const std::string &nm) {
      return (skip_strange && nm == "Strange") || (skip_gauge && nm == "Gauge");
    };
    LatticeGaugeField force(&Grid_);
    const int nref = (int)ForceObs.refs.size();
    std::vector<double> acc_avg(nref, 0.0), acc_max(nref, 0.0);
    Smear.set_Field(Umu);   // smear the fixed config once (deterministic)
    for (int s = 0; s < nsamp; ++s)
      for (int i = 0; i < nref; ++i) {
        if (skip(ForceObs.refs[i].name)) continue;
        Action<LatticeGaugeField> *act = ForceObs.refs[i].act;
        act->refresh(Smear, sRNG, pRNG);                 // heatbath pseudofermions
        act->deriv(Smear, force);                        // force incl. smearing chain rule
        force = PeriodicGimplR::projectForce(force);     // same projection the integrator uses
        double favg = std::sqrt(norm2(force) / force.Grid()->gSites());
        double fmax = std::sqrt(maxLocalNorm2(force));
        acc_avg[i] += favg;
        acc_max[i] += fmax;
        // Stream each level immediately so a later-level stall can't swallow the
        // numbers already computed (the end-of-loop summary is otherwise gated
        // behind every ref, including the FORCES_ONLY gauge eval).
        std::cout << GridLogMessage << "FORCES_ONLY level " << ForceObs.refs[i].name
                  << " sample " << s << " avg=" << favg << " max=" << fmax << std::endl;
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
    Grid_finalize();
    return 0;
  }

  // ── HMC ──────────────────────────────────────────────────────────────────
  IntegratorParameters MD;
  MD.name = "MinimumNorm2"; MD.MDsteps = mdsteps; MD.trajL = 1.0;

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
