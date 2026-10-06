// test_fd_force.cc  (pure-Grid HMC: per-monomial finite-difference force check)
//
// Question: is every force the HMC integrator applies exactly -dS/dU of the action it is paired
// with?  For each monomial of the production configuration separately this program
//   1. builds the monomial exactly as the driver gen_qcd_hasenbusch_tune_compact_schur.cc does
//      (same classes, constructor arguments, parameters, stout smearing, solver routes and env
//      switches; the construction code below is copied from the driver's pure-Grid branch),
//   2. refreshes its pseudofermion ONCE at U0 (the imported configuration, projected onto SU(3))
//      and then holds it fixed,
//   3. computes the force exactly as Integrator::update_P does before the momentum update
//      (stock per-action loop: Ta(deriv(Smearer)); HASEN_GRID_BATCH_SMEAR=1: deriv on the
//      smeared links, then Smearer.smeared_force, then Ta; identical for a single action),
//   4. draws a random direction P with the HMC momentum generator (traceless antihermitian),
//   5. compares the central difference (S(exp(+eps P) U0) - S(exp(-eps P) U0)) / (2 eps), with
//      the smeared links recomputed for each perturbed field (Smearer.set_Field) and the
//      perturbation applied by the integrator's own update_field (exp(eps P) U, then the group
//      projection), with the directional derivative predicted by the force.
//
// Normalisation (stock Grid 3d3eff86): H = -sum tr(P^2)/D + S(U), D = HMC_MOMENTUM_DENOMINATOR
// = 2 (GaugeImplTypes.h, CPS_MD_TIME), update_U: U <- exp(ep P) U, update_P: P <- P - ep D F with
// F = Ta(force) (Integrator.h).  dH/dt = -(2/D) tr(P dP/dt) + dS/dt = 2 tr(P F) + dS/dt, so H is
// conserved by the continuum flow iff dS/dt = -2 Re sum_{x,mu} tr(P F), independent of D.  This is
// the formula of tests/forces/Test_rect_force.cc (dS = -trace(mom*UdSdU)*dt*2.0).  For an
// antihermitian P, tr(P F) = -tr(P^dag F) = -innerProduct(P, F), so
//     pred = dS/dt|_0 = 2 Re innerProduct(P, F)            (also cross-checked via sum(trace(P F)))
//     rel  = (FD(eps) - pred) / |pred|.
// A genuine force/action mismatch gives a rel that stays flat as eps shrinks; the central
// difference's truncation error falls like eps^2; solver and rounding noise grows like 1/eps.
// Richardson extrapolations of consecutive eps pairs (eps^2 term removed) are printed too.
// The gauge action (unsmeared, no solves) is the normalisation check.
//
// Environment (the driver's, unchanged meaning): LATT, IMPORT_CFG (required), BETA, U0, CSW,
// MASS_LIGHT, MASS_STRANGE, STOUT_RHO, STOUT_NSMEAR, RAT_LO, RAT_HI, RAT_DEGREE, HASEN_LADDER,
// HMC_SEED_OFFSET, TUNE_CG_TOL_ACTION, TUNE_CG_TOL_DERIV, TUNE_CG_TOL_STRANGE, STRANGE_EVEN,
// HASEN_GRID_STRANGE_LOGDET_ODD, HASEN_GRID_MG_RUNGS, HASEN_GRID_MG_HEATBATH_RUNGS,
// HASEN_GRID_MIXED_CG_RUNGS, HASEN_GRID_MIXED_CG_HEATBATH_RUNGS, GRID_MG_*, RAT_DEGREE_MD,
// TUNE_CG_TOL_STRANGE_MD, RAT_BOUNDS_CHECK_FREQ/_TOL/_ABORT (2026-10-05), and the switches read
// inside the headers and Grid (HASEN_GRID_FUSED_CLOVER_FORCE, _DEVICE_CB, _BATCH_SMEAR,
// _IMPORT_SKIP, _SHARE_FIELDSTRENGTH, _GPU_CLOVER_INV, _CLOVER_STENCIL).
// FD-specific:
//   FD_MONOMIALS   comma list of Gauge, LightLogDet, StrangeLogDet, PF<k>, Tail, Strange, or all
//                  (default all).  Only the operators and solvers the listed monomials need are
//                  built (48^3 on one node).  Ratio rungs not listed are dropped from the Grid-MG
//                  lists exactly as the driver's FORCES_ONLY skip de-routes them (the lowest
//                  listed MG rung becomes the hierarchy donor).
//   FD_EPS         comma list (default 1e-3,3e-4,1e-4), largest first.
//   FD_NDIR        random directions per monomial (default 1); the same directions for every
//                  monomial (own RNG, FD_SEED, default 4242).
//   FD_LINKS       smeared (default, production), thin (fermion monomials on the thin links,
//                  no pullback: isolates the action's own force from the stout chain rule), both.
//   FD_TOL_STRANGE_DERIV  optional: overrides the tolerance of the strange MP force multishift
//                  only (PowerNegHalf); S and the heatbath keep TUNE_CG_TOL_STRANGE.
// Output (boss rank): FDENV (provenance), FD (one line per eps), FDRICH (Richardson), and one
// "TEST RESULT FDSUMMARY ..." line per monomial and link mode.
//
// Build: perlmutter/build_test_fused_clover.sh with SRC/BIN overrides (pure Grid, stock tree).

#include "params.h"
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCompactCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/algorithms/iterative/ConjugateGradientMixedPrec.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h>
#include "grid_mg/grid_mg_schur_solver.h"
#include "grid_mg/mixed_cg_rung_solver.h"
#include "grid_mg/ratio_action_rung_solver.h"
#include "clover_force/fused_clover_force.h"

extern char **environ;

using namespace Grid;
using namespace TXQCDProduction;

// ---------------------------------------------------------------------------
// Copied verbatim from the driver (gen_qcd_hasenbusch_tune_compact_schur.cc): the tail's class.
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
// Copied verbatim from the driver: mixed-precision CG wrapper (the tail's MD solver).
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

namespace {

std::string env_str(const char *name, const std::string &dflt)
{
  const char *v = std::getenv(name);
  return (v && *v) ? std::string(v) : dflt;
}

std::vector<std::string> split_csv(const std::string &s)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty()) out.push_back(tok);
  return out;
}

// Integrator.h's strict 0/1 gate (batch_smear_gate), for HASEN_GRID_BATCH_SMEAR.
bool gate01(const char *name)
{
  const char *v = std::getenv(name);
  if (v == nullptr || *v == '\0') return false;
  const std::string s(v);
  if (s == "1") return true;
  if (s == "0") return false;
  std::cerr << name << "=" << v << ": must be 0 or 1." << std::endl;
  exit(EXIT_FAILURE);
  return false;
}

// The force Integrator::update_P applies for ONE action, before `Mom -= force*ep*D`.
//   stock loop (Integrator.h update_P):     act->deriv(Smearer, force); force = Ta(force)
//   HASEN_GRID_BATCH_SMEAR=1 (patch 07):    smeared: act->deriv(Smearer.get_U(true), force),
//                                           summed raw, Smearer.smeared_force(sum), Ta(sum);
//                                           unsmeared: the stock call.
// For a single action the two are the same function; both branches are kept so the run executes
// the code path its switch selects.
void integrator_force(Action<LatticeGaugeField> *act, SmearedConfiguration<PeriodicGimplR> &Smear,
                      LatticeGaugeField &force, bool batch)
{
  if (batch && act->is_smeared) {
    act->deriv(Smear.get_U(true), force);
    Smear.smeared_force(force);
  } else {
    act->deriv(Smear, force);
  }
  force = PeriodicGimplR::projectForce(force);
}

struct Mono {
  std::string name;
  Action<LatticeGaugeField> *act = nullptr;
  bool fermion = false;                               // smeared in production
  std::function<void(const LatticeGaugeField &)> pre;  // Grid-MG donor gauge sync, or empty
};

std::string sci(double x, int p = 6)
{
  std::ostringstream o;
  o << std::scientific << std::setprecision(p) << x;
  return o.str();
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // ── Grid setup (driver) ───────────────────────────────────────────────────
  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid_(&Grid_);
  GridCartesian         GridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
  GridRedBlackCartesian RBGridF(&GridF);

  // ── Provenance: every variable that can change what runs ──────────────────
  {
    std::ostringstream os;
    const char *pfx[] = {"LATT", "IMPORT_CFG", "BETA", "U0", "CSW", "MASS_", "STOUT_", "RAT_",
                         "HASEN_", "HMC_SEED_OFFSET", "TUNE_CG_TOL_", "STRANGE_EVEN", "GRID_MG_",
                         "FD_", "WCF_"};
    std::vector<std::string> kv;
    for (char **e = environ; *e; ++e)
      for (const char *p : pfx)
        if (std::strncmp(*e, p, std::strlen(p)) == 0) { kv.push_back(*e); break; }
    std::sort(kv.begin(), kv.end());
    for (auto &s : kv) os << " " << s;
    std::cout << GridLogMessage << "FDENV" << os.str() << std::endl;
  }

  // ── Hasenbusch ladder (driver) ────────────────────────────────────────────
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
    std::cout << GridLogMessage << "HASEN_LADDER not set — single-level baseline." << std::endl;
  }
  const int n_ops = (int)ladder.size();
  const int n_pf = n_ops - 1;
  const int itail = n_ops - 1;
  std::cout << GridLogMessage << "Hasenbusch chain (" << n_pf << " ratios + bare-det tail at "
            << ladder.back() << "):";
  for (auto m : ladder) std::cout << " " << m;
  std::cout << std::endl;

  // ── FD options ────────────────────────────────────────────────────────────
  std::set<std::string> sel;
  {
    const std::vector<std::string> toks = split_csv(env_str("FD_MONOMIALS", "all"));
    for (auto &t : toks) {
      if (t == "all") {
        for (const char *n : {"Gauge", "LightLogDet", "StrangeLogDet", "Tail", "Strange"}) sel.insert(n);
        for (int k = 0; k < n_pf; ++k) sel.insert("PF" + std::to_string(k));
        continue;
      }
      bool ok = (t == "Gauge" || t == "LightLogDet" || t == "StrangeLogDet" || t == "Tail" ||
                 t == "Strange");
      if (!ok && t.rfind("PF", 0) == 0 && t.size() > 2) {
        char *end = nullptr;
        const long k = std::strtol(t.c_str() + 2, &end, 10);
        ok = (*end == '\0' && k >= 0 && k < n_pf);
      }
      if (!ok) {
        std::cerr << "FD_MONOMIALS: unknown monomial '" << t << "' (Gauge, LightLogDet, "
                  << "StrangeLogDet, PF0..PF" << n_pf - 1 << ", Tail, Strange, all)\n";
        exit(1);
      }
      sel.insert(t);
    }
  }
  auto selected = [&](const std::string &n) { return sel.count(n) != 0; };
  std::vector<RealD> eps_list;
  for (auto &t : split_csv(env_str("FD_EPS", "1e-3,3e-4,1e-4"))) eps_list.push_back(std::atof(t.c_str()));
  std::sort(eps_list.begin(), eps_list.end(), std::greater<RealD>());
  const int ndir = std::max(1, std::atoi(env_str("FD_NDIR", "1").c_str()));
  const int fd_seed = std::atoi(env_str("FD_SEED", "4242").c_str());
  const std::string links = env_str("FD_LINKS", "smeared");
  if (links != "smeared" && links != "thin" && links != "both") {
    std::cerr << "FD_LINKS must be smeared, thin or both.\n"; exit(1);
  }
  const bool batch = gate01("HASEN_GRID_BATCH_SMEAR");
  std::cout << GridLogMessage << "FD monomials:";
  for (auto &s : sel) std::cout << " " << s;
  std::cout << " | eps:";
  for (auto e : eps_list) std::cout << " " << e;
  std::cout << " | ndir " << ndir << " seed " << fd_seed << " links " << links
            << " | force path: " << (batch ? "batched pullback (HASEN_GRID_BATCH_SMEAR=1)"
                                           : "stock per-action pullback") << std::endl;

  // ── RNG (driver) ──────────────────────────────────────────────────────────
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid_);
  int seed_off = 0;
  if (const char *so = std::getenv("HMC_SEED_OFFSET")) seed_off = std::atoi(so);
  sRNG.SeedFixedIntegers({11 + seed_off, 12 + seed_off, 13 + seed_off,
                          14 + seed_off, 15 + seed_off});
  pRNG.SeedFixedIntegers({16 + seed_off, 17 + seed_off, 18 + seed_off,
                          19 + seed_off, 20 + seed_off});

  // ── Gauge field (driver's IMPORT_CFG branch; required here) ───────────────
  LatticeGaugeField Umu(&Grid_);
  {
    const char *ic = std::getenv("IMPORT_CFG");
    if (!ic || !*ic) { std::cerr << "IMPORT_CFG is required.\n"; exit(1); }
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
  }
  std::cout << GridLogMessage << "Initial plaquette = " << std::setprecision(16)
            << WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu) << std::endl;
  // U0 = the imported field projected onto SU(3) (the integrator's end-of-trajectory Project), so
  // the force at U0 and the perturbed fields exp(+-eps P) U0 (projected by update_field) are all
  // on the group.
  {
    LatticeGaugeField Uraw(Umu);
    PeriodicGimplR::Project(Umu);
    LatticeGaugeField d(&Grid_);
    d = Umu - Uraw;
    std::cout << GridLogMessage << "U0 = Project(imported): |dU|^2/|U|^2 = "
              << norm2(d) / norm2(Uraw) << "  plaquette = "
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu) << std::setprecision(6) << std::endl;
  }

  // ── Fermion operators (driver types and arguments; built only when needed) ─
  typedef CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<WilsonImplR>> WCF;
  typedef CompactWilsonCloverFermion<WilsonImplF, CompactCloverHelpers<WilsonImplF>> WCF_f;

  LatticeGaugeFieldF UmuF(&GridF);
  {
    LatticeColourMatrix  Ud(&Grid_); LatticeColourMatrixF Uf(&GridF);
    for (int mu = 0; mu < Nd; ++mu) {
      Ud = PeekIndex<LorentzIndex>(Umu, mu);
      precisionChange(Uf, Ud);
      PokeIndex<LorentzIndex>(UmuF, Uf, mu);
    }
  }
  WilsonImplParams impl_p, impl_pF;
  impl_p.boundary_phases  = std::vector<Complex>({1., 1., 1., -1.});
  impl_pF.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
  WilsonAnisotropyCoefficients anis;

  std::vector<std::unique_ptr<WCF>>   LightOps(n_ops);
  std::vector<std::unique_ptr<WCF_f>> LightOpsF(n_ops);
  auto light_op = [&](int i) -> WCF & {
    if (!LightOps[i]) {
      LightOps[i] = std::make_unique<WCF>(Umu, Grid_, RBGrid_, ladder[i], csw, csw, /*cF=*/1.0,
                                          anis, impl_p);
      std::cout << GridLogMessage << "[FD] built DP light operator " << i << " mass "
                << ladder[i] << std::endl;
    }
    return *LightOps[i];
  };
  auto light_opF = [&](int i) -> WCF_f & {
    if (!LightOpsF[i]) {
      LightOpsF[i] = std::make_unique<WCF_f>(UmuF, GridF, RBGridF, ladder[i], csw, csw,
                                             /*cF=*/1.0, anis, impl_pF);
      std::cout << GridLogMessage << "[FD] built SP light operator " << i << " mass "
                << ladder[i] << std::endl;
    }
    return *LightOpsF[i];
  };
  std::unique_ptr<WCF> StrangeOpP;
  std::unique_ptr<WCF_f> StrangeOpFP;
  auto strange_op = [&]() -> WCF & {
    if (!StrangeOpP)
      StrangeOpP = std::make_unique<WCF>(Umu, Grid_, RBGrid_, mass_strange, csw, csw, 1.0, anis, impl_p);
    return *StrangeOpP;
  };
  auto strange_opF = [&]() -> WCF_f & {
    if (!StrangeOpFP)
      StrangeOpFP = std::make_unique<WCF_f>(UmuF, GridF, RBGridF, mass_strange, csw, csw, 1.0, anis, impl_pF);
    return *StrangeOpFP;
  };

  // ── Solvers (driver) ──────────────────────────────────────────────────────
  const RealD cg_tol_act = TXQCDProduction::detail::env_real("TUNE_CG_TOL_ACTION", cg_tol);
  const RealD cg_tol_drv = TXQCDProduction::detail::env_real("TUNE_CG_TOL_DERIV", 1e-6);
  const RealD cg_tol_strange = TXQCDProduction::detail::env_real("TUNE_CG_TOL_STRANGE", cg_tol);
  std::cout << GridLogMessage << "CG tol: action/heatbath=" << cg_tol_act
            << " deriv=" << cg_tol_drv << " strange=" << cg_tol_strange << std::endl;
  ConjugateGradient<LatticeFermion> CG_action(cg_tol_act, cg_max);
  ConjugateGradient<LatticeFermion> CG_deriv(cg_tol_drv,  cg_max);

  // ── HASEN_GRID_FUSED_CLOVER_FORCE (driver, verbatim parse) ─────────────────
  if (const char *v = std::getenv("HASEN_GRID_FUSED_CLOVER_FORCE"); v && *v) {
    const std::string s(v);
    if (s == "1") {
      FusedCloverForceEnabled() = true;
      std::cout << GridLogMessage << "HASEN_GRID_FUSED_CLOVER_FORCE=1: fused clover force ON"
                << std::endl;
    } else if (s != "0") {
      std::cerr << "HASEN_GRID_FUSED_CLOVER_FORCE=" << v << ": must be 0 or 1.\n";
      exit(1);
    }
  }

  // ── Light log-det (driver) ────────────────────────────────────────────────
  std::unique_ptr<QCDLogDetCompactCloverEOAction<WilsonImplR>> LightLogDet;
  if (selected("LightLogDet")) {
    LightLogDet = std::make_unique<QCDLogDetCompactCloverEOAction<WilsonImplR>>(light_op(0), 2);
    LightLogDet->is_smeared = true;
  }

  // ── Ratio rungs: the driver's pure-Grid routes ────────────────────────────
  // Rungs not selected play the role of the driver's FORCES_ONLY skipped rungs: dropped from the
  // Grid-MG lists before the hierarchy is built, the lowest surviving listed rung = donor.
  std::set<int> fo_skip;
  for (int k = 0; k < n_pf; ++k)
    if (!selected("PF" + std::to_string(k))) fo_skip.insert(k);
  auto parse_grid_rung_list = [n_pf](const char *name) {
    std::set<int> out;
    const char *v = std::getenv(name);
    if (!v || !*v) return out;
    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (tok.empty()) continue;
      char *end = nullptr;
      const long k = std::strtol(tok.c_str(), &end, 10);
      if (end == tok.c_str() || *end != '\0') {
        std::cerr << name << "=" << v << ": '" << tok << "' is not a rung index.\n";
        exit(1);
      }
      if (k < 0 || k >= n_pf) {
        std::cerr << name << "=" << v << ": rung " << k << " is not a ratio rung.\n";
        exit(1);
      }
      out.insert((int)k);
    }
    if (out.empty()) { std::cerr << name << "=" << v << " lists no rung.\n"; exit(1); }
    return out;
  };
  std::unique_ptr<GridMGHierarchy> GridMGH;
  std::vector<std::unique_ptr<GridMGSchurSolver>> GridMGRungSolver(n_pf);
  std::vector<std::unique_ptr<GridMGSchurSolver>> GridMGHeatbathSolver(n_pf);
  std::vector<std::unique_ptr<MixedPrecCGRungSolver>> GridMixedRungSolver(n_pf);
  std::vector<std::unique_ptr<MixedPrecCGRungSolver>> GridMixedHeatbathSolver(n_pf);
  std::vector<std::unique_ptr<Action<LatticeGaugeField>>> RatioPF(n_pf);
  int grid_mg_donor = -1;
  {
    const std::set<int> grid_mg_listed = parse_grid_rung_list("HASEN_GRID_MG_RUNGS");
    std::set<int> grid_mg_rungs = grid_mg_listed;
    std::set<int> grid_mg_hb_rungs = parse_grid_rung_list("HASEN_GRID_MG_HEATBATH_RUNGS");
    const std::set<int> grid_mixed_rungs = parse_grid_rung_list("HASEN_GRID_MIXED_CG_RUNGS");
    const std::set<int> grid_mixed_hb_rungs = parse_grid_rung_list("HASEN_GRID_MIXED_CG_HEATBATH_RUNGS");
    // The driver's validations.
    for (int k : grid_mg_hb_rungs)
      if (!grid_mg_listed.count(k)) {
        std::cerr << "HASEN_GRID_MG_HEATBATH_RUNGS: rung " << k << " is not in HASEN_GRID_MG_RUNGS.\n";
        exit(1);
      }
    for (int k : grid_mixed_rungs)
      if (grid_mg_listed.count(k)) {
        std::cerr << "HASEN_GRID_MIXED_CG_RUNGS: rung " << k << " is also in HASEN_GRID_MG_RUNGS.\n";
        exit(1);
      }
    for (int k : grid_mixed_hb_rungs)
      if (grid_mg_hb_rungs.count(k)) {
        std::cerr << "HASEN_GRID_MIXED_CG_HEATBATH_RUNGS: rung " << k
                  << " is also in HASEN_GRID_MG_HEATBATH_RUNGS.\n";
        exit(1);
      }
    // FORCES_ONLY-style de-routing of the unselected rungs (driver).
    for (int k : fo_skip) { grid_mg_rungs.erase(k); grid_mg_hb_rungs.erase(k); }
    if (!grid_mg_rungs.empty()) {
      const int k0 = *grid_mg_rungs.begin();
      grid_mg_donor = k0;
      GridMGH = std::make_unique<GridMGHierarchy>(light_opF(k0), &GridF, &RBGridF, Odd,
                                                  GridMGParams::from_env(),
                                                  "PF" + std::to_string(k0) + " hierarchy");
      for (int k : grid_mg_rungs)
        GridMGRungSolver[k] = std::make_unique<GridMGSchurSolver>(
            *GridMGH, cg_tol_drv, Odd, "PF" + std::to_string(k), k == k0);
      std::cout << GridLogMessage << "[Ladder] Grid-MG hierarchy: donor rung " << k0
                << " mass=" << ladder[k0] << " cb=Odd fp32, outer fp64 FGCR tol=" << cg_tol_drv
                << " | " << GridMGH->Params().Summary() << std::endl;
    }
    for (int k : grid_mg_hb_rungs) {
      GRID_ASSERT(GridMGH && grid_mg_rungs.count(k));
      GridMGHeatbathSolver[k] = std::make_unique<GridMGSchurSolver>(
          *GridMGH, cg_tol_act, Odd, "PF" + std::to_string(k) + " heatbath", /*donor=*/false);
      GridMGHeatbathSolver[k]->ImportIfHierarchyStale();
    }
    for (int k : grid_mixed_rungs)
      if (!fo_skip.count(k))
        GridMixedRungSolver[k] = std::make_unique<MixedPrecCGRungSolver>(
            light_op(k), light_opF(k), &RBGridF, cg_tol_drv, cg_max, 50,
            "PF" + std::to_string(k) + " deriv/S");
    for (int k : grid_mixed_hb_rungs)
      if (!fo_skip.count(k))
        GridMixedHeatbathSolver[k] = std::make_unique<MixedPrecCGRungSolver>(
            light_op(k + 1), light_opF(k + 1), &RBGridF, cg_tol_act, cg_max, 50,
            "PF" + std::to_string(k) + " heatbath");
  }
  for (int k = 0; k < n_pf; ++k) {
    if (fo_skip.count(k)) continue;
    // Driver: M3 routes (all null when unset).
    RungSolverBase *ds_rung =
        GridMGRungSolver[k] ? static_cast<RungSolverBase *>(GridMGRungSolver[k].get())
                            : static_cast<RungSolverBase *>(GridMixedRungSolver[k].get());
    RungSolverBase *hb_rung =
        GridMGHeatbathSolver[k]
            ? static_cast<RungSolverBase *>(GridMGHeatbathSolver[k].get())
            : static_cast<RungSolverBase *>(GridMixedHeatbathSolver[k].get());
    OperatorFunction<LatticeFermion> &hb_solver =
        hb_rung ? static_cast<OperatorFunction<LatticeFermion> &>(*hb_rung)
                : static_cast<OperatorFunction<LatticeFermion> &>(CG_action);
    std::string route;
    if (ds_rung) {
      RatioPF[k] = std::make_unique<TwoFlavourSchurCloverRatioActionRungSolver<WilsonImplR, WCF>>(
          light_op(k + 1), light_op(k), *ds_rung, hb_solver);
      route = GridMGRungSolver[k] ? (k == grid_mg_donor ? "Grid-MG donor" : "Grid-MG shared")
                                  : "Grid mixed-precision CG";
    } else if (hb_rung) {
      RatioPF[k] = std::make_unique<TwoFlavourSchurCloverRatioActionHeatbathRung<WilsonImplR, WCF>>(
          light_op(k + 1), light_op(k), CG_deriv, CG_action, *hb_rung);
      route = "Grid-CG";
    } else {
      RatioPF[k] = std::make_unique<TwoFlavourSchurCloverRatioAction<WilsonImplR, WCF>>(
          light_op(k + 1), light_op(k), CG_deriv, CG_action);
      route = "Grid-CG";
    }
    RatioPF[k]->is_smeared = true;
    std::cout << GridLogMessage << "[Ladder] rung " << k << " DerivativeSolver/ActionSolver = "
              << route << ", HeatbathSolver = "
              << (GridMGHeatbathSolver[k] ? "Grid-MG shared"
                  : GridMixedHeatbathSolver[k] ? "Grid mixed-precision CG" : "Grid CG")
              << ", mass=" << ladder[k] << " / " << ladder[k + 1] << std::endl;
  }

  // ── Tail (driver: TwoFlavourSchurCloverActionMP, CG_light_md + CG_action) ──
  std::unique_ptr<SchurDifferentiableOperator<WilsonImplR>> LightTailSchurD;
  std::unique_ptr<SchurDifferentiableOperator<WilsonImplF>> LightTailSchurF;
  typedef MixedPrecCGWrapper<LatticeFermion, LatticeFermionF,
                             SchurDifferentiableOperator<WilsonImplR>,
                             SchurDifferentiableOperator<WilsonImplF>> TailMDSolver;
  std::unique_ptr<TailMDSolver> CG_light_md;
  std::unique_ptr<TwoFlavourSchurCloverActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>> TailGrid;
  if (selected("Tail")) {
    LightTailSchurD = std::make_unique<SchurDifferentiableOperator<WilsonImplR>>(light_op(itail));
    LightTailSchurF = std::make_unique<SchurDifferentiableOperator<WilsonImplF>>(light_opF(itail));
    CG_light_md = std::make_unique<TailMDSolver>(cg_tol_drv, cg_max, 50, &RBGridF,
                                                 *LightTailSchurD, *LightTailSchurF);
    TailGrid = std::make_unique<
        TwoFlavourSchurCloverActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>>(
        light_op(itail), light_opF(itail), *CG_light_md, CG_action);
    TailGrid->is_smeared = true;
  }

  // ── Strange log-det (driver's automatic parity pairing) ───────────────────
  const bool strange_even_pf = std::getenv("STRANGE_EVEN") != nullptr;
  const bool strange_logdet_odd = [strange_even_pf] {
    const char *e = std::getenv("HASEN_GRID_STRANGE_LOGDET_ODD");
    if (e && e[0] == '1') return true;
    if (e && e[0] == '0') return false;
    return strange_even_pf;
  }();
  std::unique_ptr<QCDLogDetCompactCloverEOAction<WilsonImplR>> StrangeLogDet;
  if (selected("StrangeLogDet")) {
    StrangeLogDet = std::make_unique<QCDLogDetCompactCloverEOAction<WilsonImplR>>(
        strange_op(), 1, strange_logdet_odd ? Odd : Even);
    StrangeLogDet->is_smeared = true;
    std::cout << GridLogMessage << "[StrangeLogDet] block = " << (strange_logdet_odd ? "M_oo" : "M_ee")
              << " (strange pseudofermion on " << (strange_even_pf ? "EVEN" : "ODD") << " sites)"
              << std::endl;
  }

  // ── Strange RHMC (driver) ─────────────────────────────────────────────────
  const RealD rat_lo = TXQCDProduction::detail::env_real("RAT_LO", 1e-4);
  const RealD rat_hi = TXQCDProduction::detail::env_real("RAT_HI", 100.0);
  int rat_degree = 20;
  if (const char *rd = std::getenv("RAT_DEGREE"); rd && *rd) rat_degree = std::atoi(rd);
  std::cout << GridLogMessage << "Strange rational: lo=" << rat_lo << " hi=" << rat_hi
            << " degree=" << rat_degree << std::endl;
  // Driver (2026-10-05): separate force degree / tolerance and the refresh-time bounds check,
  // same variables, defaults and parsing (RAT_DEGREE_MD, TUNE_CG_TOL_STRANGE_MD,
  // RAT_BOUNDS_CHECK_FREQ, RAT_BOUNDS_CHECK_TOL, RAT_BOUNDS_CHECK_ABORT).
  const int rat_degree_md = TXQCDProduction::detail::env_int("RAT_DEGREE_MD", rat_degree);
  const RealD cg_tol_strange_md =
      TXQCDProduction::detail::env_real("TUNE_CG_TOL_STRANGE_MD", cg_tol_strange);
  const int rat_bc_freq = TXQCDProduction::detail::env_int("RAT_BOUNDS_CHECK_FREQ", 0);
  const RealD rat_bc_tol = TXQCDProduction::detail::env_real("RAT_BOUNDS_CHECK_TOL", 1e-6);
  bool rat_bc_abort = false;
  if (const char *v = std::getenv("RAT_BOUNDS_CHECK_ABORT"); v && *v) {
    const std::string s(v);
    if (s == "1") rat_bc_abort = true;
    else if (s != "0") {
      std::cerr << "RAT_BOUNDS_CHECK_ABORT=" << v << ": must be 0 or 1.\n";
      exit(1);
    }
  }
  if (rat_degree_md < 1 || rat_bc_freq < 0 || !(cg_tol_strange_md > 0.0) || !(rat_bc_tol > 0.0)) {
    std::cerr << "RAT_DEGREE_MD=" << rat_degree_md << " RAT_BOUNDS_CHECK_FREQ=" << rat_bc_freq
              << " TUNE_CG_TOL_STRANGE_MD=" << cg_tol_strange_md << " RAT_BOUNDS_CHECK_TOL="
              << rat_bc_tol << ": need degree >= 1, freq >= 0, tolerances > 0.\n";
    exit(1);
  }
  std::cout << GridLogMessage << "Strange rational: degree_action=" << rat_degree
            << " degree_md=" << rat_degree_md << " tol_action=" << cg_tol_strange
            << " tol_md=" << cg_tol_strange_md << " bounds_check_freq=" << rat_bc_freq
            << " bounds_check_tol=" << rat_bc_tol << " bounds_check_abort=" << rat_bc_abort
            << std::endl;
  OneFlavourSchurRationalExtras strange_x;
  strange_x.md_degree = rat_degree_md;
  strange_x.md_tolerance = cg_tol_strange_md;
  strange_x.bounds_check_freq = rat_bc_freq;
  strange_x.bounds_check_tol = rat_bc_tol;
  strange_x.bounds_check_abort = rat_bc_abort;
  OneFlavourRationalParams strange_rat(rat_lo, rat_hi, cg_max, cg_tol_strange, rat_degree, 64, 100, 1e-6, 1e-4);
  std::unique_ptr<OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>> StrangeBase;
  std::unique_ptr<OneFlavourSchurCloverRationalActionEven<WilsonImplR, WCF>> StrangeEven;
  Action<LatticeGaugeField> *StrangePtr = nullptr;
  if (selected("Strange")) {
    MultiShiftFunction *neg_half = nullptr;
    if (strange_even_pf) {
      StrangeEven = std::make_unique<OneFlavourSchurCloverRationalActionEven<WilsonImplR, WCF>>(
          strange_op(), strange_rat, strange_x);
      StrangeEven->is_smeared = true;
      StrangePtr = StrangeEven.get();
      neg_half = &StrangeEven->PowerNegHalf;
      std::cout << GridLogMessage << "[Strange] STRANGE_EVEN active — Grid native EVEN-parity Schur"
                << std::endl;
    } else {
      StrangeBase = std::make_unique<
          OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>>(
          strange_op(), strange_opF(), &RBGridF, strange_rat, 50, strange_x);
      StrangeBase->is_smeared = true;
      StrangePtr = StrangeBase.get();
      neg_half = &StrangeBase->PowerNegHalf;
    }
    if (const char *t = std::getenv("FD_TOL_STRANGE_DERIV"); t && *t) {
      const RealD td = std::atof(t);
      for (auto &x : neg_half->tolerances) x = td;
      std::cout << GridLogMessage << "[Strange] FD_TOL_STRANGE_DERIV: force multishift tolerance "
                << td << " (S and heatbath keep " << cg_tol_strange << ")" << std::endl;
    }
  }

  // ── LW gauge action (driver) ──────────────────────────────────────────────
  const RealD u0_ens = TXQCDProduction::detail::env_real("U0", 1.0);
  PlaqPlusRectangleAction<PeriodicGimplR> GaugeAction(beta, -beta / (20.0 * u0_ens * u0_ens));
  GaugeAction.is_smeared = false;
  std::cout << GridLogMessage << std::setprecision(15)
            << "[Action] BETA=" << beta << " U0=" << u0_ens
            << " rect_coeff=" << -beta / (20.0 * u0_ens * u0_ens)
            << " CSW=" << csw << " MASS_LIGHT=" << mass_light << " MASS_STRANGE=" << mass_strange
            << " STOUT_RHO=" << stout_rho_inv << " STOUT_NSMEAR=" << stout_nsmear_inv
            << std::setprecision(6) << std::endl;

  // ── Stout smearing (driver) ───────────────────────────────────────────────
  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear_inv, Stout);
  Smear.set_Field(Umu);

  // ── Monomials, in test order (gauge first: the normalisation check) ────────
  std::vector<Mono> monos;
  if (selected("Gauge")) monos.push_back({"Gauge", &GaugeAction, false, {}});
  if (LightLogDet) monos.push_back({"LightLogDet", LightLogDet.get(), true, {}});
  if (StrangeLogDet) monos.push_back({"StrangeLogDet", StrangeLogDet.get(), true, {}});
  for (int k = 0; k < n_pf; ++k) {
    if (!RatioPF[k]) continue;
    Mono m{"PF" + std::to_string(k), RatioPF[k].get(), true, {}};
    // A Grid-MG sharer's preconditioner follows the donor's gauge import; the integrator always
    // evaluates the donor first at each U.  Here each monomial is evaluated alone, so the donor's
    // import of the field about to be used is replayed first (SetGauge = fp32 thin update only).
    if (GridMGRungSolver[k] && k != grid_mg_donor) {
      GridMGSchurSolver *donor = GridMGRungSolver[grid_mg_donor].get();
      m.pre = [donor](const LatticeGaugeField &U) { donor->SetGauge(U); };
    }
    monos.push_back(m);
  }
  if (TailGrid) monos.push_back({"Tail", TailGrid.get(), true, {}});
  if (StrangePtr) monos.push_back({"Strange", StrangePtr, true, {}});

  // ── Directions: HMC momenta from their own RNG (same for every monomial) ───
  std::vector<LatticeGaugeField> dirs(ndir, LatticeGaugeField(&Grid_));
  {
    GridSerialRNG sRNG_P;
    GridParallelRNG pRNG_P(&Grid_);
    sRNG_P.SeedFixedIntegers({fd_seed, fd_seed + 1, fd_seed + 2, fd_seed + 3});
    pRNG_P.SeedFixedIntegers({fd_seed + 4, fd_seed + 5, fd_seed + 6, fd_seed + 7});
    for (int d = 0; d < ndir; ++d) {
      PeriodicGimplR::generate_momenta(dirs[d], sRNG_P, pRNG_P);
      std::cout << GridLogMessage << "[FD] direction " << d << ": -sum tr(P^2)/sites = "
                << -PeriodicGimplR::FieldSquareNorm(dirs[d]) / Grid_.gSites() << std::endl;
    }
  }

  LatticeGaugeField Upert(&Grid_);
  LatticeGaugeField F(&Grid_);
  std::vector<std::string> summaries;
  const std::string switches = [] {
    std::ostringstream o;
    for (const char *v : {"HASEN_GRID_FUSED_CLOVER_FORCE", "HASEN_GRID_DEVICE_CB",
                          "HASEN_GRID_BATCH_SMEAR", "HASEN_GRID_IMPORT_SKIP",
                          "HASEN_GRID_SHARE_FIELDSTRENGTH", "HASEN_GRID_GPU_CLOVER_INV",
                          "HASEN_GRID_CLOVER_STENCIL"}) {
      const char *e = std::getenv(v);
      o << ((e && e[0] == '1') ? '1' : '0');
    }
    return o.str();
  }();

  const double t_start = usecond();
  for (auto &m : monos) {
    std::vector<bool> modes;
    if (!m.fermion || links == "smeared") modes = {true};
    else if (links == "thin") modes = {false};
    else modes = {true, false};
    for (bool use_smeared : modes) {
      Action<LatticeGaugeField> *act = m.act;
      const bool prod_smeared = act->is_smeared;
      if (m.fermion) act->is_smeared = use_smeared;
      const std::string tag = m.name + (act->is_smeared ? "/smeared" : "/thin");
      auto sync = [&]() { if (m.pre) m.pre(Smear.get_U(act->is_smeared)); };

      std::cout << GridLogMessage << "==== FD " << tag << " : " << act->action_name() << std::endl;
      double t0 = usecond();
      Smear.set_Field(Umu);
      sync();
      act->refresh(Smear, sRNG, pRNG);  // pseudofermion at U0, then frozen
      const double t_ref = (usecond() - t0) / 1e6;
      t0 = usecond();
      sync();
      const RealD S0 = act->S(Smear);
      const double t_S = (usecond() - t0) / 1e6;
      t0 = usecond();
      sync();
      integrator_force(act, Smear, F, batch);
      const double t_F = (usecond() - t0) / 1e6;
      const RealD f_avg = std::sqrt(norm2(F) / Grid_.gSites());
      const RealD f_max = std::sqrt(maxLocalNorm2(F));
      std::cout << GridLogMessage << std::setprecision(16) << "FDBASE " << tag << " S0 = " << S0
                << " force avg " << f_avg << " max " << f_max << std::setprecision(6)
                << " | refresh " << t_ref << " s, S " << t_S << " s, force " << t_F << " s"
                << std::endl;

      std::ostringstream summ;
      summ << "FDSUMMARY " << tag << " switches=" << switches;
      for (int d = 0; d < ndir; ++d) {
        LatticeGaugeField &P = dirs[d];
        const RealD pred = 2.0 * real(innerProduct(P, F));
        RealD pred2 = 0.0;
        for (int mu = 0; mu < Nd; ++mu) {
          auto Pmu = PeekIndex<LorentzIndex>(P, mu);
          auto Fmu = PeekIndex<LorentzIndex>(F, mu);
          pred2 += -2.0 * real(TensorRemove(sum(trace(Pmu * Fmu))));
        }
        std::cout << GridLogMessage << std::setprecision(16) << "FDPRED " << tag << " dir " << d
                  << " pred = 2 Re<P,F> = " << pred << "  (-2 Re sum tr(P F) = " << pred2
                  << ", rel diff " << sci((pred2 - pred) / std::fabs(pred), 2) << ")"
                  << std::setprecision(6) << std::endl;
        std::vector<RealD> fds;
        summ << " dir" << d << ":pred=" << sci(pred, 9);
        for (RealD eps : eps_list) {
          double te = usecond();
          Upert = Umu;
          PeriodicGimplR::update_field(P, Upert, +eps);
          Smear.set_Field(Upert);
          sync();
          const RealD Sp = act->S(Smear);
          Upert = Umu;
          PeriodicGimplR::update_field(P, Upert, -eps);
          Smear.set_Field(Upert);
          sync();
          const RealD Sm = act->S(Smear);
          const RealD fd = (Sp - Sm) / (2.0 * eps);
          const RealD rel = (fd - pred) / std::fabs(pred);
          fds.push_back(fd);
          std::cout << GridLogMessage << std::setprecision(16) << "FD " << tag << " dir " << d
                    << " eps " << sci(eps, 2) << " S+ " << Sp << " S- " << Sm << " S+-S- "
                    << (Sp - Sm) << " fd " << fd << " pred " << pred << " rel "
                    << sci(rel, 4) << std::setprecision(6) << " (" << (usecond() - te) / 1e6
                    << " s)" << std::endl;
          summ << " rel(" << sci(eps, 1) << ")=" << sci(rel, 3);
        }
        for (size_t i = 0; i + 1 < fds.size(); ++i) {
          const RealD r = eps_list[i] / eps_list[i + 1];
          const RealD rich = (r * r * fds[i + 1] - fds[i]) / (r * r - 1.0);
          const RealD relr = (rich - pred) / std::fabs(pred);
          std::cout << GridLogMessage << "FDRICH " << tag << " dir " << d << " eps "
                    << sci(eps_list[i], 2) << "+" << sci(eps_list[i + 1], 2) << " rel "
                    << sci(relr, 4) << std::endl;
          summ << " rich(" << sci(eps_list[i], 1) << "," << sci(eps_list[i + 1], 1)
               << ")=" << sci(relr, 3);
        }
      }
      Smear.set_Field(Umu);
      act->is_smeared = prod_smeared;
      summaries.push_back(summ.str());
      std::cout << GridLogMessage << "TEST RESULT " << summ.str() << std::endl;
      std::cout << GridLogMessage << "[FD] " << tag << " done, elapsed total "
                << (usecond() - t_start) / 1e6 << " s" << std::endl;
    }
  }

  std::cout << GridLogMessage << "==== FD summary (rel = (FD - pred)/|pred|; mismatch = flat in eps)"
            << std::endl;
  for (auto &s : summaries) std::cout << GridLogMessage << "TEST RESULT " << s << std::endl;
  Grid_finalize();
  return 0;
}
