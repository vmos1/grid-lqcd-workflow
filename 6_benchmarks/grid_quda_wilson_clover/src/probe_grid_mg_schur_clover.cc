// probe_grid_mg_schur_clover.cc
//
// Phase 0 probe for the MG arm of the Grid-vs-QUDA campaign.
// Design doc: __docs/2026_09_10_mg_comparison_project_goals.md
//
// Deliberately Grid-only -- it includes ONLY <Grid/Grid.h> and links no QUDA --
// so every failure localises here rather than inside the 63k-line benchmark with
// a live QUDA session attached.
//
// WHAT THIS ANSWERS. Three facts the doc needs that cannot be settled by reading
// source or by a compile-only probe, because all three are RUNTIME aborts:
//
//   1. Does Grid's general-coarsening MG run at all on a 4D red-black compact
//      clover Schur operator? No test in Grid's tree exercises that combination
//      (they are 5D/DWF, or 4D on the FULL operator).
//   2. Does the coarse grid actually divide the red-black grid at our
//      decompositions? (`subdivides()` asserts on _processors, _simd_layout and
//      _rdimensions.) Doc §4 works this out on paper; this checks it.
//   3. Is the coarse operator really the Galerkin projection of the operator we
//      think we coarsened? Getting this wrong yields a valid-looking, meaningless
//      coarse space with NO error -- see the Galerkin check below.
//
// WHICH OPERATOR IS COARSENED, AND WHY IT IS *NOT* Mpc^dag Mpc.
// Multigrid preconditions Mpc directly and never squares it; squaring is a
// CG-specific device for obtaining an HPD operator, and a squared operator is the
// wrong thing to hand a multigrid. GeneralCoarsenedMatrix::CoarsenOperator calls
// linop.Op(), and for a Schur operator Op() IS Mpc
// (Grid/algorithms/LinearOperator.h:340) -- so SchurDiagMooeeOperator is passed
// straight in with no wrapper. This is the opposite of what an earlier revision of
// the design doc said; see doc §2 and §6. The `HermOpAdaptor` those revisions
// called mandatory belongs to a CG-type (HPD) outer solver such as TwoLevelADEF2,
// which is not what this path uses.
//
// EVEN CHECKERBOARD. CoarsenOperator builds `FineComplexField one(grid)` with no
// checkerboard set, so it defaults to Even (Lattice_base.h:268); blockZAXPY
// propagates that to the phase field (Lattice_transfer.h:346); and the subsequent
// `phaV = phaF[p]*V.subspace[i]` goes through CBFromExpression (Lattice_ET.h:248),
// which hard-asserts that every leaf shares a checkerboard. An Odd subspace
// therefore ABORTS. --probe-checkerboard odd exists precisely to demonstrate that,
// so the restriction is recorded as observed behaviour rather than a source
// reading.

#include <Grid/Grid.h>

// Probe-local FlexibleGCR (Grid's PrecGCRNonHermitian minus four unnecessary costs) and the
// operator-application counters. See __docs/2026_09_15_grid_mg_fix_plan.md.
#include "bench_nvtx.h"
#include "probe_mg_solvers.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace Grid;

namespace {

// Number of null-space vectors generated. COMPILE-TIME: it is a template argument
// shared by Aggregation and the coarse matrix, so changing it needs a rebuild.
//
// 24 matches QUDA once chiral doubling is counted on both sides. Grid generates
// nbasis vectors and doubles them with gamma5 into a 2*nbasis aggregation => 48
// coarse dof per block. QUDA reaches the same 48 via n_vec=24 with
// spin_block_size[0]=2. Compare 48 against 48, never 24 against 24.
constexpr int kNbasis = 24;

// ---------------------------------------------------------------------------
// Local helpers. Neither is in the Grid library.
// ---------------------------------------------------------------------------

// Shifted Schur operator, for the smoother.
//
// Grid's ShiftedNonHermitianLinearOperator cannot be used here: it is templated on
// a *Matrix* and calls _Mat.M()/_Mat.Mdag(), whereas a Schur operator is a
// LinearOperatorBase exposing Mpc()/MpcDag(). This is the direct analogue --
// Op() = Mpc + shift, AdjOp() = MpcDag + shift.
//
// OpDiag/OpDir/OpDirAll assert: they exist only for the OLD CoarsenedMatrix
// coarsening path, which cannot take a Schur operator at all (doc §6). Nothing on
// the GeneralCoarsenedMatrix path calls them.
template <class Field>
class ShiftedSchurOperator : public LinearOperatorBase<Field> {
  SchurOperatorBase<Field> &schur_;
  RealD shift_;

public:
  ShiftedSchurOperator(SchurOperatorBase<Field> &schur, RealD shift) : schur_(schur), shift_(shift) {}

  void Op(const Field &in, Field &out) override
  {
    schur_.Mpc(in, out);
    out = out + shift_ * in;
  }
  void AdjOp(const Field &in, Field &out) override
  {
    schur_.MpcDag(in, out);
    out = out + shift_ * in;
  }
  void OpDiag(const Field &in, Field &out) override { GRID_ASSERT(0); }
  void OpDir(const Field &in, Field &out, int dir, int disp) override { GRID_ASSERT(0); }
  void OpDirAll(const Field &in, std::vector<Field> &out) override { GRID_ASSERT(0); }
  void HermOpAndNorm(const Field &in, Field &out, RealD &n1, RealD &n2) override { GRID_ASSERT(0); }
  void HermOp(const Field &in, Field &out) override { GRID_ASSERT(0); }
};

// Precision-changing adaptor: wraps a SINGLE-precision LinearFunction so it can be
// used as a preconditioner by a DOUBLE-precision outer solver.
//
// WHY THIS SHAPE, AND NOT AN ALL-SINGLE SOLVE. The probe's gate is an independent
// residual at 1e-10, which fp32 (~1e-7 relative) cannot reach. The standard remedy
// -- and what QUDA does -- is to keep the OUTER Krylov and all gates in double and
// run only the PRECONDITIONER in reduced precision, where the arithmetic merely has
// to be approximately right. Grid's own precedent is
// tests/solver/Test_wilson_mg_mp.cc:98,106: an all-single MG hierarchy driven by a
// MixedPrecisionFlexibleGeneralisedMinimalResidual<LatticeFermionD,LatticeFermionF>.
//
// This is the CONFIGURATION THAT MATCHES QUDA, which runs a single-precision
// preconditioner (cuda_prec_precondition) with half null vectors under a double
// outer GCR. It is NOT the "single coarse operator under a double fine operator"
// hybrid -- that one additionally needs vComplexD2 fine fields to make SIMD lane
// counts match (Lattice_transfer.h:41 subdivides()) and is a much larger change.
//
// precisionChange is Grid's supported double<->single conversion
// (Lattice_transfer.h:1377) and is the same primitive its mixed-precision CG uses.
template <class FieldD, class FieldF>
class PrecisionChangeAdaptor : public LinearFunction<FieldD> {
public:
  using LinearFunction<FieldD>::operator();

private:
  LinearFunction<FieldF> &inner_;
  GridBase *fine_f_;

public:
  PrecisionChangeAdaptor(LinearFunction<FieldF> &inner, GridBase *fine_f)
      : inner_(inner), fine_f_(fine_f)
  {
  }

  void operator()(const FieldD &in, FieldD &out) override
  {
    FieldF in_f(fine_f_);
    FieldF out_f(fine_f_);
    in_f.Checkerboard() = in.Checkerboard();
    out_f.Checkerboard() = in.Checkerboard();
    precisionChange(in_f, in);
    out_f = Zero();
    out_f.Checkerboard() = in.Checkerboard();
    inner_(in_f, out_f);
    precisionChange(out, out_f);
    out.Checkerboard() = in.Checkerboard();
  }
};

// Two-level MG preconditioner, copied from Grid's own
// tests/debug/Test_general_coarse_wilson.cc (the class lives in the test file, not
// the library). Structure is unchanged; the only edit is a `quiet` flag, because
// the original prints five timing lines per application and the outer PGCR applies
// it hundreds of times.
//
// NOTE this is a *preconditioner* (a LinearFunction), unlike TwoLevelADEF2 which
// is itself the outer solver. It is driven by
// PrecGeneralisedConjugateResidualNonHermitian below.
template <class Fobj, class CComplex, int nbasis>
class MGPreconditioner : public LinearFunction<Lattice<Fobj>> {
public:
  using LinearFunction<Lattice<Fobj>>::operator();

  typedef Aggregation<Fobj, CComplex, nbasis> Aggregates;
  typedef typename Aggregation<Fobj, CComplex, nbasis>::FineField FineField;
  typedef typename Aggregation<Fobj, CComplex, nbasis>::CoarseVector CoarseVector;
  typedef LinearOperatorBase<FineField> FineOperator;
  typedef LinearFunction<FineField> FineSmoother;
  typedef LinearOperatorBase<CoarseVector> CoarseOperator;
  typedef LinearFunction<CoarseVector> CoarseSolver;

  Aggregates &_Aggregates;
  FineOperator &_FineOperator;
  FineSmoother &_PreSmoother;
  FineSmoother &_PostSmoother;
  CoarseOperator &_CoarseOperator;
  CoarseSolver &_CoarseSolve;
  bool quiet;

  // Stage 1 switches (defaults reproduce the original behaviour).
  //
  // fast_project: use blockProjectFast instead of Aggregation::ProjectToSubspace.
  //   blockProject (Lattice_transfer.h:267) does, per basis vector, BOTH a blockInnerProductD
  //   and a blockZAXPY that subtracts the projection back out of a working copy, plus a
  //   whole-field copy at entry -- about 2.5x the memory traffic of blockProjectFast
  //   (Lattice_transfer.h:1796). The subtraction is a stabilisation for a NON-orthonormal
  //   basis and CANNOT change the result here: CoarsenOperator block-orthogonalises the
  //   subspace in place (GeneralCoarsenedMatrix.h:322), which is the same fact the Galerkin
  //   check already relies on. Measured cost of the difference: ~11.5 ms of a 145.7 ms V-cycle.
  //   ⛔ Only valid AFTER CoarsenOperator has run. Do not move this call earlier.
  bool fast_project = false;
  // D7: 0/1 select stock ProjectToSubspace / blockProjectFast; 2 selects the fused
  // projector in probe_mg_solvers.h. See that file for why.
  int project_mode = 0;
  // Gate on the per-region accelerator_barrier(). OFF by default -- see the long
  // comment at the tick() lambda in operator().
  bool instrument = false;
  // persistent_temps: hoist the four per-application fields out of the V-cycle (D4). At C3 a
  //   fine rb vector is ~64 MB and a coarse one ~2 MB, and field destruction costs ~0.3 ms
  //   EACH regardless of size.
  bool persistent_temps = false;

  int level;
  void Level(int lv) { level = lv; }

  MGPreconditioner(Aggregates &Agg, FineOperator &Fine, FineSmoother &PreSmoother,
                   FineSmoother &PostSmoother, CoarseOperator &CoarseOperator_,
                   CoarseSolver &CoarseSolve_, bool quiet_ = true)
      : _Aggregates(Agg), _FineOperator(Fine), _PreSmoother(PreSmoother),
        _PostSmoother(PostSmoother), _CoarseOperator(CoarseOperator_),
        _CoarseSolve(CoarseSolve_), quiet(quiet_), level(1)
  {
  }

  // Accumulated per-stage cost, summed over every application. Reported once at
  // the end rather than printed per call -- the outer PGCR applies this ~17 times,
  // and 85 scattered lines are harder to read than one table.
  //
  // ⚠️ Each timestamp is preceded by accelerator_barrier(), without which Grid's
  // asynchronous dispatch would attribute time to whichever stage happens to
  // synchronise. That barrier is itself a perturbation, so these numbers are a
  // DIAGNOSTIC BREAKDOWN, not a measurement: the total here will exceed the
  // unbarriered solve time, and the split is what to read, not the absolute values.
  mutable double t_presmooth = 0.0, t_fineop = 0.0, t_project = 0.0;
  mutable double t_coarse = 0.0, t_promote = 0.0, t_postsmooth = 0.0;
  mutable long long n_apply = 0;

  void report_breakdown() const
  {
    const double total = t_presmooth + t_fineop + t_project + t_coarse + t_promote + t_postsmooth;
    if (total <= 0.0) return;
    auto line = [&](const char *name, double t) {
      std::cout << GridLogMessage << "    " << name << " " << (t / 1.0e6) << " s  ("
                << (100.0 * t / total) << "%)" << std::endl;
    };
    std::cout << GridLogMessage << "  MG preconditioner breakdown over " << n_apply
              << " applications (barriered; diagnostic only):" << std::endl;
    line("pre-smoother ", t_presmooth);
    line("fine operator", t_fineop);
    line("project      ", t_project);
    line("coarse solve ", t_coarse);
    line("promote      ", t_promote);
    line("post-smoother", t_postsmooth);
    std::cout << GridLogMessage << "    barriered total " << (total / 1.0e6) << " s" << std::endl;
  }

  // Persistent V-cycle temporaries, created on first application (D4).
  std::unique_ptr<CoarseVector> pCsrc, pCsol;
  std::unique_ptr<FineField> pvec1, pvec2;

  virtual void operator()(const FineField &in, FineField &out)
  {
    GridBase *CoarseGrid = _Aggregates.CoarseGrid;
    // ⛔ Only allocate the persistent set when it will actually be used. Allocating it in the
    // control arm too would hold ~130 MB of fine fields per rank for nothing, and the fp32
    // path at C3 already fails to fit (results §11: it needs DEVICE_MEM_MB=8000 because both
    // hierarchies are resident).
    if (persistent_temps && !pCsrc) {
      pCsrc.reset(new CoarseVector(CoarseGrid));
      pCsol.reset(new CoarseVector(CoarseGrid));
      pvec1.reset(new FineField(in.Grid()));
      pvec2.reset(new FineField(in.Grid()));
    }
    // Control arm: allocate and destroy per application, exactly as the original did.
    std::unique_ptr<CoarseVector> tCsrc, tCsol;
    std::unique_ptr<FineField> tvec1, tvec2;
    if (!persistent_temps) {
      tCsrc.reset(new CoarseVector(CoarseGrid));
      tCsol.reset(new CoarseVector(CoarseGrid));
      tvec1.reset(new FineField(in.Grid()));
      tvec2.reset(new FineField(in.Grid()));
    }
    CoarseVector &Csrc = persistent_temps ? *pCsrc : *tCsrc;
    CoarseVector &Csol = persistent_temps ? *pCsol : *tCsol;
    FineField &vec1 = persistent_temps ? *pvec1 : *tvec1;
    FineField &vec2 = persistent_temps ? *pvec2 : *tvec2;
    vec1.Checkerboard() = in.Checkerboard();
    vec2.Checkerboard() = in.Checkerboard();

    // ⛔⛔ THIS BARRIER USED TO BE UNCONDITIONAL. There are 14 tick() calls per
    // V-cycle, each a full accelerator_barrier() (device synchronise), so every
    // Grid MG timing this campaign ever produced was measured with ~14 forced
    // syncs per V-cycle inside the timed region -- while the QUDA probe times its
    // solve only at the outer boundary and has NO internal barriers. That is not
    // an apples-to-apples comparison, and the sub-timers sum to only ~62-67% of
    // the solve's wall time at every volume tested, which is the tell.
    //
    // Default is now OFF: no barriers, no breakdown. --probe-instrument-vcycle 1
    // restores the breakdown, and its total must not be compared against an
    // uninstrumented run's.
    const bool instr = instrument;
    auto tick = [instr]() -> double {
      if (!instr) return 0.0;
      accelerator_barrier();
      return usecond();
    };
    double t0;
    ++n_apply;

    out = Zero();
    out.Checkerboard() = in.Checkerboard();
    t0 = tick();
    _PreSmoother(in, out);
    t_presmooth += tick() - t0;

    t0 = tick();
    _FineOperator.Op(out, vec1);
    sub(vec1, in, vec1);
    t_fineop += tick() - t0;

    t0 = tick();
    if (project_mode == 2)
      ProbeMG::blockProjectFused(Csrc, vec1, _Aggregates.subspace);
    else if (fast_project)
      blockProjectFast(Csrc, vec1, _Aggregates.subspace);
    else
      _Aggregates.ProjectToSubspace(Csrc, vec1);
    t_project += tick() - t0;

    t0 = tick();
    Csol = Zero();
    _CoarseSolve(Csrc, Csol);
    t_coarse += tick() - t0;

    t0 = tick();
    _Aggregates.PromoteFromSubspace(Csol, vec1);
    add(out, out, vec1);
    t_promote += tick() - t0;

    t0 = tick();
    _FineOperator.Op(out, vec1);
    sub(vec1, in, vec1);
    t_fineop += tick() - t0;

    t0 = tick();
    vec2 = Zero();
    vec2.Checkerboard() = in.Checkerboard();
    _PostSmoother(vec1, vec2);
    t_postsmooth += tick() - t0;

    add(out, out, vec2);
  }
};

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

std::string read_string(int argc, char **argv, const std::string &option, const std::string &fallback)
{
  char **itr = std::find(argv, argv + argc, option);
  if (itr != argv + argc && ++itr != argv + argc) return std::string(*itr);
  return fallback;
}

double read_double(int argc, char **argv, const std::string &option, double fallback)
{
  const std::string payload = read_string(argc, argv, option, "");
  return payload.empty() ? fallback : std::stod(payload);
}

int read_int(int argc, char **argv, const std::string &option, int fallback)
{
  const std::string payload = read_string(argc, argv, option, "");
  return payload.empty() ? fallback : std::stoi(payload);
}

// Median of the timed repeats, matching what the QUDA probe reports
// (probe_quda_mg_clover.cc:385). With an odd count the median discards a single
// slow first solve, which is why it is preferred to the mean here.
double median_of(std::vector<double> v)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

Coordinate parse_block(const std::string &spec)
{
  Coordinate block(4);
  std::size_t pos = 0;
  for (int d = 0; d < 4; ++d) {
    const std::size_t dot = spec.find('.', pos);
    block[d] = std::stoi(spec.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos));
    if (dot == std::string::npos && d != 3) {
      std::cerr << "probe: --probe-block needs four dot-separated values, got " << spec << std::endl;
      exit(2);
    }
    pos = (dot == std::string::npos) ? pos : dot + 1;
  }
  return block;
}

std::string coordinate_string(const Coordinate &c)
{
  std::string s;
  for (std::size_t d = 0; d < c.size(); ++d) s += (d ? "." : "") + std::to_string(c[d]);
  return s;
}

} // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  const std::string cb_name = read_string(argc, argv, "--probe-checkerboard", "even");
  const Coordinate Block = parse_block(read_string(argc, argv, "--probe-block", "4.4.4.4"));
  const double mass = read_double(argc, argv, "--probe-mass", -0.2416);
  const double csw = read_double(argc, argv, "--probe-csw", 1.20536588031793);
  const double tol = read_double(argc, argv, "--probe-tol", 1.0e-10);
  const int maxiter = read_int(argc, argv, "--probe-maxiter", 1000);
  const int cg_maxiter = read_int(argc, argv, "--probe-cg-maxiter", 50000);
  const int run_cg = read_int(argc, argv, "--probe-run-cg", 1);
  // double | mixed | both. `mixed` is Grid's MixedPrecisionConjugateGradient (fp32 inner CG,
  // fp64 residual restarts) -- the solver the production driver runs for the light tail and the
  // strange RHMC (MixedPrecCGWrapper), and the counterpart of QUDA's precise=double /
  // sloppy=single CG. Needs the fp32 operator, so it requires --probe-mg-precision single.
  // `both` runs the two back to back after one MG setup.
  const std::string cg_precision = read_string(argc, argv, "--probe-cg-precision", "double");
  // Inner fp32 tolerance of the mixed CG. Default = tol, which is what the production driver
  // uses (it never sets InnerTolerance); Grid's own HMC drivers use 1e-8, and QUDA's nearest
  // analogue is reliable_delta = 1e-3. A comma-separated list runs one mixed-CG row per value,
  // all after the SAME MG setup -- at C3 a second launch would repeat a minute of setup.
  std::vector<double> cg_inner_tols;
  {
    std::stringstream ss(read_string(argc, argv, "--probe-cg-inner-tol", ""));
    for (std::string item; std::getline(ss, item, ',');)
      if (!item.empty()) cg_inner_tols.push_back(std::stod(item));
    if (cg_inner_tols.empty()) cg_inner_tols.push_back(tol);
  }
  // Restart cap for the mixed CG. 50 = the production driver's value.
  const int cg_mixed_outer = read_int(argc, argv, "--probe-cg-mixed-outer", 50);
  // Timed repeats per solver, median reported. Matches the QUDA probe's
  // --probe-solve-repeats so both sides aggregate the same way.
  const int solve_repeats = std::max(1, read_int(argc, argv, "--probe-solve-repeats", 3));
  // Account for the ~27% of the MG solve that the preconditioner's breakdown does
  // not see, i.e. the outer GCR's own work. ⚠️ Adds an accelerator_barrier() per
  // sub-region, so the instrumented TOTAL is not comparable with an uninstrumented
  // one -- read shares, not absolutes.
  const int instrument_outer = read_int(argc, argv, "--probe-instrument-outer", 0);
  // Re-enable the V-cycle's per-region barriers and breakdown. ⛔ Costs real time --
  // that is the point of making it optional.
  const int instrument_vcycle = read_int(argc, argv, "--probe-instrument-vcycle", 0);
  // Run the OUTER solver and the reference CG in fp32 as well, not just the MG
  // hierarchy. Needed because QUDA's MG forces its outer sloppy precision to match
  // the hierarchy, so Grid-single (fp64 outer) vs QUDA-single was never matched.
  // Requires --probe-mg-precision single, and a tolerance fp32 can reach (1e-6).
  const bool outer_single =
      (read_string(argc, argv, "--probe-outer-precision", "double") == "single");
  // Coarse stencil range, in hops. THIS IS NOT A FREE TUNING KNOB -- it has a
  // correct minimum set by the operator being coarsened, and job 58207223 was run
  // below it.
  //
  // CoarsenOperator reconstructs the coarse matrix by inverting an
  // npoint x npoint momentum matrix, which is exact only if the operator's
  // block-to-block couplings all lie within those npoint shifts. Anything outside
  // ALIASES into the retained points -- silently, with no error.
  //
  // The Schur operator Mpc = Moo - Moe Mee^-1 Meo is TWO hops (odd -> even
  // neighbour -> site-local clover inverse -> odd neighbour), so a site at a coarse
  // block corner reaches DIAGONALLY into the corner-adjacent block, e.g. shift
  // (-1,-1,0,0). A 9-point nearest-neighbour stencil has no such shift.
  //
  // Measured at C2 with hops=1: Galerkin deviation 4.7e-2 against a control of 2.3,
  // i.e. the coarse operator was visibly not the Galerkin projection. Grid's own
  // Schur drivers (Test_general_coarse_hdcg{,_phys48}.cc) use hops=4; only
  // Test_general_coarse_wilson.cc uses hops=1, and that one coarsens the FULL
  // one-hop operator, where 9 points is exact. The design doc inherited the wrong
  // one from that test.
  //
  //   1 -> 9 points  (nearest)           -- known insufficient for Mpc, kept to reproduce
  //   2 -> 33 points (next-to-nearest)   -- the analytic minimum for a 2-hop operator
  //   4 -> 81 points (all of {-1,0,1}^4) -- what Grid's own Schur drivers use
  // Coarse-solver Krylov depth. See the CoarseSolver construction for why this is
  // a memory knob as much as an algorithmic one.
  const int coarse_mmax = read_int(argc, argv, "--probe-coarse-mmax", 8);
  const int coarse_nstep = read_int(argc, argv, "--probe-coarse-nstep", 8);

  // Post-smoother Krylov depth and tolerance. The SAME allocation pathology as
  // the coarse solver (PrecGCRNonHermitian.h:138 allocates q(mmax)+p(mmax) on
  // every GCRnStep) but on FINE vectors: at C3 a fine rb vector is ~64 MB, so
  // mmax=4 churns ~510 MB per smoother application. The smoother is 29-31% of
  // the tuned solve, so this is the largest untested lever.
  //
  // smoother_tol is the other half: we ask 0.1 where QUDA asks 0.25, i.e. we may
  // simply be over-solving the smoother. QUDA's shape for reference is
  // nu_pre=0 / nu_post=8 CA_GCR at tol 0.25 (quda_grid_bridge.h:483-487).
  const int smoother_mmax = read_int(argc, argv, "--probe-smoother-mmax", 4);
  const int smoother_nstep = read_int(argc, argv, "--probe-smoother-nstep", 4);
  const double smoother_tol = read_double(argc, argv, "--probe-smoother-tol", 0.1);
  const int smoother_maxiter = read_int(argc, argv, "--probe-smoother-maxiter", 1);

  // Outer MG-preconditioned GCR depth. Restart length, not a memory knob at the
  // same scale (16 fine vectors ~ 1 GB at C3, allocated once per solve).
  const int outer_mmax = read_int(argc, argv, "--probe-outer-mmax", 16);
  const int outer_nstep = read_int(argc, argv, "--probe-outer-nstep", 16);

  // ---- Stage 1: the waste-removal switch ----------------------------------
  //
  // --probe-fast-mg 0 reproduces Grid's PrecGCRNonHermitian and blockProject exactly and is
  // the CONTROL; 1 enables the cleaned path. One binary runs both arms, which is what lets the
  // pair be measured in a single allocation -- across allocations this configuration has
  // ranged 1.602-2.292 s and no comparison would survive it.
  //
  // ⇒ THE GATE: at C3 clover fp64 both arms must give 11 outer PGCR steps and an independent
  // residual of 2.711e-11. Every removal is an algebraic identity; if convergence moves, the
  // cleaned path has a bug and the timing is meaningless.
  const int fast_mg = read_int(argc, argv, "--probe-fast-mg", 0);
  // Escape hatches, so a single suspect change can be isolated without a rebuild.
  const int fast_gcr = read_int(argc, argv, "--probe-fast-gcr", fast_mg);
  const int fast_project = read_int(argc, argv, "--probe-fast-project", fast_mg);
  const int persistent_temps = read_int(argc, argv, "--probe-persistent-temps", fast_mg);
  // D2's report block: an extra operator application per converged solve, for a log line.
  const int verify_residual = read_int(argc, argv, "--probe-verify-residual", fast_mg ? 0 : 1);

  // ---- Stage 2: the preconditioner-strength knobs -------------------------
  //
  // Grid is configured WEAKER than QUDA at every level, which is where the 11-vs-6 outer
  // iteration gap comes from. Defaults below reproduce the values inherited from Grid's own
  // test; QUDA's counterparts are named alongside.
  //
  // Coarse-solver tolerance. Grid's test value is 2e-1; QUDA asks 0.1
  // (quda_grid_bridge.h:491). Was hardcoded until now.
  const double coarse_tol = read_double(argc, argv, "--probe-coarse-tol", 2.0e-1);
  // Coarse-solver RESTART cap (GCR cycles of coarse_nstep steps each). Was hardcoded 50, i.e.
  // effectively "solve to coarse_tol". At C3 clover that costs ~113 coarse applications per
  // V-cycle, where QUDA caps its coarse solve at 12 iterations total (quda_grid_bridge.h:562).
  // QUDA's budget is coarse_maxiter 1 with coarse_nstep = coarse_mmax = 12.
  const int coarse_maxiter = read_int(argc, argv, "--probe-coarse-maxiter", 50);
  // Workstream A (plan v2 §5): which coarse-operator APPLY the coarse solver drives. The
  // coarsening is ALWAYS done by GeneralCoarsenedMatrix; only the apply changes.
  //   general : GeneralCoarsenedMatrix::Mult -- the control (padded volume, PaddedCell exchange)
  //   stencil : Grid's old CoarsenedMatrix apply (CartesianStencil halo, interior only); hops=1
  //   mrhs    : Grid's MultiGeneralCoarsenedMatrix (batched cuBLAS, interior only); fp64 only
  // See probe_mg_solvers.h for why each restriction exists.
  const std::string coarse_apply = read_string(argc, argv, "--probe-coarse-apply", "general");
  // N timed applications of each available apply on one random coarse vector, plus the
  // agreement gate against `general`. 0 = off. Builds every alternative, so costs memory.
  const int coarse_apply_check = read_int(argc, argv, "--probe-coarse-apply-check", 0);
  // Null-vector solve tolerance and inverse-iteration rounds. Grid hardcodes 1e-3 and 3
  // rounds (Aggregates.h:136); QUDA's setup_tol is 5e-6 (quda_grid_bridge.h:479).
  // ⚠️ THE LEADING CANDIDATE for the iteration gap, and the one that also retires the
  // "Grid's MG setup is 2.5x faster than QUDA's" claim -- a cheaper setup that buys a weaker
  // preconditioner is not a win.
  const double subspace_tol = read_double(argc, argv, "--probe-subspace-tol", 1.0e-3);
  const int subspace_rounds = read_int(argc, argv, "--probe-subspace-rounds", 3);
  const int subspace_mmax = read_int(argc, argv, "--probe-subspace-mmax", 10);
  const int subspace_maxiter = read_int(argc, argv, "--probe-subspace-maxiter", 30);

  // Precision of the MG PRECONDITIONER (the outer solver and every gate stay in
  // double either way -- see PrecisionChangeAdaptor for why).
  //
  //   double : the whole hierarchy in fp64. Grid's only supported MG precision
  //            until now, and what every Grid MG number in this campaign used.
  //   single : subspace, coarse operator, coarse solver and smoother all in fp32,
  //            converted at the preconditioner boundary. This is the configuration
  //            that MATCHES QUDA's (single preconditioner under a double outer).
  //
  // ⛔ THERE IS NO `half` OPTION AND THERE CANNOT BE ONE. Grid's vComplexH is a
  // uint16_t container with NO fp16 arithmetic -- its own type traits say so:
  // "Fixme this is incomplete until Grid supports fp16 or bfp16 arithmetic types"
  // (Grid/tensors/Tensor_traits.h:231-232, 245-246). There is no LatticeComplexH,
  // the mixed-precision solvers are SFINAE-gated to exactly double<->single
  // (getPrecision, Tensor_traits.h:394-404), and the only consumers of H are
  // precisionChange targets and the Dslash comms compressor
  // (FermionOperatorImpl.h:96-111). So QUDA's half null vectors / half coarse
  // halos have NO Grid counterpart; that asymmetry is a finding, not a gap in
  // this probe.
  const std::string precision_name = read_string(argc, argv, "--probe-mg-precision", "double");
  if (precision_name != "double" && precision_name != "single") {
    std::cerr << "probe: --probe-mg-precision must be double|single"
              << " (half is impossible in Grid -- see the source comment)" << std::endl;
    Grid_finalize();
    return 2;
  }
  const bool mg_single = (precision_name == "single");

  if (cg_precision != "double" && cg_precision != "mixed" && cg_precision != "both") {
    std::cerr << "probe: --probe-cg-precision must be double|mixed|both" << std::endl;
    Grid_finalize();
    return 2;
  }
  if (cg_precision != "double" && !mg_single) {
    std::cerr << "probe: --probe-cg-precision mixed|both needs the fp32 operator, i.e."
              << " --probe-mg-precision single" << std::endl;
    Grid_finalize();
    return 2;
  }

  const int stencil_hops = read_int(argc, argv, "--probe-stencil-hops", 2);
  if (stencil_hops < 1 || stencil_hops > 4) {
    std::cerr << "probe: --probe-stencil-hops must be 1..4" << std::endl;
    Grid_finalize();
    return 2;
  }
  if (coarse_apply != "general" && coarse_apply != "stencil" && coarse_apply != "mrhs") {
    std::cerr << "probe: --probe-coarse-apply must be general|stencil|mrhs" << std::endl;
    Grid_finalize();
    return 2;
  }
  if (coarse_apply == "stencil" && stencil_hops != 1) {
    std::cerr << "probe: --probe-coarse-apply stencil needs --probe-stencil-hops 1"
              << " (Grid's old Geometry has no corner points)" << std::endl;
    Grid_finalize();
    return 2;
  }
  if (coarse_apply == "mrhs" && mg_single) {
    std::cerr << "probe: --probe-coarse-apply mrhs needs --probe-mg-precision double: Grid's"
              << " MultiGeneralCoarsenedMatrix hard-wires ComplexD GEMM (see probe_mg_solvers.h)"
              << std::endl;
    Grid_finalize();
    return 2;
  }

  if (cb_name != "even" && cb_name != "odd") {
    std::cerr << "probe: --probe-checkerboard must be even|odd" << std::endl;
    Grid_finalize();
    return 2;
  }
  const int cb = (cb_name == "odd") ? Odd : Even;

  // wilson | clover. Wilson is doc §4's control -- MG is EXPECTED to lose on it,
  // because the solve converges in far fewer iterations than clover and the MG
  // setup cost is unchanged. csw is ignored for wilson.
  const std::string action_name = read_string(argc, argv, "--probe-action", "clover");
  if (action_name != "wilson" && action_name != "clover") {
    std::cerr << "probe: --probe-action must be wilson|clover" << std::endl;
    Grid_finalize();
    return 2;
  }

  // Schur convention. See the operator construction below for why the default is
  // symmetric; parsed here because the geometry banner reports it.
  const std::string schur_name = read_string(argc, argv, "--probe-schur", "one");
  if (schur_name != "one" && schur_name != "mooee") {
    std::cerr << "probe: --probe-schur must be one|mooee" << std::endl;
    Grid_finalize();
    return 2;
  }

  GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  const bool boss = UGrid->IsBoss();

  // ---- Coarse grid, and the geometry gate (doc §4) -------------------------
  //
  // The x ratio is 2 rather than 4 because a red-black grid halves
  // _rdimensions[0]: a block of 2 red-black x-sites spans 4 physical x. So the
  // PHYSICAL aggregation block is Block[], matching QUDA's geo_block_size, while
  // the ratio Grid sees in dim 0 is Block[0]/2. Block[0] must therefore be even.
  Coordinate clatt = GridDefaultLatt();
  for (int d = 0; d < 4; ++d) {
    if (clatt[d] % Block[d] != 0) {
      if (boss)
        std::cerr << "probe: lattice dim " << d << " (" << clatt[d] << ") is not divisible by block "
                  << Block[d] << std::endl;
      Grid_finalize();
      return 2;
    }
    clatt[d] = clatt[d] / Block[d];
  }
  if (Block[0] % 2 != 0) {
    if (boss) std::cerr << "probe: Block[0] must be even on a 4D red-black grid (checker dim is x)" << std::endl;
    Grid_finalize();
    return 2;
  }

  GridCartesian *Coarse4d = SpaceTimeGrid::makeFourDimGrid(
      clatt, GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());

  if (boss) {
    std::cout << GridLogMessage << "=== probe_grid_mg_schur_clover ===" << std::endl;
    std::cout << GridLogMessage << "action            " << action_name << std::endl;
    std::cout << GridLogMessage << "checkerboard      " << cb_name << " (" << cb << ")" << std::endl;
    std::cout << GridLogMessage << "aggregation block " << coordinate_string(Block) << std::endl;
    std::cout << GridLogMessage << "stencil hops      " << stencil_hops << std::endl;
    std::cout << GridLogMessage << "schur convention  " << schur_name
              << (schur_name == "one" ? " (symmetric; comparable with QUDA MG)"
                                      : " (asymmetric; QUDA MG CANNOT coarsen this)")
              << std::endl;
    std::cout << GridLogMessage << "nbasis            " << kNbasis << " -> coarse dof " << 2 * kNbasis
              << " (gamma5 doubled; matches QUDA n_vec=24 x spin_block_size=2)" << std::endl;
    std::cout << GridLogMessage << "coarse   mmax/nstep " << coarse_mmax << "/" << coarse_nstep << std::endl;
    std::cout << GridLogMessage << "smoother mmax/nstep " << smoother_mmax << "/" << smoother_nstep
              << " tol " << smoother_tol << " maxiter " << smoother_maxiter << std::endl;
    std::cout << GridLogMessage << "outer    mmax/nstep " << outer_mmax << "/" << outer_nstep << std::endl;
    std::cout << GridLogMessage << "coarse   tol        " << coarse_tol << " (QUDA asks 0.1)"
              << std::endl;
    std::cout << GridLogMessage << "coarse   apply      " << coarse_apply
              << " (check " << coarse_apply_check << ")" << std::endl;
    std::cout << GridLogMessage << "subspace tol/rounds " << subspace_tol << "/" << subspace_rounds
              << " mmax " << subspace_mmax << " maxiter " << subspace_maxiter
              << " (QUDA setup_tol 5e-6)" << std::endl;
    std::cout << GridLogMessage << "fast-mg           " << fast_mg << " (gcr " << fast_gcr
              << ", project " << fast_project << ", persistent-temps " << persistent_temps
              << ", verify-residual " << verify_residual << ")"
              << (fast_mg ? "  [Stage 1: waste removed]" : "  [CONTROL: reproduces Grid]")
              << std::endl;
    std::cout << GridLogMessage << "--- geometry gate ---" << std::endl;
    std::cout << GridLogMessage << "fine   full  gdim " << coordinate_string(UGrid->GlobalDimensions())
              << " rdim " << coordinate_string(UGrid->_rdimensions) << std::endl;
    std::cout << GridLogMessage << "fine   rb    gdim " << coordinate_string(UrbGrid->GlobalDimensions())
              << " rdim " << coordinate_string(UrbGrid->_rdimensions) << std::endl;
    std::cout << GridLogMessage << "coarse       gdim " << coordinate_string(Coarse4d->GlobalDimensions())
              << " rdim " << coordinate_string(Coarse4d->_rdimensions) << std::endl;
    std::cout << GridLogMessage << "procs fine/coarse " << coordinate_string(UrbGrid->_processors) << " / "
              << coordinate_string(Coarse4d->_processors) << std::endl;
    std::cout << GridLogMessage << "simd  fine/coarse " << coordinate_string(UrbGrid->_simd_layout) << " / "
              << coordinate_string(Coarse4d->_simd_layout) << std::endl;
    Coordinate ratio(4);
    for (int d = 0; d < 4; ++d) ratio[d] = UrbGrid->_rdimensions[d] / Coarse4d->_rdimensions[d];
    std::cout << GridLogMessage << "rdim ratio        " << coordinate_string(ratio)
              << "   (expect Block with dim0 halved)" << std::endl;
  }
  // The real gate: this is what blockProject/blockPromote call internally. If the
  // decomposition is wrong it aborts here, before anything expensive runs.
  subdivides(Coarse4d, UrbGrid);
  if (boss) std::cout << GridLogMessage << "subdivides(coarse, fine_rb): PASSED" << std::endl;

  // ---- Gauge field and operator -------------------------------------------
  std::vector<int> seeds4({1, 2, 3, 4});
  std::vector<int> cseeds({5, 6, 7, 8});
  // The fine RNG lives on the FULL grid on purpose: GridParallelRNG::fill on a
  // checkerboarded field builds its temporary on the RNG's own grid and then
  // pickCheckerboard()s it (Lattice_rng.h:370), so an rb-grid RNG would not do.
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(seeds4);
  // Coarse fields need their OWN RNG. RNGfillable_general requires the field's
  // grid to be finer than or equal to the RNG's, and the coarse grid is coarser
  // than UGrid -- filling a CoarseVector from RNG4 aborts at runtime.
  GridParallelRNG CRNG(Coarse4d);
  CRNG.SeedFixedIntegers(cseeds);

  LatticeGaugeFieldD Umu(UGrid);
  const std::string cfg = read_string(argc, argv, "--probe-cfg", "");
  if (cfg.empty()) {
    SU<Nc>::HotConfiguration(RNG4, Umu);
  } else {
    FieldMetaData header;
    IldgReader reader;
    reader.open(cfg);
    reader.readConfiguration(Umu, header);
    reader.close();
  }
  const double plaquette = WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu);
  if (boss)
    std::cout << GridLogMessage << "gauge " << (cfg.empty() ? "hot" : cfg) << " plaquette " << plaquette
              << std::endl;

  // ---- Stout smearing ------------------------------------------------------
  //
  // The production clover action is DEFINED with stout-smeared links, so an
  // unsmeared probe inverts a different operator from the one HMC inverts. This
  // ensemble's own metadata records <STOUT_FERM_STATE> rho=0.125, n_smear=1,
  // orthog_dir=-1; orthog_dir=-1 means all four directions, which is exactly
  // Grid's Smear_Stout default.
  //
  // Smearing does NOT simply make the problem harder: it sharply reduces the
  // additive mass renormalisation, so m_c moves toward zero and the SAME bare
  // mass represents a far lighter quark. That is why m0 = -0.2450 is a light
  // point for the smeared action and a heavy one unsmeared (75 CG iterations).
  //
  // The gate is the smeared plaquette, which the ensemble publishes in
  // plaquette_sm1_rho0.125_dat_a.xml -- check it before trusting any timing.
  const int stout_nsmear = read_int(argc, argv, "--probe-stout-nsmear", 0);
  const double stout_rho = read_double(argc, argv, "--probe-stout-rho", 0.125);
  if (stout_nsmear > 0) {
    Smear_Stout<PeriodicGimplD> stout(stout_rho);
    LatticeGaugeFieldD Usmear(UGrid);
    for (int n = 0; n < stout_nsmear; ++n) {
      stout.smear(Usmear, Umu);
      Umu = Usmear;
    }
    const double plaq_smeared = WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu);
    if (boss)
      std::cout << GridLogMessage << "stout smearing rho " << stout_rho << " n_smear "
                << stout_nsmear << " -> smeared plaquette " << plaq_smeared << std::endl;
  }

  // Same operator the benchmark's clover_impl=compact path builds: antiperiodic
  // time via boundary_phases, csw_r = csw_t = csw, cF = 1.0 (bulk boundary).
  std::vector<Complex> phases(Nd, 1.0);
  phases[Nd - 1] = -1.0;
  WilsonFermionD::ImplParams implParams;
  implParams.boundary_phases = phases;
  WilsonAnisotropyCoefficients anisotropy;

  // Wilson and compact clover share one code path via a WilsonFermionD base
  // pointer: CompactWilsonCloverFermion derives from WilsonFermion, so
  // Meooe/Mooee/MooeeInv dispatch virtually and SchurDiag*Operator<WilsonFermionD,
  // ...> binds to either. Same idiom as grid_wilson_clover_operator.h.
  //
  // Wilson is doc §4's CONTROL, and here it is also a controlled experiment on the
  // clover result: it shares the entire coarse-grid and transfer path and changes
  // only the fine operator. Since the fine operator is ~11% of Grid's MG time, a
  // Wilson deficit of the same order as clover's confirms the cost is in the
  // coarse machinery; near-parity on Wilson would refute that.
  std::unique_ptr<WilsonFermionD> fermop;
  if (action_name == "wilson") {
    fermop.reset(new WilsonFermionD(Umu, *UGrid, *UrbGrid, mass, implParams));
  } else {
    // cF = 1.0: bulk boundary, so this is the same physics as the standard clover.
    fermop.reset(new CompactWilsonCloverFermionD(Umu, *UGrid, *UrbGrid, mass, csw, csw, 1.0,
                                                 anisotropy, implParams));
  }
  WilsonFermionD &Dc = *fermop;

  // WHICH SCHUR CONVENTION, AND WHY IT IS NOT THE ONE THE CG ROWS USE.
  //
  //   SchurDiagOneOperator    1 - Moo^-1 Moe Mee^-1 Meo   == QUDA "symmetric"
  //   SchurDiagMooeeOperator  Moo - Moe Mee^-1 Meo        == QUDA "asymmetric"
  //
  // QUDA multigrid CANNOT coarsen the asymmetric form -- quda/lib/coarse_op.cuh:990
  // rejects both asymmetric matpc types outright. So the MG comparison is forced
  // onto the SYMMETRIC operator on both backends. Grid itself has no such
  // restriction, which is why this is a runtime flag: `mooee` measures whether the
  // convention costs Grid anything, so the forced choice is a recorded datum
  // rather than an untested assumption.
  //
  // Default `one` = symmetric = comparable with QUDA. (Parsed with the other flags
  // at the top of main, because the geometry banner reports it.)
  //
  // Both forms contain Moe ... Meo, so both are 2-hop and the coarse stencil
  // finding (hops >= 2) applies unchanged to either.
  //
  // Held by base pointer so CoarsenOperator (LinearOperatorBase&), the outer PGCR
  // and ShiftedSchurOperator (SchurOperatorBase&) all bind to either choice.
  // Op() = Mpc, AdjOp() = MpcDag, HermOp() = Mpc^dag Mpc; CoarsenOperator and PGCR
  // both use Op(), so it goes in unwrapped -- see the header comment.
  std::unique_ptr<SchurOperatorBase<LatticeFermionD>> schur_holder;
  if (schur_name == "one")
    schur_holder.reset(new SchurDiagOneOperator<WilsonFermionD, LatticeFermionD>(Dc));
  else
    schur_holder.reset(new SchurDiagMooeeOperator<WilsonFermionD, LatticeFermionD>(Dc));
  SchurOperatorBase<LatticeFermionD> &SchurOp = *schur_holder;

  // ---- Subspace ------------------------------------------------------------
  typedef Aggregation<vSpinColourVectorD, vTComplexD, kNbasis> Subspace;
  typedef Aggregation<vSpinColourVectorD, vTComplexD, 2 * kNbasis> CombinedSubspace;
  typedef GeneralCoarsenedMatrix<vSpinColourVectorD, vTComplexD, 2 * kNbasis> LittleDiracOperator;
  typedef LittleDiracOperator::CoarseVector CoarseVector;

  // Constructed from the runtime hop count rather than one of the named
  // subclasses: NearestStencilGeometry4D etc. are just fixed-hops wrappers over
  // this same base, so this covers all of them and lets one binary scan the range.
  NonLocalStencilGeometry4D geom(Coarse4d, stencil_hops);

  CombinedSubspace CombinedUV(Coarse4d, UrbGrid, cb);
  double subspace_seconds = 0.0;
  double t0 = 0.0;

  if (cb == Even) {
    Subspace Aggregates(Coarse4d, UrbGrid, cb);

    if (boss) std::cout << GridLogMessage << "--- subspace generation (GCR) ---" << std::endl;
    accelerator_barrier();
    UGrid->Barrier();
    t0 = usecond();
    // GCR, not Chebyshev: CreateSubspaceChebyshev filters a HERMITIAN operator and
    // does not apply to Mpc. CreateSubspaceGCR drives DiracOp.Op() directly.
    //
    // Note it builds its noise as `FineField noise(FineGrid)`, whose checkerboard
    // defaults to Even, so the subspace it returns is Even whatever the Aggregation
    // was constructed with. That is the same default that makes CoarsenOperator
    // Even-only, and it is why this branch is guarded on cb == Even.
    // Probe-local reimplementation of Grid's CreateSubspaceGCR with the tolerance and round
    // count exposed (Stage 2). Defaults reproduce Grid exactly; see probe_mg_solvers.h.
    ProbeMG::create_subspace_gcr(RNG4, SchurOp, Aggregates, kNbasis, subspace_tol,
                                 subspace_rounds, subspace_mmax, subspace_mmax,
                                 subspace_maxiter, fast_gcr != 0, fast_gcr != 0);
    accelerator_barrier();
    UGrid->Barrier();
    subspace_seconds = (usecond() - t0) / 1.0e6;

    // gamma5 chirality doubling, as in Test_general_coarse_wilson.cc. This is Grid's
    // counterpart to QUDA's spin_block_size[0] = 2.
    Gamma G5(Gamma::Algebra::Gamma5);
    for (int b = 0; b < kNbasis; ++b) {
      CombinedUV.subspace[b] = Aggregates.subspace[b];
      CombinedUV.subspace[b + kNbasis] = G5 * Aggregates.subspace[b];
    }
  } else {
    // ODD: the Even-only restriction demonstration (doc §6).
    //
    // A genuinely Odd-parity subspace, built by hand because CreateSubspaceGCR
    // cannot produce one. Its numerical quality is irrelevant: CoarsenOperator
    // asserts inside CBFromExpression on the very first `phaF[p] * subspace[i]`,
    // which is BEFORE the first fine-operator application -- so this costs nothing
    // and the abort we are demonstrating is the genuine one, reached by the genuine
    // route, not a relabelling trick.
    if (boss) {
      std::cout << GridLogMessage << "--- ODD checkerboard: demonstrating the Even-only restriction ---"
                << std::endl;
      std::cout << GridLogMessage
                << "Building a genuine Odd-parity subspace and calling CoarsenOperator." << std::endl;
      std::cout << GridLogMessage
                << "EXPECTED: abort in CBFromExpression (Lattice_ET.h:248) -- CoarsenOperator's"
                << std::endl;
      std::cout << GridLogMessage
                << "internal phase field is Even (Lattice_base.h:268 + Lattice_transfer.h:346)."
                << std::endl;
      std::cout << GridLogMessage << "A clean run here would mean the doc's claim is WRONG." << std::endl;
    }
    LatticeFermionD full(UGrid);
    for (int b = 0; b < 2 * kNbasis; ++b) {
      random(RNG4, full);
      CombinedUV.subspace[b].Checkerboard() = cb;
      pickCheckerboard(cb, CombinedUV.subspace[b], full);
    }
  }

  // ---- Coarsen -------------------------------------------------------------
  LittleDiracOperator LittleDiracOp(geom, UrbGrid, Coarse4d);

  if (boss) std::cout << GridLogMessage << "--- coarsening (" << geom.npoint << " points x " << 2 * kNbasis
                      << " vectors = " << geom.npoint * 2 * kNbasis << " Mpc applications) ---" << std::endl;
  accelerator_barrier();
  UGrid->Barrier();
  t0 = usecond();
  LittleDiracOp.CoarsenOperator(SchurOp, CombinedUV);
  accelerator_barrier();
  UGrid->Barrier();
  const double coarsen_seconds = (usecond() - t0) / 1.0e6;

  // ---- Galerkin check ------------------------------------------------------
  //
  // Replaces runChecks(), which exists only on the OLD CoarsenedMatrix path
  // (tests/solver/Test_multigrid_common.h:391) and has no counterpart here.
  //
  // Verifies A_coarse == P^dag A P, i.e. that the coarse matrix really is the
  // Galerkin projection of the fine operator we believe we coarsened. Exact only
  // because CoarsenOperator block-orthogonalises the subspace in place, so this
  // must run AFTER it. Pattern from Test_general_coarse_wilson.cc:243-256.
  //
  // The control re-runs the same comparison against Mpc^dag Mpc. That is the
  // operator an HermOpAdaptor would have coarsened, so a large deviation there is
  // positive evidence that the check can tell the two apart -- without which a
  // small deviation above proves nothing.
  auto galerkin_deviation = [&](bool use_herm) {
    CoarseVector c_src(Coarse4d);
    CoarseVector c_res(Coarse4d);
    CoarseVector c_proj(Coarse4d);
    LatticeFermionD f_prom(UrbGrid);
    LatticeFermionD f_mat(UrbGrid);
    f_prom.Checkerboard() = cb;
    f_mat.Checkerboard() = cb;

    random(CRNG, c_src);

    CombinedUV.PromoteFromSubspace(c_src, f_prom);
    if (use_herm)
      SchurOp.HermOp(f_prom, f_mat);
    else
      SchurOp.Op(f_prom, f_mat);
    CombinedUV.ProjectToSubspace(c_proj, f_mat);

    LittleDiracOp.M(c_src, c_res);

    c_proj = c_proj - c_res;
    return std::sqrt(norm2(c_proj) / norm2(c_res));
  };

  const double galerkin_mpc = galerkin_deviation(false);
  const double galerkin_herm = galerkin_deviation(true);
  const double galerkin_tolerance = 1.0e-10;
  const bool galerkin_passed = (galerkin_mpc <= galerkin_tolerance) && (galerkin_herm > 1.0e-3);

  if (boss) {
    std::cout << GridLogMessage << "--- Galerkin check ---" << std::endl;
    std::cout << GridLogMessage << "  vs Mpc          (expect <= " << galerkin_tolerance << ") : "
              << galerkin_mpc << std::endl;
    std::cout << GridLogMessage << "  vs Mpc^dag Mpc  (control, expect O(1))    : " << galerkin_herm
              << std::endl;
    std::cout << GridLogMessage << "  Galerkin check: " << (galerkin_passed ? "PASSED" : "FAILED")
              << std::endl;
  }

  // ---- Solver stack --------------------------------------------------------
  //
  // Three levels of PrecGeneralisedConjugateResidualNonHermitian, exactly as
  // Test_general_coarse_wilson.cc: coarse solver, fine smoother, and outer solver,
  // with MGPreconditioner between them. Every one of them uses only Linop.Op().
  TrivialPrecon<CoarseVector> coarse_trivial;
  TrivialPrecon<LatticeFermionD> fine_trivial;

  // ---- Stage 0: operator-application counters -----------------------------
  //
  // The diagnosis rests on a COUNT -- 9 fine Mpc and 4 coarse Mults per V-cycle, two of each
  // unnecessary -- so the count is measured, not inferred. Reset immediately before the timed
  // solve, because CoarsenOperator applies Mpc npoint*2*nbasis times during setup.
  //
  // Every operator the solvers actually drive is wrapped:
  //   CountedSchurOp          the MG preconditioner's two residual computations, and the
  //                           outer PGCR's own Op
  //   CountedShiftedSchurOp   the post-smoother
  //   CountedCoarse           the coarse solver
  // LinOpCoarse is NOT wrapped: MGPreconditioner stores it as _CoarseOperator but never
  // applies it -- the V-cycle goes through _CoarseSolve.
  ProbeMG::OpCounts counts;

  NonHermitianLinearOperator<LittleDiracOperator, CoarseVector> LinOpCoarse(LittleDiracOp);
  ShiftedNonHermitianLinearOperator<LittleDiracOperator, CoarseVector> ShiftedLinOpCoarse(LittleDiracOp, 0.001);

  // ---- Workstream A: alternative coarse APPLIES, fp64 hierarchy (plan v2 §5) -------------
  //
  // Same coarse matrix and the same 0.001 shift, different apply routine. Built only when
  // selected or when the agreement check is on, since each holds its own copy of the links.
  // Skipped entirely under --probe-mg-precision single: the fp64 coarse solver is never driven
  // there (OuterPrecon selects the fp32 hierarchy) and memory at C3 is already tight.
  typedef ProbeMG::StencilCoarseApply<vSpinColourVectorD, vTComplexD, 2 * kNbasis> StencilCoarseD;
  typedef ProbeMG::MrhsCoarseApply<vSpinColourVectorD, vTComplexD, 2 * kNbasis> MrhsCoarseD;
  std::unique_ptr<StencilCoarseD> StencilCoarse;
  std::unique_ptr<MrhsCoarseD> MrhsCoarse;
  if (!mg_single) {
    if (coarse_apply == "stencil" || (coarse_apply_check > 0 && stencil_hops == 1))
      StencilCoarse.reset(new StencilCoarseD(LittleDiracOp, Coarse4d, 0.001));
    if (coarse_apply == "mrhs" || coarse_apply_check > 0)
      MrhsCoarse.reset(new MrhsCoarseD(LittleDiracOp, Coarse4d, 0.001));
    if (coarse_apply_check > 0) {
      if (boss) std::cout << GridLogMessage << "--- coarse apply check (fp64) ---" << std::endl;
      CoarseVector c_probe(Coarse4d);
      random(CRNG, c_probe);
      if (StencilCoarse)
        ProbeMG::check_coarse_apply<CoarseVector>("stencil", ShiftedLinOpCoarse, *StencilCoarse,
                                                  c_probe, UGrid, coarse_apply_check, boss);
      if (MrhsCoarse)
        ProbeMG::check_coarse_apply<CoarseVector>("mrhs   ", ShiftedLinOpCoarse, *MrhsCoarse,
                                                  c_probe, UGrid, coarse_apply_check, boss);
    }
  }
  LinearOperatorBase<CoarseVector> &SelectedCoarse =
      (StencilCoarse && coarse_apply == "stencil")
          ? static_cast<LinearOperatorBase<CoarseVector> &>(*StencilCoarse)
          : (MrhsCoarse && coarse_apply == "mrhs")
                ? static_cast<LinearOperatorBase<CoarseVector> &>(*MrhsCoarse)
                : static_cast<LinearOperatorBase<CoarseVector> &>(ShiftedLinOpCoarse);
  ProbeMG::CountingLinearOperator<CoarseVector> CountedCoarse(SelectedCoarse, counts.coarse);
  ProbeMG::CountingLinearOperator<LatticeFermionD> CountedSchurOp(SchurOp, counts.fine);
  // ⛔ mmax IS A MEMORY KNOB, NOT JUST AN ALGORITHM KNOB.
  // PrecGeneralisedConjugateResidualNonHermitian allocates
  //     std::vector<Field> q(mmax,grid);  std::vector<Field> p(mmax,grid);
  // on EVERY GCRnStep call (PrecGCRNonHermitian.h:138). At C3 a coarse vector is
  // ~2 MB, so the mmax=50 inherited from Test_general_coarse_wilson.cc allocated
  // ~200 MB per coarse solve, ~22 times per MG solve -- several GB of pure
  // allocation churn, for a solve that converges in TWO steps. That test runs on a
  // tiny lattice where 100 coarse vectors cost nothing; at physical volume it
  // dominates.
  //
  // Default 8: comfortably above the 2 steps actually needed, 6x smaller than 50.
  // If the coarse solve ever needs more than nstep steps the outer loop simply
  // restarts GCR, so this trades a (never-taken) restart for a large allocation.
  // ⚠️ BOTH solvers are constructed and one is SELECTED. The control arm is therefore stock
  // Grid itself, not a reproduction of it -- which matters because the largest single item in
  // D4 is Grid's per-call allocation of q(mmax)+p(mmax), and any emulation that kept a
  // persistent workspace would have quietly removed it from the control and understated the
  // result. Construction is free: both classes only store references until first applied.
  PrecGeneralisedConjugateResidualNonHermitian<CoarseVector> CoarseSolverStock(
      coarse_tol, coarse_maxiter, CountedCoarse, coarse_trivial, coarse_mmax, coarse_nstep);
  CoarseSolverStock.Level(3);
  ProbeMG::FlexibleGCR<CoarseVector> CoarseSolverFast(
      coarse_tol, coarse_maxiter, CountedCoarse, coarse_trivial, coarse_mmax, coarse_nstep);
  CoarseSolverFast.Level(3);
  CoarseSolverFast.verify_residual = (verify_residual != 0);
  // The probe zeroes Csol before every coarse solve, so the entry residual is exactly the
  // source and the initial operator application is pure waste (D1).
  CoarseSolverFast.zero_guess = true;
  LinearFunction<CoarseVector> &CoarseSolver =
      fast_gcr ? static_cast<LinearFunction<CoarseVector> &>(CoarseSolverFast)
               : static_cast<LinearFunction<CoarseVector> &>(CoarseSolverStock);

  ShiftedSchurOperator<LatticeFermionD> ShiftedSchurOp(SchurOp, 0.01);
  ProbeMG::CountingLinearOperator<LatticeFermionD> CountedShiftedSchurOp(ShiftedSchurOp, counts.fine);
  PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD> SmootherStock(
      smoother_tol, smoother_maxiter, CountedShiftedSchurOp, fine_trivial, smoother_mmax, smoother_nstep);
  SmootherStock.Level(2);
  ProbeMG::FlexibleGCR<LatticeFermionD> SmootherFast(
      smoother_tol, smoother_maxiter, CountedShiftedSchurOp, fine_trivial, smoother_mmax, smoother_nstep);
  SmootherFast.Level(2);
  SmootherFast.verify_residual = (verify_residual != 0);
  // Likewise vec2 is zeroed before every smoother application.
  SmootherFast.zero_guess = true;
  LinearFunction<LatticeFermionD> &Smoother =
      fast_gcr ? static_cast<LinearFunction<LatticeFermionD> &>(SmootherFast)
               : static_cast<LinearFunction<LatticeFermionD> &>(SmootherStock);

  MGPreconditioner<vSpinColourVectorD, vTComplexD, 2 * kNbasis> Precon(
      CombinedUV, CountedSchurOp, fine_trivial, Smoother, LinOpCoarse, CoarseSolver);
  Precon.fast_project = (fast_project != 0);
  Precon.project_mode = fast_project;
  Precon.instrument = (instrument_vcycle != 0);
  Precon.persistent_temps = (persistent_temps != 0);

  // ---- Single-precision preconditioner hierarchy ---------------------------
  //
  // Built only when --probe-mg-precision single. Everything here mirrors the
  // double stack above with F types, and is reached by the outer (double) solver
  // through PrecisionChangeAdaptor. The outer Krylov, the independent residual and
  // every gate stay in double regardless -- see PrecisionChangeAdaptor for why an
  // all-single solve could not meet a 1e-10 gate.
  //
  // The SUBSPACE is converted from the double one rather than regenerated, so both
  // precisions precondition with the SAME null space and the measurement isolates
  // arithmetic precision alone. (QUDA instead generates its null vectors at the
  // reduced precision; regenerating here would confound two changes at once.)
  typedef Aggregation<vSpinColourVectorF, vTComplexF, 2 * kNbasis> CombinedSubspaceF;
  typedef GeneralCoarsenedMatrix<vSpinColourVectorF, vTComplexF, 2 * kNbasis> LittleDiracOperatorF;
  typedef LittleDiracOperatorF::CoarseVector CoarseVectorF;

  GridCartesian *UGridF = nullptr;
  GridRedBlackCartesian *UrbGridF = nullptr;
  GridCartesian *Coarse4dF = nullptr;
  std::unique_ptr<LatticeGaugeFieldF> UmuF;
  std::unique_ptr<WilsonFermionF> fermopF;
  std::unique_ptr<SchurOperatorBase<LatticeFermionF>> schur_holderF;
  std::unique_ptr<CombinedSubspaceF> CombinedUVF;
  // ⛔ MUST OUTLIVE LittleDiracOpF: GeneralCoarsenedMatrix stores the geometry BY
  // REFERENCE (`NonLocalStencilGeometry &geom;`, GeneralCoarsenedMatrix.h:61), so
  // a block-scoped geometry dangles the moment the block exits. Declaring it
  // inside the `if (mg_single)` block segfaulted immediately AFTER coarsening
  // completed -- and at C3 that same bug surfaced first as a bogus multi-exabyte
  // allocation, i.e. it masqueraded as an out-of-memory condition.
  std::unique_ptr<NonLocalStencilGeometry4D> geomF;
  std::unique_ptr<LittleDiracOperatorF> LittleDiracOpF;
  std::unique_ptr<TrivialPrecon<CoarseVectorF>> coarse_trivialF;
  std::unique_ptr<TrivialPrecon<LatticeFermionF>> fine_trivialF;
  std::unique_ptr<NonHermitianLinearOperator<LittleDiracOperatorF, CoarseVectorF>> LinOpCoarseF;
  std::unique_ptr<ShiftedNonHermitianLinearOperator<LittleDiracOperatorF, CoarseVectorF>> ShiftedLinOpCoarseF;
  // Workstream A on the fp32 hierarchy: stencil apply only (mrhs is fp64-only, see the header).
  typedef ProbeMG::StencilCoarseApply<vSpinColourVectorF, vTComplexF, 2 * kNbasis> StencilCoarseF_t;
  std::unique_ptr<StencilCoarseF_t> StencilCoarseF;
  std::unique_ptr<ProbeMG::CountingLinearOperator<CoarseVectorF>> CountedCoarseF;
  std::unique_ptr<ProbeMG::CountingLinearOperator<LatticeFermionF>> CountedSchurOpF;
  std::unique_ptr<ProbeMG::CountingLinearOperator<LatticeFermionF>> CountedShiftedSchurOpF;
  std::unique_ptr<PrecGeneralisedConjugateResidualNonHermitian<CoarseVectorF>> CoarseSolverStockF;
  std::unique_ptr<ProbeMG::FlexibleGCR<CoarseVectorF>> CoarseSolverFastF;
  std::unique_ptr<ShiftedSchurOperator<LatticeFermionF>> ShiftedSchurOpF;
  std::unique_ptr<PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionF>> SmootherStockF;
  std::unique_ptr<ProbeMG::FlexibleGCR<LatticeFermionF>> SmootherFastF;
  std::unique_ptr<MGPreconditioner<vSpinColourVectorF, vTComplexF, 2 * kNbasis>> PreconF;
  std::unique_ptr<PrecisionChangeAdaptor<LatticeFermionD, LatticeFermionF>> PreconMixed;
  double coarsen_f_seconds = 0.0;

  if (mg_single) {
    if (boss)
      std::cout << GridLogMessage << "--- building SINGLE-precision MG hierarchy ---" << std::endl;

    // Separate grids: vComplexF has twice vComplexD's SIMD lanes, so the F fields
    // need their own layout. precisionChange handles the cross-grid copy; this is
    // the same construction Grid's mixed-precision CG uses.
    UGridF = SpaceTimeGrid::makeFourDimGrid(
        GridDefaultLatt(), GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
    UrbGridF = SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);
    Coarse4dF = SpaceTimeGrid::makeFourDimGrid(
        clatt, GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
    subdivides(Coarse4dF, UrbGridF);

    UmuF.reset(new LatticeGaugeFieldF(UGridF));
    precisionChange(*UmuF, Umu);

    if (action_name == "wilson")
      fermopF.reset(new WilsonFermionF(*UmuF, *UGridF, *UrbGridF, mass, implParams));
    else
      fermopF.reset(new CompactWilsonCloverFermionF(*UmuF, *UGridF, *UrbGridF, mass, csw, csw, 1.0,
                                                    anisotropy, implParams));

    if (schur_name == "one")
      schur_holderF.reset(new SchurDiagOneOperator<WilsonFermionF, LatticeFermionF>(*fermopF));
    else
      schur_holderF.reset(new SchurDiagMooeeOperator<WilsonFermionF, LatticeFermionF>(*fermopF));

    // Subspace converted from the double one, vector by vector.
    //
    // Converting (rather than regenerating with CreateSubspaceGCR on the F
    // operator) is deliberate: both precisions then precondition with the SAME
    // null space, so the measurement isolates arithmetic precision alone. The
    // cost is that the fp64 hierarchy must be built first and stays resident --
    // see the memory note below, which is why this row does not fit at C3/16 GPU.
    CombinedUVF.reset(new CombinedSubspaceF(Coarse4dF, UrbGridF, cb));
    for (int b = 0; b < 2 * kNbasis; ++b) {
      CombinedUVF->subspace[b].Checkerboard() = cb;
      precisionChange(CombinedUVF->subspace[b], CombinedUV.subspace[b]);
    }

    geomF.reset(new NonLocalStencilGeometry4D(Coarse4dF, stencil_hops));
    LittleDiracOpF.reset(new LittleDiracOperatorF(*geomF, UrbGridF, Coarse4dF));
    if (boss) std::cout << GridLogMessage << "--- coarsening in single ---" << std::endl;
    accelerator_barrier();
    UGrid->Barrier();
    const double tf0 = usecond();
    LittleDiracOpF->CoarsenOperator(*schur_holderF, *CombinedUVF);
    accelerator_barrier();
    UGrid->Barrier();
    coarsen_f_seconds = (usecond() - tf0) / 1.0e6;
    if (boss)
      std::cout << GridLogMessage << "single coarsening " << coarsen_f_seconds << " s" << std::endl;

    coarse_trivialF.reset(new TrivialPrecon<CoarseVectorF>());
    fine_trivialF.reset(new TrivialPrecon<LatticeFermionF>());
    LinOpCoarseF.reset(
        new NonHermitianLinearOperator<LittleDiracOperatorF, CoarseVectorF>(*LittleDiracOpF));
    ShiftedLinOpCoarseF.reset(
        new ShiftedNonHermitianLinearOperator<LittleDiracOperatorF, CoarseVectorF>(*LittleDiracOpF, 0.001));
    // Workstream A, fp32: the stencil apply of the same fp32 coarse matrix (+0.001).
    if (coarse_apply == "stencil" || (coarse_apply_check > 0 && stencil_hops == 1))
      StencilCoarseF.reset(new StencilCoarseF_t(*LittleDiracOpF, Coarse4dF, 0.001));
    if (coarse_apply_check > 0 && StencilCoarseF) {
      if (boss) std::cout << GridLogMessage << "--- coarse apply check (fp32) ---" << std::endl;
      // CRNG lives on the fp64 coarse grid; draw there and convert, as the subspace is.
      CoarseVector c_probeD(Coarse4d);
      random(CRNG, c_probeD);
      CoarseVectorF c_probeF(Coarse4dF);
      precisionChange(c_probeF, c_probeD);
      ProbeMG::check_coarse_apply<CoarseVectorF>("stencil", *ShiftedLinOpCoarseF, *StencilCoarseF,
                                                 c_probeF, UGrid, coarse_apply_check, boss);
    }
    LinearOperatorBase<CoarseVectorF> &SelectedCoarseF =
        (StencilCoarseF && coarse_apply == "stencil")
            ? static_cast<LinearOperatorBase<CoarseVectorF> &>(*StencilCoarseF)
            : static_cast<LinearOperatorBase<CoarseVectorF> &>(*ShiftedLinOpCoarseF);
    // The fp32 hierarchy shares the SAME counters as the fp64 one: only one of the two is ever
    // driven by the outer solver (OuterPrecon selects), so the totals stay unambiguous.
    CountedCoarseF.reset(
        new ProbeMG::CountingLinearOperator<CoarseVectorF>(SelectedCoarseF, counts.coarse));
    CountedSchurOpF.reset(
        new ProbeMG::CountingLinearOperator<LatticeFermionF>(*schur_holderF, counts.fine));
    CoarseSolverStockF.reset(new PrecGeneralisedConjugateResidualNonHermitian<CoarseVectorF>(
        coarse_tol, coarse_maxiter, *CountedCoarseF, *coarse_trivialF, coarse_mmax, coarse_nstep));
    CoarseSolverStockF->Level(3);
    CoarseSolverFastF.reset(new ProbeMG::FlexibleGCR<CoarseVectorF>(
        coarse_tol, coarse_maxiter, *CountedCoarseF, *coarse_trivialF, coarse_mmax, coarse_nstep));
    CoarseSolverFastF->Level(3);
    CoarseSolverFastF->verify_residual = (verify_residual != 0);
    CoarseSolverFastF->zero_guess = true;

    ShiftedSchurOpF.reset(new ShiftedSchurOperator<LatticeFermionF>(*schur_holderF, 0.01));
    CountedShiftedSchurOpF.reset(
        new ProbeMG::CountingLinearOperator<LatticeFermionF>(*ShiftedSchurOpF, counts.fine));
    SmootherStockF.reset(new PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionF>(
        smoother_tol, smoother_maxiter, *CountedShiftedSchurOpF, *fine_trivialF, smoother_mmax, smoother_nstep));
    SmootherStockF->Level(2);
    SmootherFastF.reset(new ProbeMG::FlexibleGCR<LatticeFermionF>(
        smoother_tol, smoother_maxiter, *CountedShiftedSchurOpF, *fine_trivialF, smoother_mmax, smoother_nstep));
    SmootherFastF->Level(2);
    SmootherFastF->verify_residual = (verify_residual != 0);
    SmootherFastF->zero_guess = true;

    LinearFunction<CoarseVectorF> &CoarseSolverFsel =
        fast_gcr ? static_cast<LinearFunction<CoarseVectorF> &>(*CoarseSolverFastF)
                 : static_cast<LinearFunction<CoarseVectorF> &>(*CoarseSolverStockF);
    LinearFunction<LatticeFermionF> &SmootherFsel =
        fast_gcr ? static_cast<LinearFunction<LatticeFermionF> &>(*SmootherFastF)
                 : static_cast<LinearFunction<LatticeFermionF> &>(*SmootherStockF);

    PreconF.reset(new MGPreconditioner<vSpinColourVectorF, vTComplexF, 2 * kNbasis>(
        *CombinedUVF, *CountedSchurOpF, *fine_trivialF, SmootherFsel, *LinOpCoarseF, CoarseSolverFsel));
    PreconF->fast_project = (fast_project != 0);
    PreconF->project_mode = fast_project;
    PreconF->instrument = (instrument_vcycle != 0);
    PreconF->persistent_temps = (persistent_temps != 0);
    PreconMixed.reset(
        new PrecisionChangeAdaptor<LatticeFermionD, LatticeFermionF>(*PreconF, UrbGridF));

    // ⛔ RELEASE THE DOUBLE HIERARCHY'S DEVICE MEMORY -- WITHOUT THIS THE fp32 ROW
    // CANNOT RUN AT C3. Both hierarchies are live at this point: the fp64 coarse
    // operator (geom.npoint x 48 x 48 complex per coarse site, in _A and _Adag)
    // plus its 48-vector fine subspace, and now the fp32 equivalents. Measured
    // 2026-09-15: OOM immediately after "single coarsening" at device-mem 16000
    // (raw cudaMalloc failure, Lattice_base.h) and at 13000; at 10000 a single
    // allocation exceeds the cache instead (bytes<DeviceMaxBytes,
    // MemoryManagerCache.cc). No cache size fixes it because the total genuinely
    // does not fit -- the fp64 copy has to go.
    //
    // Safe because in the fp32 path nothing downstream touches them: the outer
    // solver preconditions through PreconMixed, and the Galerkin check has already
    // run above. `Precon` itself is never applied (OuterPrecon selects PreconMixed).
    // The double SchurOp is still needed -- it IS the operator being solved.
    // ⛔⛔ DO NOT TRY TO FREE THE fp64 HIERARCHY HERE. Both attempts failed, and
    // both failed as CORRUPTION rather than as a clean error:
    //   CombinedUV.subspace.clear() -> Aggregation reads subspace[0].Checkerboard()
    //       unguarded (Aggregates.h:79) => request to allocate 49 TB.
    //   LittleDiracOp._A/_Adag.clear() -> still read afterwards => request to
    //       allocate 2.7 EB ("EvictVictims bytes 2688987614179115008").
    // Both look superficially like memory pressure and are not; a garbage
    // allocation SIZE in EvictVictims means use-after-free, so do not respond to
    // it by tuning --device-mem.
    //
    // The real constraint: at C3 the fp32 row needs BOTH hierarchies resident
    // (the fp64 one is built first and the subspace is converted from it), plus a
    // second set of grids whose halo buffers live OUTSIDE the managed cache. That
    // does not fit in 40 GB at 16 GPUs. Options, none of them a one-liner:
    //   - generate the fp32 subspace directly with CreateSubspaceGCR on the F
    //     operator, so the fp64 hierarchy is never built (changes what is being
    //     compared: QUDA also generates null vectors at reduced precision, so this
    //     is arguably MORE faithful, but it is a different measurement);
    //   - run the fp32 row at more GPUs, where per-rank volume is smaller;
    //   - run it at C2, where both hierarchies fit.
  }

  // The outer solver's preconditioner: the double hierarchy, or the single one
  // behind its precision-changing adaptor.
  LinearFunction<LatticeFermionD> &OuterPrecon =
      mg_single ? static_cast<LinearFunction<LatticeFermionD> &>(*PreconMixed)
                : static_cast<LinearFunction<LatticeFermionD> &>(Precon);

  PrecGeneralisedConjugateResidualNonHermitian<LatticeFermionD> MGSolverStock(
      tol, maxiter, CountedSchurOp, OuterPrecon, outer_mmax, outer_nstep);
  MGSolverStock.Level(1);
  ProbeMG::FlexibleGCR<LatticeFermionD> MGSolverFast(
      tol, maxiter, CountedSchurOp, OuterPrecon, outer_mmax, outer_nstep);
  MGSolverFast.Level(1);
  // The outer solver keeps its residual report: it runs ONCE per solve, and it is the
  // independent confirmation that the recurrence residual has not drifted.
  MGSolverFast.verify_residual = true;
  // ⛔ The OUTER solver does NOT get zero_guess. mg_sol is zeroed before the first call, but
  // GCRnStep is re-entered on restart with a non-zero iterate, and skipping the residual
  // computation there would solve the wrong system. FlexibleGCR additionally gates D1 on
  // k == 0; this is the second half of the same guard.
  MGSolverFast.zero_guess = false;
  MGSolverFast.instrument = (instrument_outer != 0);
  LinearFunction<LatticeFermionD> &MGSolver =
      fast_gcr ? static_cast<LinearFunction<LatticeFermionD> &>(MGSolverFast)
               : static_cast<LinearFunction<LatticeFermionD> &>(MGSolverStock);

  // ---- Solve ---------------------------------------------------------------
  //
  // ⛔ RESEED IMMEDIATELY BEFORE THE DRAW. Two reasons, both bugs before this line
  // existed: (1) RNG4 has already been consumed by CreateSubspaceGCR above, so the
  // source silently depended on how many vectors the setup drew; (2) the QUDA probe
  // reseeds to these same integers before ITS draw, so without this the two probes
  // solved DIFFERENT right-hand sides and no Grid-vs-QUDA number was like-for-like.
  // Keep these seeds identical to probe_quda_mg_clover.cc.
  RNG4.SeedFixedIntegers(std::vector<int>({11, 22, 33, 44}));
  LatticeFermionD full_src(UGrid);
  random(RNG4, full_src);
  LatticeFermionD src(UrbGrid);
  src.Checkerboard() = cb;
  pickCheckerboard(cb, src, full_src);
  const double source_norm2 = norm2(src);

  LatticeFermionD mg_sol(UrbGrid);
  mg_sol.Checkerboard() = cb;
  mg_sol = Zero();

  // ---- D7 transfer microbenchmark -----------------------------------------
  //
  // Restriction and prolongation are adjoints over the same data and are called
  // the same number of times per V-cycle, so their costs should be comparable.
  // They are not (35.1% vs 11.8%). This times the three implementations in
  // isolation, and checks the fused one against blockProjectFast numerically --
  // they agree to rounding, not exactly, because the block reduction order differs.
  const int transfer_bench = read_int(argc, argv, "--probe-transfer-bench", 0);
  if (transfer_bench > 0) {
    CoarseVector c_fast(Coarse4d), c_fused(Coarse4d);
    LatticeFermionD f_tmp(UrbGrid);
    f_tmp.Checkerboard() = cb;
    LatticeFermionD f_probe(UrbGrid);
    f_probe.Checkerboard() = cb;
    random(RNG4, full_src);
    pickCheckerboard(cb, f_probe, full_src);

    // Warm every path before timing any of it.
    blockProjectFast(c_fast, f_probe, CombinedUV.subspace);
    ProbeMG::blockProjectFused(c_fused, f_probe, CombinedUV.subspace);
    CombinedUV.PromoteFromSubspace(c_fast, f_tmp);
    accelerator_barrier();
    UGrid->Barrier();

    auto time_it = [&](const char *name, auto &&fn) {
      accelerator_barrier();
      UGrid->Barrier();
      const double s = usecond();
      for (int i = 0; i < transfer_bench; ++i) fn();
      accelerator_barrier();
      UGrid->Barrier();
      const double ms = (usecond() - s) / 1.0e3 / double(transfer_bench);
      if (boss)
        std::cout << GridLogMessage << "transfer bench  " << name << "  " << ms << " ms/call"
                  << std::endl;
      return ms;
    };

    const double ms_fast =
        time_it("blockProjectFast ", [&] { blockProjectFast(c_fast, f_probe, CombinedUV.subspace); });
    const double ms_fused = time_it(
        "blockProjectFused", [&] { ProbeMG::blockProjectFused(c_fused, f_probe, CombinedUV.subspace); });
    const double ms_prom =
        time_it("blockPromote     ", [&] { CombinedUV.PromoteFromSubspace(c_fast, f_tmp); });

    // Numerical agreement gate: relative difference of the two coarse results.
    blockProjectFast(c_fast, f_probe, CombinedUV.subspace);
    ProbeMG::blockProjectFused(c_fused, f_probe, CombinedUV.subspace);
    CoarseVector c_diff(Coarse4d);
    c_diff = c_fast - c_fused;
    const double rel = std::sqrt(norm2(c_diff) / norm2(c_fast));
    if (boss) {
      std::cout << GridLogMessage << "transfer bench  fused/fast speedup " << (ms_fast / ms_fused)
                << "x, project/promote was " << (ms_fast / ms_prom) << "x, now "
                << (ms_fused / ms_prom) << "x" << std::endl;
      std::cout << GridLogMessage << "transfer bench  fused-vs-fast relative difference " << rel
                << (rel < 1.0e-10 ? "  PASSED" : "  FAILED") << std::endl;
    }
  }

  // ---- FULLY SINGLE-PRECISION PATH ----------------------------------------
  //
  // WHY. `MG_PRECISION=single` alone gives an fp32 HIERARCHY under an fp64 OUTER
  // solver, while QUDA's MG *requires* the outer sloppy precision to match the
  // hierarchy (SLOPPY=double + MG aborts, "Precisions 4 8 do not match",
  // coarse_op_24.cu:115). So Grid-single vs QUDA-single still compared an fp64
  // outer against an fp32-sloppy one, in QUDA's favour. fp64-vs-fp64 cannot fix
  // it either: QUDA's fp64 MG is not compiled (multigrid.h:10). The only way to
  // match both sides is to run EVERYTHING in fp32, which is what this does.
  //
  // ⚠️ fp32 cannot reach 1e-10 -- run all four comparisons at TOL=1e-6.
  // The independent residual is still evaluated in fp64 on the promoted solution,
  // so the gate stays honest even though the solve is fp32 throughout.
  if (outer_single) {
    if (!mg_single) {
      if (boss)
        std::cout << GridLogMessage
                  << "ERROR: --probe-outer-precision single requires --probe-mg-precision single"
                  << std::endl;
      Grid_finalize();
      return 1;
    }
    LatticeFermionF srcF(UrbGridF), solF(UrbGridF);
    srcF.Checkerboard() = cb;
    solF.Checkerboard() = cb;
    precisionChange(srcF, src);
    const double srcF_norm2 = norm2(srcF);

    ProbeMG::FlexibleGCR<LatticeFermionF> MGSolverF(tol, maxiter, *CountedSchurOpF, *PreconF,
                                                    outer_mmax, outer_nstep);
    MGSolverF.Level(1);
    MGSolverF.zero_guess = false;

    // MG: warm, then timed repeats, median -- identical treatment to the fp64 path.
    solF = Zero();
    MGSolverF(srcF, solF);
    std::vector<double> mgF_times;
    int mgF_steps = 0;
    long long mgF_fine = 0, mgF_coarse = 0;
    for (int r = 0; r < solve_repeats; ++r) {
      solF = Zero();
      counts.reset();
      accelerator_barrier();
      UGrid->Barrier();
      const double s = usecond();
      MGSolverF(srcF, solF);
      accelerator_barrier();
      UGrid->Barrier();
      mgF_times.push_back((usecond() - s) / 1.0e6);
      mgF_steps = MGSolverF.steps;
      mgF_fine = counts.fine;
      mgF_coarse = counts.coarse;
    }
    // Grade in fp64 on the promoted solution against the fp64 operator.
    LatticeFermionD promoted(UrbGrid), resid(UrbGrid);
    promoted.Checkerboard() = cb;
    resid.Checkerboard() = cb;
    precisionChange(promoted, solF);
    SchurOp.Op(promoted, resid);
    resid = resid - src;
    const double mgF_residual = std::sqrt(norm2(resid) / source_norm2);

    // CG on Mpc^dag Mpc, also fully fp32.
    std::vector<double> cgF_times;
    long long cgF_iters = 0;
    double cgF_residual = 0.0;
    if (run_cg) {
      LatticeFermionF cgF(UrbGridF);
      cgF.Checkerboard() = cb;
      ConjugateGradient<LatticeFermionF> CGF(tol, cg_maxiter, false);
      cgF = Zero();
      CGF(*CountedSchurOpF, srcF, cgF);
      for (int r = 0; r < solve_repeats; ++r) {
        cgF = Zero();
        accelerator_barrier();
        UGrid->Barrier();
        const double s = usecond();
        CGF(*CountedSchurOpF, srcF, cgF);
        accelerator_barrier();
        UGrid->Barrier();
        cgF_times.push_back((usecond() - s) / 1.0e6);
        cgF_iters = static_cast<long long>(CGF.IterationsToComplete);
      }
      precisionChange(promoted, cgF);
      SchurOp.HermOp(promoted, resid);
      resid = resid - src;
      cgF_residual = std::sqrt(norm2(resid) / source_norm2);
    }

    if (boss) {
      std::cout << GridLogMessage << "=== SUMMARY (FULLY fp32: hierarchy + outer + CG) ==="
                << std::endl;
      std::cout << GridLogMessage << "action                  " << action_name << std::endl;
      std::cout << GridLogMessage << "tolerance               " << tol
                << "  (fp32 cannot reach 1e-10)" << std::endl;
      std::cout << GridLogMessage << "MG solve (Mpc)          " << median_of(mgF_times) << " s, "
                << mgF_steps << " outer steps, fp64-graded residual " << mgF_residual << std::endl;
      if (mgF_steps > 0)
        std::cout << GridLogMessage << "operator applications   fine " << mgF_fine << " ("
                  << double(mgF_fine) / double(mgF_steps) << " / V-cycle), coarse " << mgF_coarse
                  << std::endl;
      if (run_cg)
        std::cout << GridLogMessage << "CG solve (Mpc^dag Mpc)  " << median_of(cgF_times) << " s, "
                  << cgF_iters << " iters, fp64-graded residual " << cgF_residual << std::endl;
      const bool ok = (mgF_residual <= 1.0e-4) && (!run_cg || cgF_residual <= 1.0e-4);
      std::cout << GridLogMessage << "PROBE RESULT: " << (ok ? "PASS" : "FAIL") << std::endl;
    }
    Grid_finalize();
    return 0;
  }

  if (boss) std::cout << GridLogMessage << "--- MG-preconditioned GCR solve of Mpc ---" << std::endl;

  // ⛔ WARM SOLVE, UNTIMED. Without it the first timed solve carries CUDA autotune
  // and first-touch allocation. This also removes an asymmetry that used to flatter
  // CG: the MG solve ran FIRST, so the CG reference below inherited an already-warm
  // machine while MG paid for warming it. Both are now warmed and both take the
  // median of `solve_repeats`, matching the QUDA probe.
  MGSolver(src, mg_sol);

  if (instrument_outer) MGSolverFast.reset_timers();
  // NVTX scoping for nsys. Setup is 7-10 s against a 0.08 s solve, so whole-run
  // kernel statistics are useless here -- the range is what makes it possible to
  // ask "how many kernels, and how much wall time between them" for the SOLVE.
  // Compiled out unless -DBENCH_NVTX; see bench_nvtx.h.
  // ⛔ Profiler overhead is NOT backend-neutral (nsys inflated Grid 3.5-5.2% but
  // QUDA 0.9-1.7% at C2/1 GPU) -- use this for WITHIN-run attribution only, never
  // for a Grid-vs-QUDA ratio.
  std::vector<double> mg_times;
  long long mg_fine_applies = 0;
  long long mg_coarse_applies = 0;
  int mg_steps = 0;
  for (int r = 0; r < solve_repeats; ++r) {
    mg_sol = Zero();
    // Reset AFTER setup: CoarsenOperator applies Mpc npoint*2*nbasis times, and
    // CreateSubspaceGCR far more, neither of which belongs in the per-V-cycle count.
    counts.reset();
    accelerator_barrier();
    UGrid->Barrier();
    t0 = usecond();
    {
      BENCH_NVTX_RANGE("mg_solve");
      MGSolver(src, mg_sol);
    }
    accelerator_barrier();
    UGrid->Barrier();
    mg_times.push_back((usecond() - t0) / 1.0e6);
    // Deterministic across repeats (same source, seeded setup); taking the last is
    // equivalent to taking any, and a disagreement here would itself be a finding.
    mg_fine_applies = counts.fine;
    mg_coarse_applies = counts.coarse;
    mg_steps = fast_gcr ? MGSolverFast.steps : MGSolverStock.steps;
  }
  const double mg_seconds = median_of(mg_times);

  // Independent residual, outside every timed region, evaluated with Grid's own
  // Schur operator -- the MG row solves Mpc, so it is graded against Mpc.
  LatticeFermionD residual(UrbGrid);
  residual.Checkerboard() = cb;
  SchurOp.Op(mg_sol, residual);
  residual = residual - src;
  const double mg_residual = std::sqrt(norm2(residual) / source_norm2);

  // ---- Reference CG on Mpc^dag Mpc ----------------------------------------
  //
  // NOT a like-for-like comparison -- different operator, different Krylov method.
  // It is here to place the MG time on a familiar scale, and because it is what
  // the harness's quantity-1 control will be.
  double cg_seconds = 0.0;
  double cg_residual = 0.0;
  long long cg_iterations = 0;
  long long cg_fine_applies = 0;
  const bool cg_double = (cg_precision == "double" || cg_precision == "both");
  const bool cg_mixed = (cg_precision == "mixed" || cg_precision == "both");
  if (run_cg && cg_double) {
    LatticeFermionD cg_sol(UrbGrid);
    cg_sol.Checkerboard() = cb;
    cg_sol = Zero();
    ConjugateGradient<LatticeFermionD> CG(tol, cg_maxiter, false);
    if (boss) std::cout << GridLogMessage << "--- reference CG on Mpc^dag Mpc ---" << std::endl;
    // Warm solve, untimed -- same treatment as the MG row above, so neither solver
    // is charged for warming the machine on the other's behalf.
    CG(CountedSchurOp, src, cg_sol);

    std::vector<double> cg_times;
    for (int r = 0; r < solve_repeats; ++r) {
      cg_sol = Zero();
      // Counted too: CG's applications-per-second is the calibration that converts the MG
      // application COUNT into a time, and it is the only place a bare Mpc cost can be read off.
      counts.reset();
      accelerator_barrier();
      UGrid->Barrier();
      t0 = usecond();
      CG(CountedSchurOp, src, cg_sol);
      accelerator_barrier();
      UGrid->Barrier();
      cg_times.push_back((usecond() - t0) / 1.0e6);
      cg_iterations = static_cast<long long>(CG.IterationsToComplete);
      cg_fine_applies = counts.fine;
    }
    cg_seconds = median_of(cg_times);

    SchurOp.HermOp(cg_sol, residual);
    residual = residual - src;
    cg_residual = std::sqrt(norm2(residual) / source_norm2);
  }

  // ---- Mixed-precision CG on Mpc^dag Mpc -----------------------------------
  //
  // Grid's stock MixedPrecisionConjugateGradient, built exactly as the production driver's
  // MixedPrecCGWrapper builds it: fp32 inner CG on the fp32 Schur operator, fp64 residual
  // restarts on the fp64 one, then a final fp64 patch-up CG. Same source, same fp64 grading.
  struct MixedCGRow {
    double inner_tol, seconds, residual;
    long long inner, restarts, final_iters;
  };
  std::vector<MixedCGRow> cgm_rows;
  if (run_cg && cg_mixed) {
    LatticeFermionD cgm_sol(UrbGrid);
    cgm_sol.Checkerboard() = cb;
    for (const double inner_tol : cg_inner_tols) {
      MixedPrecisionConjugateGradient<LatticeFermionD, LatticeFermionF> MPCG(
          tol, cg_maxiter, cg_mixed_outer, UrbGridF, *CountedSchurOpF, CountedSchurOp);
      MPCG.InnerTolerance = inner_tol;
      if (boss)
        std::cout << GridLogMessage << "--- mixed-precision CG on Mpc^dag Mpc, inner tol " << inner_tol
                  << " ---" << std::endl;
      cgm_sol = Zero();
      MPCG(src, cgm_sol);  // warm, untimed

      MixedCGRow row{inner_tol, 0.0, 0.0, 0, 0, 0};
      std::vector<double> cgm_times;
      for (int r = 0; r < solve_repeats; ++r) {
        cgm_sol = Zero();
        accelerator_barrier();
        UGrid->Barrier();
        t0 = usecond();
        MPCG(src, cgm_sol);
        accelerator_barrier();
        UGrid->Barrier();
        cgm_times.push_back((usecond() - t0) / 1.0e6);
        row.inner = static_cast<long long>(MPCG.TotalInnerIterations);
        row.restarts = static_cast<long long>(MPCG.TotalOuterIterations);
        row.final_iters = static_cast<long long>(MPCG.TotalFinalStepIterations);
      }
      row.seconds = median_of(cgm_times);

      SchurOp.HermOp(cgm_sol, residual);
      residual = residual - src;
      row.residual = std::sqrt(norm2(residual) / source_norm2);
      cgm_rows.push_back(row);
    }
  }

  // ---- Summary -------------------------------------------------------------
  const bool converged = mg_residual <= std::max(1.0e-8, 100.0 * tol);

  // ⛔ The Galerkin check is a GATE only at hops >= 2.
  //
  // At hops=1 the 9-point coarse stencil cannot represent the 2-hop Mpc exactly, so the check
  // fails BY CONSTRUCTION (4.68e-2 at C3) -- and hops=1 is the TUNED configuration, being an
  // equally effective preconditioner at 3.2x less time (results doc §5). Grading on it made
  // every tuned run report `PROBE RESULT: FAIL` while passing its actual correctness test,
  // which is the independent residual. The must-fail control is still required at every hops,
  // since a check that cannot distinguish Mpc from Mpc^dag Mpc proves nothing.
  const bool galerkin_control_ok = (galerkin_herm > 1.0e-3);
  const bool probe_passed =
      converged && galerkin_control_ok && ((stencil_hops < 2) || galerkin_passed);
  if (boss) {
    std::cout << GridLogMessage << "=== SUMMARY ===" << std::endl;
    std::cout << GridLogMessage << "action                  " << action_name << std::endl;
    std::cout << GridLogMessage << "checkerboard            " << cb_name << std::endl;
    std::cout << GridLogMessage << "stencil hops / npoint   " << stencil_hops << " / " << geom.npoint
              << std::endl;
    std::cout << GridLogMessage << "coarse apply            " << coarse_apply << std::endl;
    std::cout << GridLogMessage << "MG precondition prec    " << precision_name
              << (mg_single ? " (hierarchy fp32; outer solver + gates fp64)" : " (fp64 throughout)")
              << std::endl;
    std::cout << GridLogMessage << "subspace generation     " << subspace_seconds << " s" << std::endl;
    std::cout << GridLogMessage << "coarsen operator        " << coarsen_seconds << " s" << std::endl;
    if (mg_single)
      std::cout << GridLogMessage << "coarsen operator (fp32) " << coarsen_f_seconds << " s" << std::endl;
    std::cout << GridLogMessage << "setup total             "
              << (subspace_seconds + coarsen_seconds + coarsen_f_seconds) << " s" << std::endl;
    std::cout << GridLogMessage << "MG solve (Mpc)          " << mg_seconds << " s, " << mg_steps
              << " outer PGCR steps, independent residual " << mg_residual << std::endl;
    if (instrument_vcycle) {
      if (mg_single)
        PreconF->report_breakdown();
      else
        Precon.report_breakdown();
    } else {
      std::cout << GridLogMessage
                << "  V-cycle breakdown SUPPRESSED (--probe-instrument-vcycle 0): its 14 "
                   "accelerator_barrier() calls per cycle are inside the timed region"
                << std::endl;
    }
    if (instrument_outer && fast_gcr)
      std::cout << GridLogMessage << MGSolverFast.timer_report() << std::endl;

    // ---- Operator-application census (Stage 0) ----------------------------
    //
    // This is the quantity the whole diagnosis turns on. The baseline prediction at C3 clover
    // is 9 fine / 4 coarse PER V-CYCLE; the cleaned path should read 7 / 2.
    if (mg_steps > 0) {
      const double per_cycle_fine = double(mg_fine_applies) / double(mg_steps);
      const double per_cycle_coarse = double(mg_coarse_applies) / double(mg_steps);
      std::cout << GridLogMessage << "operator applications   fine " << mg_fine_applies
                << " (" << per_cycle_fine << " / V-cycle), coarse " << mg_coarse_applies << " ("
                << per_cycle_coarse << " / V-cycle)" << std::endl;
    }

    if (run_cg && cg_double) {
      std::cout << GridLogMessage << "CG solve (Mpc^dag Mpc)  " << cg_seconds << " s, " << cg_iterations
                << " iters, independent residual " << cg_residual << std::endl;
      // Bare cost of one Mpc application, the calibration that turns the MG census into a time.
      if (cg_fine_applies > 0)
        std::cout << GridLogMessage << "CG operator cost        " << cg_fine_applies
                  << " Mpc applications, " << (1.0e3 * cg_seconds / double(cg_fine_applies))
                  << " ms each" << std::endl;
    }
    for (const auto &row : cgm_rows)
      std::cout << GridLogMessage << "CG mixed solve (Mpc^dag Mpc) " << row.seconds << " s, "
                << (row.inner + row.final_iters) << " iters (" << row.inner << " fp32 inner, "
                << row.restarts << " restarts, " << row.final_iters << " fp64 final), inner tol "
                << row.inner_tol << ", independent residual " << row.residual << std::endl;
    std::cout << GridLogMessage << "Galerkin check          " << (galerkin_passed ? "PASSED" : "FAILED")
              << (stencil_hops >= 2 ? " (gate)" : " (diagnostic only at hops=1)") << std::endl;
    std::cout << GridLogMessage << "MG converged            " << (converged ? "YES" : "NO") << std::endl;
    std::cout << GridLogMessage << "PROBE RESULT: " << (probe_passed ? "PASS" : "FAIL") << std::endl;
  }

  Grid_finalize();
  return probe_passed ? 0 : 1;
}
