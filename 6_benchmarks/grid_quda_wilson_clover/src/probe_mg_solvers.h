// probe_mg_solvers.h
//
// Probe-local solver components for the MG arm of the Grid-vs-QUDA campaign.
// Diagnosis and staged plan: __docs/2026_09_15_grid_mg_fix_plan.md
//
// WHY THIS FILE EXISTS.
//
// Reading the PGCR trace timestamps in runs/2026_9_11_grid_mg_probe/rep_fp64_1/probe.log
// against the probe's own barriered breakdown shows that ONE V-CYCLE issues
//
//     9 fine Mpc applications and 4 coarse Mults,
//
// and that TWO OF EACH ARE UNNECESSARY. The probe's `fine operator` timer sees 2 of the 9,
// which is where the published "the fine operator is 4-6% of Grid's MG solve" comes from.
//
// Grid's PrecGeneralisedConjugateResidualNonHermitian is a header-only template, so the fixed
// version lives here rather than as an edit to Grid/. That means NO GRID REBUILD, and the
// upstream clone in Grid/ -- which we do not commit to -- stays untouched.
//
// ⚠️ THE CONTROL IS STOCK GRID, NOT AN EMULATION OF IT. The probe constructs BOTH this class
// and Grid's own, and selects between them at runtime (--probe-fast-gcr). An earlier draft of
// this file carried an `if (!fast)` path that reproduced Grid's behaviour inside FlexibleGCR;
// that was abandoned because it silently dropped Grid's per-call allocation of q(mmax)+p(mmax)
// -- the single largest item in D4 -- and so would have made the control faster than stock
// Grid and understated the very effect being measured.
//
// WHAT IS REMOVED RELATIVE TO GRID, and why each is an ALGEBRAIC IDENTITY rather than a tuning
// change. Line numbers are
// Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h.
//
//   D1  :149  `Linop.Op(psi,Az)` forms the initial residual r = src - A.psi. Both inner
//             solvers are entered with psi identically Zero (the probe zeroes Csol before the
//             coarse solve and vec2 before the smoother), so this computes A.0 and subtracts
//             nothing. Costs 1 fine + 1 coarse application per V-cycle.
//             ⛔ The OUTER solver's psi is NOT zero on restart, so this may only be skipped on
//             the FIRST GCRnStep of a solve -- see `zero_guess` and the k==0 gate below.
//
//   D2  :102-104  On the converged exit, operator() applies the operator AGAIN plus an axpy
//             and a norm2, purely so the log line can print "true residual" beside the
//             recurrence residual it already holds. The coarse solver converges on every
//             V-cycle, so it pays a whole extra coarse Mult every time.
//             Evidence it is real: the smoother's CONVERGING cycle took 31.4 ms and its
//             non-converging one 27.1 ms -- the 4.3 ms difference is exactly one Mpc.
//             Kept available behind `verify_residual` for debugging.
//
//   D4  :129-139  r, z, tmp, ttmp, Az plus q(mmax) and p(mmax) are constructed on entry and
//             destroyed on exit of EVERY call. `tmp` and `ttmp` are never used at all.
//             Destruction measured at ~0.3 ms per field INDEPENDENT OF SIZE (21 coarse fields
//             = 6.1 ms; 105 fields at the old mmax=50 = 65.9 ms of dead time between the
//             `step[2]` log line and the `PGCR(50,50)` one, with no computation in between).
//             ⇒ the mmax 50 -> 8 gain recorded in the results doc §5 was per-object
//             bookkeeping, not allocation volume, which is why mmax=4 was WORSE.
//             Also: Lattice(GridBase*) is not explicit (Lattice_base.h:264), so
//             `std::vector<Field> q(mmax,grid)` selects the (count,value) overload and
//             DEEP-COPIES mmax times. The unique_ptr storage below avoids both.
//
//   D5  :151, :217  `zAAz = norm2(Az)` -- the value is read at neither site (the one at :175
//             IS used, for qq[0], and is kept). :158 evaluates norm2(r) INSIDE the log
//             statement, so a global reduction is paid to print a number.
//
// WHAT IS NOT CHANGED: the flexible-orthogonalisation recursion is copied verbatim, including
// its use of classical (not modified) Gram-Schmidt -- `b` is formed against the ORIGINAL Az,
// not the partially-updated q[peri_kp]. Changing that would change the iterates.
//
// ⇒ THE GATE. Against the fp64 C3 clover baseline these must be UNCHANGED:
//       11 outer PGCR steps, independent residual 2.711e-11.
//    Any movement is a bug, not a result.

#pragma once

#include <Grid/Grid.h>

#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace ProbeMG {

using namespace Grid;

// ---------------------------------------------------------------------------
// D7: fused block projection (restriction)
// ---------------------------------------------------------------------------
//
// MEASURED 2026-09-18, smeared Wilson 16^3x48, 1 GPU: `project` is 35.1% of the
// V-cycle while `promote` -- its ADJOINT, same data, same call count -- is 11.8%.
// The asymmetry is structural, not algorithmic:
//
//   blockPromote  (Lattice_transfer.h:641)  ONE accelerator_for over FINE sites.
//       The nbasis basis vectors are gathered into a device view array and all
//       accumulated in a SINGLE pass.  One kernel, no temporaries.
//
//   blockProjectFast (Lattice_transfer.h:1809)  a HOST-side loop over nbasis,
//       each iteration calling blockInnerProductD (:398), which for EVERY basis
//       vector:
//         - allocates a FULL FINE-SIZED temporary `fine_inner`,
//         - writes it   (localInnerProductD, one full fine pass),
//         - reads it    (blockSum, another full fine pass),
//         - runs a third kernel to convert into the coarse field.
//       At nbasis=48 that is ~48 fine-sized allocations, ~96 full fine-sized
//       passes and ~190 kernel launches per blockProject call.
//
// This routine does what blockPromote already does, in the other direction: one
// kernel, no fine-sized temporaries, the basis gathered into a device view array.
// It parallelises over (coarse site x basis vector) -- nbasis times more threads
// than a coarse-site-only decomposition, which matters here because the coarse
// grid is small (2304 sites at this volume, far below what an A100 wants).
//
// ⚠️ It still READS fineData once per basis vector; what it removes is the
// fine-sized WRITE + re-READ of `fine_inner`, the allocations, and the launches.
// A variant reading fineData exactly once would need nbasis accumulators live per
// thread (96 reals at nbasis=48) and would spill, so it is not obviously better --
// hence the microbenchmark rather than an assumption.
//
// ⛔ NOT bit-identical to blockProjectFast: the reduction order over the block
// differs, so results agree to rounding, not exactly.
template <class vobj, class CComplex, int nbasis, class VLattice>
inline void blockProjectFused(Lattice<iVector<CComplex, nbasis>> &coarseData,
                              const Lattice<vobj> &fineData,
                              const VLattice &Basis)
{
  GridBase *fine = fineData.Grid();
  GridBase *coarse = coarseData.Grid();
  const int _ndimension = coarse->_ndimension;

  GRID_ASSERT(nbasis == static_cast<int>(Basis.size()));
  subdivides(coarse, fine);

  Coordinate block_r(_ndimension);
  int blockVol = 1;
  for (int d = 0; d < _ndimension; d++) {
    block_r[d] = fine->_rdimensions[d] / coarse->_rdimensions[d];
    blockVol *= block_r[d];
  }

  autoView(fineData_, fineData, AcceleratorRead);
  autoView(coarseData_, coarseData, AcceleratorWrite);

  // Same device-view-array idiom blockPromote uses (Lattice_transfer.h:631-637).
  typedef LatticeView<vobj> Vview;
  std::vector<Vview> basis_views_h;
  basis_views_h.reserve(nbasis);
  for (int v = 0; v < nbasis; v++) basis_views_h.push_back(Basis[v].View(AcceleratorRead));
  static deviceVector<Vview> basis_views;
  basis_views.resize(nbasis);
  acceleratorCopyToDevice(&basis_views_h[0], &basis_views[0], nbasis * sizeof(Vview));
  auto Basis_p = &basis_views[0];

  Coordinate frdimensions = fine->_rdimensions;
  Coordinate crdimensions = coarse->_rdimensions;

  accelerator_for2d(sc, coarse->oSites(), v, nbasis, vobj::Nsimd(), {
    Coordinate coor_c(_ndimension);
    Coordinate coor_b(_ndimension);
    Coordinate coor_f(_ndimension);
    Lexicographic::CoorFromIndex(coor_c, sc, crdimensions);

    decltype(innerProduct(Basis_p[0](0), fineData_(0))) sum = Zero();
    for (int b = 0; b < blockVol; b++) {
      Lexicographic::CoorFromIndex(coor_b, b, block_r);
      for (int d = 0; d < _ndimension; d++) coor_f[d] = coor_c[d] * block_r[d] + coor_b[d];
      int sf;
      Lexicographic::IndexFromCoor(coor_f, sf, frdimensions);
      sum = sum + innerProduct(Basis_p[v](sf), fineData_(sf));
    }
    convertType(coarseData_[sc](v), sum);
  });

  for (int v = 0; v < nbasis; v++) basis_views_h[v].ViewClose();
}

// ---------------------------------------------------------------------------
// Operator-application counters (Stage 0)
// ---------------------------------------------------------------------------
//
// The whole diagnosis rests on a COUNT, so the count is measured rather than inferred. Every
// claimed saving is then confirmed structurally -- "two fewer applications per V-cycle" -- and
// not merely by a stopwatch, which at +-7% within an allocation could not resolve it.
struct OpCounts {
  long long fine = 0;
  long long coarse = 0;
  void reset()
  {
    fine = 0;
    coarse = 0;
  }
};

// Forwarding decorator that tallies Op/AdjOp/HermOp on whatever it wraps.
//
// Wraps a LinearOperatorBase, so it composes with both the Schur operator and the shifted
// forms the smoother and coarse solver actually drive. Counting at this level rather than
// inside the fermion operator is deliberate: one Mpc application is the unit the plan is
// written in, and Mpc is itself built from several Meooe/Mooee calls.
template <class Field>
class CountingLinearOperator : public LinearOperatorBase<Field> {
  LinearOperatorBase<Field> &inner_;
  long long &count_;

public:
  CountingLinearOperator(LinearOperatorBase<Field> &inner, long long &count)
      : inner_(inner), count_(count)
  {
  }

  void Op(const Field &in, Field &out) override
  {
    ++count_;
    inner_.Op(in, out);
  }
  void AdjOp(const Field &in, Field &out) override
  {
    ++count_;
    inner_.AdjOp(in, out);
  }
  void HermOp(const Field &in, Field &out) override
  {
    // Mpc^dag Mpc is two applications, and the reference CG row goes through here.
    count_ += 2;
    inner_.HermOp(in, out);
  }
  void HermOpAndNorm(const Field &in, Field &out, RealD &n1, RealD &n2) override
  {
    count_ += 2;
    inner_.HermOpAndNorm(in, out, n1, n2);
  }
  void OpDiag(const Field &in, Field &out) override { inner_.OpDiag(in, out); }
  void OpDir(const Field &in, Field &out, int dir, int disp) override
  {
    inner_.OpDir(in, out, dir, disp);
  }
  void OpDirAll(const Field &in, std::vector<Field> &out) override { inner_.OpDirAll(in, out); }
};

// ---------------------------------------------------------------------------
// FlexibleGCR -- PrecGeneralisedConjugateResidualNonHermitian without D1/D2/D4/D5
// ---------------------------------------------------------------------------
//
// Same Krylov method (VPGCR, Abe and Zhang 2005), same recursion, same iterates. Drop-in for
// Grid's class: same constructor signature, same `steps` and `Level()`, so the probe can hold
// either behind a LinearFunction reference and switch at runtime.
template <class Field>
class FlexibleGCR : public LinearFunction<Field> {
public:
  using LinearFunction<Field>::operator();

  RealD Tolerance;
  Integer MaxIterations;
  int mmax;
  int nstep;
  int steps = 0;

  // ---- Accounting for the UNATTRIBUTED time ------------------------------
  //
  // The MGPreconditioner's barriered breakdown captures only ~73% of the MG solve
  // at 16^3x48; the remaining ~27% is this class's own work and appears in no
  // timer. That is the largest single block of waste left, and `OUTER_MMAX` 6->1
  // buys only 2.5%, so it is NOT the O(m^2) Gram-Schmidt. These split it.
  //
  // ⚠️ Every tick() carries an accelerator_barrier(), so these timers themselves
  // serialise the pipeline. They are a DIAGNOSTIC: read the shares, and do not
  // quote a total measured with instrument=true against one measured without.
  bool instrument = false;
  double t_op = 0.0;      // Linop.Op -- the outer Krylov step's own application
  double t_prec = 0.0;    // Preconditioner() -- the V-cycle, for cross-checking
  double t_linalg = 0.0;  // innerProduct / axpy / axpy_norm, per step
  double t_gs = 0.0;      // the classical Gram-Schmidt history loop
  double t_qq = 0.0;      // norm2 of the new q
  double t_setup = 0.0;   // r0 formation + the pre-loop precon/op pair
  double t_log = 0.0;     // ostringstream construction (built even when !verbose)

  void reset_timers()
  {
    t_op = t_prec = t_linalg = t_gs = t_qq = t_setup = t_log = 0.0;
  }

  std::string timer_report() const
  {
    const double tot = t_op + t_prec + t_linalg + t_gs + t_qq + t_setup + t_log;
    std::ostringstream os;
    auto row = [&](const char *n, double v) {
      os << "\n      " << n << " " << (v / 1.0e6) << " s";
      if (tot > 0.0) os << "  (" << (100.0 * v / tot) << "%)";
    };
    os << "  outer GCR breakdown (barriered; diagnostic only):";
    row("preconditioner (V-cycle)", t_prec);
    row("operator Linop.Op       ", t_op);
    row("step linalg (ip/axpy)   ", t_linalg);
    row("Gram-Schmidt history    ", t_gs);
    row("qq norm2                ", t_qq);
    row("pre-loop setup          ", t_setup);
    row("log string construction ", t_log);
    os << "\n      barriered total " << (tot / 1.0e6) << " s";
    return os.str();
  }
  int level = 1;
  int verbose = 1;

  // Set by the caller when it guarantees psi == 0 on entry (D1). Applied only to the FIRST
  // GCRnStep of a solve; a restart carries a non-zero iterate.
  bool zero_guess = false;
  // D2's report block: an extra operator application per converged solve, for a log line.
  bool verify_residual = false;

  LinearFunction<Field> &Preconditioner;
  LinearOperatorBase<Field> &Linop;

  void Level(int lv) { level = lv; }

  FlexibleGCR(RealD tol, Integer maxit, LinearOperatorBase<Field> &linop,
              LinearFunction<Field> &prec, int mmax_, int nstep_)
      : Tolerance(tol), MaxIterations(maxit), mmax(mmax_), nstep(nstep_), Preconditioner(prec),
        Linop(linop)
  {
  }

private:
  // Persistent workspace (D4). unique_ptr rather than std::vector<Field> for two reasons: a
  // Lattice has no default constructor, and vector<Field>(n, grid) would deep-copy.
  std::unique_ptr<Field> r_, z_, Az_;
  std::vector<std::unique_ptr<Field>> q_, p_;
  std::vector<RealD> qq_;
  GridBase *workspace_grid_ = nullptr;

  void ensure_workspace(GridBase *grid)
  {
    if (workspace_grid_ == grid && static_cast<int>(q_.size()) == mmax) return;
    r_.reset(new Field(grid));
    z_.reset(new Field(grid));
    Az_.reset(new Field(grid));
    q_.clear();
    p_.clear();
    q_.reserve(mmax);
    p_.reserve(mmax);
    for (int m = 0; m < mmax; ++m) {
      q_.emplace_back(new Field(grid));
      p_.emplace_back(new Field(grid));
    }
    qq_.assign(mmax, 0.0);
    workspace_grid_ = grid;
  }

  void log(const std::string &msg) const
  {
    if (verbose)
      std::cout << GridLogMessage << std::string(level, '\t') << " Level " << level << " " << msg
                << std::endl;
  }

public:
  void operator()(const Field &src, Field &psi) override
  {
    const RealD ssq = norm2(src);
    const RealD rsq = Tolerance * Tolerance * ssq;

    steps = 0;
    for (Integer k = 0; k < MaxIterations; k++) {

      // D1 is legal only where psi is genuinely zero: the caller's guarantee, and only on the
      // FIRST inner call. A restart carries a non-zero iterate and must form the residual.
      const bool guess_is_zero = zero_guess && (k == 0);

      const RealD cp = GCRnStep(src, psi, rsq, guess_is_zero);

      {
        std::ostringstream os;
        os << "PGCR(" << mmax << "," << nstep << ") " << steps << " steps cp = " << cp
           << " target " << rsq;
        log(os.str());
      }

      if (cp < rsq) {
        std::ostringstream os;
        os << "PGCR: Converged on iteration " << steps << " computed residual "
           << std::sqrt(cp / ssq);
        if (verify_residual) {
          // D2: an extra operator application, an axpy and a norm2, for the log line only.
          Field r(src.Grid());
          r.Checkerboard() = src.Checkerboard();
          Linop.Op(psi, r);
          axpy(r, -1.0, src, r);
          os << " true residual " << std::sqrt(norm2(r) / ssq);
        }
        os << " target " << Tolerance;
        log(os.str());
        return;
      }
    }
    log("Variable Preconditioned GCR did not converge");
  }

  RealD GCRnStep(const Field &src, Field &psi, RealD rsq, bool guess_is_zero)
  {
    GridBase *grid = src.Grid();
    ensure_workspace(grid);

    Field &r = *r_;
    Field &z = *z_;
    Field &Az = *Az_;
    r.Checkerboard() = src.Checkerboard();
    z.Checkerboard() = src.Checkerboard();
    Az.Checkerboard() = src.Checkerboard();

    RealD cp;
    ComplexD a, b;
    ComplexD rq;

    //////////////////////////////////////////////////////////////////////////
    // r0 = src - A psi.  With a guaranteed-zero psi this is r0 = src, and the
    // operator application is skipped entirely (D1).
    //////////////////////////////////////////////////////////////////////////
    // Instrumentation tick. Identical form to MGPreconditioner's so the two
    // breakdowns are commensurate; a no-op returning 0 when instrument is false,
    // so the uninstrumented path pays neither a barrier nor a timer.
    auto tick = [this]() -> double {
      if (!instrument) return 0.0;
      accelerator_barrier();
      return usecond();
    };
    double tt;

    tt = tick();
    if (guess_is_zero) {
      r = src;
    } else {
      Linop.Op(psi, Az);
      r = src - Az;
    }

    /////////////////////
    // p = Prec(r)
    /////////////////////
    Preconditioner(r, z);
    Linop.Op(z, Az);

    *p_[0] = z;
    *q_[0] = Az;
    qq_[0] = norm2(Az);

    cp = norm2(r);
    t_setup += tick() - tt;

    for (int k = 0; k < nstep; k++) {

      steps++;

      const int kp = k + 1;
      const int peri_k = k % mmax;
      const int peri_kp = kp % mmax;

      tt = tick();
      rq = innerProduct(*q_[peri_k], r);
      a = rq / qq_[peri_k];

      axpy(psi, a, *p_[peri_k], psi);

      cp = axpy_norm(r, -a, *q_[peri_k], r);
      t_linalg += tick() - tt;

      // ⚠️ The stream is CONSTRUCTED unconditionally and only then does log()
      // test `verbose`, so this is paid on every step of every solve. Timed to
      // find out whether that matters.
      tt = tick();
      {
        std::ostringstream os;
        os << "PGCR step[" << steps << "]  resid " << cp << " target " << rsq;
        log(os.str());
      }
      t_log += tick() - tt;

      if ((k == nstep - 1) || (cp < rsq)) {
        return cp;
      }

      tt = tick();
      Preconditioner(r, z); // solve Az = r
      t_prec += tick() - tt;

      tt = tick();
      Linop.Op(z, Az);
      t_op += tick() - tt;

      *q_[peri_kp] = Az;
      *p_[peri_kp] = z;

      // Classical Gram-Schmidt against the history, verbatim from Grid: `b` is formed against
      // the ORIGINAL Az, not the partially-updated q[peri_kp]. Do not "improve" this.
      tt = tick();
      const int northog = ((kp) > (mmax - 1)) ? (mmax - 1) : (kp);
      for (int back = 0; back < northog; back++) {

        const int peri_back = (k - back) % mmax;
        GRID_ASSERT((k - back) >= 0);

        b = -real(innerProduct(*q_[peri_back], Az)) / qq_[peri_back];
        *p_[peri_kp] = *p_[peri_kp] + b * (*p_[peri_back]);
        *q_[peri_kp] = *q_[peri_kp] + b * (*q_[peri_back]);
      }
      t_gs += tick() - tt;

      tt = tick();
      qq_[peri_kp] = norm2(*q_[peri_kp]);
      t_qq += tick() - tt;
    }
    GRID_ASSERT(0); // never reached
    return cp;
  }
};

// ---------------------------------------------------------------------------
// Null-space generation with a configurable tolerance (Stage 2)
// ---------------------------------------------------------------------------
//
// Grid's Aggregation::CreateSubspaceGCR (Aggregates.h:128-176) hardcodes its solve at
//
//     PrecGeneralisedConjugateResidualNonHermitian GCR(0.001, 30, DiracOp, trivial, 10, 10)
//
// with three rounds of inverse iteration per basis vector. QUDA's setup_tol is 5e-6
// (quda_grid_bridge.h:479) -- 200x sharper. Grid also runs 4 smoother steps to QUDA's
// nu_post=8 and a coarse tolerance of 0.2 to QUDA's 0.1, so Grid is configured to a WEAKER
// preconditioner at every level. That is where the 11-vs-6 outer-iteration gap comes from,
// and it is also why "Grid's MG setup is 2.5x faster than QUDA's" is not the win the results
// doc records it as -- it is doing less work.
//
// This reproduces Grid's procedure with the tolerance and round count exposed, so the
// iteration gap can be attacked directly. Defaults reproduce Grid exactly.
//
// ⚠️ Grid's version applies the operator twice per basis vector purely to PRINT
// <n|Op|n> and <f|Op|f> diagnostics -- 2*nbasis extra fine applications in setup. Those are
// skipped when `quiet`, which is why fast-mode setup is faster for a reason that has nothing
// to do with null-space quality; do not read that as a preconditioner change.
template <class Aggregates>
void create_subspace_gcr(GridParallelRNG &RNG,
                         LinearOperatorBase<typename Aggregates::FineField> &DiracOp,
                         Aggregates &Agg, int nn, RealD tol, int rounds, int mmax, int nstep,
                         Integer maxiter, bool use_fast_gcr, bool quiet)
{
  typedef typename Aggregates::FineField FineField;
  GridBase *FineGrid = Agg.FineGrid;

  TrivialPrecon<FineField> simple_fine;
  PrecGeneralisedConjugateResidualNonHermitian<FineField> gcr_stock(tol, maxiter, DiracOp,
                                                                    simple_fine, mmax, nstep);
  FlexibleGCR<FineField> gcr_fast(tol, maxiter, DiracOp, simple_fine, mmax, nstep);
  gcr_fast.zero_guess = true; // `guess` is zeroed before every solve below
  gcr_fast.verify_residual = false;
  LinearFunction<FineField> &GCR =
      use_fast_gcr ? static_cast<LinearFunction<FineField> &>(gcr_fast)
                   : static_cast<LinearFunction<FineField> &>(gcr_stock);

  FineField noise(FineGrid);
  FineField src(FineGrid);
  FineField guess(FineGrid);
  FineField Mn(FineGrid);

  for (int b = 0; b < nn; b++) {

    Agg.subspace[b] = Zero();
    gaussian(RNG, noise);
    noise = noise * RealD(std::pow(norm2(noise), -0.5));

    if (!quiet) {
      DiracOp.Op(noise, Mn);
      std::cout << GridLogMessage << "noise   [" << b << "] <n|Op|n> " << innerProduct(noise, Mn)
                << std::endl;
    }

    // Inverse iteration: each round solves against the previous iterate, pulling the vector
    // towards the low modes the coarse space has to represent.
    for (int i = 0; i < rounds; i++) {
      src = noise;
      guess = Zero();
      GCR(src, guess);
      Agg.subspace[b] = guess;
      noise = Agg.subspace[b];
      noise = noise * RealD(std::pow(norm2(noise), -0.5));
    }

    if (!quiet) {
      DiracOp.Op(noise, Mn);
      std::cout << GridLogMessage << "filtered[" << b << "] <f|Op|f> " << innerProduct(noise, Mn)
                << " <f|OpDagOp|f>" << norm2(Mn) << std::endl;
    }

    Agg.subspace[b] = noise;
  }
}

// ---------------------------------------------------------------------------
// Workstream A (plan v2, __docs/2026_09_25_mg_plan_v2_coarse_apply_first.md §5):
// alternative COARSE-OPERATOR APPLIES built from Grid's own classes.
// ---------------------------------------------------------------------------
//
// WHY. At C3 clover the coarse apply of GeneralCoarsenedMatrix::Mult costs ~3.2 ms and is
// ~89% of a 410 ms outer step (LEDGER L152). Its Mult expands the input into a halo-PADDED
// copy through PaddedCell (Cshift-based, not the stencil), runs the 48x48 multiply over the
// padded volume (2.37x the real sites at C3/16 GPU), keeps npoint padded scratch vectors, then
// extracts the interior. A streaming bound for the real links is ~0.3 ms.
//
// Both alternatives reuse the GENERAL class for COARSENING (the only path that accepts a Schur
// operator, L131) and replace only the APPLY. Neither edits Grid/.
//
//   A2  StencilCoarseApply : the apply of Grid's OLD CoarsenedMatrix (CartesianStencil::
//       HaloExchange, interior volume; the same halo path the fine Dslash uses and the CG
//       campaign's comms flags tune). Its link matrices are filled from the general operator's
//       by matching shift vectors. Kernel conventions differ:
//         general (GeneralCoarsenedMatrix.h:197-201): res(b) += A[p](bb,b) * nbr(bb)  => A^T nbr
//         old     (CoarsenedMatrix.h:158)           : res(b) += A[pt](b,bb) * nbr(bb) => A   nbr
//       so A_old[pt] = TRANSPOSE(A_general[p]) for the matching shift. `transpose_links`
//       exists so the agreement check can falsify this reading. hops==1 only (the old
//       Geometry has no corner points). CoarsenedMatrix also builds red-black members we never
//       use; the rb grid it needs is constructed with a checker mask on x only, because the
//       default all-dims mask asserts on odd reduced dimensions (rdim 3 at C3).
//
//   A1  MrhsCoarseApply : Grid's MultiGeneralCoarsenedMatrix (batched cuBLAS GEMM over the
//       INTERIOR sites), the class both physical-volume HDCG drivers use for the solve. It
//       wants nrhs right-hand-side SLOTS as dimension 0 of a 5D grid; slot 0 carries the one
//       real vector and the rest stay zero. ⛔ fp64 COARSE TYPE ONLY: its GEMM pointer tables
//       and the gemmBatched call are hard-wired ComplexD (GeneralCoarsenedMatrixMultiRHS.h:
//       70-72, 695) whatever CComplex is, so with fp32 links it would run Zgemm over ComplexF
//       data. Every Grid test instantiates it with vTComplex (double). Diagnostic only here.
//       ⚠️ It still does the padded exchange, on a vector nrhs x larger.
//
// GATE for both: agreement with the general apply on a random coarse vector (~1e-6 fp32,
// ~1e-12 fp64), then an UNCHANGED outer iteration count and independent residual in the solve.

// Site-wise transpose of a lattice of nbasis x nbasis matrices. A free function because CUDA
// refuses an extended device lambda inside a constructor ("must allow its address to be taken").
template <class CComplex, int nbasis>
void transpose_site_matrices(Lattice<iMatrix<CComplex, nbasis>> &dst,
                             const Lattice<iMatrix<CComplex, nbasis>> &src)
{
  conformable(dst.Grid(), src.Grid());
  autoView(dst_v, dst, AcceleratorWrite);
  autoView(src_v, src, AcceleratorRead);
  const int Nsimd = CComplex::Nsimd();
  accelerator_for(ss, src.Grid()->oSites(), Nsimd, {
    auto s = coalescedRead(src_v[ss]);
    auto t = s;
    for (int i = 0; i < nbasis; ++i)
      for (int j = 0; j < nbasis; ++j) t(i, j) = s(j, i);
    coalescedWrite(dst_v[ss], t);
  });
}

// A2 ------------------------------------------------------------------------
template <class Fobj, class CComplex, int nbasis>
class StencilCoarseApply : public LinearOperatorBase<Lattice<iVector<CComplex, nbasis>>> {
public:
  typedef GeneralCoarsenedMatrix<Fobj, CComplex, nbasis> GeneralOp;
  typedef CoarsenedMatrix<Fobj, CComplex, nbasis> OldOp;
  typedef Lattice<iVector<CComplex, nbasis>> CoarseVector;
  typedef Lattice<iMatrix<CComplex, nbasis>> CoarseMatrix;

  GridRedBlackCartesian *CoarseRB = nullptr;
  std::unique_ptr<OldOp> op;
  RealD shift = 0.0;
  bool transposed = true;

  StencilCoarseApply(GeneralOp &general, GridCartesian *Coarse4d, RealD shift_ = 0.0,
                     bool transpose_links = true)
      : shift(shift_), transposed(transpose_links)
  {
    GRID_ASSERT(general.geom.npoint == 9); // hops==1 only
    // Red-black grid checkered on x only. CoarsenedMatrix never applies its rb members here;
    // this grid exists so the constructor can build them without tripping the all-dims
    // rdim-even assert (Cartesian_red_black.h:214) at C3 (rdim 12.6.3.3 / 12.3.3.3).
    Coordinate mask({1, 0, 0, 0});
    CoarseRB = new GridRedBlackCartesian(Coarse4d, mask, 0);
    op.reset(new OldOp(*Coarse4d, *CoarseRB, 1)); // hermitian=1: Mdag aliases M (never called)

    const int Nd = 4;
    for (int point = 0; point < op->geom.npoint; ++point) {
      const int dir = op->geom.directions[point];
      const int disp = op->geom.displacements[point];
      Coordinate want(Nd, 0);
      want[dir] = disp;
      int found = -1;
      for (int p = 0; p < general.geom.npoint; ++p)
        if (general.geom.shifts[p] == want) found = p;
      GRID_ASSERT(found >= 0);
      // The general operator's links live on the PADDED grid after ExchangeCoarseLinks;
      // Extract returns the interior on Coarse4d, which is where op->A[point] lives.
      CoarseMatrix Aup = general.Cell.Extract(general._A[found]);
      if (transposed)
        transpose_site_matrices<CComplex, nbasis>(op->A[point], Aup);
      else
        op->A[point] = Aup;
    }
  }
  ~StencilCoarseApply()
  {
    op.reset();
    delete CoarseRB;
  }

  void Op(const CoarseVector &in, CoarseVector &out) override
  {
    op->M(in, out);
    if (shift != 0.0) out = out + shift * in;
  }
  void AdjOp(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
  void OpDiag(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
  void OpDir(const CoarseVector &in, CoarseVector &out, int dir, int disp) override { GRID_ASSERT(0); }
  void OpDirAll(const CoarseVector &in, std::vector<CoarseVector> &out) override { GRID_ASSERT(0); }
  void HermOpAndNorm(const CoarseVector &in, CoarseVector &out, RealD &n1, RealD &n2) override { GRID_ASSERT(0); }
  void HermOp(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
};

// A1 ------------------------------------------------------------------------
template <class Fobj, class CComplex, int nbasis>
class MrhsCoarseApply : public LinearOperatorBase<Lattice<iVector<CComplex, nbasis>>> {
public:
  typedef GeneralCoarsenedMatrix<Fobj, CComplex, nbasis> GeneralOp;
  typedef MultiGeneralCoarsenedMatrix<Fobj, CComplex, nbasis> MrhsOp;
  typedef Lattice<iVector<CComplex, nbasis>> CoarseVector;

  GridCartesian *CoarseMrhs = nullptr; // 5D: {nrhs, c0, c1, c2, c3}, all SIMD lanes in dim 0
  std::unique_ptr<MrhsOp> op;
  std::unique_ptr<CoarseVector> vm, wm; // persistent (D4 lesson)
  int nrhs = 0;
  RealD shift = 0.0;

  MrhsCoarseApply(GeneralOp &general, GridCartesian *Coarse4d, RealD shift_ = 0.0) : shift(shift_)
  {
    // See the header comment: the class's GEMM is ComplexD whatever CComplex is.
    GRID_ASSERT((std::is_same<typename CComplex::scalar_type, ComplexD>::value));
    // Smallest legal batch: one full SIMD vector of RHS slots (the class divides by Nsimd).
    nrhs = CComplex::Nsimd();
    Coordinate clatt = Coarse4d->GlobalDimensions();
    Coordinate mpi = Coarse4d->_processors;
    // 5D, not the HDCG drivers' 6D: their fine operator is 5D (DWF) so their coarse geometry
    // already skips one dim; ours is 4D and MultiGeneralCoarsenedMatrix adds exactly one
    // skipped dim for the RHS index. A 6D grid here would shift the wrong dimensions.
    Coordinate rhLatt({nrhs, clatt[0], clatt[1], clatt[2], clatt[3]});
    Coordinate rhSimd({nrhs, 1, 1, 1, 1});
    Coordinate rhMpi({1, mpi[0], mpi[1], mpi[2], mpi[3]});
    CoarseMrhs = new GridCartesian(rhLatt, rhSimd, rhMpi);
    op.reset(new MrhsOp(general.geom, CoarseMrhs));
    op->CopyMatrix(general); // unpadded links straight from the general operator
    vm.reset(new CoarseVector(CoarseMrhs));
    wm.reset(new CoarseVector(CoarseMrhs));
    *vm = Zero();
    *wm = Zero();
  }
  ~MrhsCoarseApply()
  {
    op.reset();
    vm.reset();
    wm.reset();
    delete CoarseMrhs;
  }

  void Op(const CoarseVector &in, CoarseVector &out) override
  {
    out.Checkerboard() = in.Checkerboard();
    // Slot 0 carries the real vector; the other slots are zero from the constructor and, M
    // being linear, stay zero.
    InsertSliceFast(in, *vm, 0, 0);
    op->M(*vm, *wm);
    ExtractSliceFast(out, *wm, 0, 0);
    if (shift != 0.0) out = out + shift * in;
  }
  void AdjOp(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
  void OpDiag(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
  void OpDir(const CoarseVector &in, CoarseVector &out, int dir, int disp) override { GRID_ASSERT(0); }
  void OpDirAll(const CoarseVector &in, std::vector<CoarseVector> &out) override { GRID_ASSERT(0); }
  void HermOpAndNorm(const CoarseVector &in, CoarseVector &out, RealD &n1, RealD &n2) override { GRID_ASSERT(0); }
  void HermOp(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
};

// Agreement + timing of an alternative apply against the reference one on the given vector.
// Returns the relative difference; prints ms/apply for both (`reps` timed calls each, after
// the agreement call has warmed both paths).
template <class CoarseVector>
double check_coarse_apply(const char *name, LinearOperatorBase<CoarseVector> &reference,
                          LinearOperatorBase<CoarseVector> &candidate, const CoarseVector &src,
                          GridBase *UGrid, int reps, bool boss)
{
  CoarseVector ref(src.Grid()), alt(src.Grid()), diff(src.Grid());
  ref.Checkerboard() = src.Checkerboard();
  alt.Checkerboard() = src.Checkerboard();
  reference.Op(src, ref);
  candidate.Op(src, alt);
  diff = ref - alt;
  const double rel = std::sqrt(norm2(diff) / norm2(ref));

  auto time_it = [&](LinearOperatorBase<CoarseVector> &L) {
    accelerator_barrier();
    UGrid->Barrier();
    const double s = usecond();
    for (int i = 0; i < reps; ++i) L.Op(src, alt);
    accelerator_barrier();
    UGrid->Barrier();
    return (usecond() - s) / 1.0e3 / double(reps);
  };
  const double ms_ref = time_it(reference);
  const double ms_alt = time_it(candidate);
  if (boss) {
    std::cout << GridLogMessage << "coarse apply  " << name << "  rel diff vs general " << rel
              << (rel < 1.0e-5 ? "  AGREES" : "  DISAGREES") << "  general " << ms_ref
              << " ms/apply  " << name << " " << ms_alt << " ms/apply  (" << (ms_ref / ms_alt)
              << "x)" << std::endl;
  }
  return rel;
}

} // namespace ProbeMG
