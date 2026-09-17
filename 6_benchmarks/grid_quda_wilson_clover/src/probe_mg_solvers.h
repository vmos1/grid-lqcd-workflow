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
#include <vector>

namespace ProbeMG {

using namespace Grid;

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

    for (int k = 0; k < nstep; k++) {

      steps++;

      const int kp = k + 1;
      const int peri_k = k % mmax;
      const int peri_kp = kp % mmax;

      rq = innerProduct(*q_[peri_k], r);
      a = rq / qq_[peri_k];

      axpy(psi, a, *p_[peri_k], psi);

      cp = axpy_norm(r, -a, *q_[peri_k], r);

      {
        std::ostringstream os;
        os << "PGCR step[" << steps << "]  resid " << cp << " target " << rsq;
        log(os.str());
      }

      if ((k == nstep - 1) || (cp < rsq)) {
        return cp;
      }

      Preconditioner(r, z); // solve Az = r
      Linop.Op(z, Az);

      *q_[peri_kp] = Az;
      *p_[peri_kp] = z;

      // Classical Gram-Schmidt against the history, verbatim from Grid: `b` is formed against
      // the ORIGINAL Az, not the partially-updated q[peri_kp]. Do not "improve" this.
      const int northog = ((kp) > (mmax - 1)) ? (mmax - 1) : (kp);
      for (int back = 0; back < northog; back++) {

        const int peri_back = (k - back) % mmax;
        GRID_ASSERT((k - back) >= 0);

        b = -real(innerProduct(*q_[peri_back], Az)) / qq_[peri_back];
        *p_[peri_kp] = *p_[peri_kp] + b * (*p_[peri_back]);
        *q_[peri_kp] = *q_[peri_kp] + b * (*q_[peri_back]);
      }
      qq_[peri_kp] = norm2(*q_[peri_kp]);
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

} // namespace ProbeMG
