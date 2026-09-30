// grid_mg_schur_solver.h  (pure-Grid HMC, M2: the `best2` multigrid as a rung solver)
//
// GridMGSchurSolver: one per MG rung. Fills both the DerivativeSolver and the ActionSolver slot
// of a TwoFlavourSchurCloverRatioAction (through TwoFlavourSchurCloverRatioActionRungSolver),
// i.e. it is called as Solver(linop, src, sol) with linop = SchurDifferentiableOperator(DenOp)
// (Mpc = M_oo - M_oe M_ee^-1 M_eo), src/sol on the Odd checkerboard, and must return
//
//     sol = (Mpc^dag Mpc)^-1 src
//
// carrying the Odd checkerboard. Realised, as the QUDA hybrid does, by two Mpc solves through
// gamma5-hermiticity (Mpc^dag = g5 Mpc g5 on a half lattice, so no zero-padding):
//
//     y = g5 Mpc^-1 (g5 src),   sol = Mpc^-1 y.
//
// Each Mpc solve is a FRESH fp64 ProbeMG::FlexibleGCR on the action's own linop from a zero
// guess, preconditioned by the shared GridMGHierarchy (fp32). Correctness therefore never
// depends on the hierarchy's age: the multigrid only preconditions. After each solve one extra
// linop.Op gives the true relative residual |Mpc x - b| / |b|, which is logged and drives the
// two rebuild tiers:
//   soft  outer iterations >= threshold_count  => H.NotePendingRebuild() (next SetGauge)
//   hard  true residual > rsd_tol_factor * tol  => H.ForceRebuildNow(), re-solve once,
//         GRID_ASSERT if still above the factor.
// Spec: __docs/2026_09_29_pure_grid_m2_mg_solver_design.md §1 and §2 item 6.

#pragma once

#include <Grid/Grid.h>

#include "grid_mg_hierarchy.h"
#include "mg_solvers.h"
#include "rung_solver_base.h"

#include <cmath>
#include <string>

namespace Grid {

class GridMGSchurSolver : public RungSolverBase {
 public:
  using OperatorFunction<LatticeFermion>::operator();

  struct SolveStats {
    int iterations = 0;       // outer FlexibleGCR steps (all restart cycles)
    RealD true_residual = 0;  // |linop.Op(x) - b| / |b|, fp64
    double seconds = 0.0;     // solve + residual check (+ a hard-tier rebuild and re-solve)
    bool rebuilt = false;     // the hard tier fired on this solve
  };

  // donor: this rung's SetGauge forwards to the hierarchy. Sharing rungs pass false: within a
  // force evaluation the gauge is the same object, so the donor's import is exact for them too.
  GridMGSchurSolver(GridMGHierarchy &H, RealD tol, int cb, std::string name, bool donor)
      : H_(H), tol_(tol), cb_(cb), name_(std::move(name)), donor_(donor)
  {
    GRID_ASSERT(cb_ == Even || cb_ == Odd);
    if (H_.Checkerboard() != cb_) {
      std::cout << GridLogError << Tag() << "checkerboard " << cb_
                << " differs from the hierarchy's " << H_.Checkerboard() << std::endl;
      GRID_ASSERT(0);
    }
    GRID_ASSERT(tol_ > 0.0);
  }

  void SetGauge(const LatticeGaugeField &U) override
  {
    if (donor_) H_.SetGauge(U);
  }

  void operator()(LinearOperatorBase<LatticeFermion> &linop, const LatticeFermion &src,
                  LatticeFermion &sol) override
  {
    if (src.Checkerboard() != cb_) {
      std::cout << GridLogError << Tag() << "source checkerboard " << src.Checkerboard()
                << ", solver built for " << cb_ << std::endl;
      GRID_ASSERT(0);
    }
    if (!H_.Built() || H_.NeedsBuild()) H_.Build();
    ++calls_;

    Gamma G5(Gamma::Algebra::Gamma5);
    GridBase *grid = src.Grid();

    // y = g5 Mpc^-1 (g5 src)
    LatticeFermion b1(grid);
    b1 = G5 * src;
    b1.Checkerboard() = cb_;
    LatticeFermion y(grid);
    y.Checkerboard() = cb_;
    last_[0] = Solve(linop, b1, y, 1);

    // sol = Mpc^-1 y
    LatticeFermion b2(grid);
    b2 = G5 * y;
    b2.Checkerboard() = cb_;
    sol.Checkerboard() = cb_;
    last_[1] = Solve(linop, b2, sol, 2);
    sol.Checkerboard() = cb_;
  }

  // Diagnostics of the most recent call; which = 1 (the g5 solve) or 2.
  const SolveStats &Last(int which) const
  {
    GRID_ASSERT(which == 1 || which == 2);
    return last_[which - 1];
  }
  long long Calls() const { return calls_; }
  RealD Tolerance() const { return tol_; }
  bool Donor() const { return donor_; }
  const std::string &Name() const { return name_; }

 private:
  std::string Tag() const { return "[GridMG " + name_ + "] "; }

  // One Mpc solve from a zero guess with its true-residual check. No tier logic.
  SolveStats SolveOnce(LinearOperatorBase<LatticeFermion> &linop, const LatticeFermion &b,
                       LatticeFermion &x, RealD bnorm2)
  {
    const GridMGParams &p = H_.Params();
    SolveStats st;
    const double t0 = usecond();
    x = Zero();
    x.Checkerboard() = b.Checkerboard();
    ProbeMG::FlexibleGCR<LatticeFermionD> gcr(tol_, p.outer_maxiter, linop, H_.Preconditioner(),
                                              p.outer_mmax, p.outer_nstep);
    gcr.Level(1);
    gcr.verbose = (p.verbose >= 2) ? 1 : 0;
    // The probe's outer settings: zero_guess stays false (one Op on the first cycle, the
    // measured best2 behaviour) and the verify block is off because the residual is checked
    // below instead, by the same single extra Op.
    gcr.zero_guess = false;
    gcr.verify_residual = false;
    gcr(b, x);
    st.iterations = gcr.steps;

    LatticeFermion r(b.Grid());
    r.Checkerboard() = b.Checkerboard();
    linop.Op(x, r);
    r = r - b;
    st.true_residual = std::sqrt(norm2(r) / bnorm2);
    st.seconds = (usecond() - t0) / 1.0e6;
    return st;
  }

  SolveStats Solve(LinearOperatorBase<LatticeFermion> &linop, const LatticeFermion &b,
                   LatticeFermion &x, int which)
  {
    const GridMGParams &p = H_.Params();
    const RealD bnorm2 = norm2(b);
    if (bnorm2 == 0.0) {  // FlexibleGCR would never meet a zero target
      x = Zero();
      x.Checkerboard() = b.Checkerboard();
      return SolveStats();
    }
    const long long fine0 = H_.Counts().fine, coarse0 = H_.Counts().coarse;
    SolveStats st = SolveOnce(linop, b, x, bnorm2);
    Log(which, st, fine0, coarse0, "");

    if (st.iterations >= p.threshold_count) H_.NotePendingRebuild();

    const RealD bound = p.rsd_tol_factor * tol_;
    if (!(st.true_residual <= bound)) {
      std::cout << GridLogMessage << Tag() << "solve " << which << "/2 true residual "
                << st.true_residual << " > " << p.rsd_tol_factor << " x tol " << tol_
                << ": HARD-TIER rebuild and re-solve" << std::endl;
      const double seconds_before = st.seconds;
      H_.ForceRebuildNow();
      const long long fine1 = H_.Counts().fine, coarse1 = H_.Counts().coarse;
      st = SolveOnce(linop, b, x, bnorm2);
      st.seconds += seconds_before;
      st.rebuilt = true;
      Log(which, st, fine1, coarse1, " (re-solve after hard-tier rebuild)");
      if (!(st.true_residual <= bound)) {
        std::cout << GridLogError << Tag() << "solve " << which << "/2 still at true residual "
                  << st.true_residual << " > " << bound << " after a fresh hierarchy: ABORT"
                  << std::endl;
        GRID_ASSERT(0);
      }
    }
    return st;
  }

  void Log(int which, const SolveStats &st, long long fine0, long long coarse0,
           const char *suffix) const
  {
    if (H_.Params().verbose < 1) return;
    std::cout << GridLogMessage << Tag() << "solve " << which << "/2 iters " << st.iterations
              << " true_rel_residual " << st.true_residual << " tol " << tol_ << " seconds "
              << st.seconds << " gen " << H_.Generation() << " build " << H_.Builds()
              << " vcycle_applies fine " << (H_.Counts().fine - fine0) << " coarse "
              << (H_.Counts().coarse - coarse0) << suffix << std::endl;
  }

  GridMGHierarchy &H_;
  RealD tol_;
  int cb_;
  std::string name_;
  bool donor_;
  long long calls_ = 0;
  SolveStats last_[2];
};

}  // namespace Grid
