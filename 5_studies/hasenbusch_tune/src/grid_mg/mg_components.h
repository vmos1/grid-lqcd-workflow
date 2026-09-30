// mg_components.h  (pure-Grid HMC, M2: the `best2` multigrid as a rung solver)
//
// The three multigrid building blocks of the probe, COPIED VERBATIM (classes and comments)
// from grid-lqcd-workflow/6_benchmarks/grid_quda_wilson_clover/src/probe_grid_mg_schur_clover.cc
// lines 86-391 at workflow commit 9c6e99683d34 (`git show HEAD:<path>`, not the working tree):
//
//   ShiftedSchurOperator      Mpc + shift, the fine smoother's operator
//   PrecisionChangeAdaptor    fp32 LinearFunction exposed to an fp64 solver (persistent maps)
//   MGPreconditioner          one V-cycle (Grid's tests/debug/Test_general_coarse_wilson.cc)
//
// The only difference from the probe is the enclosing namespace: the probe had them in an
// anonymous namespace of its .cc; here they live in ProbeMG (with mg_solvers.h), since an
// anonymous namespace in a header would give every translation unit its own copy. The
// benchmark counters are optional as they were in the probe: MGPreconditioner's per-stage
// timers only tick when `instrument` is set (default false, no barriers), and the operator
// counters (ProbeMG::OpCounts / CountingLinearOperator) are a wrapper chosen by the caller.
// See src/grid_mg/PROVENANCE.md.

#pragma once

#include <Grid/Grid.h>

#include "mg_solvers.h"

#include <memory>
#include <vector>

namespace ProbeMG {

using namespace Grid;

// ---- verbatim from probe_grid_mg_schur_clover.cc:86-391 follows ----

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
//
// ⚠️ The two-argument precisionChange(out,in) REBUILDS its site/lane map on every call
// (Lattice_transfer.h:1489-1493: a fresh precisionChangeWorkspace, built on the host with a
// thread_for over all sites, then copied to the device). Grid's own
// MixedPrecisionFlexibleGeneralisedMinimalResidual pays the same cost per iteration
// (MixedPrecisionFlexibleGeneralisedMinimalResidual.h:210-219). The workspace class exists
// precisely so callers can build the map once; `persistent` does that, and also keeps the two
// fp32 temporaries alive across calls (D4 lesson). Default OFF reproduces every earlier number.
template <class FieldD, class FieldF>
class PrecisionChangeAdaptor : public LinearFunction<FieldD> {
public:
  using LinearFunction<FieldD>::operator();

private:
  LinearFunction<FieldF> &inner_;
  GridBase *fine_f_;
  bool persistent_ = false;
  std::unique_ptr<precisionChangeWorkspace> ws_down_, ws_up_; // fp64->fp32, fp32->fp64
  std::unique_ptr<FieldF> in_f_, out_f_;

public:
  PrecisionChangeAdaptor(LinearFunction<FieldF> &inner, GridBase *fine_f, bool persistent = false)
      : inner_(inner), fine_f_(fine_f), persistent_(persistent)
  {
  }

  void operator()(const FieldD &in, FieldD &out) override
  {
    if (persistent_) {
      if (!ws_down_) {
        ws_down_.reset(new precisionChangeWorkspace(fine_f_, in.Grid()));
        ws_up_.reset(new precisionChangeWorkspace(in.Grid(), fine_f_));
        in_f_.reset(new FieldF(fine_f_));
        out_f_.reset(new FieldF(fine_f_));
      }
      in_f_->Checkerboard() = in.Checkerboard();
      out_f_->Checkerboard() = in.Checkerboard();
      precisionChange(*in_f_, in, *ws_down_);
      *out_f_ = Zero();
      out_f_->Checkerboard() = in.Checkerboard();
      inner_(*in_f_, *out_f_);
      precisionChange(out, *out_f_, *ws_up_);
      out.Checkerboard() = in.Checkerboard();
      return;
    }
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

// ---- end of verbatim copy ----

} // namespace ProbeMG
