#pragma once
// mixed_cg_rung_solver.h  (pure-Grid HMC, M3)
//
// MixedPrecCGRungSolver: Grid's MixedPrecisionConjugateGradient (fp32 inner CG, fp64 defect
// correction, final fp64 patch-up CG) as a RungSolverBase, for the ratio rungs' CG solves:
// HASEN_GRID_MIXED_CG_RUNGS (deriv + S, one object at cg_tol_drv) and
// HASEN_GRID_MIXED_CG_HEATBATH_RUNGS (heatbath, cg_tol_act); driver
// gen_qcd_hasenbusch_tune_compact_schur.cc. Before M3 only the tail ran mixed CG.
//
// operator()(linop, src, sol): sol = (Mpc^dag Mpc)^-1 src on src's checkerboard (Odd), with
// SchurDiagMooeeOperator(opD) in fp64 and SchurDiagMooeeOperator(opF) in fp32, constructed and
// called exactly as the driver's MixedPrecCGWrapper (the tail's CG_light_md: same max_inner =
// cg_max = 30000, max_outer = 50). The linop argument is ignored, as MixedPrecCGWrapper ignores
// it: opD must be the operator the action passes in (DenOp = LightOps[k] for deriv/S, NumOp =
// LightOps[k+1] for the heatbath). The action has already imported the gauge into opD
// (TwoFlavourSchurCloverRatioAction::refresh/S/deriv call ImportGauge before their solve).
// sol is a start guess (the action zeroes it) and leaves with src's checkerboard.
//
// SetGauge(U): U -> persistent fp32 gauge field -> opF.ImportGauge, the pattern of the tail's
// TwoFlavourSchurCloverActionMP::deriv, so the fp32 inner operator sees the same U as opD.
// A redundant import is skipped, keyed to the OPERATOR, not to this solver: one fp32 operator
// has several importers (LightOpsF[k]: rung k's deriv solver and rung k-1's heatbath solver;
// the tail's LightOpsF[itail]: the tail's MP deriv and the last rung's heatbath solver; the
// Grid-MG donor's: its hierarchy). A per-solver "U equals my last import" test would be WRONG:
// after a rejected trajectory the refresh gauge equals this solver's last import while another
// importer has since put a different field into opF. So a registry per opF records the
// fingerprint (gauge_fingerprint.h) of the last U any MixedPrecCGRungSolver imported into it
// and of opF's doubled links right after that import; the import is skipped only if the new U
// matches AND opF's links are still the ones that import produced (the second test catches
// every importer outside the registry). A duplicate import of the same U is harmless anyway;
// the skip only saves the fp32 clover rebuild. Cost of the test: two trace-sum reductions.

#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/ConjugateGradientMixedPrec.h>

#include "gauge_fingerprint.h"
#include "rung_solver_base.h"

#include <map>
#include <memory>
#include <string>

namespace Grid {

class MixedPrecCGRungSolver : public RungSolverBase {
 public:
  // The driver's WCF / WCF_f: the compact clover instantiations libGrid provides.
  typedef CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<WilsonImplR>> WCF;
  typedef CompactWilsonCloverFermion<WilsonImplF, CompactCloverHelpers<WilsonImplF>> WCF_f;
  using OperatorFunction<LatticeFermion>::operator();

  MixedPrecCGRungSolver(WCF &opD, WCF_f &opF, GridRedBlackCartesian *RBGridF, RealD tol,
                        int max_inner, int max_outer, std::string name)
      : opD_(opD), opF_(opF), RBGridF_(RBGridF), tol_(tol), max_inner_(max_inner),
        max_outer_(max_outer), name_(std::move(name))
  {
    if (opF_.FermionRedBlackGrid() != (GridBase *)RBGridF_) {
      std::cout << GridLogError << Tag() << "opF was not built on the RBGridF passed in"
                << std::endl;
      GRID_ASSERT(0);
    }
    if (opD_.Mass() != opF_.Mass()) {
      std::cout << GridLogError << Tag() << "fp64 operator mass " << opD_.Mass()
                << " != fp32 operator mass " << opF_.Mass() << std::endl;
      GRID_ASSERT(0);
    }
    GRID_ASSERT(tol_ > 0.0);
    GRID_ASSERT(max_inner_ >= 1 && max_outer_ >= 1);
  }

  void SetGauge(const LatticeGaugeField &U) override
  {
    const GaugeFingerprint fpU = GaugeTraceFingerprint(U);
    std::map<const void *, Stamp> &reg = Registry();
    auto it = reg.find(&opF_);
    if (it != reg.end() && it->second.gauge == fpU &&
        GaugeTraceFingerprint(opF_.Umu) == it->second.links) {
      ++skips_;
      return;
    }
    if (!UF_) {
      UF_.reset(new LatticeGaugeFieldF(opF_.GaugeGrid()));
      ws_.reset(new precisionChangeWorkspace(opF_.GaugeGrid(), U.Grid()));
      ws_in_ = U.Grid();
    }
    GRID_ASSERT(U.Grid() == ws_in_);
    precisionChange(*UF_, U, *ws_);
    opF_.ImportGauge(*UF_);
    ++imports_;
    Stamp st;
    st.gauge = fpU;
    st.links = GaugeTraceFingerprint(opF_.Umu);
    reg[&opF_] = st;
  }

  void operator()(LinearOperatorBase<LatticeFermion> &, const LatticeFermion &src,
                  LatticeFermion &sol) override
  {
    const double t0 = usecond();
    SchurDiagMooeeOperator<WCF, LatticeFermion> sd(opD_);
    SchurDiagMooeeOperator<WCF_f, LatticeFermionF> sf(opF_);
    MixedPrecisionConjugateGradient<LatticeFermion, LatticeFermionF> MPCG(
        tol_, max_inner_, max_outer_, RBGridF_, sf, sd);
    MPCG(src, sol);
    sol.Checkerboard() = src.Checkerboard();
    ++calls_;
    std::cout << GridLogMessage << Tag() << "solve " << calls_ << " inner "
              << MPCG.TotalInnerIterations << " restarts " << MPCG.TotalOuterIterations
              << " final_fp64 " << MPCG.TotalFinalStepIterations << " true_rel_residual "
              << MPCG.TrueResidual << " tol " << tol_ << " seconds "
              << (usecond() - t0) / 1.0e6 << " | fp32 gauge imports " << imports_ << " skipped "
              << skips_ << std::endl;
  }

  long long Calls() const { return calls_; }
  long long Imports() const { return imports_; }
  long long SkippedImports() const { return skips_; }
  RealD Tolerance() const { return tol_; }
  const std::string &Name() const { return name_; }

 private:
  struct Stamp {
    GaugeFingerprint gauge;  // the U last imported into this opF by a MixedPrecCGRungSolver
    GaugeFingerprint links;  // opF's doubled links right after that import
  };
  // One registry for the program, keyed by the fp32 operator's address (a function-local
  // static of an inline function: a single instance).
  static std::map<const void *, Stamp> &Registry()
  {
    static std::map<const void *, Stamp> registry;
    return registry;
  }
  std::string Tag() const { return "[GridMixedCG " + name_ + "] "; }

  WCF &opD_;
  WCF_f &opF_;
  GridRedBlackCartesian *RBGridF_;
  RealD tol_;
  int max_inner_, max_outer_;
  std::string name_;
  std::unique_ptr<LatticeGaugeFieldF> UF_;  // persistent fp32 gauge, allocated at first import
  std::unique_ptr<precisionChangeWorkspace> ws_;
  GridBase *ws_in_ = nullptr;
  long long calls_ = 0, imports_ = 0, skips_ = 0;
};

}  // namespace Grid
