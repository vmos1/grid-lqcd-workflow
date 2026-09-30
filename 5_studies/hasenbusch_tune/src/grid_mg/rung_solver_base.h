#pragma once
// rung_solver_base.h  (pure-Grid HMC, M2)
//
// Renamed copy of the Grid-TXQCD fork's Grid/algorithms/iterative/QudaRungSolverBase.h
// (fork commit 0ed7c199, 2026-07-14), which has no QUDA dependency: QudaRungSolverBase ->
// RungSolverBase, comment rewritten, code otherwise identical. See src/grid_mg/PROVENANCE.md.
//
// Common interface for a rung solver that fills a TwoFlavourSchurCloverRatioAction's
// DerivativeSolver/ActionSolver slots (TwoFlavourSchurCloverRatioActionRungSolver, in
// ratio_action_rung_solver.h) and keeps state that depends on the gauge field: the Grid
// multigrid's fp32 operator and hierarchy (GridMGSchurSolver). SetGauge() is not part of Grid's
// OperatorFunction interface, so this adds it as a virtual, letting the action wrapper hold any
// such solver behind one reference without a template parameter.

#include <Grid/Grid.h>
#include <Grid/algorithms/LinearOperator.h>

namespace Grid {

class RungSolverBase : public OperatorFunction<LatticeFermion> {
 public:
  virtual void SetGauge(const LatticeGaugeField &U) = 0;
  virtual ~RungSolverBase() = default;
};

}  // namespace Grid
