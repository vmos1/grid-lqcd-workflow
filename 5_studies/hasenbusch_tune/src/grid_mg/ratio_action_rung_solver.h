#pragma once
// ratio_action_rung_solver.h  (pure-Grid HMC, M2)
//
// Renamed copy of the Grid-TXQCD fork's
// Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioActionQuda.h (fork commit 0ed7c199,
// 2026-07-14), which has no QUDA dependency: TwoFlavourSchurCloverRatioActionQuda ->
// TwoFlavourSchurCloverRatioActionRungSolver, QudaRungSolverBase -> RungSolverBase, includes and
// comment adapted, code otherwise identical. See src/grid_mg/PROVENANCE.md.
//
// Wires a rung solver (a RungSolverBase: GridMGSchurSolver, grid_mg_schur_solver.h, or since M3
// MixedPrecCGRungSolver, mixed_cg_rung_solver.h) into
// the carried TwoFlavourSchurCloverRatioAction's DerivativeSolver/ActionSolver slots. Both act
// on Mpc(DenOp), the lighter mass of the rung. HeatbathSolver acts on Vpc(NumOp), the heavier
// operator, and is normally the plain Grid CG; it must NOT go through a solver built for DenOp's
// mass. If the caller passes a RungSolverBase built at NumOp's mass as heatbath_cg, its SetGauge
// is kept in sync too (detected by dynamic_cast; null for the plain CG).
//
// Overrides refresh/S/deriv only to call solver.SetGauge(U) BEFORE delegating to the base, so the
// solver's own gauge-dependent state (the multigrid's fp32 operator) sees the same U that the
// base then imports into NumOp/DenOp. Parity is the base's: Odd.
//
// Generic over FermionOp like TwoFlavourSchurCloverRatioAction itself (the driver instantiates
// it with WilsonImplR and CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<...>>).

#include <Grid/Grid.h>
// The carried header (5_studies/hasenbusch_tune/include/, found first via -I$HB/include).
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioAction.h>

#include "rung_solver_base.h"

namespace Grid {

template <class Impl, class FermionOp = WilsonCloverFermion<Impl, CloverHelpers<Impl>>>
class TwoFlavourSchurCloverRatioActionRungSolver
    : public TwoFlavourSchurCloverRatioAction<Impl, FermionOp> {
 public:
  typedef TwoFlavourSchurCloverRatioAction<Impl, FermionOp> Base;
  typedef typename Impl::GaugeField GaugeField;

  TwoFlavourSchurCloverRatioActionRungSolver(
      FermionOp &NumOp, FermionOp &DenOp, RungSolverBase &solver,
      OperatorFunction<typename Base::FermionField> &heatbath_cg)
      : Base(NumOp, DenOp, solver, solver, heatbath_cg), solver_(solver),
        heatbath_rung_(dynamic_cast<RungSolverBase *>(&heatbath_cg)) {}

  void refresh(const GaugeField &U, GridSerialRNG &sRNG, GridParallelRNG &pRNG) override {
    solver_.SetGauge(U);
    if (heatbath_rung_) heatbath_rung_->SetGauge(U);
    Base::refresh(U, sRNG, pRNG);
  }
  RealD S(const GaugeField &U) override {
    solver_.SetGauge(U);
    return Base::S(U);
  }
  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    solver_.SetGauge(U);
    Base::deriv(U, dSdU);
  }

 private:
  RungSolverBase &solver_;
  RungSolverBase *heatbath_rung_;
};

// ---------------------------------------------------------------------------------------------
// M3 addition (new code, not from the fork): a ratio rung whose deriv and S keep the plain
// Grid CG solvers of the default route (separate DS and AS objects, i.e. the base's 4-argument
// behaviour: CG_deriv at cg_tol_drv, CG_action at cg_tol_act) while its HeatbathSolver is a
// RungSolverBase with gauge-dependent state (the driver's HASEN_GRID_MIXED_CG_HEATBATH_RUNGS
// on a rung in neither HASEN_GRID_MG_RUNGS nor HASEN_GRID_MIXED_CG_RUNGS). The one-solver
// wrapper above cannot serve here: it would put S on the deriv solver. Only refresh() is
// overridden, to call heatbath.SetGauge(U) before Base::refresh (the only solve of refresh is
// the heatbath's); S and deriv are the base's, untouched.
template <class Impl, class FermionOp = WilsonCloverFermion<Impl, CloverHelpers<Impl>>>
class TwoFlavourSchurCloverRatioActionHeatbathRung
    : public TwoFlavourSchurCloverRatioAction<Impl, FermionOp> {
 public:
  typedef TwoFlavourSchurCloverRatioAction<Impl, FermionOp> Base;
  typedef typename Impl::GaugeField GaugeField;

  TwoFlavourSchurCloverRatioActionHeatbathRung(
      FermionOp &NumOp, FermionOp &DenOp, OperatorFunction<typename Base::FermionField> &DS,
      OperatorFunction<typename Base::FermionField> &AS, RungSolverBase &heatbath)
      : Base(NumOp, DenOp, DS, AS, heatbath), heatbath_(heatbath) {}

  void refresh(const GaugeField &U, GridSerialRNG &sRNG, GridParallelRNG &pRNG) override {
    heatbath_.SetGauge(U);
    Base::refresh(U, sRNG, pRNG);
  }

 private:
  RungSolverBase &heatbath_;
};

}  // namespace Grid
