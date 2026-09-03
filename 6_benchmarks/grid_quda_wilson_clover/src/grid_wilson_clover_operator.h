#pragma once

// grid_wilson_clover_operator.h
//
// Pure-Grid Wilson / Wilson-clover operator: the Grid-side counterpart to
// QudaOperator in quda_grid_bridge.h. This header is deliberately QUDA-free --
// it includes ONLY <Grid/Grid.h> and uses only Grid classes, so the Grid solve
// path can be read and reasoned about in isolation. The benchmark harness
// (benchmark_grid_quda_wilson_clover.cc) owns the paired comparison, source
// generation, timing, correctness gates and JSONL emission; this class owns the
// Grid fermion operator, its optional single-precision twin, and the Schur
// (even-odd preconditioned) normal operators, and exposes the exact operations
// the harness times.
//
// The three actions share one code path via a WilsonFermionD base pointer:
// WilsonCloverFermion and CompactWilsonCloverFermion both derive from
// WilsonFermion<Impl>, so Dhop/DhopOE/M/Mooee/MooeeInv dispatch virtually to the
// correct override and SchurDiagMooeeOperator<WilsonFermionD,...> binds to any of
// them. FermionOperator has a virtual destructor, so deleting the derived object
// through the base unique_ptr is well defined.

#include <Grid/Grid.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace grid_quda_benchmark {

using namespace Grid;

// Result of a double-precision CG solve (the fields the harness records).
struct GridSolveResult {
  long long iterations = 0;
  double true_residual = 0.0;
};

// Result of a mixed-precision CG solve. MixedPrecisionConjugateGradient reports
// inner iterations, outer restarts, and a final double-precision cleanup step
// separately; all three are surfaced because none is comparable to a plain CG
// count on its own.
struct GridMixedSolveResult {
  long long inner_iterations = 0;
  long long outer_iterations = 0;
  long long final_step_iterations = 0;
  double true_residual = 0.0;
};

class GridWilsonCloverOperator {
public:
  enum class Action { Wilson, CloverStandard, CloverCompact };

  // mass/csw are in Grid's mass-normalized convention (as used throughout the
  // benchmark). When mixed==true the single-precision operator and its Schur
  // operator are also built -- MixedPrecisionConjugateGradient needs a linear
  // operator in BOTH precisions at construction, and the single-precision grid
  // is genuinely separate (vComplexF has a different SIMD width from vComplexD),
  // so UGrid_f/UrbGrid_f/Umu_f must be supplied. tol/maxiter seed the
  // ConjugateGradient constructed per solve, exactly as the previous inline code.
  GridWilsonCloverOperator(Action action,
                           GridCartesian *UGrid, GridRedBlackCartesian *UrbGrid,
                           LatticeGaugeField &Umu, double mass, double csw, bool mixed,
                           GridCartesian *UGrid_f, GridRedBlackCartesian *UrbGrid_f,
                           LatticeGaugeFieldF *Umu_f, double tol, int maxiter)
      : action_(action), clover_(action != Action::Wilson),
        UrbGrid_(UrbGrid), UrbGrid_f_(UrbGrid_f), tol_(tol), maxiter_(maxiter),
        pc_dslash_scratch_(UrbGrid)
  {
    pc_dslash_scratch_.Checkerboard() = Odd;

    // Antiperiodic time boundary via boundary_phases, matching the previous
    // inline construction exactly (phases[Nd-1] = -1).
    std::vector<Complex> phases(Nd, 1.0);
    phases[Nd - 1] = -1.0;
    WilsonAnisotropyCoefficients anisotropy;

    if (action_ == Action::Wilson) {
      WilsonFermionD::ImplParams impl;
      impl.boundary_phases = phases;
      op_d_.reset(new WilsonFermionD(Umu, *UGrid, *UrbGrid, mass, impl));
    } else if (action_ == Action::CloverStandard) {
      WilsonCloverFermionD::ImplParams impl;
      impl.boundary_phases = phases;
      op_d_.reset(new WilsonCloverFermionD(Umu, *UGrid, *UrbGrid, mass, csw, csw, anisotropy, impl));
    } else { // CloverCompact
      CompactWilsonCloverFermionD::ImplParams impl;
      impl.boundary_phases = phases;
      // cF = 1.0: bulk boundary (no open-boundary clover improvement), so the
      // compact operator is the same physics as the standard clover above.
      op_d_.reset(new CompactWilsonCloverFermionD(Umu, *UGrid, *UrbGrid, mass, csw, csw, 1.0, anisotropy, impl));
    }
    herm_d_.reset(new SchurDiagMooeeOperator<WilsonFermionD, LatticeFermion>(*op_d_));

    if (mixed) {
      if (UGrid_f == nullptr || UrbGrid_f == nullptr || Umu_f == nullptr)
        throw std::runtime_error(
            "GridWilsonCloverOperator: mixed precision requested without single-precision grids/gauge");
      if (action_ == Action::Wilson) {
        WilsonFermionF::ImplParams impl;
        impl.boundary_phases = phases;
        op_f_.reset(new WilsonFermionF(*Umu_f, *UGrid_f, *UrbGrid_f, mass, impl));
      } else if (action_ == Action::CloverStandard) {
        WilsonCloverFermionF::ImplParams impl;
        impl.boundary_phases = phases;
        op_f_.reset(new WilsonCloverFermionF(*Umu_f, *UGrid_f, *UrbGrid_f, mass, csw, csw, anisotropy, impl));
      } else {
        CompactWilsonCloverFermionF::ImplParams impl;
        impl.boundary_phases = phases;
        op_f_.reset(new CompactWilsonCloverFermionF(*Umu_f, *UGrid_f, *UrbGrid_f, mass, csw, csw, 1.0, anisotropy, impl));
      }
      herm_f_.reset(new SchurDiagMooeeOperator<WilsonFermionF, LatticeFermionF>(*op_f_));
    }
  }

  GridWilsonCloverOperator(const GridWilsonCloverOperator &) = delete;
  GridWilsonCloverOperator &operator=(const GridWilsonCloverOperator &) = delete;

  bool is_clover() const { return clover_; }
  bool has_mixed() const { return op_f_ != nullptr; }

  // Preconditioned Dslash, Even -> Odd. Wilson uses the raw DhopOE on both sides
  // of the benchmark; clover's dslashQuda applies A_oo^-1 D_oe, so compose the
  // same operations explicitly here (DhopOE then MooeeInv). The intermediate
  // field is a persistent member so no allocation enters the timed region.
  void apply_pc_dslash(const LatticeFermion &src_even, LatticeFermion &out)
  {
    op_d_->DhopOE(src_even, pc_dslash_scratch_, DaggerNo);
    if (clover_)
      op_d_->MooeeInv(pc_dslash_scratch_, out);
    else
      out = pc_dslash_scratch_;
  }

  // Full matrix M on a full 4D field.
  void apply_mat(const LatticeFermion &src, LatticeFermion &out) { op_d_->M(src, out); }

  // Preconditioned normal operator Mpc^dag Mpc on the odd checkerboard. Also the
  // reference the harness uses for its independent residual checks.
  void apply_normal(const LatticeFermion &src_odd, LatticeFermion &out) { herm_d_->MpcDagMpc(src_odd, out); }

  // Local clover inverse on a checkerboard (clover action only).
  void apply_clover_inv(const LatticeFermion &src_odd, LatticeFermion &out)
  {
    if (!clover_) throw std::logic_error("clover_inv requested for a Wilson operator");
    op_d_->MooeeInv(src_odd, out);
  }

  // Double CG: M_pc^dag M_pc x = b on the odd checkerboard. The caller owns
  // zeroing `sol` and the timing barriers, so the timed region matches the
  // previous inline loop exactly. ConjugateGradient construction allocates no
  // lattice fields, so building it here is negligible.
  GridSolveResult solve_double(const LatticeFermion &src_odd, LatticeFermion &sol)
  {
    ConjugateGradient<LatticeFermion> CG(tol_, maxiter_, false);
    CG(*herm_d_, src_odd, sol);
    return {static_cast<long long>(CG.IterationsToComplete), CG.TrueResidual};
  }

  // Mixed-precision CG (double outer, single inner) -- the same scheme QUDA uses
  // for its sloppy solve. Caller owns zeroing + timing, as for solve_double.
  GridMixedSolveResult solve_mixed(const LatticeFermion &src_odd, LatticeFermion &sol, int outer_iterations)
  {
    if (!has_mixed())
      throw std::logic_error("solve_mixed requested but the single-precision operator was not built");
    MixedPrecisionConjugateGradient<LatticeFermion, LatticeFermionF> mpcg(
        tol_, maxiter_, outer_iterations, UrbGrid_f_, *herm_f_, *herm_d_);
    mpcg(src_odd, sol);
    return {static_cast<long long>(mpcg.TotalInnerIterations),
            static_cast<long long>(mpcg.TotalOuterIterations),
            static_cast<long long>(mpcg.TotalFinalStepIterations), mpcg.TrueResidual};
  }

private:
  Action action_;
  bool clover_;
  GridRedBlackCartesian *UrbGrid_;
  GridRedBlackCartesian *UrbGrid_f_;
  double tol_;
  int maxiter_;
  std::unique_ptr<WilsonFermionD> op_d_;
  std::unique_ptr<WilsonFermionF> op_f_;
  std::unique_ptr<SchurDiagMooeeOperator<WilsonFermionD, LatticeFermion>> herm_d_;
  std::unique_ptr<SchurDiagMooeeOperator<WilsonFermionF, LatticeFermionF>> herm_f_;
  LatticeFermion pc_dslash_scratch_;
};

// ---------------------------------------------------------------------------
// Multi-RHS (batched) compact clover operator.
//
// CompactWilsonCloverFermion5D solves N right-hand sides at once. The fifth
// dimension is an RHS-BATCHING index, NOT a domain-wall dimension: the clover
// term is the same compact deGrand-Rossi form as CloverCompact above, and the
// operator is block diagonal in the 5th index. Batching amortizes the gauge and
// clover reads across all N sources, which is why it helps valence work (many
// propagator sources) and does nothing for HMC (one solve at a time).
//
// This is a SEPARATE class rather than another Action on GridWilsonCloverOperator
// because CompactWilsonCloverFermion5D derives from WilsonFermion5D, not from
// WilsonFermion, so it cannot share that class's WilsonFermionD base pointer.
// Keeping it separate also leaves the gate-validated 4D path byte-for-byte
// untouched.
//
// Two properties of Grid's 5D grids that this class relies on, both verified
// against the Grid source rather than assumed:
//   * makeFiveDimRedBlackGrid builds the checkerboard with mask {0,1,1,1,1} --
//     the Ls dimension does NOT participate in the parity. Every Ls slice
//     therefore carries the full 4D checkerboard pattern, so the batched odd
//     problem is exactly N independent copies of the 4D odd problem.
//   * InsertSlice requires the low-dimensional field to have matching LOCAL
//     dimensions and the orthogonal direction to be undistributed. That holds
//     for full 4D -> full 5D with orthog=0, so sources are assembled on FULL
//     grids and the checkerboard is taken afterwards.
class GridCompactClover5DOperator {
public:
  // Ls == number of right-hand sides. The 4D grids are borrowed; the 5D grids
  // are created and owned here.
  GridCompactClover5DOperator(GridCartesian *UGrid, GridRedBlackCartesian *UrbGrid,
                              LatticeGaugeField &Umu, double mass, double csw,
                              int nrhs, double tol, int maxiter)
      : nrhs_(nrhs), tol_(tol), maxiter_(maxiter)
  {
    if (nrhs_ < 1) throw std::runtime_error("GridCompactClover5DOperator: nrhs must be >= 1");

    FGrid_.reset(SpaceTimeGrid::makeFiveDimGrid(nrhs_, UGrid));
    FrbGrid_.reset(SpaceTimeGrid::makeFiveDimRedBlackGrid(nrhs_, UGrid));

    std::vector<Complex> phases(Nd, 1.0);
    phases[Nd - 1] = -1.0;
    CompactWilsonCloverFermion5DD::ImplParams impl;
    impl.boundary_phases = phases;

    // Same physics as the 4D compact operator: csw_r = csw_t = csw, cF = 1.0
    // (bulk boundary). Note the 5D constructor takes no anisotropy argument.
    op_.reset(new CompactWilsonCloverFermion5DD(Umu, *FGrid_, *FrbGrid_, *UGrid, *UrbGrid,
                                                mass, csw, csw, 1.0, impl));
    herm_.reset(new SchurDiagMooeeOperator<CompactWilsonCloverFermion5DD, LatticeFermion>(*op_));
  }

  GridCompactClover5DOperator(const GridCompactClover5DOperator &) = delete;
  GridCompactClover5DOperator &operator=(const GridCompactClover5DOperator &) = delete;

  int nrhs() const { return nrhs_; }
  GridCartesian *five_dim_grid() { return FGrid_.get(); }
  GridRedBlackCartesian *five_dim_rb_grid() { return FrbGrid_.get(); }

  // Pack N full 4D sources into one 5D odd-checkerboard field, slice s <- src4[s].
  // Assembled on the full 5D grid, then checkerboarded (see class comment).
  void batch_sources(const std::vector<LatticeFermion> &src4_full, LatticeFermion &src5_odd)
  {
    if (static_cast<int>(src4_full.size()) != nrhs_)
      throw std::runtime_error("GridCompactClover5DOperator: source count != nrhs");
    LatticeFermion src5_full(FGrid_.get());
    src5_full = Zero();
    for (int s = 0; s < nrhs_; ++s) InsertSlice(src4_full[s], src5_full, s, 0);
    pickCheckerboard(Odd, src5_odd, src5_full);
  }

  // Preconditioned normal operator on the 5D odd checkerboard -- the same
  // Mpc^dag Mpc the 4D path solves, applied to all N right-hand sides at once.
  void apply_normal(const LatticeFermion &src5_odd, LatticeFermion &out) { herm_->MpcDagMpc(src5_odd, out); }

  // One batched CG over all N right-hand sides. NOTE: because the operator is
  // block diagonal in the RHS index, this converges when the WORST-conditioned
  // right-hand side converges -- the iteration count is not comparable to a
  // single-RHS count, and the N sequential QUDA solves each converge
  // independently. Caller owns zeroing and timing.
  GridSolveResult solve_double(const LatticeFermion &src5_odd, LatticeFermion &sol5_odd)
  {
    ConjugateGradient<LatticeFermion> CG(tol_, maxiter_, false);
    CG(*herm_, src5_odd, sol5_odd);
    return {static_cast<long long>(CG.IterationsToComplete), CG.TrueResidual};
  }

private:
  int nrhs_;
  double tol_;
  int maxiter_;
  // Declared before op_/herm_ so they outlive the operator that references them
  // (members are destroyed in reverse declaration order).
  std::unique_ptr<GridCartesian> FGrid_;
  std::unique_ptr<GridRedBlackCartesian> FrbGrid_;
  std::unique_ptr<CompactWilsonCloverFermion5DD> op_;
  std::unique_ptr<SchurDiagMooeeOperator<CompactWilsonCloverFermion5DD, LatticeFermion>> herm_;
};

} // namespace grid_quda_benchmark
