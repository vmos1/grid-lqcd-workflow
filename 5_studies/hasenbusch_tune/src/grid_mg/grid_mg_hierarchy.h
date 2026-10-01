// grid_mg_hierarchy.h  (pure-Grid HMC, M2: the `best2` multigrid as a rung solver)
//
// GridMGHierarchy: the fp32 three-level multigrid preconditioner of the gq-mg campaign's `best2`
// recipe, built on ONE fp32 compact-clover operator (the donor mass) and exposed as a
// LinearFunction<LatticeFermionD> for an fp64 outer solver. One hierarchy per donor mass; rungs
// that share it use it unchanged (the hybrid's model).
// Spec: __docs/2026_09_29_pure_grid_m2_mg_solver_design.md §2 item 4.
//
// Construction order, solver types and every numerical choice follow the probe's
// `--probe-mg-precision single --probe-coarse-precon mg --probe-coarse-apply stencil` path
// (probe_grid_mg_schur_clover.cc lines 1448-1671 at workflow commit 9c6e99683d34), with the
// probe's fp64 subspace, fp64 coarse operator and fp64 V-cycle dropped (under best2 they were
// never applied: "dead weight, 4 GB per rank at C3"). Level 1 = fine / block on the red-black
// fp32 grid, level 2 = level 1 / block2 on the full level-1 grid.
//
//   SetGauge(U)  fp64 U -> persistent fp32 UmuF (persistent precision-change map), then
//                opF.ImportGauge(UmuF); ++generation. A hierarchy that is not built yet, or has
//                a soft-tier rebuild pending, is marked needs-build. Nothing else: the coarse
//                levels stay frozen between rebuilds (the hybrid's "thin update").
//   SetGaugeIfNew(U)  (M3, the MG heatbath) imports U only if the CURRENT generation's import
//                was not of U: the gauge fingerprint (gauge_fingerprint.h) of U and of opF's
//                doubled links are recorded per generation once TrackGauge() is on; a
//                fingerprint recorded at an older generation never counts. TrackGauge() off
//                (every run without HASEN_GRID_MG_HEATBATH_RUNGS) = SetGauge exactly as in M2.
//   Build()      destroys every level object and re-creates it (re-coarsening in place is not
//                possible: _A lives on the padded grid after ExchangeCoarseLinks), reseeding
//                the setup RNGs from params.seed first, so a build is a pure function of the
//                current gauge field and the parameters. Grids, geometries and RNGs are
//                created once in the constructor and reused.
//
// ⛔ Lifetimes. GeneralCoarsenedMatrix holds its NonLocalStencilGeometry BY REFERENCE, and the
// solvers and V-cycles hold references to operators and aggregations, so all of them are owned
// here, destroyed in reverse order before a rebuild and in the destructor. The OpCounts object
// is per hierarchy (never share one between hierarchies).

#pragma once

#include <Grid/Grid.h>

#include "gauge_fingerprint.h"
#include "grid_mg_params.h"
#include "mg_components.h"
#include "mg_solvers.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace Grid {

class GridMGHierarchy {
 public:
  // The driver's WCF_f (gen_qcd_hasenbusch_tune_compact_schur.cc): the only instantiation
  // libGrid provides for the compact clover operator.
  typedef CompactWilsonCloverFermion<WilsonImplF, CompactCloverHelpers<WilsonImplF>> WCF_f;
  // The operator the HMC ratio action solves (SchurDifferentiableOperator is a
  // SchurDiagMooeeOperator): Mpc = M_oo - M_oe M_ee^-1 M_eo on the Odd checkerboard.
  typedef SchurDiagMooeeOperator<WCF_f, LatticeFermionF> SchurOpF;

  // Level 1 (instantiations exactly as the probe's fp32 hierarchy).
  typedef Aggregation<vSpinColourVectorF, vTComplexF, kNbasis> Subspace1Half;
  typedef Aggregation<vSpinColourVectorF, vTComplexF, 2 * kNbasis> Subspace1;
  typedef GeneralCoarsenedMatrix<vSpinColourVectorF, vTComplexF, 2 * kNbasis> Level1Op;
  typedef Level1Op::CoarseVector CoarseVector1;
  typedef ProbeMG::StencilCoarseApply<vSpinColourVectorF, vTComplexF, 2 * kNbasis> Stencil1;
  // Level 2: Fobj = the level-1 site vector, and the coarse complex type one tensor level deeper
  // (iScalar<vTComplexF>, probe L158 / Test_dwf_hdcr.cc:291).
  typedef Level1Op::siteVector L1Site;  // iVector<vTComplexF, 48>
  typedef iScalar<vTComplexF> CComplex2;
  typedef Aggregation<L1Site, CComplex2, kNbasis2> Subspace2;
  typedef GeneralCoarsenedMatrix<L1Site, CComplex2, kNbasis2> Level2Op;
  typedef Level2Op::CoarseVector CoarseVector2;
  typedef ProbeMG::StencilCoarseApply<L1Site, CComplex2, kNbasis2> Stencil2;

  // Empty string = the geometry is usable; otherwise a message naming the offending dimension.
  // Checks every divisibility the grid constructors and subdivides() would assert on, so a bad
  // (lattice, --mpi, block, block2) combination fails with a readable line instead of a bare
  // assert. Public so a caller can test candidate blocks.
  static std::string CheckGeometry(GridBase *GridF, const Coordinate &block,
                                   const Coordinate &block2)
  {
    const Coordinate fine = GridF->GlobalDimensions();
    const Coordinate procs = GridF->_processors;
    const Coordinate simd = GridF->_simd_layout;
    std::ostringstream err;
    if (fine.size() != 4 || block.size() != 4 || block2.size() != 4)
      return "lattice, block and block2 must all be 4D";
    if (block[0] % 2 != 0) {
      err << "block " << GridMGParams::CoordinateString(block)
          << ": x block must be even (the red-black grid halves x)";
      return err.str();
    }
    for (int d = 0; d < 4; ++d) {
      if (fine[d] % block[d] != 0) {
        err << "dim " << d << ": lattice " << fine[d] << " not divisible by block " << block[d];
        return err.str();
      }
      const int c1 = fine[d] / block[d];
      if (c1 % procs[d] != 0) {
        err << "dim " << d << ": level-1 extent " << c1 << " not divisible by " << procs[d]
            << " ranks";
        return err.str();
      }
      const int c1loc = c1 / procs[d];
      if (c1loc % simd[d] != 0) {
        err << "dim " << d << ": level-1 local extent " << c1loc
            << " not divisible by the fp32 SIMD layout " << simd[d] << " (change --mpi or block)";
        return err.str();
      }
      if (c1 % block2[d] != 0) {
        err << "dim " << d << ": level-1 extent " << c1 << " not divisible by block2 "
            << block2[d];
        return err.str();
      }
      const int c2 = c1 / block2[d];
      if (c2 % procs[d] != 0) {
        err << "dim " << d << ": level-2 extent " << c2 << " not divisible by " << procs[d]
            << " ranks";
        return err.str();
      }
      const int c2loc = c2 / procs[d];
      if (c2loc % simd[d] != 0) {
        err << "dim " << d << ": level-2 local extent " << c2loc
            << " not divisible by the fp32 SIMD layout " << simd[d]
            << " (change --mpi or block2)";
        return err.str();
      }
    }
    return "";
  }

  GridMGHierarchy(WCF_f &opF, GridCartesian *GridF, GridRedBlackCartesian *RBGridF, int cb,
                  const GridMGParams &params, std::string name)
      : opF_(opF), GridF_(GridF), RBGridF_(RBGridF), cb_(cb), params_(params),
        name_(std::move(name)), UmuF_(GridF), schurF_(opF),
        countedSchurF_(schurF_, counts_.fine)
  {
    GRID_ASSERT(cb_ == Even || cb_ == Odd);
    if (opF_.FermionRedBlackGrid() != (GridBase *)RBGridF_ || opF_.GaugeGrid() != (GridBase *)GridF_) {
      std::cout << GridLogError << Tag() << "opF was not built on the GridF/RBGridF passed in"
                << std::endl;
      GRID_ASSERT(0);
    }
    const std::string bad = CheckGeometry(GridF_, params_.block, params_.block2);
    if (!bad.empty()) {
      std::cout << GridLogError << Tag() << "unusable MG geometry: " << bad << " (lattice "
                << GridMGParams::CoordinateString(GridF_->GlobalDimensions()) << ", mpi "
                << GridMGParams::CoordinateString(GridF_->_processors) << ", fp32 simd "
                << GridMGParams::CoordinateString(GridF_->_simd_layout) << ", block "
                << GridMGParams::CoordinateString(params_.block) << ", block2 "
                << GridMGParams::CoordinateString(params_.block2) << ")" << std::endl;
      GRID_ASSERT(0);
    }
    Coordinate c1 = GridF_->GlobalDimensions();
    Coordinate c2 = c1;
    for (int d = 0; d < 4; ++d) {
      c1[d] = c1[d] / params_.block[d];
      c2[d] = c1[d] / params_.block2[d];
    }
    Coarse1F_.reset(new GridCartesian(c1, GridF_->_simd_layout, GridF_->_processors));
    Coarse2F_.reset(new GridCartesian(c2, GridF_->_simd_layout, GridF_->_processors));
    // The real gates (what blockProject/blockPromote assert internally), before anything costly.
    subdivides(Coarse1F_.get(), RBGridF_);
    subdivides(Coarse2F_.get(), Coarse1F_.get());

    geom1_.reset(new NonLocalStencilGeometry4D(Coarse1F_.get(), GridMGParams::stencil_hops));
    geom2_.reset(new NonLocalStencilGeometry4D(Coarse2F_.get(), GridMGParams::stencil_hops));

    // Fine noise must come from an RNG on the FULL fp32 grid: GridParallelRNG::fill of a
    // checkerboarded field builds its temporary on the RNG's own grid (Lattice_rng.h:370).
    // Level-2 noise lives on the level-1 grid and needs its own RNG there.
    rngF_.reset(new GridParallelRNG(GridF_));
    rng1_.reset(new GridParallelRNG(Coarse1F_.get()));

    std::cout << GridLogMessage << Tag() << "cb " << (cb_ == Odd ? "Odd" : "Even") << " mass "
              << opF_.Mass() << " | fine "
              << GridMGParams::CoordinateString(GridF_->GlobalDimensions()) << " level-1 "
              << GridMGParams::CoordinateString(Coarse1F_->GlobalDimensions()) << " (rdim "
              << GridMGParams::CoordinateString(Coarse1F_->_rdimensions) << ", npoint "
              << geom1_->npoint << ") level-2 "
              << GridMGParams::CoordinateString(Coarse2F_->GlobalDimensions()) << " (rdim "
              << GridMGParams::CoordinateString(Coarse2F_->_rdimensions) << ", npoint "
              << geom2_->npoint << ")" << std::endl;
  }

  ~GridMGHierarchy() { DestroyLevels(); }

  GridMGHierarchy(const GridMGHierarchy &) = delete;
  GridMGHierarchy &operator=(const GridMGHierarchy &) = delete;

  // ---- gauge update: fp32 import only (the "thin update") ------------------------------
  void SetGauge(const LatticeGaugeField &U)
  {
    ImportGaugeF(U);
    if (track_gauge_) RecordGauge(GaugeTraceFingerprint(U));  // M3 only; off = the M2 path
  }

  // ---- M3: gauge identity per generation (the MG heatbath, HASEN_GRID_MG_HEATBATH_RUNGS) ----
  // From now on every import records the fingerprint of U and of opF's doubled links, tagged
  // with the generation it produced. Call before the first SetGauge (the driver does, when it
  // builds the heatbath solvers); an earlier untracked generation simply never matches.
  void TrackGauge() { track_gauge_ = true; }
  bool TracksGauge() const { return track_gauge_; }
  // True iff the current generation's import was of a field with U's fingerprint AND opF still
  // holds the links that import produced (another importer into the same fp32 operator, e.g. a
  // mixed-CG heatbath on the rung below, is detected by the second check).
  bool Carries(const LatticeGaugeField &U) const { return Carries(GaugeTraceFingerprint(U)); }
  // Import U unless the current generation already carries it. Returns true if it imported.
  bool SetGaugeIfNew(const LatticeGaugeField &U)
  {
    GRID_ASSERT(track_gauge_);
    const GaugeFingerprint fp = GaugeTraceFingerprint(U);
    if (Carries(fp)) return false;
    ImportGaugeF(U);
    RecordGauge(fp);
    return true;
  }

  // ---- (re)build every level on the current fp32 operator --------------------------------
  void Build()
  {
    if (generation_ == 0) {
      std::cout << GridLogError << Tag() << "Build() before any SetGauge(): no gauge imported"
                << std::endl;
      GRID_ASSERT(0);
    }
    const GridMGParams &p = params_;
    const int inner_verbose = (p.verbose >= 3) ? 1 : 0;
    const std::string reason = !build_reason_.empty() ? build_reason_
                               : builds_ == 0         ? std::string("initial")
                               : rebuild_pending_     ? std::string("pending (soft tier)")
                                                      : std::string("explicit");
    build_reason_.clear();

    DestroyLevels();
    ++builds_;
    auto sync = [this]() -> double {
      accelerator_barrier();
      GridF_->Barrier();
      return usecond();
    };
    const double t0 = sync();

    // ---- level 1: null vectors on the fp32 Schur operator, gamma5-doubled in fp32 --------
    rngF_->SeedFixedIntegers(
        std::vector<int>({p.seed, p.seed + 1, p.seed + 2, p.seed + 3}));
    UV_.reset(new Subspace1(Coarse1F_.get(), RBGridF_, cb_));
    {
      Subspace1Half Agg(Coarse1F_.get(), RBGridF_, cb_);
      ProbeMG::create_subspace_gcr(*rngF_, schurF_, Agg, kNbasis, p.subspace_tol,
                                   p.subspace_rounds, p.subspace_mmax, p.subspace_nstep,
                                   p.subspace_maxiter, /*use_fast_gcr=*/true, /*quiet=*/true,
                                   /*relax=*/false, /*from_subspace=*/false, inner_verbose);
      // gamma5 chirality doubling, as Test_general_coarse_wilson.cc (QUDA spin_block_size 2).
      Gamma G5(Gamma::Algebra::Gamma5);
      for (int b = 0; b < kNbasis; ++b) {
        UV_->subspace[b] = Agg.subspace[b];
        UV_->subspace[b + kNbasis] = G5 * Agg.subspace[b];
      }
    }
    for (int b = 0; b < 2 * kNbasis; ++b) GRID_ASSERT(UV_->subspace[b].Checkerboard() == cb_);
    const double t1 = sync();

    // ---- level 1: coarsen Mpc (patch 04 makes CoarsenOperator parity-agnostic) ------------
    L1_.reset(new Level1Op(*geom1_, RBGridF_, Coarse1F_.get()));
    L1_->CoarsenOperator(schurF_, *UV_);
    linop1_.reset(new NonHermitianLinearOperator<Level1Op, CoarseVector1>(*L1_));
    if (geom1_->npoint == 9) {
      stencil1_.reset(new Stencil1(*L1_, Coarse1F_.get(), p.coarse_shift));
    } else {
      // Probe fallback (lines 1590-1601): an extent-2 dimension merges the +1/-1 shifts.
      shifted1_.reset(
          new ShiftedNonHermitianLinearOperator<Level1Op, CoarseVector1>(*L1_, p.coarse_shift));
      std::cout << GridLogMessage << Tag() << "level-1 stencil apply unavailable (npoint "
                << geom1_->npoint << " != 9); using the general apply" << std::endl;
    }
    LinearOperatorBase<CoarseVector1> &apply1 =
        stencil1_ ? static_cast<LinearOperatorBase<CoarseVector1> &>(*stencil1_)
                  : static_cast<LinearOperatorBase<CoarseVector1> &>(*shifted1_);
    counted1_.reset(new ProbeMG::CountingLinearOperator<CoarseVector1>(apply1, counts_.coarse));
    const double t2 = sync();

    // ---- level 2: the shifted level-1 stencil operator aggregated again ------------------
    rng1_->SeedFixedIntegers(
        std::vector<int>({p.seed + 4, p.seed + 5, p.seed + 6, p.seed + 7}));
    Agg2_.reset(new Subspace2(Coarse2F_.get(), Coarse1F_.get(), Even));  // full grid: cb moot
    ProbeMG::create_subspace_gcr(*rng1_, *counted1_, *Agg2_, kNbasis2, p.l2_subspace_tol,
                                 p.l2_subspace_rounds, p.l2_subspace_mmax, p.l2_subspace_nstep,
                                 p.l2_subspace_maxiter, /*use_fast_gcr=*/true, /*quiet=*/true,
                                 /*relax=*/false, /*from_subspace=*/false, inner_verbose);
    const double t3 = sync();
    // The level-1 operator is 9-point, so a 9-point level-2 stencil is exact.
    L2_.reset(new Level2Op(*geom2_, Coarse1F_.get(), Coarse2F_.get()));
    L2_->CoarsenOperator(*counted1_, *Agg2_);
    linop2_.reset(new NonHermitianLinearOperator<Level2Op, CoarseVector2>(*L2_));
    if (geom2_->npoint == 9) {
      stencil2_.reset(new Stencil2(*L2_, Coarse2F_.get(), p.coarse2_shift));
    } else {
      shifted2_.reset(
          new ShiftedNonHermitianLinearOperator<Level2Op, CoarseVector2>(*L2_, p.coarse2_shift));
      if (builds_ == 1)
        std::cout << GridLogMessage << Tag() << "level-2 stencil apply unavailable (npoint "
                  << geom2_->npoint << " != 9: a level-2 dimension has extent 2); using the"
                  << " general apply" << std::endl;
    }
    LinearOperatorBase<CoarseVector2> &apply2 =
        stencil2_ ? static_cast<LinearOperatorBase<CoarseVector2> &>(*stencil2_)
                  : static_cast<LinearOperatorBase<CoarseVector2> &>(*shifted2_);
    counted2_.reset(new ProbeMG::CountingLinearOperator<CoarseVector2>(apply2, counts_.coarse2));

    // Coarsest solve (probe Coarse2SolverF).
    coarse2Solver_.reset(new ProbeMG::FlexibleGCR<CoarseVector2>(
        p.coarse2_tol, p.coarse2_maxiter, *counted2_, trivial2_, p.coarse2_mmax, p.coarse2_nstep));
    coarse2Solver_->Level(5);
    coarse2Solver_->verbose = inner_verbose;
    coarse2Solver_->zero_guess = true;
    coarse2Solver_->verify_residual = false;
    // Level-1 smoother inside the level-2 cycle (probe L1SmootherF).
    l1Smoother_.reset(new ProbeMG::FlexibleGCR<CoarseVector1>(
        p.l1_smoother_tol, p.l1_smoother_maxiter, *counted1_, trivial1_, p.l1_smoother_mmax,
        p.l1_smoother_nstep));
    l1Smoother_->Level(4);
    l1Smoother_->verbose = inner_verbose;
    l1Smoother_->zero_guess = true;
    l1Smoother_->verify_residual = false;
    // Level-2 V-cycle (probe Precon2F).
    precon2_.reset(new ProbeMG::MGPreconditioner<L1Site, CComplex2, kNbasis2>(
        *Agg2_, *counted1_, trivial1_, *l1Smoother_, *linop2_, *coarse2Solver_));
    precon2_->Level(3);
    precon2_->fast_project = (p.project_mode != 0);
    precon2_->project_mode = p.project_mode;
    precon2_->instrument = false;
    precon2_->persistent_temps = p.persistent_temps;
    // Level-1 solve preconditioned by the level-2 V-cycle (probe CoarseSolverFastF).
    coarse1Solver_.reset(new ProbeMG::FlexibleGCR<CoarseVector1>(
        p.coarse_tol, p.coarse_maxiter, *counted1_, *precon2_, p.coarse_mmax, p.coarse_nstep));
    coarse1Solver_->Level(3);
    coarse1Solver_->verbose = inner_verbose;
    coarse1Solver_->verify_residual = false;
    coarse1Solver_->zero_guess = true;
    const double t4 = sync();

    // ---- fine level: shifted-Mpc smoother, V-cycle, fp64 boundary ------------------------
    shiftedSchurF_.reset(new ProbeMG::ShiftedSchurOperator<LatticeFermionF>(schurF_, p.smoother_shift));
    countedShiftedSchurF_.reset(
        new ProbeMG::CountingLinearOperator<LatticeFermionF>(*shiftedSchurF_, counts_.fine));
    smootherF_.reset(new ProbeMG::FlexibleGCR<LatticeFermionF>(
        p.smoother_tol, p.smoother_maxiter, *countedShiftedSchurF_, trivialF_, p.smoother_mmax,
        p.smoother_nstep));
    smootherF_->Level(2);
    smootherF_->verbose = inner_verbose;
    smootherF_->verify_residual = false;
    smootherF_->zero_guess = true;
    preconF_.reset(new ProbeMG::MGPreconditioner<vSpinColourVectorF, vTComplexF, 2 * kNbasis>(
        *UV_, countedSchurF_, trivialF_, *smootherF_, *linop1_, *coarse1Solver_));
    preconF_->fast_project = (p.project_mode != 0);
    preconF_->project_mode = p.project_mode;
    preconF_->instrument = false;
    preconF_->persistent_temps = p.persistent_temps;
    preconMixed_.reset(new ProbeMG::PrecisionChangeAdaptor<LatticeFermionD, LatticeFermionF>(
        *preconF_, RBGridF_, p.persistent_precchange));
    const double t5 = sync();

    built_ = true;
    needs_build_ = false;
    rebuild_pending_ = false;
    built_generation_ = generation_;
    last_setup_s_ = (t5 - t0) / 1.0e6;
    std::cout << GridLogMessage << Tag() << "build #" << builds_ << " (" << reason << ", gen "
              << generation_ << "): level-1 vectors " << (t1 - t0) / 1.0e6
              << " s, level-1 coarsen " << (t2 - t1) / 1.0e6 << " s, level 2 "
              << (t4 - t2) / 1.0e6 << " s (vectors " << (t3 - t2) / 1.0e6 << ", coarsen+solvers "
              << (t4 - t3) / 1.0e6 << "), fine stack " << (t5 - t4) / 1.0e6 << " s, total "
              << last_setup_s_ << " s" << std::endl;
  }

  // ---- interface used by GridMGSchurSolver -------------------------------------------------
  LinearFunction<LatticeFermionD> &Preconditioner()
  {
    GRID_ASSERT(built_ && preconMixed_);
    return *preconMixed_;
  }
  // Soft tier: rebuild at the next SetGauge.
  void NotePendingRebuild()
  {
    if (!rebuild_pending_ && params_.verbose >= 1)
      std::cout << GridLogMessage << Tag() << "rebuild pending (soft tier), executed at the next"
                << " SetGauge" << std::endl;
    rebuild_pending_ = true;
  }
  // Hard tier: rebuild now, on the fp32 operator's current gauge field.
  void ForceRebuildNow()
  {
    build_reason_ = "forced (hard tier)";
    Build();
  }
  int Generation() const { return generation_; }
  bool Built() const { return built_; }
  bool NeedsBuild() const { return needs_build_; }
  bool RebuildPending() const { return rebuild_pending_; }
  int BuiltGeneration() const { return built_generation_; }
  int Builds() const { return builds_; }
  double LastSetupSeconds() const { return last_setup_s_; }
  int Checkerboard() const { return cb_; }
  const std::string &Name() const { return name_; }
  const GridMGParams &Params() const { return params_; }
  const ProbeMG::OpCounts &Counts() const { return counts_; }

  // ---- diagnostics access (the standalone test's Galerkin checks) ---------------------------
  SchurOpF &FineSchur() { return schurF_; }
  Subspace1 &Level1Subspace() { GRID_ASSERT(built_); return *UV_; }
  Level1Op &Level1Operator() { GRID_ASSERT(built_); return *L1_; }
  // The level-1 operator the level-1 solve and the level-2 coarsening use: A1 + coarse_shift.
  LinearOperatorBase<CoarseVector1> &Level1Apply() { GRID_ASSERT(built_); return *counted1_; }
  Subspace2 &Level2Subspace() { GRID_ASSERT(built_); return *Agg2_; }
  Level2Op &Level2Operator() { GRID_ASSERT(built_); return *L2_; }
  GridCartesian *Coarse1Grid() { return Coarse1F_.get(); }
  GridCartesian *Coarse2Grid() { return Coarse2F_.get(); }
  GridCartesian *FineGridF() { return GridF_; }
  GridRedBlackCartesian *FineRBGridF() { return RBGridF_; }

 private:
  std::string Tag() const { return "[GridMG " + name_ + "] "; }

  // The M2 SetGauge body: fp64 U -> persistent fp32 UmuF_ -> opF_.ImportGauge; ++generation.
  void ImportGaugeF(const LatticeGaugeField &U)
  {
    if (!ws_gauge_) {
      ws_gauge_.reset(new precisionChangeWorkspace(GridF_, U.Grid()));
      ws_gauge_in_ = U.Grid();
    }
    GRID_ASSERT(U.Grid() == ws_gauge_in_);
    precisionChange(UmuF_, U, *ws_gauge_);
    opF_.ImportGauge(UmuF_);
    ++generation_;
    if (!built_) needs_build_ = true;
    if (rebuild_pending_) needs_build_ = true;
  }
  // M3: tag the import just made (generation_) with U's fingerprint and opF's doubled links'.
  void RecordGauge(const GaugeFingerprint &fpU)
  {
    fp_gauge_ = fpU;
    fp_links_ = GaugeTraceFingerprint(opF_.Umu);
    fp_generation_ = generation_;
  }
  bool Carries(const GaugeFingerprint &fpU) const
  {
    // The generation check: a fingerprint recorded for an older generation (an untracked
    // import happened since) never counts.
    if (!track_gauge_ || generation_ == 0 || fp_generation_ != generation_) return false;
    if (fpU != fp_gauge_) return false;
    return GaugeTraceFingerprint(opF_.Umu) == fp_links_;
  }

  void DestroyLevels()
  {
    // Reverse order of creation: every object below references something above it.
    preconMixed_.reset();
    preconF_.reset();
    smootherF_.reset();
    countedShiftedSchurF_.reset();
    shiftedSchurF_.reset();
    coarse1Solver_.reset();
    precon2_.reset();
    l1Smoother_.reset();
    coarse2Solver_.reset();
    counted2_.reset();
    stencil2_.reset();
    shifted2_.reset();
    linop2_.reset();
    L2_.reset();
    Agg2_.reset();
    counted1_.reset();
    stencil1_.reset();
    shifted1_.reset();
    linop1_.reset();
    L1_.reset();
    UV_.reset();
    built_ = false;
  }

  // ---- fixed for the hierarchy's life (declared first => destroyed last) ----
  WCF_f &opF_;
  GridCartesian *GridF_;
  GridRedBlackCartesian *RBGridF_;
  int cb_;
  GridMGParams params_;
  std::string name_;
  std::unique_ptr<GridCartesian> Coarse1F_, Coarse2F_;
  std::unique_ptr<NonLocalStencilGeometry4D> geom1_, geom2_;
  std::unique_ptr<GridParallelRNG> rngF_, rng1_;
  LatticeGaugeFieldF UmuF_;
  std::unique_ptr<precisionChangeWorkspace> ws_gauge_;
  GridBase *ws_gauge_in_ = nullptr;
  SchurOpF schurF_;
  ProbeMG::OpCounts counts_;
  ProbeMG::CountingLinearOperator<LatticeFermionF> countedSchurF_;
  TrivialPrecon<LatticeFermionF> trivialF_;
  TrivialPrecon<CoarseVector1> trivial1_;
  TrivialPrecon<CoarseVector2> trivial2_;

  // ---- level objects, destroyed and re-created by Build() ----
  std::unique_ptr<Subspace1> UV_;
  std::unique_ptr<Level1Op> L1_;
  std::unique_ptr<NonHermitianLinearOperator<Level1Op, CoarseVector1>> linop1_;
  std::unique_ptr<ShiftedNonHermitianLinearOperator<Level1Op, CoarseVector1>> shifted1_;
  std::unique_ptr<Stencil1> stencil1_;
  std::unique_ptr<ProbeMG::CountingLinearOperator<CoarseVector1>> counted1_;
  std::unique_ptr<Subspace2> Agg2_;
  std::unique_ptr<Level2Op> L2_;
  std::unique_ptr<NonHermitianLinearOperator<Level2Op, CoarseVector2>> linop2_;
  std::unique_ptr<ShiftedNonHermitianLinearOperator<Level2Op, CoarseVector2>> shifted2_;
  std::unique_ptr<Stencil2> stencil2_;
  std::unique_ptr<ProbeMG::CountingLinearOperator<CoarseVector2>> counted2_;
  std::unique_ptr<ProbeMG::FlexibleGCR<CoarseVector2>> coarse2Solver_;
  std::unique_ptr<ProbeMG::FlexibleGCR<CoarseVector1>> l1Smoother_;
  std::unique_ptr<ProbeMG::MGPreconditioner<L1Site, CComplex2, kNbasis2>> precon2_;
  std::unique_ptr<ProbeMG::FlexibleGCR<CoarseVector1>> coarse1Solver_;
  std::unique_ptr<ProbeMG::ShiftedSchurOperator<LatticeFermionF>> shiftedSchurF_;
  std::unique_ptr<ProbeMG::CountingLinearOperator<LatticeFermionF>> countedShiftedSchurF_;
  std::unique_ptr<ProbeMG::FlexibleGCR<LatticeFermionF>> smootherF_;
  std::unique_ptr<ProbeMG::MGPreconditioner<vSpinColourVectorF, vTComplexF, 2 * kNbasis>> preconF_;
  std::unique_ptr<ProbeMG::PrecisionChangeAdaptor<LatticeFermionD, LatticeFermionF>> preconMixed_;

  // ---- state ----
  bool built_ = false;
  bool needs_build_ = false;
  bool rebuild_pending_ = false;
  int generation_ = 0;
  int built_generation_ = -1;
  int builds_ = 0;
  double last_setup_s_ = 0.0;
  std::string build_reason_;
  // M3 gauge identity (TrackGauge); unused when off.
  bool track_gauge_ = false;
  int fp_generation_ = -1;
  GaugeFingerprint fp_gauge_, fp_links_;
};

}  // namespace Grid
