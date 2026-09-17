#pragma once

#include <Grid/Grid.h>
#include <quda.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace grid_quda_benchmark {

using namespace Grid;

inline int local_volume(GridBase *grid)
{
  Coordinate local = grid->LocalDimensions();
  int volume = 1;
  for (int extent : local) volume *= extent;
  return volume;
}

inline int rank_from_coordinates(const int *coordinates, void *data)
{
  auto *grid = static_cast<CartesianCommunicator *>(data);
  Coordinate coordinate(4);
  for (int dimension = 0; dimension < 4; ++dimension) coordinate[dimension] = coordinates[dimension];
  return grid->RankFromProcessorCoor(coordinate);
}

class QudaSession {
public:
  explicit QudaSession(GridBase *grid, int device = -1) : active_(false), init_seconds_(0.0)
  {
    Coordinate mpi = GridDefaultMpi();
    int dimensions[4] = {mpi[0], mpi[1], mpi[2], mpi[3]};
    auto *communicator = static_cast<CartesianCommunicator *>(grid);
    accelerator_barrier();
    grid->Barrier();
    const double start = usecond();
    setMPICommHandleQuda(static_cast<void *>(&communicator->communicator));
    initCommsGridQuda(4, dimensions, rank_from_coordinates, communicator);
    initQuda(device);
    accelerator_barrier();
    grid->Barrier();
    init_seconds_ = (usecond() - start) / 1.0e6;
    grid->GlobalMax(init_seconds_);
    active_ = true;
  }

  QudaSession(const QudaSession &) = delete;
  QudaSession &operator=(const QudaSession &) = delete;

  ~QudaSession()
  {
    if (active_) endQuda();
  }

  double init_seconds() const { return init_seconds_; }

private:
  bool active_;
  double init_seconds_;
};

inline void lexicographic_to_even_odd(const double *source, double *destination, int volume, int values_per_site,
                                      const Coordinate &local)
{
  assert((local[0] & 1) == 0);
  const int checkerboard_volume = volume / 2;
  thread_for(site, volume, {
    int remainder = site;
    const int x = remainder % local[0];
    remainder /= local[0];
    const int y = remainder % local[1];
    remainder /= local[1];
    const int z = remainder % local[2];
    const int t = remainder / local[2];
    const int parity = (x + y + z + t) & 1;
    const int destination_site = parity * checkerboard_volume + (site >> 1);
    std::memcpy(&destination[destination_site * values_per_site], &source[site * values_per_site],
                values_per_site * sizeof(double));
  });
}

inline void even_odd_to_lexicographic(const double *source, double *destination, int volume, int values_per_site,
                                      const Coordinate &local)
{
  assert((local[0] & 1) == 0);
  const int checkerboard_volume = volume / 2;
  thread_for(site, volume, {
    int remainder = site;
    const int x = remainder % local[0];
    remainder /= local[0];
    const int y = remainder % local[1];
    remainder /= local[1];
    const int z = remainder % local[2];
    const int t = remainder / local[2];
    const int parity = (x + y + z + t) & 1;
    const int source_site = parity * checkerboard_volume + (site >> 1);
    std::memcpy(&destination[site * values_per_site], &source[source_site * values_per_site],
                values_per_site * sizeof(double));
  });
}

template <class FermionField>
inline void fermion_to_even_odd(const FermionField &field, double *buffer)
{
  using SiteSpinor = typename FermionField::scalar_object;
  static_assert(sizeof(SiteSpinor) == 24 * sizeof(double), "unexpected Grid spinor layout");
  const int volume = local_volume(field.Grid());
  std::vector<SiteSpinor> sites;
  unvectorizeToLexOrdArray(sites, field);
  lexicographic_to_even_odd(reinterpret_cast<const double *>(sites.data()), buffer, volume, 24,
                            field.Grid()->LocalDimensions());
}

template <class FermionField>
inline void even_odd_to_fermion(const double *buffer, FermionField &field)
{
  using SiteSpinor = typename FermionField::scalar_object;
  static_assert(sizeof(SiteSpinor) == 24 * sizeof(double), "unexpected Grid spinor layout");
  const int volume = local_volume(field.Grid());
  std::vector<SiteSpinor> sites(volume);
  even_odd_to_lexicographic(buffer, reinterpret_cast<double *>(sites.data()), volume, 24,
                            field.Grid()->LocalDimensions());
  vectorizeFromLexOrdArray(sites, field);
}

template <class FermionField>
inline void fermion_rb_to_buffer(const FermionField &field, double *buffer)
{
  using SiteSpinor = typename FermionField::scalar_object;
  static_assert(sizeof(SiteSpinor) == 24 * sizeof(double), "unexpected Grid spinor layout");
  std::vector<SiteSpinor> sites;
  unvectorizeToLexOrdArray(sites, field);
  std::memcpy(buffer, sites.data(), sites.size() * sizeof(SiteSpinor));
}

template <class FermionField>
inline void buffer_to_fermion_rb(const double *buffer, FermionField &field)
{
  using SiteSpinor = typename FermionField::scalar_object;
  static_assert(sizeof(SiteSpinor) == 24 * sizeof(double), "unexpected Grid spinor layout");
  std::vector<SiteSpinor> sites(local_volume(field.Grid()));
  std::memcpy(sites.data(), buffer, sites.size() * sizeof(SiteSpinor));
  vectorizeFromLexOrdArray(sites, field);
}

inline void gauge_to_lexicographic(const LatticeGaugeField &gauge, std::array<std::vector<double>, 4> &buffers)
{
  using SiteGauge = LatticeGaugeField::scalar_object;
  static_assert(sizeof(SiteGauge) == 4 * 18 * sizeof(double), "unexpected Grid gauge layout");

  const int volume = local_volume(gauge.Grid());
  std::vector<SiteGauge> sites;
  unvectorizeToLexOrdArray(sites, gauge);
  const auto *source = reinterpret_cast<const double *>(sites.data());
  for (auto &buffer : buffers) buffer.resize(18 * volume);

  thread_for(site, volume, {
    for (int direction = 0; direction < 4; ++direction) {
      std::memcpy(&buffers[direction][18 * site], &source[72 * site + 18 * direction], 18 * sizeof(double));
    }
  });
}

inline void lexicographic_to_gauge(const std::array<std::vector<double>, 4> &buffers, LatticeGaugeField &gauge)
{
  using SiteGauge = LatticeGaugeField::scalar_object;
  const int volume = local_volume(gauge.Grid());
  std::vector<SiteGauge> sites(volume);
  auto *destination = reinterpret_cast<double *>(sites.data());

  thread_for(site, volume, {
    for (int direction = 0; direction < 4; ++direction) {
      std::memcpy(&destination[72 * site + 18 * direction], &buffers[direction][18 * site], 18 * sizeof(double));
    }
  });
  vectorizeFromLexOrdArray(sites, gauge);
}

inline bool exact_buffer_match(const double *left, const double *right, std::size_t count)
{
  return std::memcmp(left, right, count * sizeof(double)) == 0;
}

// Exercise every host-side layout bridge before QUDA is initialized.  The
// deterministic payload catches misplaced sites/components exactly (all of
// these paths only copy bytes), while the RB comparison also checks the
// assumption that Grid's odd-checkerboard order equals QUDA's odd EO slab.
inline void validate_conversion_round_trips(GridBase *grid, GridRedBlackCartesian *rb_grid)
{
  const int volume = local_volume(grid);
  const Coordinate local = grid->LocalDimensions();
  if ((local[0] & 1) != 0) throw std::runtime_error("x dimension must be even for QUDA even-odd ordering");

  // Multi-rank parity constraint. lexicographic_to_even_odd() derives a site's
  // checkerboard from its LOCAL coordinates, whereas Grid defines the
  // checkerboard from GLOBAL ones. A rank's local origin sits at
  // (p0*L0, p1*L1, p2*L2, p3*L3), so the two agree on every rank if and only if
  // every local extent is even -- then the origin's coordinate sum is even for
  // all p and the parities coincide. With an odd local extent, ranks at odd
  // processor coordinates get their checkerboard inverted and Grid's Odd half
  // no longer corresponds to QUDA's odd slab.
  //
  // The red-black round trip below does detect this, but reports it as an
  // opaque "ordering does not match" failure. Name the real cause instead:
  // it is a property of the chosen decomposition, not of the data.
  //
  // Production 48.48.48.96 on MPI=1.2.2.4 gives local 48x24x24x24 -- all even,
  // so the constraint is satisfied there and on every geometry used so far.
  for (int dimension = 0; dimension < 4; ++dimension) {
    if ((local[dimension] & 1) != 0)
      throw std::runtime_error(
          "every local lattice extent must be even: local dimension " + std::to_string(dimension) +
          " is " + std::to_string(local[dimension]) +
          ", which inverts the checkerboard on ranks at odd processor coordinates and breaks the "
          "Grid/QUDA even-odd correspondence. Choose a decomposition with even local extents.");
  }

  std::vector<double> lex(24 * volume);
  for (std::size_t index = 0; index < lex.size(); ++index)
    lex[index] = static_cast<double>((index % 104729) + 1);
  std::vector<double> even_odd(lex.size());
  std::vector<double> lex_back(lex.size());
  lexicographic_to_even_odd(lex.data(), even_odd.data(), volume, 24, local);
  even_odd_to_lexicographic(even_odd.data(), lex_back.data(), volume, 24, local);
  if (!exact_buffer_match(lex.data(), lex_back.data(), lex.size()))
    throw std::runtime_error("full-spinor lexicographic/even-odd conversion round trip failed");

  LatticeFermion full_source(grid);
  LatticeFermion full_result(grid);
  std::vector<typename LatticeFermion::scalar_object> full_sites(volume);
  for (int site = 0; site < volume; ++site)
    std::memcpy(&full_sites[site], &lex[24 * site], 24 * sizeof(double));
  vectorizeFromLexOrdArray(full_sites, full_source);
  fermion_to_even_odd(full_source, even_odd.data());
  even_odd_to_fermion(even_odd.data(), full_result);
  LatticeFermion full_difference(grid);
  full_difference = full_source - full_result;
  if (norm2(full_difference) != 0.0)
    throw std::runtime_error("full-spinor Grid/buffer conversion round trip failed");

  LatticeFermion rb_source(rb_grid);
  LatticeFermion rb_result(rb_grid);
  rb_source.Checkerboard() = Odd;
  rb_result.Checkerboard() = Odd;
  pickCheckerboard(Odd, rb_source, full_source);
  std::vector<double> rb_buffer(12 * volume);
  fermion_rb_to_buffer(rb_source, rb_buffer.data());
  if (!exact_buffer_match(&even_odd[12 * volume], rb_buffer.data(), rb_buffer.size()))
    throw std::runtime_error("red-black spinor ordering does not match QUDA odd-half ordering");
  buffer_to_fermion_rb(rb_buffer.data(), rb_result);
  LatticeFermion rb_difference(rb_grid);
  rb_difference.Checkerboard() = Odd;
  rb_difference = rb_source - rb_result;
  if (norm2(rb_difference) != 0.0)
    throw std::runtime_error("red-black spinor Grid/buffer conversion round trip failed");

  LatticeGaugeField gauge_source(grid);
  LatticeGaugeField gauge_result(grid);
  std::vector<typename LatticeGaugeField::scalar_object> gauge_sites(volume);
  auto *gauge_values = reinterpret_cast<double *>(gauge_sites.data());
  for (std::size_t index = 0; index < 72ULL * static_cast<std::size_t>(volume); ++index)
    gauge_values[index] = static_cast<double>((index % 130363) + 1);
  vectorizeFromLexOrdArray(gauge_sites, gauge_source);
  std::array<std::vector<double>, 4> gauge_buffers;
  gauge_to_lexicographic(gauge_source, gauge_buffers);
  for (int direction = 0; direction < 4; ++direction) {
    std::vector<double> gauge_even_odd(18 * volume);
    std::vector<double> gauge_lex_back(18 * volume);
    lexicographic_to_even_odd(gauge_buffers[direction].data(), gauge_even_odd.data(), volume, 18, local);
    even_odd_to_lexicographic(gauge_even_odd.data(), gauge_lex_back.data(), volume, 18, local);
    if (!exact_buffer_match(gauge_buffers[direction].data(), gauge_lex_back.data(), gauge_lex_back.size()))
      throw std::runtime_error("gauge lexicographic/even-odd conversion round trip failed");
  }
  lexicographic_to_gauge(gauge_buffers, gauge_result);
  LatticeGaugeField gauge_difference(grid);
  gauge_difference = gauge_source - gauge_result;
  if (norm2(gauge_difference) != 0.0)
    throw std::runtime_error("gauge Grid/buffer conversion round trip failed");
}

inline void apply_antiperiodic_time_boundary(std::array<std::vector<double>, 4> &gauge, GridBase *grid)
{
  const Coordinate local = grid->LocalDimensions();
  const Coordinate processors = grid->ProcessorGrid();
  const Coordinate processor = grid->ThisProcessorCoor();
  if (processor[3] != processors[3] - 1) return;

  const int volume = local_volume(grid);
  const int final_local_time = local[3] - 1;
  for (int site = 0; site < volume; ++site) {
    int remainder = site;
    remainder /= local[0];
    remainder /= local[1];
    remainder /= local[2];
    const int time = remainder;
    if (time != final_local_time) continue;
    for (int component = 0; component < 18; ++component) gauge[3][18 * site + component] *= -1.0;
  }
}

inline QudaGaugeParam make_gauge_param(GridBase *grid, QudaPrecision precise, QudaPrecision sloppy,
                                       QudaReconstructType precise_reconstruct,
                                       QudaReconstructType sloppy_reconstruct,
                                       bool antiperiodic_time)
{
  QudaGaugeParam param = newQudaGaugeParam();
  const Coordinate local = grid->LocalDimensions();
  for (int dimension = 0; dimension < 4; ++dimension) param.X[dimension] = local[dimension];
  param.type = QUDA_WILSON_LINKS;
  param.location = QUDA_CPU_FIELD_LOCATION;
  param.cpu_prec = QUDA_DOUBLE_PRECISION;
  param.cuda_prec = precise;
  param.cuda_prec_sloppy = sloppy;
  param.cuda_prec_refinement_sloppy = sloppy;
  param.cuda_prec_precondition = sloppy;
  param.reconstruct = precise_reconstruct;
  param.reconstruct_sloppy = sloppy_reconstruct;
  param.reconstruct_refinement_sloppy = sloppy_reconstruct;
  param.reconstruct_precondition = sloppy_reconstruct;
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = antiperiodic_time ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
  param.anisotropy = 1.0;
  param.tadpole_coeff = 1.0;
  param.scale = 1.0;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  const int x_face = local[1] * local[2] * local[3] / 2;
  const int y_face = local[0] * local[2] * local[3] / 2;
  const int z_face = local[0] * local[1] * local[3] / 2;
  const int t_face = local[0] * local[1] * local[2] / 2;
  param.ga_pad = std::max({x_face, y_face, z_face, t_face});
  param.struct_size = sizeof(param);
  return param;
}

inline QudaInvertParam make_invert_param(bool clover, double mass, double csw, QudaPrecision precise,
                                         QudaPrecision sloppy, double tolerance, int maximum_iterations)
{
  QudaInvertParam param = newQudaInvertParam();
  const double kappa = 1.0 / (2.0 * (4.0 + mass));
  param.dslash_type = clover ? QUDA_CLOVER_WILSON_DSLASH : QUDA_WILSON_DSLASH;
  param.mass = mass;
  param.kappa = kappa;
  param.Ls = 1;
  param.clover_coeff = csw * kappa;
  param.clover_csw = csw;
  param.inv_type = QUDA_CG_INVERTER;
  param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  param.solve_type = QUDA_NORMOP_PC_SOLVE;
  // Action-dependent by necessity, not by choice: DiracWilsonPC::M
  // (quda/lib/dirac_wilson.cpp) errorQuda()s on anything but
  // QUDA_MATPC_EVEN_EVEN / QUDA_MATPC_ODD_ODD, because a Wilson diagonal block
  // is the scalar (4+mass) and the symmetric/asymmetric distinction collapses.
  // Both choices below are the one that matches Grid's
  // SchurDiagMooeeOperator (Moo - Moe Mee^-1 Meo) on the odd checkerboard:
  //   Wilson  QUDA_MATPC_ODD_ODD:       1     - kappa^2 D_oe D_eo
  //   Clover  QUDA_MATPC_ODD_ODD_ASYM:  A_oo  - kappa^2 D_oe A_ee^-1 D_eo
  // and in both cases Mpc_quda = 2*kappa*Mpc_grid, so the 2*kappa / 4*kappa^2
  // corrections in apply_mat()/apply_normal()/solve() are unchanged.
  param.matpc_type = clover ? QUDA_MATPC_ODD_ODD_ASYMMETRIC : QUDA_MATPC_ODD_ODD;
  param.dagger = QUDA_DAG_NO;
  param.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;
  param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  param.use_init_guess = QUDA_USE_INIT_GUESS_NO;
  param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
  param.tol = tolerance;
  param.maxiter = maximum_iterations;
  param.reliable_delta = 1e-3;
  param.use_sloppy_partial_accumulator = 0;
  param.solution_accumulator_pipeline = 1;
  param.pipeline = 0;
  param.tol_hq = 0.0;
  param.Nsteps = 5;
  param.input_location = QUDA_CPU_FIELD_LOCATION;
  param.output_location = QUDA_CPU_FIELD_LOCATION;
  param.dirac_order = QUDA_DIRAC_ORDER;
  param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  param.cpu_prec = QUDA_DOUBLE_PRECISION;
  param.cuda_prec = precise;
  param.cuda_prec_sloppy = sloppy;
  param.cuda_prec_refinement_sloppy = sloppy;
  param.cuda_prec_precondition = sloppy;
  param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
  param.clover_cuda_prec = precise;
  param.clover_cuda_prec_sloppy = sloppy;
  param.clover_cuda_prec_refinement_sloppy = sloppy;
  param.clover_cuda_prec_precondition = sloppy;
  param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  param.clover_location = QUDA_CUDA_FIELD_LOCATION;
  param.compute_clover = clover ? 1 : 0;
  param.compute_clover_inverse = clover ? 1 : 0;
  param.return_clover = 0;
  param.return_clover_inverse = 0;
  param.verbosity = QUDA_SUMMARIZE;
  param.struct_size = sizeof(param);
  return param;
}

// ---------------------------------------------------------------------------
// Multigrid.
//
// WHAT THE MG ROW SOLVES, AND WHY IT IS NOT Mpc^dag Mpc.
// Multigrid preconditions Mpc directly and never squares it -- squaring is a
// CG-specific device for obtaining an HPD operator. QUDA supports exactly this on
// a single checkerboard:
//
//   outer:  QUDA_GCR_INVERTER + inv_type_precondition = QUDA_MG_INVERTER
//           + QUDA_DIRECT_PC_SOLVE + QUDA_MATPC_SOLUTION
//           + matpc_type inherited from the CG rows (ODD_ODD_ASYMMETRIC)
//
// Permitted at quda/tests/invert_test.cpp:452; quda/tests/utils/set_params.cpp:536-548
// describes preconditioned-outer + preconditioned-smoother as "the optimal
// combination in general for Wilson-type operators". QUDA_NORMOP_PC_SOLVE is NOT
// supported with MG, and is not wanted.
//
// Consequently the production gamma5 two-solve route (QudaMGSchurSolver.h) is not
// used here: it exists because the HMC *force* needs (Mpc^dag Mpc)^-1 Phi, not
// because MG is restricted to the full lattice.
//
// TWO SEPARATE QudaInvertParams. quda/lib/interface_quda.cpp:2869's
// "Outer MG solver can only use QUDA_DIRECT_SOLVE" constrains
// mg_param.invert_param -- the MG hierarchy's OWN operator -- not the outer param
// passed to invertQuda. The inner one below therefore keeps DIRECT_SOLVE /
// MAT_SOLUTION while the outer runs DIRECT_PC_SOLVE / MATPC_SOLUTION.
//
// Parameters are the NPLQCD production values carried by
// Grid-TXQCD/Grid/util/QudaMultigridConfig.h, so the benchmark's QUDA MG is the
// MG production runs. The HMC cadence knobs of that file are dropped entirely
// rather than set to zero: the gauge field never changes here, so one setup and N
// solves is the only mode, and a knob that cannot fire is a knob that can mislead.
// DEFAULTS ARE TRANSCRIBED FROM THE PRODUCTION CONFIGURATION'S OWN METADATA, not
// from QudaMultigridConfig.h. The `.lime` file carries the <MULTIGRIDParams> block
// Chroma actually ran for the light quark (Mass -0.2416), and it differs from that
// header in blocking, per-level n_vec, coarse iteration counts, omega, setup
// maxiter and Krylov depths. Reading the recipe off the configuration we are
// actually inverting is the only way it cannot drift.
//
// ⚠️ TWO PRODUCTION VALUES ARE DELIBERATELY NOT COPIED:
//
//   Precision HALF  -> we use the loaded sloppy precision instead. Chroma loaded a
//                      half gauge copy; this harness loads only `precise` and
//                      `sloppy`, so HALF would reference an unloaded copy and give
//                      a garbage volume. Our own QudaCloverInverter made the same
//                      substitution for the same reason.
//   RsdTarget 1e-12 -> the benchmark's tolerance is a campaign-wide knob (1e-10),
//                      shared with the CG rows so the two are comparable.
//
// ⚠️ Production also runs this on a STOUT-SMEARED field (rho 0.125, n_smear 1).
// Phase 1 is unsmeared, so these parameters are being applied to an operator they
// were not tuned for -- doc §4's "unsmeared understates MG" caveat, concretely.
struct QudaMgParams {
  // 3 = the production configuration. 2 = the like-for-like row, since Grid's path
  // is two-level by construction. See like_for_like() below.
  int n_level = 3;
  // Level l -> l+1 blocking. Config: <Blocking> 3 3 3 4 / 2 2 2 2.
  // ⛔ {3,3,3,4} is NOT reproducible on Grid: its red-black grid halves
  // _rdimensions[0], so the x-block must be EVEN. That is why the like-for-like
  // row uses {4,4,4,4} and this one is QUDA-only.
  std::vector<std::array<int, 4>> geo_block_size = {{{3, 3, 3, 4}}, {{2, 2, 2, 2}}};
  // Config: <NullVectors>24 32</NullVectors> -- PER LEVEL, not a single value.
  // Level 0's 24 with spin_block_size[0]=2 gives 48 coarse dof, which is what Grid
  // reaches via nbasis=24 gamma5-doubled. Compare 48 against 48, never 24 vs 24.
  // Level 1's 32 has no Grid counterpart at all (Grid is two-level).
  std::vector<int> n_vec_levels = {24, 32};
  int n_vec = 24; // fallback for levels beyond n_vec_levels
  // Config: <SubspaceSolver>CG</SubspaceSolver>, MaxIterSubspaceCreate 500,
  // RsdTargetSubspaceCreate 5e-06.
  QudaInverterType setup_inv = QUDA_CG_INVERTER;
  int setup_maxiter = 500;
  double setup_tol = 5e-6;
  // Config: <SmootherType>CA_GCR</SmootherType>, SmootherTol 0.25,
  // Pre-SmootherApplications 0, Post-SmootherApplications 8,
  // RelaxationOmegaMG 1.0 (NOT the 0.95 in QudaMultigridConfig.h).
  QudaInverterType smoother = QUDA_CA_GCR_INVERTER;
  int nu_pre = 0;
  int nu_post = 8;
  double omega = 1.0;
  double smoother_tol = 0.25;
  // Config: <CoarseResidual>0.1 0.1 0.1</CoarseResidual>,
  // <MaxCoarseIterations>12 12 8</MaxCoarseIterations>,
  // <CoarseSolverType>GCR, CA_GCR</CoarseSolverType>.
  double coarse_solver_tol = 0.1;
  std::vector<int> coarse_solver_maxiter_levels = {12, 12, 8};
  int coarse_solver_maxiter = 12; // fallback
  // ⛔ ONLY THE COARSEST LEVEL MAY USE A COMMUNICATION-AVOIDING SOLVER.
  // QUDA attaches the next-coarser level to every other level as an EXPLICIT
  // preconditioner, and quda/lib/solver.cpp:52 accepts an explicit preconditioner
  // for GCR and PCG only -- a CA_GCR there aborts with
  //   "Explicit preconditoner not supported for 22 solver".
  // (Found by running: the 3-level leg died there while the 2-level leg, whose
  // only coarse level IS the coarsest, passed.)
  //
  // The config's <CoarseSolverType>GCR, CA_GCR</CoarseSolverType> says exactly
  // this: GCR on the intermediate level, CA_GCR on the coarsest. It is expressed
  // as two named fields rather than an indexed vector because QUDA reads
  // coarse_solver[param.level + 1] (multigrid.cpp:576), so a vector here invites
  // an off-by-one that only shows up at n_level >= 3.
  QudaInverterType intermediate_coarse_solver = QUDA_GCR_INVERTER;
  QudaInverterType coarsest_solver = QUDA_CA_GCR_INVERTER;
  // Config: <OuterGCRNKrylov>20</OuterGCRNKrylov> (the inner PrecondGCRNKrylov is
  // 10, set on the MG's own invert param).
  int outer_gcr_nkrylov = 20;
  // ⛔ MUST MATCH A GAUGE COPY THAT IS ACTUALLY RESIDENT ON THE GPU.
  // make_gauge_param loads exactly two copies, `precise` and `sloppy`, and sets
  // gauge_param.cuda_prec_precondition = sloppy. A preconditioner precision that
  // is neither references an unloaded copy and QUDA reports a garbage volume
  // rather than an error -- this is the failure the production
  // QudaMultigridConfig.h header warns about for HALF.
  //
  // The NPLQCD reference MG uses half and production uses single, but neither is
  // safe to hardcode HERE, because this benchmark's sloppy precision is a campaign
  // knob: PRECISION=strict gives sloppy=DOUBLE, PRECISION=production gives
  // sloppy=SINGLE. Hardcoding SINGLE would silently break every strict-precision
  // MG row.
  //
  // QUDA_INVALID_PRECISION (the default) means "inherit whatever the gauge was
  // actually loaded at"; build_multigrid resolves it and rejects any explicit
  // value that does not match.
  QudaPrecision precondition_prec = QUDA_INVALID_PRECISION;
  // Precision of the near-null vectors and of the COARSE-level halo exchanges.
  // Production (and the NPLQCD reference) run both at HALF.
  //
  // ⛔ DO NOT SET THESE TO QUDA_DOUBLE_PRECISION against quda-install-mpi-mg.
  // Double-precision MG is a COMPILE-TIME cmake option (QUDA_MULTIGRID_DOUBLE),
  // OFF in this build and off by default upstream, so a double null-vector
  // precision does not fail validation -- it aborts partway through MG SETUP at
  // `is_enabled_multigrid_double()`, quda/build-mpi-mg/lib/block_orthogonalize_24_32.cu:290.
  // Measured 2026-09-15: run c3_qudamg_fp64, 16 ranks, all aborted there.
  // They remain knobs so that a QUDA rebuilt with QUDA_MULTIGRID_DOUBLE=ON can
  // supply the all-fp64 row that would isolate machinery quality from precision
  // in the Grid comparison. See probe_quda_mg_clover.cc for why that row matters.
  QudaPrecision null_prec = QUDA_HALF_PRECISION;
  QudaPrecision coarse_halo_prec = QUDA_HALF_PRECISION;
  // QUDA's own MG verification. Slow; worth turning on once during bring-up.
  bool run_verify = false;
  QudaVerbosity verbosity = QUDA_SUMMARIZE;

  // The row that may be compared against Grid. Everything production does that
  // Grid structurally cannot is stripped out, and each removal is a recorded
  // mismatch rather than a silent simplification:
  //   3 levels     -> 2          (Grid's MGPreconditioner holds ONE coarse level)
  //   {3,3,3,4}    -> {4,4,4,4}  (Grid's red-black x-block must be EVEN)
  //   n_vec 24,32  -> 24         (level 1 has no Grid counterpart)
  // Tolerances, smoother counts and coarse iteration caps are KEPT, because those
  // Grid can match -- so the parameters that differ are exactly the ones that
  // cannot be matched, and nothing is changed merely for convenience.
  static QudaMgParams like_for_like()
  {
    QudaMgParams p;
    p.n_level = 2;
    p.geo_block_size = {{{4, 4, 4, 4}}};
    p.n_vec_levels = {24};
    p.coarse_solver_maxiter_levels = {12, 12};
    // Plain GCR on the coarsest level, NOT the config's CA_GCR: Grid's coarse
    // solver is PrecGeneralisedConjugateResidualNonHermitian, which is a plain
    // GCR. Matching it is the point of this variant.
    p.coarsest_solver = QUDA_GCR_INVERTER;
    return p;
  }
};

// Inner invert param -- the MG hierarchy's own operator. Copies only the physics
// fields from the outer and resets every solver/control field explicitly:
// inheriting wholesale makes the MG internals see the outer's preconditioner
// pointer and report garbage volumes.
inline QudaInvertParam make_mg_inner_invert_param(const QudaInvertParam &outer, const QudaMgParams &mg)
{
  QudaInvertParam p = newQudaInvertParam();

  p.dslash_type = outer.dslash_type;
  p.kappa = outer.kappa;
  p.mass = outer.mass;
  p.Ls = outer.Ls;
  p.clover_csw = outer.clover_csw;
  p.clover_coeff = outer.clover_coeff;
  p.compute_clover = 0; // already computed on the outer load
  p.compute_clover_inverse = 0;
  // QUDA's MG expects KAPPA normalization on its inner param even when the outer
  // differs; a mismatch shows up as "Spinor volume X doesn't match gauge volume Y"
  // with garbage Y rather than as an explicit error. The outer here is KAPPA too.
  p.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  p.solver_normalization = outer.solver_normalization;
  p.gamma_basis = outer.gamma_basis;
  p.dirac_order = outer.dirac_order;
  p.input_location = outer.input_location;
  p.output_location = outer.output_location;
  p.matpc_type = outer.matpc_type; // keeps the MG Schur-consistent with the outer
  p.dagger = QUDA_DAG_NO;

  p.cpu_prec = outer.cpu_prec;
  p.cuda_prec = outer.cuda_prec;
  p.cuda_prec_sloppy = outer.cuda_prec_sloppy;
  p.cuda_prec_refinement_sloppy = outer.cuda_prec_refinement_sloppy;
  p.cuda_prec_precondition = mg.precondition_prec;
  p.clover_cpu_prec = outer.clover_cpu_prec;
  p.clover_cuda_prec = outer.clover_cuda_prec;
  p.clover_cuda_prec_sloppy = outer.clover_cuda_prec_sloppy;
  p.clover_cuda_prec_refinement_sloppy = outer.clover_cuda_prec_refinement_sloppy;
  p.clover_cuda_prec_precondition = mg.precondition_prec;
  p.clover_order = outer.clover_order;

  p.inv_type = QUDA_GCR_INVERTER;
  p.tol = 1e-10;
  p.maxiter = 1000;
  p.reliable_delta = 1e-5;
  p.gcrNkrylov = 10;
  // REQUIRED: interface_quda.cpp:2869 rejects anything else on THIS param. It is a
  // statement about the MG hierarchy's internal operator, not about the solve the
  // caller may request -- the outer param below runs DIRECT_PC_SOLVE.
  p.solve_type = QUDA_DIRECT_SOLVE;
  p.solution_type = QUDA_MAT_SOLUTION;
  p.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  p.preconditioner = nullptr; // MG is not nested

  p.verbosity_precondition = QUDA_SILENT;
  p.verbosity = mg.verbosity;
  p.struct_size = sizeof(p);
  return p;
}

// Caller must set mg_param.invert_param before calling newMultigridQuda.
inline QudaMultigridParam make_multigrid_param(const QudaMgParams &mg)
{
  QudaMultigridParam p = newQudaMultigridParam();
  p.n_level = mg.n_level;
  assert(mg.n_level <= QUDA_MAX_MG_LEVEL);
  assert(static_cast<int>(mg.geo_block_size.size()) >= mg.n_level - 1);

  for (int l = 0; l < mg.n_level; ++l) {
    if (l < mg.n_level - 1) {
      for (int d = 0; d < 4; ++d) p.geo_block_size[l][d] = mg.geo_block_size[l][d];
    } else {
      for (int d = 0; d < 4; ++d) p.geo_block_size[l][d] = 1;
    }
    // QUDA_MAX_DIM extra dims must be 1 or QUDA computes a coarse volume from
    // uninitialized memory.
    for (int d = 4; d < QUDA_MAX_DIM; ++d) p.geo_block_size[l][d] = 1;
    // Chiral spin-halving at the fine level only; this is the counterpart of
    // Grid's gamma5 doubling.
    p.spin_block_size[l] = (l == 0) ? 2 : 1;

    // Per-level where the production config is per-level, falling back to the
    // scalar for any level it does not name.
    p.n_vec[l] = (l < static_cast<int>(mg.n_vec_levels.size())) ? mg.n_vec_levels[l] : mg.n_vec;
    p.precision_null[l] = mg.null_prec;
    p.n_block_ortho[l] = 1;
    p.block_ortho_two_pass[l] = QUDA_BOOLEAN_TRUE;

    p.setup_inv_type[l] = mg.setup_inv;
    p.num_setup_iter[l] = 1;
    p.setup_maxiter[l] = mg.setup_maxiter;
    p.setup_maxiter_refresh[l] = 0; // fixed gauge: no refresh cadence exists
    p.setup_tol[l] = mg.setup_tol;
    p.setup_ca_basis[l] = QUDA_POWER_BASIS;
    p.setup_ca_basis_size[l] = 4;
    p.setup_ca_lambda_min[l] = 0.0;
    p.setup_ca_lambda_max[l] = -1.0;
    p.n_vec_batch[l] = 1;

    // QUDA reads this as coarse_solver[param.level + 1], i.e. "the solver used ON
    // level l", so index 0 is never consulted. CA only at the coarsest -- see
    // QudaMgParams::coarsest_solver.
    p.coarse_solver[l] = (l == mg.n_level - 1) ? mg.coarsest_solver : mg.intermediate_coarse_solver;
    p.coarse_solver_tol[l] = mg.coarse_solver_tol;
    p.coarse_solver_maxiter[l] = (l < static_cast<int>(mg.coarse_solver_maxiter_levels.size()))
                                     ? mg.coarse_solver_maxiter_levels[l]
                                     : mg.coarse_solver_maxiter;
    p.coarse_solver_ca_basis[l] = QUDA_POWER_BASIS;
    p.coarse_solver_ca_basis_size[l] = 4;
    p.coarse_solver_ca_lambda_min[l] = 0.0;
    p.coarse_solver_ca_lambda_max[l] = -1.0;

    p.smoother[l] = mg.smoother;
    p.smoother_tol[l] = mg.smoother_tol;
    p.nu_pre[l] = mg.nu_pre;
    p.nu_post[l] = mg.nu_post;
    p.omega[l] = mg.omega;
    // QUDA promotes coarse link storage to HALF internally; the halo precision has
    // to match, so only level 0 follows the precondition precision.
    p.smoother_halo_precision[l] = (l == 0) ? mg.precondition_prec : mg.coarse_halo_prec;
    p.smoother_schwarz_type[l] = QUDA_INVALID_SCHWARZ;
    p.smoother_schwarz_cycle[l] = 1;
    p.smoother_solver_ca_basis[l] = QUDA_POWER_BASIS;
    p.smoother_solver_ca_lambda_min[l] = 0.0;
    p.smoother_solver_ca_lambda_max[l] = -1.0;

    p.cycle_type[l] = QUDA_MG_CYCLE_RECURSIVE;
    // THE PAIRING THAT MAKES A SINGLE-PARITY OUTER SOLVE LEGAL. With
    // smoother_solve_type = QUDA_DIRECT_PC_SOLVE below, level 0 = MATPC_SOLUTION is
    // QUDA's "scenario 3": single-parity residual coarsening throughout.
    // multigrid.cpp:1165 forbids only the reverse (outer MATPC with inner MAT), and
    // multigrid.cpp:1168 requires DIRECT_PC_SOLVE whenever the inner type is MATPC.
    p.coarse_grid_solution_type[l] = (l == mg.n_level - 1) ? QUDA_MAT_SOLUTION : QUDA_MATPC_SOLUTION;
    p.smoother_solve_type[l] = QUDA_DIRECT_PC_SOLVE;
    p.global_reduction[l] = QUDA_BOOLEAN_TRUE;
    p.location[l] = QUDA_CUDA_FIELD_LOCATION;
    p.setup_location[l] = QUDA_CUDA_FIELD_LOCATION;
    p.use_eig_solver[l] = QUDA_BOOLEAN_FALSE;
    p.verbosity[l] = mg.verbosity;
    p.setup_use_mma[l] = QUDA_BOOLEAN_FALSE;
    p.dslash_use_mma[l] = QUDA_BOOLEAN_FALSE;
    p.transfer_use_mma[l] = QUDA_BOOLEAN_FALSE;
    p.vec_load[l] = QUDA_BOOLEAN_FALSE;
    p.vec_store[l] = QUDA_BOOLEAN_FALSE;
    std::snprintf(p.vec_infile[l], sizeof(p.vec_infile[l]), "%s", "");
    std::snprintf(p.vec_outfile[l], sizeof(p.vec_outfile[l]), "%s", "");
  }

  p.setup_type = QUDA_NULL_VECTOR_SETUP;
  p.pre_orthonormalize = QUDA_BOOLEAN_FALSE;
  p.post_orthonormalize = QUDA_BOOLEAN_TRUE;
  p.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;
  p.generate_all_levels = QUDA_BOOLEAN_TRUE;
  p.run_verify = mg.run_verify ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  p.run_low_mode_check = QUDA_BOOLEAN_FALSE;
  p.run_oblique_proj_check = QUDA_BOOLEAN_FALSE;
  p.coarse_guess = QUDA_BOOLEAN_FALSE;
  p.preserve_deflation = QUDA_BOOLEAN_FALSE;
  p.allow_truncation = QUDA_BOOLEAN_FALSE;
  p.staggered_kd_dagger_approximation = QUDA_BOOLEAN_FALSE;
  p.thin_update_only = QUDA_BOOLEAN_FALSE;

  p.struct_size = sizeof(p);
  return p;
}

class QudaOperator {
public:
  QudaOperator(GridBase *grid, const LatticeGaugeField &gauge, bool clover, double mass, double csw,
               QudaPrecision precise, QudaPrecision sloppy,
               QudaReconstructType precise_reconstruct, QudaReconstructType sloppy_reconstruct,
               double tolerance, int maximum_iterations, bool antiperiodic_time)
      : grid_(grid), clover_(clover), volume_(local_volume(grid)),
        gauge_param_(make_gauge_param(grid, precise, sloppy, precise_reconstruct,
                                      sloppy_reconstruct, antiperiodic_time)),
        invert_param_(make_invert_param(clover, mass, csw, precise, sloppy, tolerance, maximum_iterations)),
        full_input_(24 * volume_), full_output_(24 * volume_), rb_input_(12 * volume_), rb_output_(12 * volume_),
        gauge_pack_seconds_(0.0), gauge_upload_seconds_(0.0), clover_setup_seconds_(0.0)
  {
    accelerator_barrier();
    grid_->Barrier();
    double start = usecond();
    gauge_to_lexicographic(gauge, gauge_buffers_);
    if (antiperiodic_time) apply_antiperiodic_time_boundary(gauge_buffers_, grid_);
    std::array<std::vector<double>, 4> gauge_even_odd;
    std::array<void *, 4> gauge_pointers;
    const Coordinate local = grid_->LocalDimensions();
    for (int direction = 0; direction < 4; ++direction) {
      gauge_even_odd[direction].resize(18 * volume_);
      lexicographic_to_even_odd(gauge_buffers_[direction].data(), gauge_even_odd[direction].data(), volume_, 18,
                                local);
      gauge_pointers[direction] = gauge_even_odd[direction].data();
    }
    accelerator_barrier();
    grid_->Barrier();
    gauge_pack_seconds_ = (usecond() - start) / 1.0e6;
    grid_->GlobalMax(gauge_pack_seconds_);

    start = usecond();
    loadGaugeQuda(gauge_pointers.data(), &gauge_param_);
    accelerator_barrier();
    grid_->Barrier();
    gauge_upload_seconds_ = (usecond() - start) / 1.0e6;
    grid_->GlobalMax(gauge_upload_seconds_);

    if (clover_) {
      start = usecond();
      loadCloverQuda(nullptr, nullptr, &invert_param_);
      accelerator_barrier();
      grid_->Barrier();
      clover_setup_seconds_ = (usecond() - start) / 1.0e6;
      grid_->GlobalMax(clover_setup_seconds_);
    }
  }

  QudaOperator(const QudaOperator &) = delete;
  QudaOperator &operator=(const QudaOperator &) = delete;

  ~QudaOperator()
  {
    // Order matters: the MG hierarchy holds references to the resident gauge and
    // clover fields, so it must be torn down before they are freed.
    destroy_multigrid();
    if (clover_) freeCloverQuda();
    freeGaugeQuda();
  }

  // dslashQuda always constructs a *preconditioned* Dirac object (it hardcodes
  // pc=true; see quda/lib/interface_quda.cpp), so the operation it performs is
  // action-dependent and the shared "pc_dslash" label means:
  //
  //   Wilson:  DiracWilsonPC::Dslash = DiracWilson::Dslash
  //              = ApplyWilson(..., kappa=0, ...)     -> raw hopping sum D_quda
  //   Clover:  DiracCloverPC::Dslash
  //              = ApplyWilsonCloverPreconditioned(..., kappa=0, ...)
  //                                                   -> A_quda^{-1} D_quda
  //
  // Neither path rescales for mass_normalization: dslashQuda's blas::ax branch
  // fires only for staggered/asqtad dslash types. So the raw output is in
  // QUDA's own convention and must be converted to Grid's mass-normalized one.
  //
  // Grid imports the hopping links pre-scaled by -1/2
  // (WilsonFermionImplementation.h: `HUmu = _Umu * (-0.5)`), while QUDA's
  // D_quda is the bare sum, hence
  //
  //   D_grid = -0.5 * D_quda                                        ... (1)
  //
  // and Grid's clover diagonal carries diag_mass = 4+mass with no kappa in it
  // (WilsonCloverFermionImplementation.h), whereas QUDA's resident clover field
  // is the kappa-normalized A_quda = 1 + csw*kappa*sigma.F, i.e.
  //
  //   A_quda = 2*kappa*A_grid                                       ... (2)
  //
  // (1) and (2) are exactly what makes M_quda = 2*kappa*M_grid hold for the
  // full operator, which is the relation apply_mat() below relies on.
  // Combining them for the clover preconditioned Dslash:
  //
  //   A_grid^{-1} D_grid = (2*kappa*A_quda^{-1}) * (-0.5*D_quda)
  //                      = -kappa * A_quda^{-1} D_quda
  //
  // Hence the conversion factor applied here is -0.5 for Wilson and -kappa for
  // clover. Both are source-derived, but the runtime correctness gate in
  // measure_op() is still the authority -- no timing is collected for an
  // operation whose two sides disagree.
  template <class FermionField> void apply_pc_dslash(const FermionField &input, FermionField &output,
                                                     QudaParity output_parity)
  {
    fermion_rb_to_buffer(input, rb_input_.data());
    dslashQuda(rb_output_.data(), rb_input_.data(), &invert_param_, output_parity);
    const double scale = clover_ ? -invert_param_.kappa : -0.5;
    for (double &value : rb_output_) value *= scale;
    buffer_to_fermion_rb(rb_output_.data(), output);
  }

  // cloverQuda applies QUDA's resident clover field with NO mass_normalization
  // rescaling of any kind (see quda/lib/interface_quda.cpp's cloverQuda: it has
  // no blas::ax branch at all, unlike dslashQuda/MatQuda/MatDagMatQuda). Under
  // QUDA_KAPPA_NORMALIZATION that resident field is the kappa-normalized clover
  // term A_quda = 1 + clover_coeff*sigma.F with clover_coeff = csw*kappa, i.e.
  // A_quda = 2*kappa*A_grid exactly like the M/Mpc relation documented at the
  // top of this file (Grid's diag_mass = 4+mass has no kappa baked in either --
  // see WilsonCloverHelpers.h's ModifyBoundaries: diag = diag_mass -+ csw_t).
  // So the non-inverse clover multiply must be divided by 2*kappa to land in
  // Grid's mass-normalized convention, and the clover inverse -- being the
  // inverse of that same rescaled matrix -- must be multiplied by 2*kappa.
  //
  // NOTE: only inverse=true is reachable. QUDA's ApplyClover
  // (quda/lib/dslash_clover_helper.cu:25) begins with an unconditional
  //     if (!inverse) errorQuda("Unsupported direct application");
  // so cloverQuda(..., inverse=0) calls errorQuda -> MPI_Abort and kills the
  // whole job. Throwing here instead converts that into a normal C++ error
  // that main()'s catch block reports cleanly, rather than losing the run and
  // every record not yet flushed. Confirmed on smoke job 57603507.
  template <class FermionField> void apply_clover(const FermionField &input, FermionField &output, QudaParity parity, bool inverse)
  {
    if (!clover_) throw std::logic_error("clover operation requested for Wilson operator");
    if (!inverse)
      throw std::runtime_error(
          "QUDA has no standalone direct clover application (ApplyClover in "
          "dslash_clover_helper.cu rejects inverse=false); compare the direct clover term "
          "through the fused mat/normal_pc cases instead");
    fermion_rb_to_buffer(input, rb_input_.data());
    cloverQuda(rb_output_.data(), rb_input_.data(), &invert_param_, parity, inverse ? 1 : 0);
    const double two_kappa = 2.0 * invert_param_.kappa;
    const double scale = inverse ? two_kappa : (1.0 / two_kappa);
    for (double &value : rb_output_) value *= scale;
    buffer_to_fermion_rb(rb_output_.data(), output);
  }

  // MatQuda under QUDA_KAPPA_NORMALIZATION applies no mass_normalization
  // rescaling either (that branch in interface_quda.cpp only fires for
  // QUDA_MASS_NORMALIZATION/QUDA_ASYMMETRIC_MASS_NORMALIZATION), so its raw
  // output is QUDA's kappa-normalized M = 2*kappa*M_grid -- the same relation
  // documented for Mpc at the top of this file, one power of (2 kappa) instead
  // of two. Undo it here so this comparison lands in Grid's mass-normalized
  // convention, matching apply_normal()/solve() below.
  template <class FermionField> void apply_mat(const FermionField &input, FermionField &output)
  {
    fermion_to_even_odd(input, full_input_.data());
    const QudaSolutionType saved_solution_type = invert_param_.solution_type;
    const QudaSolveType saved_solve_type = invert_param_.solve_type;
    invert_param_.solution_type = QUDA_MAT_SOLUTION;
    invert_param_.solve_type = QUDA_DIRECT_SOLVE;
    MatQuda(full_output_.data(), full_input_.data(), &invert_param_);
    invert_param_.solution_type = saved_solution_type;
    invert_param_.solve_type = saved_solve_type;
    const double inverse_two_kappa = 1.0 / (2.0 * invert_param_.kappa);
    for (double &value : full_output_) value *= inverse_two_kappa;
    even_odd_to_fermion(full_output_.data(), output);
  }

  template <class FermionField> void apply_normal(const FermionField &input, FermionField &output)
  {
    fermion_rb_to_buffer(input, rb_input_.data());
    MatDagMatQuda(rb_output_.data(), rb_input_.data(), &invert_param_);
    const double inverse_four_kappa_squared = 1.0 / (4.0 * invert_param_.kappa * invert_param_.kappa);
    for (double &value : rb_output_) value *= inverse_four_kappa_squared;
    buffer_to_fermion_rb(rb_output_.data(), output);
  }

  template <class FermionField> void solve(const FermionField &source, FermionField &solution)
  {
    fermion_rb_to_buffer(source, rb_input_.data());
    // (2*kappa)^2 for the asymmetric normal operator; 1 for the symmetric one --
    // see matpc_is_symmetric(). The v3 CG rows are asymmetric, so they take the
    // 4*kappa^2 branch exactly as before and are bit-for-bit unaffected.
    const double four_kappa_squared =
        matpc_is_symmetric() ? 1.0 : (4.0 * invert_param_.kappa * invert_param_.kappa);
    for (double &value : rb_input_) value *= four_kappa_squared;
    std::fill(rb_output_.begin(), rb_output_.end(), 0.0);
    invertQuda(rb_output_.data(), rb_input_.data(), &invert_param_);
    buffer_to_fermion_rb(rb_output_.data(), solution);
  }

  // Batched multi-RHS solve: N sources in ONE invertMultiSrcQuda call.
  //
  // SAME SOLVER BODY as solve() above. invertQuda is itself a one-element
  // solve({hp_x}, {hp_b}, param, gauge) and invertMultiSrcQuda is
  // solve(_x, _b, param, gauge) (quda/lib/interface_quda.cpp); the batch size is
  // taken from the pointer-vector length inside solve(), so the ONLY difference
  // between the two routes is how many sources are handed over. That is what
  // makes the batched-vs-batched comparison in run_multirhs() an apples-to-apples
  // one rather than an approximation.
  //
  // Our configuration reaches no multi-RHS guard: clover dslash_type,
  // QUDA_NORMOP_PC_SOLVE, QUDA_MATPCDAG_MATPC_SOLUTION,
  // QUDA_MATPC_ODD_ODD_ASYMMETRIC and plain CG are all natively vectorised over
  // right-hand sides (DiracCloverPC's Dslash/DslashXpay/M all take cvector_ref,
  // and CG::operator() carries per-RHS beta/pAp/sigma/r2). Every errorQuda on
  // this path lives in the split-grid branch, which the unsplit configuration
  // below does not take.
  //
  // UNSPLIT DELIBERATELY (split_grid = {1,1,1,1}, num_src_per_sub_partition ==
  // num_src). Unsplit means all ranks work on all N sources in one cvector_ref
  // solve, so any gain comes from amortising the gauge and clover READS across
  // the batch. That is exactly the mechanism Grid's 5D path uses, hence the
  // like-for-like instrument. Split-grid (prod(split_grid) > 1) instead
  // partitions the MPI communicator and solves different sources at LOWER
  // parallelism, and replicates the gauge field per sub-partition -- a different
  // lever, at different parallelism, with different memory ceilings, so it is
  // not a backend ratio and is deliberately not offered here.
  //
  // ⛔ LAYOUT TRAP, and why the buffers are ONE allocation. QUDA builds a single
  // ColorSpinorParam from _hp_b[0] and then merely swaps the `v` pointer for each
  // subsequent source (quda/lib/solve.cpp), while checkInvertParam validates
  // against _hp_b[0] alone (quda/lib/interface_quda.cpp). A source whose layout
  // differed from source 0's would therefore be SILENTLY MISREAD rather than
  // rejected -- it would produce a plausible-looking wrong answer. Deriving all N
  // pointers as uniform strides into one contiguous vector makes an inconsistent
  // layout unrepresentable rather than merely unlikely.
  template <class FermionField>
  void solve_batched(const std::vector<FermionField> &sources, std::vector<FermionField> &solutions)
  {
    const int n = static_cast<int>(sources.size());
    if (n < 1) throw std::runtime_error("solve_batched: at least one source is required");
    if (static_cast<int>(solutions.size()) != n)
      throw std::runtime_error("solve_batched: solution count does not match source count");
    // Hard cap in QUDA (errorQuda in check_params.h), so catching it here turns an
    // MPI_Abort that loses the whole run into a normal C++ error main() reports.
    // Note QUDA_MAX_MULTI_RHS is a different, SOFT per-launch limit: exceeding it
    // recursively bisects the batch rather than failing.
    if (n > QUDA_MAX_MULTI_SRC)
      throw std::runtime_error("solve_batched: num_src " + std::to_string(n) +
                               " exceeds QUDA_MAX_MULTI_SRC " + std::to_string(QUDA_MAX_MULTI_SRC));

    const std::size_t stride = 12ULL * static_cast<std::size_t>(volume_);
    // One contiguous allocation each -- see the layout trap above. resize() is a
    // no-op on every call after the first at a given n, so once the harness has
    // warmed up nothing is allocated inside a timed region.
    batch_input_.resize(stride * static_cast<std::size_t>(n));
    batch_output_.resize(stride * static_cast<std::size_t>(n));
    std::vector<void *> source_pointers(n);
    std::vector<void *> solution_pointers(n);

    // The same per-RHS 4*kappa^2 source scaling the single-RHS solve() applies,
    // for the same reason: it converts Grid's mass-normalized Mpc^dag Mpc
    // right-hand side into QUDA's kappa-normalized one. Applying it to only some
    // slices would yield a solution that looks converged on QUDA's own residual
    // yet fails the cross-backend gate, which is why that gate runs before any
    // batched timing is trusted.
    const double four_kappa_squared = 4.0 * invert_param_.kappa * invert_param_.kappa;
    for (int s = 0; s < n; ++s) {
      double *slice = batch_input_.data() + stride * static_cast<std::size_t>(s);
      fermion_rb_to_buffer(sources[s], slice);
      for (std::size_t index = 0; index < stride; ++index) slice[index] *= four_kappa_squared;
      source_pointers[s] = slice;
      solution_pointers[s] = batch_output_.data() + stride * static_cast<std::size_t>(s);
    }
    std::fill(batch_output_.begin(), batch_output_.end(), 0.0);

    // num_src is a per-call property and the harness drives single-RHS
    // invertQuda solves through this SAME operator, so save and restore it.
    // solve() takes its batch size from the pointer-vector length rather than
    // from num_src, so a leak would be benign today; restoring keeps it that way
    // if that ever changes, and keeps the parameter block describing the call it
    // is actually about to make.
    const int saved_num_src = invert_param_.num_src;
    const int saved_num_src_per_sub_partition = invert_param_.num_src_per_sub_partition;
    int saved_split_grid[4];
    for (int dimension = 0; dimension < 4; ++dimension)
      saved_split_grid[dimension] = invert_param_.split_grid[dimension];

    invert_param_.num_src = n;
    invert_param_.num_src_per_sub_partition = n;
    for (int dimension = 0; dimension < 4; ++dimension) invert_param_.split_grid[dimension] = 1;

    invertMultiSrcQuda(solution_pointers.data(), source_pointers.data(), &invert_param_);

    invert_param_.num_src = saved_num_src;
    invert_param_.num_src_per_sub_partition = saved_num_src_per_sub_partition;
    for (int dimension = 0; dimension < 4; ++dimension)
      invert_param_.split_grid[dimension] = saved_split_grid[dimension];

    for (int s = 0; s < n; ++s)
      buffer_to_fermion_rb(batch_output_.data() + stride * static_cast<std::size_t>(s), solutions[s]);
  }

  // Per-source relative residuals from the most recent solve, as QUDA computed
  // them. compute_true_res defaults to 1 and CG fills every entry of
  // true_res[0..n-1] (quda/lib/inv_cg_quda.cpp), so a batched solve reports one
  // residual per source rather than a single aggregate.
  //
  // These are QUDA's OWN numbers and carry its ~1e-9 run-to-run autotuning
  // floor, so they are recorded for diagnosis and are NOT what a correctness
  // gate is graded on -- gates use independently evaluated Grid-side residuals.
  std::vector<double> true_residuals(int count) const
  {
    if (count < 0 || count > QUDA_MAX_MULTI_SRC)
      throw std::runtime_error("true_residuals: count out of range");
    return std::vector<double>(invert_param_.true_res, invert_param_.true_res + count);
  }

  // Iterations taken by the most recent solve. For a BATCHED solve this is the
  // worst case over sources, not the mean: Solver::convergenceL2 returns false
  // if any right-hand side is unconverged (quda/lib/solver.cpp), so the batch
  // iterates until the slowest source converges. Grid's 5D path shares that
  // property, which is what makes a per-iteration comparison between the two
  // batched routes meaningful at all.
  long long last_iterations() const { return static_cast<long long>(invert_param_.iter); }

  // ---- Multigrid -------------------------------------------------------
  //
  // Build AFTER the constructor has run loadGaugeQuda + loadCloverQuda: QUDA's MG
  // setup coarsens the resident gauge/clover fields, so a build before they exist
  // dereferences nothing useful.
  //
  // The setup time returned here is reported as its OWN column, never folded into
  // a time-to-solution. In HMC the cost is amortised over many solves under a
  // rebuild cadence; on a fixed gauge field it is one setup and N solves, so a
  // combined number would answer neither question.
  void build_multigrid(const QudaMgParams &mg)
  {
    if (mg_preconditioner_ != nullptr) throw std::logic_error("multigrid already built");
    // Wilson is supported: quda/tests/invert_test.cpp:446 lists QUDA_WILSON_DSLASH
    // among the MG-capable dslash types. An earlier revision of this method
    // rejected it, which was simply wrong.
    mg_params_ = mg;

    // Resolve / validate the preconditioner precision against the gauge copies
    // that actually exist on the GPU (see QudaMgParams::precondition_prec). A
    // mismatch here does not raise a QUDA error -- it yields a garbage volume
    // deep inside the MG setup -- so it is checked up front instead.
    const QudaPrecision loaded_precondition_prec = gauge_param_.cuda_prec_precondition;
    if (mg_params_.precondition_prec == QUDA_INVALID_PRECISION) {
      mg_params_.precondition_prec = loaded_precondition_prec;
    } else if (mg_params_.precondition_prec != loaded_precondition_prec) {
      throw std::runtime_error(
          "multigrid precondition_prec does not match the loaded gauge precondition precision; "
          "the gauge field is only resident at `precise` and `sloppy`, so any other value "
          "references an unloaded copy and produces a garbage volume rather than an error");
    }
    // Built from mg_params_, not the caller's `mg`, so the resolved
    // precondition_prec above is the one that reaches QUDA.
    mg_invert_param_ = make_mg_inner_invert_param(invert_param_, mg_params_);
    mg_param_ = make_multigrid_param(mg_params_);
    mg_param_.invert_param = &mg_invert_param_;

    accelerator_barrier();
    grid_->Barrier();
    const double start = usecond();
    mg_preconditioner_ = newMultigridQuda(&mg_param_);
    accelerator_barrier();
    grid_->Barrier();
    mg_setup_seconds_ = (usecond() - start) / 1.0e6;
    grid_->GlobalMax(mg_setup_seconds_);
  }

  void destroy_multigrid()
  {
    if (mg_preconditioner_ == nullptr) return;
    destroyMultigridQuda(mg_preconditioner_);
    mg_preconditioner_ = nullptr;
    invert_param_.preconditioner = nullptr;
  }

  bool has_multigrid() const { return mg_preconditioner_ != nullptr; }
  double mg_setup_seconds() const { return mg_setup_seconds_; }
  const QudaMgParams &mg_params() const { return mg_params_; }

  // Single-parity MG solve of Mpc x = b on the checkerboard selected by
  // matpc_type -- NOT the normal operator solve()/apply_normal() use.
  //
  // Scaling: Mpc_quda = 2*kappa*Mpc_grid, so solving Mpc_quda x = 2*kappa*b_grid
  // gives x = Mpc_grid^-1 b_grid. One factor of 2*kappa, against solve()'s
  // 4*kappa^2 for the squared operator -- same derivation, one power lower.
  //
  // The outer solver fields are saved and restored around the call so the CG rows
  // in the same job are bit-for-bit unaffected. That matters: those rows are the
  // in-run gate that the job reproduces v3.
  template <class FermionField> void solve_mg(const FermionField &source, FermionField &solution)
  {
    if (mg_preconditioner_ == nullptr) throw std::logic_error("solve_mg called before build_multigrid");

    const QudaInverterType saved_inv_type = invert_param_.inv_type;
    const QudaInverterType saved_precon = invert_param_.inv_type_precondition;
    const QudaSolveType saved_solve_type = invert_param_.solve_type;
    const QudaSolutionType saved_solution_type = invert_param_.solution_type;
    const QudaSchwarzType saved_schwarz = invert_param_.schwarz_type;
    void *saved_preconditioner = invert_param_.preconditioner;
    const int saved_precondition_cycle = invert_param_.precondition_cycle;
    const double saved_tol_precondition = invert_param_.tol_precondition;
    const int saved_maxiter_precondition = invert_param_.maxiter_precondition;
    const int saved_gcr_nkrylov = invert_param_.gcrNkrylov;

    // Config: <OuterGCRNKrylov>20</OuterGCRNKrylov>. The inner PrecondGCRNKrylov
    // of 10 lives on the MG's own invert param.
    invert_param_.gcrNkrylov = mg_params_.outer_gcr_nkrylov;
    invert_param_.inv_type = QUDA_GCR_INVERTER;
    invert_param_.inv_type_precondition = QUDA_MG_INVERTER;
    invert_param_.solve_type = QUDA_DIRECT_PC_SOLVE;
    invert_param_.solution_type = QUDA_MATPC_SOLUTION;
    invert_param_.schwarz_type = QUDA_INVALID_SCHWARZ;
    invert_param_.precondition_cycle = 1;
    invert_param_.tol_precondition = 1e-1;
    invert_param_.maxiter_precondition = 1;
    invert_param_.preconditioner = mg_preconditioner_;

    fermion_rb_to_buffer(source, rb_input_.data());
    // 1 for symmetric, 2*kappa for asymmetric -- see matpc_is_symmetric(). MG
    // requires symmetric, so in practice this is 1; the asymmetric branch exists
    // so the method stays correct if it is ever called on a non-MG symmetric path.
    const double scale = matpc_is_symmetric() ? 1.0 : (2.0 * invert_param_.kappa);
    if (scale != 1.0) {
      for (double &value : rb_input_) value *= scale;
    }
    std::fill(rb_output_.begin(), rb_output_.end(), 0.0);
    invertQuda(rb_output_.data(), rb_input_.data(), &invert_param_);
    buffer_to_fermion_rb(rb_output_.data(), solution);

    mg_last_iterations_ = invert_param_.iter;
    mg_last_internal_seconds_ = invert_param_.secs;
    mg_last_true_residual_ = invert_param_.true_res[0];

    invert_param_.inv_type = saved_inv_type;
    invert_param_.inv_type_precondition = saved_precon;
    invert_param_.solve_type = saved_solve_type;
    invert_param_.solution_type = saved_solution_type;
    invert_param_.schwarz_type = saved_schwarz;
    invert_param_.precondition_cycle = saved_precondition_cycle;
    invert_param_.tol_precondition = saved_tol_precondition;
    invert_param_.maxiter_precondition = saved_maxiter_precondition;
    invert_param_.preconditioner = saved_preconditioner;
    invert_param_.gcrNkrylov = saved_gcr_nkrylov;
  }

  int mg_last_iterations() const { return mg_last_iterations_; }
  double mg_last_internal_seconds() const { return mg_last_internal_seconds_; }
  double mg_last_true_residual() const { return mg_last_true_residual_; }

  // ⛔ THE GRID<->QUDA SCALE FACTOR DEPENDS ON matpc_type, AND IS 1 FOR SYMMETRIC.
  //
  // Grid pre-scales the hopping links by -1/2 and its clover diagonal carries no
  // kappa, while QUDA's resident clover is A_quda = 2*kappa*A_grid. Working both
  // conventions through the two Schur forms:
  //
  //   ASYMMETRIC  Grid Moo - Moe Mee^-1 Meo
  //                 = (1/2kappa) [ A_oo - kappa^2 D_oe A_ee^-1 D_eo ]
  //               => Mpc_quda = 2*kappa * Mpc_grid          (scale 2*kappa)
  //
  //   SYMMETRIC   Grid 1 - Moo^-1 Moe Mee^-1 Meo
  //                 = 1 - kappa^2 A_oo^-1 D_oe A_ee^-1 D_eo
  //               => Mpc_quda = Mpc_grid                    (scale 1)
  //
  // The (2*kappa)^-1 prefactor of the asymmetric form is exactly cancelled by the
  // two A^-1 factors the symmetric form introduces. Applying the asymmetric
  // correction to a symmetric solve returns a solution scaled by 2*kappa (or
  // 4*kappa^2 for the normal operator) -- a perfectly converged answer to the
  // wrong system, which QUDA's own residual reports as fine. Measured exactly that
  // way on the first symmetric run: Grid-side residuals 0.7337 = |1 - 2*kappa| and
  // 0.9291 = |1 - 4*kappa^2|.
  bool matpc_is_symmetric() const
  {
    return invert_param_.matpc_type == QUDA_MATPC_EVEN_EVEN
           || invert_param_.matpc_type == QUDA_MATPC_ODD_ODD;
  }

  const QudaInvertParam &invert_param() const { return invert_param_; }
  QudaInvertParam &invert_param() { return invert_param_; }
  const QudaGaugeParam &gauge_param() const { return gauge_param_; }
  double gauge_pack_seconds() const { return gauge_pack_seconds_; }
  double gauge_upload_seconds() const { return gauge_upload_seconds_; }
  double clover_setup_seconds() const { return clover_setup_seconds_; }
  std::array<double, 3> plaquette() const
  {
    std::array<double, 3> value;
    plaqQuda(value.data());
    return value;
  }

private:
  GridBase *grid_;
  bool clover_;
  int volume_;
  QudaGaugeParam gauge_param_;
  QudaInvertParam invert_param_;
  std::array<std::vector<double>, 4> gauge_buffers_;
  std::vector<double> full_input_;
  std::vector<double> full_output_;
  std::vector<double> rb_input_;
  std::vector<double> rb_output_;
  // Multi-RHS staging, sized on first use by solve_batched() and left empty
  // otherwise, so a single-RHS run allocates nothing extra and every archived
  // single-RHS measurement keeps its memory profile. One contiguous allocation
  // each, deliberately -- see the layout trap in solve_batched().
  std::vector<double> batch_input_;
  std::vector<double> batch_output_;
  double gauge_pack_seconds_;
  double gauge_upload_seconds_;
  double clover_setup_seconds_;
  // Multigrid state. mg_param_ holds a pointer to mg_invert_param_, so both must
  // outlive the preconditioner and neither may be copied -- QudaOperator is
  // non-copyable, which is what keeps that pointer valid.
  void *mg_preconditioner_ = nullptr;
  QudaMgParams mg_params_;
  QudaMultigridParam mg_param_;
  QudaInvertParam mg_invert_param_;
  double mg_setup_seconds_ = 0.0;
  int mg_last_iterations_ = 0;
  double mg_last_internal_seconds_ = 0.0;
  double mg_last_true_residual_ = 0.0;
};

} // namespace grid_quda_benchmark
