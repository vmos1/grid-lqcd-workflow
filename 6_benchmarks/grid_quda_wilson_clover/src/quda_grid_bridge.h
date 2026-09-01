#pragma once

#include <Grid/Grid.h>
#include <quda.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
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
    const double four_kappa_squared = 4.0 * invert_param_.kappa * invert_param_.kappa;
    for (double &value : rb_input_) value *= four_kappa_squared;
    std::fill(rb_output_.begin(), rb_output_.end(), 0.0);
    invertQuda(rb_output_.data(), rb_input_.data(), &invert_param_);
    buffer_to_fermion_rb(rb_output_.data(), solution);
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
  double gauge_pack_seconds_;
  double gauge_upload_seconds_;
  double clover_setup_seconds_;
};

} // namespace grid_quda_benchmark
