// benchmark_grid_quda_wilson_clover.cc
//
// Fixed-work throughput and matched-CG-solve comparison between stock Grid's
// Wilson / Wilson-clover fermion operator and stock QUDA's equivalent,
// built against the benchmark-local bridge in quda_grid_bridge.h.
//
// Scope note: Grid's own arithmetic is kept in double precision throughout.
// --benchmark-precision only changes QUDA's sloppy precision/reconstruction:
//   strict     -> precise/sloppy double, precise/sloppy reconstruct = NO
//   production -> precise double/reconstruct-NO, sloppy single/reconstruct-12
// This isolates "does QUDA's kernel beat Grid's even head-to-head in double"
// (strict) from "how much further does QUDA's normal production mixed
// precision path pull ahead" (production). It does not exercise a
// Grid-side mixed-precision solver -- that is a distinct question.
//
// Convention notes (see quda_grid_bridge.h for the load-bearing details):
//   - Both sides solve the ODD-checkerboard normal-equation Schur system
//     directly: given an odd-checkerboard right-hand side b, find x with
//     M_pc^dag M_pc x = b, where M_pc is Grid's SchurDiagMooeeOperator
//     (Moo - Moe Mee^-1 Meo) and QUDA's matching preconditioned operator. The
//     QUDA matpc enum is action-dependent because a Wilson diagonal block is a
//     scalar and QUDA rejects the asymmetric enum for it:
//         wilson -> QUDA_MATPC_ODD_ODD
//         clover -> QUDA_MATPC_ODD_ODD_ASYMMETRIC
//     Both give Mpc_quda = 2*kappa*Mpc_grid. No even-reconstruction step is
//     exercised on either side; that step is identical bookkeeping on both
//     backends and not part of what this benchmark is isolating.
//   - QUDA's MatQuda/MatDagMatQuda/cloverQuda(non-inverse) all return output
//     in QUDA's kappa-normalized convention (2*kappa times Grid's mass-
//     normalized convention; 4*kappa^2 for the squared normal operator; the
//     inverse of 2*kappa for a clover inverse). The bridge's apply_mat(),
//     apply_clover(), apply_normal(), and solve() all undo this internally,
//     so this file passes/consumes plain Grid-convention (mass-normalized)
//     fields with no extra scaling.
//   - Preconditioned-Dslash parity direction: an Even input produces an Odd
//     output. dslashQuda always builds a preconditioned Dirac object, so the
//     "pc_dslash" case is deliberately action-dependent -- it is raw
//     checkerboard D for Wilson and A_oo^-1 D_oe for clover. Grid mirrors that
//     exactly (DhopOE alone, versus MooeeInv composed after DhopOE), and the
//     bridge converts QUDA's output with a factor of -0.5 (Wilson) or -kappa
//     (clover). The correctness gate terminates the run before timing if these
//     conventions are not equivalent.
//
// Correctness is a hard gate: validate each paired operation before collecting
// any timing for it. A mismatch is recorded in the JSONL output and terminates
// the run, so an invalid convention can never produce publishable timings.

#include <Grid/Grid.h>
#include "quda_grid_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <memory>
#include <vector>

using namespace Grid;
using namespace grid_quda_benchmark;

namespace {

// ---------------------------------------------------------------------
// CLI parsing (same idiom as tests/core/Test_compact_wilson_clover_speedup.cc)
// ---------------------------------------------------------------------

bool flag_present(int argc, char **argv, const std::string &option)
{
  return GridCmdOptionExists(argv, argv + argc, option);
}

std::string flag_payload(int argc, char **argv, const std::string &option)
{
  return GridCmdOptionPayload(argv, argv + argc, option);
}

std::string read_string(int argc, char **argv, const std::string &option, const std::string &fallback)
{
  return flag_present(argc, argv, option) ? flag_payload(argc, argv, option) : fallback;
}

int read_int(int argc, char **argv, const std::string &option, int fallback)
{
  if (!flag_present(argc, argv, option)) return fallback;
  std::string arg = flag_payload(argc, argv, option);
  int value = fallback;
  GridCmdOptionInt(arg, value);
  return value;
}

double read_double(int argc, char **argv, const std::string &option, double fallback)
{
  if (!flag_present(argc, argv, option)) return fallback;
  std::string arg = flag_payload(argc, argv, option);
  double value = fallback;
  GridCmdOptionFloat(arg, value);
  return value;
}

struct BenchmarkOptions {
  std::string action = "both";       // wilson | clover | both
  std::string input = "hot";         // hot | physical
  double mass = -0.2416;
  double csw = 1.20536588031793;
  std::string precision = "strict";  // strict | production
  // Independent QUDA knobs. `precision` is only a PRESET for these two; when
  // either is given explicitly it wins. Bundling them (as strict/production
  // does) makes it impossible to see which knob bought which speedup, which is
  // the whole point of the precision/reconstruct scan.
  //   sloppy_precision:   double | single | half     ("" = take from preset)
  //   sloppy_reconstruct: no     | 12     | 8        ("" = take from preset)
  std::string sloppy_precision;
  std::string sloppy_reconstruct;
  // Which Grid solver to time in addition to plain double CG.
  //   double = double CG only
  //   mixed  = also run Grid's MixedPrecisionConjugateGradient (double outer,
  //            single inner) -- the same scheme QUDA's sloppy precision uses,
  //            so it measures how much of QUDA's advantage Grid can recover.
  std::string grid_solver = "double";  // double | mixed
  int samples = 7;
  int warmups = 2;
  int repetitions = 20;
  int solve_repeats = 5;
  double tol = 1e-10;
  int maxiter = 50000;
  // Largest relative iteration-count span (across backends and repeats) that
  // still permits attributing the solve-time ratio to operator throughput.
  double iteration_tolerance = 0.02;
  std::string output;
  std::string cache_state = "unknown";
  std::string cfg;
};

[[noreturn]] void fail(const std::string &message)
{
  throw std::runtime_error(message);
}

BenchmarkOptions parse_options(int argc, char **argv)
{
  BenchmarkOptions o;
  o.action        = read_string(argc, argv, "--benchmark-action", o.action);
  o.input         = read_string(argc, argv, "--benchmark-input", o.input);
  o.mass          = read_double(argc, argv, "--benchmark-mass", o.mass);
  o.csw           = read_double(argc, argv, "--benchmark-csw", o.csw);
  o.precision     = read_string(argc, argv, "--benchmark-precision", o.precision);
  o.sloppy_precision   = read_string(argc, argv, "--benchmark-sloppy-precision", o.sloppy_precision);
  o.sloppy_reconstruct = read_string(argc, argv, "--benchmark-sloppy-reconstruct", o.sloppy_reconstruct);
  o.grid_solver        = read_string(argc, argv, "--benchmark-grid-solver", o.grid_solver);
  o.samples       = read_int(argc, argv, "--benchmark-samples", o.samples);
  o.warmups       = read_int(argc, argv, "--benchmark-warmups", o.warmups);
  o.repetitions   = read_int(argc, argv, "--benchmark-repetitions", o.repetitions);
  o.solve_repeats = read_int(argc, argv, "--benchmark-solve-repeats", o.solve_repeats);
  o.tol           = read_double(argc, argv, "--benchmark-tol", o.tol);
  o.maxiter       = read_int(argc, argv, "--benchmark-maxiter", o.maxiter);
  o.iteration_tolerance =
      read_double(argc, argv, "--benchmark-iteration-tolerance", o.iteration_tolerance);
  o.output        = read_string(argc, argv, "--benchmark-output", o.output);
  o.cache_state   = read_string(argc, argv, "--benchmark-cache-state", o.cache_state);
  o.cfg           = read_string(argc, argv, "--benchmark-cfg", o.cfg);

  if (o.action != "wilson" && o.action != "clover" && o.action != "both")
    fail("--benchmark-action must be wilson|clover|both");
  if (o.input != "hot" && o.input != "physical")
    fail("--benchmark-input must be hot|physical");
  if (o.precision != "strict" && o.precision != "production")
    fail("--benchmark-precision must be strict|production");
  // Resolve the preset into explicit knobs, then let explicit flags override.
  // After this block both fields are always concrete, so the record written to
  // JSONL states exactly what ran rather than a preset name that has to be
  // decoded later.
  if (o.sloppy_precision.empty())
    o.sloppy_precision = (o.precision == "strict") ? "double" : "single";
  if (o.sloppy_reconstruct.empty())
    o.sloppy_reconstruct = (o.precision == "strict") ? "no" : "12";
  if (o.sloppy_precision != "double" && o.sloppy_precision != "single" && o.sloppy_precision != "half")
    fail("--benchmark-sloppy-precision must be double|single|half");
  if (o.sloppy_reconstruct != "no" && o.sloppy_reconstruct != "12" && o.sloppy_reconstruct != "8")
    fail("--benchmark-sloppy-reconstruct must be no|12|8");
  if (o.grid_solver != "double" && o.grid_solver != "mixed")
    fail("--benchmark-grid-solver must be double|mixed");
  if (o.output.empty())
    fail("--benchmark-output is required");
  if (o.input == "physical" && o.cfg.empty())
    fail("--benchmark-cfg is required when --benchmark-input=physical");
  if (o.samples < 1 || o.repetitions < 1 || o.warmups < 0 || o.solve_repeats < 1)
    fail("--benchmark-samples/--benchmark-repetitions/--benchmark-solve-repeats must be positive "
         "(--benchmark-warmups may be zero)");
  if (o.iteration_tolerance < 0.0)
    fail("--benchmark-iteration-tolerance must not be negative");
  return o;
}

// ---------------------------------------------------------------------
// Statistics + JSONL emission
// ---------------------------------------------------------------------

struct Stat {
  double mean = 0.0;
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
  double stddev = 0.0;
};

Stat summarize(const std::vector<double> &values)
{
  Stat s;
  const double n = static_cast<double>(values.size());
  s.min = *std::min_element(values.begin(), values.end());
  s.max = *std::max_element(values.begin(), values.end());
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
  std::vector<double> sorted(values);
  std::sort(sorted.begin(), sorted.end());
  const std::size_t middle = sorted.size() / 2;
  s.median = (sorted.size() & 1) ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
  double variance = 0.0;
  for (double v : values) variance += (v - s.mean) * (v - s.mean);
  variance = (n > 1.0) ? variance / (n - 1.0) : 0.0;
  s.stddev = std::sqrt(variance);
  return s;
}

// Per-repeat integer summaries. Iteration counts are reported as a median plus
// explicit min/max/constancy so a run whose repeats disagree can never be read
// as a single clean number.
struct IterationStat {
  long long median = 0;
  long long min = 0;
  long long max = 0;
  bool constant = true;
};

IterationStat summarize_iterations(const std::vector<long long> &values)
{
  IterationStat s;
  if (values.empty()) return s;
  std::vector<long long> sorted(values);
  std::sort(sorted.begin(), sorted.end());
  s.min = sorted.front();
  s.max = sorted.back();
  s.median = sorted[sorted.size() / 2];
  s.constant = (s.min == s.max);
  return s;
}

double median_of(const std::vector<double> &values)
{
  if (values.empty()) return 0.0;
  std::vector<double> sorted(values);
  std::sort(sorted.begin(), sorted.end());
  const std::size_t middle = sorted.size() / 2;
  return (sorted.size() & 1) ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
}

double max_of(const std::vector<double> &values)
{
  return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
}

// Every call to a Grid collective (Dhop/M/Mooee/CG/...) and every QUDA call
// (dslashQuda/MatQuda/.../invertQuda) is collective over the same
// communicator, so wall time is measured with a Barrier immediately before
// and after so it reflects the slowest rank, matching benchmarks/Benchmark_wilson.cc.
template <class Fn>
Stat time_operation(GridBase *grid, int warmups, int samples, int repetitions, Fn &&fn)
{
  for (int w = 0; w < warmups; ++w) fn();
  accelerator_barrier();
  std::vector<double> per_call_seconds(samples);
  for (int s = 0; s < samples; ++s) {
    accelerator_barrier();
    grid->Barrier();
    double t0 = usecond();
    for (int r = 0; r < repetitions; ++r) fn();
    accelerator_barrier();
    grid->Barrier();
    double t1 = usecond();
    per_call_seconds[s] = (t1 - t0) / 1.0e6 / static_cast<double>(repetitions);
  }
  return summarize(per_call_seconds);
}

class JsonlWriter {
public:
  // KNOWN multi-rank hazard, deliberately left as-is: this throw fires on the
  // boss rank only. If the output path were unwritable, rank 0 would unwind
  // while the others carried on into the next collective and the job would
  // hang rather than fail cleanly. Making it uniform needs a communicator here
  // to broadcast the failure, which this class does not hold. It is tolerated
  // because the runner mkdir -p's RUN_DIR immediately before launching, so the
  // path is essentially always writable; a filesystem outage is the only
  // trigger. Any NEW rank-conditional throw should be treated as a bug -- see
  // the norm2(Umu) deadlock that cost jobs 57607090 and 57607372.
  JsonlWriter(const std::string &path, bool boss) : boss_(boss)
  {
    if (boss_) {
      out_.open(path, std::ios::out | std::ios::trunc);
      if (!out_.is_open()) fail("could not open --benchmark-output for writing: " + path);
    }
  }

  class Record {
  public:
    Record &add(const std::string &key, const std::string &value)
    {
      sep();
      body_ << "\"" << key << "\":\"" << escape(value) << "\"";
      return *this;
    }
    Record &add(const std::string &key, double value)
    {
      sep();
      body_ << "\"" << key << "\":" << std::setprecision(10) << value;
      return *this;
    }
    Record &add(const std::string &key, long long value)
    {
      sep();
      body_ << "\"" << key << "\":" << value;
      return *this;
    }
    Record &add(const std::string &key, bool value)
    {
      sep();
      body_ << "\"" << key << "\":" << (value ? "true" : "false");
      return *this;
    }
    // Per-repeat arrays: the summarizer must be able to see every measured
    // repeat, not just an aggregate, so a suspicious spread cannot hide.
    Record &add(const std::string &key, const std::vector<double> &values)
    {
      sep();
      body_ << "\"" << key << "\":[";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) body_ << ",";
        body_ << std::setprecision(10) << values[index];
      }
      body_ << "]";
      return *this;
    }
    Record &add(const std::string &key, const std::vector<long long> &values)
    {
      sep();
      body_ << "\"" << key << "\":[";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) body_ << ",";
        body_ << values[index];
      }
      body_ << "]";
      return *this;
    }
    std::string str() const { return "{" + body_.str() + "}"; }

  private:
    void sep()
    {
      if (!first_) body_ << ",";
      first_ = false;
    }
    static std::string escape(const std::string &value)
    {
      std::string out;
      out.reserve(value.size());
      for (char c : value) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
      }
      return out;
    }
    std::ostringstream body_;
    bool first_ = true;
  };

  void emit(const Record &record)
  {
    if (!boss_) return;
    out_ << record.str() << "\n";
    out_.flush();
  }

private:
  bool boss_;
  std::ofstream out_;
};

// ---------------------------------------------------------------------
// One measurement helper shared by every op: time both backends, compare,
// emit two JSONL records (grid/quda) plus a correctness record.
// ---------------------------------------------------------------------

struct MeasureContext {
  JsonlWriter &writer;
  GridBase *grid;
  const BenchmarkOptions &opt;
  std::string action_name;
  bool boss;
};

void add_case_fields(JsonlWriter::Record &record, const BenchmarkOptions &opt,
                     const std::string &action, const std::string &op)
{
  std::ostringstream lattice;
  std::ostringstream mpi;
  Coordinate global = GridDefaultLatt();
  Coordinate processors = GridDefaultMpi();
  for (int d = 0; d < Nd; ++d) {
    if (d) {
      lattice << ".";
      mpi << ".";
    }
    lattice << global[d];
    mpi << processors[d];
  }
  record.add("input", opt.input)
      .add("lattice", lattice.str())
      .add("mpi", mpi.str())
      .add("mass", opt.mass)
      .add("csw", opt.csw)
      .add("action", action)
      .add("op", op)
      .add("precision", opt.precision)
      // The resolved knobs, not just the preset name: a record must say what
      // actually ran. `precision` is retained for backward compatibility with
      // records written before the knobs were separable.
      .add("sloppy_precision", opt.sloppy_precision)
      .add("sloppy_reconstruct", opt.sloppy_reconstruct)
      .add("cache_state", opt.cache_state);
}

template <class GridFn, class QudaFn, class Field>
void measure_op(MeasureContext &ctx, const std::string &op_name,
                GridFn &&grid_fn, QudaFn &&quda_fn,
                Field &grid_reference_output, Field &quda_check_output,
                double tolerance)
{
  // Validate first. Timing an operation whose output does not match would make
  // it too easy to publish a fast but inequivalent convention accidentally.
  grid_fn();
  quda_fn();

  Field diff(grid_reference_output.Grid());
  diff = grid_reference_output - quda_check_output;
  double diff_norm2 = norm2(diff);
  double ref_norm2 = norm2(grid_reference_output);
  double rel_dev = (ref_norm2 > 0.0) ? std::sqrt(diff_norm2 / ref_norm2) : std::sqrt(diff_norm2);
  bool passed = rel_dev <= tolerance;

  JsonlWriter::Record check;
  check.add("kind", std::string("correctness"));
  add_case_fields(check, ctx.opt, ctx.action_name, op_name);
  check.add("rel_deviation", rel_dev)
      .add("tolerance", tolerance)
      .add("passed", passed);
  ctx.writer.emit(check);

  if (ctx.boss) {
    std::cout << GridLogMessage << ctx.action_name << "/" << op_name
              << ": pre-timing relative L2 deviation " << rel_dev << " -> "
              << (passed ? "PASSED" : "FAILED") << std::endl;
  }
  if (!passed) fail(ctx.action_name + "/" + op_name + " correctness gate failed");

  Stat grid_stat = time_operation(ctx.grid, ctx.opt.warmups, ctx.opt.samples, ctx.opt.repetitions,
                                  [&]() { grid_fn(); });
  Stat quda_stat = time_operation(ctx.grid, ctx.opt.warmups, ctx.opt.samples, ctx.opt.repetitions,
                                  [&]() { quda_fn(); });

  if (ctx.boss) {
    std::cout << GridLogMessage << ctx.action_name << "/" << op_name
              << ": grid median " << grid_stat.median << " s, quda public-API median "
              << quda_stat.median << " s, wall ratio(grid/quda) "
              << (grid_stat.median / quda_stat.median) << std::endl;
  }

  auto emit_backend = [&](const std::string &backend, const Stat &stat) {
    JsonlWriter::Record r;
    r.add("kind", std::string("timing"));
    add_case_fields(r, ctx.opt, ctx.action_name, op_name);
    r.add("backend", backend)
        .add("timing_scope", backend == "grid" ? std::string("resident_grid") : std::string("public_api_end_to_end"))
        .add("samples", static_cast<long long>(ctx.opt.samples))
        .add("repetitions", static_cast<long long>(ctx.opt.repetitions))
        .add("median_s", stat.median)
        .add("mean_s", stat.mean)
        .add("min_s", stat.min)
        .add("max_s", stat.max)
        .add("stddev_s", stat.stddev);
    ctx.writer.emit(r);
  };
  emit_backend("grid", grid_stat);
  emit_backend("quda", quda_stat);
}

// ---------------------------------------------------------------------
// Per-action benchmark: Dslash, full matrix, preconditioned normal operator,
// (clover only) local clover multiply/inverse, and the matched CG solve.
//
// FermionAction is WilsonFermionD (Wilson control) or a WilsonFermionD&
// reference bound to a WilsonCloverFermionD object (WilsonCloverFermion
// derives from WilsonFermion, so Dhop/DhopOE/DhopEO/M/Mooee/MooeeInv all
// resolve correctly through the base reference).
// ---------------------------------------------------------------------

// The two knobs that decompose QUDA's speedup. `precise` deliberately stays
// double / RECONSTRUCT_NO in every configuration: it is the reference against
// which the sloppy inner solve is refined, so varying it would change the
// answer rather than the speed.
QudaPrecision sloppy_precision(const std::string &sloppy)
{
  if (sloppy == "double") return QUDA_DOUBLE_PRECISION;
  if (sloppy == "single") return QUDA_SINGLE_PRECISION;
  return QUDA_HALF_PRECISION;
}

QudaReconstructType precise_gauge_reconstruct(const std::string &)
{
  return QUDA_RECONSTRUCT_NO;
}

QudaReconstructType sloppy_gauge_reconstruct(const std::string &reconstruct)
{
  if (reconstruct == "no") return QUDA_RECONSTRUCT_NO;
  if (reconstruct == "12") return QUDA_RECONSTRUCT_12;
  return QUDA_RECONSTRUCT_8;
}

// Fixed source-field stream, re-seeded per action. Grid and QUDA already share
// each individual source; re-seeding additionally makes the Wilson control and
// the clover case consume the *same* source fields op-for-op, so a
// Wilson-versus-clover ratio is not confounded by different right-hand sides.
const std::vector<int> kSourceSeeds({11, 22, 33, 44});

// Restart cap for Grid's MixedPrecisionConjugateGradient. This is a TUNING
// KNOB and its default is not necessarily optimal, exactly as QUDA's
// reliable_delta is not. Grid's own tests use values around 50; 100 gives
// headroom without masking a solver that is failing to converge, since a run
// that actually needs 100 restarts will show it in restarts_all.
constexpr int kMixedOuterIterations = 100;

// Dw_f / UrbGrid_f are the single-precision counterparts, non-null only when
// --benchmark-grid-solver=mixed. They exist because Grid's
// MixedPrecisionConjugateGradient needs a linear operator in BOTH precisions
// at construction; there is no way to derive the single-precision one from the
// double. The single-precision grid is genuinely separate, not a view: vComplexF
// has a different SIMD width from vComplexD.
void run_action(const std::string &action_name, bool clover, WilsonFermionD &Dw,
                WilsonFermionF *Dw_f, GridRedBlackCartesian *UrbGrid_f,
                GridCartesian *UGrid, GridRedBlackCartesian *UrbGrid,
                LatticeGaugeField &Umu, GridParallelRNG &pRNG,
                const BenchmarkOptions &opt, JsonlWriter &writer)
{
  const bool boss = UGrid->IsBoss();
  MeasureContext ctx{writer, UGrid, opt, action_name, boss};

  if (boss) std::cout << GridLogMessage << "=== action " << action_name << " ===" << std::endl;

  pRNG.SeedFixedIntegers(kSourceSeeds);

  QudaOperator qop(UGrid, Umu, clover, opt.mass, opt.csw,
                   QUDA_DOUBLE_PRECISION, sloppy_precision(opt.sloppy_precision),
                   precise_gauge_reconstruct(opt.precision),
                   sloppy_gauge_reconstruct(opt.sloppy_reconstruct), opt.tol, opt.maxiter,
                   /*antiperiodic_time=*/true);

  auto emit_setup = [&](const std::string &kind, const std::string &stage, double seconds) {
    JsonlWriter::Record record;
    record.add("kind", kind);
    add_case_fields(record, opt, action_name, stage);
    record.add("backend", std::string("quda"))
        .add("stage", stage)
        .add("seconds", seconds);
    writer.emit(record);
  };
  emit_setup("setup", "gauge_pack", qop.gauge_pack_seconds());
  emit_setup("upload", "gauge_upload", qop.gauge_upload_seconds());
  if (clover) emit_setup("setup", "clover_construct", qop.clover_setup_seconds());

  const double grid_plaquette = WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu);
  const std::array<double, 3> quda_plaquette = qop.plaquette();
  const double plaquette_difference = std::abs(grid_plaquette - quda_plaquette[0]);
  const double plaquette_tolerance = 1e-11;
  const bool plaquette_passed = plaquette_difference <= plaquette_tolerance;
  {
    JsonlWriter::Record check;
    check.add("kind", std::string("correctness"));
    add_case_fields(check, opt, action_name, "gauge_upload");
    check.add("grid_plaquette", grid_plaquette)
        .add("quda_plaquette", quda_plaquette[0])
        .add("abs_deviation", plaquette_difference)
        .add("tolerance", plaquette_tolerance)
        .add("passed", plaquette_passed);
    writer.emit(check);
  }
  if (!plaquette_passed) fail(action_name + "/gauge_upload plaquette correctness gate failed");

  SchurDiagMooeeOperator<WilsonFermionD, LatticeFermion> HermOpEO(Dw);

  // ---- Preconditioned Dslash, Even -> Odd. Wilson has A=constant and the
  // benchmark uses the raw DhopOE on both sides. Clover's dslashQuda applies
  // A_oo^-1 D_oe, so compose the same operations explicitly in Grid. ----
  {
    LatticeFermion full_src(UGrid);
    random(pRNG, full_src);
    LatticeFermion src_even(UrbGrid);
    src_even.Checkerboard() = Even;
    pickCheckerboard(Even, src_even, full_src);

    LatticeFermion grid_out(UrbGrid);
    grid_out.Checkerboard() = Odd;
    LatticeFermion quda_out(UrbGrid);
    quda_out.Checkerboard() = Odd;
    LatticeFermion hop_out(UrbGrid);
    hop_out.Checkerboard() = Odd;

    measure_op(ctx, "pc_dslash",
               [&]() {
                 Dw.DhopOE(src_even, hop_out, DaggerNo);
                 if (clover)
                   Dw.MooeeInv(hop_out, grid_out);
                 else
                   grid_out = hop_out;
               },
               [&]() { qop.apply_pc_dslash(src_even, quda_out, QUDA_ODD_PARITY); },
               grid_out, quda_out, 1e-8);
  }

  // ---- Full matrix M (full 4D field, no checkerboarding) ----
  {
    LatticeFermion src(UGrid);
    random(pRNG, src);
    LatticeFermion grid_out(UGrid);
    LatticeFermion quda_out(UGrid);

    measure_op(ctx, "mat",
               [&]() { Dw.M(src, grid_out); },
               [&]() { qop.apply_mat(src, quda_out); },
               grid_out, quda_out, 1e-8);
  }

  // ---- Preconditioned normal operator on the Odd checkerboard: Mpc^dag Mpc ----
  {
    LatticeFermion full_src(UGrid);
    random(pRNG, full_src);
    LatticeFermion src_odd(UrbGrid);
    src_odd.Checkerboard() = Odd;
    pickCheckerboard(Odd, src_odd, full_src);

    LatticeFermion grid_out(UrbGrid);
    grid_out.Checkerboard() = Odd;
    LatticeFermion quda_out(UrbGrid);
    quda_out.Checkerboard() = Odd;

    measure_op(ctx, "normal_pc",
               [&]() { HermOpEO.MpcDagMpc(src_odd, grid_out); },
               [&]() { qop.apply_normal(src_odd, quda_out); },
               grid_out, quda_out, 1e-6);
  }

  // ---- Local clover inverse (clover action only) ----
  //
  // There is deliberately NO paired "clover_mult" case. QUDA exposes no
  // standalone direct clover application in this version: ApplyClover in
  // quda/lib/dslash_clover_helper.cu:25 opens with
  //
  //     if (!inverse) errorQuda("Unsupported direct application");
  //
  // an unconditional guard, so cloverQuda(..., inverse=0) -- which is what
  // DiracClover::Clover reaches -- aborts the job rather than returning a
  // result. This was confirmed empirically on the 2026-08-25 one-GPU smoke
  // (job 57603507), which died exactly there after every other clover gate
  // had already passed at ~1e-16.
  //
  // Consequence for the study: the direct clover term is observable in QUDA
  // only *fused* inside the stencil (ApplyWilsonClover computes A*x + kappa*D*in
  // in one kernel), i.e. through the `mat` and `normal_pc` cases. Do not
  // reintroduce a paired clover_mult against this QUDA revision.
  if (clover) {
    LatticeFermion full_src(UGrid);
    random(pRNG, full_src);
    LatticeFermion src_odd(UrbGrid);
    src_odd.Checkerboard() = Odd;
    pickCheckerboard(Odd, src_odd, full_src);

    {
      LatticeFermion grid_out(UrbGrid);
      grid_out.Checkerboard() = Odd;
      LatticeFermion quda_out(UrbGrid);
      quda_out.Checkerboard() = Odd;
      measure_op(ctx, "clover_inv",
                 [&]() { Dw.MooeeInv(src_odd, grid_out); },
                 [&]() { qop.apply_clover(src_odd, quda_out, QUDA_ODD_PARITY, /*inverse=*/true); },
                 grid_out, quda_out, 1e-6);
    }
  }

  // ---- Matched CG solve: M_pc^dag M_pc x = b on the Odd checkerboard, one
  // fixed b solved --benchmark-solve-repeats times per backend so the two
  // backends' timing/iteration-count distributions are compared on exactly
  // the same linear system. ----
  {
    LatticeFermion full_src(UGrid);
    random(pRNG, full_src);
    LatticeFermion src_odd(UrbGrid);
    src_odd.Checkerboard() = Odd;
    pickCheckerboard(Odd, src_odd, full_src);

    std::vector<double> grid_seconds, quda_seconds, quda_internal_seconds;
    std::vector<double> grid_solver_residuals, quda_solver_residuals;
    std::vector<double> grid_measured_residuals, quda_measured_residuals;
    std::vector<double> grid_seconds_per_iteration, quda_seconds_per_iteration;
    // Internal-time-based per-iteration cost. QUDA-only: its public API copies
    // source/solution across PCIe on every call, so the wall-clock vector above
    // is not a kernel-throughput number by itself -- see the ratio computed
    // from internal_s below and the campaign doc's "never use wall" rule.
    std::vector<double> quda_internal_seconds_per_iteration;
    std::vector<long long> grid_iters, quda_iters;
    LatticeFermion grid_sol(UrbGrid);
    grid_sol.Checkerboard() = Odd;
    LatticeFermion quda_sol(UrbGrid);
    quda_sol.Checkerboard() = Odd;

    // Untimed warm solves trigger any first-use setup. They are also the
    // correctness gate, so no measured solve is run unless both solutions are
    // independently valid under Grid's Schur operator.
    ConjugateGradient<LatticeFermion> warm_grid_cg(opt.tol, opt.maxiter, false);
    grid_sol = Zero();
    warm_grid_cg(HermOpEO, src_odd, grid_sol);
    qop.solve(src_odd, quda_sol);

    LatticeFermion diff(UrbGrid);
    diff.Checkerboard() = Odd;
    diff = grid_sol - quda_sol;
    const double diff_norm2 = norm2(diff);
    const double solution_norm2 = norm2(grid_sol);
    const double rel_dev = solution_norm2 > 0.0 ? std::sqrt(diff_norm2 / solution_norm2) : std::sqrt(diff_norm2);
    LatticeFermion residual(UrbGrid);
    residual.Checkerboard() = Odd;
    HermOpEO.MpcDagMpc(grid_sol, residual);
    residual = residual - src_odd;
    const double source_norm2 = norm2(src_odd);
    const double grid_independent_residual = std::sqrt(norm2(residual) / source_norm2);
    HermOpEO.MpcDagMpc(quda_sol, residual);
    residual = residual - src_odd;
    const double quda_independent_residual = std::sqrt(norm2(residual) / source_norm2);

    // A converged CG solution is only unique to O(tol), so allow a conservative
    // multiple of the requested tolerance for cross-backend solution agreement.
    const double solve_tolerance = std::max(1e-8, 100.0 * opt.tol);
    const double residual_tolerance = std::max(1e-8, 100.0 * opt.tol);
    const bool warm_iterations_match = warm_grid_cg.IterationsToComplete == qop.invert_param().iter;
    const bool passed = rel_dev <= solve_tolerance
                        && grid_independent_residual <= residual_tolerance
                        && quda_independent_residual <= residual_tolerance;

    // These residuals belong to the WARM solve only. They are recorded on the
    // correctness record and deliberately not reused as measured-repeat
    // residuals further down; each measured repeat evaluates its own.
    JsonlWriter::Record check;
    check.add("kind", std::string("correctness"));
    add_case_fields(check, opt, action_name, "solve");
    check.add("phase", std::string("warm"))
        .add("rel_deviation", rel_dev)
        .add("tolerance", solve_tolerance)
        .add("grid_independent_residual", grid_independent_residual)
        .add("quda_independent_residual", quda_independent_residual)
        .add("residual_tolerance", residual_tolerance)
        .add("grid_iterations", static_cast<long long>(warm_grid_cg.IterationsToComplete))
        .add("quda_iterations", static_cast<long long>(qop.invert_param().iter))
        .add("iterations_match", warm_iterations_match)
        .add("passed", passed);
    writer.emit(check);
    if (!passed) fail(action_name + "/solve correctness gate failed");

    // Independently evaluate the Schur residual of each measured repeat's own
    // returned solution. This is outside every timed region, so it cannot
    // perturb the numbers it validates.
    auto independent_residual_of = [&](const LatticeFermion &solution) {
      LatticeFermion check_residual(UrbGrid);
      check_residual.Checkerboard() = Odd;
      HermOpEO.MpcDagMpc(solution, check_residual);
      check_residual = check_residual - src_odd;
      return std::sqrt(norm2(check_residual) / source_norm2);
    };

    for (int r = 0; r < opt.solve_repeats; ++r) {
      ConjugateGradient<LatticeFermion> CG(opt.tol, opt.maxiter, false);
      grid_sol = Zero();
      accelerator_barrier();
      UGrid->Barrier();
      double t0 = usecond();
      CG(HermOpEO, src_odd, grid_sol);
      accelerator_barrier();
      UGrid->Barrier();
      double t1 = usecond();
      const double seconds = (t1 - t0) / 1.0e6;
      const long long iterations = static_cast<long long>(CG.IterationsToComplete);
      grid_seconds.push_back(seconds);
      grid_iters.push_back(iterations);
      grid_solver_residuals.push_back(CG.TrueResidual);
      grid_measured_residuals.push_back(independent_residual_of(grid_sol));
      if (iterations > 0) grid_seconds_per_iteration.push_back(seconds / static_cast<double>(iterations));
    }

    for (int r = 0; r < opt.solve_repeats; ++r) {
      accelerator_barrier();
      UGrid->Barrier();
      double t0 = usecond();
      qop.solve(src_odd, quda_sol);
      accelerator_barrier();
      UGrid->Barrier();
      double t1 = usecond();
      const double seconds = (t1 - t0) / 1.0e6;
      const long long iterations = static_cast<long long>(qop.invert_param().iter);
      quda_seconds.push_back(seconds);
      quda_internal_seconds.push_back(qop.invert_param().secs);
      quda_iters.push_back(iterations);
      quda_solver_residuals.push_back(qop.invert_param().true_res[0]);
      quda_measured_residuals.push_back(independent_residual_of(quda_sol));
      if (iterations > 0) {
        quda_seconds_per_iteration.push_back(seconds / static_cast<double>(iterations));
        quda_internal_seconds_per_iteration.push_back(
            quda_internal_seconds.back() / static_cast<double>(iterations));
      }
    }

    Stat grid_stat = summarize(grid_seconds);
    Stat quda_stat = summarize(quda_seconds);
    Stat quda_internal_stat = summarize(quda_internal_seconds);
    IterationStat grid_iter_stat = summarize_iterations(grid_iters);
    IterationStat quda_iter_stat = summarize_iterations(quda_iters);

    // Attributing the solve-time ratio to operator throughput requires both
    // backends to have done the same amount of work. Grid's CG and QUDA's CG
    // do not use identical stopping arithmetic (QUDA applies reliable updates
    // at reliable_delta), so a small difference is expected and tolerated;
    // anything larger makes the ratio a convergence result, not a kernel one.
    const long long iteration_span = std::max(grid_iter_stat.max, quda_iter_stat.max)
                                     - std::min(grid_iter_stat.min, quda_iter_stat.min);
    const long long iteration_scale = std::max<long long>(1, std::max(grid_iter_stat.median, quda_iter_stat.median));
    const double iteration_mismatch =
        static_cast<double>(iteration_span) / static_cast<double>(iteration_scale);
    const bool iterations_match = grid_iter_stat.median == quda_iter_stat.median;
    const bool repeats_constant = grid_iter_stat.constant && quda_iter_stat.constant;
    const bool attributable = repeats_constant && iteration_mismatch <= opt.iteration_tolerance;

    if (!attributable && boss) {
      std::cout << GridLogWarning << action_name
                << "/solve: iteration counts are not matched across backends/repeats (grid "
                << grid_iter_stat.min << "-" << grid_iter_stat.max << ", quda "
                << quda_iter_stat.min << "-" << quda_iter_stat.max << ", relative span "
                << iteration_mismatch << " > tolerance " << opt.iteration_tolerance
                << "); the solve-time ratio is NOT attributable to operator throughput"
                << std::endl;
    }
    if (boss) {
      std::cout << GridLogMessage << action_name << "/solve: grid median " << grid_stat.median
                << " s (" << grid_iter_stat.median << " iters, max measured independent res "
                << max_of(grid_measured_residuals) << "), quda wall median " << quda_stat.median
                << " s (" << quda_iter_stat.median << " iters, max measured independent res "
                << max_of(quda_measured_residuals) << "), wall ratio(grid/quda) "
                << (grid_stat.median / quda_stat.median) << ", warm solution relative L2 deviation "
                << rel_dev << ", iterations " << (iterations_match ? "MATCH" : "DIFFER")
                << ", attributable " << (attributable ? "YES" : "NO") << std::endl;
    }

    auto emit_backend = [&](const std::string &backend, const Stat &stat, const IterationStat &iter_stat,
                            const std::vector<long long> &iterations,
                            const std::vector<double> &seconds,
                            const std::vector<double> &solver_residuals,
                            const std::vector<double> &measured_residuals,
                            const std::vector<double> &seconds_per_iteration,
                            const std::vector<double> &internal_seconds_per_iteration,
                            double internal_s, bool has_internal_time) {
      JsonlWriter::Record r;
      r.add("kind", std::string("solve"));
      add_case_fields(r, opt, action_name, "solve");
      r.add("backend", backend)
          .add("timing_scope", backend == "grid" ? std::string("end_to_end_grid") : std::string("public_api_end_to_end"))
          .add("solve_repeats", static_cast<long long>(opt.solve_repeats))
          .add("median_s", stat.median)
          .add("mean_s", stat.mean)
          .add("min_s", stat.min)
          .add("max_s", stat.max)
          .add("stddev_s", stat.stddev)
          .add("sample_seconds", seconds)
          // "iterations" is the per-repeat median; iterations_constant says
          // whether quoting one number for this backend is honest at all.
          .add("iterations", iter_stat.median)
          .add("iterations_min", iter_stat.min)
          .add("iterations_max", iter_stat.max)
          .add("iterations_constant", iter_stat.constant)
          .add("iterations_all", iterations)
          // Worst case over the measured repeats, not the warm solve.
          .add("true_residual", max_of(solver_residuals))
          .add("true_residual_all", solver_residuals)
          .add("independent_residual", max_of(measured_residuals))
          .add("independent_residual_all", measured_residuals)
          .add("residual_scope", std::string("max_over_measured_repeats"))
          // Wall-clock per-iteration cost. For QUDA this includes the public
          // API's per-call PCIe transfer, so it is NOT a kernel-throughput
          // number on its own -- see s_per_iteration_median_internal below.
          .add("s_per_iteration_median_wall", median_of(seconds_per_iteration))
          .add("iterations_match_across_backends", iterations_match)
          .add("iteration_relative_span", iteration_mismatch)
          .add("iteration_tolerance", opt.iteration_tolerance)
          .add("attributable_to_operator_throughput", attributable)
          .add("requested_tol", opt.tol)
          .add("maxiter", static_cast<long long>(opt.maxiter));
      if (has_internal_time) {
        r.add("internal_s", internal_s);
        r.add("s_per_iteration_median_internal", median_of(internal_seconds_per_iteration));
      }
      writer.emit(r);
    };
    static const std::vector<double> kNoInternalPerIteration;
    emit_backend("grid", grid_stat, grid_iter_stat, grid_iters, grid_seconds, grid_solver_residuals,
                 grid_measured_residuals, grid_seconds_per_iteration, kNoInternalPerIteration, 0.0, false);
    emit_backend("quda", quda_stat, quda_iter_stat, quda_iters, quda_seconds, quda_solver_residuals,
                 quda_measured_residuals, quda_seconds_per_iteration, quda_internal_seconds_per_iteration,
                 quda_internal_stat.median, true);

    // ---- Grid mixed-precision CG (double outer, single inner) ----
    //
    // Same scheme QUDA uses for its sloppy solve, so this is what says how much
    // of QUDA's precision advantage Grid can recover with stock code. Note it
    // canNOT recover the gauge-reconstruction part: Grid always stores full
    // 18-real links, so `sloppy_reconstruct` has no Grid counterpart.
    //
    // Emitted as backend "grid_mixed" against the SAME source and the SAME
    // requested tolerance, and gated on the same independently evaluated
    // Grid-Schur residual as the other two. Iteration counts are NOT comparable
    // to the double CG -- the mixed solver reports inner iterations, restarts,
    // and a final-step count separately -- so all three are recorded and
    // "iterations" carries the inner total, which is the work that dominates.
    if (opt.grid_solver == "mixed" && Dw_f != nullptr && UrbGrid_f != nullptr) {
      SchurDiagMooeeOperator<WilsonFermionF, LatticeFermionF> HermOpEO_f(*Dw_f);

      std::vector<double> mixed_seconds, mixed_residuals, mixed_solver_residuals;
      std::vector<long long> mixed_inner, mixed_outer, mixed_final;
      LatticeFermion mixed_sol(UrbGrid);
      mixed_sol.Checkerboard() = Odd;

      // Warm solve doubles as the correctness gate, exactly as above.
      {
        MixedPrecisionConjugateGradient<LatticeFermion, LatticeFermionF> warm_mpcg(
            opt.tol, opt.maxiter, kMixedOuterIterations, UrbGrid_f, HermOpEO_f, HermOpEO);
        mixed_sol = Zero();
        warm_mpcg(src_odd, mixed_sol);
      }
      const double mixed_warm_residual = independent_residual_of(mixed_sol);
      LatticeFermion mixed_diff(UrbGrid);
      mixed_diff.Checkerboard() = Odd;
      mixed_diff = grid_sol - mixed_sol;
      const double mixed_rel_dev = std::sqrt(norm2(mixed_diff) / solution_norm2);
      const bool mixed_passed = mixed_warm_residual <= residual_tolerance && mixed_rel_dev <= solve_tolerance;

      {
        JsonlWriter::Record check;
        check.add("kind", std::string("correctness"));
        add_case_fields(check, opt, action_name, "solve_mixed");
        check.add("phase", std::string("warm"))
            .add("rel_deviation", mixed_rel_dev)
            .add("tolerance", solve_tolerance)
            .add("independent_residual", mixed_warm_residual)
            .add("residual_tolerance", residual_tolerance)
            .add("passed", mixed_passed);
        writer.emit(check);
      }
      if (!mixed_passed) fail(action_name + "/solve_mixed correctness gate failed");

      for (int r = 0; r < opt.solve_repeats; ++r) {
        MixedPrecisionConjugateGradient<LatticeFermion, LatticeFermionF> mpcg(
            opt.tol, opt.maxiter, kMixedOuterIterations, UrbGrid_f, HermOpEO_f, HermOpEO);
        mixed_sol = Zero();
        accelerator_barrier();
        UGrid->Barrier();
        double t0 = usecond();
        mpcg(src_odd, mixed_sol);
        accelerator_barrier();
        UGrid->Barrier();
        double t1 = usecond();
        mixed_seconds.push_back((t1 - t0) / 1.0e6);
        mixed_inner.push_back(static_cast<long long>(mpcg.TotalInnerIterations));
        mixed_outer.push_back(static_cast<long long>(mpcg.TotalOuterIterations));
        mixed_final.push_back(static_cast<long long>(mpcg.TotalFinalStepIterations));
        mixed_solver_residuals.push_back(mpcg.TrueResidual);
        mixed_residuals.push_back(independent_residual_of(mixed_sol));
      }

      Stat mixed_stat = summarize(mixed_seconds);
      IterationStat mixed_inner_stat = summarize_iterations(mixed_inner);

      if (boss) {
        std::cout << GridLogMessage << action_name << "/solve_mixed: grid mixed median "
                  << mixed_stat.median << " s (" << mixed_inner_stat.median << " inner iters, "
                  << summarize_iterations(mixed_outer).median << " restarts, max measured independent res "
                  << max_of(mixed_residuals) << "), speedup vs grid double "
                  << (grid_stat.median / mixed_stat.median) << ", vs quda internal "
                  << (mixed_stat.median / quda_internal_stat.median) << std::endl;
      }

      JsonlWriter::Record r;
      r.add("kind", std::string("solve"));
      add_case_fields(r, opt, action_name, "solve");
      r.add("backend", std::string("grid_mixed"))
          .add("timing_scope", std::string("end_to_end_grid"))
          .add("solve_repeats", static_cast<long long>(opt.solve_repeats))
          .add("median_s", mixed_stat.median)
          .add("mean_s", mixed_stat.mean)
          .add("min_s", mixed_stat.min)
          .add("max_s", mixed_stat.max)
          .add("stddev_s", mixed_stat.stddev)
          .add("sample_seconds", mixed_seconds)
          .add("iterations", mixed_inner_stat.median)
          .add("iterations_min", mixed_inner_stat.min)
          .add("iterations_max", mixed_inner_stat.max)
          .add("iterations_constant", mixed_inner_stat.constant)
          .add("iterations_all", mixed_inner)
          .add("iterations_scope", std::string("inner_cg_total"))
          .add("restarts_all", mixed_outer)
          .add("final_step_iterations_all", mixed_final)
          .add("true_residual", max_of(mixed_solver_residuals))
          .add("true_residual_all", mixed_solver_residuals)
          .add("independent_residual", max_of(mixed_residuals))
          .add("independent_residual_all", mixed_residuals)
          .add("residual_scope", std::string("max_over_measured_repeats"))
          .add("outer_iterations_cap", static_cast<long long>(kMixedOuterIterations))
          .add("requested_tol", opt.tol)
          .add("maxiter", static_cast<long long>(opt.maxiter));
      writer.emit(r);
    }
  }
}

} // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  try {
    static_assert(Nc == 3, "this benchmark targets Nc=3 (QUDA is 3-color only)");

    BenchmarkOptions opt = parse_options(argc, argv);

    GridCartesian *UGrid =
        SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
    const bool boss = UGrid->IsBoss();

    validate_conversion_round_trips(UGrid, UrbGrid);
    if (boss) std::cout << GridLogMessage << "host conversion round trips: PASSED" << std::endl;

    std::vector<int> seeds({1, 2, 3, 4});
    GridParallelRNG pRNG(UGrid);
    pRNG.SeedFixedIntegers(seeds);

    LatticeGaugeField Umu(UGrid);
    if (opt.input == "hot") {
      SU<Nc>::HotConfiguration(pRNG, Umu);
    } else {
      FieldMetaData header;
      IldgReader reader;
      reader.open(opt.cfg);
      reader.readConfiguration(Umu, header);
      reader.close();
    }
    // norm2() is COLLECTIVE -- it reduces across every rank. It must therefore
    // be evaluated outside the boss guard: with it inside, rank 0 entered the
    // global sum while ranks 1..n-1 skipped the block, and the job deadlocked.
    // Invisible at one rank (boss is the only rank), which is why every
    // single-rank run passed and this only appeared at 4 GPUs -- see jobs
    // 57607090 and 57607372, both hung immediately after the conversion gate.
    const double gauge_norm2 = norm2(Umu);
    if (boss) {
      std::cout << GridLogMessage << "gauge input=" << opt.input << " norm2(Umu)=" << gauge_norm2 << std::endl;
    }

    {
    // Session-wide QUDA MPI/device init; must outlive every QudaOperator
    // constructed below (loadGaugeQuda/freeGaugeQuda happen per action).
    QudaSession session(UGrid);

    JsonlWriter writer(opt.output, boss);
    {
      JsonlWriter::Record record;
      record.add("kind", std::string("setup"));
      add_case_fields(record, opt, "all", "quda_init");
      record.add("backend", std::string("quda"))
          .add("stage", std::string("quda_init"))
          .add("seconds", session.init_seconds());
      writer.emit(record);
    }

    const bool do_wilson = (opt.action == "wilson" || opt.action == "both");
    const bool do_clover = (opt.action == "clover" || opt.action == "both");
    const bool do_mixed = (opt.grid_solver == "mixed");

    // Single-precision counterparts, built only when a mixed-precision Grid
    // solve is requested. These are separate grid objects rather than views:
    // vComplexF has a different SIMD width from vComplexD, so the site layout
    // differs and precisionChange() must do real work.
    std::unique_ptr<GridCartesian> UGrid_f;
    std::unique_ptr<GridRedBlackCartesian> UrbGrid_f;
    std::unique_ptr<LatticeGaugeFieldF> Umu_f;
    if (do_mixed) {
      UGrid_f.reset(SpaceTimeGrid::makeFourDimGrid(
          GridDefaultLatt(), GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi()));
      UrbGrid_f.reset(SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f.get()));
      Umu_f.reset(new LatticeGaugeFieldF(UGrid_f.get()));
      precisionChange(*Umu_f, Umu);
      if (boss)
        std::cout << GridLogMessage << "single-precision gauge prepared for mixed-precision Grid CG"
                  << std::endl;
    }

    if (do_wilson) {
      std::vector<Complex> phases(Nd, 1.0);
      phases[Nd - 1] = -1.0;
      WilsonFermionD::ImplParams implParams;
      implParams.boundary_phases = phases;
      WilsonFermionD Dw(Umu, *UGrid, *UrbGrid, opt.mass, implParams);

      std::unique_ptr<WilsonFermionF> Dw_f;
      if (do_mixed) {
        WilsonFermionF::ImplParams implParamsF;
        implParamsF.boundary_phases = phases;
        Dw_f.reset(new WilsonFermionF(*Umu_f, *UGrid_f, *UrbGrid_f, opt.mass, implParamsF));
      }
      run_action("wilson", /*clover=*/false, Dw, Dw_f.get(), UrbGrid_f.get(),
                 UGrid, UrbGrid, Umu, pRNG, opt, writer);
    }

    if (do_clover) {
      std::vector<Complex> phases(Nd, 1.0);
      phases[Nd - 1] = -1.0;
      WilsonCloverFermionD::ImplParams implParams;
      implParams.boundary_phases = phases;
      WilsonAnisotropyCoefficients anisotropy;
      WilsonCloverFermionD Dwc(Umu, *UGrid, *UrbGrid, opt.mass, opt.csw, opt.csw, anisotropy, implParams);

      std::unique_ptr<WilsonCloverFermionF> Dwc_f;
      if (do_mixed) {
        WilsonCloverFermionF::ImplParams implParamsF;
        implParamsF.boundary_phases = phases;
        WilsonAnisotropyCoefficients anisotropyF;
        Dwc_f.reset(new WilsonCloverFermionF(*Umu_f, *UGrid_f, *UrbGrid_f, opt.mass, opt.csw, opt.csw,
                                             anisotropyF, implParamsF));
      }
      run_action("clover", /*clover=*/true, Dwc, Dwc_f.get(), UrbGrid_f.get(),
                 UGrid, UrbGrid, Umu, pRNG, opt, writer);
    }
    } // QudaSession torn down here, after all QudaOperator instances are gone

    if (boss) std::cout << GridLogMessage << "benchmark complete, records written to " << opt.output << std::endl;
  } catch (const std::exception &error) {
    std::cerr << "benchmark_grid_quda_wilson_clover: ERROR: " << error.what() << std::endl;
    Grid_finalize();
    return 2;
  }

  Grid_finalize();
  return 0;
}
