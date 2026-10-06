// probe_half_apply_selftest.cc
//
// Standalone self-test of ProbeMG::HalfStencilCoarseApply (probe_mg_half_apply.h): fp16 link
// STORAGE, fp32 arithmetic, for the level-1 coarse apply of the Grid MG probe.
//
// Builds a CoarsenedMatrix<vSpinColourVectorF, vTComplexF, 48> directly on the coarse lattice
// given by --grid (default 12.12.12.24, the production level-1 lattice), fills its 9 link
// fields with Gaussian site matrices, and compares
//   (a) CoarsenedMatrix::M                    fp32 links (the apply the probe uses today)
//   (b) HalfStencilCoarseApply::Op            the same links stored as binary16
// then times both. Three numerical checks, all printed:
//   1. ||a-b||/||a||      fp16 rounding level, expect ~3e-4 for O(1) links. THE GATE (< 1e-2).
//   2. shift path         Op with shift s against a + s*in, expect the same ~3e-4.
//   3. rounded-link check M applied to links rounded through binary16 in place, against (b).
//      Same link values, same accumulation order, so expect ~1e-7 or exactly 0. A value near
//      check 1 would mean the kernel reads the wrong elements (lane / index bug), not precision.
//      Runs LAST because it overwrites the fp32 links.
//
// No physics operator is involved: the links are random, so this tests the kernel, its layout
// and its bandwidth, not the MG solve. Launch through run_selftest.sh (4 nodes x 4 GPUs).
//
// Options (besides Grid's --grid --mpi --accelerator-threads --shm --shm-mpi --comms-overlap
// --device-mem):
//   --selftest-reps N          timed applies of each (default 50)
//   --selftest-link-scale s    multiply the Gaussian links by s (default 1.0)
//   --selftest-shift s         shift used by check 2 (default 0.25)

#include <Grid/Grid.h>

#include "probe_mg_half_apply.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace Grid;

// Named (not anonymous) namespace: it holds a function with an extended device lambda.
namespace HalfSelftest {

std::string read_string(int argc, char **argv, const std::string &option, const std::string &fallback)
{
  char **itr = std::find(argv, argv + argc, option);
  if (itr != argv + argc && ++itr != argv + argc) return std::string(*itr);
  return fallback;
}

double read_double(int argc, char **argv, const std::string &option, double fallback)
{
  const std::string payload = read_string(argc, argv, option, "");
  return payload.empty() ? fallback : std::stod(payload);
}

int read_int(int argc, char **argv, const std::string &option, int fallback)
{
  const std::string payload = read_string(argc, argv, option, "");
  return payload.empty() ? fallback : std::stoi(payload);
}

std::string coordinate_string(const Coordinate &c)
{
  std::string s;
  for (std::size_t d = 0; d < c.size(); ++d) s += (d ? "." : "") + std::to_string(c[d]);
  return s;
}

// Round every fp32 word of a lattice through binary16, in place. Each float is rounded on its
// own, so the SIMD-lane / tensor order of the words does not matter here.
template <class vobj>
void round_lattice_through_half(Lattice<vobj> &l)
{
  static_assert(sizeof(vobj) % sizeof(float) == 0, "fp32 lattice expected");
  autoView(l_v, l, AcceleratorWrite);
  float *f = (float *)&l_v[0];
  const uint64_t n = (uint64_t)l.Grid()->oSites() * (sizeof(vobj) / sizeof(float));
  accelerator_for(i, n, 1, { f[i] = ProbeMG::half_round_float(f[i]); });
}

// Boss-only progress marker, flushed, so a hang is localised to one phase. Rank 0's view only:
// a line means rank 0 got past the phase, not that every rank did.
void progress(bool boss, const char *phase)
{
  if (boss) std::cout << GridLogMessage << "PROGRESS " << phase << std::endl << std::flush;
}

// ms per call of f(), barriered on the device and across ranks.
template <class F>
double time_ms(F &&f, int reps, GridBase *g)
{
  accelerator_barrier();
  g->Barrier();
  const double s = usecond();
  for (int i = 0; i < reps; ++i) f();
  accelerator_barrier();
  g->Barrier();
  return (usecond() - s) / 1.0e3 / double(reps);
}

} // namespace HalfSelftest

int main(int argc, char **argv)
{
  using namespace HalfSelftest;
  Grid_init(&argc, &argv);

  const int reps = std::max(1, read_int(argc, argv, "--selftest-reps", 50));
  const double link_scale = read_double(argc, argv, "--selftest-link-scale", 1.0);
  const double test_shift = read_double(argc, argv, "--selftest-shift", 0.25);

  Coordinate clatt({12, 12, 12, 24});
  if (GridCmdOptionExists(argv, argv + argc, "--grid")) clatt = GridDefaultLatt();

  constexpr int nbasis = 48;
  typedef CoarsenedMatrix<vSpinColourVectorF, vTComplexF, nbasis> OldOp;
  typedef OldOp::CoarseVector CoarseVector;
  typedef ProbeMG::HalfStencilCoarseApply<vSpinColourVectorF, vTComplexF, nbasis> HalfOp;

  GridCartesian *Coarse4d =
      SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
  // Checkered on x only, exactly as StencilCoarseApply: CoarsenedMatrix builds red-black members
  // this test never applies, and the default all-dims mask asserts on odd reduced dimensions.
  Coordinate mask({1, 0, 0, 0});
  GridRedBlackCartesian *CoarseRB = new GridRedBlackCartesian(Coarse4d, mask, 0);
  const bool boss = Coarse4d->IsBoss();
  bool pass = false;

  {
    const int ranks = Coarse4d->ProcessorCount();
    const double gsites = (double)Coarse4d->gSites();
    const double npoint = 9.0;
    const double elems = gsites * npoint * nbasis * nbasis;  // complex link elements, global
    const double bytes_f32 = elems * 8.0;                     // 2 x fp32
    const double bytes_f16 = elems * 4.0;                     // 2 x binary16

    if (boss) {
      std::cout << GridLogMessage << "half-apply selftest  nbasis " << nbasis << "  CComplex vTComplexF"
                << "  Nsimd " << vTComplexF::Nsimd() << std::endl;
      std::cout << GridLogMessage << "coarse gdim " << coordinate_string(Coarse4d->GlobalDimensions())
                << "  mpi " << coordinate_string(Coarse4d->_processors) << "  ldim "
                << coordinate_string(Coarse4d->LocalDimensions()) << "  simd "
                << coordinate_string(Coarse4d->_simd_layout) << "  rdim "
                << coordinate_string(Coarse4d->_rdimensions) << "  oSites " << Coarse4d->oSites()
                << "  ranks " << ranks << std::endl;
      std::cout << GridLogMessage << "reps " << reps << "  link scale " << link_scale << "  shift check "
                << test_shift << std::endl;
    }

    OldOp op(*Coarse4d, *CoarseRB, 1);  // hermitian=1 as in StencilCoarseApply (M ignores it)
    GridParallelRNG CRNG(Coarse4d);
    CRNG.SeedFixedIntegers(std::vector<int>({101, 202, 303, 404}));
    for (int p = 0; p < op.geom.npoint; ++p) {
      gaussian(CRNG, op.A[p]);
      if (link_scale != 1.0) op.A[p] = link_scale * op.A[p];
    }

    CoarseVector in(Coarse4d), a(Coarse4d), b(Coarse4d), c(Coarse4d), diff(Coarse4d);
    gaussian(CRNG, in);

    HalfOp half(op, 0.0);

    if (boss) {
      std::cout << GridLogMessage << "link max |re|,|im| (global, fp32) " << half.max_abs
                << "   binary16 max finite 65504" << std::endl;
      std::cout << GridLogMessage << "link storage fp32 " << bytes_f32 / 1.0e6 << " MB total, "
                << bytes_f32 / 1.0e6 / ranks << " MB/rank;  fp16 " << bytes_f16 / 1.0e6 << " MB total, "
                << bytes_f16 / 1.0e6 / ranks << " MB/rank  (fp16 store this rank "
                << (double)half.bytes() / 1.0e6 << " MB)" << std::endl;
    }

    progress(boss, "import done");

    // ⛔ Every collective (norm2, GlobalSum/Max, Barrier, HaloExchange) is called by ALL ranks,
    // outside any `if (boss)`. An earlier revision called norm2(in) inside the boss-only print
    // below: rank 0 sat in the allreduce while the other ranks reached the timing Barrier, and
    // the 16-rank run hung right after the link-storage line.

    // 1. fp16 storage against fp32
    op.M(in, a);
    progress(boss, "fp32 apply done");
    half.Op(in, b);
    progress(boss, "half apply done");
    diff = a - b;
    const double na = norm2(a);
    const double nin = norm2(in);
    const double rel = std::sqrt(norm2(diff) / na);

    // 2. shift path
    half.shift = test_shift;
    half.Op(in, c);
    half.shift = 0.0;
    diff = a + test_shift * in;
    diff = diff - c;
    const double rel_shift = std::sqrt(norm2(diff) / norm2(c));
    progress(boss, "checks 1-2 done");

    if (boss) {
      std::cout << GridLogMessage << "||a|| " << std::sqrt(na) << "  ||in|| " << std::sqrt(nin) << std::endl;
      std::cout << GridLogMessage << "relative difference ||M_fp32 in - M_fp16 in|| / ||M_fp32 in|| = " << rel
                << (rel < 1.0e-2 ? "  (below 1e-2)" : "  (ABOVE 1e-2)") << std::endl;
      std::cout << GridLogMessage << "shift check rel diff " << rel_shift
                << (rel_shift < 1.0e-2 ? "  AGREES" : "  DISAGREES") << std::endl;
    }

    // Timing: one warm-up round, then the timed round.
    for (int i = 0; i < 3; ++i) {
      op.M(in, a);
      half.Op(in, b);
    }
    progress(boss, "warm-up done");
    const double ms_f32 = time_ms([&]() { op.M(in, a); }, reps, Coarse4d);
    progress(boss, "fp32 timing loop done");
    const double ms_f16 = time_ms([&]() { half.Op(in, b); }, reps, Coarse4d);
    progress(boss, "half timing loop done");
    if (boss) {
      const double bw_f32 = bytes_f32 / (ms_f32 * 1.0e-3) / 1.0e9;  // aggregate GB/s, links only
      const double bw_f16 = bytes_f16 / (ms_f16 * 1.0e-3) / 1.0e9;
      std::cout << GridLogMessage << "half apply " << ms_f16 << " ms, fp32 apply " << ms_f32 << " ms, ratio "
                << (ms_f32 / ms_f16) << "  (fp32/half, " << reps << " applies each)" << std::endl;
      std::cout << GridLogMessage << "effective link bandwidth: half " << bw_f16 << " GB/s aggregate ("
                << bw_f16 / ranks << " GB/s per GPU), fp32 " << bw_f32 << " GB/s aggregate ("
                << bw_f32 / ranks << " GB/s per GPU)" << std::endl;
    }

    // 3. rounded-link check (destroys the fp32 links, so last)
    for (int p = 0; p < op.geom.npoint; ++p) round_lattice_through_half(op.A[p]);
    half.Op(in, b);
    op.M(in, a);
    diff = a - b;
    const double rel_round = std::sqrt(norm2(diff) / norm2(a));
    progress(boss, "rounded-link check done");
    if (boss) {
      std::cout << GridLogMessage << "rounded-link check (M on fp16-rounded fp32 links vs half apply) rel diff "
                << rel_round
                << (rel_round < 1.0e-5 ? "  AGREES (indexing exact)" : "  DISAGREES (indexing suspect)")
                << std::endl;
    }

    pass = (rel < 1.0e-2);
    if (boss) std::cout << GridLogMessage << "SELFTEST RESULT: " << (pass ? "PASS" : "FAIL") << std::endl;
  }

  delete CoarseRB;
  Grid_finalize();
  return pass ? 0 : 1;
}
