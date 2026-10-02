#pragma once
// C2: fast ImportGauge for the compact clover operator, project side (libGrid untouched).
//
// FastImportGauge(op, U) reproduces CompactWilsonCloverFermion<Impl, CompactCloverHelpers<Impl>>
// ::ImportGauge (stock Grid 3d3eff86, CompactWilsonCloverFermionImplementation.h:392-470) phase by
// phase, with the stock helper calls in the stock order, and changes two phases behind runtime
// switches (read once from the environment, on iff the value starts with '1', default OFF):
//
//   HASEN_GRID_SHARE_FIELDSTRENGTH=1 (C2b, exact). The six field-strength fields depend only on U
//     (WilsonLoops<Impl>::FieldStrength with periodic link shifts; mass, csw and the fermion
//     boundary phases do not enter), so every operator imported at the same U can use the same
//     six fields. One cache per Impl, i.e. per precision: the fp32 operator gets an fp32 field
//     strength computed from its own fp32 U, exactly as stock; an fp64 field strength is never fed
//     to an fp32 operator. Keyed like the C1 skip (import_guard.h): a shadow copy of the U the six
//     fields were computed from, reuse only if norm2(U_mu - shadow_mu) == 0.0 for every mu.
//     Bitwise-equal U => bitwise-equal field strength => bit-identical import.
//   HASEN_GRID_GPU_CLOVER_INV=1 (C2a, roundoff). The stock inversion of the clover term
//     (CompactWilsonCloverHelpers::Invert: a host thread_for with one Eigen 12x12 PartialPivLU
//     inverse per site, plus the device->host->device round trip of the four compact fields that
//     its CpuRead/CpuWrite views force) is replaced by one accelerator_for over sites that
//     rebuilds each site's two 6x6 blocks with Invert's exact index map, inverts each in fp64
//     complex arithmetic (Gauss-Jordan, partial pivoting; for an fp32 operator the fp32 entries are
//     widened to fp64 and the inverse rounded back, as Invert does) and writes the inverse's
//     diagonal and upper triangle back. Same matrix, different algorithm and operation order than
//     Eigen's: roundoff-level differences (src/gauge_import/test_fast_import.cc measures them).
//     The eight checkerboard picks then also run on the device (acceleratorPickCheckerboard, an
//     exact copy) so that the import never moves the clover fields through the host.
//
// Dispatch: GuardedImportGauge (import_guard.h) calls ImportGaugeDispatch, which takes this path
// when either gate is on (composing with HASEN_GRID_IMPORT_SKIP, which decides first whether an
// import happens at all) and otherwise calls the stock op.ImportGauge(U), untouched.
//
// HASEN_GRID_IMPORT_TIMERS=1: one "ImportTimer" line per performed import on the boss rank, with
// phase seconds (accelerator_barrier between phases) and cumulative count/time per precision.
// With both gates off it times the stock call as a whole (stock prints its own phase breakdown
// only at GridLogDebug). The barriers change timing only, never numerics.
//
// Supported operators: exactly the driver's two, CompactWilsonCloverFermion<WilsonImplD|WilsonImplF,
// CompactCloverHelpers<...>>; any other type aborts with a message if a gate is on. Open
// boundaries (fixedBoundaries, i.e. a zero time boundary phase) are asserted absent: stock would
// call ModifyBoundaries there; the driver is antiperiodic in time.
//
// Memory (per precision, only with SHARE on): shadow GaugeField + six GaugeLinkFields = ten link
// fields, persistent; at 48^3x96 on 16 GPUs (663,552 sites per GPU) 0.96 GB fp64 and 0.48 GB fp32
// per GPU. The GPU inversion allocates nothing.
//
// Hazards (as C1): a field changed in place keeps its Grid pointer, so the shadow comparison is
// the only key; it costs four PeekIndex + subtract + norm2 per import, a few ms at 48^3.
#include <Grid/Grid.h>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <sstream>
#include <type_traits>

namespace Grid {

inline bool FastImportEnv(const char *name) {
  const char *e = std::getenv(name);
  return e && e[0] == '1';
}
inline bool ShareFieldStrengthEnabled() {
  static const bool on = FastImportEnv("HASEN_GRID_SHARE_FIELDSTRENGTH");
  return on;
}
inline bool GpuCloverInvEnabled() {
  static const bool on = FastImportEnv("HASEN_GRID_GPU_CLOVER_INV");
  return on;
}
inline bool ImportTimersEnabled() {
  static const bool on = FastImportEnv("HASEN_GRID_IMPORT_TIMERS");
  return on;
}
inline bool FastImportEnabled() { return ShareFieldStrengthEnabled() || GpuCloverInvEnabled(); }

// ---- the operator types the fast path supports: the driver's WCF and WCF_f ----
template <class Op> struct FastImportSupported : std::false_type {};
template <>
struct FastImportSupported<CompactWilsonCloverFermion<WilsonImplD, CompactCloverHelpers<WilsonImplD>>>
    : std::true_type {};
template <>
struct FastImportSupported<CompactWilsonCloverFermion<WilsonImplF, CompactCloverHelpers<WilsonImplF>>>
    : std::true_type {};

// ---- field strength: the six fields of stock ImportGauge, same (mu, nu) and order ----
template <class Impl>
inline void FastImportFieldStrength(typename Impl::GaugeLinkField *F[6], const typename Impl::GaugeField &U) {
  WilsonLoops<Impl>::FieldStrength(*F[0], U, Zdir, Ydir);  // Bx
  WilsonLoops<Impl>::FieldStrength(*F[1], U, Zdir, Xdir);  // By
  WilsonLoops<Impl>::FieldStrength(*F[2], U, Ydir, Xdir);  // Bz
  WilsonLoops<Impl>::FieldStrength(*F[3], U, Tdir, Xdir);  // Ex
  WilsonLoops<Impl>::FieldStrength(*F[4], U, Tdir, Ydir);  // Ey
  WilsonLoops<Impl>::FieldStrength(*F[5], U, Tdir, Zdir);  // Ez
}

template <class Impl>
struct FieldStrengthCache {
  typedef typename Impl::GaugeField GaugeField;
  typedef typename Impl::GaugeLinkField GaugeLinkField;
  std::unique_ptr<GaugeField> shadow;      // the U the six fields were computed from
  std::unique_ptr<GaugeLinkField> F[6];    // Bx By Bz Ex Ey Ez
  long hits = 0;
  long misses = 0;
};
template <class Impl>
inline FieldStrengthCache<Impl> &FieldStrengthCacheFor() {
  static FieldStrengthCache<Impl> c;  // one per Impl = one per precision
  return c;
}

// Fill F[] with the field strength of U from the cache (computing it on a miss). hit = reused.
template <class Impl>
inline void SharedFieldStrength(const typename Impl::GaugeField &U, const typename Impl::GaugeLinkField *F[6],
                                bool &hit) {
  typedef typename Impl::GaugeField GaugeField;
  typedef typename Impl::GaugeLinkField GaugeLinkField;
  auto &c = FieldStrengthCacheFor<Impl>();
  hit = false;
  if (c.shadow && c.shadow->Grid() == U.Grid()) {
    // Exact device-side test, one link direction at a time (as import_guard.h).
    hit = true;
    for (int mu = 0; mu < Nd && hit; ++mu) {
      auto d = PeekIndex<LorentzIndex>(U, mu);
      d = d - PeekIndex<LorentzIndex>(*c.shadow, mu);
      if (norm2(d) != 0.0) hit = false;
    }
  }
  if (hit) {
    ++c.hits;
  } else {
    if (!c.shadow || c.shadow->Grid() != U.Grid()) {
      c.shadow.reset(new GaugeField(U.Grid()));
      for (int i = 0; i < 6; ++i) c.F[i].reset(new GaugeLinkField(U.Grid()));
    }
    GaugeLinkField *Fw[6];
    for (int i = 0; i < 6; ++i) Fw[i] = c.F[i].get();
    FastImportFieldStrength<Impl>(Fw, U);
    *c.shadow = U;
    ++c.misses;
  }
  for (int i = 0; i < 6; ++i) F[i] = c.F[i].get();
}

// ---- GPU inversion of the compact clover term ----
// One SIMD lane of one site (scalar objects): rebuild each 6x6 block exactly as
// CompactWilsonCloverHelpers::Invert does (diagonal entry from `diagonal`, i<j from `triangle`,
// i>j its conjugate; Invert's 12x12 is block diagonal with block b = rows/cols 6b..6b+5), widen to
// fp64, invert in place by Gauss-Jordan with partial pivoting (pivot = largest |a_ik|^2 in the
// column), and store the inverse's diagonal and upper triangle (Invert stores the same entries).
template <class Impl, class sobjD, class sobjT>
accelerator_inline void FastImportInvertLane(const sobjD &d, const sobjT &t, sobjD &di, sobjT &ti) {
  typedef CompactWilsonCloverHelpers<Impl> CH;
  typedef typename sobjD::scalar_type S;  // ComplexD or ComplexF
  typedef typename S::value_type SR;
  constexpr int N = CompactWilsonCloverTypes<Impl>::Nred;  // 6
  for (int b = 0; b < CompactWilsonCloverTypes<Impl>::Nblock; ++b) {
    double ar[N][N], ai[N][N];
    int piv[N];
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        if (i == j) {
          ar[i][j] = (double)d()(b)(i).real();
          ai[i][j] = (double)d()(b)(i).imag();
        } else if (i < j) {
          ar[i][j] = (double)t()(b)(CH::triangle_index(i, j)).real();
          ai[i][j] = (double)t()(b)(CH::triangle_index(i, j)).imag();
        } else {
          ar[i][j] = (double)t()(b)(CH::triangle_index(i, j)).real();
          ai[i][j] = -(double)t()(b)(CH::triangle_index(i, j)).imag();
        }
      }
    }
    for (int k = 0; k < N; ++k) {
      int p = k;
      double best = ar[k][k] * ar[k][k] + ai[k][k] * ai[k][k];
      for (int i = k + 1; i < N; ++i) {
        const double v = ar[i][k] * ar[i][k] + ai[i][k] * ai[i][k];
        if (v > best) { best = v; p = i; }
      }
      piv[k] = p;
      if (p != k) {
        for (int j = 0; j < N; ++j) {
          double tr = ar[k][j], tq = ai[k][j];
          ar[k][j] = ar[p][j]; ai[k][j] = ai[p][j];
          ar[p][j] = tr;       ai[p][j] = tq;
        }
      }
      // 1/pivot; the pivot slot becomes 1 and is then scaled with the row (in-place inverse)
      const double pr = ar[k][k], pq = ai[k][k];
      const double den = pr * pr + pq * pq;
      const double rr = pr / den, rq = -pq / den;
      ar[k][k] = 1.0; ai[k][k] = 0.0;
      for (int j = 0; j < N; ++j) {
        const double xr = ar[k][j], xq = ai[k][j];
        ar[k][j] = xr * rr - xq * rq;
        ai[k][j] = xr * rq + xq * rr;
      }
      for (int i = 0; i < N; ++i) {
        if (i == k) continue;
        const double fr = ar[i][k], fq = ai[i][k];
        ar[i][k] = 0.0; ai[i][k] = 0.0;
        for (int j = 0; j < N; ++j) {
          ar[i][j] -= fr * ar[k][j] - fq * ai[k][j];
          ai[i][j] -= fr * ai[k][j] + fq * ar[k][j];
        }
      }
    }
    // undo the row interchanges as column interchanges, last first
    for (int k = N - 1; k >= 0; --k) {
      const int p = piv[k];
      if (p == k) continue;
      for (int i = 0; i < N; ++i) {
        double tr = ar[i][k], tq = ai[i][k];
        ar[i][k] = ar[i][p]; ai[i][k] = ai[i][p];
        ar[i][p] = tr;       ai[i][p] = tq;
      }
    }
    for (int i = 0; i < N; ++i) {
      di()(b)(i) = S((SR)ar[i][i], (SR)ai[i][i]);
      for (int j = i + 1; j < N; ++j) ti()(b)(CH::triangle_index(i, j)) = S((SR)ar[i][j], (SR)ai[i][j]);
    }
  }
}

// Site dispatch: under SIMT (device pass) coalescedRead hands each thread its lane's scalar
// object; on the host pass (and on CPU builds) the full SIMD object, inverted lane by lane.
template <class Impl, class objD, class objT>
accelerator_inline void FastImportInvertSite(const objD &d, const objT &t, objD &di, objT &ti) {
#ifdef GRID_SIMT
  FastImportInvertLane<Impl>(d, t, di, ti);
#else
  for (int l = 0; l < objD::Nsimd(); ++l) {
    typename objD::scalar_object ds = extractLane(l, d), dis;
    typename objT::scalar_object ts = extractLane(l, t), tis;
    FastImportInvertLane<Impl>(ds, ts, dis, tis);
    insertLane(l, di, dis);
    insertLane(l, ti, tis);
  }
#endif
}

// Device replacement for CompactWilsonCloverHelpers<Impl>::Invert (same arguments, same outputs).
template <class Impl>
void FastImportInvertCloverDevice(const typename CompactWilsonCloverTypes<Impl>::CloverDiagonalField &diagonal,
                                  const typename CompactWilsonCloverTypes<Impl>::CloverTriangleField &triangle,
                                  typename CompactWilsonCloverTypes<Impl>::CloverDiagonalField &diagonalInv,
                                  typename CompactWilsonCloverTypes<Impl>::CloverTriangleField &triangleInv) {
  conformable(diagonal, diagonalInv);
  conformable(triangle, triangleInv);
  conformable(diagonal, triangle);
  diagonalInv.Checkerboard() = diagonal.Checkerboard();
  triangleInv.Checkerboard() = triangle.Checkerboard();
  autoView(d_v, diagonal, AcceleratorRead);
  autoView(t_v, triangle, AcceleratorRead);
  autoView(di_v, diagonalInv, AcceleratorWrite);
  autoView(ti_v, triangleInv, AcceleratorWrite);
  typedef typename Impl::Simd Simd;
  accelerator_for(ss, diagonal.Grid()->oSites(), Simd::Nsimd(), {
    auto dd = coalescedRead(d_v[ss]);
    auto tt = coalescedRead(t_v[ss]);
    decltype(dd) ddi;
    decltype(tt) tti;
    FastImportInvertSite<Impl>(dd, tt, ddi, tti);
    coalescedWrite(di_v[ss], ddi);
    coalescedWrite(ti_v[ss], tti);
  });
}

// ---- the import itself ----
template <class Impl>
void FastImportGaugeCompact(CompactWilsonCloverFermion<Impl, CompactCloverHelpers<Impl>> &op,
                            const typename Impl::GaugeField &U, bool share, bool gpuinv, bool timers) {
  typedef CompactWilsonCloverFermion<Impl, CompactCloverHelpers<Impl>> Op;
  typedef WilsonCloverHelpers<Impl> Helpers;
  typedef CompactWilsonCloverHelpers<Impl> CompactHelpers;
  typedef CompactCloverHelpers<Impl> CloverHelpers;
  typedef typename Impl::GaugeLinkField GaugeLinkField;
  typedef typename Op::CloverField CloverField;

  // Stock calls ModifyBoundaries when fixedBoundaries (open BC); not reproduced, not ours.
  GRID_ASSERT(!op.fixedBoundaries);

  double t0 = usecond();
  // Doubled-link store: the base-class import by qualified name (no virtual dispatch).
  WilsonFermion<Impl> &base = op;
  base.WilsonFermion<Impl>::ImportGauge(U);
  if (timers) accelerator_barrier();
  double t1 = usecond();

  conformable(U.Grid(), op.GaugeGrid());
  GridBase *grid = U.Grid();

  // Field strength: shared (cache) or per import (stock).
  bool hit = false;
  std::unique_ptr<GaugeLinkField> own[6];
  const GaugeLinkField *F[6];
  if (share) {
    SharedFieldStrength<Impl>(U, F, hit);
  } else {
    GaugeLinkField *Fw[6];
    for (int i = 0; i < 6; ++i) {
      own[i].reset(new GaugeLinkField(grid));
      Fw[i] = own[i].get();
    }
    FastImportFieldStrength<Impl>(Fw, U);
    for (int i = 0; i < 6; ++i) F[i] = own[i].get();
  }
  if (timers) accelerator_barrier();
  double t2 = usecond();

  // Clover term, exactly the stock expressions (csw_r, csw_t already carry the 0.5).
  CloverField TmpOriginal(grid);
  CloverField TmpInverse(grid);
  TmpOriginal  = Helpers::fillCloverYZ(*F[0]) * op.csw_r;
  TmpOriginal += Helpers::fillCloverXZ(*F[1]) * op.csw_r;
  TmpOriginal += Helpers::fillCloverXY(*F[2]) * op.csw_r;
  TmpOriginal += Helpers::fillCloverXT(*F[3]) * op.csw_t;
  TmpOriginal += Helpers::fillCloverYT(*F[4]) * op.csw_t;
  TmpOriginal += Helpers::fillCloverZT(*F[5]) * op.csw_t;
  if (timers) accelerator_barrier();
  double t3 = usecond();

  CloverHelpers::InstantiateClover(TmpOriginal, TmpInverse, op.csw_t, op.diag_mass);
  CompactHelpers::ConvertLayout(TmpOriginal, op.Diagonal, op.Triangle);
  if (timers) accelerator_barrier();
  double t4 = usecond();

  if (gpuinv) {
    FastImportInvertCloverDevice<Impl>(op.Diagonal, op.Triangle, op.DiagonalInv, op.TriangleInv);
  } else {
    CloverHelpers::InvertClover(TmpInverse, op.Diagonal, op.Triangle, op.DiagonalInv, op.TriangleInv,
                                op.fixedBoundaries);
  }
  if (timers) accelerator_barrier();
  double t5 = usecond();

  if (gpuinv) {
    // Device-resident import: the eight checkerboard copies on the device as well. The stock
    // pickCheckerboard is a host thread_for (CpuWrite/CpuRead views, Lattice_transfer.h); after the
    // device inversion it would force a device->host copy of the four full fields here and a
    // host->device copy of the eight half fields at their next device use.
    // acceleratorPickCheckerboard (same header) is the same element copy on the device: exact.
    acceleratorPickCheckerboard(Even, op.DiagonalEven,    op.Diagonal);
    acceleratorPickCheckerboard(Even, op.TriangleEven,    op.Triangle);
    acceleratorPickCheckerboard(Odd,  op.DiagonalOdd,     op.Diagonal);
    acceleratorPickCheckerboard(Odd,  op.TriangleOdd,     op.Triangle);
    acceleratorPickCheckerboard(Even, op.DiagonalInvEven, op.DiagonalInv);
    acceleratorPickCheckerboard(Even, op.TriangleInvEven, op.TriangleInv);
    acceleratorPickCheckerboard(Odd,  op.DiagonalInvOdd,  op.DiagonalInv);
    acceleratorPickCheckerboard(Odd,  op.TriangleInvOdd,  op.TriangleInv);
  } else {
    pickCheckerboard(Even, op.DiagonalEven,    op.Diagonal);
    pickCheckerboard(Even, op.TriangleEven,    op.Triangle);
    pickCheckerboard(Odd,  op.DiagonalOdd,     op.Diagonal);
    pickCheckerboard(Odd,  op.TriangleOdd,     op.Triangle);
    pickCheckerboard(Even, op.DiagonalInvEven, op.DiagonalInv);
    pickCheckerboard(Even, op.TriangleInvEven, op.TriangleInv);
    pickCheckerboard(Odd,  op.DiagonalInvOdd,  op.DiagonalInv);
    pickCheckerboard(Odd,  op.TriangleInvOdd,  op.TriangleInv);
  }
  if (timers) {
    accelerator_barrier();
    double t6 = usecond();
    static long n = 0;
    static double cum = 0.0;
    ++n;
    cum += (t6 - t0) / 1.0e6;
    if (grid->IsBoss()) {
      // Formatted in a local stream: std::cout's flags and precision stay as the driver set them.
      const bool dp = std::is_same<Impl, WilsonImplD>::value;
      std::ostringstream s;
      s << std::fixed << std::setprecision(6) << "ImportTimer fast prec=" << (dp ? "D" : "F")
        << " fs=" << (share ? (hit ? "hit" : "miss") : "own") << " inv=" << (gpuinv ? "gpu" : "host")
        << " wilson_s=" << (t1 - t0) / 1e6 << " fieldstrength_s=" << (t2 - t1) / 1e6
        << " fill_s=" << (t3 - t2) / 1e6 << " inst_convert_s=" << (t4 - t3) / 1e6
        << " invert_s=" << (t5 - t4) / 1e6 << " pick_s=" << (t6 - t5) / 1e6
        << " total_s=" << (t6 - t0) / 1e6 << " n=" << n << " cum_total_s=" << cum;
      std::cout << GridLogMessage << s.str() << std::endl;
    }
  }
}

template <class Op, bool ok = FastImportSupported<Op>::value>
struct FastImportDispatch {
  template <class GF>
  static void run(Op &, const GF &, bool, bool, bool) {
    std::cout << GridLogError << "FastImportGauge: unsupported operator type; HASEN_GRID_SHARE_FIELDSTRENGTH"
              << " and HASEN_GRID_GPU_CLOVER_INV support only CompactWilsonCloverFermion<WilsonImplD|F,"
              << " CompactCloverHelpers>" << std::endl;
    GRID_ASSERT(0);
  }
};
template <class Op>
struct FastImportDispatch<Op, true> {
  template <class GF>
  static void run(Op &op, const GF &U, bool share, bool gpuinv, bool timers) {
    FastImportGaugeCompact(op, U, share, gpuinv, timers);
  }
};

// Explicit switches (unit test); FastImportGauge reads them from the environment.
template <class Op, class GF>
inline void FastImportGaugeWith(Op &op, const GF &U, bool share, bool gpuinv, bool timers) {
  FastImportDispatch<Op>::run(op, U, share, gpuinv, timers);
}
template <class Op, class GF>
inline void FastImportGauge(Op &op, const GF &U) {
  FastImportGaugeWith(op, U, ShareFieldStrengthEnabled(), GpuCloverInvEnabled(), ImportTimersEnabled());
}

// What GuardedImportGauge calls for a performed import. Both gates off: the stock op.ImportGauge
// (timed as a whole under HASEN_GRID_IMPORT_TIMERS=1).
template <class Op, class GF>
inline void ImportGaugeDispatch(Op &op, const GF &U) {
  if (FastImportEnabled()) {
    FastImportGauge(op, U);
    return;
  }
  if (!ImportTimersEnabled()) {
    op.ImportGauge(U);
    return;
  }
  accelerator_barrier();
  const double t0 = usecond();
  op.ImportGauge(U);
  accelerator_barrier();
  const double t1 = usecond();
  static long n = 0;
  static double cum = 0.0;
  ++n;
  cum += (t1 - t0) / 1.0e6;
  if (U.Grid()->IsBoss()) {
    const bool dp = sizeof(typename GF::scalar_type) == sizeof(ComplexD);
    std::ostringstream s;
    s << std::fixed << std::setprecision(6) << "ImportTimer stock prec=" << (dp ? "D" : "F")
      << " total_s=" << (t1 - t0) / 1e6 << " n=" << n << " cum_total_s=" << cum;
    std::cout << GridLogMessage << s.str() << std::endl;
  }
}

}  // namespace Grid
