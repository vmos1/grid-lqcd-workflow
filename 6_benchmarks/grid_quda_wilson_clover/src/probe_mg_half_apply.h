// probe_mg_half_apply.h
//
// fp16-STORAGE coarse apply for the level-1 operator of the Grid MG probe (prototype).
//
// WHY. At the production volume the level-1 apply is Grid's old CoarsenedMatrix::M
// (CoarsenedMatrix.h:112-169, reached through ProbeMG::StencilCoarseApply). It streams the 9
// link fields, 41472 sites x 9 x 48 x 48 complex fp32 = 6.9 GB per apply over 16 GPUs, is
// memory-bound (~0.57 ms) and runs ~42 times per outer iteration. QUDA stores its coarse links
// in 16 bits. Halving the link bytes is therefore the one change that attacks the bound itself.
//
// WHAT. HalfStencilCoarseApply keeps a DEVICE copy of the 9 link fields in IEEE binary16 and
// runs a kernel that is CoarsenedMatrix::M line for line (same HaloExchange + SimpleCompressor,
// same GetEntry / coalescedReadPermute / CommBuf neighbour read, same point-then-bb
// accumulation order, same fp32 complex multiply-add), except that each link element is read
// as fp16 and widened to fp32 before the multiply. Only STORAGE is half; arithmetic is fp32.
// Grid has no fp16 arithmetic; the conversions are CUDA's __float2half_rn / __half2float on
// device, with Grid's sfw_float_to_half / sfw_half_to_float (Grid_vector_types.h:43-104) on
// the host pass of the lambdas (compiled, never executed, in a GPU build).
//
// LAYOUT. One 32-bit word per complex link element: binary16 real part in the low 16 bits,
// imaginary part in the high 16 bits (the same byte order as CUDA's __half2 {x, y}). Words are
// ordered
//
//     links[point][ss][bb][b][lane]        (lane fastest)
//
// where ss is the outer (SIMD-vector) site, b the output basis index, bb the input basis index,
// lane the SIMD lane, and the element is lane `lane` of A[point][ss](b,bb). WHY THIS ORDER:
// accelerator_for(sss, oSites*nbasis, Nsimd, ...) launches blocks of (Nsimd, acceleratorThreads)
// threads with threadIdx.x = lane and consecutive threadIdx.y = consecutive sss = consecutive b
// at fixed ss (nbasis is a multiple of 32/Nsimd). A warp therefore holds 32/Nsimd consecutive b
// times all Nsimd lanes; at each (point, bb) iteration it reads 32 CONSECUTIVE words (128 B,
// one full line). Grid's own fp32 layout has lanes innermost too but b outside bb, so a warp
// reads 32/Nsimd separate 64 B segments per iteration.
//
// USE. Construct from the CoarsenedMatrix that already holds the fp32 links, e.g. the `op`
// inside StencilCoarseApply:  HalfStencilCoarseApply<F,C,n> H(*stencil.op, stencil.shift);
// The CoarsenedMatrix must outlive this object (its Stencil does the halo exchange). The fp32
// links are left untouched; freeing them to recover memory is the caller's decision.
//
// GATE. Agreement with CoarsenedMatrix::M on a random vector at the fp16 rounding level
// (~3e-4 relative for O(1) links), and bit-level agreement (~1e-7) with M applied to links that
// were rounded through fp16 first -- the latter isolates indexing from precision. See
// probe_half_apply_selftest.cc.

#pragma once

#include <Grid/Grid.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace ProbeMG {

using namespace Grid;

// ---------------------------------------------------------------------------
// fp16 helpers. Non-template, host+device. The #ifdef lives HERE, never inside an
// accelerator_for body (a preprocessor directive inside a macro argument is undefined).
// ---------------------------------------------------------------------------
accelerator_inline uint32_t half_pack_pair(float re, float im)
{
#ifdef __CUDA_ARCH__
  const uint32_t lo = (uint32_t)__half_as_ushort(__float2half_rn(re));
  const uint32_t hi = (uint32_t)__half_as_ushort(__float2half_rn(im));
#else
  const uint32_t lo = (uint32_t)::sfw_float_to_half(re).x;
  const uint32_t hi = (uint32_t)::sfw_float_to_half(im).x;
#endif
  return lo | (hi << 16);
}

accelerator_inline void half_unpack_pair(uint32_t w, float &re, float &im)
{
#ifdef __CUDA_ARCH__
  re = __half2float(__ushort_as_half((unsigned short)(w & 0xffffu)));
  im = __half2float(__ushort_as_half((unsigned short)(w >> 16)));
#else
  re = ::sfw_half_to_float(::Grid_half((uint16_t)(w & 0xffffu)));
  im = ::sfw_half_to_float(::Grid_half((uint16_t)(w >> 16)));
#endif
}

// float -> nearest binary16 -> float. Used by the self-test to build the rounded-link reference.
accelerator_inline float half_round_float(float x)
{
  float re, im;
  half_unpack_pair(half_pack_pair(x, 0.0f), re, im);
  return re;
}

// Real/imag of one extracted link element. On the DEVICE pass coalescedRead returns the scalar
// object and TensorRemove gives ComplexF, which selects the first overload. The HOST pass of a
// GPU build compiles the same lambda with coalescedRead returning the SIMD vector; that body
// never runs, so the template fallback only has to compile. HalfStencilCoarseApply
// static_asserts that the scalar type is ComplexF, so the fallback cannot be hit on device.
accelerator_inline void half_complex_parts(const ComplexF &z, float &re, float &im)
{
  re = z.real();
  im = z.imag();
}
template <class T>
accelerator_inline void half_complex_parts(const T &, float &re, float &im)
{
  re = 0.0f;
  im = 0.0f;
}

// ---------------------------------------------------------------------------
template <class Fobj, class CComplex, int nbasis>
class HalfStencilCoarseApply : public LinearOperatorBase<Lattice<iVector<CComplex, nbasis>>> {
public:
  typedef CoarsenedMatrix<Fobj, CComplex, nbasis> OldOp;
  typedef iVector<CComplex, nbasis> siteVector;
  typedef Lattice<siteVector> CoarseVector;
  typedef typename CComplex::scalar_type ScalarC;

  static_assert(std::is_same<ScalarC, ComplexF>::value,
                "HalfStencilCoarseApply: fp32 coarse type (vTComplexF) only");
  static_assert(sizeof(CComplex) == CComplex::Nsimd() * sizeof(ComplexF),
                "HalfStencilCoarseApply: CComplex must be one SIMD word of ComplexF lanes");

  OldOp &op;          // owns the Stencil (halo exchange) and the fp32 links A[point]
  RealD shift = 0.0;  // out = M in + shift * in, as StencilCoarseApply
  GridBase *grid = nullptr;
  int npoint = 0;
  uint64_t osites = 0;
  deviceVector<uint32_t> links;  // [point][ss][bb][b][lane], one packed binary16 pair per word
  float max_abs = 0.0f;          // global max |re|,|im| over all fp32 links at import

  HalfStencilCoarseApply(OldOp &op_, RealD shift_ = 0.0) : op(op_), shift(shift_)
  {
    grid = op.Grid();
    npoint = op.geom.npoint;
    osites = grid->oSites();
    GRID_ASSERT(npoint == 9);  // hops==1 CoarsenedMatrix
    GRID_ASSERT((int)op.A.size() == npoint);
    // No device lambda may sit in a constructor (CUDA: "must allow its address to be taken"),
    // so the conversion kernel is a separate member function.
    ImportLinks();
  }

  // Words (= complex elements) per point and total bytes of the fp16 link store (this rank).
  uint64_t words_per_point() const { return osites * nbasis * nbasis * (uint64_t)CComplex::Nsimd(); }
  uint64_t bytes() const { return (uint64_t)npoint * words_per_point() * sizeof(uint32_t); }

  // (Re)convert op.A into the fp16 store. Call again if op.A changes.
  void ImportLinks()
  {
    const int Nsimd = CComplex::Nsimd();
    const uint64_t nsites = osites;
    const uint64_t per_point = words_per_point();
    links.resize((uint64_t)npoint * per_point);

    // Per-thread partial maxima -> host -> max. Only used to detect fp16 overflow.
    const uint64_t npartial = nsites * nbasis * (uint64_t)Nsimd;
    deviceVector<float> partial(npartial);
    std::vector<float> hpartial(npartial);
    float *partial_p = &partial[0];
    float mx_all = 0.0f;

    for (int point = 0; point < npoint; ++point) {
      autoView(A_v, op.A[point], AcceleratorRead);
      uint32_t *dst = &links[0] + (uint64_t)point * per_point;
      accelerator_for(sss, nsites * nbasis, Nsimd, {
        const uint64_t ss = sss / nbasis;
        const int b = (int)(sss % nbasis);
        const int lane = acceleratorSIMTlane(Nsimd);
        float mx = 0.0f;
        for (int bb = 0; bb < nbasis; bb++) {
          float re, im;
          half_complex_parts(TensorRemove(coalescedRead(A_v[ss](b, bb))), re, im);
          const float ar = re < 0.0f ? -re : re;
          const float ai = im < 0.0f ? -im : im;
          mx = ar > mx ? ar : mx;
          mx = ai > mx ? ai : mx;
          dst[((ss * nbasis + bb) * nbasis + b) * Nsimd + lane] = half_pack_pair(re, im);
        }
        partial_p[sss * Nsimd + lane] = mx;
      });
      acceleratorCopyFromDevice(partial_p, &hpartial[0], npartial * sizeof(float));
      for (uint64_t i = 0; i < npartial; ++i) mx_all = hpartial[i] > mx_all ? hpartial[i] : mx_all;
    }
    RealF g = mx_all;
    grid->GlobalMax(g);
    max_abs = g;
    // binary16 max finite is 65504; anything above converts to Inf and poisons the solve.
    if (!(max_abs < 65504.0f)) {
      std::cout << GridLogError << "HalfStencilCoarseApply: link max |value| " << max_abs
                << " exceeds the binary16 range" << std::endl;
      GRID_ASSERT(0);
    }
  }

  void Op(const CoarseVector &in, CoarseVector &out) override
  {
    conformable(grid, in.Grid());
    conformable(in.Grid(), out.Grid());
    out.Checkerboard() = in.Checkerboard();

    SimpleCompressor<siteVector> compressor;
    op.Stencil.HaloExchange(in, compressor);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    autoView(Stencil_v, op.Stencil, AcceleratorRead);

    // Locals only inside the kernel: a [=] device lambda must not capture `this`.
    const int np = npoint;
    const int Nsimd = CComplex::Nsimd();
    const uint64_t nsites = osites;
    const uint64_t site_stride = (uint64_t)nbasis * nbasis * Nsimd;  // one (point, ss) block
    const uint64_t bb_stride = (uint64_t)nbasis * Nsimd;
    const uint32_t *links_p = &links[0];

    typedef decltype(coalescedRead(in_v[0])) calcVector;
    typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

    accelerator_for(sss, nsites * nbasis, Nsimd, {
      const uint64_t ss = sss / nbasis;
      const int b = (int)(sss % nbasis);
      const int lane = acceleratorSIMTlane(Nsimd);
      calcComplex res = Zero();
      calcVector nbr;
      int ptype;
      StencilEntry *SE;

      for (int point = 0; point < np; point++) {
        SE = Stencil_v.GetEntry(ptype, point, ss);

        if (SE->_is_local) {
          nbr = coalescedReadPermute(in_v[SE->_offset], ptype, SE->_permute);
        } else {
          nbr = coalescedRead(Stencil_v.CommBuf()[SE->_offset]);
        }
        acceleratorSynchronise();

        const uint32_t *lp =
            links_p + ((uint64_t)point * nsites + ss) * site_stride + (uint64_t)(b * Nsimd + lane);
        for (int bb = 0; bb < nbasis; bb++) {
          float re, im;
          half_unpack_pair(lp[bb * bb_stride], re, im);
          const calcComplex a = ScalarC(re, im);  // same element type M multiplies with
          res = res + a * nbr(bb);
        }
      }
      coalescedWrite(out_v[ss](b), res);
    });

    if (shift != 0.0) out = out + shift * in;
  }
  void AdjOp(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
  void OpDiag(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
  void OpDir(const CoarseVector &in, CoarseVector &out, int dir, int disp) override { GRID_ASSERT(0); }
  void OpDirAll(const CoarseVector &in, std::vector<CoarseVector> &out) override { GRID_ASSERT(0); }
  void HermOpAndNorm(const CoarseVector &in, CoarseVector &out, RealD &n1, RealD &n2) override { GRID_ASSERT(0); }
  void HermOp(const CoarseVector &in, CoarseVector &out) override { GRID_ASSERT(0); }
};

} // namespace ProbeMG
