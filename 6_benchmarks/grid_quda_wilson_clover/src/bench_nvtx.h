#pragma once

// Harness-level NVTX ranges, so an Nsight Systems timeline can be attributed to
// a named region (which backend, which action) rather than guessed at from
// kernel symbols and wall-clock adjacency.
//
// Compiled out entirely unless -DBENCH_NVTX is passed, so the default build is
// unchanged and the untraced binary stays available for clean timing.
//
// This is INDEPENDENT of Grid's --enable-tracing=nvtx: NVTX3 is header-only
// (CUDA >= 10), so these ranges need no Grid rebuild and no extra library.
// Note that CUDA 12.9 ships the header ONLY under nvtx3/ -- <nvToolsExt.h> at
// the include root no longer exists, which is also why Grid's own tracing
// option does not build here without adding -I${CUDA_HOME}/include/nvtx3.
//
// ⚠️ Profiler overhead is NOT backend-neutral: at C2/1 GPU, nsys inflated Grid
// by 3.5-5.2% but QUDA by only 0.9-1.7%, moving the clover ratio 1.386 -> 1.422.
// Ranges are for WITHIN-run attribution; never quote a backend ratio measured
// under a profiler.

#include <string>

#if defined(BENCH_NVTX)

#include <nvtx3/nvToolsExt.h>

namespace bench {

// RAII push/pop. Holds the name by value: nvtxRangePushA copies the string at
// push time, but keeping it alive costs nothing and removes the question.
class NvtxRange
{
public:
  explicit NvtxRange(const std::string &name) : name_(name) { nvtxRangePushA(name_.c_str()); }
  ~NvtxRange() { nvtxRangePop(); }

  NvtxRange(const NvtxRange &) = delete;
  NvtxRange &operator=(const NvtxRange &) = delete;

private:
  std::string name_;
};

}  // namespace bench

#define BENCH_NVTX_CONCAT_INNER(a, b) a##b
#define BENCH_NVTX_CONCAT(a, b) BENCH_NVTX_CONCAT_INNER(a, b)
#define BENCH_NVTX_RANGE(name) \
  ::bench::NvtxRange BENCH_NVTX_CONCAT(bench_nvtx_scope_, __LINE__)(name)

#else

#define BENCH_NVTX_RANGE(name) \
  do {                         \
  } while (0)

#endif
