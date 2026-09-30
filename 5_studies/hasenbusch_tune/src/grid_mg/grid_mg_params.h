// grid_mg_params.h  (pure-Grid HMC, M2: the `best2` multigrid as a rung solver)
//
// Every tunable of the Grid multigrid rung solver, with the `best2` recipe of the gq-mg
// campaign as the defaults (runs/2026_9_29_mg_final_bench_c3/c3_final_bench.sh, `grid=(...)`
// block; __docs/2026_09_29_grid_multigrid_experimentation.md §5), mapped from the probe's
// command-line knobs (probe_grid_mg_schur_clover.cc) as named in each comment.
// Spec: __docs/2026_09_29_pure_grid_m2_mg_solver_design.md §2 item 3.
//
// GridMGParams::from_env() is the ONLY place in the grid_mg/ code that reads the environment.
// Unset variable = the default below. A malformed value, or a GRID_MG_* variable this file
// does not know (a typo would otherwise be silently ignored), aborts with a message.
//
// Two differences from the probe's best2 row, both from the spec:
//   * the hierarchy coarsens SchurDiagMooeeOperator (the operator the HMC action solves),
//     not the probe's SCHUR=one; the One route is deferred (spec §6);
//   * the outer restart cap is 1000 GCR cycles (the benchmark ran MAXITER=200).

#pragma once

#include <Grid/Grid.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern char **environ;

namespace Grid {

// Level-1 null vectors, COMPILE-TIME: a template argument of the Aggregation and of the coarse
// matrix, so changing it needs a rebuild (-DGRID_MG_NBASIS=<n>, as the probe's PROBE_NBASIS).
// Gamma5-doubled to 2*kNbasis = 48 coarse dof per block (QUDA: n_vec 24 x spin_block_size 2).
#ifndef GRID_MG_NBASIS
#define GRID_MG_NBASIS 24
#endif
constexpr int kNbasis = GRID_MG_NBASIS;
// Level-2 null vectors, compile-time for the same reason; no gamma5 doubling below level 1.
constexpr int kNbasis2 = 32;

struct GridMGParams {
  // ---- compile-time (documented here, not settable) ----
  static constexpr int nbasis = kNbasis;    // level-1 null vectors (48 coarse dof)
  static constexpr int nbasis2 = kNbasis2;  // level-2 null vectors
  // Coarse stencil range at BOTH levels. ProbeMG::StencilCoarseApply supports hops 1 only
  // (Grid's old Geometry has no corner points); best2 runs STENCIL_HOPS=1.
  static constexpr int stencil_hops = 1;

  // ---- aggregation blocks (physical sites; block[0] must be even on the red-black grid) ----
  Coordinate block;   // GRID_MG_BLOCK   default 4.4.4.4 (probe BLOCK)
  Coordinate block2;  // GRID_MG_BLOCK2  default 2.3.3.3 (probe BLOCK2), in level-1 sites

  // ---- level-1 null vectors: GCR inverse iteration on the fp32 Schur operator ----
  RealD subspace_tol = 1.0e-3;  // GRID_MG_SUBSPACE_TOL      (SUBSPACE_TOL)
  int subspace_rounds = 3;      // GRID_MG_SUBSPACE_ROUNDS   (SUBSPACE_ROUNDS)
  int subspace_mmax = 10;       // GRID_MG_SUBSPACE_MMAX     (SUBSPACE_MMAX)
  int subspace_nstep = 10;      // GRID_MG_SUBSPACE_NSTEP    (probe passes SUBSPACE_MMAX here)
  int subspace_maxiter = 30;    // GRID_MG_SUBSPACE_MAXITER  (SUBSPACE_MAXITER)
  // ---- level-2 null vectors: same generator on the shifted level-1 stencil operator ----
  RealD l2_subspace_tol = 1.0e-3;  // GRID_MG_L2_SUBSPACE_TOL      (L2_SUBSPACE_TOL)
  int l2_subspace_rounds = 3;      // GRID_MG_L2_SUBSPACE_ROUNDS   (L2_SUBSPACE_ROUNDS)
  int l2_subspace_mmax = 10;       // GRID_MG_L2_SUBSPACE_MMAX     (probe hard-codes 10)
  int l2_subspace_nstep = 10;      // GRID_MG_L2_SUBSPACE_NSTEP    (probe hard-codes 10)
  int l2_subspace_maxiter = 10;    // GRID_MG_L2_SUBSPACE_MAXITER  (L2_SUBSPACE_MAXITER=10)

  // ---- fine smoother: FlexibleGCR on Mpc + smoother_shift ----
  RealD smoother_shift = 0.01;  // GRID_MG_SMOOTHER_SHIFT    (probe hard-codes 0.01)
  RealD smoother_tol = 0.1;     // GRID_MG_SMOOTHER_TOL      (SMOOTHER_TOL)
  int smoother_maxiter = 1;     // GRID_MG_SMOOTHER_MAXITER  (SMOOTHER_MAXITER)
  int smoother_mmax = 1;        // GRID_MG_SMOOTHER_MMAX     (SMOOTHER_MMAX)
  int smoother_nstep = 8;       // GRID_MG_SMOOTHER_NSTEP    (SMOOTHER_NSTEP)

  // ---- level 1: stencil apply of the coarse Mpc + coarse_shift, FlexibleGCR solve ----
  RealD coarse_shift = 0.001;  // GRID_MG_COARSE_SHIFT    (probe hard-codes 0.001)
  RealD coarse_tol = 0.2;      // GRID_MG_COARSE_TOL      (COARSE_TOL)
  int coarse_maxiter = 1;      // GRID_MG_COARSE_MAXITER  (COARSE_MAXITER)
  int coarse_mmax = 12;        // GRID_MG_COARSE_MMAX     (COARSE_MMAX)
  int coarse_nstep = 12;       // GRID_MG_COARSE_NSTEP    (COARSE_NSTEP)

  // ---- level 2 (preconditions the level-1 solve with a V-cycle) ----
  RealD coarse2_shift = 0.001;   // GRID_MG_COARSE2_SHIFT          (probe hard-codes 0.001)
  RealD l1_smoother_tol = 0.1;   // GRID_MG_L1_SMOOTHER_TOL        (L2_SMOOTHER_TOL)
  int l1_smoother_maxiter = 1;   // GRID_MG_L1_SMOOTHER_MAXITER    (probe hard-codes 1)
  int l1_smoother_mmax = 1;      // GRID_MG_L1_SMOOTHER_MMAX       (probe hard-codes 1)
  int l1_smoother_nstep = 2;     // GRID_MG_L1_SMOOTHER_NSTEP      (L2_SMOOTHER_NSTEP)
  RealD coarse2_tol = 0.2;       // GRID_MG_COARSE2_TOL            (L2_COARSE_TOL)
  int coarse2_maxiter = 1;       // GRID_MG_COARSE2_MAXITER        (L2_COARSE_MAXITER)
  int coarse2_mmax = 4;          // GRID_MG_COARSE2_MMAX           (L2_COARSE_MMAX)
  int coarse2_nstep = 4;         // GRID_MG_COARSE2_NSTEP          (L2_COARSE_NSTEP)

  // ---- outer: fp64 FlexibleGCR on the action's own linop; tol comes from the action ----
  int outer_maxiter = 1000;  // GRID_MG_OUTER_MAXITER  restart cycles (probe MAXITER)
  int outer_mmax = 6;        // GRID_MG_OUTER_MMAX     (OUTER_MMAX)
  int outer_nstep = 6;       // GRID_MG_OUTER_NSTEP    (OUTER_NSTEP)

  // ---- implementation switches (best2 = FAST_MG=1 FAST_PROJECT=2 PERSISTENT_PRECCHANGE=1) ----
  int project_mode = 2;              // GRID_MG_PROJECT_MODE  0 stock, 1 blockProjectFast, 2 fused
  bool persistent_temps = true;      // GRID_MG_PERSISTENT_TEMPS       V-cycle temporaries kept
  bool persistent_precchange = true; // GRID_MG_PERSISTENT_PRECCHANGE  fp64<->fp32 maps kept

  // ---- rebuild cadence (the hybrid's two tiers) ----
  // Soft: an Mpc solve needing >= threshold_count outer iterations marks a rebuild pending,
  // executed at the hierarchy's next SetGauge.
  int threshold_count = 64;    // GRID_MG_THRESHOLD_COUNT
  // Hard: a true relative residual > rsd_tol_factor * tol rebuilds at once and re-solves once;
  // still bad => GRID_ASSERT.
  RealD rsd_tol_factor = 8.0;  // GRID_MG_RSD_TOL_FACTOR

  // ---- setup RNG and logging ----
  int seed = 4711;  // GRID_MG_SEED  the setup RNGs are reseeded from it at EVERY build
  // GRID_MG_VERBOSE: 0 = rebuild notices and hard-tier events only; 1 = + one line per Mpc
  // solve and the setup timing split; 2 = + the outer GCR's per-step log; 3 = + the inner
  // solvers' per-step logs (null-vector GCR, smoothers, coarse solves; very long).
  int verbose = 1;

  GridMGParams()
      : block(std::vector<int>({4, 4, 4, 4})), block2(std::vector<int>({2, 3, 3, 3}))
  {
  }

  static std::string CoordinateString(const Coordinate &c)
  {
    std::string s;
    for (int d = 0; d < (int)c.size(); ++d) s += (d ? "." : "") + std::to_string(c[d]);
    return s;
  }

  // One line with every effective value, for the driver's [Ladder] banner and the test.
  std::string Summary() const
  {
    std::ostringstream os;
    os << "nbasis " << nbasis << " (x2 = " << 2 * nbasis << ") nbasis2 " << nbasis2 << " hops "
       << stencil_hops << " block " << CoordinateString(block) << " block2 "
       << CoordinateString(block2) << " | subspace tol/rounds/mmax/nstep/maxiter " << subspace_tol
       << "/" << subspace_rounds << "/" << subspace_mmax << "/" << subspace_nstep << "/"
       << subspace_maxiter << " l2 " << l2_subspace_tol << "/" << l2_subspace_rounds << "/"
       << l2_subspace_mmax << "/" << l2_subspace_nstep << "/" << l2_subspace_maxiter
       << " | smoother shift/tol/maxiter/mmax/nstep " << smoother_shift << "/" << smoother_tol
       << "/" << smoother_maxiter << "/" << smoother_mmax << "/" << smoother_nstep
       << " | coarse shift/tol/maxiter/mmax/nstep " << coarse_shift << "/" << coarse_tol << "/"
       << coarse_maxiter << "/" << coarse_mmax << "/" << coarse_nstep
       << " | l1-smoother tol/maxiter/mmax/nstep " << l1_smoother_tol << "/" << l1_smoother_maxiter
       << "/" << l1_smoother_mmax << "/" << l1_smoother_nstep
       << " | coarse2 shift/tol/maxiter/mmax/nstep " << coarse2_shift << "/" << coarse2_tol << "/"
       << coarse2_maxiter << "/" << coarse2_mmax << "/" << coarse2_nstep
       << " | outer maxiter/mmax/nstep " << outer_maxiter << "/" << outer_mmax << "/"
       << outer_nstep << " | project_mode " << project_mode << " persistent_temps "
       << persistent_temps << " persistent_precchange " << persistent_precchange
       << " | threshold_count " << threshold_count << " rsd_tol_factor " << rsd_tol_factor
       << " seed " << seed << " verbose " << verbose;
    return os.str();
  }

  // The single environment reader of the grid_mg/ code.
  static GridMGParams from_env()
  {
    GridMGParams p;

    auto fail = [](const std::string &name, const char *value, const std::string &why) {
      std::cout << GridLogError << "GridMGParams: " << name << "='" << value << "' " << why
                << std::endl;
      std::cerr << "GridMGParams: " << name << "='" << value << "' " << why << std::endl;
      exit(1);
    };
    auto get_real = [&](const char *name, RealD &dst) {
      const char *v = std::getenv(name);
      if (!v || !*v) return;
      errno = 0;
      char *end = nullptr;
      const double x = std::strtod(v, &end);
      if (errno != 0 || end == v || *end != '\0') fail(name, v, "is not a real number");
      dst = x;
    };
    auto get_int = [&](const char *name, int &dst) {
      const char *v = std::getenv(name);
      if (!v || !*v) return;
      errno = 0;
      char *end = nullptr;
      const long x = std::strtol(v, &end, 10);
      if (errno != 0 || end == v || *end != '\0') fail(name, v, "is not an integer");
      dst = (int)x;
    };
    auto get_bool = [&](const char *name, bool &dst) {
      int x = dst ? 1 : 0;
      get_int(name, x);
      if (x != 0 && x != 1) fail(name, std::getenv(name), "must be 0 or 1");
      dst = (x != 0);
    };
    auto get_block = [&](const char *name, Coordinate &dst) {
      const char *v = std::getenv(name);
      if (!v || !*v) return;
      std::vector<int> parts;
      std::string item;
      std::stringstream ss(v);
      while (std::getline(ss, item, '.')) {
        errno = 0;
        char *end = nullptr;
        const long x = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || errno != 0 || *end != '\0' || x < 1)
          fail(name, v, "must be four dot-separated positive integers, e.g. 4.4.4.4");
        parts.push_back((int)x);
      }
      if (parts.size() != 4) fail(name, v, "must be four dot-separated positive integers");
      dst = Coordinate(parts);
    };

    // Every variable this function understands. Keep in step with the reads below.
    static const char *known[] = {
        "GRID_MG_BLOCK", "GRID_MG_BLOCK2",
        "GRID_MG_SUBSPACE_TOL", "GRID_MG_SUBSPACE_ROUNDS", "GRID_MG_SUBSPACE_MMAX",
        "GRID_MG_SUBSPACE_NSTEP", "GRID_MG_SUBSPACE_MAXITER",
        "GRID_MG_L2_SUBSPACE_TOL", "GRID_MG_L2_SUBSPACE_ROUNDS", "GRID_MG_L2_SUBSPACE_MMAX",
        "GRID_MG_L2_SUBSPACE_NSTEP", "GRID_MG_L2_SUBSPACE_MAXITER",
        "GRID_MG_SMOOTHER_SHIFT", "GRID_MG_SMOOTHER_TOL", "GRID_MG_SMOOTHER_MAXITER",
        "GRID_MG_SMOOTHER_MMAX", "GRID_MG_SMOOTHER_NSTEP",
        "GRID_MG_COARSE_SHIFT", "GRID_MG_COARSE_TOL", "GRID_MG_COARSE_MAXITER",
        "GRID_MG_COARSE_MMAX", "GRID_MG_COARSE_NSTEP",
        "GRID_MG_COARSE2_SHIFT", "GRID_MG_L1_SMOOTHER_TOL", "GRID_MG_L1_SMOOTHER_MAXITER",
        "GRID_MG_L1_SMOOTHER_MMAX", "GRID_MG_L1_SMOOTHER_NSTEP",
        "GRID_MG_COARSE2_TOL", "GRID_MG_COARSE2_MAXITER", "GRID_MG_COARSE2_MMAX",
        "GRID_MG_COARSE2_NSTEP",
        "GRID_MG_OUTER_MAXITER", "GRID_MG_OUTER_MMAX", "GRID_MG_OUTER_NSTEP",
        "GRID_MG_PROJECT_MODE", "GRID_MG_PERSISTENT_TEMPS", "GRID_MG_PERSISTENT_PRECCHANGE",
        "GRID_MG_THRESHOLD_COUNT", "GRID_MG_RSD_TOL_FACTOR", "GRID_MG_SEED", "GRID_MG_VERBOSE"};
    for (char **e = environ; e && *e; ++e) {
      if (std::strncmp(*e, "GRID_MG_", 8) != 0) continue;
      const char *eq = std::strchr(*e, '=');
      const std::string name = eq ? std::string(*e, eq - *e) : std::string(*e);
      bool ok = false;
      for (const char *k : known) ok = ok || (name == k);
      if (!ok) fail(name, eq ? eq + 1 : "", "is not a GridMGParams variable (typo?)");
    }

    get_block("GRID_MG_BLOCK", p.block);
    get_block("GRID_MG_BLOCK2", p.block2);
    get_real("GRID_MG_SUBSPACE_TOL", p.subspace_tol);
    get_int("GRID_MG_SUBSPACE_ROUNDS", p.subspace_rounds);
    get_int("GRID_MG_SUBSPACE_MMAX", p.subspace_mmax);
    get_int("GRID_MG_SUBSPACE_NSTEP", p.subspace_nstep);
    get_int("GRID_MG_SUBSPACE_MAXITER", p.subspace_maxiter);
    get_real("GRID_MG_L2_SUBSPACE_TOL", p.l2_subspace_tol);
    get_int("GRID_MG_L2_SUBSPACE_ROUNDS", p.l2_subspace_rounds);
    get_int("GRID_MG_L2_SUBSPACE_MMAX", p.l2_subspace_mmax);
    get_int("GRID_MG_L2_SUBSPACE_NSTEP", p.l2_subspace_nstep);
    get_int("GRID_MG_L2_SUBSPACE_MAXITER", p.l2_subspace_maxiter);
    get_real("GRID_MG_SMOOTHER_SHIFT", p.smoother_shift);
    get_real("GRID_MG_SMOOTHER_TOL", p.smoother_tol);
    get_int("GRID_MG_SMOOTHER_MAXITER", p.smoother_maxiter);
    get_int("GRID_MG_SMOOTHER_MMAX", p.smoother_mmax);
    get_int("GRID_MG_SMOOTHER_NSTEP", p.smoother_nstep);
    get_real("GRID_MG_COARSE_SHIFT", p.coarse_shift);
    get_real("GRID_MG_COARSE_TOL", p.coarse_tol);
    get_int("GRID_MG_COARSE_MAXITER", p.coarse_maxiter);
    get_int("GRID_MG_COARSE_MMAX", p.coarse_mmax);
    get_int("GRID_MG_COARSE_NSTEP", p.coarse_nstep);
    get_real("GRID_MG_COARSE2_SHIFT", p.coarse2_shift);
    get_real("GRID_MG_L1_SMOOTHER_TOL", p.l1_smoother_tol);
    get_int("GRID_MG_L1_SMOOTHER_MAXITER", p.l1_smoother_maxiter);
    get_int("GRID_MG_L1_SMOOTHER_MMAX", p.l1_smoother_mmax);
    get_int("GRID_MG_L1_SMOOTHER_NSTEP", p.l1_smoother_nstep);
    get_real("GRID_MG_COARSE2_TOL", p.coarse2_tol);
    get_int("GRID_MG_COARSE2_MAXITER", p.coarse2_maxiter);
    get_int("GRID_MG_COARSE2_MMAX", p.coarse2_mmax);
    get_int("GRID_MG_COARSE2_NSTEP", p.coarse2_nstep);
    get_int("GRID_MG_OUTER_MAXITER", p.outer_maxiter);
    get_int("GRID_MG_OUTER_MMAX", p.outer_mmax);
    get_int("GRID_MG_OUTER_NSTEP", p.outer_nstep);
    get_int("GRID_MG_PROJECT_MODE", p.project_mode);
    get_bool("GRID_MG_PERSISTENT_TEMPS", p.persistent_temps);
    get_bool("GRID_MG_PERSISTENT_PRECCHANGE", p.persistent_precchange);
    get_int("GRID_MG_THRESHOLD_COUNT", p.threshold_count);
    get_real("GRID_MG_RSD_TOL_FACTOR", p.rsd_tol_factor);
    get_int("GRID_MG_SEED", p.seed);
    get_int("GRID_MG_VERBOSE", p.verbose);

    // Sanity: values that would hang or silently misbehave.
    auto positive = [&](const char *name, long x) {
      if (x < 1) fail(name, std::to_string(x).c_str(), "must be >= 1");
    };
    positive("GRID_MG_SUBSPACE_ROUNDS", p.subspace_rounds);
    positive("GRID_MG_SUBSPACE_MMAX", p.subspace_mmax);
    positive("GRID_MG_SUBSPACE_NSTEP", p.subspace_nstep);
    positive("GRID_MG_SUBSPACE_MAXITER", p.subspace_maxiter);
    positive("GRID_MG_L2_SUBSPACE_ROUNDS", p.l2_subspace_rounds);
    positive("GRID_MG_L2_SUBSPACE_MMAX", p.l2_subspace_mmax);
    positive("GRID_MG_L2_SUBSPACE_NSTEP", p.l2_subspace_nstep);
    positive("GRID_MG_L2_SUBSPACE_MAXITER", p.l2_subspace_maxiter);
    positive("GRID_MG_SMOOTHER_MAXITER", p.smoother_maxiter);
    positive("GRID_MG_SMOOTHER_MMAX", p.smoother_mmax);
    positive("GRID_MG_SMOOTHER_NSTEP", p.smoother_nstep);
    positive("GRID_MG_COARSE_MAXITER", p.coarse_maxiter);
    positive("GRID_MG_COARSE_MMAX", p.coarse_mmax);
    positive("GRID_MG_COARSE_NSTEP", p.coarse_nstep);
    positive("GRID_MG_L1_SMOOTHER_MAXITER", p.l1_smoother_maxiter);
    positive("GRID_MG_L1_SMOOTHER_MMAX", p.l1_smoother_mmax);
    positive("GRID_MG_L1_SMOOTHER_NSTEP", p.l1_smoother_nstep);
    positive("GRID_MG_COARSE2_MAXITER", p.coarse2_maxiter);
    positive("GRID_MG_COARSE2_MMAX", p.coarse2_mmax);
    positive("GRID_MG_COARSE2_NSTEP", p.coarse2_nstep);
    positive("GRID_MG_OUTER_MAXITER", p.outer_maxiter);
    positive("GRID_MG_OUTER_MMAX", p.outer_mmax);
    positive("GRID_MG_OUTER_NSTEP", p.outer_nstep);
    positive("GRID_MG_THRESHOLD_COUNT", p.threshold_count);
    if (p.project_mode < 0 || p.project_mode > 2)
      fail("GRID_MG_PROJECT_MODE", std::to_string(p.project_mode).c_str(), "must be 0, 1 or 2");
    if (!(p.rsd_tol_factor >= 1.0))
      fail("GRID_MG_RSD_TOL_FACTOR", std::to_string(p.rsd_tol_factor).c_str(), "must be >= 1");
    if (p.block[0] % 2 != 0)
      fail("GRID_MG_BLOCK", CoordinateString(p.block).c_str(),
           "needs an even x block (the red-black grid halves x)");
    return p;
  }
};

}  // namespace Grid
