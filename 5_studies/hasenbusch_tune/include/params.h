#pragma once
#include <Grid/Grid.h>
#include <cstdlib>
#include <random>

using namespace Grid;

namespace TXQCDProduction {

namespace detail {
inline RealD env_real(const char *name, RealD def) {
  if (const char *s = std::getenv(name); s && *s) return std::atof(s);
  return def;
}
inline int env_int(const char *name, int def) {
  if (const char *s = std::getenv(name); s && *s) return std::atoi(s);
  return def;
}
inline Coordinate env_latt(const char *name,
                            const std::vector<int> &def) {
  const char *s = std::getenv(name);
  if (!s || !*s) return Coordinate(def);
  std::vector<int> v; std::string t;
  for (const char *p = s; *p; ++p) {
    if (*p == '.') { if (!t.empty()) v.push_back(std::atoi(t.c_str())); t.clear(); }
    else t.push_back(*p);
  }
  if (!t.empty()) v.push_back(std::atoi(t.c_str()));
  if ((int)v.size() != Nd) return Coordinate(def);
  return Coordinate(v);
}
}

// ===== Lattice geometry (override: LATT=L.L.L.T) =====
inline Coordinate lattice_size() {
  return detail::env_latt("LATT", std::vector<int>{16, 16, 16, 48});
}

// Runtime-overridable physics params (env var in parentheses).  Defaults are
// production values; set the env var to test small-lattice / bisection points.
inline const RealD mass_light   = detail::env_real("MASS_LIGHT",   -0.2450); // MASS_LIGHT
inline const RealD mass_strange = detail::env_real("MASS_STRANGE", -0.2450); // MASS_STRANGE
inline const RealD csw          = detail::env_real("CSW",           1.24930970916466); // CSW

// ===== Gauge action =====
inline const RealD beta   = detail::env_real("BETA",   6.1);               // BETA
inline const RealD u0     = detail::env_real("U0",     0.832605301399891); // U0
inline const RealD lambda = detail::env_real("LAMBDA", 0.5);               // LAMBDA

// ===== Stout smearing for inversions =====
inline const RealD stout_rho_inv   = detail::env_real("STOUT_RHO",    0.125); // STOUT_RHO
inline const int   stout_nsmear_inv = detail::env_int("STOUT_NSMEAR", 1);      // STOUT_NSMEAR

// ===== Stout smearing for source/sink construction =====
constexpr RealD stout_rho_src = 0.16;
constexpr int   stout_nsmear_src = 3;

// ===== Gaussian source/sink smearing =====
constexpr RealD gauss_width = 2.1;
constexpr int   gauss_niter = 20;

// ===== Connected source grid =====
//
// Defaults: 4 spatial × 12 temporal per dim → 4³×12 = 768 sources.  For quick
// per-cfg comparisons (basin diagnostics, λ sweeps), set MEAS_SPACE_SRC and
// MEAS_TIME_SRC to small values (e.g. 1×2 = 2 sources, ~30× faster).
inline int space_src_per_dim_runtime() {
  return detail::env_int("MEAS_SPACE_SRC", 4);
}
inline int time_src_per_dim_runtime() {
  return detail::env_int("MEAS_TIME_SRC", 12);
}
constexpr int space_src_per_dim = 4;   // kept for BWC; runtime accessor preferred
constexpr int time_src_per_dim  = 12;
// Per-cfg deterministic source-grid origin shift (active cfg >= 1000 to keep
// older data matched).  Reduces connected measurement autocorrelation by
// breaking the fixed-source-pattern noise component that locks in across
// consecutive cfgs.  At λ=5 16³×48, measured ρ(1) reduction ~50% on the
// connected pion (from per-src vs cfg-avg ρ(1) split).  Shift is deterministic:
// seeded by cfg number so reruns are reproducible.  Written to HDF5 in conn
// as 'src_shift' (4-element int vector).  Disco does not use this — its Z2
// noise sources are translation-invariant in expectation.  For cfg < 1000
// returns the legacy (0,0,0,0) origin.
inline Coordinate src_grid_origin(int traj) {
  Coordinate latt = lattice_size();
  // SHIFT_ALL_CFGS=1 forces deterministic per-cfg translation averaging for
  // every cfg regardless of cfg number — used for ensembles like b6.5 whose
  // cfg numbering starts low (~10) but where we still want translation
  // averaging from the start.  Default: legacy gate on traj < 1000.
  bool shift_all = false;
  if (const char *e = std::getenv("SHIFT_ALL_CFGS");
      e && *e && !(e[0] == '0' && e[1] == '\0'))
    shift_all = true;
  if (!shift_all && traj < 1000) {
    return Coordinate(std::vector<int>{0, 0, 0, 0});
  }
  // Deterministic per-cfg shift, seeded by cfg number.
  std::mt19937_64 rng(static_cast<uint64_t>(traj));
  rng.discard(8);  // burn in
  Coordinate s(Nd);
  for (int d = 0; d < Nd; ++d) {
    std::uniform_int_distribution<int> dist(0, latt[d] - 1);
    s[d] = dist(rng);
  }
  return s;
}

// ===== VEV monitoring =====
constexpr int n_vev_noise = 8;

// ===== Disconnected =====
constexpr int n_noise_disco = 32;

// ===== HMC =====
constexpr int n_therm = 100;
constexpr int n_prod = 1000;
// meas_skip is the checkpoint-save cadence (used in gen_*_cfgs.cc via
// CheckpointerParameters::saveInterval).  Env-overridable as N_SKIP so
// short runs can save every traj (N_SKIP=1).  Default 10 for production.
inline int meas_skip_runtime() { return detail::env_int("N_SKIP", 10); }
#define meas_skip (TXQCDProduction::meas_skip_runtime())

// ===== Solver =====
// Default tol 1e-8; override at runtime with MEAS_CG_TOL for sloppy/refined runs.
inline RealD cg_tol_runtime() { return detail::env_real("MEAS_CG_TOL", 1e-8); }
constexpr RealD cg_tol = 1e-8;  // legacy constexpr, kept for places that don't read env
constexpr int   cg_max = 30000;

// ===== Config paths =====
inline std::string lambda_tag() {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(4) << lambda;
  return "lam" + ss.str();
}
inline std::string suffix_tag() {
  if (const char *s = std::getenv("SUFFIX"); s && *s) return std::string(s);
  return "";
}
inline std::string qcd_suffix_tag() {
  if (const char *s = std::getenv("QCD_SUFFIX"); s && *s) return std::string(s);
  return "";
}
inline std::string txqcd_cfg_dir()  { return "cfgs/txqcd_" + lambda_tag() + suffix_tag(); }
inline std::string txqcd_data_dir() { return "meas_2pt/txqcd_" + lambda_tag() + suffix_tag(); }
inline std::string qcd_cfg_dir()    { return "cfgs/qcd" + qcd_suffix_tag(); }
inline std::string qcd_data_dir()   { return "meas_2pt/qcd" + qcd_suffix_tag(); }

// ===== Measurement trajectories =====
inline std::vector<int> meas_trajs() {
  std::vector<int> v;
  for (int t = n_therm; t < n_therm + n_prod; t += meas_skip)
    v.push_back(t);
  return v;
}

// ===== Utility =====
inline void mkdir_p(const std::string &path) {
  std::string cmd = "mkdir -p " + path;
  system(cmd.c_str());
}

inline bool file_exists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

}  // namespace TXQCDProduction
