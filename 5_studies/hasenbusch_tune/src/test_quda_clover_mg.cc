// MG validation + amortized timing test for QudaCloverInverter's multigrid path.
//
// Compares plain CG and MG-GCR paths on the same gauge + sources.  Reports:
//   • per-solve wallclock (so we can amortize MG's one-time setup over N solves)
//   • residual + bit-exactness vs CG
//
// Configurable via env vars (so we don't need a separate binary per regime):
//   TEST_MASS         (default -0.245) — bare Wilson mass
//   TEST_CSW          (default 1.24930970916466) — clover coefficient
//   MG_NLEVEL         (default 2) — 2 or 3
//   MG_BLOCK_L0       (default "4 4 4 4") — level-0 → level-1 geometric block
//   MG_BLOCK_L1       (default "2 2 2 2") — level-1 → level-2 (if NLEVEL=3)
//   MG_NVEC           (default 24) — null-vector count per level
//   N_SOLVES          (default 1) — how many sources to invert (amortizes MG setup)
//   USE_RUN_VERIFY    (default 0) — run QUDA's MG verification at setup (slow)
//   STOUT_NSMEAR      (default 0) — stout-smear the loaded config N times before
//                                   solving. The HMC light force inverts the
//                                   STOUT-SMEARED operator, so a faithful timing
//                                   /iteration test must smear too (production=1).
//   STOUT_RHO         (default 0.125) — stout rho (production value)
//
// Solves three ways on the SAME (optionally smeared) operator + sources:
//   (a) QUDA-CG   (b) QUDA MG-GCR   (c) pure-Grid CG on M†M (the production backend)
// -> separates "operator is hard" (CG iters explode) from "Grid Dslash is slow".
//
// Examples:
//   # 16^3 x 48 at m=-0.245, 2-level MG (single-rank smoke):
//   srun --overlap --mpi=pmix -N 1 -n 1 ./test_quda_clover_mg \
//        --grid 16.16.16.48 --mpi 1.1.1.1
//
//   # 48^3 x 96 at m=-0.2416 (b6.3), 3-level MG, 16 amortized solves:
//   TEST_MASS=-0.2416 TEST_CSW=1.20536588031793 \
//   MG_NLEVEL=3 MG_BLOCK_L0='4 4 4 4' MG_BLOCK_L1='1 2 2 2' N_SOLVES=16 \
//   srun --overlap --mpi=pmix -N 2 -n 8 ./test_quda_clover_mg \
//        --grid 48.48.48.96 --mpi 1.1.2.4

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>
#include <Grid/algorithms/iterative/ConjugateGradient.h>
#include <Grid/qcd/smearing/StoutSmearing.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <cstdio>

using namespace Grid;

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

static std::array<int,4> parse_block(const char *s, std::array<int,4> dflt) {
  if (!s) return dflt;
  std::array<int,4> b = dflt;
  std::istringstream iss(s);
  iss >> b[0] >> b[1] >> b[2] >> b[3];
  return b;
}

static double getd(const char *k, double dflt) {
  const char *v = std::getenv(k);
  return v ? std::atof(v) : dflt;
}
static int geti(const char *k, int dflt) {
  const char *v = std::getenv(k);
  return v ? std::atoi(v) : dflt;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian         GridX(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGridX(&GridX);

  GridParallelRNG pRNG(&GridX);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  // Pass the gauge grid so QUDA inherits Grid's MPI communicator + rank map
  // (MPI-comms build). Without it, >=2 partitioned directions (e.g. 48^3 at
  // --mpi 1.1.2.4) hit QUDA's own lexicographic rank map -> few-% halo error
  // (the rank-map bug fixed in the strange campaign). Single-rank 16^3 is
  // unaffected. Mirrors TwoFlavourSchurCloverQudaForceActionMP.h's init.
  Quda::initialize(-1, nullptr, &GridX);

  // ---- Config ---------------------------------------------------------------
  RealD mass = getd("TEST_MASS", -0.245);
  RealD csw  = getd("TEST_CSW",  1.24930970916466);
  int n_level     = geti("MG_NLEVEL", 2);
  int n_vec       = geti("MG_NVEC",   24);
  int n_solves    = geti("N_SOLVES",  1);
  bool run_verify = geti("USE_RUN_VERIFY", 0) != 0;
  auto block_l0 = parse_block(std::getenv("MG_BLOCK_L0"), {4,4,4,4});
  auto block_l1 = parse_block(std::getenv("MG_BLOCK_L1"), {2,2,2,2});

  std::cout << GridLogMessage << "Config: mass=" << mass << " csw=" << csw
            << " mg_n_level=" << n_level << " n_vec=" << n_vec
            << " n_solves=" << n_solves
            << " block_l0={" << block_l0[0] << "," << block_l0[1] << "," << block_l0[2] << "," << block_l0[3] << "}";
  if (n_level >= 3)
    std::cout << " block_l1={" << block_l1[0] << "," << block_l1[1] << "," << block_l1[2] << "," << block_l1[3] << "}";
  std::cout << std::endl;

  // ---- Gauge + operator ----------------------------------------------------
  LatticeGaugeField U(&GridX);
  const char *cfg_file = std::getenv("CFG_FILE");
  if (cfg_file && *cfg_file) {
    // Detect format and load.  NERSC = "BEGIN_HEADER", LIME/ILDG = magic
    // 0x456789AB.
    FILE *fp = std::fopen(cfg_file, "rb");
    if (!fp) { std::cerr << "cannot open " << cfg_file << std::endl; return 2; }
    unsigned char buf[16] = {0};
    std::fread(buf, 1, sizeof(buf), fp); std::fclose(fp);
    FieldMetaData header;
    typedef GaugeStatistics<PeriodicGimplR> GS;
    if (std::memcmp(buf, "BEGIN_HEADER", 12) == 0) {
      NerscIO::readConfiguration<GS>(U, header, cfg_file);
    } else if (buf[0]==0x45 && buf[1]==0x67 && buf[2]==0x89 && buf[3]==0xAB) {
      IldgReader rd; rd.open(cfg_file); rd.readConfiguration(U, header); rd.close();
    } else {
      std::cerr << "unknown format for " << cfg_file << std::endl; return 2;
    }
    RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U);
    std::cout << GridLogMessage << "Loaded cfg: " << cfg_file
              << "  plaq = " << std::setprecision(10) << plaq << std::endl;
  } else {
    SU<Nc>::HotConfiguration(pRNG, U);
    std::cout << GridLogMessage << "Hot random gauge (no CFG_FILE env)" << std::endl;
  }

  // ---- Optional stout smearing (match the production light operator) --------
  // The HMC light force inverts the STOUT-smeared gauge (is_smeared=true), so a
  // faithful iteration/timing comparison must smear the loaded config the same
  // way (production: STOUT_RHO=0.125, STOUT_NSMEAR=1). STOUT_NSMEAR=0 (default)
  // leaves the thin links = the raw-config baseline.
  int    stout_nsmear = geti("STOUT_NSMEAR", 0);
  double stout_rho    = getd("STOUT_RHO",    0.125);
  if (stout_nsmear > 0) {
    Smear_Stout<PeriodicGimplR> stout(stout_rho);
    LatticeGaugeField Usm(&GridX), Utmp(&GridX);
    Usm = U;
    for (int i = 0; i < stout_nsmear; ++i) { stout.smear(Utmp, Usm); Usm = Utmp; }
    U = Usm;
    RealD plaq_sm = WilsonLoops<PeriodicGimplR>::avgPlaquette(U);
    std::cout << GridLogMessage << "Stout-smeared: nsmear=" << stout_nsmear
              << " rho=" << stout_rho << "  smeared plaq = "
              << std::setprecision(10) << plaq_sm << std::endl;
  }

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF D(U, GridX, RBGridX, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(D);

  // Generate N_SOLVES random sources up-front so both paths solve the SAME RHSs.
  std::vector<LatticeFermion> srcs;
  for (int s = 0; s < n_solves; ++s) {
    srcs.emplace_back(&GridX);
    random(pRNG, srcs.back());
  }

  // ---- (a) QUDA-CG ----------------------------------------------------------
  QudaCloverParams qp_cg;
  qp_cg.mass = mass; qp_cg.csw = csw;
  qp_cg.anti_periodic_t = true; qp_cg.tol = 1e-10; qp_cg.max_iter = 50000;
  qp_cg.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  qp_cg.use_multigrid = false;

  std::cout << GridLogMessage << "=== Path (a): QUDA-CG (n=" << n_solves << " solves) ===" << std::endl;
  QudaCloverInverter cg(&GridX, qp_cg);
  cg.SetGauge(U);

  std::vector<LatticeFermion> x_cg;
  for (int s = 0; s < n_solves; ++s) x_cg.emplace_back(&GridX);
  auto t0 = std::chrono::steady_clock::now();
  int  total_cg_iters = 0;
  for (int s = 0; s < n_solves; ++s) {
    x_cg[s] = Zero();
    cg(HermOp, srcs[s], x_cg[s]);
    total_cg_iters += cg.LastIter();
  }
  auto t1 = std::chrono::steady_clock::now();
  double cg_total_secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << GridLogMessage << "  CG total iters=" << total_cg_iters
            << "  total_secs=" << cg_total_secs
            << "  per-solve secs=" << (cg_total_secs / n_solves)
            << std::endl;

  // ---- (b) MG-preconditioned GCR -------------------------------------------
  QudaCloverParams qp_mg = qp_cg;
  qp_mg.use_multigrid = true;
  qp_mg.mg.n_level = n_level;
  qp_mg.mg.geo_block_size = (n_level >= 3) ? std::vector<std::array<int,4>>{ block_l0, block_l1 }
                                            : std::vector<std::array<int,4>>{ block_l0 };
  qp_mg.mg.n_vec = n_vec;
  qp_mg.mg.run_verify = run_verify;

  std::cout << GridLogMessage << "=== Path (b): MG-GCR (n=" << n_solves << " solves) ===" << std::endl;
  auto t_setup_start = std::chrono::steady_clock::now();
  QudaCloverInverter mg(&GridX, qp_mg);
  mg.SetGauge(U);
  auto t_setup_end = std::chrono::steady_clock::now();
  double mg_setup_secs = std::chrono::duration<double>(t_setup_end - t_setup_start).count();
  std::cout << GridLogMessage << "  MG setup secs = " << mg_setup_secs << std::endl;

  std::vector<LatticeFermion> x_mg;
  for (int s = 0; s < n_solves; ++s) x_mg.emplace_back(&GridX);
  auto tA = std::chrono::steady_clock::now();
  int  total_mg_iters = 0;
  for (int s = 0; s < n_solves; ++s) {
    x_mg[s] = Zero();
    mg(HermOp, srcs[s], x_mg[s]);
    total_mg_iters += mg.LastIter();
  }
  auto tB = std::chrono::steady_clock::now();
  double mg_solve_secs = std::chrono::duration<double>(tB - tA).count();
  std::cout << GridLogMessage << "  MG total iters=" << total_mg_iters
            << "  total solve_secs=" << mg_solve_secs
            << "  per-solve secs=" << (mg_solve_secs / n_solves)
            << "  (setup amortized over " << n_solves << " solves: "
            << ((mg_setup_secs + mg_solve_secs) / n_solves) << " s/solve)"
            << std::endl;

  // ---- (c) pure-Grid CG on M†M (the production backend) --------------------
  // Grid's CG solves the normal eq (M†M)x=b; QUDA's CG is internally CGNE on the
  // same M†M, so iteration counts are directly comparable. The rhs convention
  // differs (b vs M†b), so we do NOT cross-check x_grid against x_quda — we read
  // the iteration count (operator hardness) and time the backend. Double prec:
  // a mixed-precision Grid CG would be ~2-3x faster, so this is a conservative
  // Grid baseline.
  std::cout << GridLogMessage << "=== Path (c): pure-Grid CG on MdagM (n=" << n_solves << " solves) ===" << std::endl;
  ConjugateGradient<LatticeFermion> GridCG(1.0e-10, 50000);
  std::vector<LatticeFermion> x_gcg;
  for (int s = 0; s < n_solves; ++s) x_gcg.emplace_back(&GridX);
  auto tg0 = std::chrono::steady_clock::now();
  int total_gcg_iters = 0;
  for (int s = 0; s < n_solves; ++s) {
    x_gcg[s] = Zero();
    GridCG(HermOp, srcs[s], x_gcg[s]);
    total_gcg_iters += GridCG.IterationsToComplete;
  }
  auto tg1 = std::chrono::steady_clock::now();
  double gcg_total_secs = std::chrono::duration<double>(tg1 - tg0).count();
  std::cout << GridLogMessage << "  Grid-CG total iters=" << total_gcg_iters
            << "  total_secs=" << gcg_total_secs
            << "  per-solve secs=" << (gcg_total_secs / n_solves) << std::endl;

  // ---- Bit-exact compare ----------------------------------------------------
  RealD max_rel_diff = 0.0;
  for (int s = 0; s < n_solves; ++s) {
    LatticeFermion diff(&GridX);
    diff = x_mg[s] - x_cg[s];
    RealD rel = std::sqrt(norm2(diff) / norm2(x_cg[s]));
    if (rel > max_rel_diff) max_rel_diff = rel;
  }

  // Verify M·x_mg ≈ src for each solve
  RealD max_M_resid = 0.0;
  for (int s = 0; s < n_solves; ++s) {
    LatticeFermion Mx(&GridX), res(&GridX);
    D.M(x_mg[s], Mx);
    res = Mx - srcs[s];
    RealD rel = std::sqrt(norm2(res) / norm2(srcs[s]));
    if (rel > max_M_resid) max_M_resid = rel;
  }

  double per_solve_speedup = (cg_total_secs / n_solves) /
                             ((mg_setup_secs + mg_solve_secs) / n_solves);
  std::cout << GridLogMessage
            << "============================================" << std::endl;
  std::cout << GridLogMessage << "Summary:" << std::endl;
  std::cout << GridLogMessage << "  QUDA-CG per-solve: " << (cg_total_secs / n_solves)  << " s, " << (total_cg_iters / n_solves) << " iters" << std::endl;
  std::cout << GridLogMessage << "  Grid-CG per-solve: " << (gcg_total_secs / n_solves) << " s, " << (total_gcg_iters / n_solves) << " iters (double, M†M)" << std::endl;
  std::cout << GridLogMessage << "  QUDA-MG per-solve: " << (mg_solve_secs / n_solves)  << " s, " << (total_mg_iters / n_solves) << " iters (no setup)" << std::endl;
  std::cout << GridLogMessage << "  MG  setup:     " << mg_setup_secs << " s (one-time)" << std::endl;
  std::cout << GridLogMessage << "  MG  amortized (setup/n + solve/n): "
            << ((mg_setup_secs + mg_solve_secs) / n_solves) << " s/solve" << std::endl;
  std::cout << GridLogMessage << "  per-solve speedup (QUDA-CG / MG-amortized) = " << per_solve_speedup << "x" << std::endl;
  std::cout << GridLogMessage << "  backend speedup (Grid-CG / QUDA-CG)        = "
            << (cg_total_secs > 0 ? gcg_total_secs / cg_total_secs : 0.0) << "x" << std::endl;
  std::cout << GridLogMessage << "  max ||x_mg - x_cg|| / ||x_cg||         = " << max_rel_diff << std::endl;
  std::cout << GridLogMessage << "  max ||M·x_mg - src|| / ||src||         = " << max_M_resid << std::endl;
  bool pass = (max_rel_diff < 1e-5 && max_M_resid < 1e-6);
  std::cout << GridLogMessage
            << (pass ? "PASS" : "FAIL") << " — MG vs CG agreement"
            << std::endl;

  Quda::finalize();
  Grid_finalize();
  return pass ? 0 : 1;
}
