// QUDA multishift SOLVER-CONFIG sweep — fork of test_quda_multishift_ours.cc.
//
// WHY A FORK (not an edit): other tests/runs are in flight and sessions get
// cleared, so we keep test_quda_multishift_ours.cc + every existing binary
// bit-identical.  This file links against the EXISTING libGrid (no Grid-TXQCD
// edit, no libGrid rebuild) and only ADDS env-gated QUDA solver-config knobs,
// set via the inverter's by-reference InvertParam()/GaugeParam().  With NO new
// env set it reproduces today's exact divergence (the baseline).
//
// Root cause being tested (2026-06-22): on the 48^3 b6.3 near-critical strange
// operator, Grid's multishift converges the smallest-offset (base) shift in
// ~487 iters; QUDA's multishift bails at ~63-71 iters when its reliable-update
// guard sees the true residual increase (41->71->265).  reliable_delta=1e-1
// (hardcoded in QudaCloverMultiShiftInverter::setup_params_) is the prime
// suspect.  This sweep finds the knob(s) that let QUDA's base shift converge.
//
// Run (2 nodes, A100):
//   IMPORT_CFG=.../cfg_2000.lime srun ... ./test_quda_ms_solver_sweep \
//       --grid 48.48.48.96 --mpi 1.2.2.2 --accelerator-threads 8 --shm 512
//
// Env (physics/sweep — inherited from the original):
//   IMPORT_CFG       NERSC or ILDG/LIME gauge (else hot start)
//   MASS_STRANGE     bare mass (default -0.2050)
//   CSW              clover coeff (default 1.20537)
//   MS_SHIFTS        comma list of shifts (default 1e-5,1e-4,1e-3,1e-2,0.05,0.5,5.0)
//   MS_TOL           per-shift tolerance (default 1e-4, matches the force run)
//   MS_MATPC=sym     symmetric matpc (tests the asymmetric 4kappa^2 rescale path)
//   QUDA_FORCE_SLOPPY_DP  run the multishift fully double
//   SKIP_GRID        skip the (slow) Grid A/B reference
//
// Env (NEW — QUDA solver-config knobs; UNSET => inverter default => baseline):
//   MS_RELIABLE_DELTA        inv.reliable_delta            (default 1e-1)
//   MS_RELIABLE_DELTA_REFINE inv.reliable_delta_refinement
//   MS_MAX_RES_INC           inv.max_res_increase          (consec. tolerated)
//   MS_MAX_RES_INC_TOTAL     inv.max_res_increase_total    (total tolerated)
//   MS_ALT_RELIABLE          inv.use_alternative_reliable  (0/1)
//   MS_ACC_PIPELINE          inv.solution_accumulator_pipeline
//   MS_SLOPPY_ACC            inv.use_sloppy_partial_accumulator (0/1)
//   MS_MAXITER               inv.maxiter
//   MS_RECON_SLOPPY          gauge recon for sloppy+refine+precond: 18=NO,12,8

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShift.h>
#include <Grid/qcd/action/pseudofermion/EvenOddSchurDifferentiable.h>

#include <cstdlib>
#include <sstream>
#include <vector>

using namespace Grid;
typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

static double env_d(const char *k, double dflt) {
  const char *v = std::getenv(k);
  return (v && *v) ? std::atof(v) : dflt;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian Grid4(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid(&Grid4);   // solve_rb_even works on red-black (V_eo) fields

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});
  Quda::initialize(-1, nullptr, &Grid4);   // pass grid → QUDA_GRID_RANKMAP uses Grid's rank map

  const RealD mass = env_d("MASS_STRANGE", -0.2050);
  const RealD csw  = env_d("CSW", 1.20537);

  // ── Gauge: load our thermalized config (near-critical conditioning is the
  //    whole point) or fall back to a hot gauge. ────────────────────────────
  LatticeGaugeField U(&Grid4);
  if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic << std::endl;
    FILE *fp = std::fopen(ic, "rb"); char magic[16] = {0};
    if (fp) { std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      NerscIO::readConfiguration<GaugeStatistics<PeriodicGimplR>>(U, header, std::string(ic));
    } else {
      IldgReader IR; IR.open(std::string(ic));
      IR.readConfiguration(U, header); IR.close();
    }
  } else {
    std::cout << GridLogMessage << "No IMPORT_CFG — hot start." << std::endl;
    SU<Nc>::HotConfiguration(pRNG, U);
  }
  std::cout << GridLogMessage << "plaquette = " << std::setprecision(10)
            << WilsonLoops<PeriodicGimplR>::avgPlaquette(U) << std::endl;

  // ── Shift sweep (default spans tiny RHMC poles → large) ──────────────────
  std::vector<RealD> shifts;
  if (const char *s = std::getenv("MS_SHIFTS"); s && *s) {
    std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) shifts.push_back(std::atof(tok.c_str()));
  } else {
    shifts = {1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.5, 5.0};
  }
  const RealD tol = env_d("MS_TOL", 1e-4);

  QudaCloverParams qp;
  qp.mass = mass; qp.csw = csw; qp.anti_periodic_t = true;
  qp.tol = tol; qp.max_iter = 20000;
  qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  if (std::getenv("QUDA_FORCE_SLOPPY_DP")) qp.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;

  QudaCloverMultiShiftSpec spec;
  // matpc selectable: default ASYMMETRIC (= the strange force); MS_MATPC=sym tests
  // whether the wrapper's hand-rolled asymmetric 4kappa^2 rescale is the culprit.
  bool matpc_sym = false;
  if (const char *m = std::getenv("MS_MATPC"); m && (*m == 's' || *m == 'S')) {
    spec.matpc_type = QUDA_MATPC_EVEN_EVEN;  // symmetric: QUDA normalizes natively
    matpc_sym = true;
  } else {
    spec.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  }
  spec.shifts = shifts;
  spec.tols.assign(shifts.size(), tol);

  std::cout << GridLogMessage << "mass=" << mass << " csw=" << csw
            << " tol=" << tol << " nshift=" << shifts.size()
            << " matpc=" << (matpc_sym ? "SYMMETRIC" : "ASYMMETRIC")
            << " sloppy_dp=" << (std::getenv("QUDA_FORCE_SLOPPY_DP") ? 1 : 0)
            << std::endl;

  QudaCloverMultiShiftInverter quda_ms(&Grid4, qp, spec);

  // ── NEW: Phase-1 solver-config knobs (env-gated). Applied to the inverter's
  //    by-reference InvertParam()/GaugeParam() BEFORE SetGauge (so recon takes
  //    effect on the gauge load) and before solve_rb_even.  Any knob left unset
  //    keeps the inverter default → reproduces today's baseline divergence. ──
  {
    QudaInvertParam &inv = quda_ms.InvertParam();
    QudaGaugeParam  &gp  = quda_ms.GaugeParam();
    auto set_d = [](const char *k, double &dst) {
      if (const char *v = std::getenv(k); v && *v) {
        dst = std::atof(v);
        std::cout << GridLogMessage << "  [knob] " << k << " -> " << dst << std::endl;
      }
    };
    auto set_i = [](const char *k, int &dst) {
      if (const char *v = std::getenv(k); v && *v) {
        dst = std::atoi(v);
        std::cout << GridLogMessage << "  [knob] " << k << " -> " << dst << std::endl;
      }
    };
    set_d("MS_RELIABLE_DELTA",        inv.reliable_delta);
    set_d("MS_RELIABLE_DELTA_REFINE", inv.reliable_delta_refinement);
    set_i("MS_MAX_RES_INC",           inv.max_res_increase);
    set_i("MS_MAX_RES_INC_TOTAL",     inv.max_res_increase_total);
    set_i("MS_ALT_RELIABLE",          inv.use_alternative_reliable);
    set_i("MS_ACC_PIPELINE",          inv.solution_accumulator_pipeline);
    set_i("MS_SLOPPY_ACC",            inv.use_sloppy_partial_accumulator);
    set_i("MS_MAXITER",               inv.maxiter);
    if (const char *v = std::getenv("MS_RECON_SLOPPY"); v && *v) {
      int r = std::atoi(v);
      QudaReconstructType rt = (r == 12) ? QUDA_RECONSTRUCT_12
                             : (r == 8)  ? QUDA_RECONSTRUCT_8
                                         : QUDA_RECONSTRUCT_NO;   // 18 / anything else
      gp.reconstruct_sloppy            = rt;
      gp.reconstruct_refinement_sloppy = rt;
      gp.reconstruct_precondition      = rt;
      std::cout << GridLogMessage << "  [knob] MS_RECON_SLOPPY -> " << r << std::endl;
    }
    std::cout << GridLogMessage
              << "  [knob] EFFECTIVE: reliable_delta=" << inv.reliable_delta
              << " reliable_delta_refine=" << inv.reliable_delta_refinement
              << " max_res_increase=" << inv.max_res_increase
              << " max_res_increase_total=" << inv.max_res_increase_total
              << " alt_reliable=" << inv.use_alternative_reliable
              << " acc_pipeline=" << inv.solution_accumulator_pipeline
              << " sloppy_acc=" << inv.use_sloppy_partial_accumulator
              << " recon_sloppy=" << (int)gp.reconstruct_sloppy
              << " maxiter=" << inv.maxiter
              << std::endl;
  }

  quda_ms.SetGauge(U);

  // Even-parity source + solutions on the red-black grid (what solve_rb_even
  // reads/writes; full-grid fields trip the V_eo vs lSites() assert).
  LatticeFermion src_full(&Grid4); random(pRNG, src_full);
  LatticeFermion src(&RBGrid); src.Checkerboard() = Even;
  pickCheckerboard(Even, src, src_full);
  std::vector<LatticeFermion> sols(shifts.size(), LatticeFermion(&RBGrid));
  for (auto &s : sols) { s.Checkerboard() = Even; s = Zero(); }

  std::cout << GridLogMessage << "=== solve_rb_even on " << shifts.size()
            << " shifts ===" << std::endl;
  quda_ms.solve_rb_even(src, sols, /*make_resident=*/false);

  std::cout << GridLogMessage << "multishift: " << quda_ms.LastIter()
            << " iter, " << quda_ms.LastSecs() << " s" << std::endl;

  int nfail = 0;
  auto res = quda_ms.LastResPerShift();
  for (int k = 0; k < (int)shifts.size(); ++k) {
    bool conv = (res[k] < 10 * tol);
    if (!conv) ++nfail;
    std::cout << GridLogMessage
              << "  shift[" << k << "] = " << std::scientific << std::setprecision(3)
              << shifts[k] << "  true_res = " << res[k]
              << "  norm2(x) = " << norm2(sols[k])
              << (conv ? "  OK" : "  <-- FAIL (> 10*tol)") << std::endl;
  }
  std::cout << GridLogMessage << "RESULT: " << (shifts.size() - nfail) << "/"
            << shifts.size() << " shifts converged; "
            << (nfail ? "REPRODUCED divergence" : "all OK") << std::endl;

  // ──────────────────────────────────────────────────────────────────────────
  // A/B reference: Grid's own ConjugateGradientMultiShift on the SAME operator,
  // source and shifts — the exact Schur op the strange force uses
  // (SchurDifferentiableOperator on WilsonCloverFermion, EVEN_EVEN_ASYMMETRIC,
  // mass form).  The wrapper's 4kappa^2 rescale makes QUDA's output the solution
  // of (Mpc_grid^dag Mpc_grid + sigma) x = b in the SAME mass normalization, so
  // the two are directly comparable.  Grid converges where QUDA diverges; the
  // cross-check (cos~1 on the shift QUDA DID converge) proves it's the identical
  // operator -> minimal, self-validating A/B bug-report artifact.  SKIP_GRID=1
  // disables the (slow, >100 s) Grid solve.
  // ──────────────────────────────────────────────────────────────────────────
  if (std::getenv("SKIP_GRID") == nullptr) {
    std::cout << GridLogMessage
              << "=== Grid ConjugateGradientMultiShift (A/B reference) ===" << std::endl;
    WilsonImplParams impl_p;
    impl_p.boundary_phases.resize(Nd, 1.0);
    impl_p.boundary_phases[Nd - 1] = -1.0;   // antiperiodic-t (matches qp.anti_periodic_t)
    WCF Dgrid(U, Grid4, RBGrid, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);
    SchurDifferentiableOperator<WilsonImplR> Mpc(Dgrid);

    MultiShiftFunction msf;
    msf.order = (int)shifts.size();
    msf.norm  = 0.0;
    msf.poles = shifts;                       // (Mpc^dag Mpc + pole_k) x_k = src
    msf.residues.assign(shifts.size(), 1.0);
    msf.tolerances.assign(shifts.size(), tol);

    std::vector<LatticeFermion> sols_grid(shifts.size(), LatticeFermion(&RBGrid));
    for (auto &s : sols_grid) { s.Checkerboard() = Even; s = Zero(); }

    ConjugateGradientMultiShift<LatticeFermion> msCG(qp.max_iter, msf);
    double t0 = usecond();
    msCG(Mpc, src, sols_grid);
    double tg = (usecond() - t0) * 1e-6;
    std::cout << GridLogMessage << "Grid multishift wall: " << tg << " s" << std::endl;

    int gfail = 0;
    LatticeFermion tmp(&RBGrid), r(&RBGrid);
    for (int k = 0; k < (int)shifts.size(); ++k) {
      Mpc.HermOp(sols_grid[k], tmp);          // Mpc^dag Mpc x_k
      tmp = tmp + shifts[k] * sols_grid[k];   // (Mpc^dag Mpc + sigma_k) x_k
      r = tmp - src;
      RealD tr = std::sqrt(norm2(r) / norm2(src));
      bool conv = (tr < 10 * tol);
      if (!conv) ++gfail;
      std::cout << GridLogMessage << "  [GRID] shift[" << k << "] = "
                << std::scientific << std::setprecision(3) << shifts[k]
                << "  true_res = " << tr << "  norm2(x) = " << norm2(sols_grid[k])
                << (conv ? "  OK" : "  <-- FAIL") << std::endl;
    }
    std::cout << GridLogMessage << "GRID RESULT: " << (shifts.size() - gfail) << "/"
              << shifts.size() << " shifts converged" << std::endl;

    std::cout << GridLogMessage
              << "=== cross-check Grid vs QUDA per shift (cos -> ~1 means same op) ==="
              << std::endl;
    for (int k = 0; k < (int)shifts.size(); ++k) {
      RealD ng = norm2(sols_grid[k]), nq = norm2(sols[k]);
      ComplexD ip = innerProduct(sols_grid[k], sols[k]);
      RealD cosv  = (ng > 0 && nq > 0) ? real(ip) / std::sqrt(ng * nq) : 0.0;
      RealD ratio = (nq > 0) ? std::sqrt(ng / nq) : 0.0;
      std::cout << GridLogMessage << "  shift[" << k << "] = "
                << std::scientific << std::setprecision(3) << shifts[k]
                << "  cos = " << std::fixed << std::setprecision(6) << cosv
                << "  |grid|/|quda| = " << std::setprecision(4) << ratio << std::endl;
    }
  }

  Quda::finalize();
  Grid_finalize();
  return nfail ? 1 : 0;
}
