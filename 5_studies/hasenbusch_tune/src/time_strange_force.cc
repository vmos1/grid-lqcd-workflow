// time_strange_force.cc
//
// Standalone strange-quark RHMC FORCE timer.
//
// Isolates and instruments the ~350 s "Strange" cost seen in
// gen_qcd_hasenbusch_tune_compact.cc (FORCES_ONLY mode) so we can answer the
// open question -- "is the 350 s the multishift SOLVER, or the per-pole force
// ASSEMBLY?" -- with EXPLICIT per-phase timers (not inferred from log gaps), and
// compare the number cleanly across machines (Perlmutter vs lq).
//
// It builds ONLY the strange sector: the exact production strange action
// OneFlavourSchurCloverRationalActionMP at the cfg_2000 b6.3 48^3x96 parameters,
// on an imported thermalized config, then times -- on the SAME fixed gauge --
// the three pseudofermion phases:
//
//   refresh()  heatbath   ->  multishift CG SOLVE only          (no force)
//   S()        action     ->  multishift CG SOLVE only          (no force)
//   deriv()    MD force   ->  multishift CG SOLVE + per-pole force ASSEMBLY
//
// refresh()/S() are pure solves, so they give the SOLVER-only cost directly;
// deriv() is solve+assembly.  Hence  assembly ~= deriv - solve.  Grid's
// ConjugateGradientMultiShift{,MixedPrec} also self-logs its own solve time
// INSIDE each call, so the exact solve/assembly split is readable from the log
// too (look for the "ConjugateGradientMultiShiftMixedPrec ... Total" line that
// prints during deriv()).
//
// PURE GRID (no QUDA) by construction -- uses Grid's mixed-precision multishift
// (OneFlavourSchurCloverRationalActionMP), exactly as the tune binary's non-QUDA
// path.  The QUDA strange-force path is deliberately NOT exercised here.
//
// Operator: COMPACT clover by default (faithful to gen_qcd_hasenbusch_tune_compact).
//   Build with -DUSE_NONCOMPACT to use non-compact WilsonCloverFermion instead.
//   The clover FORCE (MeeDeriv/MooDeriv) is gauge-only and bit-identical between
//   the two operators, so the timings are directly comparable -- use the
//   non-compact build if a machine's libGrid does not instantiate the compact
//   operator (e.g. on lq, if the compact build link-fails).
//
// ---------------------------------------------------------------------------
// Build (Perlmutter, pure-Grid, COMPACT default):
//   cd grid-lqcd-workflow/5_studies/hasenbusch_tune
//   SRC=$PWD/src/time_strange_force.cc BIN=$PWD/bin/time_strange_force \
//     bash perlmutter/build_driver_puregrid.sh
//
//   non-compact: append -DUSE_NONCOMPACT to the compile (one-off):
//     CXX=$(../../../Grid-TXQCD/build/grid-config --cxx); ... add -DUSE_NONCOMPACT
//   (see the run script / lq handoff for the exact line).
//
// Run (2 nodes / 8 GPU):
//   IMPORT_CFG=.../cfg_2000.lime  MASS_STRANGE=-0.2050  CSW=1.20537 \
//   srun ... ./time_strange_force --grid 48.48.48.96 --mpi 1.2.2.2 \
//        --accelerator-threads 8 --shm 2048 --shm-mpi 1
//
// Env (all optional; defaults = production cfg_2000 strange):
//   IMPORT_CFG    NERSC or ILDG/LIME gauge (REQUIRED for a meaningful number;
//                 hot start otherwise -- smoke test only)
//   MASS_STRANGE  bare strange mass            (default -0.2050)
//   CSW           clover coefficient           (default 1.20537)
//   RAT_LO        rational lower bound         (default 1e-4)   <- production
//   RAT_HI        rational upper bound         (default 100.0)  <- production
//   RAT_DEGREE    rational degree / #poles     (default 20)     <- production
//   RAT_MDTOL     MD (force) per-pole tol      (default 1e-6)   <- production
//   CG_TOL        action/heatbath solve tol    (default 1e-8)   <- production
//   CG_MAX        CG max iters                 (default 30000)
//   N_DERIV       number of deriv() repeats    (default 1)

#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#ifdef USE_NONCOMPACT
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#else
#include <Grid/qcd/action/fermion/CompactWilsonCloverFermion.h>
#endif
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>

using namespace Grid;

#ifdef USE_NONCOMPACT
typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>>  WCF;
typedef WilsonCloverFermion<WilsonImplF, CloverHelpers<WilsonImplF>>  WCF_f;
static const char *kOpKind  = "non-compact WilsonCloverFermion";
static const char *kOpShort = "noncompact";
#else
typedef CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<WilsonImplR>> WCF;
typedef CompactWilsonCloverFermion<WilsonImplF, CompactCloverHelpers<WilsonImplF>> WCF_f;
static const char *kOpKind  = "compact CompactWilsonCloverFermion";
static const char *kOpShort = "compact";
#endif

static double env_d(const char *k, double dflt) {
  const char *v = std::getenv(k); return (v && *v) ? std::atof(v) : dflt;
}
static long env_i(const char *k, long dflt) {
  const char *v = std::getenv(k); return (v && *v) ? std::atol(v) : dflt;
}
static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid4(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid4);
  // Single-precision grids for the mixed-precision strange multishift.
  GridCartesian         GridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
  GridRedBlackCartesian RBGridF(&GridF);

  GridSerialRNG   sRNG;          sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
  GridParallelRNG pRNG(&Grid4);  pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});

  const RealD mass = env_d("MASS_STRANGE", -0.2050);
  const RealD csw  = env_d("CSW", 1.20537);

  // ── Gauge ──────────────────────────────────────────────────────────────────
  LatticeGaugeField Umu(&Grid4);
  if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic << std::endl;
    FILE *fp = std::fopen(ic, "rb"); char magic[16] = {0};
    if (fp) { std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      NerscIO::readConfiguration<GaugeStatistics<PeriodicGimplR>>(Umu, header, std::string(ic));
    } else {
      IldgReader IR; IR.open(std::string(ic));
      IR.readConfiguration(Umu, header); IR.close();
    }
  } else {
    std::cout << GridLogMessage << "No IMPORT_CFG — hot start (smoke test only)." << std::endl;
    SU<Nc>::HotConfiguration(pRNG, Umu);
  }
  std::cout << GridLogMessage << "plaquette = " << std::setprecision(10)
            << WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu)
            << "   (cfg_2000 b6.3 ref ≈ 0.5435082)" << std::endl;

  // SP gauge for the mixed-precision strange operator.
  LatticeGaugeFieldF UmuF(&GridF);
  {
    LatticeColourMatrix  Ud(&Grid4); LatticeColourMatrixF Uf(&GridF);
    for (int mu = 0; mu < Nd; ++mu) {
      Ud = PeekIndex<LorentzIndex>(Umu, mu);
      precisionChange(Uf, Ud);
      PokeIndex<LorentzIndex>(UmuF, Uf, mu);
    }
  }

  // ── Strange operator (antiperiodic-t), DP + SP ─────────────────────────────
  WilsonImplParams impl_p, impl_pF;
  impl_p.boundary_phases  = std::vector<Complex>({1., 1., 1., -1.});
  impl_pF.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
  WilsonAnisotropyCoefficients anis;
#ifdef USE_NONCOMPACT
  WCF   StrangeOp (Umu,  Grid4, RBGrid,  mass, csw, csw, anis, impl_p);
  WCF_f StrangeOpF(UmuF, GridF, RBGridF, mass, csw, csw, anis, impl_pF);
#else
  WCF   StrangeOp (Umu,  Grid4, RBGrid,  mass, csw, csw, /*cF=*/1.0, anis, impl_p);
  WCF_f StrangeOpF(UmuF, GridF, RBGridF, mass, csw, csw, /*cF=*/1.0, anis, impl_pF);
#endif

  // ── Strange RHMC action — EXACT production params ───────────────────────────
  //   gen_qcd_hasenbusch_tune_compact.cc:286
  //   OneFlavourRationalParams(lo, hi, MaxIter, tol, degree, precision,
  //                            BoundsCheckFreq, mdtol, BoundsCheckTol)
  const RealD lo     = env_d("RAT_LO", 1e-4);
  const RealD hi     = env_d("RAT_HI", 100.0);
  const int   degree = (int)env_i("RAT_DEGREE", 20);
  const RealD mdtol  = env_d("RAT_MDTOL", 1e-6);
  const RealD cg_tol = env_d("CG_TOL", 1e-8);
  const int   cg_max = (int)env_i("CG_MAX", 30000);
  OneFlavourRationalParams strange_rat(lo, hi, cg_max, cg_tol, degree, 64, 100, mdtol, 1e-4);

  OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF, WCF, WCF_f>
      Strange(StrangeOp, StrangeOpF, &RBGridF, strange_rat, 50);

  std::cout << GridLogMessage << "Strange RHMC: " << kOpKind << std::endl;
  std::cout << GridLogMessage << "  mass=" << mass << " csw=" << csw
            << "  rat=[" << std::scientific << std::setprecision(2) << lo << "," << hi
            << "] deg=" << degree << " mdtol=" << mdtol << " cg_tol=" << cg_tol
            << std::fixed << std::endl;

  // ── Time the three phases on the fixed gauge ───────────────────────────────
  LatticeGaugeField force(&Grid4);

  std::cout << GridLogMessage << "==== [1/3] refresh()  heatbath  (multishift SOLVE only) ====" << std::endl;
  double t0 = now_s();
  Strange.refresh(Umu, sRNG, pRNG);
  const double t_refresh = now_s() - t0;
  std::cout << GridLogMessage << "PHASE refresh = " << std::fixed << std::setprecision(2)
            << t_refresh << " s" << std::endl;

  std::cout << GridLogMessage << "==== [2/3] S()        action    (multishift SOLVE only) ====" << std::endl;
  t0 = now_s();
  const RealD Sval = Strange.S(Umu);
  const double t_S = now_s() - t0;
  std::cout << GridLogMessage << "PHASE S       = " << t_S << " s   (S=" << Sval << ")" << std::endl;

  std::cout << GridLogMessage << "==== [3/3] deriv()    MD force   (multishift SOLVE + per-pole ASSEMBLY) ====" << std::endl;
  const int nderiv = (int)env_i("N_DERIV", 1);
  double t_deriv_sum = 0;
  for (int i = 0; i < nderiv; ++i) {
    t0 = now_s();
    Strange.deriv(Umu, force);
    const double td = now_s() - t0;
    t_deriv_sum += td;
    const double fnorm = std::sqrt(norm2(force) / force.Grid()->gSites());
    std::cout << GridLogMessage << "PHASE deriv[" << i << "] = " << std::fixed << td
              << " s   (|force|_avg=" << std::scientific << std::setprecision(4) << fnorm
              << std::fixed << ")" << std::endl;
  }
  const double t_deriv = t_deriv_sum / nderiv;

  // ── Summary ────────────────────────────────────────────────────────────────
  // refresh()/S() are pure DP multishift solves; deriv() is MP solve + assembly.
  // The MP solve is <= the DP solve, so (deriv - S) is a LOWER bound on the
  // per-pole assembly; for the exact split read the self-logged
  // "ConjugateGradientMultiShiftMixedPrec ... Total" time printed inside deriv().
  std::cout << GridLogMessage << "================ STRANGE FORCE TIMING ================" << std::endl;
  std::cout << GridLogMessage << "  op kind          : " << kOpKind << std::endl;
  std::cout << GridLogMessage << "  refresh (solve)  : " << std::setprecision(2) << t_refresh << " s" << std::endl;
  std::cout << GridLogMessage << "  S()     (solve)  : " << t_S << " s" << std::endl;
  std::cout << GridLogMessage << "  deriv  (slv+asm) : " << t_deriv << " s   (avg of " << nderiv << ")" << std::endl;
  std::cout << GridLogMessage << "  assembly (>=)    : " << (t_deriv - t_S)
            << " s   (= deriv - S-solve; lower bound — see MP-solve log line for exact)" << std::endl;
  std::cout << GridLogMessage << "  TIMING_CSV op=" << kOpShort
            << " refresh=" << t_refresh << " S=" << t_S << " deriv=" << t_deriv
            << " assembly_lb=" << (t_deriv - t_S) << " degree=" << degree << std::endl;
  std::cout << GridLogMessage << "=====================================================" << std::endl;

  Grid_finalize();
  return 0;
}
