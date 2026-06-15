// Lo-sweep: do the REAL strange-RHMC poles trip QUDA, and does raising `lo` fix it?
//
// For each lo in LO_SWEEP, build the ACTUAL poles the strange force solves
// (degree-DEGREE Remez of x^(-1/2) over [lo,hi], exactly as
// OneFlavourSchurCloverRationalActionEven constructs PowerNegHalf), then solve
// that pole set with BOTH:
//   - QUDA  : QudaCloverMultiShiftInverter::solve_rb_even (the exact force path)
//   - Grid  : ConjugateGradientMultiShift (the working baseline)
// on the SAME thermalized operator / gauge / source.
//
// Question answered: the A/B with hand-picked shifts showed QUDA fails every
// sigma <~ 0.5.  Raising `lo` raises the SMALLEST real pole.  Does QUDA's
// converged-count climb to N as lo rises (=> a tunable fix), or does it keep
// failing the small poles (=> not a knob; a QUDA build/version difference vs the
// collaborator's working run)?
//
// Run (2 nodes, A100):
//   IMPORT_CFG=.../cfg_2000.lime QUDA_ENABLE_MPS=1 QUDA_ENABLE_P2P=0 \
//   LO_SWEEP=1e-4,1e-3,1e-2,1e-1 srun ... ./test_quda_losweep \
//       --grid 48.48.48.96 --mpi 1.2.2.2 --accelerator-threads 8 --shm 512 --shm-mpi 1 --comms-sequential
//
// Env:
//   IMPORT_CFG     NERSC or ILDG/LIME gauge (else hot start)
//   MASS_STRANGE   bare mass (default -0.2050)
//   CSW            clover coeff (default 1.20537)
//   LO_SWEEP       comma list of lo values (default 1e-4,1e-3,1e-2,1e-1)
//   HI             rational upper bound (default 100.0)
//   DEGREE         rational degree / #poles (default 20)
//   POLE_TOL       per-pole solve tolerance (default 1e-6 = production md tol)
//   REMEZ_PREC     Remez mantissa bits (default 64)
//   MAXIT          CG max iters (default 20000)
//   SKIP_GRID      set => QUDA only (skip the Grid reference)

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShift.h>
#include <Grid/qcd/action/pseudofermion/EvenOddSchurDifferentiable.h>

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <vector>

using namespace Grid;
typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

static double env_d(const char *k, double dflt) {
  const char *v = std::getenv(k);
  return (v && *v) ? std::atof(v) : dflt;
}
static long env_i(const char *k, long dflt) {
  const char *v = std::getenv(k);
  return (v && *v) ? std::atol(v) : dflt;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian Grid4(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid(&Grid4);

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});
  Quda::initialize();

  const RealD mass = env_d("MASS_STRANGE", -0.2050);
  const RealD csw  = env_d("CSW", 1.20537);

  // ── Gauge ────────────────────────────────────────────────────────────────
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

  // ── Sweep config ───────────────────────────────────────────────────────────
  const RealD hi       = env_d("HI", 100.0);
  const int   degree   = (int)env_i("DEGREE", 20);
  const RealD pole_tol = env_d("POLE_TOL", 1e-6);
  const int   rprec    = (int)env_i("REMEZ_PREC", 64);
  const int   maxit    = (int)env_i("MAXIT", 20000);
  const bool  skip_grid = (std::getenv("SKIP_GRID") != nullptr);

  std::vector<RealD> los;
  if (const char *s = std::getenv("LO_SWEEP"); s && *s) {
    std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) los.push_back(std::atof(tok.c_str()));
  } else {
    los = {1e-4, 1e-3, 1e-2, 1e-1};
  }

  std::cout << GridLogMessage << "mass=" << mass << " csw=" << csw
            << " hi=" << hi << " degree=" << degree << " pole_tol=" << pole_tol
            << " skip_grid=" << (skip_grid ? 1 : 0) << std::endl;

  // ── Grid operator + shared source (fixed gauge across the whole sweep) ──────
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF Dgrid(U, Grid4, RBGrid, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);
  SchurDifferentiableOperator<WilsonImplR> Mpc(Dgrid);

  LatticeFermion src_full(&Grid4); random(pRNG, src_full);
  LatticeFermion src(&RBGrid); src.Checkerboard() = Even;
  pickCheckerboard(Even, src, src_full);

  QudaCloverParams qp;
  qp.mass = mass; qp.csw = csw; qp.anti_periodic_t = true;
  qp.tol = pole_tol; qp.max_iter = maxit;
  qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  std::cout << GridLogMessage
            << "############ LO SWEEP (real RHMC poles) ############" << std::endl;

  for (RealD lo : los) {
    // ── Build the EXACT poles the strange force solves ──────────────────────
    AlgRemez remez(lo, hi, rprec);
    remez.generateApprox(degree, 1, 2);          // x^(1/2)
    MultiShiftFunction PowerNegHalf;
    PowerNegHalf.Init(remez, pole_tol, true);    // -> x^(-1/2), poles = shifts
    std::vector<RealD> shifts = PowerNegHalf.poles;
    const int N = (int)shifts.size();
    RealD pmin = *std::min_element(shifts.begin(), shifts.end());
    RealD pmax = *std::max_element(shifts.begin(), shifts.end());

    std::cout << GridLogMessage << "---- lo=" << std::scientific << std::setprecision(3) << lo
              << "  N=" << N << "  min_pole=" << pmin << "  max_pole=" << pmax << " ----" << std::endl;

    // ── QUDA multishift on those poles ──────────────────────────────────────
    QudaCloverMultiShiftSpec spec;
    spec.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    spec.shifts = shifts;
    spec.tols.assign(N, pole_tol);

    QudaCloverMultiShiftInverter quda_ms(&Grid4, qp, spec);
    quda_ms.SetGauge(U);
    std::vector<LatticeFermion> sols(N, LatticeFermion(&RBGrid));
    for (auto &s : sols) { s.Checkerboard() = Even; s = Zero(); }
    quda_ms.solve_rb_even(src, sols, /*make_resident=*/false);
    auto qres = quda_ms.LastResPerShift();

    int qconv = 0;
    for (int k = 0; k < N; ++k) {
      bool c = (qres[k] < 10 * pole_tol); if (c) ++qconv;
      std::cout << GridLogMessage << "  [QUDA] pole[" << k << "]=" << std::scientific
                << std::setprecision(3) << shifts[k] << "  true_res=" << qres[k]
                << (c ? "  OK" : "  FAIL") << std::endl;
    }

    // ── Grid reference on the SAME poles ────────────────────────────────────
    int gconv = 0;
    if (!skip_grid) {
      std::vector<LatticeFermion> solsg(N, LatticeFermion(&RBGrid));
      for (auto &s : solsg) { s.Checkerboard() = Even; s = Zero(); }
      ConjugateGradientMultiShift<LatticeFermion> msCG(maxit, PowerNegHalf);
      msCG(Mpc, src, solsg);
      LatticeFermion tmp(&RBGrid), r(&RBGrid);
      for (int k = 0; k < N; ++k) {
        Mpc.HermOp(solsg[k], tmp);
        tmp = tmp + shifts[k] * solsg[k];
        r = tmp - src;
        RealD tr = std::sqrt(norm2(r) / norm2(src));
        bool c = (tr < 10 * pole_tol); if (c) ++gconv;
        std::cout << GridLogMessage << "  [GRID] pole[" << k << "]=" << std::scientific
                  << std::setprecision(3) << shifts[k] << "  true_res=" << tr
                  << (c ? "  OK" : "  FAIL") << std::endl;
      }
    }

    std::cout << GridLogMessage << "SWEEP  lo=" << std::scientific << std::setprecision(3) << lo
              << "  min_pole=" << pmin << "  QUDA=" << qconv << "/" << N
              << "  Grid=" << (skip_grid ? -1 : gconv) << "/" << N << std::endl;
  }

  Quda::finalize();
  Grid_finalize();
  return 0;
}
