// Spectrum probe for the strange Schur operator (Mpc^dag Mpc) on a config.
//
// Pins the conditioning that the strange RHMC multishift sees:
//   lambda_max  via Grid's PowerMethod (matrix applies only, cheap)
//   lambda_min  via inverse iteration (CG solve per step) + Rayleigh quotient
//
// The decisive number: is the RHMC lower bound lo=1e-4 REAL (the operator
// genuinely has eigenvalues down near 1e-4 => lo can't be raised) or OVERLY
// AGGRESSIVE (lambda_min >> 1e-4 => lo manufactures poles below the spectrum,
// and raising lo may let QUDA's multishift converge)?  See lo/lambda_min at end.
//
// Run (2 nodes, A100) — pure Grid, NO QUDA:
//   IMPORT_CFG=.../cfg_2000.lime srun ... ./eig_probe_strange \
//       --grid 48.48.48.96 --mpi 1.2.2.2 --accelerator-threads 8 --shm 512
//
// Env:
//   IMPORT_CFG     NERSC or ILDG/LIME gauge (else hot start)
//   MASS_STRANGE   bare mass (default -0.2050)
//   CSW            clover coeff (default 1.20537)
//   EIG_OUTER      inverse-iteration steps (default 20)
//   EIG_CG_TOL     inner CG tolerance (default 1e-5)
//   EIG_CG_MAXIT   inner CG max iters (default 20000)

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/EvenOddSchurDifferentiable.h>

#include <cstdlib>

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

  const RealD mass = env_d("MASS_STRANGE", -0.2050);
  const RealD csw  = env_d("CSW", 1.20537);

  // ── Gauge: load thermalized config, else hot start. ──────────────────────
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

  // ── Strange Schur operator (same as OneFlavourSchurCloverRationalActionEven) ──
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;   // antiperiodic-t
  WCF D(U, Grid4, RBGrid, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);
  SchurDifferentiableOperator<WilsonImplR> Mpc(D);   // HermOp = Mpc^dag Mpc

  // Even-checkerboard source.
  LatticeFermion src_full(&Grid4); random(pRNG, src_full);
  LatticeFermion x(&RBGrid); x.Checkerboard() = Even;
  pickCheckerboard(Even, x, src_full);

  // ── lambda_max ────────────────────────────────────────────────────────────
  std::cout << GridLogMessage << "=== PowerMethod: lambda_max(Mpc^dag Mpc) ===" << std::endl;
  PowerMethod<LatticeFermion> pm;
  RealD lmax = pm(Mpc, x);

  // ── lambda_min via inverse iteration: x_{k+1} = normalize(A^{-1} x_k) ──────
  const int   outer = (int)env_i("EIG_OUTER", 20);
  const RealD cgtol = env_d("EIG_CG_TOL", 1e-5);
  const int   cgmax = (int)env_i("EIG_CG_MAXIT", 20000);
  ConjugateGradient<LatticeFermion> CG(cgtol, cgmax);

  pickCheckerboard(Even, x, src_full);
  { RealD n = std::sqrt(norm2(x)); x = x * (1.0 / n); }
  LatticeFermion y(&RBGrid),  Ax(&RBGrid);
  y.Checkerboard() = Even; Ax.Checkerboard() = Even;

  RealD lmin = 0.0;
  std::cout << GridLogMessage << "=== inverse iteration: lambda_min (Rayleigh) ===" << std::endl;
  for (int k = 0; k < outer; k++) {
    y = Zero(); y.Checkerboard() = Even;     // fresh CG (RHS changes each step)
    CG(Mpc, x, y);                           // solve (Mpc^dag Mpc) y = x
    { RealD n = std::sqrt(norm2(y)); y = y * (1.0 / n); }
    x = y;
    Mpc.HermOp(x, Ax);
    lmin = real(innerProduct(x, Ax)) / norm2(x);
    std::cout << GridLogMessage << "  iter " << k << "  lambda_min = "
              << std::scientific << std::setprecision(6) << lmin << std::endl;
  }

  // ── Verdict ────────────────────────────────────────────────────────────────
  std::cout << GridLogMessage << "================ SPECTRUM ================" << std::endl;
  std::cout << GridLogMessage << "mass=" << mass << " csw=" << csw << std::endl;
  std::cout << GridLogMessage << "lambda_max       = " << std::scientific << std::setprecision(6) << lmax << std::endl;
  std::cout << GridLogMessage << "lambda_min       = " << lmin << std::endl;
  std::cout << GridLogMessage << "cond = lmax/lmin = " << lmax / lmin << std::endl;
  std::cout << GridLogMessage << "RHMC lo = 1e-4    => lo/lambda_min = " << 1e-4 / lmin << std::endl;
  std::cout << GridLogMessage << "  ( ~1  => lo is REAL (eigenvalues that small exist; can't raise lo);" << std::endl;
  std::cout << GridLogMessage << "    <<1 => lo OVERLY AGGRESSIVE (poles below the spectrum; raising lo may fix QUDA) )" << std::endl;

  Grid_finalize();
  return 0;
}
