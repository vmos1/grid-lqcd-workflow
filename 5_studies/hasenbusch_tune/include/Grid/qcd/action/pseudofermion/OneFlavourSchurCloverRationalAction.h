#pragma once
// One-flavour Schur-complement RHMC for Wilson-Clover.
//
// det(M†M)^{1/2} = det(Mee†Mee)^{1/2} × det(Mpc†Mpc)^{1/2}
//
// This action handles: det(Mpc†Mpc)^{1/2} via rational approximation.
// Paired with QCDLogDetCloverEOAction(nf=1) for the even-even block.
//
// Unlike OneFlavourEvenOddRationalPseudoFermionAction, this does NOT assume ConstEE.
// The force includes clover corrections (MooDeriv, MeeDeriv chain rule).

#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
// A3: every ImportGauge into FermOp goes through the exact-skip guard (HASEN_GRID_IMPORT_SKIP=1,
// default off = plain ImportGauge). The strange operator is also imported by the MP deriv and by
// the strange log-det action, all guarded, so the guard sees ALL imports into it (hazard 1).
#include "gauge_import/import_guard.h"

NAMESPACE_BEGIN(Grid);

// Separate force (MD) approximation and run-time bounds check (2026-10-05, L191; grid_qcd docs
// 2026_10_02_agent1_pure_grid_physics_review.md s.3-4). A default-constructed value reproduces the
// behaviour before 10-05 bit for bit: x^(-1/2) at param.degree and param.tolerance, no check. The
// OneFlavourRationalParams fields mdtolerance, BoundsCheckFreq and BoundsCheckTol stay unread
// (every caller passes legacy values there); these replace them. Also used by
// OneFlavourSchurCloverRationalActionEven.h. The driver tests the macro, so it still builds
// against the Grid-TXQCD fork copies of these headers, which lack it.
#define HASEN_STRANGE_RATIONAL_EXTRAS 1
struct OneFlavourSchurRationalExtras {
  int   md_degree = 0;              // degree of the force approximation x^(-1/2); <= 0: param.degree
  RealD md_tolerance = 0.0;         // tolerance of the force multishift; <= 0: param.tolerance
  int   bounds_check_freq = 0;      // 0: off; n: check at refreshes 1, 1+n, 1+2n, ...
  RealD bounds_check_tol = 1e-6;    // margin on err_action / err_md above p x Remez error (below)
  bool  bounds_check_abort = false; // false: WARNING line and continue; true: abort
};

// Run-time spectral bounds check, modelled on stock Grid's HighBoundCheck and
// InverseSqrtBoundsCheck / InversePowerBoundsCheck (Grid/qcd/action/pseudofermion/Bounds.h, called
// from OneFlavourEvenOddRational.h S()), but reporting instead of asserting. On the probe X (the
// refresh's Gaussian noise eta, so no random number is drawn and the chain is unchanged), with
// Q = Mpc^dag Mpc and every solve fp64 at the action tolerance param.tolerance:
//   lambda_max    200 power iterations of Q from X (Rayleigh quotient, as PowerMethod.h); < hi
//   err_action    |X - Q r_-1/4(Q)^4 X| / |X|, r_-1/4 = the action/heatbath approximation (4 solves)
//   err_md        |X - Q r_-1/2(Q)^2 X| / |X|, r_-1/2 = the force approximation (2 solves)
//   force_action  |r_-1/4(Q)^2 X - r_-1/2(Q) X| / |r_-1/2(Q) X| (no extra solve; information)
// Each approximation is good to its Remez error delta (relative, AlgRemez::generateApprox) on
// [lo, hi] only: there |1 - x r(x)^p| <= p delta, so in range err_action <= 4 delta_action and
// err_md <= 2 delta_md plus the solver floor, while an eigenvalue of Q below lo or above hi lifts
// them far above it (stock's threshold is tolerance*100). PASS = lambda_max < hi,
// err_action < lim_action = tol + 4 delta_action and err_md < lim_md = tol + 2 delta_md (so a
// deliberately low RAT_DEGREE_MD, e.g. 12 with delta 1.4e-6 on [1e-4, 100], does not fire on its
// own error). One parseable "[StrangeBoundsCheck]" line per check; on FAIL a WARNING line, then
// GRID_ASSERT if abort is set.
template <class Field>
bool OneFlavourSchurRationalBoundsCheck(LinearOperatorBase<Field> &Q, const Field &eta,
                                        MultiShiftFunction &NegQuarter,
                                        const MultiShiftFunction &NegHalf,
                                        RealD remez_action, RealD remez_md,
                                        const OneFlavourRationalParams &param,
                                        const OneFlavourSchurRationalExtras &x,
                                        const std::string &who, long refresh) {
  GridBase *g = eta.Grid();
  Field X(g), A(g), B(g), R2(g);
  X = eta;
  const RealD nx = norm2(X);

  RealD lambda_max = 0.0;
  A = X;
  for (int i = 0; i < 200; i++) {
    A = A * (1.0 / std::sqrt(norm2(A)));
    Q.HermOp(A, B);
    lambda_max = real(innerProduct(A, B));
    A = B;
  }

  ConjugateGradientMultiShift<Field> msA(param.MaxIter, NegQuarter);
  msA(Q, X, A);   // r_-1/4 X
  msA(Q, A, B);   // r_-1/4^2 X: what S uses for Q^-1/2
  R2 = B;
  msA(Q, B, A);
  msA(Q, A, B);   // r_-1/4^4 X
  Q.HermOp(B, A);
  B = X - A;
  const RealD err_action = std::sqrt(norm2(B) / nx);

  MultiShiftFunction md = NegHalf;
  for (auto &t : md.tolerances) t = param.tolerance;
  ConjugateGradientMultiShift<Field> msM(param.MaxIter, md);
  msM(Q, X, A);   // r_-1/2 X
  B = R2 - A;
  const RealD force_action = std::sqrt(norm2(B) / norm2(A));
  msM(Q, A, B);   // r_-1/2^2 X
  Q.HermOp(B, A);
  B = X - A;
  const RealD err_md = std::sqrt(norm2(B) / nx);

  const RealD lim_action = x.bounds_check_tol + 4.0 * remez_action;
  const RealD lim_md = x.bounds_check_tol + 2.0 * remez_md;
  const bool pass = lambda_max < param.hi && err_action < lim_action && err_md < lim_md;
  std::ostringstream os;
  os << std::setprecision(6) << "[StrangeBoundsCheck] action=" << who << " refresh=" << refresh
     << " lo=" << param.lo << " hi=" << param.hi << " degree=" << param.degree
     << " degree_md=" << NegHalf.order << " lambda_max=" << lambda_max
     << " err_action=" << err_action << " err_md=" << err_md << " force_action=" << force_action
     << " remez_action=" << remez_action << " remez_md=" << remez_md << " tol=" << x.bounds_check_tol
     << " lim_action=" << lim_action << " lim_md=" << lim_md
     << " status=" << (pass ? "PASS" : "FAIL");
  std::cout << GridLogMessage << os.str() << std::endl;
  if (!pass) {
    std::cout << GridLogMessage << "[StrangeBoundsCheck] WARNING: the strange rational approximation"
              << " on [" << param.lo << ", " << param.hi << "] does not cover the spectrum of"
              << " Q = Mpc^dag Mpc (lambda_max >= hi, or |1 - Q r(Q)^p| >= lim: an eigenvalue below"
              << " lo or above hi). S, heatbath and force are wrong on those modes and the force is"
              << " not -dS/dU there (L191). Lower RAT_LO / raise RAT_HI." << std::endl;
    if (x.bounds_check_abort) {
      std::cout << GridLogMessage << "[StrangeBoundsCheck] RAT_BOUNDS_CHECK_ABORT=1: aborting."
                << std::endl;
      GRID_ASSERT(pass && "strange rational bounds check failed (RAT_BOUNDS_CHECK_ABORT=1)");
    }
  }
  return pass;
}

template <class Impl,
          class FermionOp = WilsonCloverFermion<Impl, Grid::CloverHelpers<Impl>>>
class OneFlavourSchurCloverRationalAction
    : public Action<typename Impl::GaugeField> {
public:
  INHERIT_IMPL_TYPES(Impl);

  typedef OneFlavourRationalParams Params;
  typedef FermionOp FermionOperator;

  Params param;
  OneFlavourSchurRationalExtras extras;
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  MultiShiftFunction PowerNegHalf;  // the force (MD) approximation: extras.md_degree / md_tolerance
  RealD RemezErrorAction = 0.0;     // AlgRemez errors of x^(1/4) and of the force's x^(1/2)
  RealD RemezErrorMD = 0.0;

protected:
  FermionOperator &FermOp;
  FermionField PhiOdd;
  long refreshes = 0;

public:
  OneFlavourSchurCloverRationalAction(FermionOperator &Op, Params &p,
                                      const OneFlavourSchurRationalExtras &x =
                                          OneFlavourSchurRationalExtras())
      : FermOp(Op), PhiOdd(Op.FermionRedBlackGrid()), param(p), extras(x) {
    AlgRemez remez(param.lo, param.hi, param.precision);
    const int md_degree = extras.md_degree > 0 ? extras.md_degree : param.degree;
    const RealD md_tol = extras.md_tolerance > 0.0 ? extras.md_tolerance : param.tolerance;

    std::cout << GridLogMessage << "Generating degree " << param.degree
              << " for x^(1/4)" << std::endl;
    RemezErrorAction = remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);

    std::cout << GridLogMessage << "Generating degree " << md_degree
              << " for x^(1/2)" << std::endl;
    RemezErrorMD = remez.generateApprox(md_degree, 1, 2);
    PowerNegHalf.Init(remez, md_tol, true);
  }

  std::string action_name() override {
    return "OneFlavourSchurCloverRationalAction";
  }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] lo=" << param.lo
       << " hi=" << param.hi << " degree=" << param.degree << std::endl;
    return os.str();
  }

  void refresh(const GaugeField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    RealD scale = std::sqrt(0.5);

    FermionField eta(FermOp.FermionGrid());
    FermionField etaOdd(FermOp.FermionRedBlackGrid());

    gaussian(pRNG, eta);
    eta = eta * scale;
    pickCheckerboard(Odd, etaOdd, eta);

    GuardedImportGauge(FermOp, U);

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter, PowerQuarter);
    msCG(Mpc, etaOdd, PhiOdd);

    const long n = ++refreshes;
    if (extras.bounds_check_freq > 0 && (n - 1) % extras.bounds_check_freq == 0)
      OneFlavourSchurRationalBoundsCheck<FermionField>(Mpc, etaOdd, PowerNegQuarter, PowerNegHalf,
                                                       RemezErrorAction, RemezErrorMD, param,
                                                       extras, action_name(), n);
  }

  RealD S(const GaugeField &U) override {
    GuardedImportGauge(FermOp, U);

    FermionField Y(FermOp.FermionRedBlackGrid());

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter,
                                                   PowerNegQuarter);
    msCG(Mpc, PhiOdd, Y);

    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    const int Npole = PowerNegHalf.poles.size();

    std::vector<FermionField> MPhi_k(Npole, FermOp.FermionRedBlackGrid());

    FermionField X(FermOp.FermionRedBlackGrid());
    FermionField Y(FermOp.FermionRedBlackGrid());

    GaugeField tmp(FermOp.GaugeGrid());
    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();

    GuardedImportGauge(FermOp, U);

    SchurDifferentiableOperator<Impl> Mpc(FermOp);
    ConjugateGradientMultiShift<FermionField> msCG(param.MaxIter, PowerNegHalf);
    msCG(Mpc, PhiOdd, MPhi_k);

    dSdU = Zero();
    for (int k = 0; k < Npole; k++) {
      RealD ak = PowerNegHalf.residues[k];

      X = MPhi_k[k];
      Mpc.Mpc(X, Y);

      // 1. Hopping forces
      Mpc.MpcDeriv(tmp, Y, X);
      dSdU = dSdU + ak * tmp;
      Mpc.MpcDagDeriv(tmp, X, Y);
      dSdU = dSdU + ak * tmp;

      // 2. Odd-site clover force
      FermOp.MooDeriv(tmp, Y, X, DaggerNo);
      dSdU = dSdU + ak * tmp;
      FermOp.MooDeriv(tmp, X, Y, DaggerYes);
      dSdU = dSdU + ak * tmp;

      // 3. Even-site clover force (chain rule through Mee^{-1})
      FermionField W_e(fcbgrid), Z_e(fcbgrid), tmp1(fcbgrid);

      FermOp.Meooe(X, tmp1);
      FermOp.MooeeInv(tmp1, W_e);

      FermOp.MeooeDag(Y, tmp1);
      FermOp.MooeeInvDag(tmp1, Z_e);

      FermOp.MeeDeriv(tmp, Z_e, W_e, DaggerNo);
      dSdU = dSdU + ak * tmp;
      FermOp.MeeDeriv(tmp, W_e, Z_e, DaggerYes);
      dSdU = dSdU + ak * tmp;
    }
  }
};

NAMESPACE_END(Grid);
