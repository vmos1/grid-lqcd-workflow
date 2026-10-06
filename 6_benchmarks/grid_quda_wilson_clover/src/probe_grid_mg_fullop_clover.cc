// probe_grid_mg_fullop_clover.cc
//
// gq-mg step 1, the structural experiment (__docs/2026_09_29_grid_multigrid_experimentation.md §8):
// solve the even-odd Schur system with a multigrid hierarchy built from the FULL clover
// operator D, as QUDA does, instead of from the Schur operator itself (probe_grid_mg_schur_clover.cc).
//
// WHY IT IS EXACT. With D = [[M_ee, M_eo], [M_oe, M_oo]], the even-even block of D^-1 is the
// inverse of the Schur complement S = M_ee - M_eo M_oo^-1 M_oe, which is Grid's
// SchurDiagMooeeOperator. A coarse-grid approximation V Dc^-1 V^dag of D^-1 (Dc = V^dag D V) therefore
// gives an approximation of S^-1 by keeping its even-even block:
//     S^-1 r  ~=  [ V Dc^-1 V^dag (r on even sites, 0 on odd) ]_even
// That is exactly what QUDA does for a MATPC solve (multigrid.cpp MG::operator(): the transfer runs
// in parity mode and restricts the even residual with the even-site components of its full-lattice
// null vectors). Here the transfer runs on FULL-lattice fields (embed, project, prolong, pick even),
// which is the same arithmetic and allows blocks Grid's red-black grid cannot take (3.3.3.4).
// For the symmetric operator One = M_ee^-1 S, One^-1 = S^-1 M_ee, so the rhs is multiplied by M_ee
// first (--probe-schur one).
//
// WHAT CHANGES AGAINST THE SCHUR-OPERATOR PROBE, AND WHAT DOES NOT.
//   - null vectors: generated on the fp32 FULL operator D (same GCR inverse iteration and knobs)
//   - level-1 coarse operator: Galerkin projection of D, which is ONE hop, so the 9-point stencil
//     is EXACT and the fast stencil apply can be used (the Schur operator's 9-point operator is not)
//   - aggregation block: on the full lattice (--probe-block, default 3.3.3.4 = QUDA's)
//   - unchanged: fine smoother (GCR on the Schur operator + 0.01), level-1 GCR preconditioned by
//     a level-2 V-cycle, level-2 construction, fp64 outer flexible GCR on the Schur operator
//
// CLOVER-PRECONDITIONED HIERARCHY (env FULLOP_OPERATOR=precond; default D = the path above).
// QUDA's production setup coarsens A~ = Mdiag^-1 D, Mdiag = blockdiag(M_ee, M_oo) (multigrid.cpp
// preconditioned_coarsen), not D. A~ is still one hop, so the 9-point Galerkin stencil stays exact.
// Its diagonal blocks are the identity, so its even Schur complement is One = M_ee^-1 S itself and
//     (A~^-1)_ee = One^-1          (SCHUR=one: rhs used as is, no M_ee multiplication)
//     S^-1 = (A~^-1)_ee M_ee^-1    (SCHUR=mooee: rhs multiplied by M_ee^-1 first)
// Env FULLOP_NULLVEC=full|schur (default full with D, schur with precond):
//   full   GCR inverse iteration on the full-lattice operator being coarsened (D or A~)
//   schur  QUDA's scheme: the even halves x_e come from the same GCR iteration on the fp32 Schur
//          operator (red-black grid, same knobs and seeds as probe_grid_mg_schur_clover.cc), the odd
//          halves from the odd-site equation x_o = -M_oo^-1 M_oe x_e. Needs an even x block.
// Grid's MooeeInv accepts a full-lattice field directly (full-grid clover inverse), so A~ needs no
// checkerboard split; a boss line checks it against the per-checkerboard inverse.
//
// fp32 hierarchy only (the production case), fp64 outer. Knob names are the ones
// run_probe_grid_mg.sh already passes, so this binary runs through that wrapper with BIN=...;
// flags it does not use are ignored. Grid itself is not modified.

#include <Grid/Grid.h>

#include "bench_nvtx.h"
#include "probe_mg_solvers.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace Grid;

namespace {

#ifndef PROBE_NBASIS
#define PROBE_NBASIS 24
#endif
constexpr int kNbasis = PROBE_NBASIS; // gamma5-doubled to 2*kNbasis level-1 dof, as QUDA's 24 x 2
constexpr int kNbasis2 = 32;          // level-2 dof, QUDA production's second <NullVectors> value

// Shifted Schur operator for the smoother (same as probe_grid_mg_schur_clover.cc).
template <class Field>
class ShiftedSchurOperator : public LinearOperatorBase<Field> {
  SchurOperatorBase<Field> &schur_;
  RealD shift_;

public:
  ShiftedSchurOperator(SchurOperatorBase<Field> &schur, RealD shift) : schur_(schur), shift_(shift) {}
  void Op(const Field &in, Field &out) override
  {
    schur_.Mpc(in, out);
    out = out + shift_ * in;
  }
  void AdjOp(const Field &in, Field &out) override
  {
    schur_.MpcDag(in, out);
    out = out + shift_ * in;
  }
  void OpDiag(const Field &in, Field &out) override { GRID_ASSERT(0); }
  void OpDir(const Field &in, Field &out, int dir, int disp) override { GRID_ASSERT(0); }
  void OpDirAll(const Field &in, std::vector<Field> &out) override { GRID_ASSERT(0); }
  void HermOpAndNorm(const Field &in, Field &out, RealD &n1, RealD &n2) override { GRID_ASSERT(0); }
  void HermOp(const Field &in, Field &out) override { GRID_ASSERT(0); }
};

// A~ = Mdiag^-1 D on the full lattice (FULLOP_OPERATOR=precond), Mdiag = blockdiag(M_ee, M_oo).
// MooeeInv is applied to the full-grid field directly: CompactWilsonCloverFermion::MooeeInv uses its
// full-grid DiagonalInv/TriangleInv when the grid is not checkerboarded (the per-parity arrays are
// pickCheckerboard copies of them), WilsonFermion::MooeeInv is a scalar. Checked at run time.
class PrecondFullOperator : public LinearOperatorBase<LatticeFermionF> {
  WilsonFermionF &op_;
  LatticeFermionF tmp_;

public:
  PrecondFullOperator(WilsonFermionF &op, GridBase *full) : op_(op), tmp_(full) {}
  void Op(const LatticeFermionF &in, LatticeFermionF &out) override
  {
    op_.M(in, tmp_);
    op_.MooeeInv(tmp_, out);
  }
  void AdjOp(const LatticeFermionF &in, LatticeFermionF &out) override
  {
    op_.MooeeInvDag(in, tmp_);
    op_.Mdag(tmp_, out);
  }
  void OpDiag(const LatticeFermionF &in, LatticeFermionF &out) override { GRID_ASSERT(0); }
  void OpDir(const LatticeFermionF &in, LatticeFermionF &out, int dir, int disp) override { GRID_ASSERT(0); }
  void OpDirAll(const LatticeFermionF &in, std::vector<LatticeFermionF> &out) override { GRID_ASSERT(0); }
  void HermOpAndNorm(const LatticeFermionF &in, LatticeFermionF &out, RealD &n1, RealD &n2) override
  {
    GRID_ASSERT(0);
  }
  void HermOp(const LatticeFermionF &in, LatticeFermionF &out) override { GRID_ASSERT(0); }
};

// fp64 outer / fp32 preconditioner adaptor with the precision-change maps built once
// (the PERSISTENT_PRECCHANGE=1 path of the Schur-operator probe).
template <class FieldD, class FieldF>
class PrecisionChangeAdaptor : public LinearFunction<FieldD> {
public:
  using LinearFunction<FieldD>::operator();

private:
  LinearFunction<FieldF> &inner_;
  GridBase *fine_f_;
  std::unique_ptr<precisionChangeWorkspace> ws_down_, ws_up_;
  std::unique_ptr<FieldF> in_f_, out_f_;

public:
  PrecisionChangeAdaptor(LinearFunction<FieldF> &inner, GridBase *fine_f) : inner_(inner), fine_f_(fine_f) {}
  void operator()(const FieldD &in, FieldD &out) override
  {
    if (!ws_down_) {
      ws_down_.reset(new precisionChangeWorkspace(fine_f_, in.Grid()));
      ws_up_.reset(new precisionChangeWorkspace(in.Grid(), fine_f_));
      in_f_.reset(new FieldF(fine_f_));
      out_f_.reset(new FieldF(fine_f_));
    }
    in_f_->Checkerboard() = in.Checkerboard();
    out_f_->Checkerboard() = in.Checkerboard();
    precisionChange(*in_f_, in, *ws_down_);
    *out_f_ = Zero();
    out_f_->Checkerboard() = in.Checkerboard();
    inner_(*in_f_, *out_f_);
    precisionChange(out, *out_f_, *ws_up_);
    out.Checkerboard() = in.Checkerboard();
  }
};

// Level-2 V-cycle on the level-1 operator: the MGPreconditioner of Test_general_coarse_wilson.cc
// with the fused projection and persistent temporaries (same structure as the Schur probe's).
template <class Fobj, class CComplex, int nbasis>
class MGPreconditioner : public LinearFunction<Lattice<Fobj>> {
public:
  using LinearFunction<Lattice<Fobj>>::operator();
  typedef Aggregation<Fobj, CComplex, nbasis> Aggregates;
  typedef typename Aggregates::FineField FineField;
  typedef typename Aggregates::CoarseVector CoarseVector;

  Aggregates &agg_;
  LinearOperatorBase<FineField> &fine_;
  LinearFunction<FineField> &post_;
  LinearFunction<CoarseVector> &coarse_solve_;
  std::unique_ptr<CoarseVector> csrc_, csol_;
  std::unique_ptr<FineField> v1_, v2_;

  MGPreconditioner(Aggregates &agg, LinearOperatorBase<FineField> &fine, LinearFunction<FineField> &post,
                   LinearFunction<CoarseVector> &coarse_solve)
      : agg_(agg), fine_(fine), post_(post), coarse_solve_(coarse_solve)
  {
  }

  void operator()(const FineField &in, FineField &out) override
  {
    if (!csrc_) {
      csrc_.reset(new CoarseVector(agg_.CoarseGrid));
      csol_.reset(new CoarseVector(agg_.CoarseGrid));
      v1_.reset(new FineField(in.Grid()));
      v2_.reset(new FineField(in.Grid()));
    }
    // no pre-smoothing: out = 0, so the residual is `in`
    ProbeMG::blockProjectFused(*csrc_, in, agg_.subspace);
    *csol_ = Zero();
    coarse_solve_(*csrc_, *csol_);
    agg_.PromoteFromSubspace(*csol_, out);
    fine_.Op(out, *v1_);
    sub(*v1_, in, *v1_);
    *v2_ = Zero();
    post_(*v1_, *v2_);
    add(out, out, *v2_);
  }
};

// The new piece: a V-cycle for the even-odd Schur system whose coarse correction comes from a
// FULL-operator hierarchy (see the header comment for why it is exact).
//   rhs  = M_ee r  (D hierarchy, One form) or r (D, Mooee form; A~ hierarchy, One form)
//          or M_ee^-1 r (A~ hierarchy, Mooee form)
//   c    = V^dag (rhs on even, 0 on odd)          full-lattice projection
//   e    = Dc^-1 c                                level-1 GCR, preconditioned by level 2
//   out  = [V e]_even                             full-lattice prolongation, keep even sites
//   out += PostSmooth(r - S out)                  smoother on the Schur system, as QUDA
//
// Two transfer modes, same arithmetic:
//   full  embed on the full lattice, project with the full vectors, prolong, keep even sites.
//         Works for any block (3.3.3.4 included) but moves twice the data per transfer.
//   even  project/prolong the even residual with the EVEN-SITE HALVES of the full vectors on the
//         red-black grid (QUDA's parity mode). V^dag (r,0) = V_e^dag r and [V e]_even = V_e e, so
//         the result is identical, at the transfer cost of the Schur-operator hierarchy. Needs an
//         even x block (Grid's red-black grid halves x).
template <class CComplex, int nbasis>
class SchurFromFullMG : public LinearFunction<LatticeFermionF> {
public:
  using LinearFunction<LatticeFermionF>::operator();
  typedef Aggregation<vSpinColourVectorF, CComplex, nbasis> Aggregates;
  typedef typename Aggregates::CoarseVector CoarseVector;

  Aggregates &agg_;
  Aggregates *even_agg_; // non-null: even-site transfer
  LinearOperatorBase<LatticeFermionF> &schur_;
  WilsonFermionF *mee_;    // non-null: rhs = M_ee r    (D hierarchy, One form)
  WilsonFermionF *meeinv_; // non-null: rhs = M_ee^-1 r (A~ hierarchy, Mooee form)
  LinearFunction<CoarseVector> &coarse_solve_;
  LinearFunction<LatticeFermionF> &post_;
  std::unique_ptr<LatticeFermionF> full_, rhs_, v1_, v2_;
  std::unique_ptr<CoarseVector> csrc_, csol_;

  SchurFromFullMG(Aggregates &agg, Aggregates *even_agg, LinearOperatorBase<LatticeFermionF> &schur,
                  WilsonFermionF *mee, WilsonFermionF *meeinv, LinearFunction<CoarseVector> &coarse_solve,
                  LinearFunction<LatticeFermionF> &post)
      : agg_(agg), even_agg_(even_agg), schur_(schur), mee_(mee), meeinv_(meeinv), coarse_solve_(coarse_solve),
        post_(post)
  {
  }

  void operator()(const LatticeFermionF &in, LatticeFermionF &out) override
  {
    const int cb = in.Checkerboard();
    if (!full_) {
      full_.reset(new LatticeFermionF(agg_.FineGrid));
      rhs_.reset(new LatticeFermionF(in.Grid()));
      v1_.reset(new LatticeFermionF(in.Grid()));
      v2_.reset(new LatticeFermionF(in.Grid()));
      csrc_.reset(new CoarseVector(agg_.CoarseGrid));
      csol_.reset(new CoarseVector(agg_.CoarseGrid));
    }
    rhs_->Checkerboard() = cb;
    v1_->Checkerboard() = cb;
    v2_->Checkerboard() = cb;
    out.Checkerboard() = cb;
    const LatticeFermionF *rhs = &in;
    if (mee_) {
      mee_->Mooee(in, *rhs_);
      rhs = rhs_.get();
    } else if (meeinv_) {
      meeinv_->MooeeInv(in, *rhs_);
      rhs = rhs_.get();
    }
    if (even_agg_) {
      ProbeMG::blockProjectFused(*csrc_, *rhs, even_agg_->subspace);
      *csol_ = Zero();
      coarse_solve_(*csrc_, *csol_);
      even_agg_->PromoteFromSubspace(*csol_, out);
    } else {
      *full_ = Zero();
      setCheckerboard(*full_, *rhs);
      ProbeMG::blockProjectFused(*csrc_, *full_, agg_.subspace);
      *csol_ = Zero();
      coarse_solve_(*csrc_, *csol_);
      agg_.PromoteFromSubspace(*csol_, *full_);
      pickCheckerboard(cb, out, *full_);
    }
    schur_.Op(out, *v1_);
    sub(*v1_, in, *v1_);
    *v2_ = Zero();
    post_(*v1_, *v2_);
    add(out, out, *v2_);
  }
};

std::string read_string(int argc, char **argv, const std::string &option, const std::string &fallback)
{
  char **itr = std::find(argv, argv + argc, option);
  if (itr != argv + argc && ++itr != argv + argc) return std::string(*itr);
  return fallback;
}
double read_double(int argc, char **argv, const std::string &option, double fallback)
{
  const std::string payload = read_string(argc, argv, option, "");
  return payload.empty() ? fallback : std::stod(payload);
}
int read_int(int argc, char **argv, const std::string &option, int fallback)
{
  const std::string payload = read_string(argc, argv, option, "");
  return payload.empty() ? fallback : std::stoi(payload);
}
double median_of(std::vector<double> v)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
Coordinate parse_block(const std::string &spec)
{
  Coordinate block(4);
  std::size_t pos = 0;
  for (int d = 0; d < 4; ++d) {
    const std::size_t dot = spec.find('.', pos);
    block[d] = std::stoi(spec.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos));
    pos = (dot == std::string::npos) ? pos : dot + 1;
  }
  return block;
}
std::string coordinate_string(const Coordinate &c)
{
  std::string s;
  for (std::size_t d = 0; d < c.size(); ++d) s += (d ? "." : "") + std::to_string(c[d]);
  return s;
}
double seconds_since(double t0) { return (usecond() - t0) / 1.0e6; }

} // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  const std::string action_name = read_string(argc, argv, "--probe-action", "clover");
  const std::string schur_name = read_string(argc, argv, "--probe-schur", "one");
  const std::string cb_name = read_string(argc, argv, "--probe-checkerboard", "even");
  const Coordinate Block = parse_block(read_string(argc, argv, "--probe-block", "3.3.3.4"));
  const Coordinate Block2 = parse_block(read_string(argc, argv, "--probe-block2", "2.2.2.3"));
  const double mass = read_double(argc, argv, "--probe-mass", -0.2416);
  const double csw = read_double(argc, argv, "--probe-csw", 1.20536588031793);
  const double tol = read_double(argc, argv, "--probe-tol", 1.0e-11);
  const int maxiter = read_int(argc, argv, "--probe-maxiter", 200);
  const int solve_repeats = read_int(argc, argv, "--probe-solve-repeats", 3);
  const int stout_nsmear = read_int(argc, argv, "--probe-stout-nsmear", 0);
  const double stout_rho = read_double(argc, argv, "--probe-stout-rho", 0.125);
  const std::string cfg = read_string(argc, argv, "--probe-cfg", "");
  // level-1 null vectors (FULLOP_NULLVEC: on the full-lattice operator, or on the Schur operator)
  const double subspace_tol = read_double(argc, argv, "--probe-subspace-tol", 1.0e-3);
  const int subspace_rounds = read_int(argc, argv, "--probe-subspace-rounds", 3);
  const int subspace_mmax = read_int(argc, argv, "--probe-subspace-mmax", 10);
  const int subspace_maxiter = read_int(argc, argv, "--probe-subspace-maxiter", 30);
  // fine smoother, on the Schur operator
  const int smoother_mmax = read_int(argc, argv, "--probe-smoother-mmax", 1);
  const int smoother_nstep = read_int(argc, argv, "--probe-smoother-nstep", 8);
  const double smoother_tol = read_double(argc, argv, "--probe-smoother-tol", 0.1);
  const int smoother_maxiter = read_int(argc, argv, "--probe-smoother-maxiter", 1);
  // level-1 solve
  const double coarse_tol = read_double(argc, argv, "--probe-coarse-tol", 0.2);
  const int coarse_maxiter = read_int(argc, argv, "--probe-coarse-maxiter", 1);
  const int coarse_nstep = read_int(argc, argv, "--probe-coarse-nstep", 12);
  const int coarse_mmax = read_int(argc, argv, "--probe-coarse-mmax", 12);
  // level 2
  const double l2_subspace_tol = read_double(argc, argv, "--probe-l2-subspace-tol", 1.0e-3);
  const int l2_subspace_rounds = read_int(argc, argv, "--probe-l2-subspace-rounds", 3);
  const int l2_subspace_maxiter = read_int(argc, argv, "--probe-l2-subspace-maxiter", 10);
  const int l2_smoother_nstep = read_int(argc, argv, "--probe-l2-smoother-nstep", 2);
  const double l2_smoother_tol = read_double(argc, argv, "--probe-l2-smoother-tol", 0.1);
  const double l2_coarse_tol = read_double(argc, argv, "--probe-l2-coarse-tol", 0.2);
  const int l2_coarse_maxiter = read_int(argc, argv, "--probe-l2-coarse-maxiter", 1);
  const int l2_coarse_nstep = read_int(argc, argv, "--probe-l2-coarse-nstep", 4);
  const int l2_coarse_mmax = read_int(argc, argv, "--probe-l2-coarse-mmax", 4);
  // outer
  const int outer_mmax = read_int(argc, argv, "--probe-outer-mmax", 6);
  const int outer_nstep = read_int(argc, argv, "--probe-outer-nstep", 6);

  GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()),
                                                        GridDefaultMpi());
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  const bool boss = UGrid->IsBoss();
  auto fail = [&](const std::string &msg) {
    if (boss) std::cerr << "probe: " << msg << std::endl;
    Grid_finalize();
    return 2;
  };
  if (cb_name != "even") return fail("--probe-checkerboard must be even");
  if (schur_name != "one" && schur_name != "mooee") return fail("--probe-schur must be one|mooee");
  if (action_name != "clover" && action_name != "wilson") return fail("--probe-action must be clover|wilson");
  const int cb = Even;
  // Hierarchy operator and null-vector scheme (see the header comment). Env, like FULLOP_TRANSFER;
  // unset or empty selects the default.
  const char *operator_env = std::getenv("FULLOP_OPERATOR");
  const std::string fullop_operator = (operator_env && *operator_env) ? operator_env : "D";
  if (fullop_operator != "D" && fullop_operator != "precond") return fail("FULLOP_OPERATOR must be D|precond");
  const bool precond = fullop_operator == "precond";
  const char *nullvec_env = std::getenv("FULLOP_NULLVEC");
  const std::string nullvec = (nullvec_env && *nullvec_env) ? nullvec_env : (precond ? "schur" : "full");
  if (nullvec != "full" && nullvec != "schur") return fail("FULLOP_NULLVEC must be full|schur");
  if (nullvec == "schur" && Block[0] % 2 != 0) return fail("FULLOP_NULLVEC=schur needs an even x block");

  // Level-1 coarse grid from the FULL lattice: no red-black constraint on the x block.
  Coordinate clatt = GridDefaultLatt();
  for (int d = 0; d < 4; ++d) {
    if (clatt[d] % Block[d] != 0) return fail("lattice not divisible by --probe-block in dim " + std::to_string(d));
    clatt[d] /= Block[d];
  }
  GridCartesian *UGridF = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexF::Nsimd()),
                                                         GridDefaultMpi());
  GridRedBlackCartesian *UrbGridF = SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);
  GridCartesian *Coarse4dF = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplexF::Nsimd()),
                                                            GridDefaultMpi());
  subdivides(Coarse4dF, UGridF); // aborts early on a bad block
  if (nullvec == "schur") subdivides(Coarse4dF, UrbGridF);

  if (boss) {
    std::cout << GridLogMessage << "=== probe_grid_mg_fullop_clover ===" << std::endl;
    if (precond)
      std::cout << GridLogMessage
                << "hierarchy         clover-preconditioned operator Mdiag^-1 M (1 hop, 9-point Galerkin exact)"
                << std::endl;
    else
      std::cout << GridLogMessage << "hierarchy         FULL operator D (1 hop, 9-point Galerkin exact)" << std::endl;
    if (nullvec == "schur")
      std::cout << GridLogMessage << "null vectors      schur (even-site GCR, odd sites reconstructed)" << std::endl;
    else
      std::cout << GridLogMessage << "null vectors      full (GCR on " << (precond ? "Mdiag^-1 M" : "D") << ")"
                << std::endl;
    std::cout << GridLogMessage << "action            " << action_name << std::endl;
    std::cout << GridLogMessage << "schur convention  " << schur_name << " (outer solve and smoother)" << std::endl;
    std::cout << GridLogMessage << "aggregation block " << coordinate_string(Block) << " (full lattice) -> level 1 "
              << coordinate_string(clatt) << ", rdim " << coordinate_string(Coarse4dF->_rdimensions) << std::endl;
    std::cout << GridLogMessage << "nbasis            " << kNbasis << " -> coarse dof " << 2 * kNbasis
              << " (gamma5 doubled)" << std::endl;
    std::cout << GridLogMessage << "subspace tol/rounds/mmax/maxiter " << subspace_tol << "/" << subspace_rounds << "/"
              << subspace_mmax << "/" << subspace_maxiter << std::endl;
    std::cout << GridLogMessage << "smoother nstep/tol " << smoother_nstep << "/" << smoother_tol << "  coarse nstep/tol "
              << coarse_nstep << "/" << coarse_tol << "  outer mmax/nstep " << outer_mmax << "/" << outer_nstep
              << "  tol " << tol << std::endl;
  }

  // ---- gauge field, smearing, operators (as probe_grid_mg_schur_clover.cc) ----
  std::vector<int> seeds4({1, 2, 3, 4});
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(seeds4);
  LatticeGaugeFieldD Umu(UGrid);
  if (cfg.empty()) {
    SU<Nc>::HotConfiguration(RNG4, Umu);
  } else {
    FieldMetaData header;
    IldgReader reader;
    reader.open(cfg);
    reader.readConfiguration(Umu, header);
    reader.close();
  }
  // ⛔ avgPlaquette is a global reduction: every rank must call it. Calling it inside `if (boss)`
  // deadlocks every multi-rank run (2026-09-29: the first C3 launch hung here for 55 minutes
  // until the allocation's time limit; the 1-rank C2 shakedown could not show it).
  const double plaquette = WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu);
  if (boss)
    std::cout << GridLogMessage << "gauge " << (cfg.empty() ? "hot" : cfg) << " plaquette " << plaquette << std::endl;
  if (stout_nsmear > 0) {
    Smear_Stout<PeriodicGimplD> stout(stout_rho);
    LatticeGaugeFieldD Usmear(UGrid);
    for (int n = 0; n < stout_nsmear; ++n) {
      stout.smear(Usmear, Umu);
      Umu = Usmear;
    }
    const double plaq_smeared = WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu);
    if (boss)
      std::cout << GridLogMessage << "stout smearing rho " << stout_rho << " n_smear " << stout_nsmear
                << " -> smeared plaquette " << plaq_smeared << std::endl;
  }
  std::vector<Complex> phases(Nd, 1.0);
  phases[Nd - 1] = -1.0;
  WilsonFermionD::ImplParams implParams;
  implParams.boundary_phases = phases;
  WilsonAnisotropyCoefficients anisotropy;

  std::unique_ptr<WilsonFermionD> fermop;
  if (action_name == "wilson")
    fermop.reset(new WilsonFermionD(Umu, *UGrid, *UrbGrid, mass, implParams));
  else
    fermop.reset(new CompactWilsonCloverFermionD(Umu, *UGrid, *UrbGrid, mass, csw, csw, 1.0, anisotropy, implParams));
  std::unique_ptr<SchurOperatorBase<LatticeFermionD>> schurD;
  if (schur_name == "one")
    schurD.reset(new SchurDiagOneOperator<WilsonFermionD, LatticeFermionD>(*fermop));
  else
    schurD.reset(new SchurDiagMooeeOperator<WilsonFermionD, LatticeFermionD>(*fermop));

  LatticeGaugeFieldF UmuF(UGridF);
  precisionChange(UmuF, Umu);
  std::unique_ptr<WilsonFermionF> fermopF;
  if (action_name == "wilson")
    fermopF.reset(new WilsonFermionF(UmuF, *UGridF, *UrbGridF, mass, implParams));
  else
    fermopF.reset(
        new CompactWilsonCloverFermionF(UmuF, *UGridF, *UrbGridF, mass, csw, csw, 1.0, anisotropy, implParams));
  std::unique_ptr<SchurOperatorBase<LatticeFermionF>> schurF;
  if (schur_name == "one")
    schurF.reset(new SchurDiagOneOperator<WilsonFermionF, LatticeFermionF>(*fermopF));
  else
    schurF.reset(new SchurDiagMooeeOperator<WilsonFermionF, LatticeFermionF>(*fermopF));
  NonHermitianLinearOperator<WilsonFermionF, LatticeFermionF> FullOpF(*fermopF);
  // The operator the hierarchy is built from: D, or A~ = Mdiag^-1 D (FULLOP_OPERATOR=precond).
  std::unique_ptr<PrecondFullOperator> PrecondOpF;
  if (precond) PrecondOpF.reset(new PrecondFullOperator(*fermopF, UGridF));
  LinearOperatorBase<LatticeFermionF> &CoarsenOpF =
      precond ? static_cast<LinearOperatorBase<LatticeFermionF> &>(*PrecondOpF)
              : static_cast<LinearOperatorBase<LatticeFermionF> &>(FullOpF);
  // One fp32 RNG on the FULL grid, seeds4, drawn from only by the null-vector generation below. The
  // schur scheme fills red-black fields from it exactly as probe_grid_mg_schur_clover.cc does
  // (GridParallelRNG::fill of a checkerboarded field draws on the RNG's own grid and picks the
  // parity, so an RNG constructed ON the red-black grid would recurse; it must be a full-grid one).
  GridParallelRNG RNG4F(UGridF);
  RNG4F.SeedFixedIntegers(seeds4);

  // ---- level 1: null vectors, gamma5 doubling, Galerkin coarsening of CoarsenOpF ----
  typedef Aggregation<vSpinColourVectorF, vTComplexF, kNbasis> SubspaceF;
  typedef Aggregation<vSpinColourVectorF, vTComplexF, 2 * kNbasis> CombinedSubspaceF;
  typedef GeneralCoarsenedMatrix<vSpinColourVectorF, vTComplexF, 2 * kNbasis> LittleDiracOperatorF;
  typedef LittleDiracOperatorF::CoarseVector CoarseVectorF;

  accelerator_barrier();
  UGrid->Barrier();
  const double t_setup = usecond();
  CombinedSubspaceF CombinedUVF(Coarse4dF, UGridF, 0);
  double subspace_seconds = 0.0;
  double nullvec_odd_residual = 0.0; // schur scheme: |[D x]_odd| / |x| of vector 0, zero up to fp32 roundoff
  {
    SubspaceF AggF(Coarse4dF, UGridF, 0);
    if (nullvec == "schur") {
      // Even halves: the Schur-operator probe's generation (fp32 Schur operator, red-black grid).
      SubspaceF AggRbF(Coarse4dF, UrbGridF, Even);
      ProbeMG::create_subspace_gcr(RNG4F, *schurF, AggRbF, kNbasis, subspace_tol, subspace_rounds, subspace_mmax,
                                   subspace_mmax, subspace_maxiter, true, true);
      // Odd halves from the odd-site equation (D x)_o = 0: x_o = -M_oo^-1 M_oe x_e.
      LatticeFermionF tmp_o(UrbGridF), x_o(UrbGridF);
      for (int b = 0; b < kNbasis; ++b) {
        AggRbF.subspace[b].Checkerboard() = Even;
        tmp_o.Checkerboard() = Odd;
        fermopF->Meooe(AggRbF.subspace[b], tmp_o);
        fermopF->MooeeInv(tmp_o, x_o);
        x_o = x_o * RealD(-1.0);
        setCheckerboard(AggF.subspace[b], AggRbF.subspace[b]);
        setCheckerboard(AggF.subspace[b], x_o);
      }
    } else {
      ProbeMG::create_subspace_gcr(RNG4F, CoarsenOpF, AggF, kNbasis, subspace_tol, subspace_rounds, subspace_mmax,
                                   subspace_mmax, subspace_maxiter, true, true);
    }
    accelerator_barrier();
    UGrid->Barrier();
    subspace_seconds = seconds_since(t_setup);
    if (nullvec == "schur") { // global reductions: every rank
      LatticeFermionF f_chk(UGridF), o_chk(UrbGridF);
      FullOpF.Op(AggF.subspace[0], f_chk);
      o_chk.Checkerboard() = Odd;
      pickCheckerboard(Odd, o_chk, f_chk);
      nullvec_odd_residual = std::sqrt(norm2(o_chk) / norm2(AggF.subspace[0]));
      if (boss)
        std::cout << GridLogMessage << "null vectors      odd-site residual |[D x]_o|/|x| " << nullvec_odd_residual
                  << (nullvec_odd_residual < 1.0e-4 ? "  PASSED" : "  FAILED") << std::endl;
    }
    Gamma G5(Gamma::Algebra::Gamma5);
    for (int b = 0; b < kNbasis; ++b) {
      CombinedUVF.subspace[b] = AggF.subspace[b];
      CombinedUVF.subspace[b + kNbasis] = G5 * AggF.subspace[b];
    }
  }
  if (boss) std::cout << GridLogMessage << "subspace generation     " << subspace_seconds << " s" << std::endl;

  NonLocalStencilGeometry4D geomF(Coarse4dF, 1); // must outlive the operator (held by reference)
  LittleDiracOperatorF LittleDiracOpF(geomF, UGridF, Coarse4dF);
  const double t_c1 = usecond();
  LittleDiracOpF.CoarsenOperator(CoarsenOpF, CombinedUVF);
  accelerator_barrier();
  UGrid->Barrier();
  const double coarsen_seconds = seconds_since(t_c1);

  // Galerkin gate: Dc c == V^dag D V c (D = CoarsenOpF). Exact to fp32 roundoff for a one-hop operator.
  GridParallelRNG CRNG(Coarse4dF);
  CRNG.SeedFixedIntegers(std::vector<int>({5, 6, 7, 8}));
  double galerkin1 = -1.0;
  double mooeeinv_check = 0.0; // precond: full-grid MooeeInv vs the per-checkerboard one
  {
    CoarseVectorF c_src(Coarse4dF), c_res(Coarse4dF), c_proj(Coarse4dF);
    LatticeFermionF f_prom(UGridF), f_mat(UGridF);
    random(CRNG, c_src);
    CombinedUVF.PromoteFromSubspace(c_src, f_prom);
    CoarsenOpF.Op(f_prom, f_mat);
    CombinedUVF.ProjectToSubspace(c_proj, f_mat);
    LittleDiracOpF.M(c_src, c_res);
    c_proj = c_proj - c_res;
    galerkin1 = std::sqrt(norm2(c_proj) / norm2(c_res));
    if (precond) { // global reductions: every rank
      LatticeFermionF f_inv(UGridF);
      LatticeFermionF h_in(UrbGridF), h_ref(UrbGridF), h_got(UrbGridF);
      fermopF->MooeeInv(f_prom, f_inv);
      double num = 0.0;
      for (int p = 0; p < 2; ++p) {
        const int parity = p == 0 ? Even : Odd;
        h_in.Checkerboard() = parity;
        h_got.Checkerboard() = parity;
        pickCheckerboard(parity, h_in, f_prom);
        pickCheckerboard(parity, h_got, f_inv);
        fermopF->MooeeInv(h_in, h_ref);
        h_got = h_got - h_ref;
        num += norm2(h_got);
      }
      mooeeinv_check = std::sqrt(num / norm2(f_inv));
    }
  }
  if (boss) {
    std::cout << GridLogMessage << "level-1 coarsening      " << coarsen_seconds << " s, Galerkin " << galerkin1
              << (galerkin1 < 1.0e-4 ? "  PASSED" : "  FAILED") << std::endl;
    if (precond)
      std::cout << GridLogMessage << "MooeeInv full-grid check " << mooeeinv_check
                << (mooeeinv_check < 1.0e-6 ? "  PASSED" : "  FAILED") << std::endl;
  }

  ProbeMG::OpCounts counts;
  ProbeMG::StencilCoarseApply<vSpinColourVectorF, vTComplexF, 2 * kNbasis> StencilCoarseF(LittleDiracOpF, Coarse4dF,
                                                                                         0.001);
  ProbeMG::CountingLinearOperator<CoarseVectorF> CountedCoarseF(StencilCoarseF, counts.coarse);
  if (boss) std::cout << GridLogMessage << "coarse apply            stencil" << std::endl;

  // ---- level 2: the level-1 operator aggregated again (unchanged from the Schur probe) ----
  typedef LittleDiracOperatorF::siteVector L1SiteF;
  typedef iScalar<vTComplexF> CComplex2F;
  typedef Aggregation<L1SiteF, CComplex2F, kNbasis2> Subspace2F;
  typedef GeneralCoarsenedMatrix<L1SiteF, CComplex2F, kNbasis2> L2OperatorF;
  typedef L2OperatorF::CoarseVector CoarseVector2F;
  Coordinate clatt2 = Coarse4dF->GlobalDimensions();
  for (int d = 0; d < 4; ++d) {
    if (clatt2[d] % Block2[d] != 0) return fail("level-1 lattice not divisible by --probe-block2 in dim " + std::to_string(d));
    clatt2[d] /= Block2[d];
  }
  GridCartesian *Coarse2F = SpaceTimeGrid::makeFourDimGrid(clatt2, GridDefaultSimd(Nd, vComplexF::Nsimd()),
                                                           GridDefaultMpi());
  if (boss)
    std::cout << GridLogMessage << "level-2      gdim " << coordinate_string(Coarse2F->GlobalDimensions()) << " rdim "
              << coordinate_string(Coarse2F->_rdimensions) << " (level-1 rdim "
              << coordinate_string(Coarse4dF->_rdimensions) << ")" << std::endl;
  subdivides(Coarse2F, Coarse4dF);

  const double t_l2 = usecond();
  GridParallelRNG CRNG1F(Coarse4dF);
  CRNG1F.SeedFixedIntegers(std::vector<int>({9, 10, 11, 12}));
  Subspace2F Agg2F(Coarse2F, Coarse4dF, Even);
  ProbeMG::create_subspace_gcr(CRNG1F, CountedCoarseF, Agg2F, kNbasis2, l2_subspace_tol, l2_subspace_rounds, 10, 10,
                               l2_subspace_maxiter, true, true);
  accelerator_barrier();
  UGrid->Barrier();
  const double l2_subspace_seconds = seconds_since(t_l2);
  NonLocalStencilGeometry4D geom2F(Coarse2F, 1);
  L2OperatorF L2OpF(geom2F, Coarse4dF, Coarse2F);
  const double t_l2c = usecond();
  L2OpF.CoarsenOperator(CountedCoarseF, Agg2F);
  accelerator_barrier();
  UGrid->Barrier();
  const double l2_coarsen_seconds = seconds_since(t_l2c);
  double galerkin2 = -1.0;
  {
    CoarseVectorF f_rand(Coarse4dF), f_prom(Coarse4dF), f_mat(Coarse4dF);
    random(CRNG1F, f_rand);
    CoarseVector2F c_src(Coarse2F), c_res(Coarse2F), c_proj(Coarse2F);
    Agg2F.ProjectToSubspace(c_src, f_rand);
    Agg2F.PromoteFromSubspace(c_src, f_prom);
    CountedCoarseF.Op(f_prom, f_mat);
    Agg2F.ProjectToSubspace(c_proj, f_mat);
    L2OpF.M(c_src, c_res);
    c_proj = c_proj - c_res;
    galerkin2 = std::sqrt(norm2(c_proj) / norm2(c_res));
  }
  if (boss)
    std::cout << GridLogMessage << "level-2 setup           " << l2_subspace_seconds + l2_coarsen_seconds
              << " s (subspace " << l2_subspace_seconds << ", coarsen " << l2_coarsen_seconds << "), Galerkin "
              << galerkin2 << std::endl;

  std::unique_ptr<ProbeMG::StencilCoarseApply<L1SiteF, CComplex2F, kNbasis2>> Stencil2F;
  ShiftedNonHermitianLinearOperator<L2OperatorF, CoarseVector2F> Shifted2F(L2OpF, 0.001);
  if (geom2F.npoint == 9) {
    Stencil2F.reset(new ProbeMG::StencilCoarseApply<L1SiteF, CComplex2F, kNbasis2>(L2OpF, Coarse2F, 0.001));
  } else if (boss) {
    std::cout << GridLogMessage << "level-2 stencil apply UNAVAILABLE (npoint " << geom2F.npoint
              << "); using the general apply" << std::endl;
  }
  LinearOperatorBase<CoarseVector2F> &Selected2F =
      Stencil2F ? static_cast<LinearOperatorBase<CoarseVector2F> &>(*Stencil2F)
                : static_cast<LinearOperatorBase<CoarseVector2F> &>(Shifted2F);
  ProbeMG::CountingLinearOperator<CoarseVector2F> Counted2F(Selected2F, counts.coarse2);
  TrivialPrecon<CoarseVector2F> trivial2;
  ProbeMG::FlexibleGCR<CoarseVector2F> Coarse2Solver(l2_coarse_tol, l2_coarse_maxiter, Counted2F, trivial2, l2_coarse_mmax,
                                                     l2_coarse_nstep);
  Coarse2Solver.Level(5);
  Coarse2Solver.verbose = 0;
  Coarse2Solver.zero_guess = true;
  Coarse2Solver.verify_residual = false;
  TrivialPrecon<CoarseVectorF> trivial1;
  ProbeMG::FlexibleGCR<CoarseVectorF> L1Smoother(l2_smoother_tol, 1, CountedCoarseF, trivial1, 1, l2_smoother_nstep);
  L1Smoother.Level(4);
  L1Smoother.verbose = 0;
  L1Smoother.zero_guess = true;
  L1Smoother.verify_residual = false;
  MGPreconditioner<L1SiteF, CComplex2F, kNbasis2> Precon2F(Agg2F, CountedCoarseF, L1Smoother, Coarse2Solver);

  ProbeMG::FlexibleGCR<CoarseVectorF> CoarseSolver(coarse_tol, coarse_maxiter, CountedCoarseF, Precon2F, coarse_mmax,
                                                   coarse_nstep);
  CoarseSolver.Level(3);
  CoarseSolver.verbose = 0;
  CoarseSolver.zero_guess = true;
  CoarseSolver.verify_residual = false;

  // ---- fine level: smoother on the Schur operator, V-cycle over the full hierarchy ----
  ProbeMG::CountingLinearOperator<LatticeFermionF> CountedSchurF(*schurF, counts.fine);
  ShiftedSchurOperator<LatticeFermionF> ShiftedSchurF(*schurF, 0.01);
  ProbeMG::CountingLinearOperator<LatticeFermionF> CountedShiftedSchurF(ShiftedSchurF, counts.fine);
  TrivialPrecon<LatticeFermionF> trivial0;
  ProbeMG::FlexibleGCR<LatticeFermionF> Smoother(smoother_tol, smoother_maxiter, CountedShiftedSchurF, trivial0,
                                                 smoother_mmax, smoother_nstep);
  Smoother.Level(2);
  Smoother.verbose = 0;
  Smoother.zero_guess = true;
  Smoother.verify_residual = false;
  // Transfer mode (see SchurFromFullMG). Env FULLOP_TRANSFER=even|full; the default is `even`
  // whenever the block allows it. Env rather than a flag, so that run_probe_grid_mg.sh (whose srun
  // exports the environment) did not have to change.
  const char *transfer_env = std::getenv("FULLOP_TRANSFER");
  const std::string transfer = transfer_env ? transfer_env : (Block[0] % 2 == 0 ? "even" : "full");
  if (transfer != "even" && transfer != "full") return fail("FULLOP_TRANSFER must be even|full");
  if (transfer == "even" && Block[0] % 2 != 0) return fail("FULLOP_TRANSFER=even needs an even x block");
  std::unique_ptr<CombinedSubspaceF> EvenUVF;
  if (transfer == "even") {
    subdivides(Coarse4dF, UrbGridF);
    EvenUVF.reset(new CombinedSubspaceF(Coarse4dF, UrbGridF, Even));
    // Taken AFTER CoarsenOperator, which block-orthonormalised CombinedUVF in place: the even
    // halves must be those of the same vectors the coarse operator was built from.
    for (int b = 0; b < 2 * kNbasis; ++b) {
      EvenUVF->subspace[b].Checkerboard() = Even;
      pickCheckerboard(Even, EvenUVF->subspace[b], CombinedUVF.subspace[b]);
    }
  }
  if (boss) std::cout << GridLogMessage << "transfer          " << transfer
                      << (transfer == "even" ? " (even-site halves, QUDA parity mode)" : " (full lattice)") << std::endl;
  // rhs map (see SchurFromFullMG): D hierarchy: M_ee for One, none for Mooee.
  // A~ hierarchy: none for One ((A~^-1)_ee = One^-1), M_ee^-1 for Mooee.
  WilsonFermionF *rhs_mee = (!precond && schur_name == "one") ? fermopF.get() : nullptr;
  WilsonFermionF *rhs_meeinv = (precond && schur_name == "mooee") ? fermopF.get() : nullptr;
  SchurFromFullMG<vTComplexF, 2 * kNbasis> VCycle(CombinedUVF, EvenUVF.get(), CountedSchurF, rhs_mee, rhs_meeinv,
                                                  CoarseSolver, Smoother);
  PrecisionChangeAdaptor<LatticeFermionD, LatticeFermionF> PreconMixed(VCycle, UrbGridF);
  const double setup_seconds = seconds_since(t_setup);
  if (boss) std::cout << GridLogMessage << "setup total             " << setup_seconds << " s" << std::endl;

  // ---- outer solve (fp64 flexible GCR on the Schur operator) ----
  ProbeMG::CountingLinearOperator<LatticeFermionD> CountedSchurD(*schurD, counts.fine);
  ProbeMG::FlexibleGCR<LatticeFermionD> Outer(tol, maxiter, CountedSchurD, PreconMixed, outer_mmax, outer_nstep);
  Outer.Level(1);
  Outer.verify_residual = true;
  Outer.zero_guess = false;

  // Same source as the Schur probe and the QUDA probe.
  RNG4.SeedFixedIntegers(std::vector<int>({11, 22, 33, 44}));
  LatticeFermionD full_src(UGrid);
  random(RNG4, full_src);
  LatticeFermionD src(UrbGrid);
  src.Checkerboard() = cb;
  pickCheckerboard(cb, src, full_src);
  const double source_norm2 = norm2(src);
  LatticeFermionD sol(UrbGrid);
  sol.Checkerboard() = cb;

  std::vector<double> times;
  long long fine = 0, coarse = 0, coarse2 = 0;
  int steps = 0;
  for (int r = 0; r < solve_repeats; ++r) {
    sol = Zero();
    counts.reset();
    accelerator_barrier();
    UGrid->Barrier();
    const double t0 = usecond();
    Outer(src, sol);
    accelerator_barrier();
    UGrid->Barrier();
    times.push_back(seconds_since(t0));
    fine = counts.fine;
    coarse = counts.coarse;
    coarse2 = counts.coarse2;
    steps = Outer.steps;
  }
  LatticeFermionD residual(UrbGrid);
  residual.Checkerboard() = cb;
  schurD->Op(sol, residual);
  residual = residual - src;
  const double mg_residual = std::sqrt(norm2(residual) / source_norm2);
  const bool pass = std::isfinite(mg_residual) && mg_residual <= 1.1 * tol && galerkin1 < 1.0e-4 && steps < maxiter &&
                    (!precond || mooeeinv_check < 1.0e-6) && (nullvec != "schur" || nullvec_odd_residual < 1.0e-4);

  if (boss) {
    const double v = steps > 0 ? static_cast<double>(steps) : 1.0;
    std::cout << GridLogMessage << "--- summary (full-operator hierarchy) ---" << std::endl;
    std::cout << GridLogMessage << "mode                    operator " << (precond ? "Mdiag^-1 M" : "D")
              << ", null vectors " << nullvec << ", transfer " << transfer << ", schur " << schur_name << std::endl;
    std::cout << GridLogMessage << "setup total             " << setup_seconds << " s" << std::endl;
    std::cout << GridLogMessage << "MG solve (Mpc)          " << median_of(times) << " s, " << steps
              << " outer PGCR steps, independent residual " << mg_residual << std::endl;
    std::cout << GridLogMessage << "operator applications   fine " << fine << " (" << fine / v << " / V-cycle), coarse "
              << coarse << " (" << coarse / v << " / V-cycle), coarse2 " << coarse2 << " (" << coarse2 / v
              << " / V-cycle)" << std::endl;
    std::cout << GridLogMessage << "PROBE RESULT: " << (pass ? "PASS" : "FAIL") << std::endl;
  }
  Grid_finalize();
  return pass ? 0 : 1;
}
