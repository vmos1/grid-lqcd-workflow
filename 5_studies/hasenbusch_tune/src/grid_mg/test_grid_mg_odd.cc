// test_grid_mg_odd.cc  (pure-Grid HMC, M2 acceptance test 1)
//
// Standalone check of the Grid multigrid rung solver (src/grid_mg/) on the checkerboard the HMC
// uses, before any HMC run. Spec: __docs/2026_09_29_pure_grid_m2_mg_solver_design.md §2 item 8
// and §4 item 1. Validates patch 04 (parity-agnostic CoarsenOperator) and the port.
//
// For each requested checkerboard it
//   1. builds a GridMGHierarchy on the fp32 compact-clover operator (stout-smeared gauge field,
//      exactly as the driver builds and smears it) and times the setup;
//   2. checks the Galerkin property of level 1 on a random coarse vector,
//        |P^dag Mpc P v - A_1 v| / |A_1 v|   (expect ~5e-2 at hops 1: the 9-point stencil cannot
//        hold the 2-hop Mpc exactly, the probe's documented inexactness, 4.7e-2 at 16^3),
//      with the probe's must-differ control against Mpc^dag Mpc (expect O(1)), and the level-1
//      stencil apply against the general apply it was copied from (expect ~1e-6, fp32);
//   3. checks the Galerkin property of level 2 (expect ~1e-6: the level-1 operator is 9-point,
//      so a 9-point level-2 stencil is exact up to fp32 roundoff);
//   4. runs one GridMGSchurSolver normal-equation solve on a Gaussian source of that parity at
//      --tol and grades |Mpc^dag Mpc sol - src| / |src| in fp64.
//
// Thresholds (PASS needs all, on every checkerboard run):
//   GALERKIN L1 rel < 1e-1 and control > 1e-3; STENCIL L1 rel < 1e-4; GALERKIN L2 rel < 1e-4;
//   SOLVE true_rel_residual <= 1e-10 with sol.Checkerboard() == cb and no assert.
//
// Usage (inside an allocation, 4 GPUs, one rank per GPU):
//   test_grid_mg_odd --grid 16.16.16.48 --mpi 1.1.2.2 [--cfg <lime>] [--mass -0.245]
//                    [--csw 1.24930970916466] [--tol 1e-11] [--cb even|odd|both]
//                    [--block2 auto|a.b.c.d] [--stout-rho 0.125] [--stout-nsmear 1]
// GRID_MG_* variables tune the hierarchy (grid_mg_params.h). --block2 overrides GRID_MG_BLOCK2;
// `auto` (default) keeps the configured block2 if the geometry admits it and otherwise lowers
// each dimension to the largest usable value, printing the choice: best2's 2.3.3.3 cannot divide
// the 16^3 level-1 lattice 4.4.4.12 on any decomposition. ⚠️ --mpi 1.1.1.4 (the M1 smoke's) is
// unusable for any MG geometry at 16^3: the level-1 local T extent 3 does not divide the fp32
// SIMD layout 1.2.2.2. Use 1.1.2.2 (block2 auto -> 2.2.1.3) or 1.2.2.1 on 4 GPUs.

#include <Grid/Grid.h>

#include "grid_mg/grid_mg_hierarchy.h"
#include "grid_mg/grid_mg_params.h"
#include "grid_mg/grid_mg_schur_solver.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace Grid;

namespace {

typedef CompactWilsonCloverFermion<WilsonImplR, CompactCloverHelpers<WilsonImplR>> WCF;
typedef GridMGHierarchy::WCF_f WCF_f;

std::string arg_string(int argc, char **argv, const std::string &opt, const std::string &fallback)
{
  if (!GridCmdOptionExists(argv, argv + argc, opt)) return fallback;
  return GridCmdOptionPayload(argv, argv + argc, opt);
}
double arg_double(int argc, char **argv, const std::string &opt, double fallback)
{
  const std::string s = arg_string(argc, argv, opt, "");
  return s.empty() ? fallback : std::stod(s);
}
int arg_int(int argc, char **argv, const std::string &opt, int fallback)
{
  const std::string s = arg_string(argc, argv, opt, "");
  return s.empty() ? fallback : std::stoi(s);
}

bool parse_block(const std::string &s, Coordinate &out)
{
  std::vector<int> v;
  std::size_t pos = 0;
  while (pos <= s.size()) {
    const std::size_t dot = s.find('.', pos);
    const std::string item = s.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos);
    if (item.empty()) return false;
    for (char c : item)
      if (c < '0' || c > '9') return false;
    v.push_back(std::stoi(item));
    if (dot == std::string::npos) break;
    pos = dot + 1;
  }
  if (v.size() != 4) return false;
  for (int x : v)
    if (x < 1) return false;
  out = Coordinate(v);
  return true;
}

// Largest block2 <= `want` per dimension that the geometry admits (see file header).
Coordinate auto_block2(GridBase *GridF, const Coordinate &block, const Coordinate &want)
{
  if (GridMGHierarchy::CheckGeometry(GridF, block, want).empty()) return want;
  const Coordinate fine = GridF->GlobalDimensions();
  const Coordinate procs = GridF->_processors;
  const Coordinate simd = GridF->_simd_layout;
  Coordinate b2 = want;
  for (int d = 0; d < 4; ++d) {
    const int c1 = fine[d] / block[d];
    int pick = 1;
    for (int b = want[d]; b >= 1; --b) {
      if (c1 % b != 0) continue;
      const int c2 = c1 / b;
      if (c2 % procs[d] != 0) continue;
      if ((c2 / procs[d]) % simd[d] != 0) continue;
      pick = b;
      break;
    }
    b2[d] = pick;
  }
  return b2;
}

}  // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  bool all_pass = true;
  {
    const std::string cfg = arg_string(
        argc, argv, "--cfg",
        "/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/data/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime");
    const double mass = arg_double(argc, argv, "--mass", -0.245);
    const double csw = arg_double(argc, argv, "--csw", 1.24930970916466);
    const double tol = arg_double(argc, argv, "--tol", 1.0e-11);
    const std::string cb_arg = arg_string(argc, argv, "--cb", "both");
    const std::string block2_arg = arg_string(argc, argv, "--block2", "auto");
    const double stout_rho = arg_double(argc, argv, "--stout-rho", 0.125);
    const int stout_nsmear = arg_int(argc, argv, "--stout-nsmear", 1);

    std::vector<int> cbs;
    if (cb_arg == "even") cbs = {Even};
    else if (cb_arg == "odd") cbs = {Odd};
    else if (cb_arg == "both") cbs = {Even, Odd};
    else {
      std::cerr << "test_grid_mg_odd: --cb must be even|odd|both, got " << cb_arg << std::endl;
      Grid_finalize();
      return 2;
    }

    // ---- grids: as the driver builds them ----
    const Coordinate latt = GridDefaultLatt();
    const Coordinate mpi = GridDefaultMpi();
    GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), mpi);
    GridRedBlackCartesian RBGrid_(&Grid_);
    GridCartesian GridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
    GridRedBlackCartesian RBGridF(&GridF);
    const bool boss = Grid_.IsBoss();

    // ---- parameters ----
    GridMGParams params = GridMGParams::from_env();
    if (block2_arg == "auto") {
      const Coordinate chosen = auto_block2(&GridF, params.block, params.block2);
      if (boss && !(chosen == params.block2))
        std::cout << "BLOCK2 auto: " << GridMGParams::CoordinateString(params.block2)
                  << " does not fit this geometry; using " << GridMGParams::CoordinateString(chosen)
                  << std::endl;
      params.block2 = chosen;
    } else if (!parse_block(block2_arg, params.block2)) {
      std::cerr << "test_grid_mg_odd: --block2 must be auto or a.b.c.d, got " << block2_arg
                << std::endl;
      Grid_finalize();
      return 2;
    }
    const std::string geom_err = GridMGHierarchy::CheckGeometry(&GridF, params.block, params.block2);
    if (!geom_err.empty()) {
      if (boss)
        std::cout << "GEOMETRY unusable: " << geom_err << " (lattice "
                  << GridMGParams::CoordinateString(latt) << ", mpi "
                  << GridMGParams::CoordinateString(mpi) << ")" << std::endl
                  << "TEST RESULT: FAIL" << std::endl;
      Grid_finalize();
      return 1;
    }
    if (boss) {
      std::cout << "PARAMS " << params.Summary() << std::endl;
      std::cout << "RUN lattice " << GridMGParams::CoordinateString(latt) << " mpi "
                << GridMGParams::CoordinateString(mpi) << " cfg " << cfg << " mass " << mass
                << " csw " << csw << " tol " << tol << " cb " << cb_arg << " stout rho "
                << stout_rho << " nsmear " << stout_nsmear << std::endl;
    }

    // ---- gauge field: read, as the driver's IMPORT_CFG branch ----
    LatticeGaugeField Umu(&Grid_);
    {
      FILE *fp = std::fopen(cfg.c_str(), "rb");
      if (!fp) {
        if (boss) std::cout << "cannot open --cfg " << cfg << std::endl << "TEST RESULT: FAIL" << std::endl;
        Grid_finalize();
        return 1;
      }
      char magic[16] = {0};
      const size_t nread = std::fread(magic, 1, sizeof(magic), fp);
      std::fclose(fp);
      FieldMetaData header;
      if (nread >= 12 && std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
        NerscIO::readConfiguration<GaugeStatistics<PeriodicGimplR>>(Umu, header, cfg);
      } else {
#ifdef HAVE_LIME
        IldgReader IR;
        IR.open(cfg);
        IR.readConfiguration(Umu, header);
        IR.close();
#else
        if (boss) std::cout << "this Grid has no LIME; cannot read " << cfg << std::endl;
        GRID_ASSERT(0);
#endif
      }
    }
    const double plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);

    // ---- stout smearing: as the driver (SmearedConfiguration over Smear_Stout, rho 0.125) ----
    Smear_Stout<PeriodicGimplR> Stout(stout_rho);
    SmearedConfiguration<PeriodicGimplR> Smear(&Grid_, stout_nsmear, Stout);
    Smear.set_Field(Umu);
    LatticeGaugeField &Usm = (stout_nsmear > 0) ? Smear.get_SmearedU() : Umu;
    const double plaq_sm = WilsonLoops<PeriodicGimplR>::avgPlaquette(Usm);
    if (boss)
      std::cout << "GAUGE plaquette " << plaq << " smeared " << plaq_sm << std::endl;

    // ---- operators: as the driver's LightOps / LightOpsF (antiperiodic T, cF = 1) ----
    WilsonImplParams impl_p, impl_pF;
    impl_p.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
    impl_pF.boundary_phases = std::vector<Complex>({1., 1., 1., -1.});
    WilsonAnisotropyCoefficients anis;
    WCF op(Usm, Grid_, RBGrid_, mass, csw, csw, /*cF=*/1.0, anis, impl_p);
    LatticeGaugeFieldF UsmF(&GridF);
    precisionChange(UsmF, Usm);
    WCF_f opF(UsmF, GridF, RBGridF, mass, csw, csw, /*cF=*/1.0, anis, impl_pF);
    // The operator the ratio action hands its solver.
    SchurDifferentiableOperator<WilsonImplR> Mpc(op);

    for (const int cb : cbs) {
      const std::string cbn = (cb == Odd) ? "odd" : "even";
      bool pass = true;

      GridMGHierarchy H(opF, &GridF, &RBGridF, cb, params, "test_" + cbn);
      H.SetGauge(Usm);
      H.Build();
      const double setup_s = H.LastSetupSeconds();

      // ---- level-1 Galerkin (fp32), control, and stencil agreement ----
      typedef GridMGHierarchy::CoarseVector1 CV1;
      typedef GridMGHierarchy::CoarseVector2 CV2;
      GridParallelRNG CRNG1(H.Coarse1Grid());
      CRNG1.SeedFixedIntegers(std::vector<int>({5, 6, 7, 8}));
      auto galerkin1 = [&](bool herm) {
        CV1 c_src(H.Coarse1Grid()), c_res(H.Coarse1Grid()), c_proj(H.Coarse1Grid());
        LatticeFermionF f_prom(&RBGridF), f_mat(&RBGridF);
        f_prom.Checkerboard() = cb;
        f_mat.Checkerboard() = cb;
        random(CRNG1, c_src);
        H.Level1Subspace().PromoteFromSubspace(c_src, f_prom);
        GRID_ASSERT(f_prom.Checkerboard() == cb);
        if (herm)
          H.FineSchur().HermOp(f_prom, f_mat);
        else
          H.FineSchur().Op(f_prom, f_mat);
        H.Level1Subspace().ProjectToSubspace(c_proj, f_mat);
        H.Level1Operator().M(c_src, c_res);
        c_proj = c_proj - c_res;
        return std::sqrt(norm2(c_proj) / norm2(c_res));
      };
      const double g1 = galerkin1(false);
      const double g1_ctrl = galerkin1(true);
      double st1 = 0.0;
      {
        CV1 c_src(H.Coarse1Grid()), c_gen(H.Coarse1Grid()), c_st(H.Coarse1Grid());
        random(CRNG1, c_src);
        H.Level1Operator().M(c_src, c_gen);
        H.Level1Apply().Op(c_src, c_st);  // A_1 v + coarse_shift v
        c_st = c_st - params.coarse_shift * c_src;
        c_st = c_st - c_gen;
        st1 = std::sqrt(norm2(c_st) / norm2(c_gen));
      }
      // ---- level-2 Galerkin (probe lines 1570-1585) ----
      double g2 = 0.0;
      {
        CV1 f_rand(H.Coarse1Grid()), f_prom(H.Coarse1Grid()), f_mat(H.Coarse1Grid());
        CV2 c_src(H.Coarse2Grid()), c_res(H.Coarse2Grid()), c_proj(H.Coarse2Grid());
        random(CRNG1, f_rand);
        H.Level2Subspace().ProjectToSubspace(c_src, f_rand);
        H.Level2Subspace().PromoteFromSubspace(c_src, f_prom);
        H.Level1Apply().Op(f_prom, f_mat);
        H.Level2Subspace().ProjectToSubspace(c_proj, f_mat);
        H.Level2Operator().M(c_src, c_res);
        c_proj = c_proj - c_res;
        g2 = std::sqrt(norm2(c_proj) / norm2(c_res));
      }
      const bool g1_ok = (g1 < 1.0e-1) && (g1_ctrl > 1.0e-3);
      const bool st1_ok = (st1 < 1.0e-4);
      const bool g2_ok = (g2 < 1.0e-4);
      pass = pass && g1_ok && st1_ok && g2_ok;
      if (boss) {
        std::cout << "GALERKIN L1 cb=" << cbn << " rel=" << g1 << " control_MdagM=" << g1_ctrl
                  << (g1_ok ? " ok" : " BAD") << std::endl;
        std::cout << "STENCIL L1 cb=" << cbn << " rel=" << st1 << (st1_ok ? " ok" : " BAD")
                  << std::endl;
        std::cout << "GALERKIN L2 cb=" << cbn << " rel=" << g2 << (g2_ok ? " ok" : " BAD")
                  << std::endl;
      }

      // ---- one normal-equation solve through the rung solver ----
      GridMGSchurSolver S(H, tol, cb, "test_" + cbn, /*donor=*/true);
      S.SetGauge(Usm);  // what the action wrapper does before each solve
      GridParallelRNG RNG4(&Grid_);
      RNG4.SeedFixedIntegers(std::vector<int>({11, 22, 33, 44}));
      LatticeFermionD full(&Grid_);
      gaussian(RNG4, full);
      LatticeFermionD src(&RBGrid_), sol(&RBGrid_), res(&RBGrid_);
      pickCheckerboard(cb, src, full);
      sol = Zero();
      sol.Checkerboard() = cb;
      const double ts = usecond();
      S(Mpc, src, sol);
      const double solve_s = (usecond() - ts) / 1.0e6;
      const bool cb_ok = (sol.Checkerboard() == cb);
      Mpc.HermOp(sol, res);
      res = res - src;
      const double rel = std::sqrt(norm2(res) / norm2(src));
      const bool solve_ok = cb_ok && (rel <= 1.0e-10);
      pass = pass && solve_ok;
      if (boss) {
        std::cout << "SOLVE cb=" << cbn << " iters1=" << S.Last(1).iterations
                  << " iters2=" << S.Last(2).iterations << " true_rel_residual=" << rel
                  << " setup_s=" << setup_s << " solve_s=" << solve_s
                  << " mpc_residuals=" << S.Last(1).true_residual << "," << S.Last(2).true_residual
                  << " sol_cb=" << sol.Checkerboard() << (solve_ok ? " ok" : " BAD") << std::endl;
        std::cout << "CHECKERBOARD " << cbn << (pass ? " PASS" : " FAIL") << std::endl;
      }
      all_pass = all_pass && pass;
    }
    if (boss) std::cout << "TEST RESULT: " << (all_pass ? "PASS" : "FAIL") << std::endl;
  }
  Grid_finalize();
  return all_pass ? 0 : 1;
}
