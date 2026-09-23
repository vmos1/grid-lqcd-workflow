// probe_quda_mg_clover.cc
//
// Phase 0 probe for the QUDA side of the MG arm.
// Design doc: __docs/2026_09_10_mg_comparison_project_goals.md
// Counterpart: probe_grid_mg_schur_clover.cc (the Grid side).
//
// PURE QUDA IN EVERY TIMED REGION. Grid appears only outside them, for exactly two
// jobs, both agreed as out of scope for the comparison:
//
//   1. reading the ILDG configuration and laying the gauge field out, and
//   2. GRADING the returned solutions.
//
// (2) is not a compromise, it is the correct choice: QUDA's own reported
// `true_res` carries a ~1e-9 run-to-run autotuning floor, so every gate in this
// campaign is graded on independently evaluated GRID-side fields. Letting QUDA
// grade itself here would be strictly worse than using Grid.
//
// Note production is NOT this shape: there, Grid drives and calls QUDA modules for
// the solves. This probe and its Grid counterpart are each ONE backend solving on
// its own, which is what makes the pair comparable.
//
// SELF-CONTAINED. Includes only <Grid/Grid.h>, <quda.h> and quda_grid_bridge.h
// from this campaign directory. Nothing from Grid-TXQCD or any production tree is
// included or linked, so this file plus quda_grid_bridge.h can be handed to a QUDA
// developer as-is.
//
// WHAT IT SOLVES. Mpc on a single checkerboard, via GCR preconditioned by QUDA
// multigrid (QUDA_DIRECT_PC_SOLVE + QUDA_MATPC_SOLUTION). NOT the normal operator:
// multigrid preconditions Mpc directly and never squares it. The reference CG at
// the end solves Mpc^dag Mpc, because CG needs an HPD operator -- that asymmetry is
// deliberate and is what doc §3 quantities 1 and 2 are.
//
// PARAMETERS come from the production configuration's own <MULTIGRIDParams>
// metadata (see QudaMgParams in quda_grid_bridge.h), not from a hand-copied
// header, so they cannot drift from the ensemble being inverted.

#include <Grid/Grid.h>

#include "quda_grid_bridge.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace Grid;
using namespace grid_quda_benchmark;

namespace {

std::string read_string(int argc, char **argv, const std::string &option, const std::string &fallback)
{
  return GridCmdOptionExists(argv, argv + argc, option) ? GridCmdOptionPayload(argv, argv + argc, option)
                                                        : fallback;
}

int read_int(int argc, char **argv, const std::string &option, int fallback)
{
  if (!GridCmdOptionExists(argv, argv + argc, option)) return fallback;
  std::string arg = GridCmdOptionPayload(argv, argv + argc, option);
  int value = fallback;
  GridCmdOptionInt(arg, value);
  return value;
}

double read_double(int argc, char **argv, const std::string &option, double fallback)
{
  if (!GridCmdOptionExists(argv, argv + argc, option)) return fallback;
  std::string arg = GridCmdOptionPayload(argv, argv + argc, option);
  double value = fallback;
  GridCmdOptionFloat(arg, value);
  return value;
}

double median_of(std::vector<double> v)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

} // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);
  int status = 0;
  try {
    // wilson | clover. Wilson is doc §4's control: MG is EXPECTED to lose on it,
    // since the solve converges in far fewer iterations while MG setup is
    // unchanged. csw is ignored for wilson.
    const std::string action_name = read_string(argc, argv, "--probe-action", "clover");
    if (action_name != "wilson" && action_name != "clover")
      throw std::runtime_error("--probe-action must be wilson|clover");
    const bool clover = (action_name == "clover");

    const std::string cb_name = read_string(argc, argv, "--probe-checkerboard", "even");
    const std::string cfg = read_string(argc, argv, "--probe-cfg", "");
    const std::string mg_mode = read_string(argc, argv, "--probe-mg-mode", "like_for_like");
    const std::string sloppy_name = read_string(argc, argv, "--probe-sloppy", "single");
    // mixed = QUDA's production default (fp32 precon, HALF null vectors + halos);
    // single = whole hierarchy fp32, the like-for-like row against Grid's fp32.
    const std::string mg_precision = read_string(argc, argv, "--probe-mg-precision", "mixed");
    const double mass = read_double(argc, argv, "--probe-mass", -0.2416);
    const double csw = read_double(argc, argv, "--probe-csw", 1.20536588031793);
    const double tol = read_double(argc, argv, "--probe-tol", 1.0e-10);
    const int maxiter = read_int(argc, argv, "--probe-maxiter", 5000);
    const int repeats = read_int(argc, argv, "--probe-solve-repeats", 3);
    const int run_cg = read_int(argc, argv, "--probe-run-cg", 1);
    // ⛔ REQUIRED for a fp64 CG row. QUDA's MG setup asserts that the sloppy
    // precision matches the hierarchy's ("Precisions 4 8 do not match",
    // coarse_op_24.cu:115), so --probe-sloppy double ABORTS during MG setup --
    // before the CG reference is ever reached, since MG is built first. Skipping
    // MG is the only way to measure QUDA CG in double.
    const int run_mg = read_int(argc, argv, "--probe-run-mg", 1);
    const int run_verify = read_int(argc, argv, "--probe-mg-verify", 0);

    if (cb_name != "even" && cb_name != "odd") throw std::runtime_error("--probe-checkerboard must be even|odd");
    if (mg_mode != "like_for_like" && mg_mode != "production")
      throw std::runtime_error("--probe-mg-mode must be like_for_like|production");
    if (mg_precision != "mixed" && mg_precision != "single")
      throw std::runtime_error("--probe-mg-precision must be mixed|single"
                               " (double cannot be compiled -- see the source comment)");
    const int cb = (cb_name == "odd") ? Odd : Even;

    // Production runs sloppy=SINGLE with RECONS_12 (config <CudaSloppyPrecision>,
    // <CudaSloppyReconstruct>). The v3 CG rows used the `strict` preset
    // (double/no), so the MG rows cannot share a precision setting with them --
    // that is recorded rather than silently reconciled.
    // Outer ("precise") working precision. Default double; `single` makes the
    // WHOLE QUDA solve fp32, which is what a like-for-like comparison against a
    // fully-fp32 Grid solve needs -- QUDA's MG forces its sloppy precision to
    // match the hierarchy, so an fp64 outer is not available to it with MG on.
    // ⚠️ fp32 cannot reach 1e-10; run these at tol 1e-6.
    const std::string precise_name = read_string(argc, argv, "--probe-precise", "double");
    QudaPrecision precise = QUDA_DOUBLE_PRECISION;
    if (precise_name == "single") {
      precise = QUDA_SINGLE_PRECISION;
    } else if (precise_name != "double") {
      throw std::runtime_error("--probe-precise must be single|double");
    }

    QudaPrecision sloppy = QUDA_SINGLE_PRECISION;
    QudaReconstructType sloppy_recon = QUDA_RECONSTRUCT_12;
    if (sloppy_name == "double") {
      sloppy = QUDA_DOUBLE_PRECISION;
      sloppy_recon = QUDA_RECONSTRUCT_NO;
    } else if (sloppy_name != "single") {
      throw std::runtime_error("--probe-sloppy must be single|double");
    }

    GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
        GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
    const bool boss = UGrid->IsBoss();

    // ---- Gauge field: the ONE place Grid does real work, deliberately untimed ---
    std::vector<int> seeds({1, 2, 3, 4});
    GridParallelRNG pRNG(UGrid);
    pRNG.SeedFixedIntegers(seeds);

    LatticeGaugeField Umu(UGrid);
    if (cfg.empty()) {
      SU<Nc>::HotConfiguration(pRNG, Umu);
    } else {
      FieldMetaData header;
      IldgReader reader;
      reader.open(cfg);
      reader.readConfiguration(Umu, header);
      reader.close();
    }
    double grid_plaquette = WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu);

    // ---- Stout smearing, applied through GRID's Smear_Stout --------------------
    //
    // Deliberately the SAME code path as probe_grid_mg_schur_clover.cc rather than
    // a QUDA-side equivalent: two independent smearing implementations is exactly
    // the kind of thing that silently differs and voids the comparison. QUDA is
    // handed the already-smeared links below.
    //
    // This ensemble's metadata records <STOUT_FERM_STATE> rho=0.125, n_smear=1,
    // orthog_dir=-1 (all four directions, Grid's default).
    const int stout_nsmear = read_int(argc, argv, "--probe-stout-nsmear", 0);
    const double stout_rho = read_double(argc, argv, "--probe-stout-rho", 0.125);
    if (stout_nsmear > 0) {
      Smear_Stout<PeriodicGimplD> stout(stout_rho);
      LatticeGaugeField Usmear(UGrid);
      for (int n = 0; n < stout_nsmear; ++n) {
        stout.smear(Usmear, Umu);
        Umu = Usmear;
      }
      grid_plaquette = WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu);
      if (boss)
        std::cout << GridLogMessage << "stout smearing rho " << stout_rho << " n_smear "
                  << stout_nsmear << " -> smeared plaquette " << grid_plaquette << std::endl;
    }

    if (boss) {
      std::cout << GridLogMessage << "=== probe_quda_mg_clover ===" << std::endl;
      std::cout << GridLogMessage << "action         " << action_name << std::endl;
      std::cout << GridLogMessage << "checkerboard   " << cb_name << std::endl;
      std::cout << GridLogMessage << "mg mode        " << mg_mode << std::endl;
      std::cout << GridLogMessage << "sloppy         " << sloppy_name << std::endl;
      std::cout << GridLogMessage << "mg precision   " << mg_precision
                << (mg_precision == "single" ? " (hierarchy ALL fp32; like-for-like vs Grid fp32)"
                                             : " (fp32 precon + HALF null vectors/halos)")
                << std::endl;
      std::cout << GridLogMessage << "gauge          " << (cfg.empty() ? "hot" : cfg) << std::endl;
      std::cout << GridLogMessage << "grid plaquette " << grid_plaquette << std::endl;
    }

    // Grid-side reference operator. Used ONLY to grade returned solutions, never
    // inside a timed region. Same construction as the Grid probe so the two
    // probes are graded against an identical operator.
    std::vector<Complex> phases(Nd, 1.0);
    phases[Nd - 1] = -1.0;
    WilsonFermionD::ImplParams implParams;
    implParams.boundary_phases = phases;
    WilsonAnisotropyCoefficients anisotropy;
    // Base pointer so both actions share one path -- CompactWilsonCloverFermion
    // derives from WilsonFermion and dispatches virtually.
    std::unique_ptr<WilsonFermionD> fermop;
    if (clover)
      fermop.reset(new CompactWilsonCloverFermionD(Umu, *UGrid, *UrbGrid, mass, csw, csw, 1.0,
                                                   anisotropy, implParams));
    else
      fermop.reset(new WilsonFermionD(Umu, *UGrid, *UrbGrid, mass, implParams));
    WilsonFermionD &Dc = *fermop;

    // ⛔ SYMMETRIC Schur, not the asymmetric one the CG rows use. This is FORCED by
    // QUDA, not chosen: quda/lib/coarse_op.cuh:990 is
    //
    //   if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    //     errorQuda("Unsupported coarsening of matpc = %d", matpc);
    //
    // so QUDA multigrid cannot coarsen either asymmetric form. (Found the hard way:
    // the first run of this probe aborted there. Our own production
    // QudaCloverInverter uses QUDA_MATPC_EVEN_EVEN for the same reason.)
    //
    // Matching the conventions (enum_quda.h:212, LinearOperator.h:358/383):
    //   QUDA asymmetric  A_ee - k^2 D_eo A_oo^-1 D_oe   == Grid SchurDiagMooeeOperator
    //   QUDA symmetric   1 - k^2 A_ee^-1 D_eo A_oo^-1 D_oe == Grid SchurDiagOneOperator
    //
    // So the MG rows solve the SYMMETRIC operator on BOTH backends -- still
    // like-for-like with each other, but a different preconditioning from the CG
    // rows. Both give the solution to the same full system, so the "MG or CG?"
    // question is unaffected; it just must not be described as one operator.
    //
    // Still 2-hop (M_oe ... M_eo), so the coarse stencil finding from the Grid
    // probe carries over unchanged.
    SchurDiagOneOperator<WilsonFermionD, LatticeFermionD> SchurOp(Dc);

    {
      QudaSession session(UGrid);

      QudaOperator qop(UGrid, Umu, clover, mass, csw, precise, sloppy,
                       QUDA_RECONSTRUCT_NO, sloppy_recon, tol, maxiter, /*antiperiodic_time=*/true);

      // SYMMETRIC (see the SchurOp comment above -- asymmetric coarsening is
      // rejected by QUDA). Must be set BEFORE build_multigrid:
      // make_mg_inner_invert_param copies matpc_type from the outer param, which is
      // what keeps the MG hierarchy Schur-consistent with the solve requested.
      qop.invert_param().matpc_type = (cb == Odd) ? QUDA_MATPC_ODD_ODD : QUDA_MATPC_EVEN_EVEN;

      // ---- Gate 1: the gauge field QUDA holds is the one Grid read ------------
      const std::array<double, 3> quda_plaquette = qop.plaquette();
      const double plaquette_deviation = std::abs(grid_plaquette - quda_plaquette[0]);
      // ⚠️ The threshold MUST track the working precision. QUDA stores the gauge
      // field at `precise`, so with --probe-precise single the plaquette can only
      // agree with Grid's fp64 value to ~1e-7; 1e-11 would fail a perfectly
      // correct fp32 run. This is a precision floor, not a slackened gate.
      const double plaquette_tol = (precise == QUDA_SINGLE_PRECISION) ? 1e-6 : 1e-11;
      const bool plaquette_passed = plaquette_deviation <= plaquette_tol;
      if (boss)
        std::cout << GridLogMessage << "quda plaquette " << quda_plaquette[0] << ", deviation "
                  << plaquette_deviation << " -> " << (plaquette_passed ? "PASSED" : "FAILED") << std::endl;
      if (!plaquette_passed) throw std::runtime_error("plaquette gate failed");

      // ---- MG setup (timed, pure QUDA) ---------------------------------------
      QudaMgParams mg = (mg_mode == "production") ? QudaMgParams{} : QudaMgParams::like_for_like();
      mg.run_verify = (run_verify != 0);

      // --probe-mg-precision mixed|single.
      //
      //   mixed  (default, = production): fp32 preconditioner, but HALF null
      //          vectors and HALF coarse halos. QUDA's own best configuration.
      //   single: raises those two to fp32 so the ENTIRE hierarchy is fp32.
      //
      // WHY THE `single` ROW EXISTS. Grid's MG can run an fp32 preconditioner but
      // CANNOT run half at all -- vComplexH has no fp16 arithmetic
      // (Tensor_traits.h:231). So Grid-fp32 vs QUDA-mixed still compares different
      // arithmetic. Grid-fp32 vs QUDA-single is the ONE like-for-like precision
      // comparison available between the two libraries, since the fp64-vs-fp64
      // row cannot be compiled (see below).
      //
      // ⛔ The `single` row is a DIAGNOSTIC: it denies QUDA its half-precision
      // advantage and is therefore expected to be SLOWER than QUDA's best. Never
      // quote it as QUDA's capability; the production-relevant row is `mixed`.
      if (mg_precision == "single") {
        mg.null_prec = QUDA_SINGLE_PRECISION;
        mg.coarse_halo_prec = QUDA_SINGLE_PRECISION;
      }

      // ⛔⛔ AN ALL-DOUBLE QUDA MG ROW IS NOT AVAILABLE IN THIS BUILD, AND THE
      // KNOBS BELOW ARE DELIBERATELY LEFT AT THEIR HALF DEFAULTS BECAUSE OF IT.
      //
      // Setting mg.null_prec / mg.coarse_halo_prec to DOUBLE and running
      // --probe-sloppy double dies inside MG SETUP, not at parameter validation:
      //   "Double precision multigrid has not been enabled"
      //   quda/build-mpi-mg/lib/block_orthogonalize_24_32.cu:290
      // That is a `if constexpr (is_enabled_multigrid_double())` COMPILE-TIME
      // guard -- double-precision MG is a cmake option (QUDA_MULTIGRID_DOUBLE),
      // OFF in quda-install-mpi-mg and off by default upstream. No runtime flag
      // reaches it; enabling it requires rebuilding QUDA.
      //
      // WHY ANYONE WOULD WANT IT: Grid's general-coarsened MG path is fp64
      // throughout, so the Grid-vs-QUDA ratio confounds machinery quality with
      // arithmetic skipped. An all-double QUDA row would separate them.
      // ⛔ It would be a DIAGNOSTIC ONLY -- slower than QUDA's best, so it
      // FLATTERS Grid and must never be quoted as the headline ratio.

      // ⛔ BLOCKING MUST DIVIDE THE LATTICE AT EVERY LEVEL, and QUDA will NOT tell
      // you if it does not. The production blocking {3,3,3,4} is built for 48^3x96
      // (48/3 = 16, 96/4 = 24); on C2's 16^3x48 the x/y/z dimensions give 16/3.
      // QUDA silently built some adjusted coarse grid and spent 586 s in setup --
      // 27x the 2-level leg -- producing a configuration that means nothing.
      // Checked here so the failure is a message rather than ten wasted minutes,
      // or worse, a plausible number at C3.
      {
        Coordinate latt = GridDefaultLatt();
        for (int l = 0; l < mg.n_level - 1; ++l) {
          for (int d = 0; d < 4; ++d) {
            const int b = mg.geo_block_size[l][d];
            if (b > 1 && latt[d] % b != 0) {
              throw std::runtime_error(
                  "MG blocking does not divide the lattice at level " + std::to_string(l) + ", dim "
                  + std::to_string(d) + ": " + std::to_string(latt[d]) + " % " + std::to_string(b)
                  + " != 0. The production blocking targets 48^3x96; --probe-mg-mode production is "
                    "a C3-only row.");
            }
            latt[d] /= b;
          }
        }
      }
      if (boss)
        std::cout << GridLogMessage << "--- MG setup: n_level=" << mg.n_level << ", block "
                  << mg.geo_block_size[0][0] << "." << mg.geo_block_size[0][1] << "."
                  << mg.geo_block_size[0][2] << "." << mg.geo_block_size[0][3] << " ---" << std::endl;
      if (run_mg) {
        qop.build_multigrid(mg);
        if (boss)
          std::cout << GridLogMessage << "MG setup " << qop.mg_setup_seconds() << " s" << std::endl;
      } else if (boss) {
        std::cout << GridLogMessage << "MG SKIPPED (--probe-run-mg 0): CG-only row" << std::endl;
      }

      // ---- Source -------------------------------------------------------------
      pRNG.SeedFixedIntegers(std::vector<int>({11, 22, 33, 44}));
      LatticeFermionD full_src(UGrid);
      random(pRNG, full_src);
      LatticeFermionD src(UrbGrid);
      src.Checkerboard() = cb;
      pickCheckerboard(cb, src, full_src);
      const double source_norm2 = norm2(src);

      LatticeFermionD sol(UrbGrid);
      sol.Checkerboard() = cb;
      LatticeFermionD residual(UrbGrid);
      residual.Checkerboard() = cb;

      // Grid-side grading. Mpc for the MG solve, Mpc^dag Mpc for the CG reference,
      // because those are the systems each actually solves.
      auto mg_residual_of = [&](const LatticeFermionD &x) {
        SchurOp.Op(x, residual);
        residual = residual - src;
        return std::sqrt(norm2(residual) / source_norm2);
      };
      auto cg_residual_of = [&](const LatticeFermionD &x) {
        SchurOp.HermOp(x, residual);
        residual = residual - src;
        return std::sqrt(norm2(residual) / source_norm2);
      };

      // ---- Warm MG solve doubles as the correctness gate ----------------------
      // This is also the untimed warm-up: it absorbs QUDA's autotune so the first
      // TIMED repeat below is already warm, matching the CG row's explicit warm
      // and the Grid probe's warm solve on both of its rows.
      bool mg_passed = true;
      if (run_mg) {
        sol = Zero();
        qop.solve_mg(src, sol);
        const double warm_residual = mg_residual_of(sol);
        mg_passed = warm_residual <= std::max(1e-8, 100.0 * tol);
        if (boss)
          std::cout << GridLogMessage << "MG warm solve: " << qop.mg_last_iterations()
                    << " iters, Grid-side residual " << warm_residual << " -> "
                    << (mg_passed ? "PASSED" : "FAILED") << std::endl;
      }

      // ---- Timed MG repeats ---------------------------------------------------
      std::vector<double> mg_seconds, mg_residuals;
      std::vector<long long> mg_iters;
      for (int r = 0; r < (run_mg ? repeats : 0); ++r) {
        sol = Zero();
        accelerator_barrier();
        UGrid->Barrier();
        const double t0 = usecond();
        qop.solve_mg(src, sol);
        accelerator_barrier();
        UGrid->Barrier();
        mg_seconds.push_back((usecond() - t0) / 1.0e6);
        mg_iters.push_back(qop.mg_last_iterations());
        mg_residuals.push_back(mg_residual_of(sol));
      }

      // ---- Reference: QUDA CG on Mpc^dag Mpc ---------------------------------
      // Same backend, same source, different operator and different Krylov method.
      // This is doc §3 quantity 2 -- immune to Grid-side tuning, because QUDA is
      // its own control.
      std::vector<double> cg_seconds;
      std::vector<long long> cg_iters;
      double cg_residual = 0.0;
      if (run_cg) {
        LatticeFermionD cg_sol(UrbGrid);
        cg_sol.Checkerboard() = cb;
        cg_sol = Zero();
        qop.solve(src, cg_sol); // warm
        for (int r = 0; r < repeats; ++r) {
          cg_sol = Zero();
          accelerator_barrier();
          UGrid->Barrier();
          const double t0 = usecond();
          qop.solve(src, cg_sol);
          accelerator_barrier();
          UGrid->Barrier();
          cg_seconds.push_back((usecond() - t0) / 1.0e6);
          cg_iters.push_back(qop.invert_param().iter);
        }
        cg_residual = cg_residual_of(cg_sol);
      }

      const double mg_median = median_of(mg_seconds);
      const double cg_median = median_of(cg_seconds);

      if (boss) {
        std::cout << GridLogMessage << "=== SUMMARY ===" << std::endl;
        std::cout << GridLogMessage << "action                  " << action_name << std::endl;
        std::cout << GridLogMessage << "mg mode / n_level       " << mg_mode << " / " << mg.n_level
                  << std::endl;
        std::cout << GridLogMessage << "mg precision            " << mg_precision << std::endl;
        std::cout << GridLogMessage << "checkerboard            " << cb_name << std::endl;
        if (run_mg) {
          std::cout << GridLogMessage << "MG setup                " << qop.mg_setup_seconds() << " s"
                    << std::endl;
          std::cout << GridLogMessage << "MG solve (Mpc)          " << mg_median << " s, "
                    << mg_iters[mg_iters.size() / 2] << " iters, Grid-side residual "
                    << *std::max_element(mg_residuals.begin(), mg_residuals.end()) << std::endl;
        }
        if (run_cg)
          std::cout << GridLogMessage << "CG solve (Mpc^dag Mpc)  " << cg_median << " s, "
                    << cg_iters[cg_iters.size() / 2] << " iters, Grid-side residual " << cg_residual
                    << std::endl;
        if (run_mg && run_cg && cg_median > 0.0)
          std::cout << GridLogMessage << "MG/CG (time to solution, NOT per-iteration) "
                    << (mg_median / cg_median) << std::endl;
        std::cout << GridLogMessage << "PROBE RESULT: " << (mg_passed ? "PASS" : "FAIL") << std::endl;
      }
      status = mg_passed ? 0 : 1;

      // Explicit, before the QudaOperator destructor: the MG hierarchy references
      // the resident gauge and clover fields.
      if (run_mg) qop.destroy_multigrid();
    } // QudaSession torn down after every QudaOperator is gone
  } catch (const std::exception &error) {
    std::cerr << "probe_quda_mg_clover: ERROR: " << error.what() << std::endl;
    Grid_finalize();
    return 2;
  }

  Grid_finalize();
  return status;
}
