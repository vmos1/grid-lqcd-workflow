#!/bin/bash
# run_fd_force.sh -- per-monomial finite-difference force-versus-action test
# (src/fd_force/test_fd_force.cc) inside an EXISTING allocation. Sets the physics, routes,
# switches and tolerances of one profile, then hands over to the MAIN tree's
# run_test_grid_mg.sh (one node, 4 ranks x 1 GPU, select_gpu + NUMA gate, 45 min timeout), so
# the output lands in the workspace runs/<YYYY_M_D>_<RUN>/test.log. Never allocates.
#
# Usage (one line):
#   SLURM_JOB_ID=<id> RUN=<name> PROFILE=16|48 SWITCHES=on|off [FD_MONOMIALS=..] [FD_EPS=..]
#     [FD_LINKS=smeared|thin|both] [FD_NDIR=n] [FD_TOL=1e-14] [TOL_ACT=..] [TOL_DRV=..]
#     [TOL_STRANGE=..] [FD_TOL_STRANGE_DERIV=..] [RAT_LO_OVERRIDE=..] [MPI_GEOM=..] [BIN=..]
#     bash run_fd_force.sh [extra Grid args, e.g. --device-mem 20000]
#
# PROFILE=16: every value of perlmutter/smoke16_puregrid.sh (cl3 16^3x48 cfg_11100, beta 6.1,
#   csw 1.24930970916466, u0 0.832605301399891, ladder -0.245,-0.20,-0.10, strange -0.2050,
#   stout 0.125 x1, RAT 1e-4/100/20, seed 200, Grid-CG routes, MPI 1.1.1.4).
# PROFILE=48: base+G of perlmutter/traj48_puregrid.sh (cl21 48^3x96 cfg_2000, beta 6.3,
#   csw 1.20536588031793, u0 0.84570646270714, ladder -0.2416,-0.2400,-0.2320,-0.2180,-0.1870,
#   strange -0.2050, stout 0.125 x1, RAT 0.4/35/20, seed 300, the production rung routes
#   MG 0,1,2 / MG heatbath 0,1 / mixed CG 3 / mixed heatbath 2,3, MPI 1.1.2.2).
# SWITCHES=on: the seven production switches = 1 (FUSED_CLOVER_FORCE, DEVICE_CB, BATCH_SMEAR,
#   IMPORT_SKIP, SHARE_FIELDSTRENGTH, GPU_CLOVER_INV, CLOVER_STENCIL); off: all seven = 0.
# Tolerances: action/heatbath, deriv and strange all FD_TOL (default 1e-14) unless TOL_* given.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
HB=$(cd "$HERE/.." && pwd)
BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
MAIN_LAUNCHER=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune/perlmutter/run_test_grid_mg.sh
die() { echo "run_fd_force: $*" >&2; exit 1; }

: "${SLURM_JOB_ID:?set SLURM_JOB_ID to an existing allocation}"
: "${RUN:?set RUN=<name>}"
: "${PROFILE:?set PROFILE=16 or 48}"
: "${SWITCHES:?set SWITCHES=on or off}"
BIN=${BIN:-$HB/bin/test_fd_force}
[ -x "$BIN" ] || die "binary not executable: $BIN"
[ -f "$MAIN_LAUNCHER" ] || die "main launcher missing: $MAIN_LAUNCHER"

# Clear every driver knob this script does not set (a stray one would change the run).
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(QUDA_|HASEN_|HMC_MG_|USE_HMC_MG$|FORCES_|CKPT_|TXQCD_|WCF_|LAMBDA_MN2$|INTEGRATOR_VERBOSE_MEM$|NO_METROP$|IMPORT_CFG$|TUNE_CG_TOL_)')

case "$PROFILE" in
  16)
    export LATT=16.16.16.48
    MPI_GEOM=${MPI_GEOM:-1.1.1.4}
    export IMPORT_CFG=$BASE/data/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime
    export BETA=6.1 CSW=1.24930970916466 U0=0.832605301399891
    export MASS_LIGHT=-0.245 MASS_STRANGE=-0.2050
    export STOUT_RHO=0.125 STOUT_NSMEAR=1
    export RAT_LO=${RAT_LO_OVERRIDE:-1e-4} RAT_HI=100 RAT_DEGREE=20
    export HASEN_LADDER=-0.245,-0.20,-0.10
    export HMC_SEED_OFFSET=200
    ;;
  48)
    export LATT=48.48.48.96
    MPI_GEOM=${MPI_GEOM:-1.1.2.2}
    export IMPORT_CFG=$BASE/data/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime
    export BETA=6.3 CSW=1.20536588031793 U0=0.84570646270714
    export MASS_LIGHT=-0.2416 MASS_STRANGE=-0.2050
    export STOUT_RHO=0.125 STOUT_NSMEAR=1
    export RAT_LO=${RAT_LO_OVERRIDE:-0.4} RAT_HI=35.0 RAT_DEGREE=20
    export HASEN_LADDER=-0.2416,-0.2400,-0.2320,-0.2180,-0.1870
    export HMC_SEED_OFFSET=300
    export HASEN_GRID_MG_RUNGS=0,1,2 HASEN_GRID_MG_HEATBATH_RUNGS=0,1
    export HASEN_GRID_MIXED_CG_RUNGS=3 HASEN_GRID_MIXED_CG_HEATBATH_RUNGS=2,3
    ;;
  *) die "PROFILE must be 16 or 48, got $PROFILE" ;;
esac
[ -f "$IMPORT_CFG" ] || die "configuration not found: $IMPORT_CFG"

case "$SWITCHES" in
  on) SW=1 ;;
  off) SW=0 ;;
  *) die "SWITCHES must be on or off, got $SWITCHES" ;;
esac
for v in HASEN_GRID_FUSED_CLOVER_FORCE HASEN_GRID_DEVICE_CB HASEN_GRID_BATCH_SMEAR HASEN_GRID_IMPORT_SKIP HASEN_GRID_SHARE_FIELDSTRENGTH HASEN_GRID_GPU_CLOVER_INV HASEN_GRID_CLOVER_STENCIL; do
  export "$v=$SW"
done

FD_TOL=${FD_TOL:-1e-14}
export TUNE_CG_TOL_ACTION=${TOL_ACT:-$FD_TOL}
export TUNE_CG_TOL_DERIV=${TOL_DRV:-$FD_TOL}
export TUNE_CG_TOL_STRANGE=${TOL_STRANGE:-$FD_TOL}

export SLURM_CPU_BIND=cores
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=8

echo "run_fd_force: PROFILE=$PROFILE SWITCHES=$SWITCHES MPI=$MPI_GEOM FD_MONOMIALS=${FD_MONOMIALS:-all} tol act/drv/strange=$TUNE_CG_TOL_ACTION/$TUNE_CG_TOL_DERIV/$TUNE_CG_TOL_STRANGE BIN=$BIN"
exec env BIN="$BIN" LATT="$LATT" MPI_GEOM="$MPI_GEOM" bash "$MAIN_LAUNCHER" "$@"
