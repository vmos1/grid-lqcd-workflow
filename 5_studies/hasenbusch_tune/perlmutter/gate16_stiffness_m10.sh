#!/bin/bash
# =============================================================================
# gate16_stiffness_m10.sh -- the 16^3 gate of the stiffness screen (driver
# FORCES_ONLY=1 FORCES_STIFFNESS=1, binary _stock_m10; handoff
# __docs/2026_10_14_handoff_stiffness_screen.md step 1 item 7). Seven runs inside
# an EXISTING 1-node allocation, then the comparisons. Never allocates, never submits.
#
# Usage (one line):
#   SLURM_JOB_ID=<id> bash gate16_stiffness_m10.sh
# Optional: M8=, M9=, M10=<binaries>, TAG=<suffix for the run names, e.g. _r2>,
#           GATE_CASE_MIN=<per-case cap in minutes, default 20>
#
# Cases (all: the smoke16 / forces16 recipe, 16^3x48, ladder -0.245,-0.20,-0.10, tail outer,
# MDSTEPS 4, seed 200, production tolerances; CG routes unless stated):
#   f9   m9  forces16, plain FORCES_ONLY pass        \ (a) defaults unchanged, force-only path:
#   f10  m10 forces16, plain FORCES_ONLY pass        /     FORCES_ONLY lines bit-exact
#   t9   m9  smoke16, one trajectory                 \ (a) defaults unchanged, trajectory path:
#   t10  m10 smoke16, one trajectory                 /     dH and both plaquettes bit-exact
#            (m8 vs m9: gate16_levels_m9.sh, 2026-10-14, eight cases PASS)
#   s10  m10 forces16 FORCES_STIFFNESS=1, dirs each,P, eps 1e-3,2e-3, central
#            (b) the undisplaced raw force norms equal t10's first update_P lines
#                ("Force average (raw, pre-pullback)" of the batched trajectory, same seed,
#                same draws) and the gauge force equals f10's;
#            (c) gaugecheck lines: dS2 vs fd2 and dS1 vs fd1 agree;
#            (d) linearity: the contractions at eps 1e-3 and 2e-3 agree, nonlinearity small.
#   s10mg same with the Grid-MG route on rungs 0,1 (deriv and heatbath, the shared hierarchy),
#            MPI_GEOM 1.1.2.2, GRID_MG_BLOCK2=2.2.1.3 (the gated 16^3 MG recipe,
#            runs/2026_9_30_m3mghb16_1122): the trap check, the displaced
#            evaluations must re-import the gauge (contractions agree with s10 to tolerance).
# The comparison table is printed by stiffness_screen_table.py and by greps at the end.
#
# Output: runs/<date>_gate16s_<case><TAG>/ per case and runs/<date>_gate16s_summary<TAG>.txt.
# Time: minutes per case on one node; a case is capped at GATE_CASE_MIN.
# =============================================================================
set -o pipefail

BASE=/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd
HB=$BASE/grid-lqcd-workflow/5_studies/hasenbusch_tune
SMOKE=$HB/perlmutter/smoke16_puregrid.sh
FORCES=$HB/perlmutter/forces16_puregrid.sh
TABLE=$HB/data_extraction_scripts/stiffness_screen_table.py
M8=${M8:-$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m8}
M9=${M9:-$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m9}
M10=${M10:-$HB/bin/gen_qcd_hasenbusch_tune_compact_schur_stock_m10}
TAG=${TAG:-}
GATE_CASE_MIN=${GATE_CASE_MIN:-20}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[ -n "${SLURM_JOB_ID:-}" ] || die "SLURM_JOB_ID is not set (the id of an existing 1-node allocation)"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be a numeric job id"
for f in "$SMOKE" "$FORCES" "$TABLE"; do [ -f "$f" ] || die "missing: $f"; done
for b in "$M8" "$M9" "$M10"; do { [ -f "$b" ] && [ -x "$b" ]; } || die "not an executable: $b"; done
[ "$(grep -c -a -F FORCES_STIFFNESS "$M10" || true)" -gt 0 ] || die "$M10 does not contain FORCES_STIFFNESS"
JOB_STATE=$(squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null | head -1)
[ "$JOB_STATE" = RUNNING ] || die "allocation $SLURM_JOB_ID is not RUNNING (state '${JOB_STATE:-not found}')"

DATE_TAG=$(date +%Y_%-m_%-d)
SUMMARY=$BASE/runs/${DATE_TAG}_gate16s_summary${TAG}.txt
: > "$SUMMARY"
log() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
log "gate16_stiffness_m10 $(date '+%Y-%m-%d %H:%M %Z') job $SLURM_JOB_ID"
log "M8=$M8"
log "M9=$M9"
log "M10=$M10"

# No route, layout or screen knob may leak from this shell into a case: every case passes
# its own (the launchers scrub FORCES_* and HASEN_* themselves, but read the screen knobs,
# the routes and MPI_GEOM from the caller).
while read -r v; do
  unset "$v"
done < <(compgen -e | grep -E '^(FORCES_|HASEN_|GRID_MG_|CUSTOM_|DIAG_|MPI_GEOM$|HMC_SEED_OFFSET$|SRUN_EXTRA$)' || true)

# run_case <case> <launcher> <bin> [VAR=value ...]: one launcher run under a per-case cap;
# the case's variables go only to that launcher (env), never into this shell.
run_case() {
  local case=$1 launcher=$2 bin=$3; shift 3
  local run=gate16s_${case}${TAG}
  local out=$BASE/runs/${DATE_TAG}_gate16s_summary${TAG}.${case}.out
  log "---- case $case: $(basename "$launcher") $(basename "$bin") $*"
  local t0=$SECONDS
  env BIN="$bin" RUN="$run" "$@" timeout -k 30s "${GATE_CASE_MIN}m" bash "$launcher" > "$out" 2>&1
  local rc=$?
  log "case $case: exit=$rc  $((SECONDS - t0)) s  log runs/${DATE_TAG}_${run}/hmc.log"
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    log "case $case hit the ${GATE_CASE_MIN} min cap; waiting for its steps to clear"
    while [ -n "$(squeue -h -s -j "$SLURM_JOB_ID" -o %i 2>/dev/null | grep -v "^$SLURM_JOB_ID\$" || true)" ]; do sleep 10; done
  fi
}

# m8 vs m9 was gated on 2026-10-14 (gate16_levels_m9.sh, job 59785924, eight cases PASS,
# __docs/2026_10_14_general_level_driver_gate.md), so the trajectory reference here is m9.
run_case f9  "$FORCES" "$M9"
run_case f10 "$FORCES" "$M10"
run_case t9  "$SMOKE"  "$M9"
run_case t10 "$SMOKE"  "$M10"
run_case s10 "$FORCES" "$M10" FORCES_STIFFNESS=1 FORCES_STIFF_DIRS=each,P FORCES_STIFF_EPS=1e-3,2e-3
run_case s10mg "$FORCES" "$M10" FORCES_STIFFNESS=1 FORCES_STIFF_DIRS=each,P FORCES_STIFF_EPS=1e-3,2e-3 \
  HASEN_GRID_MG_RUNGS=0,1 HASEN_GRID_MG_HEATBATH_RUNGS=0,1 MPI_GEOM=1.1.2.2 GRID_MG_BLOCK2=2.2.1.3

R=$BASE/runs/${DATE_TAG}_gate16s
L_F9=${R}_f9${TAG}/hmc.log L_F10=${R}_f10${TAG}/hmc.log
L_T8=${R}_t8${TAG}/hmc.log L_T9=${R}_t9${TAG}/hmc.log L_T10=${R}_t10${TAG}/hmc.log
L_S10=${R}_s10${TAG}/hmc.log L_S10MG=${R}_s10mg${TAG}/hmc.log

log ""
log "==== (a) force-only path, m9 vs m10, FORCES_ONLY lines (must be identical) ===="
diff <(grep -a -o 'FORCES_ONLY level .* max=[^ ]*' "$L_F9") <(grep -a -o 'FORCES_ONLY level .* max=[^ ]*' "$L_F10") >> "$SUMMARY" && log "f9 == f10: IDENTICAL" || log "f9 != f10: DIFFER (see above)"
log "==== (a) trajectory path, m9 / m10: dH and plaquettes (m9 reference of 2026-10-14: dH 0.0195541474968195) ===="
for c in t9 t10; do
  f=${R}_${c}${TAG}/hmc.log
  log "$c: $(grep -a -o 'dH = [^ ]*' "$f" | head -1)  $(grep -a -o 'Plaquette: \[ 1 \] [^ ]*' "$f" | tr '\n' ' ')"
done
log "==== (b) s10 undisplaced raw forces vs t10 first update_P raw lines (same seed, same draws) ===="
log "s10 FORCES_STIFF force lines:"
grep -a 'FORCES_STIFF force ' "$L_S10" | sed 's/.*FORCES_STIFF/  /' | tee -a "$SUMMARY"
log "t10 first force lines per level-0/1 action (raw, pre-pullback for smeared; the gauge is thin):"
grep -a -m 20 -E '\[[0-9]\]\[[0-9]+\] Force (average|max)' "$L_T10" | sed 's/.*Integrator : [0-9.]* s : /  /' | tee -a "$SUMMARY"
log "f10 FORCES_ONLY gauge line: $(grep -a -o 'FORCES_ONLY level Gauge sample 0 avg=[^ ]* max=[^ ]*' "$L_F10")"
log "==== (c) gauge-action check and (d) linearity, s10 and s10mg (stiffness_screen_table.py) ===="
uv --directory "$BASE/grid-lqcd-workflow" run python "$TABLE" --levels light=0,strange=0,tail=0,gauge=1 --steps 4,16 --trajl 0.35355339059327379 s10="$L_S10" s10mg="$L_S10MG" >> "$SUMMARY" 2>&1
log "==== trap check: MG route vs CG route, <F_a, M_a v> per direction (must agree to solver tolerance) ===="
paste <(grep -a 'FORCES_STIFF stiff ' "$L_S10" | grep -o 'dir=[^ ]* eps=[^ ]* act=[^ ]* norm2_Mv=[^ ]* dot_v=[^ ]*') \
      <(grep -a 'FORCES_STIFF stiff ' "$L_S10MG" | grep -o 'norm2_Mv=[^ ]* dot_v=[^ ]*') | tee -a "$SUMMARY"
log "==== GridMG import lines in s10mg (every displaced evaluation must import) ===="
grep -a -c -E 'GridMG.*(import|Import)' "$L_S10MG" | tee -a "$SUMMARY"
grep -a -E 'GridMG.*(import|Import)' "$L_S10MG" | head -12 | tee -a "$SUMMARY"
log "summary: $SUMMARY"
