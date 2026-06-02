#!/bin/bash
# Build gen_qcd_hasenbusch_tune on Perlmutter — WITH QUDA, against YOUR OWN
# Grid-TXQCD build, linking a REUSED (prebuilt) QUDA install.
#
# Architecture (identical to Grid-TXQCD/production/Makefile):
#   * Grid-TXQCD itself is built WITHOUT QUDA — the stock Perlmutter
#     systems/Perlmutter/config-command has no --with-quda, and that's correct.
#   * QUDA is linked HERE, at the binary compile step, via -DGRID_HAVE_QUDA plus
#     -lquda -lqmp against a prebuilt QUDA you point at. The #ifdef GRID_HAVE_QUDA
#     guards in the source then compile in the QUDA strange-force action, which is
#     activated at runtime by QUDA_FORCE=1 (see perlmutter/smoke.sbatch).
#
# Prereqs:
#   1. YOUR Grid-TXQCD build is done — <GRID_TXQCD>/build/grid-config exists:
#        cd $GRID_TXQCD && source sourceme.sh
#        mkdir build && cd build && ../systems/Perlmutter/config-command && make -j 32
#      Pin Grid-TXQCD to the commit our binary targets (see PERLMUTTER_PORTING.md).
#   2. A prebuilt QUDA you can read:  $QUDA_PREFIX/{include, lib/libquda.*}.
#      REUSE the QUDA built for THIS Grid-TXQCD commit (API must match the wrapper
#      headers) — not an unrelated QUDA module. Load a cudatoolkit compatible with it.
#   3. QMP:  if your QUDA install bundles QMP, leave QMP_PREFIX=$QUDA_PREFIX;
#      otherwise point QMP_PREFIX at an install that ships libqmp.
#
# Usage:
#   source $GRID_TXQCD/sourceme.sh        # SAME modules used to build Grid-TXQCD
#   QUDA_PREFIX=/path/to/quda-install bash build.sh

set -e

# ── EDIT THESE (or pass as env) for your Perlmutter layout ───────────────────
GRID_TXQCD=${GRID_TXQCD:-$PSCRATCH/Grid-TXQCD}                 # your own checkout+build
GRID_CONFIG=${GRID_CONFIG:-${GRID_TXQCD}/build/grid-config}    # from your Grid build
TXQCD_PROD=${TXQCD_PROD:-${GRID_TXQCD}/production}             # for params.h

# Reused QUDA (you do NOT build QUDA yourself):
QUDA_PREFIX=${QUDA_PREFIX:-/CHANGE/ME/quda-install}           # has include/ and lib/libquda.*
QMP_PREFIX=${QMP_PREFIX:-${QUDA_PREFIX}}                      # ==QUDA_PREFIX if QMP is bundled
QUDA_STATIC=${QUDA_STATIC:-}                                  # set =1 to link static .a archives
QUDA_LIBDIR=${QUDA_LIBDIR:-lib}                               # some installs use lib64
# ─────────────────────────────────────────────────────────────────────────────

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${HERE}/../src/gen_qcd_hasenbusch_tune.cc
BIN=${HERE}/../bin/gen_qcd_hasenbusch_tune
mkdir -p "$(dirname "$BIN")"

if [ ! -x "$GRID_CONFIG" ]; then
  echo "ERROR: grid-config not found/executable at: $GRID_CONFIG" >&2
  echo "       Build your Grid-TXQCD first (see prereq #1)." >&2
  exit 1
fi
if [ ! -d "${QUDA_PREFIX}/include" ]; then
  echo "ERROR: QUDA_PREFIX has no include/ : ${QUDA_PREFIX}" >&2
  echo "       Set QUDA_PREFIX to a prebuilt QUDA install you can read." >&2
  exit 1
fi

CXX=$($GRID_CONFIG --cxx)
CXXFLAGS="$($GRID_CONFIG --cxxflags) -I${TXQCD_PROD}"
LDFLAGS=$($GRID_CONFIG --ldflags)
LIBS=$($GRID_CONFIG --libs)

# ── QUDA link (mirrors Grid-TXQCD/production/Makefile) ───────────────────────
QLIB=${QUDA_PREFIX}/${QUDA_LIBDIR}
MLIB=${QMP_PREFIX}/${QUDA_LIBDIR}
CXXFLAGS+=" -DGRID_HAVE_QUDA -I${QUDA_PREFIX}/include -I${QMP_PREFIX}/include"
LDFLAGS+=" -L${QLIB} -L${MLIB} -Xlinker -rpath -Xlinker ${QLIB} -Xlinker -rpath -Xlinker ${MLIB}"
if [ -n "${QUDA_STATIC}" ]; then
  LIBS="-l:libquda.a -l:libqmp.a ${LIBS}"
else
  LIBS="-lquda -lqmp ${LIBS}"
fi

echo "Compiler:    ${CXX}"
echo "grid-config: ${GRID_CONFIG}"
echo "QUDA_PREFIX: ${QUDA_PREFIX}  (lib: ${QLIB})"
echo "QMP_PREFIX:  ${QMP_PREFIX}"
echo "Source:      ${SRC}"
echo "Binary:      ${BIN}"
echo "Building (GRID_HAVE_QUDA defined) ..."

$CXX $CXXFLAGS -o "${BIN}" "${SRC}" $LDFLAGS $LIBS

echo "Done: ${BIN}"
echo "--- confirm QUDA actually linked ---"
ldd "${BIN}" 2>/dev/null | grep -iE "quda|qmp" || \
  echo "(no dynamic quda/qmp in ldd — static link, or check QUDA_STATIC / QUDA_LIBDIR)"
