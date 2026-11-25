#!/bin/bash
# =============================================================================
# build_chrono24_pm.sh -- a second Chroma `hmc` whose QUDA keeps 24 chronological-predictor
# channels instead of 12 (2026-11-25, autotune 2).
#
# Why: QUDA hard-codes 12 channels (`#define QUDA_MAX_CHRONO 12` in lib/solve.cpp, its only use)
# and Chroma's QUDA_4D_PREDICTOR takes 2 per monomial (the X and Y solves), without checking. Our
# inputs give every Hasenbusch rung and the tail a predictor, so 6 masses fill the 12 channels and
# a ladder of 7 or more aborts 2 min into its first trajectory:
#   "Requested chrono index 12 is outside of max 12"   (job 61920154, 10 masses)
# With 24 channels a ladder can have up to 12 masses. The number only sizes an array of (empty)
# vectors: nothing is allocated for a channel that is never used, and no solve changes.
#
# What it does (the existing install and build tree are not written):
#   1. compiles QUDA's solve.cpp with the exact command of the existing build (flags.make of the
#      quda_cpp target), unpatched, and checks the object equals the one inside the installed
#      libquda.a byte for byte;
#   2. relinks hmc with the existing build tree's link command and the installed libquda.a, and
#      checks the result equals the installed hmc (byte for byte; or, since the system linker
#      changed after August, the same libraries, section sizes and symbols, see step 2);
#   3. only then: patches a copy of solve.cpp (12 -> 24), compiles it the same way, puts it into a
#      copy of libquda.a in place of solve.cpp.o, relinks hmc against that copy.
# Output: $CHROMA_ROOT/install_chrono24/ bin/hmc, lib/libquda.a, solve.cpp.patch, PROVENANCE.txt
# Work files: $CHROMA_SCRATCH/chrono24/ (scratch, disposable).
# Light (one C++ file, one link, a few minutes): fine on a login node, no allocation.
# Use: launchers take CHROMA_HMC_BIN=$CHROMA_ROOT/install_chrono24/bin/hmc (default: the old hmc).
# =============================================================================
set -euo pipefail
source /global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/chroma/env_chroma_pm.sh

CHRONO=24
QB=$CHROMA_BUILD/quda-qdp-jit-double-nd4-cmake
CB=$CHROMA_BUILD/chroma-quda-qdp-jit-double-nd4-cmake
SRC=$CHROMA_SRC/quda/lib/solve.cpp
OLD_LIB=$CHROMA_INSTALL/quda-qdp-jit-double-nd4-cmake/lib/libquda.a
OLD_HMC=$CHROMA_INSTALL/chroma-quda-qdp-jit-double-nd4-cmake/bin/hmc
NEW=$CHROMA_ROOT/install_chrono24
W=$CHROMA_SCRATCH/chrono24
FM=$QB/lib/CMakeFiles/quda_cpp.dir/flags.make
LINKTXT=$CB/mainprogs/main/CMakeFiles/hmc.dir/link.txt

die() { echo "build_chrono24: $*" >&2; exit 1; }
for f in "$SRC" "$OLD_LIB" "$OLD_HMC" "$FM" "$LINKTXT"; do [ -f "$f" ] || die "missing $f"; done
[ -e "$NEW/bin/hmc" ] && die "$NEW/bin/hmc exists: not overwriting it"
grep -q "^#define QUDA_MAX_CHRONO 12$" "$SRC" || die "solve.cpp does not hold '#define QUDA_MAX_CHRONO 12'"
[ "$(grep -c "QUDA_MAX_CHRONO" "$SRC")" -ge 1 ] || die "unexpected solve.cpp"
[ "$(ar t "$OLD_LIB" | grep -c '^solve\.cpp\.o$')" = 1 ] || die "libquda.a does not hold exactly one solve.cpp.o"
mkdir -p "$W/orig" "$W/patched" "$NEW/bin" "$NEW/lib"

# ---- the existing build's compile command for solve.cpp -----------------------------------------
# make runs it through the shell (the defines carry \" quotes), hence eval.
get() { sed -n "s/^$1 = //p" "$FM"; }
CXX_DEFINES=$(get CXX_DEFINES)
CXX_INCLUDES=$(get CXX_INCLUDES)
CXX_FLAGS=$(get CXX_FLAGS)
CXX=$(sed -n 's#^.*&& \(/[^ ]*/CC\) \$(CXX_DEFINES).*solve\.cpp$#\1#p' "$QB/lib/CMakeFiles/quda_cpp.dir/build.make" | head -1)
[ -x "$CXX" ] || die "compiler of the existing build not found (got '$CXX')"
compile() {  # <source> <object>
  ( cd "$QB/lib" && eval "nice $CXX $CXX_DEFINES $CXX_INCLUDES $CXX_FLAGS -o '$2' -c '$1'" )
}

echo "== 1. unpatched solve.cpp, existing flags, against the object in the installed libquda.a"
compile "$SRC" "$W/orig/solve_rebuilt.o"
( cd "$W/orig" && ar x "$OLD_LIB" solve.cpp.o )
cmp "$W/orig/solve_rebuilt.o" "$W/orig/solve.cpp.o" || die "the recompiled solve.cpp.o differs from the installed one: flags or compiler do not reproduce the build"
echo "   identical"

# ---- the existing build's link command for hmc ----------------------------------------------------
LINK=$(cat "$LINKTXT")
relink() {  # <libquda.a> <output>
  local cmd=${LINK/ -o hmc / -o $2 }
  [ "$cmd" != "$LINK" ] || die "link.txt has no ' -o hmc '"
  local cmd2=${cmd//"$OLD_LIB"/$1}
  [ "$1" = "$OLD_LIB" ] || [ "$cmd2" != "$cmd" ] || die "link.txt does not name $OLD_LIB"
  ( cd "$CB/mainprogs/main" && eval "nice $cmd2" )
}

echo "== 2. relink hmc with the installed libquda.a, against the installed hmc"
relink "$OLD_LIB" "$W/orig/hmc_relinked"
sections() { readelf -S -W "$1" | sed -n 's/^ *\[ *[0-9]*\] //p' | awk '$1 !~ /^\.(dynsym|dynstr|hash|gnu\.hash|gnu\.version.*|rela\..*|debug_str|note\..*)$/ {print $1, $5}'; }
if cmp -s "$W/orig/hmc_relinked" "$OLD_HMC"; then
  echo "   identical"
else
  # 2026-11-25: not identical. The system linker changed since the August build (SUSE binutils
  # 2.45, 2025-11): it exports __bss_start, _edata, _end in .dynsym, which shifts every address
  # by 0x80. Accept that difference only: same libraries, same code and data section sizes,
  # same imported symbols, and no exported symbol other than those three added.
  echo "   not byte-identical; linker: $(ld --version | head -1)"
  diff <(readelf -d "$W/orig/hmc_relinked" | grep NEEDED) <(readelf -d "$OLD_HMC" | grep NEEDED) || die "NEEDED libraries differ"
  diff <(sections "$W/orig/hmc_relinked") <(sections "$OLD_HMC") || die "code or data section sizes differ"
  diff <(nm -D --undefined-only "$W/orig/hmc_relinked" | awk '{print $NF}' | sort) \
       <(nm -D --undefined-only "$OLD_HMC" | awk '{print $NF}' | sort) || die "imported symbols differ"
  EXTRA=$(comm -23 <(nm -D --defined-only "$W/orig/hmc_relinked" | awk '{print $NF}' | sort) \
                   <(nm -D --defined-only "$OLD_HMC" | awk '{print $NF}' | sort) | tr '\n' ' ')
  [ "$EXTRA" = "__bss_start _edata _end " ] || die "exported symbols differ beyond the linker's three: $EXTRA"
  echo "   same libraries, section sizes and imported symbols; exported in addition: $EXTRA(linker)"
fi

echo "== 3. patched solve.cpp (QUDA_MAX_CHRONO 12 -> $CHRONO)"
cp "$SRC" "$W/patched/solve.cpp"
sed -i "s/^#define QUDA_MAX_CHRONO 12$/#define QUDA_MAX_CHRONO $CHRONO/" "$W/patched/solve.cpp"
grep -q "^#define QUDA_MAX_CHRONO $CHRONO$" "$W/patched/solve.cpp" || die "patch did not apply"
diff -u "$SRC" "$W/patched/solve.cpp" > "$NEW/solve.cpp.patch" || true
[ "$(grep -c '^[-+]#define' "$NEW/solve.cpp.patch")" = 2 ] || die "patch is not the one-line change"
compile "$W/patched/solve.cpp" "$W/patched/solve.cpp.o"
cp "$OLD_LIB" "$NEW/lib/libquda.a"
ar r "$NEW/lib/libquda.a" "$W/patched/solve.cpp.o"
ranlib "$NEW/lib/libquda.a"
diff <(ar t "$OLD_LIB") <(ar t "$NEW/lib/libquda.a") > /dev/null || die "member list of the new libquda.a differs from the old"
relink "$NEW/lib/libquda.a" "$NEW/bin/hmc"
if ldd "$NEW/bin/hmc" | grep -q pscratch; then die "new hmc depends on scratch"; fi
if ldd "$NEW/bin/hmc" | grep -q "not found"; then die "new hmc has unresolved libraries"; fi
cmp -s "$NEW/bin/hmc" "$OLD_HMC" && die "new hmc is identical to the old one: the patch did not reach it"

{
  echo "install_chrono24: Chroma hmc with QUDA_MAX_CHRONO $CHRONO (was 12), built $(date '+%F %T %Z') on $(hostname)"
  echo "recipe: $CHROMA_ROOT/build_chrono24_pm.sh"
  echo "base: $OLD_HMC"
  echo "      sha256 $(sha256sum "$OLD_HMC" | cut -d' ' -f1)"
  echo "      QUDA $(git -C "$CHROMA_SRC/quda" log -1 --format=%H) (chromaform build of 2026-08-17)"
  echo "change: solve.cpp.patch (one line); solve.cpp.o replaced in a copy of libquda.a; hmc relinked"
  echo "        with the build tree's link command. Checked first: the unpatched compile reproduces"
  echo "        the installed solve.cpp.o byte for byte; the unpatched relink matches the installed"
  echo "        hmc (byte for byte, or with only the newer linker's three exported symbols added)."
  echo "linker: $(ld --version | head -1)"
  echo "lib/libquda.a sha256 $(sha256sum "$NEW/lib/libquda.a" | cut -d' ' -f1)"
  echo "bin/hmc       sha256 $(sha256sum "$NEW/bin/hmc" | cut -d' ' -f1)"
} > "$NEW/PROVENANCE.txt"
cat "$NEW/PROVENANCE.txt"
echo "== done"
