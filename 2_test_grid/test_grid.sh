#!/usr/bin/env bash
# Run Grid correctness tests from the build tree.
# Tests are not installed — they live in $GRID_BUILD/tests/.
#
# Known failures with --enable-simd=GEN (Mac local build):
#   Test_innerproduct_norm  — diff_ip_f == 0 (single-prec inner product, GEN SIMD)
#   Test_dwf_dslash_repro   — bytes%8==0 alignment mismatch (GEN SIMD vector width)
# These pass on GPU builds with --enable-simd=NEONv8 or AVX512.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

BIN="$GRID_BUILD/tests"
GRID_ARGS="--grid 4.4.4.4 --mpi 1.1.1.1"

TESTS=(
    Test_simd
    Test_cshift
    Test_stencil
    Test_general_stencil
    Test_gfield_shift
    Test_cayley_even_odd_vec
    Test_meson_field
    # Omitted — passes but takes ~10 min on a 4^4 lattice:
    # Test_dwf_mixedcg_prec
)

pass=0; fail=0; failed_list=()

for t in "${TESTS[@]}"; do
    printf "  %-40s" "$t ..."
    mpirun -np 1 "$BIN/$t" $GRID_ARGS > /tmp/grid_test_"$t".log 2>&1
    if [ $? -eq 0 ]; then
        echo "PASS"
        ((pass++))
    else
        echo "FAIL  (see /tmp/grid_test_${t}.log)"
        ((fail++))
        failed_list+=("$t")
    fi
done

echo ""
echo "Results: $pass passed, $fail failed"
if [ ${#failed_list[@]} -gt 0 ]; then
    echo "Failed: ${failed_list[*]}"
    exit 1
fi
