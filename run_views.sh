#!/bin/sh
# run_views.sh — three complete runs of the timed experiments on the vector build and on the scalar fallback
# build of this machine, inside the evmone-simd container.
#
#   program           what it measures                                                     output files in /out
#   price_bench       time of every added instruction, the basis of the gas schedule       <machine>_run<r>_price_<build>.txt
#   gram_views        Gram matrix: row-level and matrix-level view instructions            <machine>_run<r>_gram_<build>.txt
#   rowmacc_scaling   dense layer: group-level, row-level and matrix-level instructions    <machine>_run<r>_<build>.txt
#
# <machine> is x86 or arm64, <r> is 1, 2, 3 and <build> is avx2 or neon for the vector build and scalar for
# the fallback build.
#
# Usage (on the host):
#   mkdir -p outputs/scaling_runs
#   docker run --rm -m 6g -v "$PWD/outputs/scaling_runs:/out" evmone-simd sh /work/evmone/run_views.sh
# A subset of the programs can be chosen with -e PROGS="price_bench".
#
# The fallback library is built as in run_all.sh (RUN_FALLBACK=1): no vector flags, no auto-vectorization, on
# AArch64 the NEON path switched off. At the end, for every program, the gas tables and the check lines of its
# six files are compared (the timing lines and tables are machine and build dependent and are excluded).
# Exit status 1 if a run fails or the six files of a program do not agree.
cd /work/evmone || exit 1
OUT=${OUT:-/out}
PROGS=${PROGS:-price_bench gram_views rowmacc_scaling}
mkdir -p "$OUT"
if [ "$(uname -m)" = "x86_64" ]; then
  P=x86; V=avx2; VPAT="ymm"
  VEC_FLAGS="-march=native -mavx2 -mfma -O3 -DSIMDPP_ARCH_X86_AVX2"
  FB_FLAGS="-O3 -fno-tree-vectorize -fno-tree-slp-vectorize"
else
  P=arm64; V=neon; VPAT="sqadd"
  VEC_FLAGS="-O3 -DSIMDPP_ARCH_ARM_NEON"
  FB_FLAGS="-O3 -fno-tree-vectorize -fno-tree-slp-vectorize -DEVMONE_NO_NEON"
fi
det() { awk '/^TIME/{s=1} /^$/{s=0} !s' "$1" | grep -v 'med='; }   # gas tables and check lines only
tag() { case "$1" in price_bench) echo "price_";; gram_views) echo "gram_";; *) echo "";; esac; }
FAIL=0
for n in $PROGS; do g++ -O2 -std=c++17 -I evmc/include "$n.cpp" -o "$n" -ldl || exit 1; done
three() {   # build name: three runs of every program
  for n in $PROGS; do
    for r in 1 2 3; do
      ./"$n" > "$OUT/${P}_run${r}_$(tag "$n")$1.txt" 2>&1; rc=$?
      echo "$n run $r rc=$rc"; [ "$rc" = "0" ] || FAIL=1
    done
  done
}
echo "== vector build ($V): three runs of $PROGS =="
three "$V"
echo "== building the scalar fallback library (build_noavx2) =="
sed -i "s/add_compile_options($VEC_FLAGS)/add_compile_options($FB_FLAGS)/" CMakeLists.txt
if grep -q "add_compile_options($FB_FLAGS)" CMakeLists.txt && \
   cmake -S . -B build_noavx2 -DCMAKE_BUILD_TYPE=Release > "$OUT/rebuild_noavx2_views.log" 2>&1 && \
   cmake --build build_noavx2 --target evmone -j2 >> "$OUT/rebuild_noavx2_views.log" 2>&1; then
  cp build/lib/libevmone.so /tmp/libevmone_vec.so
  cp build_noavx2/lib/libevmone.so build/lib/libevmone.so
  echo "$VPAT instructions: fallback library $(objdump -d build_noavx2/lib/libevmone.so | grep -c "$VPAT"), vector library $(objdump -d /tmp/libevmone_vec.so | grep -c "$VPAT")"
  rm -rf build_noavx2 "$OUT/rebuild_noavx2_views.log"
  echo "== scalar fallback build: three runs of $PROGS =="
  three scalar
  cp /tmp/libevmone_vec.so build/lib/libevmone.so
else
  echo "BUILD FAIL (see rebuild_noavx2_views.log)"; FAIL=1
fi
sed -i "s/add_compile_options($FB_FLAGS)/add_compile_options($VEC_FLAGS)/" CMakeLists.txt
echo "== gas tables and check lines: the six files of every program =="
for n in $PROGS; do
  T=$(tag "$n"); REF="$OUT/${P}_run1_${T}${V}.txt"; SAME=1
  for b in "$V" scalar; do for r in 1 2 3; do
    f="$OUT/${P}_run${r}_${T}${b}.txt"
    if [ -f "$f" ] && [ "$(det "$f")" = "$(det "$REF")" ]; then :; else echo "DIFFERENT   $(basename "$f")"; SAME=0; FAIL=1; fi
  done; done
  [ "$SAME" = "1" ] && echo "identical   $n: six files, sha256 $(det "$REF" | sha256sum | cut -d' ' -f1)"
done
[ "$FAIL" = "0" ] && echo "*** TIMED RUNS COMPLETE: the six files of every program agree ***" || echo "*** INSPECT: a run failed or the files differ ***"
exit $FAIL
