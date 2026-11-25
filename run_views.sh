#!/bin/sh
# run_views.sh — three complete runs of the view experiments (gram_views) on the vector build and on the
# scalar fallback build of this machine, inside the evmone-simd container.
#
# Usage (on the host):
#   mkdir -p outputs/scaling_runs
#   docker run --rm -m 6g -v "$PWD/outputs/scaling_runs:/out" evmone-simd sh /work/evmone/run_views.sh
#
# It writes /out/<machine>_run{1,2,3}_gram_<vector build>.txt and /out/<machine>_run{1,2,3}_gram_scalar.txt,
# where <machine> is x86 or arm64 and <vector build> is avx2 or neon. The fallback library is built as in
# run_all.sh (RUN_FALLBACK=1): no vector flags, no auto-vectorization, on AArch64 the NEON path switched off.
# At the end the gas tables and the check lines of the six files are compared (the timing tables are machine
# and build dependent and are excluded). Exit status 1 if a run fails or the six files do not agree.
cd /work/evmone || exit 1
OUT=${OUT:-/out}
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
FAIL=0
g++ -O2 -std=c++17 -I evmc/include gram_views.cpp -o gram_views -ldl || exit 1
echo "== vector build ($V): three runs =="
for r in 1 2 3; do
  ./gram_views > "$OUT/${P}_run${r}_gram_${V}.txt" 2>&1; rc=$?
  echo "run $r rc=$rc"; [ "$rc" = "0" ] || FAIL=1
done
echo "== building the scalar fallback library (build_noavx2) =="
sed -i "s/add_compile_options($VEC_FLAGS)/add_compile_options($FB_FLAGS)/" CMakeLists.txt
if grep -q "add_compile_options($FB_FLAGS)" CMakeLists.txt && \
   cmake -S . -B build_noavx2 -DCMAKE_BUILD_TYPE=Release > "$OUT/rebuild_noavx2_views.log" 2>&1 && \
   cmake --build build_noavx2 --target evmone -j2 >> "$OUT/rebuild_noavx2_views.log" 2>&1; then
  cp build/lib/libevmone.so /tmp/libevmone_vec.so
  cp build_noavx2/lib/libevmone.so build/lib/libevmone.so
  echo "$VPAT instructions: fallback library $(objdump -d build_noavx2/lib/libevmone.so | grep -c "$VPAT"), vector library $(objdump -d /tmp/libevmone_vec.so | grep -c "$VPAT")"
  rm -rf build_noavx2 "$OUT/rebuild_noavx2_views.log"
  echo "== scalar fallback build: three runs =="
  for r in 1 2 3; do
    ./gram_views > "$OUT/${P}_run${r}_gram_scalar.txt" 2>&1; rc=$?
    echo "run $r rc=$rc"; [ "$rc" = "0" ] || FAIL=1
  done
  cp /tmp/libevmone_vec.so build/lib/libevmone.so
else
  echo "BUILD FAIL (see rebuild_noavx2_views.log)"; FAIL=1
fi
sed -i "s/add_compile_options($FB_FLAGS)/add_compile_options($VEC_FLAGS)/" CMakeLists.txt
echo "== gas tables and check lines of the six files =="
REF="$OUT/${P}_run1_gram_${V}.txt"
for f in "$OUT/${P}"_run[123]_gram_"${V}".txt "$OUT/${P}"_run[123]_gram_scalar.txt; do
  if [ -f "$f" ] && [ "$(det "$f")" = "$(det "$REF")" ]; then echo "identical   $(basename "$f")"; else echo "DIFFERENT   $(basename "$f")"; FAIL=1; fi
done
echo "sha256 of the gas tables and check lines: $(det "$REF" | sha256sum | cut -d' ' -f1)"
[ "$FAIL" = "0" ] && echo "*** VIEW RUNS COMPLETE: six files agree ***" || echo "*** INSPECT: a run failed or the files differ ***"
exit $FAIL
