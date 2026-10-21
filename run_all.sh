#!/bin/sh
# run_all.sh — compiles and runs every experiment program of the artifact inside the
# evmone-simd container and stores each program's output under /out.
#
# Usage (on the host):
#   mkdir -p outputs
#   docker run --rm -m 6g -v "$PWD/outputs:/out" evmone-simd sh /work/evmone/run_all.sh
#
# Optional stages (environment variables, pass with -e NAME=1):
#   RUN_TIMING=1    atv_wallclock (wall-clock time vs gas) and rowmacc_wallclock (row-level instruction timing),
#                   about 2 minutes, machine dependent; with RUN_FALLBACK=1 both are also run on the fallback
#                   build. Without RUN_TIMING=1 the timing files of an earlier run are kept, so a rerun of the
#                   gas programs never changes the timing figures reported in the paper.
#   RUN_SWEEPS=1    seed_sweep.py, seed_sweep_fashion.py, seed_sweep_fashion_cal.py (about 50 minutes);
#                   need the raw datasets mounted at /work/evmone/mnist_data and /work/evmone/fashion_data
#   RUN_FALLBACK=1  builds a second libevmone WITHOUT AVX2 (-O3 -fno-tree-vectorize -fno-tree-slp-vectorize,
#                   no -mavx2 and no SIMDPP_ARCH_X86_AVX2, so every instruction runs its scalar fallback and the
#                   compiler does not auto-vectorize it; the manifest records the count of ymm instructions in the
#                   library, which must be 0), reruns the 28 gas programs against it into /out/fallback/ and
#                   compares each output with the AVX2 run (FALLBACK_COMPARISON.txt); with RUN_TIMING=1 it also runs
#                   the two timing programs against the fallback build (fallback/atv_wallclock.txt,
#                   fallback/rowmacc_wallclock.txt, timing only, not compared); about 13 minutes (one more build)
#   RUN_PRICE=1     rebuilds libevmone with the XMACC_S price set to 26 and then 84 and reruns
#                   atv_cost_split at each price (atv_cost_split_xmaccs26.txt, atv_cost_split_xmaccs84.txt);
#                   about 2 x 2 minutes. Always the last stage, because it changes the gas table.
#
# price_neutral.py derives the time-neutral price of XMACC_N from rowmacc_wallclock.txt (AVX2 and fallback).
# A stage that is not requested leaves the output files of an earlier run untouched and notes this in
# MANIFEST.txt. The remaining training scripts (train_deep*.py, seedchk_write.py, deep_reference.py)
# regenerate the canonical files already shipped in data/ and are not run here.

cd /work/evmone || exit 1
OUT=/out
mkdir -p "$OUT"
MAN="$OUT/MANIFEST.txt"
: > "$MAN"
{
  echo "artifact run_all.sh"
  echo "date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "evmone commit: $(git rev-parse HEAD 2>/dev/null)"
  echo "evmc commit:   $(cd evmc && git rev-parse HEAD 2>/dev/null)"
  echo "compiler:      $(g++ --version | head -1)"
  echo "cpu:           $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
  echo "avx2:          $(grep -m1 -o -w avx2 /proc/cpuinfo || echo no)"
  echo
} >> "$MAN"

KERNEL_PROGS="atv_closed_form atv_cost_split atv_three_paths atv_price_sweep atv_int8 atv_solidity_scope"
MNIST_PROGS="mnist_sload mnist_layers mnist_e2e_cb mnist_loader_loop mnist_unrolled_simd mnist_batch_marginal mnist_deploy_cb mnist_swar_baseline mnist_rowmacc mnist_codecopy mnist_loops test_fullnet_500_acc fashion_onchain_acc seedchk_evm"
SEC_PROGS="test_security_opcodes test_security_matrix sec_oob sec_xgather sec_xgather2 sec_xtranspose_big sec_xmacc_n sec_jumpdest"
ALL_PROGS="$KERNEL_PROGS $MNIST_PROGS $SEC_PROGS"

compile_cpp() {
  n=$1
  g++ -O2 -std=c++17 -I evmc/include "$n.cpp" -o "$n" -ldl 2> "$OUT/$n.compile.log" && rm -f "$OUT/$n.compile.log"
}

run_cpp() {   # name [outdir]
  n=$1; d=${2:-$OUT}
  printf '%-28s ' "$n"
  if [ -x "./$n" ] || compile_cpp "$n"; then
    s=$(date +%s)
    ./"$n" > "$d/$n.txt" 2>&1; rc=$?
    e=$(date +%s)
    echo "rc=$rc  $((e-s)) s"
    printf '%-28s rc=%d  time=%ds\n' "$n" "$rc" "$((e-s))" >> "$MAN"
  else
    echo "COMPILE FAIL (see $n.compile.log)"
    printf '%-28s COMPILE FAIL\n' "$n" >> "$MAN"
  fi
}

run_py() {
  n=$1
  printf '%-28s ' "$n.py"
  s=$(date +%s)
  python3 "training/$n.py" > "$OUT/$n.txt" 2>&1; rc=$?
  e=$(date +%s)
  echo "rc=$rc  $((e-s)) s"
  printf '%-28s rc=%d  time=%ds\n' "$n.py" "$rc" "$((e-s))" >> "$MAN"
}

kept() {   # note an output kept from an earlier run
  [ -f "$OUT/$1" ] && printf '%-28s kept from an earlier run (%s not set)\n' "$1" "$2" >> "$MAN"
}

echo "== A^T v kernel programs =="
for n in $KERNEL_PROGS; do run_cpp "$n"; done
echo "== MNIST / Fashion-MNIST programs =="
for n in $MNIST_PROGS; do run_cpp "$n"; done
echo "== security programs =="
for n in $SEC_PROGS; do run_cpp "$n"; done
echo "== Python (canonical weight files only) =="
run_py int8_accuracy
run_py int8_accuracy_fashion
run_py fashion_diag
run_py overflow_bound
export OUT
run_py price_neutral     # reads the two rowmacc_wallclock.txt files (kept or just produced)

if [ "${RUN_TIMING:-0}" = "1" ]; then
  echo "== wall-clock: kernel (atv_wallclock) and row-level instruction (rowmacc_wallclock), machine dependent =="
  run_cpp atv_wallclock
  run_cpp rowmacc_wallclock
else
  kept atv_wallclock.txt RUN_TIMING
  kept rowmacc_wallclock.txt RUN_TIMING
fi

if [ "${RUN_SWEEPS:-0}" = "1" ]; then
  echo "== seed sweeps / calibration (need mnist_data/ and fashion_data/) =="
  if [ -d mnist_data ] && [ -d fashion_data ]; then
    run_py seed_sweep
    run_py seed_sweep_fashion
    run_py seed_sweep_fashion_cal
  else
    echo "mnist_data/ or fashion_data/ not mounted; sweeps skipped"
    echo "seed sweeps                  SKIPPED (raw data not mounted)" >> "$MAN"
  fi
else
  for f in seed_sweep.txt seed_sweep_fashion.txt seed_sweep_fashion_cal.txt; do kept "$f" RUN_SWEEPS; done
fi

if [ "${RUN_FALLBACK:-0}" = "1" ]; then
  echo "== scalar fallback: building libevmone without AVX2 and without auto-vectorization (build_noavx2) =="
  AVX2_FLAGS="-march=native -mavx2 -mfma -O3 -DSIMDPP_ARCH_X86_AVX2"
  NOAVX2_FLAGS="-O3 -fno-tree-vectorize -fno-tree-slp-vectorize"
  sed -i "s/add_compile_options($AVX2_FLAGS)/add_compile_options($NOAVX2_FLAGS)/" CMakeLists.txt
  if grep -q "add_compile_options($NOAVX2_FLAGS)" CMakeLists.txt; then
    if cmake -S . -B build_noavx2 -DCMAKE_BUILD_TYPE=Release > "$OUT/rebuild_noavx2.log" 2>&1 && \
       cmake --build build_noavx2 --target evmone -j2 >> "$OUT/rebuild_noavx2.log" 2>&1; then
      mkdir -p "$OUT/fallback"
      cp build/lib/libevmone.so /tmp/libevmone_avx2.so
      cp build_noavx2/lib/libevmone.so build/lib/libevmone.so
      echo "fallback library:            build_noavx2/lib/libevmone.so (compiled with $NOAVX2_FLAGS: no -mavx2, no SIMDPP_ARCH_X86_AVX2)" >> "$MAN"
      YMM_F=$(objdump -d build_noavx2/lib/libevmone.so | grep -c ymm); YMM_A=$(objdump -d /tmp/libevmone_avx2.so | grep -c ymm)
      echo "ymm (256-bit AVX) instructions:  fallback library $YMM_F, AVX2 library $YMM_A (objdump -d | grep -c ymm)" >> "$MAN"
      rm -rf build_noavx2      # the object files of the second build are not needed any more; this keeps the
                               # memory of the container (-m 6g) free for the large K=4096 kernel programs
      for n in $ALL_PROGS; do run_cpp "$n" "$OUT/fallback"; done
      if [ "${RUN_TIMING:-0}" = "1" ]; then
        echo "== wall-clock on the fallback build (timing only) =="
        run_cpp atv_wallclock "$OUT/fallback"
        run_cpp rowmacc_wallclock "$OUT/fallback"
      else
        kept fallback/atv_wallclock.txt RUN_TIMING
        kept fallback/rowmacc_wallclock.txt RUN_TIMING
      fi
      cp /tmp/libevmone_avx2.so build/lib/libevmone.so
      {
        echo "Scalar-fallback run (build_noavx2: -O3 only, no AVX2) compared with the AVX2 run, file by file (cmp):"
        for f in "$OUT"/fallback/*.txt; do
          b=$(basename "$f")
          [ "$b" = "atv_wallclock.txt" ] && continue
          [ "$b" = "rowmacc_wallclock.txt" ] && continue
          if cmp -s "$f" "$OUT/$b"; then echo "identical   $b"; else echo "DIFFERENT   $b"; fi
        done
      } > "$OUT/FALLBACK_COMPARISON.txt"
      echo "fallback comparison:         $(grep -c '^identical' "$OUT/FALLBACK_COMPARISON.txt") identical, $(grep -c '^DIFFERENT' "$OUT/FALLBACK_COMPARISON.txt") different (see FALLBACK_COMPARISON.txt)" >> "$MAN"
      cat "$OUT/FALLBACK_COMPARISON.txt"
    else
      echo "BUILD FAIL (see rebuild_noavx2.log)"
      echo "fallback build               BUILD FAIL (see rebuild_noavx2.log)" >> "$MAN"
    fi
    sed -i "s/add_compile_options($NOAVX2_FLAGS)/add_compile_options($AVX2_FLAGS)/" CMakeLists.txt
  else
    echo "compile-option line not found; fallback stage skipped"
    echo "fallback build               SKIPPED (compile-option line not found)" >> "$MAN"
  fi
else
  kept FALLBACK_COMPARISON.txt RUN_FALLBACK
fi

if [ "${RUN_PRICE:-0}" = "1" ]; then
  for g in 26 84; do
    echo "== price sensitivity: rebuilding libevmone with XMACC_S gas = $g =="
    sed -i "s/\[OP_XMACC_S\] = [0-9]*;/[OP_XMACC_S] = $g;/" lib/evmone/instructions_traits.hpp
    if grep -q "\[OP_XMACC_S\] = $g;" lib/evmone/instructions_traits.hpp; then
      if cmake --build build --target evmone -j2 > "$OUT/rebuild_xmaccs$g.log" 2>&1; then
        printf '%-28s ' "atv_cost_split (XMACC_S=$g)"
        s=$(date +%s)
        ./atv_cost_split > "$OUT/atv_cost_split_xmaccs$g.txt" 2>&1; rc=$?
        e=$(date +%s)
        echo "rc=$rc  $((e-s)) s"
        printf '%-28s rc=%d  time=%ds  (libevmone rebuilt with XMACC_S=%s)\n' "atv_cost_split_xmaccs$g" "$rc" "$((e-s))" "$g" >> "$MAN"
      else
        echo "REBUILD FAIL (see rebuild_xmaccs$g.log)"
        echo "atv_cost_split_xmaccs$g           REBUILD FAIL" >> "$MAN"
      fi
    else
      echo "sed did not find the XMACC_S gas line; price stage skipped"
      echo "atv_cost_split_xmaccs$g           SKIPPED (gas line not found)" >> "$MAN"
    fi
  done
else
  for g in 26 84; do kept "atv_cost_split_xmaccs$g.txt" RUN_PRICE; done
fi

echo >> "$MAN"
echo "sha256 of outputs:" >> "$MAN"
(cd "$OUT" && sha256sum *.txt fallback/*.txt 2>/dev/null | grep -v MANIFEST) >> "$MAN"
echo "== done; outputs in $OUT =="
cat "$MAN"
