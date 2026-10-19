#!/bin/sh
# run_all.sh — compiles and runs every experiment program of the artifact inside the
# evmone-simd container and stores each program's output under /out.
#
# Usage (on the host):
#   mkdir -p ~/evmone/artifact/outputs
#   docker run --rm -m 6g \
#       -v ~/evmone/artifact/outputs:/out \
#       -v ~/evmone/artifact/run_all.sh:/work/evmone/run_all.sh:ro \
#       evmone-simd sh /work/evmone/run_all.sh
#
# Optional stages (environment variables, pass with  -e RUN_PRICE=1  -e RUN_TIMING=1  -e RUN_SWEEPS=1 ):
#   RUN_PRICE=1   price-sensitivity measurements: rebuild libevmone with the XMACC_S gas price set to
#                 26 and then 84 and rerun atv_cost_split at each price (each rebuild takes a few minutes);
#                 outputs atv_cost_split_xmaccs26.txt and atv_cost_split_xmaccs84.txt
#   RUN_TIMING=1  run atv_wallclock (wall-clock vs gas; up to ~30 minutes, environment dependent)
#   RUN_SWEEPS=1  run the seed sweeps and the calibration experiment (training/seed_sweep*.py);
#                 these train networks and need the raw datasets mounted into the container:
#                   -v ~/evmone/mnist_data:/work/evmone/mnist_data:ro
#                   -v ~/evmone/fashion_data:/work/evmone/fashion_data:ro
#                 (standard IDX downloads; several minutes per script)
#
# The remaining training scripts (train_deep*.py, seedchk_write.py, deep_reference.py) regenerate
# the canonical files already shipped in data/ and are not run here.

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

run_cpp() {
  n=$1
  printf '%-28s ' "$n"
  if g++ -O2 -std=c++17 -I evmc/include "$n.cpp" -o "$n" -ldl 2> "$OUT/$n.compile.log"; then
    rm -f "$OUT/$n.compile.log"
    s=$(date +%s)
    ./"$n" > "$OUT/$n.txt" 2>&1; rc=$?
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

echo "== A^T v kernel programs =="
run_cpp atv_closed_form
run_cpp atv_cost_split
run_cpp atv_three_paths
run_cpp atv_price_sweep
run_cpp atv_int8
run_cpp atv_solidity_scope

echo "== MNIST / Fashion-MNIST programs =="
run_cpp mnist_sload
run_cpp mnist_layers
run_cpp mnist_e2e_cb
run_cpp mnist_loader_loop
run_cpp mnist_unrolled_simd
run_cpp mnist_batch_marginal
run_cpp mnist_deploy_cb
run_cpp test_fullnet_500_acc
run_cpp fashion_onchain_acc
run_cpp seedchk_evm

echo "== security programs =="
run_cpp test_security_opcodes
run_cpp test_security_matrix
run_cpp sec_oob
run_cpp sec_xgather
run_cpp sec_xgather2
run_cpp sec_xtranspose_big

echo "== Python (canonical weight files only) =="
run_py int8_accuracy
run_py int8_accuracy_fashion
run_py fashion_diag

if [ "${RUN_TIMING:-0}" = "1" ]; then
  echo "== wall-clock vs gas (slow) =="
  run_cpp atv_wallclock
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
fi

echo >> "$MAN"
echo "sha256 of outputs:" >> "$MAN"
(cd "$OUT" && sha256sum *.txt | grep -v MANIFEST) >> "$MAN"
echo "== done; outputs in $OUT =="
cat "$MAN"
