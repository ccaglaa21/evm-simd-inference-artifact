#!/bin/sh
# Self-test: reproduces the closed-form gas model (synthetic, needs no data files).
# Success criterion: predicted==measured with diff 0 on 10 K x 2 paths -> "ALL CHECKS PASSED".
set -e
cd /work/evmone
g++ -O2 -std=c++17 -I evmc/include atv_kapaliform.cpp -o atv_kapaliform -ldl
./atv_kapaliform | tee /tmp/selftest.out
grep -q "ALL CHECKS PASSED" /tmp/selftest.out && echo "SELFTEST: OK (closed-form reproduced, zero error)"
