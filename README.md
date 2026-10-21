# Artifact: Data Vectorization in Smart Contracts

This repository is the reproducibility artifact of the manuscript
*"Data Vectorization in Smart Contracts: SIMD Instructions and Memory Views for On-Chain Machine Learning Inference"*
(Ç. Nazar, S. Topal; submitted to the Journal of Parallel and Distributed
Computing). It contains the evmone patch that adds the
`MatrixHeader` descriptor and the SIMD instruction family, every measurement
program behind the numbers reported in the paper, the canonical trained
weights, and the reference outputs of a complete run.

The work extends the first author's M.Sc. thesis *Data Vectorization in
Blockchain Smart Contracts* (Gebze Technical University, May 2026).

## Contents

| Path | What it is |
|---|---|
| `EVMONE_COMMIT.txt`, `EVMC_COMMIT.txt` | pinned upstream commits: evmone `4938c0a3`, EVMC `496ce0f` |
| `simd_opcodes.patch` | the extension (647 lines): instruction bodies, gas table, traits, build glue |
| `xmacc_n.patch` | the row-level instruction `XMACC_N` (0xe6, element-count operand), applied after `simd_opcodes.patch` |
| `evmc_changes.patch` | one enum entry added to the EVMC header |
| `numpy_simd.hpp` | `MatrixHeader` operations (copied into `lib/evmone/` before the patch is applied) |
| `libsimdpp-2.1/` | header-only SIMD library, vendored from github.com/p12tic/libsimdpp @ `55860b50` (Boost Software License 1.0, see `THIRD_PARTY_LICENSES`) |
| `experiments/` | 30 C++ measurement programs and the Solidity baseline contract (`ATVBaseline.sol`) |
| `training/` | 11 Python scripts: training, quantization, seed sweeps, calibration experiment, int8 study, accumulator overflow bound |
| `data/` | canonical seed-42 weights and biases, 500-image batches, float reference predictions, seed-31337 spot-check weights |
| `Dockerfile`, `run_selftest.sh` | isolated build of the patched interpreter and the closed-form self-test |
| `run_all.sh` | compiles and runs every program and writes one output file per program |
| `outputs/` | reference outputs of `run_all.sh` on the declared machine, with `MANIFEST.txt` (date, commits, compiler, CPU, sha256 of every file) |

## Build and self-test (about 10 minutes, Docker)

```
docker build -t evmone-simd .
docker run --rm -m 6g evmone-simd
```

The image clones evmone and EVMC at the pinned commits, applies the two
patches, builds `libevmone.so` from scratch and runs `atv_closed_form`, which
reproduces the closed-form gas model: 10 problem sizes x 2 paths with
predicted == measured gas (difference 0), the coefficients
78.5625/10.125/23 and 4.1875/22.125/36, and the asymptote 18.76. The last
line must be `SELFTEST: OK (closed-form reproduced, zero error)`.

The patched build is compiled with `-march=native -mavx2 -mfma -O3
-DSIMDPP_ARCH_X86_AVX2` (one `add_compile_options` line in `CMakeLists.txt`),
so the Docker host must be an x86-64 machine with AVX2. Compiling with
`-O3 -fno-tree-vectorize -fno-tree-slp-vectorize` instead gives the
scalar-fallback build of the same interpreter, in which every instruction runs
its scalar path and the compiler does not auto-vectorize it (the manifest
records the number of 256-bit `ymm` instructions in each library); the
`RUN_FALLBACK=1` stage below produces that build and checks it against the
AVX2 build.

## Reproducing every number in the paper

```
mkdir -p outputs
docker run --rm -m 6g -v "$PWD/outputs:/out" evmone-simd sh /work/evmone/run_all.sh
```

This runs the 30 C++ programs and the 4 Python scripts that need only the
shipped `data/` (about 4 minutes) and writes `outputs/<program>.txt` plus
`outputs/MANIFEST.txt`. Four optional stages are enabled with environment
variables:

| Variable | Stage | Needs | Time |
|---|---|---|---|
| `-e RUN_PRICE=1` | rebuilds `libevmone` with the `XMACC_S` price set to 26 and then 84 and reruns `atv_cost_split` at each price (`atv_cost_split_xmaccs26.txt`, `atv_cost_split_xmaccs84.txt`) | nothing | about 2 x 2 min |
| `-e RUN_TIMING=1` | `atv_wallclock` (wall-clock time vs gas, time-neutral price `g*`) and `rowmacc_wallclock` (row-level instruction timing); without it the timing files of an earlier run are kept, so a rerun of the gas programs never changes the reported timing figures | nothing | about 2 min (adaptive sampling) |
| `-e RUN_FALLBACK=1` | builds a second `libevmone` without AVX2 and without auto-vectorization (`-O3 -fno-tree-vectorize -fno-tree-slp-vectorize`, directory `build_noavx2`), reruns the 28 gas programs against it into `outputs/fallback/` and compares every file with the AVX2 run (`FALLBACK_COMPARISON.txt`); with `RUN_TIMING=1` also runs `atv_wallclock` and `rowmacc_wallclock` on the fallback build (`fallback/atv_wallclock.txt`, `fallback/rowmacc_wallclock.txt`, timing only) | nothing | about 13 min |
| `-e RUN_SWEEPS=1` | `seed_sweep.py`, `seed_sweep_fashion.py`, `seed_sweep_fashion_cal.py` (five seeds each) | the raw MNIST and Fashion-MNIST IDX files mounted at `/work/evmone/mnist_data` and `/work/evmone/fashion_data` | about 50 min |

Example with everything enabled:

```
docker run --rm -m 6g -e RUN_PRICE=1 -e RUN_TIMING=1 -e RUN_FALLBACK=1 -e RUN_SWEEPS=1 \
  -v "$PWD/outputs:/out" \
  -v "$HOME/mnist_data:/work/evmone/mnist_data:ro" \
  -v "$HOME/fashion_data:/work/evmone/fashion_data:ro" \
  evmone-simd sh /work/evmone/run_all.sh
```

All gas values are deterministic: the `outputs/` shipped here were produced
on an Intel i5-1135G7 (AVX2) inside a VirtualBox virtual machine and match
the values measured on the original development host bit for bit. Only the
wall-clock figures in `atv_wallclock.txt` and `rowmacc_wallclock.txt` depend on the machine.

The dual-path claim of the paper (the AVX2 path and the scalar fallback of
every instruction give bit-identical results) is checked by the
`RUN_FALLBACK=1` stage: the same 28 programs, including the 500 + 500 + 100
image tests, are rerun against the interpreter built without AVX2, and
`outputs/FALLBACK_COMPARISON.txt` lists, file by file, whether the output is
identical to the AVX2 run. The fallback outputs are kept in
`outputs/fallback/`.

## Where each reported number comes from

| Paper item | Program | Output file |
|---|---|---|
| Pure-compute scaling table, closed-form coefficients, asymptote 18.76, static prediction with zero deviation | `experiments/atv_closed_form.cpp` | `atv_closed_form.txt` |
| Three-bucket split (SLOAD / memory / compute) at K = 256, 1024, 4096; ratios 18.39 / 18.67 / 18.74; memory cost of the physical-copy path | `experiments/atv_cost_split.cpp` | `atv_cost_split.txt` |
| End-to-end ratios 15.71 / 6.86 / 1.56 | `experiments/atv_three_paths.cpp` | `atv_three_paths.txt` |
| Price sensitivity: measured rows at 26 and 84 gas (interpreter rebuilt) | `experiments/atv_cost_split.cpp` via `RUN_PRICE=1` | `atv_cost_split_xmaccs26.txt`, `atv_cost_split_xmaccs84.txt` |
| Price sensitivity: instruction count `K^2/8` and the derived sweep (cross-check of the measured rows) | `experiments/atv_price_sweep.cpp` | `atv_price_sweep.txt` |
| Wall-clock time vs gas, throughput, time-neutral price `g*` | `experiments/atv_wallclock.cpp` via `RUN_TIMING=1` | `atv_wallclock.txt` |
| int8 storage trade-off | `experiments/atv_int8.cpp` | `atv_int8.txt` |
| Inner-product gas of the scalar path (320,064) and the reference outputs for the Solidity comparison | `experiments/atv_solidity_scope.cpp` | `atv_solidity_scope.txt` |
| Solidity baseline (see "Solidity measurement record" below) | `experiments/ATVBaseline.sol` | recorded below |
| Cold single-image cost anatomy (32,814,719; 88.6 % weight read) | `experiments/mnist_sload.cpp` | `mnist_sload.txt` |
| Per-layer compute profile (layer 1 share) | `experiments/mnist_layers.cpp` | `mnist_layers.txt` |
| End-to-end cold / warm / pure-compute ratios 1.22 / 2.40 / 4.92 | `experiments/mnist_e2e_cb.cpp` | `mnist_e2e_cb.txt` |
| Batch processing: marginal cost per image, 5.16, linear model intercept, N = 16 ratios | `experiments/mnist_batch_marginal.cpp` | `mnist_batch_marginal.txt` |
| Deployable contracts: 1,344 B / 1,774 B, 10.06, scalar loop factor | `experiments/mnist_deploy_cb.cpp` | `mnist_deploy_cb.txt` |
| Looped weight loader: 538,979 B to 1,264 B, 84.01 gas per slot, 3.42 % | `experiments/mnist_loader_loop.cpp` | `mnist_loader_loop.txt` |
| Unrolled SIMD upper bound 23.29, 3,462,418 gas, loop factor 4.74, gas per multiply-accumulate | `experiments/mnist_unrolled_simd.cpp` | `mnist_unrolled_simd.txt` |
| MNIST fidelity and accuracy (500/500, 490/500) | `experiments/test_fullnet_500_acc.cpp` | `test_fullnet_500_acc.txt` |
| Fashion-MNIST fidelity, quantization match and accuracy (500/500, 466/500, 434/500, 443/500) | `experiments/fashion_onchain_acc.cpp` | `fashion_onchain_acc.txt` |
| Seed-31337 spot check (100/100) | `experiments/seedchk_evm.cpp` (weights from `training/seedchk_write.py`) | `seedchk_evm.txt` |
| Dual-path determinism: the AVX2 build and the scalar-fallback build give bit-identical outputs (500 + 500 + 100 images and every gas value) | the 28 gas programs rerun against `build_noavx2/lib/libevmone.so` via `RUN_FALLBACK=1` | `fallback/*.txt`, `FALLBACK_COMPARISON.txt` |
| Wall-clock time of the scalar-fallback build (AVX2 versus scalar execution time of the same bytecode) | `experiments/atv_wallclock.cpp` on `build_noavx2` via `RUN_FALLBACK=1` | `fallback/atv_wallclock.txt` |
| Wall-clock time of the row-level instruction: Paths S, B and N on a layer-1-shaped kernel (128 x 784), AVX2 build and fallback build | `experiments/rowmacc_wallclock.cpp` via `RUN_TIMING=1` (and `RUN_FALLBACK=1`) | `rowmacc_wallclock.txt`, `fallback/rowmacc_wallclock.txt` |
| Packed-multiplication scalar baseline (Path S): pure computational cost of the unrolled network with standard instructions only, compared with Paths B and C | `experiments/mnist_swar_baseline.cpp` | `mnist_swar_baseline.txt` |
| Row-level instruction (Path N, `XMACC_N`): unrolled single-image compute with one instruction per neuron, and the deployable looped contract with its marginal cost per image | `experiments/mnist_rowmacc.cpp` | `mnist_rowmacc.txt` |
| Deployable contracts with a loop per neuron and an unrolled inner body, for the fixed-width path (B2) and the packed scalar baseline (S2): marginal cost per image against the per-group loops and the row-level contract | `experiments/mnist_loops.cpp` | `mnist_loops.txt` |
| Weights as contract code (code-as-storage): the Path N contract with the 13,840 weight words stored in 19 data contracts and read with `EXTCODECOPY`, compact memory map; cold and warm cost, EIP-7825 cap check, deployment cost from the gas schedule | `experiments/mnist_codecopy.cpp` | `mnist_codecopy.txt` |
| 32-bit accumulator analysis: worst-case bound per layer over the admissible input range and the largest value reached on the canonical images | `training/overflow_bound.py` | `overflow_bound.txt` |
| Quantization loss over five seeds and 10,000 images; calibration experiment | `training/seed_sweep.py`, `training/seed_sweep_fashion.py`, `training/seed_sweep_fashion_cal.py` via `RUN_SWEEPS=1` | `seed_sweep.txt`, `seed_sweep_fashion.txt`, `seed_sweep_fashion_cal.txt` |
| int8 accuracy study | `training/int8_accuracy.py`, `training/int8_accuracy_fashion.py` | `int8_accuracy.txt`, `int8_accuracy_fashion.txt` |
| Memory-safety suite | `experiments/sec_oob.cpp`, `sec_xgather.cpp`, `sec_xgather2.cpp`, `sec_xtranspose_big.cpp`, `sec_xmacc_n.cpp`, `test_security_opcodes.cpp`, `test_security_matrix.cpp` | `sec_*.txt`, `test_security_*.txt` |
| JUMPDEST analysis: the one-byte immediates of `XADD_IMM`, `XMUL_IMM`, `XSUB_IMM`, `XGT_IMM` and `XRELU` are skipped by the jump-destination analysis (an immediate 0x5b is not a jump target) | `experiments/sec_jumpdest.cpp` | `sec_jumpdest.txt` |
| Canonical weights in `data/` | `training/train_deep.py`, `training/train_deep_fashion.py`, `training/deep_reference.py` (need the raw datasets) | `data/` |

Programs that need the raw datasets read them from `mnist_data/` and
`fashion_data/` (the standard IDX files `train-images-idx3-ubyte`,
`train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`).

## Solidity measurement record

The compiler baseline of the paper was measured outside this container.
`experiments/ATVBaseline.sol` was compiled with solc `0.8.24+commit.e11b9ed9`
(optimizer enabled, 200 runs) and executed on the Remix VM with the Cancun
fork. The `gasUsed` return value of `computePacked()` was 1,414,312 gas and
that of `computeUnpacked()` was 767,144 gas; both functions returned
`out0, out1, out2 = 6031, 6284, 6137`, which are the reference values printed
by `atv_solidity_scope` (`ref[0..2]`). The paper compares these values with
the 320,064 gas of the inner-product loop of the manual scalar path
(`atv_solidity_scope.txt`). The few tens of gas consumed by the `gasleft()`
calls themselves and by the loop entry and exit are included in the Remix
figures and were not subtracted.

## Instruction set

`simd_opcodes.patch` registers twelve opcodes and `xmacc_n.patch` a thirteenth,
`XMACC_N` (0xe6). `XMACC_N` pops a group count, a weight offset and an input
offset, and accumulates that many eight-lane groups into the accumulator word on
the stack with the arithmetic of `XMACC`; it is priced at 11 gas per group
(charged inside the instruction) with no static cost. On the AVX2 build the
lanes are processed with AVX2 intrinsics (one 128-bit load and one byte
shuffle per word, 16-to-32-bit sign extension, 32-bit multiply and add); the
scalar fallback is plain integer code with the same wrapping arithmetic. Nine opcodes of the first
patch form the family evaluated in the paper: `XMACC` (0xcb), `XMACC_S` (0xe5), `XRELU` (0xcc), `XSUM32`
(0xcd), `XPACK` (0xce), `XGT_IMM` (0xca), `XTRANSPOSE` (0xdd), `XGATHER`
(0xe3) and `XBCAST` (0xe4). The remaining three, `XADD_IMM` (0xc7),
`XMUL_IMM` (0xc8) and `XSUB_IMM` (0xc9), are elementwise lane operations
that are implemented and priced but not used by any experiment. The EVMC
patch additionally registers an unused enum entry `OP_XADD` (0xc0).
Experiment programs compile with `g++ -O2 -std=c++17 -I evmc/include X.cpp -o X -ldl`
from `/work/evmone` inside the container and load `build/lib/libevmone.so` at run time.

## License

The files written for this work are released under the Apache License 2.0
(see `LICENSE`). evmone and EVMC, which the Dockerfile clones, are Apache-2.0;
libsimdpp is distributed under the Boost Software License 1.0. See
`THIRD_PARTY_LICENSES`.

## Citation

See `CITATION.cff`. The archived release of this repository (v1.0.0) is
available at Zenodo: https://doi.org/10.5281/zenodo.23495118
