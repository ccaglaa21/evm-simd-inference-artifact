FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    git g++ cmake make ca-certificates python3 python3-numpy && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY EVMONE_COMMIT.txt EVMC_COMMIT.txt simd_opcodes.patch xmacc_n.patch evmc_changes.patch ./
RUN git clone https://github.com/ethereum/evmone && cd evmone && \
    git checkout $(cat /work/EVMONE_COMMIT.txt) && \
    git submodule update --init --recursive
RUN cd /work/evmone/evmc && git checkout $(cat /work/EVMC_COMMIT.txt) && \
    git apply /work/evmc_changes.patch
COPY libsimdpp-2.1/ /usr/local/include/libsimdpp-2.1/
COPY numpy_simd.hpp /work/evmone/lib/evmone/numpy_simd.hpp
RUN cd /work/evmone && git apply /work/simd_opcodes.patch && git apply /work/xmacc_n.patch
# On a processor that is not x86-64 (64-bit ARM) the AVX2 compile flags are replaced by -O3 with the NEON backend of
# libsimdpp. The row-level instructions then run their NEON path and the element-wise instructions use the NEON
# backend of the library. On x86-64 this step changes nothing.
RUN cd /work/evmone && if [ "$(uname -m)" != "x86_64" ]; then \
      sed -i 's/add_compile_options(-march=native -mavx2 -mfma -O3 -DSIMDPP_ARCH_X86_AVX2)/add_compile_options(-O3 -DSIMDPP_ARCH_ARM_NEON)/' CMakeLists.txt && \
      grep -q 'add_compile_options(-O3 -DSIMDPP_ARCH_ARM_NEON)' CMakeLists.txt; fi
RUN cd /work/evmone && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --target evmone -j2
COPY experiments/ /work/evmone/
COPY data/ /work/evmone/
COPY training/ /work/evmone/training/
COPY run_selftest.sh run_all.sh /work/evmone/
CMD ["/work/evmone/run_selftest.sh"]
