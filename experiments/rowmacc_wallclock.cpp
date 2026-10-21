// rowmacc_wallclock.cpp — WALL-CLOCK MEASUREMENT of the row-level instruction (answer to the question
// "does the hardware vector unit contribute once one dispatch covers a whole row?").
//
// Kernel: the first layer of the MNIST network in shape (128 neurons x 784 inputs = 98 groups of 8),
// synthetic signed 16-bit weights and 8-bit inputs, all data placed in memory with CALLDATACOPY.
// Three straight-line programs compute the same 128 dot products:
//   Path S  packed-multiplication scalar (standard instructions only, 8 lanes of 32 bits per word,
//           Kronecker substitution, correction once per neuron) -- the layer-1 scheme of Path S
//   Path B  fixed-width XMACC: per group PUSH MLOAD PUSH MLOAD XMACC, then XSUM32
//   Path N  row-level XMACC_N: one instruction per neuron, then XSUM32
// Method as in atv_wallclock: steady_clock around vm->execute only, adaptive inner repetition
// (sample >= 10 ms), 7..31 samples, MEDIAN + min-max + CV%. The warm-up run checks the outputs of
// all three programs against the C++ reference (bit-exact, 32-bit wrap) and records the gas.
// This program is run on the AVX2 build and on the scalar-fallback build; the two output files are
// compared by eye (timing is excluded from the automatic comparison), the result lines of the
// bit-exact checks and the gas values must be identical.
#include <evmc/evmc.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

static evmc_vm* VM;
static const int64_t GAS = 200000000000LL;
struct Code {
    std::vector<uint8_t> c;
    void b(uint8_t x) { c.push_back(x); }
    void P1(uint8_t v) { b(0x60); b(v); }
    void PUSH(uint64_t v) { if (v < 256) { P1((uint8_t)v); return; } int n = 0; for (uint64_t t = v; t; t >>= 8) ++n;
        b((uint8_t)(0x5F + n)); for (int k = n - 1; k >= 0; --k) b((uint8_t)((v >> (8 * k)) & 0xFF)); }
    void W32(const uint8_t* w) { b(0x7F); for (int i = 0; i < 32; ++i) b(w[i]); }
    void MLOAD() { b(0x51); } void MSTORE() { b(0x52); } void MUL() { b(0x02); } void ADD() { b(0x01); } void SUB() { b(0x03); }
    void SHR() { b(0x1C); } void DUP(int n) { b((uint8_t)(0x7F + n)); } void SWAP(int n) { b((uint8_t)(0x8F + n)); } void POP() { b(0x50); }
};
static void or_bits(uint8_t* w, int bitpos, uint64_t v, int nbits) { for (int k = 0; k < nbits; ++k) if ((v >> k) & 1ULL) { int bit = bitpos + k; w[31 - bit / 8] |= (uint8_t)(1u << (bit % 8)); } }
static uint32_t rnd = 2026u; static uint32_t rnd32() { rnd = rnd * 1664525u + 1013904223u; return rnd; }

int main() {
#ifndef LOCAL_TEST
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
#endif
    const int NEU = 128, I = 784, G = I / 8, OFF = 32768, WIN = 32;
    std::vector<int16_t> W((size_t)NEU * I); std::vector<int> X(I);
    for (auto& w : W) w = (int16_t)((int)(rnd32() % 8192) - 4096);
    for (auto& x : X) x = (int)(rnd32() % 256);
    std::vector<uint32_t> ref(NEU);
    for (int n = 0; n < NEU; ++n) { int64_t s = 0; for (int i = 0; i < I; ++i) s += (int64_t)W[(size_t)n * I + i] * X[i]; ref[n] = (uint32_t)s; }
    // calldata: [XB: 98 words, 16-bit lanes] [XS: 98 words, 32-bit lanes reversed] [WB: 128*98 words] [WS: 128*98 words]
    const size_t XB = 0, XS = XB + (size_t)G * 32, WB = XS + (size_t)G * 32, WS = WB + (size_t)NEU * G * 32, CDS = WS + (size_t)NEU * G * 32;
    std::vector<uint8_t> cd(CDS, 0);
    for (int g = 0; g < G; ++g) {
        uint8_t* xb = &cd[XB + (size_t)g * 32]; uint8_t* xs = &cd[XS + (size_t)g * 32];
        for (int j = 0; j < 8; ++j) { uint16_t u = (uint16_t)X[g * 8 + j]; xb[31 - 2 * j] = u & 0xFF; xb[30 - 2 * j] = (u >> 8) & 0xFF;
            or_bits(xs, 32 * (7 - j), (uint64_t)X[g * 8 + j], 32); }                 // S: input j at lane 7-j
        for (int n = 0; n < NEU; ++n) {
            uint8_t* wb = &cd[WB + ((size_t)n * G + g) * 32]; uint8_t* ws = &cd[WS + ((size_t)n * G + g) * 32];
            for (int j = 0; j < 8; ++j) { uint16_t u = (uint16_t)W[(size_t)n * I + g * 8 + j]; wb[31 - 2 * j] = u & 0xFF; wb[30 - 2 * j] = (u >> 8) & 0xFF;
                or_bits(ws, 32 * j, (uint64_t)(W[(size_t)n * I + g * 8 + j] + OFF), 32); }    // S: weight j at lane j, offset 2^15
        }
    }
    // memory map (same for all programs): inputs at 0, weights at MW, outputs at MO
    const uint64_t MX = 0, MW = (uint64_t)G * 32, MO = MW + (uint64_t)NEU * G * 32, MEND = MO + (uint64_t)NEU * 32;
    auto copy = [&](Code& k, uint64_t cdoff, uint64_t memoff, uint64_t bytes) { k.PUSH(bytes); k.PUSH(cdoff); k.PUSH(memoff); k.b(0x37); };
    auto tail = [&](Code& k) { k.PUSH(MEND); k.PUSH(0); k.b(0xF3); };
    // Path B
    Code kB; copy(kB, XB, MX, (uint64_t)G * 32); copy(kB, WB, MW, (uint64_t)NEU * G * 32);
    for (int n = 0; n < NEU; ++n) { kB.P1(0);
        for (int g = 0; g < G; ++g) { kB.PUSH(MX + (uint64_t)g * 32); kB.MLOAD(); kB.PUSH(MW + ((uint64_t)n * G + g) * 32); kB.MLOAD(); kB.b(0xCB); }
        kB.b(0xCD); kB.PUSH(MO + (uint64_t)n * 32); kB.MSTORE(); }
    tail(kB);
    // Path N
    Code kN; copy(kN, XB, MX, (uint64_t)G * 32); copy(kN, WB, MW, (uint64_t)NEU * G * 32);
    for (int n = 0; n < NEU; ++n) { kN.P1(0); kN.PUSH(MX); kN.PUSH(MW + (uint64_t)n * G * 32); kN.PUSH(G); kN.b(0xE6); kN.b(0xCD); kN.PUSH(MO + (uint64_t)n * 32); kN.MSTORE(); }
    tail(kN);
    // Path S: correction word (sum of inputs) << 15 once, then per neuron the windowed packed dot product
    Code kS; copy(kS, XS, MX, (uint64_t)G * 32); copy(kS, WS, MW, (uint64_t)NEU * G * 32);
    const uint64_t CORR = MEND;                                  // one scratch word after the outputs
    { uint8_t ones[32]; std::memset(ones, 0, 32); for (int j = 0; j < 8; ++j) or_bits(ones, 32 * j, 1, 32);
      kS.W32(ones); kS.P1(0);
      for (int g = 0; g < G; ++g) { kS.PUSH(MX + (uint64_t)g * 32); kS.MLOAD(); kS.DUP(3); kS.MUL(); kS.ADD(); }
      kS.PUSH(224); kS.SHR(); kS.P1(15); kS.b(0x1B); kS.PUSH(CORR); kS.MSTORE(); kS.POP(); }
    for (int n = 0; n < NEU; ++n) { kS.P1(0);                    // T
        for (int g = 0; g < G; ++g) {
            if (g % WIN == 0) kS.P1(0);                          // P
            kS.PUSH(MW + ((uint64_t)n * G + g) * 32); kS.MLOAD(); kS.PUSH(MX + (uint64_t)g * 32); kS.MLOAD(); kS.MUL(); kS.ADD();
            if ((g + 1) % WIN == 0 || g == G - 1) { kS.PUSH(224); kS.SHR(); kS.ADD(); } }
        kS.PUSH(CORR); kS.MLOAD(); kS.SWAP(1); kS.SUB();         // T - correction = exact dot product (signed, 256-bit)
        kS.PUSH(MO + (uint64_t)n * 32); kS.MSTORE(); }
    kS.PUSH(MEND); kS.PUSH(0); kS.b(0xF3);
#ifdef LOCAL_TEST
    { auto dump = [](const char* f, const std::vector<uint8_t>& v) { FILE* o = std::fopen(f, "wb"); std::fwrite(v.data(), 1, v.size(), o); std::fclose(o); };
      dump("rw_S.bin", kS.c); dump("rw_B.bin", kB.c); dump("rw_N.bin", kN.c); dump("rw_cd.bin", cd);
      FILE* o = std::fopen("rw_ref.txt", "w"); for (int n = 0; n < NEU; ++n) std::fprintf(o, "%u\n", ref[n]); std::fclose(o); std::printf("dumped\n"); return 0; }
#endif
    std::printf("Bytecode: S=%zu B=%zu N=%zu bytes | calldata %zu bytes | %d neurons x %d inputs = %d multiply-accumulates\n", kS.c.size(), kB.c.size(), kN.c.size(), CDS, NEU, I, NEU * I);

    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS; msg.input_data = cd.data(); msg.input_size = cd.size();
    int fail = 0;
    auto runOnce = [&](const Code& k, bool check, int lanes /*1: XSUM32 word (uint32 at the low end), 0: S word*/, const char* name) -> long long {
        auto r = VM->execute(VM, nullptr, nullptr, EVMC_SHANGHAI, &msg, k.c.data(), k.c.size());
        long long g = GAS - r.gas_left;
        if (check) {
            bool ok = r.status_code == EVMC_SUCCESS && r.output_data && r.output_size >= MEND;
            if (ok) for (int n = 0; n < NEU && ok; ++n) { const uint8_t* w = r.output_data + MO + (size_t)n * 32;
                uint32_t got = 0; for (int i = 28; i < 32; ++i) got = (got << 8) | w[i];        // low 32 bits of the word
                (void)lanes; if (got != ref[n]) ok = false; }
            std::printf("  [%s] bit-exact against the C++ reference (128 outputs, 32-bit wrap): %s | gas %lld\n", name, ok ? "PASSED" : "FAILED", g);
            if (!ok) fail = 1;
        }
        if (r.release) r.release(&r);
        return g;
    };
    auto timePath = [&](const Code& k, int lanes, const char* name, double& med) {
        long long g1 = runOnce(k, true, lanes, name);
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now(); long long g2 = runOnce(k, false, lanes, name); auto t1 = clk::now();
        if (g2 != g1) { std::printf("  [%s] repeated-run gas differs [FAILED]\n", name); fail = 1; }
        double est = std::chrono::duration<double>(t1 - t0).count(); if (est <= 0) est = 1e-7;
        long long R = (long long)std::ceil(0.010 / est); if (R < 1) R = 1; if (R > 20000) R = 20000;
        std::vector<double> S; double tot = 0;
        while (true) { auto a = clk::now(); for (long long i = 0; i < R; ++i) runOnce(k, false, lanes, name); auto bb = clk::now();
            double dt = std::chrono::duration<double>(bb - a).count(); S.push_back(dt / R); tot += dt; size_t n = S.size();
            if (n >= 3 && ((n >= 7 && tot >= 1.0) || n >= 31 || tot >= 60.0)) break; }
        std::sort(S.begin(), S.end()); med = S[S.size() / 2];
        double mu = 0; for (double x : S) mu += x; mu /= S.size(); double sd = 0; for (double x : S) sd += (x - mu) * (x - mu); sd = std::sqrt(sd / S.size());
        std::printf("  [%s] med=%.6f s  min=%.6f  max=%.6f  CV=%.1f%%  (samples=%zu, inner-reps=%lld)\n", name, med, S.front(), S.back(), 100.0 * sd / mu, S.size(), R);
        std::fflush(stdout);
    };
    double tS, tB, tN;
    timePath(kS, 0, "S", tS); timePath(kB, 1, "B", tB); timePath(kN, 1, "N", tN);
    std::printf("\n================ ROW-LEVEL INSTRUCTION, WALL-CLOCK (layer-1 shape, %d multiply-accumulates) ================\n", NEU * I);
    std::printf("  median time per execution: S %.3f ms | B %.3f ms | N %.3f ms\n", tS * 1e3, tB * 1e3, tN * 1e3);
    std::printf("  time ratios: S/N = %.2f | B/N = %.2f | S/B = %.2f\n", tS / tN, tB / tN, tS / tB);
    std::printf("  nanoseconds per multiply-accumulate: S %.2f | B %.2f | N %.2f\n", tS * 1e9 / (NEU * I), tB * 1e9 / (NEU * I), tN * 1e9 / (NEU * I));
    std::printf("%s\n", fail ? "*** A CHECK FAILED -> timing numbers are not valid ***" : "*** ALL CHECKS PASSED -> timing numbers are valid for this build ***");
#ifndef LOCAL_TEST
    VM->destroy(VM); dlclose(h);
#endif
    return fail;
}
