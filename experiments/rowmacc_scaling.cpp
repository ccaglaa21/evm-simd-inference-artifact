// rowmacc_scaling.cpp — ROW-LENGTH SCALING of the row-level instruction (wall-clock and gas).
//
// Question: how do the gas and the execution time of the four programs of rowmacc_wallclock.cpp change with the
// row length? The kernel, the data generator, the four straight-line programs (Path S, B, N, N16), the memory
// maps, the bit-exact checks and the timing method are those of rowmacc_wallclock.cpp, unchanged. The only
// difference is that the shape is a parameter: 128 neurons x I inputs for I in a fixed list of row lengths
// (every I is a multiple of 16, so that the 8-lane and the 16-lane programs cover the row with whole words).
// The largest row is 2048 elements: with weights in [-4096, 4095] and inputs in [0, 255] every dot product
// stays below 2^31 in magnitude, so the saturating accumulation of the instructions is not reached.
// The shape I = 784 repeats the measurement of rowmacc_wallclock.cpp (same seed, same data, same programs),
// and its four gas values must be equal to the ones printed there (656224, 685199, 535823, 260625).
// This program is run on the AVX2 build and on the scalar-fallback build. Gas values and the result lines of
// the bit-exact checks must be identical on the two builds. Timing is machine dependent.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>   // a valid (empty) host interface; no host call is made by this program
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
static evmc::MockedHost g_host;

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

struct Row { int I; long long gS, gB, gN, gN16, gH, gH16; double tS, tB, tN, tN16; };
static int runShape(const int NEU, const int I, Row& out) {
    const int G = I / 8, OFF = 32768, WIN = 32;
    rnd = 2026u;                                   // same data stream for every shape (I = 784 reproduces rowmacc_wallclock)
    std::vector<int16_t> W((size_t)NEU * I); std::vector<int> X(I);
    for (auto& w : W) w = (int16_t)((int)(rnd32() % 8192) - 4096);
    for (auto& x : X) x = (int)(rnd32() % 256);
    std::vector<uint32_t> ref(NEU);
    for (int n = 0; n < NEU; ++n) { int64_t s = 0; for (int i = 0; i < I; ++i) s += (int64_t)W[(size_t)n * I + i] * X[i]; ref[n] = (uint32_t)s; }
    // calldata: [XB: 98 words, 16-bit lanes] [XS: 98 words, 32-bit lanes reversed] [WB: 128*98 words] [WS: 128*98 words]
    const int G16 = I / 16;
    const size_t XB = 0, XS = XB + (size_t)G * 32, WB = XS + (size_t)G * 32, WS = WB + (size_t)NEU * G * 32, X16 = WS + (size_t)NEU * G * 32, W16 = X16 + (size_t)G16 * 32, CDS = W16 + (size_t)NEU * G16 * 32;
    std::vector<uint8_t> cd(CDS, 0);
    for (int g = 0; g < G16; ++g) {                                                   // 16-lane words for Path N16
        uint8_t* x16 = &cd[X16 + (size_t)g * 32];
        for (int j = 0; j < 16; ++j) { uint16_t u = (uint16_t)X[g * 16 + j]; x16[31 - 2 * j] = u & 0xFF; x16[30 - 2 * j] = (u >> 8) & 0xFF; }
        for (int n = 0; n < NEU; ++n) { uint8_t* w16 = &cd[W16 + ((size_t)n * G16 + g) * 32];
            for (int j = 0; j < 16; ++j) { uint16_t u = (uint16_t)W[(size_t)n * I + g * 16 + j]; w16[31 - 2 * j] = u & 0xFF; w16[30 - 2 * j] = (u >> 8) & 0xFF; } }
    }
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
    // Path N16: inputs at 0 (G16 words), weights at MW16, outputs at MO16 (its own, smaller memory map)
    const uint64_t MW16 = (uint64_t)G16 * 32, MO16 = MW16 + (uint64_t)NEU * G16 * 32, MEND16 = MO16 + (uint64_t)NEU * 32;
    Code kN16; copy(kN16, X16, MX, (uint64_t)G16 * 32); copy(kN16, W16, MW16, (uint64_t)NEU * G16 * 32);
    for (int n = 0; n < NEU; ++n) { kN16.P1(0); kN16.PUSH(MX); kN16.PUSH(MW16 + (uint64_t)n * G16 * 32); kN16.PUSH(G16); kN16.b(0xE7); kN16.b(0xCD); kN16.PUSH(MO16 + (uint64_t)n * 32); kN16.MSTORE(); }
    kN16.PUSH(MEND16); kN16.PUSH(0); kN16.b(0xF3);
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
    std::printf("---- I = %d ----\nBytecode: S=%zu B=%zu N=%zu N16=%zu bytes | calldata %zu bytes | %d neurons x %d inputs = %d multiply-accumulates\n", I, kS.c.size(), kB.c.size(), kN.c.size(), kN16.c.size(), CDS, NEU, I, NEU * I);

    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS; msg.input_data = cd.data(); msg.input_size = cd.size();
    int fail = 0;
    uint64_t MOcur = MO, MENDcur = MEND;   // output base of the program being run (N16 has its own map)
    auto runOnce = [&](const Code& k, bool check, int lanes /*1: XSUM32 word (uint32 at the low end), 0: S word*/, const char* name) -> long long {
        auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_SHANGHAI, &msg, k.c.data(), k.c.size());
        long long g = GAS - r.gas_left;
        if (check) {
            bool ok = r.status_code == EVMC_SUCCESS && r.output_data && r.output_size >= MENDcur;
            if (ok) for (int n = 0; n < NEU && ok; ++n) { const uint8_t* w = r.output_data + MOcur + (size_t)n * 32;
                uint32_t got = 0; for (int i = 28; i < 32; ++i) got = (got << 8) | w[i];        // low 32 bits of the word
                (void)lanes; if (got != ref[n]) ok = false; }
            std::printf("  [%s] bit-exact against the C++ reference (%d outputs): %s | gas %lld\n", name, NEU, ok ? "PASSED" : "FAILED", g);
            if (!ok) fail = 1;
        }
        if (r.release) r.release(&r);
        return g;
    };
    auto timePath = [&](const Code& k, int lanes, const char* name, double& med, long long& gas) {
        long long g1 = runOnce(k, true, lanes, name); gas = g1;
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
    // harness programs: the same two copies and the same return, no arithmetic (their gas is the calldata copy and
    // the memory expansion that every path pays; "compute gas" below is total gas minus this harness gas)
    Code kH; copy(kH, XB, MX, (uint64_t)G * 32); copy(kH, WB, MW, (uint64_t)NEU * G * 32); tail(kH);
    Code kH16; copy(kH16, X16, MX, (uint64_t)G16 * 32); copy(kH16, W16, MW16, (uint64_t)NEU * G16 * 32); kH16.PUSH(MEND16); kH16.PUSH(0); kH16.b(0xF3);
    { auto rh = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_SHANGHAI, &msg, kH.c.data(), kH.c.size());
      out.gH = GAS - rh.gas_left; bool okh = rh.status_code == EVMC_SUCCESS && rh.output_size == MEND; if (rh.release) rh.release(&rh);
      auto rh16 = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_SHANGHAI, &msg, kH16.c.data(), kH16.c.size());
      out.gH16 = GAS - rh16.gas_left; okh = okh && rh16.status_code == EVMC_SUCCESS && rh16.output_size == MEND16; if (rh16.release) rh16.release(&rh16);
      std::printf("  [H] harness (copies and return only): %s | gas %lld (8-lane map) | gas %lld (16-lane map)\n", okh ? "PASSED" : "FAILED", out.gH, out.gH16);
      if (!okh) fail = 1; }
    timePath(kS, 0, "S", out.tS, out.gS); timePath(kB, 1, "B", out.tB, out.gB); timePath(kN, 1, "N", out.tN, out.gN);
    MOcur = MO16; MENDcur = MEND16; timePath(kN16, 1, "N16", out.tN16, out.gN16); MOcur = MO; MENDcur = MEND;
    out.I = I;
    return fail;
}

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    const int NEU = 128;
    const int LEN[] = {16, 32, 64, 128, 256, 512, 784, 1024, 2048};
    const int NL = (int)(sizeof(LEN) / sizeof(LEN[0]));
    std::vector<Row> R(NL); int fail = 0;
    for (int i = 0; i < NL; ++i) fail |= runShape(NEU, LEN[i], R[i]);
    // the shape I = 784 must reproduce the gas of rowmacc_wallclock.cpp
    for (int i = 0; i < NL; ++i) if (R[i].I == 784) {
        bool ok = R[i].gS == 656224 && R[i].gB == 685199 && R[i].gN == 535823 && R[i].gN16 == 260625;
        std::printf("\n[CHECK] I = 784 reproduces the gas of rowmacc_wallclock (656224, 685199, 535823, 260625): %s\n", ok ? "PASSED" : "FAILED");
        if (!ok) fail = 1; }
    std::printf("\n================ ROW-LENGTH SCALING (%d neurons x I inputs) ================\n", NEU);
    std::printf("TOTAL GAS (identical on every build)\n");
    std::printf("     I |    mult-acc |        gas S |        gas B |        gas N |      gas N16 |  S/N |  B/N | S/N16\n");
    for (auto& r : R) std::printf("%6d | %11d | %12lld | %12lld | %12lld | %12lld | %4.2f | %4.2f | %5.2f\n", r.I, NEU * r.I, r.gS, r.gB, r.gN, r.gN16,
        (double)r.gS / r.gN, (double)r.gB / r.gN, (double)r.gS / r.gN16);
    std::printf("COMPUTE GAS = total gas minus harness gas (identical on every build)\n");
    std::printf("     I |  harness 8 | harness 16 |    compute S |    compute B |    compute N |  compute N16 |  S/N |  B/N | S/N16 | N gas/MAC | N16 gas/MAC\n");
    for (auto& r : R) { long long cS = r.gS - r.gH, cB = r.gB - r.gH, cN = r.gN - r.gH, cN16 = r.gN16 - r.gH16; double m = (double)NEU * r.I;
        std::printf("%6d | %10lld | %10lld | %12lld | %12lld | %12lld | %12lld | %4.2f | %4.2f | %5.2f | %9.3f | %11.3f\n", r.I, r.gH, r.gH16, cS, cB, cN, cN16,
            (double)cS / cN, (double)cB / cN, (double)cS / cN16, cN / m, cN16 / m); }
    std::printf("TIME (median, machine and build dependent)\n");
    std::printf("     I |     S ms |     B ms |     N ms |   N16 ms |   S/N |   B/N | S/N16 | ns/MAC S | ns/MAC B | ns/MAC N | ns/MAC N16\n");
    for (auto& r : R) { double m = (double)NEU * r.I;
        std::printf("%6d | %8.4f | %8.4f | %8.4f | %8.4f | %5.2f | %5.2f | %5.2f | %8.2f | %8.2f | %8.2f | %10.2f\n", r.I, r.tS * 1e3, r.tB * 1e3, r.tN * 1e3, r.tN16 * 1e3,
            r.tS / r.tN, r.tB / r.tN, r.tS / r.tN16, r.tS * 1e9 / m, r.tB * 1e9 / m, r.tN * 1e9 / m, r.tN16 * 1e9 / m); }
    std::printf("%s\n", fail ? "*** A CHECK FAILED -> the numbers are not valid ***" : "*** ALL CHECKS PASSED -> gas is valid; timing numbers are valid for this build ***");
    VM->destroy(VM); dlclose(h);
    return fail;
}
