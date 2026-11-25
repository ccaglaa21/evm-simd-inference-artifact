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
// and its four gas values must be equal to the ones printed there (656224, 836751, 552463, 278289).
// A fifth program, Path M, computes the same layer with the matrix-level instruction: the weights are a
// 128 x I matrix view and the input is a 1 x I view (32-byte MatrixHeader, 16-bit elements), and one XMATMUL
// writes the 128 outputs. Path M has its own memory map, as Path N16 has, because its data layout is denser.
// TIMING. The interpreter analyses the bytecode on every call (it copies the code and scans it for jump
// destinations), and the straight-line programs differ in size by two orders of magnitude, so the analysis
// would be counted against the long programs. Each program is therefore timed twice: as it is (total), and
// behind a STOP opcode, which is the analysis and the call setup without any execution. The copy-and-return
// program of each memory map is timed in the same way. The compute time of a path is
//     (total - analysis) of the path  minus  (total - analysis) of its copy-and-return program,
// the counterpart in time of the compute gas. The three kinds of time are printed in separate tables.
// With NO_TIMING=1 in the environment the programs are executed and checked but not timed, and the TIME tables
// are left out (used for the sanitizer builds and for emulated runs). The gas tables and check lines are the same.
// This program is run on the vector build and on the scalar-fallback build. Gas values and the result lines of
// the bit-exact checks must be identical on the two builds. Timing is machine dependent.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>   // a valid (empty) host interface; no host call is made by this program
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
static evmc::MockedHost g_host;

static evmc_vm* VM;
static bool NOTIME = false;                    // set by NO_TIMING=1: execute and check only
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

struct Tm { double tot = 0, base = 0; };   // total time and time behind a STOP (analysis and call setup)
struct Row { int I; long long gS, gB, gN, gN16, gM, gH, gH16, gHm; Tm tS, tB, tN, tN16, tM, tH, tH16, tHm; };
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
    const size_t XB = 0, XS = XB + (size_t)G * 32, WB = XS + (size_t)G * 32, WS = WB + (size_t)NEU * G * 32, X16 = WS + (size_t)NEU * G * 32, W16 = X16 + (size_t)G16 * 32, XM = W16 + (size_t)NEU * G16 * 32;
    const size_t XMB = (32 + 2 * (size_t)I + 31) / 32 * 32, WMB = (32 + 2 * (size_t)NEU * I + 31) / 32 * 32, WM = XM + XMB, CDS = WM + WMB;   // Path M: [header, x] [header, W], 16-bit little-endian elements
    std::vector<uint8_t> cd(CDS, 0);
    { auto p64 = [&](size_t at, uint64_t v) { for (int k = 0; k < 8; ++k) cd[at + k] = (uint8_t)(v >> (8 * k)); };
      p64(XM, 1); p64(XM + 8, (uint64_t)I); p64(XM + 16, (uint64_t)I); p64(XM + 24, 1);                       // x: 1 x I, unit stride
      p64(WM, (uint64_t)NEU); p64(WM + 8, (uint64_t)I); p64(WM + 16, (uint64_t)I); p64(WM + 24, 1);           // W: NEU x I, unit stride
      for (int i = 0; i < I; ++i) { uint16_t u = (uint16_t)X[i]; cd[XM + 32 + 2 * i] = u & 0xFF; cd[XM + 33 + 2 * i] = u >> 8; }
      for (size_t e = 0; e < (size_t)NEU * I; ++e) { uint16_t u = (uint16_t)W[e]; cd[WM + 32 + 2 * e] = u & 0xFF; cd[WM + 33 + 2 * e] = u >> 8; } }
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
    // Path M: x view at 0, W view at MWm, outputs at MOm (its own memory map); one XMATMUL(W, x) writes the NEU output words
    const uint64_t MWm = XMB, MOm = MWm + WMB, MENDm = MOm + (uint64_t)NEU * 32;
    Code kM; copy(kM, XM, MX, XMB); copy(kM, WM, MWm, WMB); kM.PUSH(MOm); kM.PUSH(MX); kM.PUSH(MWm); kM.b(0xE9); kM.PUSH(MENDm); kM.PUSH(0); kM.b(0xF3);
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
    std::printf("---- I = %d ----\nBytecode: S=%zu B=%zu N=%zu N16=%zu M=%zu bytes | calldata %zu bytes | %d neurons x %d inputs = %d multiply-accumulates\n", I, kS.c.size(), kB.c.size(), kN.c.size(), kN16.c.size(), kM.c.size(), CDS, NEU, I, NEU * I);

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
    // median time of one execution of a bytecode (at least 7 samples and one second, each sample at least 10 ms)
    auto timeCode = [&](const std::vector<uint8_t>& code, const char* name, const char* what) -> double {
        auto once = [&]() { auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_SHANGHAI, &msg, code.data(), code.size()); if (r.release) r.release(&r); };
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now(); once(); auto t1 = clk::now();
        double est = std::chrono::duration<double>(t1 - t0).count(); if (est <= 0) est = 1e-7;
        long long R = (long long)std::ceil(0.010 / est); if (R < 1) R = 1; if (R > 20000) R = 20000;
        std::vector<double> S; double tot = 0;
        while (true) { auto a = clk::now(); for (long long i = 0; i < R; ++i) once(); auto bb = clk::now();
            double dt = std::chrono::duration<double>(bb - a).count(); S.push_back(dt / R); tot += dt; size_t n = S.size();
            if (n >= 3 && ((n >= 7 && tot >= 1.0) || n >= 31 || tot >= 60.0)) break; }
        std::sort(S.begin(), S.end()); const double med = S[S.size() / 2];
        double mu = 0; for (double x : S) mu += x; mu /= S.size(); double sd = 0; for (double x : S) sd += (x - mu) * (x - mu); sd = std::sqrt(sd / S.size());
        std::printf("  [%s %s] med=%.7f s  min=%.7f  max=%.7f  CV=%.1f%%  (samples=%zu, inner-reps=%lld)\n", name, what, med, S.front(), S.back(), 100.0 * sd / mu, S.size(), R);
        std::fflush(stdout);
        return med;
    };
    auto timeBoth = [&](const Code& k, const char* name, Tm& t) {
        if (NOTIME) return;
        t.tot = timeCode(k.c, name, "total");
        std::vector<uint8_t> st; st.reserve(k.c.size() + 1); st.push_back(0x00); st.insert(st.end(), k.c.begin(), k.c.end());      // STOP, then the same bytes
        t.base = timeCode(st, name, "analysis");
    };
    auto timePath = [&](const Code& k, int lanes, const char* name, Tm& t, long long& gas) {
        long long g1 = runOnce(k, true, lanes, name); gas = g1;
        long long g2 = runOnce(k, false, lanes, name);
        if (g2 != g1) { std::printf("  [%s] repeated-run gas differs [FAILED]\n", name); fail = 1; }
        timeBoth(k, name, t);
    };
    // harness programs: the same two copies and the same return, no arithmetic (their gas is the calldata copy and
    // the memory expansion that every path pays; "compute gas" below is total gas minus this harness gas)
    Code kH; copy(kH, XB, MX, (uint64_t)G * 32); copy(kH, WB, MW, (uint64_t)NEU * G * 32); tail(kH);
    Code kH16; copy(kH16, X16, MX, (uint64_t)G16 * 32); copy(kH16, W16, MW16, (uint64_t)NEU * G16 * 32); kH16.PUSH(MEND16); kH16.PUSH(0); kH16.b(0xF3);
    Code kHm; copy(kHm, XM, MX, XMB); copy(kHm, WM, MWm, WMB); kHm.PUSH(MENDm); kHm.PUSH(0); kHm.b(0xF3);
    { auto rh = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_SHANGHAI, &msg, kH.c.data(), kH.c.size());
      out.gH = GAS - rh.gas_left; bool okh = rh.status_code == EVMC_SUCCESS && rh.output_size == MEND; if (rh.release) rh.release(&rh);
      auto rh16 = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_SHANGHAI, &msg, kH16.c.data(), kH16.c.size());
      out.gH16 = GAS - rh16.gas_left; okh = okh && rh16.status_code == EVMC_SUCCESS && rh16.output_size == MEND16; if (rh16.release) rh16.release(&rh16);
      auto rhm = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_SHANGHAI, &msg, kHm.c.data(), kHm.c.size());
      out.gHm = GAS - rhm.gas_left; okh = okh && rhm.status_code == EVMC_SUCCESS && rhm.output_size == MENDm; if (rhm.release) rhm.release(&rhm);
      std::printf("  [H] harness (copies and return only): %s | gas %lld (8-lane map) | gas %lld (16-lane map) | gas %lld (matrix-view map)\n", okh ? "PASSED" : "FAILED", out.gH, out.gH16, out.gHm);
      if (!okh) fail = 1; }
    timePath(kS, 0, "S", out.tS, out.gS); timePath(kB, 1, "B", out.tB, out.gB); timePath(kN, 1, "N", out.tN, out.gN);
    MOcur = MO16; MENDcur = MEND16; timePath(kN16, 1, "N16", out.tN16, out.gN16); MOcur = MO; MENDcur = MEND;
    MOcur = MOm; MENDcur = MENDm; timePath(kM, 1, "M", out.tM, out.gM); MOcur = MO; MENDcur = MEND;
    // static gas count of Path M: the instruction charges 270 per call and, per output, 5 plus 9 per group of eight elements
    { const long long words = (long long)(MENDm / 32); const long long expect = 3 * 3 + 3 + 3 * (long long)(XMB / 32) + 3 * 3 + 3 + 3 * (long long)(WMB / 32) + 3 * 3 + 270 + (long long)NEU * (5 + 9LL * ((I + 7) / 8)) + 3 * 2 + 3 * words + words * words / 512;
      std::printf("  [M] static gas count of the program (copies, one XMATMUL, memory): %lld | measured %lld: %s\n", expect, out.gM, expect == out.gM ? "equal" : "DIFFERENT"); if (expect != out.gM) fail = 1; }
    timeBoth(kH, "H", out.tH); timeBoth(kH16, "H16", out.tH16); timeBoth(kHm, "Hm", out.tHm);
    out.I = I;
    return fail;
}

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    NOTIME = std::getenv("NO_TIMING") != nullptr;
    const int NEU = 128;
    const int LEN[] = {16, 32, 64, 128, 256, 512, 784, 1024, 2048};
    const int NL = (int)(sizeof(LEN) / sizeof(LEN[0]));
    std::vector<Row> R(NL); int fail = 0;
    for (int i = 0; i < NL; ++i) fail |= runShape(NEU, LEN[i], R[i]);
    // the shape I = 784 must reproduce the gas of rowmacc_wallclock.cpp
    for (int i = 0; i < NL; ++i) if (R[i].I == 784) {
        bool ok = R[i].gS == 656224 && R[i].gB == 836751 && R[i].gN == 552463 && R[i].gN16 == 278289;
        std::printf("\n[CHECK] I = 784 reproduces the gas of rowmacc_wallclock (656224, 836751, 552463, 278289): %s\n", ok ? "PASSED" : "FAILED");
        if (!ok) fail = 1; }
    std::printf("\n================ ROW-LENGTH SCALING (%d neurons x I inputs) ================\n", NEU);
    std::printf("TOTAL GAS (identical on every build)\n");
    std::printf("     I |    mult-acc |        gas S |        gas B |        gas N |      gas N16 |        gas M |  S/N |  B/N | S/N16 |   S/M\n");
    for (auto& r : R) std::printf("%6d | %11d | %12lld | %12lld | %12lld | %12lld | %12lld | %4.2f | %4.2f | %5.2f | %5.2f\n", r.I, NEU * r.I, r.gS, r.gB, r.gN, r.gN16, r.gM,
        (double)r.gS / r.gN, (double)r.gB / r.gN, (double)r.gS / r.gN16, (double)r.gS / r.gM);
    std::printf("COMPUTE GAS = total gas minus harness gas (identical on every build)\n");
    std::printf("     I |  harness 8 | harness 16 |  harness M |    compute S |    compute B |    compute N |  compute N16 |    compute M |  S/N |  B/N | S/N16 |   S/M |   N/M | N gas/MAC | N16 gas/MAC | M gas/MAC\n");
    for (auto& r : R) { long long cS = r.gS - r.gH, cB = r.gB - r.gH, cN = r.gN - r.gH, cN16 = r.gN16 - r.gH16, cM = r.gM - r.gHm; double m = (double)NEU * r.I;
        std::printf("%6d | %10lld | %10lld | %10lld | %12lld | %12lld | %12lld | %12lld | %12lld | %4.2f | %4.2f | %5.2f | %5.2f | %5.2f | %9.3f | %11.3f | %9.3f\n", r.I, r.gH, r.gH16, r.gHm, cS, cB, cN, cN16, cM,
            (double)cS / cN, (double)cB / cN, (double)cS / cN16, (double)cS / cM, (double)cN / cM, cN / m, cN16 / m, cM / m); }
    // times below are machine and build dependent; every table from here on starts with the word TIME
    auto comp = [](const Tm& p, const Tm& h) { return (p.tot - p.base) - (h.tot - h.base); };
    if (NOTIME) std::printf("TIME tables are left out (NO_TIMING is set)\n");
    else {
    std::printf("TIME, TOTAL of one call in ms (median; bytecode analysis, copies, computation and return)\n");
    std::printf("     I |        S |        B |        N |      N16 |        M | copy-return 8 | copy-return 16 | copy-return M\n");
    for (auto& r : R) std::printf("%6d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %13.4f | %14.4f | %13.4f\n", r.I, r.tS.tot * 1e3, r.tB.tot * 1e3, r.tN.tot * 1e3, r.tN16.tot * 1e3, r.tM.tot * 1e3, r.tH.tot * 1e3, r.tH16.tot * 1e3, r.tHm.tot * 1e3);
    std::printf("TIME, ANALYSIS in ms (median; the same bytecode behind a STOP: code analysis and call setup, nothing executed) and its share of the total\n");
    std::printf("     I |        S |        B |        N |      N16 |        M | share S | share B | share N | share N16 | share M\n");
    for (auto& r : R) std::printf("%6d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %6.1f%% | %6.1f%% | %6.1f%% | %8.1f%% | %6.1f%%\n", r.I, r.tS.base * 1e3, r.tB.base * 1e3, r.tN.base * 1e3, r.tN16.base * 1e3, r.tM.base * 1e3,
        100 * r.tS.base / r.tS.tot, 100 * r.tB.base / r.tB.tot, 100 * r.tN.base / r.tN.tot, 100 * r.tN16.base / r.tN16.tot, 100 * r.tM.base / r.tM.tot);
    std::printf("TIME, COMPUTE in ms = (total - analysis) of the path minus (total - analysis) of its copy-and-return program\n");
    std::printf("     I |        S |        B |        N |      N16 |        M |   S/N |   B/N | S/N16 |   S/M |   N/M | ns/MAC S | ns/MAC B | ns/MAC N | ns/MAC N16 | ns/MAC M | ns/gas S | ns/gas B | ns/gas N | ns/gas N16 | ns/gas M\n");
    for (auto& r : R) { double m = (double)NEU * r.I; double cS = comp(r.tS, r.tH), cB = comp(r.tB, r.tH), cN = comp(r.tN, r.tH), cN16 = comp(r.tN16, r.tH16), cM = comp(r.tM, r.tHm);
        long long gS = r.gS - r.gH, gB = r.gB - r.gH, gN = r.gN - r.gH, gN16 = r.gN16 - r.gH16, gM = r.gM - r.gHm;
        std::printf("%6d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %5.2f | %5.2f | %5.2f | %5.2f | %5.2f | %8.3f | %8.3f | %8.3f | %10.3f | %8.3f | %8.3f | %8.3f | %8.3f | %10.3f | %8.3f\n", r.I, cS * 1e3, cB * 1e3, cN * 1e3, cN16 * 1e3, cM * 1e3,
            cS / cN, cB / cN, cS / cN16, cS / cM, cN / cM, cS * 1e9 / m, cB * 1e9 / m, cN * 1e9 / m, cN16 * 1e9 / m, cM * 1e9 / m, cS * 1e9 / gS, cB * 1e9 / gB, cN * 1e9 / gN, cN16 * 1e9 / gN16, cM * 1e9 / gM); }
    }
    std::printf("\n%s\n", fail ? "*** A CHECK FAILED -> the numbers are not valid ***" : "*** ALL CHECKS PASSED -> gas is valid; timing numbers are valid for this build ***");
    VM->destroy(VM); dlclose(h);
    return fail;
}
