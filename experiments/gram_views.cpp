// gram_views.cpp — GRAM MATRIX G = A^T A of a run-time matrix: view path, copy path, packed scalar path.
//
// A is an R x C matrix of 16-bit integers (R samples, C features) that reaches the contract as call data.
// Entry G[i][j] is the dot product of columns i and j of A, so the kernel needs column access to a matrix
// that is stored row by row. With centred data, G divided by R - 1 is the covariance matrix. Three
// straight-line programs compute the same C x C result, one 256-bit word per entry:
//   Path V (view)   XTRANSPOSE turns the header of A into a transposed view (no element is moved). Every
//                   entry is one XMACC_V over rows i and j of the view, which reads the stored columns
//                   with the stride of the header (strided price, 14 gas per group of eight).
//   Path K (copy)   The transpose is materialised once with one MCOPY per element (standard instruction),
//                   a header is written for the copy, and every entry is one XMACC_V over two unit-stride
//                   rows of the copy (unit price, 11 gas per group of eight).
//   Path S (scalar) Standard instructions only, in the row-oriented form that needs no column access:
//                   G[i][:] = sum over t of A[t][i] * A[t][:]. A row is packed as eight 32-bit lanes per
//                   word with an offset of 2^11, one 256-bit MUL multiplies eight elements by the scalar,
//                   and the offset is removed with the column sums. The input reaches this path in its own
//                   packed layout, in the same row order.
// Elements lie in [-2048, 2047] and R <= 256, so every lane of Path S stays below 2^32 and no saturating
// addition of the instructions is reached; the three programs then return the same values.
// CHECKS per shape: (1) every program equals the C++ reference in all C*C entries (2) measured gas equals
// the static count of the generator plus the memory expansion (3) a repeated run gives the same gas.
// Gas is identical on every build. Timing depends on the machine and the build.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>
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
    std::vector<uint8_t> c; long long sgas = 0;
    void b(uint8_t x, long long g) { c.push_back(x); sgas += g; }
    void PUSH(uint64_t v) { int n = 0; for (uint64_t t = v; t; t >>= 8) ++n; if (n == 0) n = 1;
        b((uint8_t)(0x5F + n), 3); for (int k = n - 1; k >= 0; --k) c.push_back((uint8_t)((v >> (8 * k)) & 0xFF)); }
    void W32(const uint8_t* w) { b(0x7F, 3); for (int i = 0; i < 32; ++i) c.push_back(w[i]); }
    void MLOAD() { b(0x51, 3); } void MSTORE() { b(0x52, 3); } void MUL() { b(0x02, 5); } void ADD() { b(0x01, 3); } void SUB() { b(0x03, 3); }
    void AND() { b(0x16, 3); } void SHR() { b(0x1C, 3); } void SHL() { b(0x1B, 3); } void DUP(int n) { b((uint8_t)(0x7F + n), 3); } void SWAP(int n) { b((uint8_t)(0x8F + n), 3); } void POP() { b(0x50, 2); }
    void CDCOPY(uint64_t cdoff, uint64_t mem, uint64_t bytes) { PUSH(bytes); PUSH(cdoff); PUSH(mem); b(0x37, 3 + 3 * (long long)((bytes + 31) / 32)); }
    void MCOPY2(uint64_t src, uint64_t dst) { PUSH(2); PUSH(src); PUSH(dst); b(0x5E, 3 + 3); }   // two bytes: one word of copy cost
    void RET(uint64_t off, uint64_t size) { PUSH(size); PUSH(off); b(0xF3, 0); }
};
static long long memgas(uint64_t bytes) { long long w = (long long)((bytes + 31) / 32); return 3 * w + w * w / 512; }
static uint32_t rnd = 2026u; static uint32_t rnd32() { rnd = rnd * 1664525u + 1013904223u; return rnd; }
static uint64_t pad32(uint64_t x) { return (x + 31) / 32 * 32; }
static void put64(uint8_t* m, uint64_t v) { for (int k = 0; k < 8; ++k) m[k] = (uint8_t)(v >> (8 * k)); }

struct Row { int R, C; long long gV, gK, gS, gHh, gHs, grp; double tV, tK, tS; };
static int runShape(const int R, const int C, Row& out) {
    const int Cw = C / 8, B = 2048; const uint64_t G8 = (uint64_t)(R + 7) / 8;
    rnd = 2026u + (uint32_t)(R * 1000 + C);
    std::vector<int> A((size_t)R * C); for (auto& a : A) a = (int)(rnd32() % 4096) - 2048;
    std::vector<uint32_t> ref((size_t)C * C);
    for (int i = 0; i < C; ++i) for (int j = 0; j < C; ++j) { int64_t s = 0; for (int t = 0; t < R; ++t) s += (int64_t)A[(size_t)t * C + i] * A[(size_t)t * C + j]; ref[(size_t)i * C + j] = (uint32_t)s; }
    // call data: [header layout of A, padded to words] [packed layout of A for Path S]
    const uint64_t HB = pad32(32 + 2 * (uint64_t)R * C), PB = (uint64_t)R * Cw * 32;
    std::vector<uint8_t> cd(HB + PB, 0);
    put64(&cd[0], (uint64_t)R); put64(&cd[8], (uint64_t)C); put64(&cd[16], (uint64_t)C); put64(&cd[24], 1);
    for (size_t e = 0; e < (size_t)R * C; ++e) { uint16_t u = (uint16_t)(int16_t)A[e]; cd[32 + 2 * e] = u & 0xFF; cd[33 + 2 * e] = u >> 8; }
    for (int t = 0; t < R; ++t) for (int g = 0; g < Cw; ++g) { uint8_t* w = &cd[HB + ((size_t)t * Cw + g) * 32];
        for (int j = 0; j < 8; ++j) { uint32_t v = (uint32_t)(A[(size_t)t * C + 8 * g + j] + B); int bit = 32 * j; for (int k = 0; k < 4; ++k) w[31 - bit / 8 - k] = (uint8_t)(v >> (8 * k)); } }
    const uint64_t OUTB = (uint64_t)C * C * 32;
    // ---- Paths V and K: A at 0, outputs at MO, transposed copy (Path K only) at MT ----
    const uint64_t MA = 0, MO = HB, MT = MO + OUTB, MENDV = MO + OUTB, MENDK = MT + HB;
    auto entries = [&](Code& k, uint64_t base) { for (int i = 0; i < C; ++i) for (int j = 0; j < C; ++j) {
        k.PUSH(0); k.PUSH((uint64_t)j); k.PUSH(base); k.PUSH((uint64_t)i); k.PUSH(base); k.b(0xE8, 0); k.b(0xCD, 6); k.PUSH(MO + ((uint64_t)i * C + j) * 32); k.MSTORE(); } };
    Code kV; kV.CDCOPY(0, MA, HB); kV.PUSH(MA); kV.b(0xDD, 10); kV.POP(); entries(kV, MA); kV.RET(MO, OUTB);
    kV.sgas += (long long)C * C * (long long)G8 * 14;
    Code kK; kK.CDCOPY(0, MA, HB);
    for (int c = 0; c < C; ++c) for (int t = 0; t < R; ++t) kK.MCOPY2(MA + 32 + 2 * ((uint64_t)t * C + c), MT + 32 + 2 * ((uint64_t)c * R + t));
    { uint8_t hw[32]; put64(hw, (uint64_t)C); put64(hw + 8, (uint64_t)R); put64(hw + 16, (uint64_t)R); put64(hw + 24, 1); kK.W32(hw); kK.PUSH(MT); kK.MSTORE(); }
    entries(kK, MT); kK.RET(MO, OUTB);
    kK.sgas += (long long)C * C * (long long)G8 * 11;
    Code kHh; kHh.CDCOPY(0, MA, HB); kHh.RET(MO, OUTB);
    // ---- Path S: packed rows at 0, outputs at SO, column sums at SCS, offset terms at SBC, scalar column at SSC ----
    const uint64_t SP = 0, SO = PB, SCS = SO + OUTB, SBC = SCS + (uint64_t)Cw * 32, SSC = SBC + (uint64_t)C * 32, SEND = SSC + (uint64_t)R * 32;
    Code kS; kS.CDCOPY(HB, SP, PB);
    for (int g = 0; g < Cw; ++g) { kS.PUSH(0); for (int t = 0; t < R; ++t) { kS.PUSH(SP + ((uint64_t)t * Cw + g) * 32); kS.MLOAD(); kS.ADD(); } kS.PUSH(SCS + (uint64_t)g * 32); kS.MSTORE(); }
    auto lane = [&](Code& k, int j) { if (j) { k.PUSH((uint64_t)32 * j); k.SHR(); } k.PUSH(0xFFFFFFFFULL); k.AND(); };
    for (int j = 0; j < C; ++j) { kS.PUSH(SCS + (uint64_t)(j / 8) * 32); kS.MLOAD(); lane(kS, j % 8); kS.PUSH(11); kS.SHL(); kS.PUSH(SBC + (uint64_t)j * 32); kS.MSTORE(); }
    for (int i = 0; i < C; ++i) {
        for (int t = 0; t < R; ++t) { kS.PUSH(SP + ((uint64_t)t * Cw + i / 8) * 32); kS.MLOAD(); lane(kS, i % 8); kS.PUSH(SSC + (uint64_t)t * 32); kS.MSTORE(); }
        kS.PUSH(SBC + (uint64_t)i * 32); kS.MLOAD(); kS.PUSH((uint64_t)R * B * B); kS.SUB();          // K_i = R b^2 - b cs'_i
        for (int g = 0; g < Cw; ++g) { kS.PUSH(0);
            for (int t = 0; t < R; ++t) { kS.PUSH(SSC + (uint64_t)t * 32); kS.MLOAD(); kS.PUSH(SP + ((uint64_t)t * Cw + g) * 32); kS.MLOAD(); kS.MUL(); kS.ADD(); }
            for (int j = 0; j < 8; ++j) { kS.DUP(1); lane(kS, j); kS.DUP(3); kS.ADD(); kS.PUSH(SBC + (uint64_t)(8 * g + j) * 32); kS.MLOAD(); kS.SWAP(1); kS.SUB();
                kS.PUSH(SO + ((uint64_t)i * C + 8 * g + j) * 32); kS.MSTORE(); }
            kS.POP(); }
        kS.POP(); }
    kS.RET(SO, OUTB);
    Code kHs; kHs.CDCOPY(HB, SP, PB); kHs.RET(SO, OUTB);
    std::printf("---- R = %d samples, C = %d features ----\nBytecode: V=%zu K=%zu S=%zu bytes | %d entries x %d elements = %lld multiply-accumulates\n", R, C, kV.c.size(), kK.c.size(), kS.c.size(), C * C, R, (long long)C * C * R);

    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS; msg.input_data = cd.data(); msg.input_size = cd.size();
    int fail = 0;
    auto runOnce = [&](const Code& k, bool check, uint64_t memend, const char* name) -> long long {
        auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, k.c.data(), k.c.size());
        long long g = GAS - r.gas_left;
        if (check) {
            bool ok = r.status_code == EVMC_SUCCESS && r.output_data && r.output_size == OUTB;
            if (ok) for (size_t n = 0; n < (size_t)C * C && ok; ++n) { const uint8_t* w = r.output_data + n * 32; uint32_t got = 0; for (int i = 28; i < 32; ++i) got = (got << 8) | w[i]; if (got != ref[n]) ok = false; }
            long long pred = k.sgas + memgas(memend);
            std::printf("  [%s] equals the C++ reference (%d entries): %s | gas %lld | static count + memory %lld: %s\n", name, C * C, ok ? "PASSED" : "FAILED", g, pred, g == pred ? "equal" : "DIFFERENT");
            if (!ok || g != pred) fail = 1;
        }
        if (r.release) r.release(&r);
        return g;
    };
    auto timePath = [&](const Code& k, uint64_t memend, const char* name, double& med, long long& gas) {
        long long g1 = runOnce(k, true, memend, name); gas = g1;
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now(); long long g2 = runOnce(k, false, memend, name); auto t1 = clk::now();
        if (g2 != g1) { std::printf("  [%s] repeated-run gas differs [FAILED]\n", name); fail = 1; }
        double est = std::chrono::duration<double>(t1 - t0).count(); if (est <= 0) est = 1e-7;
        long long Rp = (long long)std::ceil(0.010 / est); if (Rp < 1) Rp = 1; if (Rp > 20000) Rp = 20000;
        std::vector<double> S; double tot = 0;
        while (true) { auto a = clk::now(); for (long long i = 0; i < Rp; ++i) runOnce(k, false, memend, name); auto bb = clk::now();
            double dt = std::chrono::duration<double>(bb - a).count(); S.push_back(dt / Rp); tot += dt; size_t n = S.size();
            if (n >= 3 && ((n >= 7 && tot >= 1.0) || n >= 31 || tot >= 60.0)) break; }
        std::sort(S.begin(), S.end()); med = S[S.size() / 2];
        double mu = 0; for (double x : S) mu += x; mu /= S.size(); double sd = 0; for (double x : S) sd += (x - mu) * (x - mu); sd = std::sqrt(sd / S.size());
        std::printf("  [%s] med=%.6f s  min=%.6f  max=%.6f  CV=%.1f%%  (samples=%zu, inner-reps=%lld)\n", name, med, S.front(), S.back(), 100.0 * sd / mu, S.size(), Rp);
        std::fflush(stdout);
    };
    { auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, kHh.c.data(), kHh.c.size()); out.gHh = GAS - r.gas_left; bool ok = r.status_code == EVMC_SUCCESS && out.gHh == kHh.sgas + memgas(MENDV); if (r.release) r.release(&r);
      auto r2 = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, kHs.c.data(), kHs.c.size()); out.gHs = GAS - r2.gas_left; ok = ok && r2.status_code == EVMC_SUCCESS && out.gHs == kHs.sgas + memgas(SO + OUTB); if (r2.release) r2.release(&r2);
      std::printf("  [H] harness (call-data copy and return only): %s | gas %lld (header layout) | gas %lld (packed layout)\n", ok ? "PASSED" : "FAILED", out.gHh, out.gHs); if (!ok) fail = 1; }
    timePath(kV, MENDV, "V", out.tV, out.gV); timePath(kK, MENDK, "K", out.tK, out.gK); timePath(kS, SEND, "S", out.tS, out.gS);
    out.R = R; out.C = C; out.grp = (long long)C * C * (long long)G8;
    return fail;
}

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    const int SH[][2] = {{128, 8}, {128, 16}, {128, 24}, {128, 32}, {128, 40}, {128, 48}, {128, 64}, {16, 32}, {32, 32}, {64, 32}, {256, 32}};
    const int NS = (int)(sizeof(SH) / sizeof(SH[0]));
    std::vector<Row> T(NS); int fail = 0;
    for (int i = 0; i < NS; ++i) fail |= runShape(SH[i][0], SH[i][1], T[i]);
    std::printf("\n================ GRAM MATRIX G = A^T A (R samples x C features) ================\n");
    std::printf("COMPUTE GAS = total gas minus harness gas (identical on every build)\n");
    std::printf("    R |   C |   mult-acc |    compute V |    compute K |    compute S |  S/V |  S/K |  K/V | cheaper | V gas/MAC | K gas/MAC | S gas/MAC\n");
    for (auto& r : T) { long long cV = r.gV - r.gHh, cK = r.gK - r.gHh, cS = r.gS - r.gHs; double m = (double)r.C * r.C * r.R;
        std::printf("%5d | %3d | %10.0f | %12lld | %12lld | %12lld | %4.2f | %4.2f | %4.2f | %7s | %9.3f | %9.3f | %9.3f\n", r.R, r.C, m, cV, cK, cS, (double)cS / cV, (double)cS / cK, (double)cK / cV, cV <= cK ? "V" : "K", cV / m, cK / m, cS / m); }
    std::printf("TIME (median, machine and build dependent)\n");
    std::printf("    R |   C |     V ms |     K ms |     S ms |   S/V |   S/K |   K/V | faster | ns/MAC V | ns/MAC K | ns/MAC S | time-neutral strided price | time-neutral unit price\n");
    for (auto& r : T) { double m = (double)r.C * r.C * r.R; long long cV = r.gV - r.gHh, cK = r.gK - r.gHh, cS = r.gS - r.gHs;
        // price per group at which the path has the nanoseconds per gas of Path S: (t_path / t_S) * computeS = compute_path - groups*price_now + groups*p
        double pv = ((r.tV / r.tS) * cS - (cV - 14.0 * r.grp)) / r.grp, pk = ((r.tK / r.tS) * cS - (cK - 11.0 * r.grp)) / r.grp;
        std::printf("%5d | %3d | %8.4f | %8.4f | %8.4f | %5.2f | %5.2f | %5.2f | %6s | %8.2f | %8.2f | %8.2f | %26.2f | %23.2f\n", r.R, r.C, r.tV * 1e3, r.tK * 1e3, r.tS * 1e3, r.tS / r.tV, r.tS / r.tK, r.tK / r.tV, r.tV <= r.tK ? "V" : "K",
            r.tV * 1e9 / m, r.tK * 1e9 / m, r.tS * 1e9 / m, pv, pk); }
    std::printf("%s\n", fail ? "*** A CHECK FAILED -> the numbers are not valid ***" : "*** ALL CHECKS PASSED -> gas is valid; timing numbers are valid for this build ***");
    VM->destroy(VM); dlclose(h);
    return fail;
}
