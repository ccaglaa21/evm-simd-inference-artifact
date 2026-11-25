// gram_views.cpp — GRAM MATRIX G = A^T A of a run-time matrix: row-level and matrix-level view instructions
// against a packed scalar program.
//
// A is an R x C matrix of 16-bit integers (R samples, C features) that reaches the contract as call data.
// Entry G[i][j] is the dot product of columns i and j of A, so the kernel needs column access to a matrix
// that is stored row by row. With centred data, G divided by R - 1 is the covariance matrix. Five
// straight-line programs compute the same C x C result, one 256-bit word per entry:
//   Path V  (row level, view)     XTRANSPOSE turns the header of A into a transposed view (no element is
//                                 moved). Every entry is one XMACC_V over rows i and j of the view, which
//                                 reads the stored columns with the stride of the header.
//   Path K  (row level, copy)     The transpose is materialised once with one MCOPY per element (standard
//                                 instruction), a header is written for the copy, and every entry is one
//                                 XMACC_V over two unit-stride rows of the copy.
//   Path MV (matrix level, view)  XTRANSPOSE, then one XMATMUL with the view as both operands. The
//                                 instruction packs the strided rows of the view once.
//   Path MK (matrix level, copy)  the copy of Path K, then one XMATMUL on the copy.
//   Path S  (scalar)              Standard instructions only, in the row-oriented form that needs no column
//                                 access: G[i][:] = sum over t of A[t][i] * A[t][:]. A row is packed as
//                                 eight 32-bit lanes per word, one 256-bit MUL multiplies eight elements by
//                                 the scalar. The input reaches this path in its own packed layout, in the
//                                 same row order.
// GAS OF THE VIEW INSTRUCTIONS (from the measured times of four builds, training/price_calibration.py):
// XMACC_V 50 per call, 10 per group of eight elements and 4 per strided element on every call. XMATMUL 270
// per call, 5 per entry, 9 per entry and group of eight elements and 6 per packed element.
// DATA. (a) Pseudo-random signed elements in [-2048, 2047] with R <= 256: Path S stores them with an offset
// of 2^11 and removes the offset with the column sums. (b) Real image data: A holds R images of MNIST or
// Fashion-MNIST (the files of the network experiments), one image per row, and the C features are the pixels
// of the central w x w window of the image (w = 8 or 16, values 0 to 255). Pixels are not negative, so Path S
// needs no offset and its correction steps are left out. In both cases every lane of Path S stays below 2^32
// and no saturating addition of the instructions is reached, so the five programs return the same values.
// No program uses the symmetry of G: all C * C entries are computed by every path.
// SCALE. The feature sweep goes up to C = 256 (8.4 million multiply-accumulates), the sample sweep up to
// R = 500 images. For R = 128 the program also finds, for every path, the largest C (a multiple of 8) whose
// gas stays within the per-transaction cap of EIP-7825 (2^24 gas), once for the execution gas alone and once
// for the whole transaction, which adds the intrinsic gas: 21000, 16 per non-zero byte and 4 per zero byte of
// the call data that the path receives (the header layout for the view paths, the packed layout for Path S).
// The size is found with the static gas count of the generator and is then confirmed by executing that shape
// and the next one.
// CHECKS per shape: (1) every program equals the C++ reference in all C*C entries (2) measured gas equals
// the static count of the generator plus the memory expansion (3) a repeated run gives the same gas.
// TIMING. The interpreter analyses the bytecode on every call (it copies the code and scans it for jump
// destinations), and the straight-line programs differ in size by several orders of magnitude, so the
// analysis would be counted against the long programs. Each program is therefore timed twice: as it is
// (total), and behind a STOP opcode, which is the analysis and the call setup without any execution. The
// copy-and-return program of each data layout is timed in the same way. The compute time of a path is
//     (total - analysis) of the path  minus  (total - analysis) of its copy-and-return program,
// the counterpart in time of the compute gas. The three kinds of time are printed in separate tables.
// With NO_TIMING=1 in the environment the programs are executed and checked but not timed, and the TIME tables
// are left out (used for the sanitizer builds and for emulated runs). The gas tables and check lines are the same.
// Gas is identical on every build. Timing depends on the machine and the build.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
static evmc::MockedHost g_host;
static evmc_vm* VM;
static bool NOTIME = false;                    // set by NO_TIMING=1: execute and check only
static const int64_t GAS = 200000000000LL;
static const long long V_CALL = 50, V_GROUP = 10, V_STRIDED = 4;                 // XMACC_V; must equal the values in the interpreter
static const long long M_CALL = 270, M_ENTRY = 5, M_GROUP = 9, M_PACKED = 6;     // XMATMUL
static const long long P_XSUM32 = 14, P_XTRANSPOSE = 22;
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

enum { PV = 0, PK = 1, PMV = 2, PMK = 3, PS = 4, NP = 5 };
static const char* PNAME[NP] = {"V", "K", "MV", "MK", "S"};
struct Tm { double tot = 0, base = 0; };              // total time and time behind a STOP (analysis and call setup)
struct Row { int R, C; long long g[NP], gHh, gHs; Tm t[NP], tHh, tHs; long long trace; const char* name; };
struct Prog { Code k[NP], kHh, kHs; uint64_t mend[NP], HEND, HSEND, OUTB; std::vector<uint8_t> cd; std::vector<uint32_t> ref; };
// img: null for pseudo-random signed elements (offset 2^11 in Path S), otherwise the pixel rows of the images
// (784 values each, no offset). withData = false builds the programs only (for the static gas count).
static void build(const int R, const int C, const std::vector<std::vector<int>>* img, bool withData, Prog& p) {
    const int Cw = C / 8, B = img ? 0 : 2048; const long long G8 = (R + 7) / 8;
    const uint64_t HB = pad32(32 + 2 * (uint64_t)R * C), PB = (uint64_t)R * Cw * 32, OUTB = (uint64_t)C * C * 32;
    p.OUTB = OUTB;
    if (withData) {
        rnd = 2026u + (uint32_t)(R * 1000 + C);
        std::vector<int> A((size_t)R * C);
        if (!img) for (auto& a : A) a = (int)(rnd32() % 4096) - 2048;
        else { int w = 1; while (w * w < C) ++w; const int r0 = (28 - w) / 2;      // central w x w window
            for (int t = 0; t < R; ++t) for (int c = 0; c < C; ++c) A[(size_t)t * C + c] = (*img)[t][(r0 + c / w) * 28 + r0 + c % w]; }
        p.ref.assign((size_t)C * C, 0);
        for (int i = 0; i < C; ++i) for (int j = i; j < C; ++j) { int64_t s = 0; for (int t = 0; t < R; ++t) s += (int64_t)A[(size_t)t * C + i] * A[(size_t)t * C + j]; p.ref[(size_t)i * C + j] = p.ref[(size_t)j * C + i] = (uint32_t)s; }
        // call data: [header layout of A, padded to words] [packed layout of A for Path S]
        p.cd.assign(HB + PB, 0); auto& cd = p.cd;
        put64(&cd[0], (uint64_t)R); put64(&cd[8], (uint64_t)C); put64(&cd[16], (uint64_t)C); put64(&cd[24], 1);
        for (size_t e = 0; e < (size_t)R * C; ++e) { uint16_t u = (uint16_t)(int16_t)A[e]; cd[32 + 2 * e] = u & 0xFF; cd[33 + 2 * e] = u >> 8; }
        for (int t = 0; t < R; ++t) for (int g = 0; g < Cw; ++g) { uint8_t* w = &cd[HB + ((size_t)t * Cw + g) * 32];
            for (int j = 0; j < 8; ++j) { uint32_t v = (uint32_t)(A[(size_t)t * C + 8 * g + j] + B); int bit = 32 * j; for (int k = 0; k < 4; ++k) w[31 - bit / 8 - k] = (uint8_t)(v >> (8 * k)); } }
    }
    // ---- header layout: A at 0, outputs at MO, transposed copy (Paths K and MK) at MT ----
    const uint64_t MA = 0, MO = HB, MT = MO + OUTB; p.mend[PV] = p.mend[PMV] = MO + OUTB; p.mend[PK] = p.mend[PMK] = MT + HB; p.HEND = MO + OUTB;
    auto entries = [&](Code& k, uint64_t base) { for (int i = 0; i < C; ++i) for (int j = 0; j < C; ++j) {
        k.PUSH(0); k.PUSH((uint64_t)j); k.PUSH(base); k.PUSH((uint64_t)i); k.PUSH(base); k.b(0xEA, 0); k.b(0xCD, P_XSUM32); k.PUSH(MO + ((uint64_t)i * C + j) * 32); k.MSTORE(); } };
    auto copyT = [&](Code& k) { for (int c = 0; c < C; ++c) for (int t = 0; t < R; ++t) k.MCOPY2(MA + 32 + 2 * ((uint64_t)t * C + c), MT + 32 + 2 * ((uint64_t)c * R + t));
        uint8_t hw[32]; put64(hw, (uint64_t)C); put64(hw + 8, (uint64_t)R); put64(hw + 16, (uint64_t)R); put64(hw + 24, 1); k.W32(hw); k.PUSH(MT); k.MSTORE(); };
    const long long E = (long long)C * C;
    Code& kV = p.k[PV]; kV.CDCOPY(0, MA, HB); kV.PUSH(MA); kV.b(0xDD, P_XTRANSPOSE); kV.POP(); entries(kV, MA); kV.RET(MO, OUTB);
    kV.sgas += E * (V_CALL + G8 * V_GROUP + 2 * (long long)R * V_STRIDED);                       // two strided rows of R elements per call
    Code& kK = p.k[PK]; kK.CDCOPY(0, MA, HB); copyT(kK); entries(kK, MT); kK.RET(MO, OUTB);
    kK.sgas += E * (V_CALL + G8 * V_GROUP);
    Code& kMV = p.k[PMV]; kMV.CDCOPY(0, MA, HB); kMV.PUSH(MA); kMV.b(0xDD, P_XTRANSPOSE); kMV.POP(); kMV.PUSH(MO); kMV.PUSH(MA); kMV.PUSH(MA); kMV.b(0xE9, 0); kMV.RET(MO, OUTB);
    kMV.sgas += M_CALL + E * (M_ENTRY + G8 * M_GROUP) + (long long)C * R * M_PACKED;            // one view as both operands: its C rows of R elements are packed once
    Code& kMK = p.k[PMK]; kMK.CDCOPY(0, MA, HB); copyT(kMK); kMK.PUSH(MO); kMK.PUSH(MT); kMK.PUSH(MT); kMK.b(0xE9, 0); kMK.RET(MO, OUTB);
    kMK.sgas += M_CALL + E * (M_ENTRY + G8 * M_GROUP);
    p.kHh.CDCOPY(0, MA, HB); p.kHh.RET(MO, OUTB);
    // ---- Path S: packed rows at 0, outputs at SO, column sums at SCS, offset terms at SBC, scalar column at SSC ----
    const uint64_t SP = 0, SO = PB, SCS = SO + OUTB, SBC = SCS + (uint64_t)Cw * 32, SSC = B ? SBC + (uint64_t)C * 32 : SO + OUTB; p.mend[PS] = SSC + (uint64_t)R * 32; p.HSEND = SO + OUTB;
    Code& kS = p.k[PS]; kS.CDCOPY(HB, SP, PB);
    auto lane = [&](Code& k, int j) { if (j) { k.PUSH((uint64_t)32 * j); k.SHR(); } k.PUSH(0xFFFFFFFFULL); k.AND(); };
    if (B) {
        for (int g = 0; g < Cw; ++g) { kS.PUSH(0); for (int t = 0; t < R; ++t) { kS.PUSH(SP + ((uint64_t)t * Cw + g) * 32); kS.MLOAD(); kS.ADD(); } kS.PUSH(SCS + (uint64_t)g * 32); kS.MSTORE(); }
        for (int j = 0; j < C; ++j) { kS.PUSH(SCS + (uint64_t)(j / 8) * 32); kS.MLOAD(); lane(kS, j % 8); kS.PUSH(11); kS.SHL(); kS.PUSH(SBC + (uint64_t)j * 32); kS.MSTORE(); }
    }
    for (int i = 0; i < C; ++i) {
        for (int t = 0; t < R; ++t) { kS.PUSH(SP + ((uint64_t)t * Cw + i / 8) * 32); kS.MLOAD(); lane(kS, i % 8); kS.PUSH(SSC + (uint64_t)t * 32); kS.MSTORE(); }
        if (B) { kS.PUSH(SBC + (uint64_t)i * 32); kS.MLOAD(); kS.PUSH((uint64_t)R * B * B); kS.SUB(); }          // K_i = R b^2 - b cs'_i
        for (int g = 0; g < Cw; ++g) { kS.PUSH(0);
            for (int t = 0; t < R; ++t) { kS.PUSH(SSC + (uint64_t)t * 32); kS.MLOAD(); kS.PUSH(SP + ((uint64_t)t * Cw + g) * 32); kS.MLOAD(); kS.MUL(); kS.ADD(); }
            for (int j = 0; j < 8; ++j) { kS.DUP(1); lane(kS, j);
                if (B) { kS.DUP(3); kS.ADD(); kS.PUSH(SBC + (uint64_t)(8 * g + j) * 32); kS.MLOAD(); kS.SWAP(1); kS.SUB(); }
                kS.PUSH(SO + ((uint64_t)i * C + 8 * g + j) * 32); kS.MSTORE(); }
            kS.POP(); }
        if (B) kS.POP(); }
    kS.RET(SO, OUTB);
    p.kHs.CDCOPY(HB, SP, PB); p.kHs.RET(SO, OUTB);
}
static int runShape(const int R, const int C, Row& out, const std::vector<std::vector<int>>* img = nullptr, const char* name = "random", bool timing = true) {
    if (NOTIME) timing = false;
    Prog p; build(R, C, img, true, p); const uint64_t OUTB = p.OUTB; const auto& ref = p.ref; const auto& cd = p.cd;
    std::printf("---- R = %d samples, C = %d features, data: %s ----\nBytecode: V=%zu K=%zu MV=%zu MK=%zu S=%zu bytes | %d entries x %d elements = %lld multiply-accumulates\n", R, C, name,
        p.k[PV].c.size(), p.k[PK].c.size(), p.k[PMV].c.size(), p.k[PMK].c.size(), p.k[PS].c.size(), C * C, R, (long long)C * C * R);
    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS; msg.input_data = cd.data(); msg.input_size = cd.size();
    int fail = 0;
    auto runOnce = [&](const Code& k, bool check, uint64_t memend, const char* nm) -> long long {
        auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, k.c.data(), k.c.size());
        long long g = GAS - r.gas_left;
        if (check) {
            bool ok = r.status_code == EVMC_SUCCESS && r.output_data && r.output_size == OUTB;
            if (ok) for (size_t n = 0; n < (size_t)C * C && ok; ++n) { const uint8_t* w = r.output_data + n * 32; uint32_t got = 0; for (int i = 28; i < 32; ++i) got = (got << 8) | w[i]; if (got != ref[n]) ok = false; }
            long long pred = k.sgas + memgas(memend);
            std::printf("  [%s] equals the C++ reference (%d entries): %s | gas %lld | static count + memory %lld: %s\n", nm, C * C, ok ? "PASSED" : "FAILED", g, pred, g == pred ? "equal" : "DIFFERENT");
            if (!ok || g != pred) fail = 1;
        }
        if (r.release) r.release(&r);
        return g;
    };
    // median time of one execution of a bytecode (at least 7 samples and one second, each sample at least 10 ms)
    auto timeCode = [&](const std::vector<uint8_t>& code, const char* nm, const char* what) -> double {
        auto once = [&]() { auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, code.data(), code.size()); if (r.release) r.release(&r); };
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now(); once(); auto t1 = clk::now();
        double est = std::chrono::duration<double>(t1 - t0).count(); if (est <= 0) est = 1e-7;
        long long Rp = (long long)std::ceil(0.010 / est); if (Rp < 1) Rp = 1; if (Rp > 20000) Rp = 20000;
        std::vector<double> S; double tot = 0;
        while (true) { auto a = clk::now(); for (long long i = 0; i < Rp; ++i) once(); auto bb = clk::now();
            double dt = std::chrono::duration<double>(bb - a).count(); S.push_back(dt / Rp); tot += dt; size_t n = S.size();
            if (n >= 3 && ((n >= 7 && tot >= 1.0) || n >= 31 || tot >= 60.0)) break; }
        std::sort(S.begin(), S.end()); const double med = S[S.size() / 2];
        double mu = 0; for (double x : S) mu += x; mu /= S.size(); double sd = 0; for (double x : S) sd += (x - mu) * (x - mu); sd = std::sqrt(sd / S.size());
        std::printf("  [%s %s] med=%.7f s  min=%.7f  max=%.7f  CV=%.1f%%  (samples=%zu, inner-reps=%lld)\n", nm, what, med, S.front(), S.back(), 100.0 * sd / mu, S.size(), Rp);
        std::fflush(stdout);
        return med;
    };
    auto timeBoth = [&](const Code& k, const char* nm, Tm& t) {
        t.tot = timeCode(k.c, nm, "total");
        std::vector<uint8_t> st; st.reserve(k.c.size() + 1); st.push_back(0x00); st.insert(st.end(), k.c.begin(), k.c.end());      // STOP, then the same bytes
        t.base = timeCode(st, nm, "analysis");
    };
    { auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, p.kHh.c.data(), p.kHh.c.size()); out.gHh = GAS - r.gas_left; bool ok = r.status_code == EVMC_SUCCESS && out.gHh == p.kHh.sgas + memgas(p.HEND); if (r.release) r.release(&r);
      auto r2 = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, p.kHs.c.data(), p.kHs.c.size()); out.gHs = GAS - r2.gas_left; ok = ok && r2.status_code == EVMC_SUCCESS && out.gHs == p.kHs.sgas + memgas(p.HSEND); if (r2.release) r2.release(&r2);
      std::printf("  [H] harness (call-data copy and return only): %s | gas %lld (header layout) | gas %lld (packed layout)\n", ok ? "PASSED" : "FAILED", out.gHh, out.gHs); if (!ok) fail = 1; }
    for (int q = 0; q < NP; ++q) {
        out.g[q] = runOnce(p.k[q], true, p.mend[q], PNAME[q]);
        if (runOnce(p.k[q], false, p.mend[q], PNAME[q]) != out.g[q]) { std::printf("  [%s] repeated-run gas differs [FAILED]\n", PNAME[q]); fail = 1; }
        if (timing) timeBoth(p.k[q], PNAME[q], out.t[q]);
    }
    if (timing) { timeBoth(p.kHh, "Hh", out.tHh); timeBoth(p.kHs, "Hs", out.tHs); }
    out.trace = 0; for (int i = 0; i < C; ++i) out.trace += (long long)ref[(size_t)i * C + i]; out.name = name; out.R = R; out.C = C;
    return fail;
}
// total execution gas of one path from the static count (no execution)
static long long staticGas(int R, int C, int path) { Prog p; build(R, C, nullptr, false, p); return p.k[path].sgas + memgas(p.mend[path]); }
// intrinsic gas of a transaction that carries the input of one path as call data: 21000, 16 per non-zero byte and
// 4 per zero byte. The view paths receive the header layout, Path S receives its packed layout.
static long long intrinsicGas(int R, int C, int path) { Prog p; build(R, C, nullptr, true, p); const size_t HB = (size_t)pad32(32 + 2 * (uint64_t)R * C);
    const size_t a = path == PS ? HB : 0, b = path == PS ? p.cd.size() : HB; long long g = 21000; for (size_t i = a; i < b; ++i) g += p.cd[i] ? 16 : 4; return g; }

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    NOTIME = std::getenv("NO_TIMING") != nullptr;
    const int SH[][2] = {{128, 8}, {128, 16}, {128, 24}, {128, 32}, {128, 40}, {128, 48}, {128, 64}, {128, 96}, {128, 128}, {128, 192}, {128, 256}, {16, 32}, {32, 32}, {64, 32}, {256, 32}};
    const int NS = (int)(sizeof(SH) / sizeof(SH[0]));
    std::vector<Row> T(NS); int fail = 0;
    for (int i = 0; i < NS; ++i) fail |= runShape(SH[i][0], SH[i][1], T[i]);
    // real image data: the first 500 images of each file, one image per row
    auto readImgs = [&](const char* file, std::vector<std::vector<int>>& im) { std::ifstream bf(file); int Nb = 0, wsc = 0; bf >> Nb >> wsc; if (Nb < 500) return false; im.assign(500, std::vector<int>(784));
        for (int t = 0; t < 500; ++t) { int lab, pred; bf >> lab >> pred; double tmp; for (int i = 0; i < 10; ++i) bf >> tmp; for (int i = 0; i < 784; ++i) bf >> im[t][i]; } return (bool)bf; };
    std::vector<std::vector<int>> mn, fa; bool okd = readImgs("mnist_batch.txt", mn) && readImgs("fashion_batch.txt", fa);
    if (!okd) { std::printf("image files not found [FAILED]\n"); fail = 1; }
    const int QS[][3] = {{64, 64, 0}, {128, 64, 0}, {256, 64, 0}, {500, 64, 0}, {128, 256, 0}, {128, 64, 1}, {500, 64, 1}, {128, 256, 1}};   // R, C, file
    const int NQ = (int)(sizeof(QS) / sizeof(QS[0])); std::vector<Row> Q(okd ? NQ : 0);
    if (okd) for (int i = 0; i < NQ; ++i) fail |= runShape(QS[i][0], QS[i][1], Q[i], QS[i][2] ? &fa : &mn, QS[i][2] ? "Fashion-MNIST" : "MNIST");
    // largest feature count within the per-transaction cap, R = 128, pseudo-random data
    // (a) execution gas only  (b) the whole transaction: intrinsic gas of the call data of the path plus execution gas
    const long long CAP = 1LL << 24; int cmax[NP], cmaxT[NP]; long long gin[NP], gout[NP], tin[NP], tout[NP], iin[NP]; bool capok[NP];
    for (int q = 0; q < NP; ++q) { int c = 8; while (staticGas(128, c + 8, q) <= CAP) c += 8; cmax[q] = c;
        c = 8; while (staticGas(128, c + 8, q) + intrinsicGas(128, c + 8, q) <= CAP) c += 8; cmaxT[q] = c; }
    std::printf("\n---- shapes at the per-transaction cap (R = 128): each is executed, without timing ----\n");
    for (int q = 0; q < NP; ++q) { const int cs[4] = {cmax[q], cmax[q] + 8, cmaxT[q], cmaxT[q] + 8}; long long ge[4] = {0, 0, 0, 0};
        for (int z = 0; z < 4; ++z) { int seen = -1; for (int y = 0; y < z; ++y) if (cs[y] == cs[z]) seen = y;
            if (seen >= 0) { ge[z] = ge[seen]; continue; }
            Row rz; fail |= runShape(128, cs[z], rz, nullptr, "random", false); ge[z] = rz.g[q]; }
        gin[q] = ge[0]; gout[q] = ge[1]; iin[q] = intrinsicGas(128, cmaxT[q], q); tin[q] = ge[2] + iin[q]; tout[q] = ge[3] + intrinsicGas(128, cmaxT[q] + 8, q);
        capok[q] = gin[q] <= CAP && gout[q] > CAP && tin[q] <= CAP && tout[q] > CAP; if (!capok[q]) fail = 1; }
    std::vector<Row*> ALL; for (auto& r : T) ALL.push_back(&r); for (auto& r : Q) ALL.push_back(&r);
    auto cg = [](const Row& r, int q) { return r.g[q] - (q == PS ? r.gHs : r.gHh); };
    std::printf("\n================ GRAM MATRIX G = A^T A (R samples x C features) ================\n");
    std::printf("COMPUTE GAS = total gas minus harness gas (identical on every build)\n");
    std::printf("          data |    R |   C |   mult-acc |    compute V |    compute K |   compute MV |   compute MK |    compute S |  S/V |  S/K | S/MV | S/MK | K/MV\n");
    for (auto* q : ALL) { auto& r = *q;
        std::printf("%14s | %4d | %3d | %10lld | %12lld | %12lld | %12lld | %12lld | %12lld | %4.2f | %4.2f | %4.2f | %4.2f | %4.2f\n", r.name, r.R, r.C, (long long)r.C * r.C * r.R, cg(r, PV), cg(r, PK), cg(r, PMV), cg(r, PMK), cg(r, PS),
            (double)cg(r, PS) / cg(r, PV), (double)cg(r, PS) / cg(r, PK), (double)cg(r, PS) / cg(r, PMV), (double)cg(r, PS) / cg(r, PMK), (double)cg(r, PK) / cg(r, PMV)); }
    std::printf("COMPUTE GAS PER MULTIPLY-ACCUMULATE AND TOTAL EXECUTION GAS (share of the per-transaction cap of 2^24 gas for Path MV)\n");
    std::printf("          data |    R |   C | V gas/MAC | K gas/MAC | MV gas/MAC | MK gas/MAC | S gas/MAC |    total V |    total K |   total MV |   total MK |    total S | MV of cap\n");
    for (auto* q : ALL) { auto& r = *q; double m = (double)r.C * r.C * r.R;
        std::printf("%14s | %4d | %3d | %9.3f | %9.3f | %10.3f | %10.3f | %9.3f | %10lld | %10lld | %10lld | %10lld | %10lld | %8.1f%%\n", r.name, r.R, r.C, cg(r, PV) / m, cg(r, PK) / m, cg(r, PMV) / m, cg(r, PMK) / m, cg(r, PS) / m,
            r.g[PV], r.g[PK], r.g[PMV], r.g[PMK], r.g[PS], 100.0 * r.g[PMV] / CAP); }
    if (okd) { std::printf("TRACE OF G ON THE IMAGE DATA (sum of the squared pixels, from the reference that all five programs equal)\n");
        for (auto& r : Q) std::printf("%14s | %4d | %3d | %14lld\n", r.name, r.R, r.C, r.trace); }
    std::printf("LARGEST FEATURE COUNT WITHIN THE PER-TRANSACTION CAP OF 2^24 = %lld GAS (R = 128, C in steps of 8). Whole transaction = intrinsic gas (21000, 16 per non-zero and 4 per zero byte of the call data of the path) + execution gas\n", CAP);
    std::printf(" path | execution gas only: largest C | entries of G |   mult-acc | execution gas at C | at C + 8 | whole transaction: largest C | entries of G |   mult-acc | intrinsic gas at C | transaction gas at C | at C + 8 | confirmed by execution\n");
    for (int q = 0; q < NP; ++q) std::printf("   %2s | %29d | %12d | %10lld | %18lld | %8lld | %28d | %12d | %10lld | %18lld | %20lld | %8lld | %s\n", PNAME[q], cmax[q], cmax[q] * cmax[q], (long long)cmax[q] * cmax[q] * 128, gin[q], gout[q],
        cmaxT[q], cmaxT[q] * cmaxT[q], (long long)cmaxT[q] * cmaxT[q] * 128, iin[q], tin[q], tout[q], capok[q] ? "yes" : "NO");
    // times below are machine and build dependent; every table from here on starts with the word TIME
    auto comp = [](const Row& r, int q) { const Tm& hh = (q == PS) ? r.tHs : r.tHh; return (r.t[q].tot - r.t[q].base) - (hh.tot - hh.base); };
    if (NOTIME) std::printf("TIME tables are left out (NO_TIMING is set)\n");
    else {
    std::printf("TIME, TOTAL of one call in ms (median; bytecode analysis, call-data copy, computation and return)\n");
    std::printf("          data |    R |   C |        V |        K |       MV |       MK |        S | copy-return, header layout | copy-return, packed layout\n");
    for (auto* q : ALL) { auto& r = *q; std::printf("%14s | %4d | %3d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %26.4f | %26.4f\n", r.name, r.R, r.C, r.t[PV].tot * 1e3, r.t[PK].tot * 1e3, r.t[PMV].tot * 1e3, r.t[PMK].tot * 1e3, r.t[PS].tot * 1e3, r.tHh.tot * 1e3, r.tHs.tot * 1e3); }
    std::printf("TIME, ANALYSIS in ms (median; the same bytecode behind a STOP: code analysis and call setup, nothing executed) and its share of the total\n");
    std::printf("          data |    R |   C |        V |        K |       MV |       MK |        S | share V | share K | share MV | share MK | share S\n");
    for (auto* q : ALL) { auto& r = *q; std::printf("%14s | %4d | %3d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %6.1f%% | %6.1f%% | %7.1f%% | %7.1f%% | %6.1f%%\n", r.name, r.R, r.C, r.t[PV].base * 1e3, r.t[PK].base * 1e3, r.t[PMV].base * 1e3, r.t[PMK].base * 1e3, r.t[PS].base * 1e3,
        100 * r.t[PV].base / r.t[PV].tot, 100 * r.t[PK].base / r.t[PK].tot, 100 * r.t[PMV].base / r.t[PMV].tot, 100 * r.t[PMK].base / r.t[PMK].tot, 100 * r.t[PS].base / r.t[PS].tot); }
    std::printf("TIME, COMPUTE in ms = (total - analysis) of the path minus (total - analysis) of its copy-and-return program\n");
    std::printf("          data |    R |   C |        V |        K |       MV |       MK |        S |   S/V |   S/K |  S/MV |  S/MK |  K/MV | ns/MAC V | ns/MAC K | ns/MAC MV | ns/MAC MK | ns/MAC S | ns/gas V | ns/gas K | ns/gas MV | ns/gas MK | ns/gas S\n");
    for (auto* q : ALL) { auto& r = *q; double m = (double)r.C * r.C * r.R; double c[NP]; for (int z = 0; z < NP; ++z) c[z] = comp(r, z);
        std::printf("%14s | %4d | %3d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %5.2f | %5.2f | %5.2f | %5.2f | %5.2f | %8.3f | %8.3f | %9.3f | %9.3f | %8.3f | %8.3f | %8.3f | %9.3f | %9.3f | %8.3f\n", r.name, r.R, r.C,
            c[PV] * 1e3, c[PK] * 1e3, c[PMV] * 1e3, c[PMK] * 1e3, c[PS] * 1e3, c[PS] / c[PV], c[PS] / c[PK], c[PS] / c[PMV], c[PS] / c[PMK], c[PK] / c[PMV],
            c[PV] * 1e9 / m, c[PK] * 1e9 / m, c[PMV] * 1e9 / m, c[PMK] * 1e9 / m, c[PS] * 1e9 / m,
            c[PV] * 1e9 / cg(r, PV), c[PK] * 1e9 / cg(r, PK), c[PMV] * 1e9 / cg(r, PMV), c[PMK] * 1e9 / cg(r, PMK), c[PS] * 1e9 / cg(r, PS)); }
    }
    std::printf("\n%s\n", fail ? "*** A CHECK FAILED -> the numbers are not valid ***" : "*** ALL CHECKS PASSED -> gas is valid; timing numbers are valid for this build ***");
    VM->destroy(VM); dlclose(h);
    return fail;
}
