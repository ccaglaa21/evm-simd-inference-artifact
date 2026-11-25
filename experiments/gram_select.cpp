// gram_select.cpp — DEPLOYABLE GRAM MATRIX CONTRACT THAT CHOOSES ITS PATH ON-CHAIN (gas only).
//
// gram_views compares straight-line programs. This program uses looped contracts whose shape is not fixed
// in the bytecode: the sample count R and the feature count C arrive with the call (two words in front of
// the matrix), and the same bytecode serves every shape. Three contracts compute G = A^T A:
//   fixed V   transposed view of A (XTRANSPOSE, no element is moved), one XMACC_V per entry, strided rows
//   fixed K   transposed copy of A (one MCOPY per element in a loop, then a header), one XMACC_V per entry
//             over unit-stride rows
//   select    evaluates on-chain, from R and C, the gas difference of the two paths and takes the cheaper
// Gas is deterministic, so the difference is a closed form in R and C:
//   gas(K) - gas(V) = k0 + C * (co + R * cb) + mem(wK) - mem(wV) - 8 * R * C^2
// where k0, co and cb are the gas of the once-only part, of one outer iteration and of one inner iteration of
// the copy loop (counted by the generator from the emitted bytecode), mem(w) = 3 w + floor(w^2 / 512) is
// the memory price of w words, wV and wK are the final memory sizes of the two paths, and 8 R is the charge
// of the strided elements of one entry on the view (two rows of R elements at 4 gas each), which the copy
// does not pay. The contract takes Path K when this difference is negative.
// CHECKS per shape: (1) the three contracts equal the C++ reference in all C*C entries (2) the gas of each
// fixed contract equals the closed form of the generator (3) the measured difference of the two fixed
// contracts equals the closed form that the selecting contract evaluates (4) the gas of the selecting contract
// equals the gas of the cheaper fixed contract plus a constant (the cost of the selection). The program has
// no timing, so its whole output is identical on every build.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <fstream>
static evmc::MockedHost g_host;
static evmc_vm* VM;
static const int64_t GAS = 200000000000LL;
struct Asm {
    std::vector<uint8_t> c; long long g = 0; std::map<std::string, size_t> lab; std::vector<std::pair<size_t, std::string>> fix;
    void b(uint8_t x, long long gas) { c.push_back(x); g += gas; }
    void PUSH(uint64_t v) { int n = 0; for (uint64_t t = v; t; t >>= 8) ++n; if (n == 0) n = 1;
        b((uint8_t)(0x5F + n), 3); for (int k = n - 1; k >= 0; --k) c.push_back((uint8_t)((v >> (8 * k)) & 0xFF)); }
    void PUSHL(const std::string& l) { b(0x61, 3); fix.push_back({c.size(), l}); c.push_back(0); c.push_back(0); }
    void LABEL(const std::string& l) { lab[l] = c.size(); b(0x5B, 1); }
    void finish() { for (auto& f : fix) { size_t a = lab.at(f.second); c[f.first] = (uint8_t)(a >> 8); c[f.first + 1] = (uint8_t)(a & 0xFF); } }
    void MLOAD() { b(0x51, 3); } void MSTORE() { b(0x52, 3); } void MSTORE8() { b(0x53, 3); } void MUL() { b(0x02, 5); } void ADD() { b(0x01, 3); } void SUB() { b(0x03, 3); }
    void SHR() { b(0x1C, 3); } void SHL() { b(0x1B, 3); } void GT() { b(0x11, 3); } void LT() { b(0x10, 3); } void DUP(int n) { b((uint8_t)(0x7F + n), 3); } void SWAP(int n) { b((uint8_t)(0x8F + n), 3); } void POP() { b(0x50, 2); }
    void JUMP() { b(0x56, 8); } void JUMPI() { b(0x57, 10); }
    void LDR() { PUSH(0); MLOAD(); } void LDC() { PUSH(32); MLOAD(); }      // R and C from their scratch words
};
// memory map: 0 R | 32 C | 64 offset of the outputs | 96 offset of the matrix that the entries read | 128 matrix A (header + elements)
static const uint64_t MA = 128;
struct Seg { long long common, sel, nosel, v, k, co, cb, ro, eb, tail; };   // gas of the code segments (see closed())
// mode: 0 fixed V, 1 fixed K, 2 selecting. K0, CO, CB: constants of the closed form placed in the selecting contract.
static Asm gen(int mode, long long K0, long long CO, long long CB, Seg& s) {
    Asm a; long long m;
    // --- common head: R, C, the matrix, the offset of the outputs ---
    m = a.g;
    a.PUSH(0); a.b(0x35, 3); a.PUSH(0); a.MSTORE();                 // R = calldata word 0
    a.PUSH(32); a.b(0x35, 3); a.PUSH(32); a.MSTORE();               // C = calldata word 1
    a.PUSH(64); a.b(0x36, 2); a.SUB(); a.PUSH(64); a.PUSH(MA); a.b(0x37, 3);   // CALLDATACOPY(MA, 64, size - 64): the 3 gas per word is added in closed()
    a.LDR(); a.LDC(); a.MUL(); a.PUSH(1); a.SHL(); a.PUSH(63); a.ADD(); a.PUSH(5); a.SHR();   // Pw = words of the matrix
    a.DUP(1); a.PUSH(5); a.SHL(); a.PUSH(MA); a.ADD(); a.PUSH(64); a.MSTORE();                // outputs at MA + 32 Pw
    s.common = a.g - m;
    // --- the choice: leaves 1 on the stack for Path K, 0 for Path V (Pw is consumed) ---
    m = a.g;
    if (mode == 2) {
        a.LDC(); a.DUP(1); a.MUL(); a.DUP(2); a.ADD(); a.PUSH(4); a.ADD();     // wV = 4 + Pw + C^2
        a.DUP(2); a.DUP(2); a.ADD();                                           // wK = wV + Pw
        a.DUP(1); a.MUL(); a.PUSH(9); a.SHR();                                 // floor(wK^2 / 512)
        a.SWAP(1); a.DUP(1); a.MUL(); a.PUSH(9); a.SHR();                      // floor(wV^2 / 512)
        a.SWAP(1); a.SUB();                                                    // difference of the quadratic terms
        a.SWAP(1); a.PUSH(3); a.MUL(); a.ADD();                                // + 3 Pw: mem(wK) - mem(wV)
        a.LDR(); a.PUSH((uint64_t)CB); a.MUL(); a.PUSH((uint64_t)CO); a.ADD(); a.LDC(); a.MUL(); a.ADD();   // + C (co + R cb)
        a.PUSH((uint64_t)K0); a.ADD();                                         // + k0: the extra gas of Path K
        a.LDC(); a.DUP(1); a.MUL(); a.LDR(); a.MUL(); a.PUSH(8); a.MUL();                                         // 8 R C^2: its saving
        a.GT();                                                                // saving > extra gas
        s.sel = a.g - m;
    } else { a.POP(); a.PUSH((uint64_t)mode); s.nosel = a.g - m; }
    a.PUSHL("K"); a.JUMPI();          // counted in the branch segments below
    // --- Path V: transposed view ---
    m = a.g;
    a.PUSH(MA); a.b(0xDD, 22); a.POP(); a.PUSH(MA); a.PUSH(96); a.MSTORE(); a.PUSHL("E"); a.JUMP();
    s.v = a.g - m;
    // --- Path K: transposed copy ---
    m = a.g;
    a.LABEL("K");
    a.PUSH(64); a.MLOAD(); a.LDC(); a.DUP(1); a.MUL(); a.PUSH(5); a.SHL(); a.ADD(); a.DUP(1); a.PUSH(96); a.MSTORE();   // MT = outputs + 32 C^2
    a.PUSH(32); a.ADD(); a.PUSH(0);                                   // stack: dst c
    long long once = a.g - m, mo = a.g;
    a.LABEL("Lc");
    a.DUP(1); a.PUSH(1); a.SHL(); a.PUSH(MA + 32); a.ADD();           // src = first element of column c
    a.DUP(1); a.LDR(); a.LDC(); a.MUL(); a.PUSH(1); a.SHL(); a.ADD(); a.SWAP(1);   // stack: dst c end src
    long long outer1 = a.g - mo, mi = a.g;
    a.LABEL("Lt");
    a.PUSH(2); a.DUP(2); a.DUP(6); a.b(0x5E, 6);                      // MCOPY(dst, src, 2)
    a.SWAP(3); a.PUSH(2); a.ADD(); a.SWAP(3);                         // dst += 2
    a.LDC(); a.PUSH(1); a.SHL(); a.ADD();                             // src += 2 C
    a.DUP(2); a.DUP(2); a.LT(); a.PUSHL("Lt"); a.JUMPI();
    s.cb = a.g - mi; long long mo2 = a.g;
    a.POP(); a.POP(); a.PUSH(1); a.ADD(); a.DUP(1); a.LDC(); a.GT(); a.PUSHL("Lc"); a.JUMPI();
    s.co = outer1 + (a.g - mo2); long long mh = a.g;
    a.POP(); a.POP();
    // header of the copy: rows = C, cols = R, stride_row = R, stride_col = 1 (little-endian fields, values below 65536)
    auto fld = [&](bool isC, uint64_t off) { for (int byte = 0; byte < 2; ++byte) { if (isC) a.LDC(); else a.LDR(); if (byte) { a.PUSH(8); a.SHR(); }
        a.PUSH(96); a.MLOAD(); if (off + byte) { a.PUSH(off + byte); a.ADD(); } a.MSTORE8(); } };
    fld(true, 0); fld(false, 8); fld(false, 16);
    a.PUSH(1); a.PUSH(96); a.MLOAD(); a.PUSH(24); a.ADD(); a.MSTORE8();
    s.k = once + (a.g - mh);
    // --- the entries: one XMACC_V per entry over rows i and j of the matrix at the offset in word 96 ---
    m = a.g;
    a.LABEL("E"); a.PUSH(64); a.MLOAD(); a.PUSH(0);                   // stack: ptr i
    long long ehead = a.g - m, mr = a.g;
    a.LABEL("Li"); a.PUSH(0);                                         // stack: ptr i j
    long long r1 = a.g - mr, me = a.g;
    a.LABEL("Lj");
    a.PUSH(0); a.DUP(2); a.PUSH(96); a.MLOAD(); a.DUP(5); a.DUP(2); a.b(0xEA, 0); a.b(0xCD, 14);
    a.DUP(4); a.MSTORE(); a.SWAP(2); a.PUSH(32); a.ADD(); a.SWAP(2);
    a.PUSH(1); a.ADD(); a.DUP(1); a.LDC(); a.GT(); a.PUSHL("Lj"); a.JUMPI();
    s.eb = a.g - me; long long mr2 = a.g;
    a.POP(); a.PUSH(1); a.ADD(); a.DUP(1); a.LDC(); a.GT(); a.PUSHL("Li"); a.JUMPI();
    s.ro = r1 + (a.g - mr2); long long mt = a.g;
    a.POP(); a.POP(); a.LDC(); a.DUP(1); a.MUL(); a.PUSH(5); a.SHL(); a.PUSH(64); a.MLOAD(); a.b(0xF3, 0);
    s.tail = ehead + (a.g - mt);
    a.finish();
    return a;
}
static long long memgas(long long words) { return 3 * words + words * words / 512; }
static const long long JI = 13;   // PUSH2 label + JUMPI after the choice
// closed form of the total gas of a fixed contract (path 0 = V, 1 = K), without the choice segment
static long long closed(const Seg& s, int path, long long R, long long C) {
    long long Pw = (2 * R * C + 63) / 32, G = (R + 7) / 8, wV = 4 + Pw + C * C, wK = wV + Pw;
    long long body = path ? s.k + C * (s.co + R * s.cb) : s.v;
    long long entries = C * C * (s.eb + 50 + G * 10 + (path ? 0 : 8 * R)) + C * s.ro;
    return s.common + 3 * Pw + JI + body + entries + s.tail + memgas(path ? wK : wV);
}
static uint32_t rnd = 2026u; static uint32_t rnd32() { rnd = rnd * 1664525u + 1013904223u; return rnd; }
struct Res { evmc_status_code st; long long gas; std::vector<uint8_t> out; };
static Res run(const Asm& a, const std::vector<uint8_t>& cd) {
    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS; msg.input_data = cd.data(); msg.input_size = cd.size();
    auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, a.c.data(), a.c.size());
    Res o; o.st = r.status_code; o.gas = GAS - r.gas_left; if (r.output_data) o.out.assign(r.output_data, r.output_data + r.output_size);
    if (r.release) r.release(&r); return o;
}

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    Seg s0{}, sV{}, sK{}, sS{};
    gen(0, 0, 0, 0, s0);                                    // first pass: the gas of the code segments
    const long long K0 = s0.k - s0.v, CO = s0.co, CB = s0.cb;
    Asm cV = gen(0, K0, CO, CB, sV), cK = gen(1, K0, CO, CB, sK), cS = gen(2, K0, CO, CB, sS);
    const long long OVH = sS.sel - sS.nosel + 0 * sV.sel;   // the selecting contract replaces POP + PUSH by the evaluation of the closed form
    const long long OVHc = sS.sel - sV.nosel;
    (void)OVH;
    std::printf("=== DEPLOYABLE GRAM CONTRACT WITH ON-CHAIN CHOICE OF THE PATH ===\n");
    std::printf("Bytecode: fixed V %zu bytes, fixed K %zu bytes, selecting %zu bytes (one bytecode for every shape)\n", cV.c.size(), cK.c.size(), cS.c.size());
    std::printf("Constants of the closed form, counted from the emitted bytecode: k0 = %lld, co = %lld, cb = %lld gas\n", K0, CO, CB);
    std::printf("Gas per entry outside XMACC_V: %lld | per row of G: %lld | XMACC_V: 50 per call, 10 per group of eight, 4 per strided element\n", sV.eb, sV.ro);
    std::printf("Gas of the selection (evaluation of the closed form in place of a constant): %lld\n\n", OVHc);
    std::vector<std::vector<int>> mn;
    { std::ifstream bf("mnist_batch.txt"); int Nb = 0, wsc = 0; bf >> Nb >> wsc; if (Nb >= 500) { mn.assign(500, std::vector<int>(784));
        for (int t = 0; t < 500; ++t) { int lab, pred; bf >> lab >> pred; double tmp; for (int i = 0; i < 10; ++i) bf >> tmp; for (int i = 0; i < 784; ++i) bf >> mn[t][i]; } if (!bf) mn.clear(); } }
    int fail = 0; if (mn.empty()) { std::printf("image file not found [FAILED]\n"); fail = 1; }
    struct Out { long long gV, gK, gS; bool ok; int pick; };
    auto shape = [&](int R, int C, bool mnist) {
        std::vector<int> A((size_t)R * C);
        if (mnist) { int w = 1; while (w * w < C) ++w; const int r0 = (28 - w) / 2; for (int t = 0; t < R; ++t) for (int c = 0; c < C; ++c) A[(size_t)t * C + c] = mn[t][(r0 + c / w) * 28 + r0 + c % w]; }
        else { rnd = 2026u + (uint32_t)(R * 1000 + C); for (auto& x : A) x = (int)(rnd32() % 4096) - 2048; }
        std::vector<uint32_t> ref((size_t)C * C);
        for (int i = 0; i < C; ++i) for (int j = i; j < C; ++j) { int64_t t = 0; for (int r = 0; r < R; ++r) t += (int64_t)A[(size_t)r * C + i] * A[(size_t)r * C + j]; ref[(size_t)i * C + j] = ref[(size_t)j * C + i] = (uint32_t)t; }
        std::vector<uint8_t> cd(64 + 32 + 2 * (size_t)R * C, 0);
        for (int k = 0; k < 8; ++k) { cd[31 - k] = (uint8_t)((uint64_t)R >> (8 * k)); cd[63 - k] = (uint8_t)((uint64_t)C >> (8 * k)); }
        auto p64 = [&](size_t at, uint64_t v) { for (int k = 0; k < 8; ++k) cd[at + k] = (uint8_t)(v >> (8 * k)); };
        p64(64, R); p64(72, C); p64(80, C); p64(88, 1);
        for (size_t e = 0; e < A.size(); ++e) { uint16_t u = (uint16_t)(int16_t)A[e]; cd[96 + 2 * e] = u & 0xFF; cd[97 + 2 * e] = u >> 8; }
        Res rV = run(cV, cd), rK = run(cK, cd), rS = run(cS, cd), rS2 = run(cS, cd);
        auto good = [&](const Res& r) { if (r.st != EVMC_SUCCESS || r.out.size() != (size_t)C * C * 32) return false;
            for (size_t n = 0; n < (size_t)C * C; ++n) { const uint8_t* w = r.out.data() + n * 32; uint32_t got = 0; for (int i = 28; i < 32; ++i) got = (got << 8) | w[i]; if (got != ref[n]) return false; } return true; };
        Out o; o.gV = rV.gas; o.gK = rK.gas; o.gS = rS.gas;
        long long fV = closed(sV, 0, R, C) + sV.nosel, fK = closed(sK, 1, R, C) + sK.nosel;
        long long Pw = (2LL * R * C + 63) / 32, wV = 4 + Pw + (long long)C * C, wK = wV + Pw;
        long long diff = K0 + (long long)C * (CO + (long long)R * CB) + memgas(wK) - memgas(wV) - 8LL * R * C * C;
        o.pick = diff < 0 ? 1 : 0;
        long long best = o.pick ? o.gK : o.gV;
        o.ok = good(rV) && good(rK) && good(rS) && rV.gas == fV && rK.gas == fK && (rK.gas - rV.gas) == diff && rS.gas == best + OVHc && rS2.gas == rS.gas
               && ((o.gK < o.gV) == (o.pick == 1));
        if (!o.ok) fail = 1;
        return o; };
    std::printf("SHAPES (gas of the whole call; 'chosen' is the path that the selecting contract takes)\n");
    std::printf("    data |    R |   C |      fixed V |      fixed K |    selecting | chosen | cheaper fixed | selecting - cheaper | K - V measured | checks\n");
    const int SH[][3] = {{16, 8, 0}, {16, 32, 0}, {16, 64, 0}, {64, 8, 0}, {64, 16, 0}, {64, 32, 0}, {64, 64, 0}, {128, 8, 0}, {128, 16, 0}, {128, 32, 0}, {128, 64, 0}, {128, 80, 0}, {128, 88, 0}, {128, 96, 0}, {128, 128, 0}, {128, 192, 0},
                         {256, 8, 0}, {256, 32, 0}, {256, 64, 0}, {256, 128, 0}, {64, 64, 1}, {128, 64, 1}, {500, 64, 1}, {128, 100, 1}, {128, 256, 1}};
    for (auto& q : SH) { if (q[2] && mn.empty()) continue; Out o = shape(q[0], q[1], q[2]);
        std::printf("%8s | %4d | %3d | %12lld | %12lld | %12lld | %6s | %13s | %19lld | %14lld | %s\n", q[2] ? "MNIST" : "random", q[0], q[1], o.gV, o.gK, o.gS, o.pick ? "K" : "V", o.gK < o.gV ? "K" : "V",
            o.gS - (o.gK < o.gV ? o.gK : o.gV), o.gK - o.gV, o.ok ? "PASSED" : "FAILED"); }
    std::printf("\nCROSSOVER (every feature count from 2 to 128 is executed with the three contracts; random data)\n");
    std::printf("    R | smallest C at which Path K is chosen | K - V at that C | K - V at C - 1 | shapes executed | all checks\n");
    for (int R : {8, 16, 32, 64, 128, 256, 500}) { int cstar = -1; long long dAt = 0, dBefore = 0, prev = 0; bool all = true; int n = 0;
        for (int C = 2; C <= 128; ++C) { Out o = shape(R, C, false); ++n; all = all && o.ok; if (o.pick && cstar < 0) { cstar = C; dAt = o.gK - o.gV; dBefore = prev; } prev = o.gK - o.gV; }
        if (cstar < 0) std::printf("%5d | %36s | %15s | %14s | %15d | %s\n", R, "none up to 128", "-", "-", n, all ? "PASSED" : "FAILED");
        else std::printf("%5d | %36d | %15lld | %14lld | %15d | %s\n", R, cstar, dAt, dBefore, n, all ? "PASSED" : "FAILED"); }
    std::printf("\n%s\n", fail ? "*** A CHECK FAILED -> the numbers are not valid ***" : "*** ALL CHECKS PASSED ***");
    VM->destroy(VM); dlclose(h);
    return fail;
}
