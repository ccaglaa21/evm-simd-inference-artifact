// mnist_rowmacc.cpp — Path N: the network written with the row-level instruction XMACC_N
// (one instruction per neuron, element-count operand). Two variants are measured:
//   N-unrolled   straight-line, unrolled loader, one XMACC_N per neuron (static prediction, single image)
//   N-deployable looped loader + batch loop + looped neuron loop (code size under EIP-170, marginal cost)
// Memory map, loader, data layout, activation and argmax code are those of the verified Path B programs
// (mnist_unrolled_simd / mnist_deploy_cb); only the per-group PUSH, MLOAD, PUSH, MLOAD, XMACC sequence
// is replaced by one XMACC_N per neuron.
// CHECKS: (1) 20 images, C++ reference == N-unrolled == N-deployable (2) N-unrolled warm predicted == measured,
// MSIZE predicted == measured (3) cold-warm == matWords*2000 (both) (4) compute cold == warm (5) deployable
// code size under 24,576 (6) repeat run identical (7) deployable marginal cost constant over N = 1..16.
// Compile with -DLOCAL_TEST to dump the bytecode, storage and calldata instead of running evmone.
#ifndef LOCAL_TEST
#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void put16(std::vector<uint8_t>& w, int lane, int val) { uint16_t u = (uint16_t)(int16_t)val; w[lane * 2] = u & 0xFF; w[lane * 2 + 1] = (u >> 8) & 0xFF; }
static void put32(std::vector<uint8_t>& w, int lane, long long val) { uint32_t u = (uint32_t)(int32_t)val; for (int b = 0; b < 4; ++b) w[lane * 4 + b] = (u >> (8 * b)) & 0xFF; }
static long long opGas(uint8_t op) {
    if (op >= 0x60 && op <= 0x7F) return 3;
    if (op >= 0x80 && op <= 0x8F) return 3;
    if (op >= 0x90 && op <= 0x9F) return 3;
    switch (op) {
        case 0x01: case 0x03: return 3;
        case 0x02: return 5;
        case 0x0B: return 5;
        case 0x13: case 0x15: case 0x16: case 0x17: return 3;
        case 0x1B: case 0x1C: return 3;
        case 0x50: return 2;
        case 0x51: case 0x52: case 0x53: return 3;
        case 0x54: return 100;      // SLOAD, warm (the prediction is for the warm run)
        case 0x59: return 2;
        case 0xF3: return 0;
        case 0xCB: return 11;       // XMACC
        case 0xCC: case 0xCD: case 0xCE: return 6;
        case 0xE6: return 0;        // XMACC_N static part; the dynamic part is added by XN()
        default: std::fprintf(stderr, "opGas: unknown %02x\n", op); std::exit(2);
    }
}
struct Code {
    std::vector<uint8_t> c; long long g = 0; long long peakB = 0;
    void mem(long long e) { if (e > peakB) peakB = e; }
    void op(uint8_t x) { c.push_back(x); g += opGas(x); }
    void raw(uint8_t x) { c.push_back(x); }
    void P1(uint8_t v) { op(0x60); raw(v); }
    void P2(int v) { op(0x61); raw((v >> 8) & 0xFF); raw(v & 0xFF); }
    void PUSH(long long v) { if (v < 256) P1((uint8_t)v); else if (v < 65536) P2((int)v); else { op(0x62); raw((v >> 16) & 0xFF); raw((v >> 8) & 0xFF); raw(v & 0xFF); } }
    void W32(std::vector<uint8_t> b) { while (b.size() < 32) b.push_back(0); op(0x7F); for (int i = 0; i < 32; ++i) raw(b[i]); }
    void XN(long long n) { op(0xE6); g += 11 * n; }   // XMACC_N: 11 gas per group
};
static long long memgas(long long bytes) { long long a = (bytes + 31) / 32; return 3 * a + (a * a) / 512; }

int main() {
    int S, SHIFT1, SHIFT2, H1, H2;
    { std::ifstream f("deep_meta.txt"); f >> S >> SHIFT1 >> SHIFT2 >> H1 >> H2; }
    const int I = 784, CDIMG = I / 8 * 32;
    std::vector<int> W1(H1 * I), W2(H2 * H1), W3(H2 * 10);
    { std::ifstream f("deep_W1.txt"); for (auto& x : W1) f >> x; }
    { std::ifstream f("deep_W2.txt"); for (auto& x : W2) f >> x; }
    { std::ifstream f("deep_W3.txt"); for (auto& x : W3) f >> x; }
    std::vector<long long> b1(H1), b2(H2);
    { std::ifstream f("deep_b1.txt"); for (auto& x : b1) f >> x; }
    { std::ifstream f("deep_b2.txt"); for (auto& x : b2) f >> x; }
    std::ifstream bf("mnist_batch.txt"); int Nb, wsc; bf >> Nb >> wsc;
    const int N_RUN = std::min(Nb, 500);
    std::vector<std::vector<int>> imgs(N_RUN, std::vector<int>(I));
    for (int im = 0; im < N_RUN; ++im) { int tlb, pfo; bf >> tlb >> pfo; double tmp; for (int i = 0; i < 10; ++i) bf >> tmp; for (int i = 0; i < I; ++i) bf >> imgs[im][i]; }
    auto rss = [&](long long v, int sh) -> int { if (v < 0) v = 0; v >>= sh; if (v > 32767) v = 32767; return (int)v; };
    auto cppScores = [&](int im, std::vector<long long>& sc) -> int {
        std::vector<int>& g = imgs[im];
        std::vector<int> h1r(H1); for (int n = 0; n < H1; ++n) { long long s = b1[n]; for (int i = 0; i < I; ++i) s += (long long)g[i] * W1[i * H1 + n]; h1r[n] = rss(s, SHIFT1); }
        std::vector<int> h2r(H2); for (int n = 0; n < H2; ++n) { long long s = b2[n]; for (int i = 0; i < H1; ++i) s += (long long)h1r[i] * W2[i * H2 + n]; h2r[n] = rss(s, SHIFT2); }
        sc.assign(10, 0); for (int c = 0; c < 10; ++c) { long long s = 0; for (int i = 0; i < H2; ++i) s += (long long)h2r[i] * W3[i * 10 + c]; sc[c] = s; }
        return (int)(std::max_element(sc.begin(), sc.end()) - sc.begin());
    };
    auto cppPred = [&](int im) -> int { std::vector<long long> sc; return cppScores(im, sc); };

    // memory map: identical to the Path B programs
    const int INP = 0, W1M = 10000, B1M = 420000, H1OUT = 430000, W2M = 440000, B2M = 480000, H2OUT = 490000;
    const int W3M = 495000, SCO = 500000;
    const int ACC = 900000, SACC = 900100, NCNT = 901000, GCNT = 901100;
    const int MAXV = 900200, MAXI = 900232, CNT2 = 900264;
    const int NIMG = 901232, ICNT = 901264, PREDS = 901296;
    (void)ACC;

    std::vector<std::pair<std::vector<uint8_t>, int>> WLOAD;
    for (int n = 0; n < H1; ++n) for (int gw = 0; gw < I / 8; ++gw) { std::vector<uint8_t> w(32, 0); for (int l = 0; l < 8; ++l) put16(w, l, W1[(gw * 8 + l) * H1 + n]); WLOAD.push_back({w, W1M + (n * (I / 8) + gw) * 32}); }
    for (int n = 0; n < H1; ++n) { std::vector<uint8_t> w(32, 0); put32(w, 0, b1[n]); WLOAD.push_back({w, B1M + n * 32}); }
    for (int n = 0; n < H2; ++n) for (int gw = 0; gw < H1 / 8; ++gw) { std::vector<uint8_t> w(32, 0); for (int l = 0; l < 8; ++l) put16(w, l, W2[(gw * 8 + l) * H2 + n]); WLOAD.push_back({w, W2M + (n * (H1 / 8) + gw) * 32}); }
    for (int n = 0; n < H2; ++n) { std::vector<uint8_t> w(32, 0); put32(w, 0, b2[n]); WLOAD.push_back({w, B2M + n * 32}); }
    for (int c = 0; c < 10; ++c) for (int gw = 0; gw < H2 / 8; ++gw) { std::vector<uint8_t> w(32, 0); for (int l = 0; l < 8; ++l) put16(w, l, W3[(gw * 8 + l) * 10 + c]); WLOAD.push_back({w, W3M + (c * (H2 / 8) + gw) * 32}); }
    const int matWords = (int)WLOAD.size();
    struct Seg { int s0, cnt, base; };
    std::vector<Seg> segs;
    for (int sl = 0; sl < matWords; ++sl) { if (segs.empty() || WLOAD[sl].second != WLOAD[sl - 1].second + 32) segs.push_back({sl, 1, WLOAD[sl].second}); else segs.back().cnt++; }
    if (segs.size() != 5) { printf("*** layout is not contiguous, stop ***\n"); return 1; }
    const long long MACS = (long long)I * H1 + (long long)H1 * H2 + (long long)H2 * 10;

    // ================= N-unrolled (straight-line, static prediction) =================
    Code k;
    k.PUSH(CDIMG); k.P1(0); k.PUSH(INP); k.raw(0x37); k.g += 3 + 3 * (CDIMG / 32); k.mem(INP + CDIMG);
    for (int sl = 0; sl < matWords; ++sl) {
        std::vector<uint8_t> key(32, 0); key[28] = (sl >> 24) & 0xFF; key[29] = (sl >> 16) & 0xFF; key[30] = (sl >> 8) & 0xFF; key[31] = sl & 0xFF;
        k.W32(key); k.op(0x54); k.PUSH(WLOAD[sl].second); k.op(0x52); k.mem(WLOAD[sl].second + 32);
    }
    const long long g_loader = k.g;
    auto unrolled_layer = [&](int M, int NGin, int GINP, int GWGT, int GBIA, int GOUT, int SHIFT) {
        for (int s = 0; s < M / 8; ++s) { k.P1(0); k.PUSH(SACC + s * 32); k.op(0x52); k.mem(SACC + s * 32 + 32); }
        for (int n = 0; n < M; ++n) {
            k.PUSH(GBIA + n * 32); k.op(0x51); k.mem(GBIA + n * 32 + 32);          // [acc] = bias word
            k.PUSH(GINP); k.PUSH(GWGT + n * NGin * 32); k.PUSH(NGin); k.XN(NGin);  // [acc'] one XMACC_N per neuron
            k.mem(GINP + NGin * 32); k.mem(GWGT + (n + 1) * NGin * 32);
            k.op(0xCD);                                                            // XSUM32 -> [sum]
            int sh = (n % 8) * 32; if (sh) { k.PUSH(sh); k.op(0x1B); }
            int sw = SACC + (n / 8) * 32;
            k.PUSH(sw); k.op(0x51); k.op(0x17); k.PUSH(sw); k.op(0x52); k.mem(sw + 32);
            if (n % 8 == 7) { k.PUSH(sw); k.op(0x51); k.op(0xCC); k.raw((uint8_t)SHIFT); k.op(0xCE); k.PUSH(GOUT + (n / 8) * 32); k.op(0x52); k.mem(GOUT + (n / 8) * 32 + 32); }
        }
    };
    unrolled_layer(H1, I / 8, INP, W1M, B1M, H1OUT, SHIFT1);
    const long long g_l1 = k.g;
    unrolled_layer(H2, H1 / 8, H1OUT, W2M, B2M, H2OUT, SHIFT2);
    for (int c = 0; c < 10; ++c) {
        k.P1(0); k.PUSH(H2OUT); k.PUSH(W3M + c * (H2 / 8) * 32); k.PUSH(H2 / 8); k.XN(H2 / 8);
        k.op(0xCD); k.P1(3); k.op(0x0B);                                           // XSUM32, SIGNEXTEND(3)
        k.PUSH(SCO + c * 32); k.op(0x52); k.mem(SCO + c * 32 + 32);
    }
    k.PUSH(SCO); k.op(0x51); k.PUSH(MAXV); k.op(0x52); k.mem(MAXV + 32);
    k.P1(0); k.PUSH(MAXI); k.op(0x52); k.mem(MAXI + 32);
    for (int c = 1; c < 10; ++c) {
        k.PUSH(MAXV); k.op(0x51); k.PUSH(SCO + c * 32); k.op(0x51); k.op(0x13); k.op(0x80);
        k.PUSH(SCO + c * 32); k.op(0x51); k.PUSH(MAXV); k.op(0x51); k.op(0x90); k.op(0x03); k.op(0x02);
        k.PUSH(MAXV); k.op(0x51); k.op(0x01); k.PUSH(MAXV); k.op(0x52);
        k.PUSH(MAXI); k.op(0x51); k.op(0x80); k.PUSH(c); k.op(0x03); k.op(0x82); k.op(0x02); k.op(0x01); k.PUSH(MAXI); k.op(0x52); k.op(0x50);
    }
    k.PUSH(MAXI); k.op(0x51); k.P1(0); k.op(0x52); k.PUSH(0x20); k.P1(0); k.op(0xF3);
    const long long predMS = ((k.peakB + 31) / 32) * 32;
    const long long predWarm = k.g + memgas(k.peakB);
    std::vector<uint8_t> codeU = k.c;
    std::vector<uint8_t> codeUm(codeU.begin(), codeU.end() - 13);
    { uint8_t t[] = {0x59, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xF3}; for (uint8_t x : t) codeUm.push_back(x); }

    // ================= N-deployable (looped loader, batch loop, looped neuron loop) =================
    std::vector<uint8_t> code;
    auto B_ = [&](uint8_t x) { code.push_back(x); };
    auto P1 = [&](uint8_t v) { B_(0x60); B_(v); };
    auto P3 = [&](int v) { B_(0x62); B_((v >> 16) & 0xFF); B_((v >> 8) & 0xFF); B_(v & 0xFF); };
    auto PUSH = [&](int v) { if (v < 256) P1((uint8_t)v); else if (v < 65536) { B_(0x61); B_((v >> 8) & 0xFF); B_(v & 0xFF); } else P3(v); };
    auto P4 = [&](uint32_t v) { B_(0x63); B_((v >> 24) & 0xFF); B_((v >> 16) & 0xFF); B_((v >> 8) & 0xFF); B_(v & 0xFF); };
    const uint8_t xgtcfg = (1 << 5) | 2;
    auto patch3 = [&](int ph, int target) { code[ph] = (target >> 16) & 0xFF; code[ph + 1] = (target >> 8) & 0xFF; code[ph + 2] = target & 0xFF; };
    auto emitLoopLoader = [&]() {
        for (auto& sg : segs) {
            P1(0); PUSH(GCNT); B_(0x52);
            int lp = (int)code.size(); B_(0x5B);
            PUSH(GCNT); B_(0x51); PUSH(sg.cnt); B_(0x11); B_(0x15); P3(0); int eph = (int)code.size() - 3; B_(0x57);
            PUSH(GCNT); B_(0x51); PUSH(sg.s0); B_(0x01); B_(0x54);
            PUSH(GCNT); B_(0x51); P1(32); B_(0x02); PUSH(sg.base); B_(0x01); B_(0x52);
            PUSH(GCNT); B_(0x51); P1(1); B_(0x01); PUSH(GCNT); B_(0x52);
            PUSH(lp); B_(0x56);
            int ed = (int)code.size(); B_(0x5B); patch3(eph, ed);
        }
    };
    auto looped_layer = [&](int M, int NGin, int GINP, int GWGT, int GBIA, int GOUT, int SHIFT) {
        for (int s = 0; s < M / 8; ++s) { P1(0); PUSH(SACC + s * 32); B_(0x52); }
        P1(0); PUSH(NCNT); B_(0x52);
        int outer = (int)code.size(); B_(0x5B);
        PUSH(NCNT); B_(0x51); PUSH(M); B_(0x11); B_(0x15); P3(0); int oe_ph = (int)code.size() - 3; B_(0x57);
        PUSH(NCNT); B_(0x51); P1(32); B_(0x02); PUSH(GBIA); B_(0x01); B_(0x51);                 // [acc] = bias
        PUSH(GINP);                                                                               // [acc, xbase]
        PUSH(NCNT); B_(0x51); PUSH(NGin * 32); B_(0x02); PUSH(GWGT); B_(0x01);                   // [acc, xbase, wbase]
        PUSH(NGin); B_(0xE6);                                                                     // XMACC_N -> [acc']
        B_(0xCD);                                                                                 // XSUM32 -> [sum]
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x06); P1(32); B_(0x02); B_(0x1B);              // sum << (NCNT%8)*32
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(SACC); B_(0x01);  // [v, saccaddr]
        B_(0x80); B_(0x51); B_(0x90); B_(0x91); B_(0x17); B_(0x90); B_(0x52);                     // sacc |= v
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x06); P1(7); B_(0x14); B_(0x15); P3(0); int skip_ph = (int)code.size() - 3; B_(0x57);
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(SACC); B_(0x01); B_(0x51);
        B_(0xCC); B_((uint8_t)SHIFT); B_(0xCE);
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(GOUT); B_(0x01); B_(0x52);
        int skip = (int)code.size(); B_(0x5B); patch3(skip_ph, skip);
        PUSH(NCNT); B_(0x51); P1(1); B_(0x01); PUSH(NCNT); B_(0x52);
        PUSH(outer); B_(0x56);
        int oe = (int)code.size(); B_(0x5B); patch3(oe_ph, oe);
    };
    auto emitBody = [&]() {
        looped_layer(H1, I / 8, INP, W1M, B1M, H1OUT, SHIFT1);
        looped_layer(H2, H1 / 8, H1OUT, W2M, B2M, H2OUT, SHIFT2);
        P1(0); PUSH(NCNT); B_(0x52);
        int co = (int)code.size(); B_(0x5B);
        PUSH(NCNT); B_(0x51); P1(10); B_(0x11); B_(0x15); P3(0); int co_ph = (int)code.size() - 3; B_(0x57);
        P1(0); PUSH(H2OUT); PUSH(NCNT); B_(0x51); PUSH((H2 / 8) * 32); B_(0x02); PUSH(W3M); B_(0x01); PUSH(H2 / 8); B_(0xE6);
        B_(0xCD);
        PUSH(NCNT); B_(0x51); P1(32); B_(0x02); PUSH(SCO); B_(0x01); B_(0x52);
        PUSH(NCNT); B_(0x51); P1(1); B_(0x01); PUSH(NCNT); B_(0x52);
        PUSH(co); B_(0x56);
        int coe = (int)code.size(); B_(0x5B); patch3(co_ph, coe);
        PUSH(SCO); B_(0x51); PUSH(MAXV); B_(0x52); P1(0); PUSH(MAXI); B_(0x52); P1(1); PUSH(CNT2); B_(0x52);
        int am = (int)code.size(); B_(0x5B);
        PUSH(CNT2); B_(0x51); P1(10); B_(0x11); B_(0x15); P3(0); int amend_ph = (int)code.size() - 3; B_(0x57);
        PUSH(MAXV); B_(0x51); PUSH(CNT2); B_(0x51); P1(32); B_(0x02); PUSH(SCO); B_(0x01); B_(0x51);
        B_(0xCA); B_(xgtcfg); P4(0xFFFFFFFF); B_(0x16); B_(0x15); P3(0); int amskip_ph = (int)code.size() - 3; B_(0x57);
        PUSH(CNT2); B_(0x51); P1(32); B_(0x02); PUSH(SCO); B_(0x01); B_(0x51); PUSH(MAXV); B_(0x52);
        PUSH(CNT2); B_(0x51); PUSH(MAXI); B_(0x52);
        int amskip = (int)code.size(); B_(0x5B); patch3(amskip_ph, amskip);
        PUSH(CNT2); B_(0x51); P1(1); B_(0x01); PUSH(CNT2); B_(0x52);
        PUSH(am); B_(0x56);
        int amend = (int)code.size(); B_(0x5B); patch3(amend_ph, amend);
    };
    code.clear();
    emitLoopLoader();
    B_(0x36); PUSH(CDIMG); B_(0x90); B_(0x04); PUSH(NIMG); B_(0x52);          // NIMG = CALLDATASIZE / 3136
    P1(0); PUSH(ICNT); B_(0x52);
    int lp = (int)code.size(); B_(0x5B);
    PUSH(ICNT); B_(0x51); PUSH(NIMG); B_(0x51); B_(0x11); B_(0x15); P3(0); int eph = (int)code.size() - 3; B_(0x57);
    PUSH(CDIMG); PUSH(ICNT); B_(0x51); PUSH(CDIMG); B_(0x02); PUSH(INP); B_(0x37);
    emitBody();
    PUSH(MAXI); B_(0x51); PUSH(ICNT); B_(0x51); P1(32); B_(0x02); PUSH(PREDS); B_(0x01); B_(0x52);
    PUSH(ICNT); B_(0x51); P1(1); B_(0x01); PUSH(ICNT); B_(0x52);
    PUSH(lp); B_(0x56);
    int ed = (int)code.size(); B_(0x5B); patch3(eph, ed);
    PUSH(NIMG); B_(0x51); P1(32); B_(0x02); PUSH(PREDS); B_(0xF3);
    std::vector<uint8_t> codeD = code;
    std::vector<uint8_t> codeDm = codeD; { for (int t = 0; t < 13; ++t) codeDm.pop_back(); uint8_t t2[] = {0x59, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xF3}; for (uint8_t x : t2) codeDm.push_back(x); }

    printf("Bytecode: N-UNROLLED=%zu B | N-DEPLOYABLE=%zu B | matWords=%d | XMACC_N priced at 11 gas per group\n", codeU.size(), codeD.size(), matWords);
    printf("[NOTE] Static prediction of the N-unrolled warm run: %lld gas (loader %lld, expected MSIZE %lld)\n\n", predWarm, g_loader, predMS);
    auto imgWords = [&](int im, std::vector<uint8_t>& dst) {
        for (int gw = 0; gw < I / 8; ++gw) { std::vector<uint8_t> w(32, 0); for (int l = 0; l < 8; ++l) { int val = imgs[im][gw * 8 + l]; uint16_t u = (uint16_t)(int16_t)val; w[l * 2] = u & 0xFF; w[l * 2 + 1] = (u >> 8) & 0xFF; } for (int i = 0; i < 32; ++i) dst.push_back(w[31 - i]); }
    };

#ifdef LOCAL_TEST
    auto dump = [](const char* name, const std::vector<uint8_t>& v) { std::ofstream f(name, std::ios::binary); f.write((const char*)v.data(), (std::streamsize)v.size()); };
    dump("rm_codeU.bin", codeU); dump("rm_codeUm.bin", codeUm); dump("rm_codeD.bin", codeD); dump("rm_codeDm.bin", codeDm);
    { std::ofstream f("rm_storage.hex"); for (int sl = 0; sl < matWords; ++sl) { for (int b = 0; b < 32; ++b) { char t[3]; std::snprintf(t, 3, "%02x", WLOAD[sl].first[31 - b]); f << t; } f << "\n"; } }
    { std::ofstream f("rm_expected.txt"); f << predWarm << " " << predMS << " " << matWords << " " << MACS << " " << SCO << " " << PREDS << "\n";
      for (int im = 0; im < std::min(N_RUN, 20); ++im) { std::vector<long long> sc; int p = cppScores(im, sc); f << p; for (auto v : sc) f << " " << v; f << "\n";
          std::vector<uint8_t> cd; imgWords(im, cd); dump(("rm_calldata_" + std::to_string(im) + ".bin").c_str(), cd); } }
    printf("LOCAL_TEST: bytecode, storage, calldata and expected values written.\n");
    (void)cppPred; (void)g_l1;
    return 0;
#else
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::cerr << "libevmone not found\n"; return 1; }
    auto create = (evmc_vm * (*)())dlsym(h, "evmc_create_evmone");
    evmc_vm* vm = create();
    evmc::address ADDR{}; ADDR.bytes[18] = 0xc0; ADDR.bytes[19] = 0xde;
    evmc::MockedHost host;
    auto keyOf = [](int sl) { evmc::bytes32 key{}; key.bytes[28] = (sl >> 24) & 0xFF; key.bytes[29] = (sl >> 16) & 0xFF; key.bytes[30] = (sl >> 8) & 0xFF; key.bytes[31] = sl & 0xFF; return key; };
    for (int sl = 0; sl < matWords; ++sl) { evmc::bytes32 val{}; for (int b = 0; b < 32; ++b) val.bytes[b] = WLOAD[sl].first[31 - b]; host.accounts[ADDR].storage[keyOf(sl)] = evmc::StorageValue{val}; }
    auto setAcc = [&](evmc_access_status st) { for (int sl = 0; sl < matWords; ++sl) host.accounts[ADDR].storage[keyOf(sl)].access_status = st; };
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();
    const int64_t GBIG = 200000000000LL;
    auto run = [&](std::vector<uint8_t>& kk, std::vector<uint8_t>& cd, std::vector<int>& preds, long long& w0) -> long long {
        evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GBIG; msg.recipient = ADDR; msg.code_address = ADDR;
        msg.input_data = cd.data(); msg.input_size = cd.size();
        auto res = vm->execute(vm, &IF, host.to_context(), EVMC_SHANGHAI, &msg, kk.data(), kk.size());
        long long g = GBIG - res.gas_left; preds.clear(); w0 = -1;
        if (res.status_code == 0 && res.output_data && res.output_size >= 32) {
            long long m = 0; for (int b = 24; b < 32; ++b) m = (m << 8) | res.output_data[b]; w0 = m;
            int n = (int)(res.output_size / 32);
            for (int i = 0; i < n; ++i) { int p = ((int32_t)res.output_data[i * 32 + 28] << 24) | ((int32_t)res.output_data[i * 32 + 29] << 16) | ((int32_t)res.output_data[i * 32 + 30] << 8) | (int32_t)res.output_data[i * 32 + 31]; preds.push_back(p); }
        } else if (res.status_code != 0) printf("  [EXECUTE ERROR] status=%d\n", (int)res.status_code);
        if (res.release) res.release(&res);
        return g;
    };

    // CHECK 1: 20 images, C++ == U == D(1)
    int ok = 0; long long w0; std::vector<int> pr;
    for (int im = 0; im < std::min(N_RUN, 20); ++im) {
        std::vector<uint8_t> cd; imgWords(im, cd); int cp = cppPred(im);
        setAcc(EVMC_ACCESS_WARM); run(codeU, cd, pr, w0); int pU = pr.empty() ? -1 : pr[0];
        setAcc(EVMC_ACCESS_WARM); run(codeD, cd, pr, w0); int pD = pr.empty() ? -1 : pr[0];
        if (pU == cp && pD == cp) ok++; else printf("  [MISMATCH] im=%d C++=%d N-unrolled=%d N-deployable=%d\n", im, cp, pU, pD);
    }
    printf("[CHECK 1] 20 images, C++ == N-unrolled == N-deployable: %d/20 %s\n", ok, ok == 20 ? "PASSED" : "FAILED");
    if (ok != 20) { printf("*** CHECK 1 FAILED -> no measurement ***\n"); vm->destroy(vm); dlclose(h); return 1; }
    bool fail = false;

    // N-unrolled: prediction, cold/warm, repeat
    std::vector<uint8_t> cd0; imgWords(0, cd0);
    long long wU, cU1, cU2, msU;
    setAcc(EVMC_ACCESS_WARM); wU = run(codeU, cd0, pr, w0);
    setAcc(EVMC_ACCESS_COLD); cU1 = run(codeU, cd0, pr, w0);
    setAcc(EVMC_ACCESS_COLD); cU2 = run(codeU, cd0, pr, w0);
    setAcc(EVMC_ACCESS_WARM); run(codeUm, cd0, pr, msU);
    printf("[CHECK 2] N-unrolled warm pred==meas: %lld/%lld diff=%lld %s | MSIZE pred/meas: %lld/%lld %s\n", predWarm, wU, wU - predWarm, wU == predWarm ? "PASSED" : "FAILED", predMS, msU, predMS == msU ? "OK" : "FAILED");
    if (wU != predWarm || predMS != msU) fail = true;
    long long memU = memgas(msU);
    long long hU_c = cU1 - (long long)matWords * 2100 - memU, hU_w = wU - (long long)matWords * 100 - memU;
    printf("[CHECK 3] cold-warm==%lld (N-unrolled): %lld %s\n", (long long)matWords * 2000, cU1 - wU, (cU1 - wU == (long long)matWords * 2000) ? "OK" : "FAILED");
    if (cU1 - wU != (long long)matWords * 2000) fail = true;
    printf("[CHECK 4] compute_cold==compute_warm (N-unrolled): %lld/%lld %s\n", hU_c, hU_w, hU_c == hU_w ? "PASSED" : "FAILED");
    if (hU_c != hU_w) fail = true;
    printf("[CHECK 5] EIP-170 (<24,576): N-deployable %zu B %s\n", codeD.size(), codeD.size() < 24576 ? "PASSED" : "FAILED");
    if (codeD.size() >= 24576) fail = true;
    printf("[CHECK 6] repeat run identical (N-unrolled cold): %s\n", cU1 == cU2 ? "PASSED" : "FAILED");
    if (cU1 != cU2) fail = true;

    // N-deployable: batch N = 1, 2, 4, 8, 16
    std::vector<int> Ns = {1, 2, 4, 8, 16};
    long long GD_c[5], GD_w[5], MSD[5];
    for (int ni = 0; ni < 5; ++ni) {
        int N = Ns[ni]; std::vector<uint8_t> cd; for (int r = 0; r < N; ++r) imgWords(0, cd);
        setAcc(EVMC_ACCESS_COLD); GD_c[ni] = run(codeD, cd, pr, w0);
        setAcc(EVMC_ACCESS_COLD); long long r2 = run(codeD, cd, pr, w0);
        if (GD_c[ni] != r2) { printf("[CHECK 7] N=%d repeat MISMATCH\n", N); fail = true; }
        setAcc(EVMC_ACCESS_WARM); GD_w[ni] = run(codeD, cd, pr, w0);
        setAcc(EVMC_ACCESS_WARM); run(codeDm, cd, pr, MSD[ni]);
        long long expMS = (((long long)PREDS + N * 32 + 31) / 32) * 32;
        bool g3 = (GD_c[ni] - GD_w[ni] == (long long)matWords * 2000), g5 = (MSD[ni] == expMS);
        printf("[N=%2d] N-deployable cold/warm = %lld/%lld | MSIZE %lld (expected %lld) | cold-warm %s | MSIZE %s\n", N, GD_c[ni], GD_w[ni], MSD[ni], expMS, g3 ? "OK" : "FAILED", g5 ? "OK" : "FAILED");
        if (!g3 || !g5) fail = true;
    }
    long long D[5]; for (int i = 0; i < 5; ++i) D[i] = GD_w[i] - memgas(MSD[i]);
    long long m01 = D[1] - D[0], m12 = (D[2] - D[1]) / 2, m23 = (D[3] - D[2]) / 4, m34 = (D[4] - D[3]) / 8;
    bool lin = (m01 == m12 && m12 == m23 && m23 == m34);
    printf("[CHECK 7] N-deployable marginal per image (memory-stripped): %lld %lld %lld %lld -> %s\n", m01, m12, m23, m34, lin ? "EXACTLY CONSTANT" : "FAILED");
    if (!lin) fail = true;
    long long memD1 = memgas(MSD[0]);
    long long hD_w = GD_w[0] - (long long)matWords * 100 - memD1;

    printf("\n================ ROW-LEVEL INSTRUCTION (Path N) ================\n");
    printf("  N-unrolled pure compute (single image): %lld  (loader %lld, layer-1 share ~%lld)\n", hU_w, g_loader - (long long)matWords * 100, g_l1 - (long long)matWords * 100 - 3 - 3 - 3 - 3 - 3 * (CDIMG / 32));
    printf("  per multiply-accumulate (%lld MACs): %.2f gas including the loader | %.2f gas without the loader\n", MACS, (double)hU_w / MACS, (double)(hU_w - (g_loader - (long long)matWords * 100)) / MACS);
    printf("  anchors (unrolled, single image): Path C 10,408,685 | Path B 446,833 | Path S 438,926\n");
    printf("  RATIOS (unrolled): C/N = %.2fx | B/N = %.2fx | S/N = %.2fx\n", 10408685.0 / hU_w, 446833.0 / hU_w, 438926.0 / hU_w);
    printf("  N-deployable: code %zu B | single image cold %lld, warm %lld (looped loader; compute %lld)\n", codeD.size(), GD_c[0], GD_w[0], hD_w);
    printf("  N-deployable marginal per image (memory-stripped): %lld\n", m01);
    printf("  anchors (deployable, marginal): Path B looped 1,992,849 | scalar looped 20,038,957\n");
    printf("  RATIOS (deployable, marginal): B/N = %.2fx | scalar/N = %.2fx\n", 1992849.0 / m01, 20038957.0 / m01);
    printf("  loop overhead of the deployable variant: marginal %lld / unrolled per-image compute %lld = %.2fx\n", m01, hU_w - (g_loader - (long long)matWords * 100), (double)m01 / (double)(hU_w - (g_loader - (long long)matWords * 100)));
    if (fail) printf("\n*** AT LEAST ONE CHECK FAILED -> numbers are UNRELIABLE, do NOT record ***\n");
    else printf("\n*** ALL CHECKS PASSED -> numbers are valid ***\n");
    vm->destroy(vm); dlclose(h); return fail ? 1 : 0;
#endif
}
