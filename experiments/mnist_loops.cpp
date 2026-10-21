// mnist_loops.cpp — deployable contracts with a loop per NEURON and an unrolled inner body, for the
// fixed-width path (Path B, XMACC per group) and for the packed-multiplication scalar baseline (Path S).
//
// The deployable contracts of mnist_deploy_cb (Path B) and mnist_e2e_cb (scalar) loop once per group of
// eight elements, so their marginal cost per image carries a jump and a counter for every group. The two
// contracts here keep the loop structure of the row-level contract of mnist_rowmacc (batch loop, loop per
// neuron) but unroll the groups of one neuron inside the loop body, which is the natural way to write a
// deployable contract without a row-level instruction. They are compared, per image and without memory,
// with the row-level contract (marginal 199,911) and with the per-group loops (Path B 1,992,849,
// byte-extraction scalar 20,038,957).
//   B2  fixed-width XMACC, memory map and looped segment loader of mnist_rowmacc, 98/16/8 groups per neuron
//   S2  Path S lanes (layer 1: 8 x 32-bit, window 32; layers 2-3: 7 x 36-bit, window 4), memory map and
//       storage layout of mnist_swar_baseline, corrections once per image, packed outputs rebuilt per image
// CHECKS: (1) 20 images, C++ reference == B2 == S2 (2) cold-warm == slots*2000 at every batch size (3) MSIZE ==
// expected (4) repeat runs identical (5) code sizes under 24,576 (6) marginal cost per image constant over
// N = 1, 2, 4, 8, 16 for both contracts.
// Compile with -DLOCAL_TEST to dump the bytecode, storage and calldata instead of running evmone.
#ifndef LOCAL_TEST
#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Word = std::array<uint8_t, 32>;   // big-endian 256-bit value, as stored in storage and memory
static void or_bits(Word& w, int bitpos, uint64_t v, int nbits) { for (int b = 0; b < nbits; ++b) if ((v >> b) & 1ULL) { int bit = bitpos + b; w[31 - bit / 8] |= (uint8_t)(1u << (bit % 8)); } }
static Word word_i256(long long v) { Word w{}; uint64_t u = (uint64_t)v; for (int b = 0; b < 8; ++b) w[31 - b] = (uint8_t)((u >> (8 * b)) & 0xFF); if (v < 0) for (int i = 0; i < 24; ++i) w[i] = 0xFF; return w; }
static Word word_lanes16(const int* v, int n) { Word w{}; for (int l = 0; l < n; ++l) { uint16_t u = (uint16_t)(int16_t)v[l]; w[31 - 2 * l] = u & 0xFF; w[30 - 2 * l] = (u >> 8) & 0xFF; } return w; }
static Word word_lane32(long long v) { Word w{}; uint32_t u = (uint32_t)(int32_t)v; for (int b = 0; b < 4; ++b) w[31 - b] = (u >> (8 * b)) & 0xFF; return w; }
static long long memgas(long long bytes) { long long a = (bytes + 31) / 32; return 3 * a + (a * a) / 512; }

struct Gen {   // bytecode builder with forward-jump patching
    std::vector<uint8_t> c;
    void B(uint8_t x) { c.push_back(x); }
    void P1(uint8_t v) { B(0x60); B(v); }
    void P3(int v) { B(0x62); B((v >> 16) & 0xFF); B((v >> 8) & 0xFF); B(v & 0xFF); }
    void PUSH(long long v) { if (v < 256) { P1((uint8_t)v); return; } int n = 0; for (long long t = v; t > 0; t >>= 8) ++n; B((uint8_t)(0x5F + n)); for (int k = n - 1; k >= 0; --k) B((uint8_t)((v >> (8 * k)) & 0xFF)); }
    void P4(uint32_t v) { B(0x63); B((v >> 24) & 0xFF); B((v >> 16) & 0xFF); B((v >> 8) & 0xFF); B(v & 0xFF); }
    void W32(const Word& w) { B(0x7F); for (int i = 0; i < 32; ++i) B(w[i]); }
    void patch3(int ph, int target) { c[ph] = (target >> 16) & 0xFF; c[ph + 1] = (target >> 8) & 0xFF; c[ph + 2] = target & 0xFF; }
    void MLOAD() { B(0x51); } void MSTORE() { B(0x52); } void ADD() { B(0x01); } void MUL() { B(0x02); } void SUB() { B(0x03); }
    void DIV() { B(0x04); } void MOD() { B(0x06); } void SHR() { B(0x1C); } void SHL() { B(0x1B); } void AND() { B(0x16); } void OR() { B(0x17); }
    void DUP(int n) { B((uint8_t)(0x7F + n)); } void SWAP(int n) { B((uint8_t)(0x8F + n)); } void POP() { B(0x50); }
    void SGT() { B(0x13); } void ISZERO() { B(0x15); } void GT() { B(0x11); } void EQ() { B(0x14); }
    // [counter word at CNT] loop head: CNT := 0; label; if CNT >= limit goto end (returns placeholder index and label)
    struct Loop { int label, endph; };
    Loop loopHead(int CNT, int limit) { P1(0); PUSH(CNT); MSTORE(); Loop L; L.label = (int)c.size(); B(0x5B); PUSH(CNT); MLOAD(); PUSH(limit); GT(); ISZERO(); P3(0); L.endph = (int)c.size() - 3; B(0x57); return L; }
    void loopTail(int CNT, const Loop& L) { PUSH(CNT); MLOAD(); P1(1); ADD(); PUSH(CNT); MSTORE(); PUSH(L.label); B(0x56); int e = (int)c.size(); B(0x5B); patch3(L.endph, e); }
    void CNTx(int CNT, long long m, long long base) { PUSH(CNT); MLOAD(); PUSH(m); MUL(); PUSH(base); ADD(); }   // -> [base + CNT*m]
};
struct Seg { int s0, cnt, base; };
static std::vector<Seg> segments(const std::vector<int>& addr) { std::vector<Seg> s; for (int k = 0; k < (int)addr.size(); ++k) { if (s.empty() || addr[k] != addr[k - 1] + 32) s.push_back({k, 1, addr[k]}); else s.back().cnt++; } return s; }
static void emitLoopLoader(Gen& g, const std::vector<Seg>& segs, int GCNT) {   // slot k = s0 + i of each segment -> base + 32 i
    for (auto& sg : segs) {
        Gen::Loop L = g.loopHead(GCNT, sg.cnt);
        g.PUSH(GCNT); g.MLOAD(); g.PUSH(sg.s0); g.ADD(); g.B(0x54);
        g.CNTx(GCNT, 32, sg.base); g.MSTORE();
        g.loopTail(GCNT, L);
    }
}

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
    auto cppPred = [&](int im) -> int {
        std::vector<int>& g = imgs[im];
        std::vector<int> h1r(H1); for (int n = 0; n < H1; ++n) { long long s = b1[n]; for (int i = 0; i < I; ++i) s += (long long)g[i] * W1[i * H1 + n]; h1r[n] = rss(s, SHIFT1); }
        std::vector<int> h2r(H2); for (int n = 0; n < H2; ++n) { long long s = b2[n]; for (int i = 0; i < H1; ++i) s += (long long)h1r[i] * W2[i * H2 + n]; h2r[n] = rss(s, SHIFT2); }
        std::vector<long long> sc(10); for (int c = 0; c < 10; ++c) { long long s = 0; for (int i = 0; i < H2; ++i) s += (long long)h2r[i] * W3[i * 10 + c]; sc[c] = s; }
        return (int)(std::max_element(sc.begin(), sc.end()) - sc.begin());
    };
    const uint8_t xgtcfg = (1 << 5) | 2;

    // ======================= B2: fixed-width XMACC, loop per neuron, unrolled groups =======================
    const int INP = 0, W1M = 10000, B1M = 420000, H1OUT = 430000, W2M = 440000, B2M = 480000, H2OUT = 490000, W3M = 495000, SCO = 500000;
    const int SACC = 900100, NCNT = 901000, GCNT = 901100, MAXV = 900200, MAXI = 900232, CNT2 = 900264, NIMG = 901232, ICNT = 901264, PREDS = 901296;
    std::vector<Word> BSLOT; std::vector<int> BADDR;
    for (int n = 0; n < H1; ++n) for (int gw = 0; gw < I / 8; ++gw) { int v[8]; for (int l = 0; l < 8; ++l) v[l] = W1[(gw * 8 + l) * H1 + n]; BSLOT.push_back(word_lanes16(v, 8)); BADDR.push_back(W1M + (n * (I / 8) + gw) * 32); }
    for (int n = 0; n < H1; ++n) { BSLOT.push_back(word_lane32(b1[n])); BADDR.push_back(B1M + n * 32); }
    for (int n = 0; n < H2; ++n) for (int gw = 0; gw < H1 / 8; ++gw) { int v[8]; for (int l = 0; l < 8; ++l) v[l] = W2[(gw * 8 + l) * H2 + n]; BSLOT.push_back(word_lanes16(v, 8)); BADDR.push_back(W2M + (n * (H1 / 8) + gw) * 32); }
    for (int n = 0; n < H2; ++n) { BSLOT.push_back(word_lane32(b2[n])); BADDR.push_back(B2M + n * 32); }
    for (int c = 0; c < 10; ++c) for (int gw = 0; gw < H2 / 8; ++gw) { int v[8]; for (int l = 0; l < 8; ++l) v[l] = W3[(gw * 8 + l) * 10 + c]; BSLOT.push_back(word_lanes16(v, 8)); BADDR.push_back(W3M + (c * (H2 / 8) + gw) * 32); }
    const int bslots = (int)BSLOT.size();
    Gen gb;
    emitLoopLoader(gb, segments(BADDR), GCNT);
    gb.B(0x36); gb.PUSH(CDIMG); gb.SWAP(1); gb.DIV(); gb.PUSH(NIMG); gb.MSTORE();          // NIMG = CALLDATASIZE / 3136
    Gen::Loop LB; gb.P1(0); gb.PUSH(ICNT); gb.MSTORE();                                      // batch loop: ICNT < MLOAD(NIMG)
    { int lbl = (int)gb.c.size(); gb.B(0x5B); gb.PUSH(ICNT); gb.MLOAD(); gb.PUSH(NIMG); gb.MLOAD(); gb.GT(); gb.ISZERO(); gb.P3(0); LB.endph = (int)gb.c.size() - 3; gb.B(0x57); LB.label = lbl; }
    gb.PUSH(CDIMG); gb.CNTx(ICNT, CDIMG, 0); gb.PUSH(INP); gb.B(0x37);                      // CALLDATACOPY(INP, ICNT*3136, 3136)
    auto layerB2 = [&](int M, int NGin, int GINP, int GWGT, int GBIA, int GOUT, int SHIFT) {
        for (int s = 0; s < M / 8; ++s) { gb.P1(0); gb.PUSH(SACC + s * 32); gb.MSTORE(); }
        Gen::Loop L = gb.loopHead(NCNT, M);
        gb.CNTx(NCNT, NGin * 32, GWGT);                                                       // [wbase]
        gb.CNTx(NCNT, 32, GBIA); gb.MLOAD();                                                  // [wbase, acc = bias]
        for (int g = 0; g < NGin; ++g) {
            gb.PUSH(GINP + g * 32); gb.MLOAD();                                               // [wbase, acc, x]
            gb.DUP(3); if (g) { gb.PUSH(g * 32); gb.ADD(); } gb.MLOAD();                      // [wbase, acc, x, w]
            gb.B(0xCB);                                                                       // XMACC -> [wbase, acc']
        }
        gb.B(0xCD); gb.SWAP(1); gb.POP();                                                     // XSUM32 -> [sum]
        gb.PUSH(NCNT); gb.MLOAD(); gb.P1(8); gb.SWAP(1); gb.MOD(); gb.P1(32); gb.MUL(); gb.SHL();          // sum << (NCNT%8)*32
        gb.PUSH(NCNT); gb.MLOAD(); gb.P1(8); gb.SWAP(1); gb.DIV(); gb.P1(32); gb.MUL(); gb.PUSH(SACC); gb.ADD();   // [v, saccaddr]
        gb.DUP(1); gb.MLOAD(); gb.SWAP(1); gb.SWAP(2); gb.OR(); gb.SWAP(1); gb.MSTORE();     // sacc |= v
        gb.PUSH(NCNT); gb.MLOAD(); gb.P1(8); gb.SWAP(1); gb.MOD(); gb.P1(7); gb.EQ(); gb.ISZERO(); gb.P3(0); int skip_ph = (int)gb.c.size() - 3; gb.B(0x57);
        gb.PUSH(NCNT); gb.MLOAD(); gb.P1(8); gb.SWAP(1); gb.DIV(); gb.P1(32); gb.MUL(); gb.PUSH(SACC); gb.ADD(); gb.MLOAD();
        gb.B(0xCC); gb.B((uint8_t)SHIFT); gb.B(0xCE);
        gb.PUSH(NCNT); gb.MLOAD(); gb.P1(8); gb.SWAP(1); gb.DIV(); gb.P1(32); gb.MUL(); gb.PUSH(GOUT); gb.ADD(); gb.MSTORE();
        int skip = (int)gb.c.size(); gb.B(0x5B); gb.patch3(skip_ph, skip);
        gb.loopTail(NCNT, L);
    };
    layerB2(H1, I / 8, INP, W1M, B1M, H1OUT, SHIFT1);
    layerB2(H2, H1 / 8, H1OUT, W2M, B2M, H2OUT, SHIFT2);
    {   // layer 3: 10 classes, loop per class, 8 groups unrolled, signed score words
        Gen::Loop L = gb.loopHead(NCNT, 10);
        gb.CNTx(NCNT, (H2 / 8) * 32, W3M); gb.P1(0);                                          // [wbase, acc = 0]
        for (int g = 0; g < H2 / 8; ++g) { gb.PUSH(H2OUT + g * 32); gb.MLOAD(); gb.DUP(3); if (g) { gb.PUSH(g * 32); gb.ADD(); } gb.MLOAD(); gb.B(0xCB); }
        gb.B(0xCD); gb.SWAP(1); gb.POP(); gb.P1(3); gb.B(0x0B);                               // XSUM32, SIGNEXTEND(3)
        gb.CNTx(NCNT, 32, SCO); gb.MSTORE();
        gb.loopTail(NCNT, L);
    }
    auto argmaxB = [&]() {   // same as mnist_rowmacc (XGT_IMM, first maximum wins)
        gb.PUSH(SCO); gb.MLOAD(); gb.PUSH(MAXV); gb.MSTORE(); gb.P1(0); gb.PUSH(MAXI); gb.MSTORE(); gb.P1(1); gb.PUSH(CNT2); gb.MSTORE();
        int am = (int)gb.c.size(); gb.B(0x5B);
        gb.PUSH(CNT2); gb.MLOAD(); gb.P1(10); gb.GT(); gb.ISZERO(); gb.P3(0); int amend_ph = (int)gb.c.size() - 3; gb.B(0x57);
        gb.PUSH(MAXV); gb.MLOAD(); gb.CNTx(CNT2, 32, SCO); gb.MLOAD();
        gb.B(0xCA); gb.B(xgtcfg); gb.P4(0xFFFFFFFF); gb.AND(); gb.ISZERO(); gb.P3(0); int amskip_ph = (int)gb.c.size() - 3; gb.B(0x57);
        gb.CNTx(CNT2, 32, SCO); gb.MLOAD(); gb.PUSH(MAXV); gb.MSTORE();
        gb.PUSH(CNT2); gb.MLOAD(); gb.PUSH(MAXI); gb.MSTORE();
        int amskip = (int)gb.c.size(); gb.B(0x5B); gb.patch3(amskip_ph, amskip);
        gb.PUSH(CNT2); gb.MLOAD(); gb.P1(1); gb.ADD(); gb.PUSH(CNT2); gb.MSTORE();
        gb.PUSH(am); gb.B(0x56);
        int amend = (int)gb.c.size(); gb.B(0x5B); gb.patch3(amend_ph, amend);
    };
    argmaxB();
    gb.PUSH(MAXI); gb.MLOAD(); gb.CNTx(ICNT, 32, PREDS); gb.MSTORE();
    gb.loopTail(ICNT, LB);
    gb.PUSH(NIMG); gb.MLOAD(); gb.P1(32); gb.MUL(); gb.PUSH(PREDS); gb.B(0xF3);
    std::vector<uint8_t> codeB = gb.c;
    std::vector<uint8_t> codeBm = codeB; { for (int t = 0; t < 13; ++t) codeBm.pop_back(); uint8_t t2[] = {0x59, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xF3}; for (uint8_t x : t2) codeBm.push_back(x); }
    const int PREDS_B = PREDS;

    // ======================= S2: packed-multiplication scalar, loop per neuron, unrolled groups =======================
    const int OFF = 32768;
    const int L1 = 8, LW1 = 32, G1 = I / L1, L2 = 7, LW2 = 36, G2 = (H1 + L2 - 1) / L2, L3 = 7, G3 = (H2 + L3 - 1) / L3, WIN1 = 32, WIN2 = 4;
    const int XIN = 0, W1S = XIN + G1 * 32, B1S = W1S + H1 * G1 * 32, W2S = B1S + H1 * 32, B2S = W2S + H2 * G2 * 32, W3S = B2S + H2 * 32;
    const int X2 = W3S + 10 * G3 * 32, X3 = X2 + G2 * 32, CORR = X3 + G3 * 32, SCOS = CORR + 3 * 32, SMAXV = SCOS + 10 * 32, SMAXI = SMAXV + 32;
    const int SCNT = SMAXI + 32, SGCNT = SCNT + 32, SNIMG = SGCNT + 32, SICNT = SNIMG + 32, SCNT2 = SICNT + 32, SPREDS = SCNT2 + 32;
    std::vector<Word> SSLOT; std::vector<int> SADDR;
    for (int n = 0; n < H1; ++n) for (int g = 0; g < G1; ++g) { Word w{}; for (int k = 0; k < L1; ++k) or_bits(w, LW1 * k, (uint64_t)(W1[(g * L1 + k) * H1 + n] + OFF), LW1); SSLOT.push_back(w); SADDR.push_back(W1S + (n * G1 + g) * 32); }
    for (int n = 0; n < H1; ++n) { SSLOT.push_back(word_i256(b1[n])); SADDR.push_back(B1S + n * 32); }
    for (int n = 0; n < H2; ++n) for (int g = 0; g < G2; ++g) { Word w{}; for (int k = 0; k < L2; ++k) { int i = g * L2 + k; if (i < H1) or_bits(w, LW2 * k, (uint64_t)(W2[i * H2 + n] + OFF), LW2); } SSLOT.push_back(w); SADDR.push_back(W2S + (n * G2 + g) * 32); }
    for (int n = 0; n < H2; ++n) { SSLOT.push_back(word_i256(b2[n])); SADDR.push_back(B2S + n * 32); }
    for (int c = 0; c < 10; ++c) for (int g = 0; g < G3; ++g) { Word w{}; for (int k = 0; k < L3; ++k) { int i = g * L3 + k; if (i < H2) or_bits(w, LW2 * k, (uint64_t)(W3[i * 10 + c] + OFF), LW2); } SSLOT.push_back(w); SADDR.push_back(W3S + (c * G3 + g) * 32); }
    const int sslots = (int)SSLOT.size();
    Word ONES8{}; for (int k = 0; k < L1; ++k) or_bits(ONES8, LW1 * k, 1, LW1);
    Word ONES7{}; for (int k = 0; k < L2; ++k) or_bits(ONES7, LW2 * k, 1, LW2);
    Gen gs;
    auto MASK36 = [&]() { gs.PUSH((1LL << 36) - 1); };
    emitLoopLoader(gs, segments(SADDR), SGCNT);
    gs.B(0x36); gs.PUSH(CDIMG); gs.SWAP(1); gs.DIV(); gs.PUSH(SNIMG); gs.MSTORE();
    Gen::Loop LS; gs.P1(0); gs.PUSH(SICNT); gs.MSTORE();
    { int lbl = (int)gs.c.size(); gs.B(0x5B); gs.PUSH(SICNT); gs.MLOAD(); gs.PUSH(SNIMG); gs.MLOAD(); gs.GT(); gs.ISZERO(); gs.P3(0); LS.endph = (int)gs.c.size() - 3; gs.B(0x57); LS.label = lbl; }
    gs.PUSH(CDIMG); gs.CNTx(SICNT, CDIMG, 0); gs.PUSH(XIN); gs.B(0x37);
    for (int g = 0; g < G2; ++g) { gs.P1(0); gs.PUSH(X2 + g * 32); gs.MSTORE(); }               // packed layer inputs are rebuilt per image
    for (int g = 0; g < G3; ++g) { gs.P1(0); gs.PUSH(X3 + g * 32); gs.MSTORE(); }
    auto emitCorrection = [&](int xbase, int groups, const Word& ones, int topbit, bool mask, int corrAddr) {
        gs.W32(ones); gs.P1(0);
        for (int g = 0; g < groups; ++g) { gs.PUSH(xbase + g * 32); gs.MLOAD(); gs.DUP(3); gs.MUL(); gs.ADD(); }
        gs.PUSH(topbit); gs.SHR(); if (mask) { MASK36(); gs.AND(); }
        gs.P1(15); gs.SHL(); gs.PUSH(corrAddr); gs.MSTORE(); gs.POP();
    };
    auto emitActivation = [&](int shift) {   // [z] -> [h], same idiom as the unrolled Path S
        gs.DUP(1); gs.P1(0); gs.SGT(); gs.ISZERO(); gs.MUL();
        gs.P1((uint8_t)shift); gs.SHR();
        gs.DUP(1); gs.PUSH(32767); gs.SWAP(1); gs.SGT();
        gs.PUSH(32767); gs.DUP(3); gs.SWAP(1); gs.SUB(); gs.MUL(); gs.ADD();
    };
    // one looped layer: wbase from the counter, dot product with window extraction, correction, bias, activation, lane packing
    auto layerS2 = [&](int M, int wbase0, int groupsPerNeuron, int xbase, int groups, int window, int topbit, bool mask, int corrAddr, int biasBase, int shift, int outX, int outLanes) {
        Gen::Loop L = gs.loopHead(SCNT, M);
        gs.CNTx(SCNT, groupsPerNeuron * 32, wbase0);                                        // [wbase]
        gs.P1(0);                                                                             // [wbase, T]
        for (int g = 0; g < groups; ++g) {
            if (g % window == 0) gs.P1(0);                                                    // [wbase, T, P]
            gs.DUP(3); if (g) { gs.PUSH(g * 32); gs.ADD(); } gs.MLOAD(); gs.PUSH(xbase + g * 32); gs.MLOAD(); gs.MUL(); gs.ADD();
            if ((g + 1) % window == 0 || g == groups - 1) { gs.PUSH(topbit); gs.SHR(); if (mask) { MASK36(); gs.AND(); } gs.ADD(); }
        }
        gs.PUSH(corrAddr); gs.MLOAD(); gs.SWAP(1); gs.SUB();                                  // [wbase, T - corr]
        if (biasBase >= 0) { gs.CNTx(SCNT, 32, biasBase); gs.MLOAD(); gs.ADD(); }
        gs.SWAP(1); gs.POP();                                                                 // [z]
        if (shift >= 0) {
            emitActivation(shift);                                                            // [h]
            // lane = outLanes - 1 - SCNT % outLanes, shift by 36*lane; word = outX + (SCNT / outLanes)*32
            gs.PUSH(SCNT); gs.MLOAD(); gs.PUSH(outLanes); gs.SWAP(1); gs.MOD(); gs.PUSH(outLanes - 1); gs.SUB(); gs.PUSH(LW2); gs.MUL(); gs.SHL();
            gs.PUSH(SCNT); gs.MLOAD(); gs.PUSH(outLanes); gs.SWAP(1); gs.DIV(); gs.P1(32); gs.MUL(); gs.PUSH(outX); gs.ADD();   // [v, addr]
            gs.DUP(1); gs.MLOAD(); gs.SWAP(1); gs.SWAP(2); gs.OR(); gs.SWAP(1); gs.MSTORE();
        } else {
            gs.CNTx(SCNT, 32, outX); gs.MSTORE();                                             // score word
        }
        gs.loopTail(SCNT, L);
    };
    emitCorrection(XIN, G1, ONES8, 224, false, CORR);
    layerS2(H1, W1S, G1, XIN, G1, WIN1, 224, false, CORR, B1S, SHIFT1, X2, L2);
    emitCorrection(X2, G2, ONES7, 216, true, CORR + 32);
    layerS2(H2, W2S, G2, X2, G2, WIN2, 216, true, CORR + 32, B2S, SHIFT2, X3, L3);
    emitCorrection(X3, G3, ONES7, 216, true, CORR + 64);
    layerS2(10, W3S, G3, X3, G3, WIN2, 216, true, CORR + 64, -1, -1, SCOS, 0);
    // argmax: branchless, signed, first maximum wins (same idiom as the unrolled Path S)
    gs.PUSH(SCOS); gs.MLOAD(); gs.PUSH(SMAXV); gs.MSTORE(); gs.P1(0); gs.PUSH(SMAXI); gs.MSTORE();
    for (int c = 1; c < 10; ++c) {
        gs.PUSH(SMAXV); gs.MLOAD(); gs.PUSH(SCOS + c * 32); gs.MLOAD(); gs.SGT(); gs.DUP(1);
        gs.PUSH(SCOS + c * 32); gs.MLOAD(); gs.PUSH(SMAXV); gs.MLOAD(); gs.SWAP(1); gs.SUB(); gs.MUL();
        gs.PUSH(SMAXV); gs.MLOAD(); gs.ADD(); gs.PUSH(SMAXV); gs.MSTORE();
        gs.PUSH(SMAXI); gs.MLOAD(); gs.DUP(1); gs.PUSH(c); gs.SUB(); gs.DUP(3); gs.MUL(); gs.ADD(); gs.PUSH(SMAXI); gs.MSTORE(); gs.POP();
    }
    gs.PUSH(SMAXI); gs.MLOAD(); gs.CNTx(SICNT, 32, SPREDS); gs.MSTORE();
    gs.loopTail(SICNT, LS);
    gs.PUSH(SNIMG); gs.MLOAD(); gs.P1(32); gs.MUL(); gs.PUSH(SPREDS); gs.B(0xF3);
    std::vector<uint8_t> codeS = gs.c;
    std::vector<uint8_t> codeSm = codeS; { for (int t = 0; t < 13; ++t) codeSm.pop_back(); uint8_t t2[] = {0x59, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xF3}; for (uint8_t x : t2) codeSm.push_back(x); }
    (void)SCNT2;

    auto cdB = [&](int im, std::vector<uint8_t>& dst) { for (int gw = 0; gw < I / 8; ++gw) { int v[8]; for (int l = 0; l < 8; ++l) v[l] = imgs[im][gw * 8 + l]; Word w = word_lanes16(v, 8); dst.insert(dst.end(), w.begin(), w.end()); } };
    auto cdS = [&](int im, std::vector<uint8_t>& dst) { for (int g = 0; g < G1; ++g) { Word w{}; for (int k = 0; k < L1; ++k) or_bits(w, LW1 * (L1 - 1 - k), (uint64_t)imgs[im][g * L1 + k], LW1); dst.insert(dst.end(), w.begin(), w.end()); } };
    printf("Bytecode: B2 (XMACC, loop per neuron) %zu B | S2 (packed scalar, loop per neuron) %zu B | slots B %d, S %d\n\n", codeB.size(), codeS.size(), bslots, sslots);

#ifdef LOCAL_TEST
    auto dump = [](const char* name, const std::vector<uint8_t>& v) { std::ofstream f(name, std::ios::binary); f.write((const char*)v.data(), (std::streamsize)v.size()); };
    dump("lp_codeB.bin", codeB); dump("lp_codeBm.bin", codeBm); dump("lp_codeS.bin", codeS); dump("lp_codeSm.bin", codeSm);
    { std::ofstream f("lp_storageB.hex"); for (auto& w : BSLOT) { for (int i = 0; i < 32; ++i) { char t[3]; std::snprintf(t, 3, "%02x", w[i]); f << t; } f << "\n"; } }
    { std::ofstream f("lp_storageS.hex"); for (auto& w : SSLOT) { for (int i = 0; i < 32; ++i) { char t[3]; std::snprintf(t, 3, "%02x", w[i]); f << t; } f << "\n"; } }
    { std::ofstream f("lp_expected.txt"); f << bslots << " " << sslots << " " << PREDS_B << " " << SPREDS << "\n";
      for (int im = 0; im < std::min(N_RUN, 20); ++im) { f << cppPred(im) << "\n"; std::vector<uint8_t> a, b; cdB(im, a); cdS(im, b);
          dump(("lp_cdB_" + std::to_string(im) + ".bin").c_str(), a); dump(("lp_cdS_" + std::to_string(im) + ".bin").c_str(), b); } }
    printf("LOCAL_TEST: bytecode, storage, calldata and expected predictions written.\n");
    return 0;
#else
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::cerr << "libevmone not found\n"; return 1; }
    auto create = (evmc_vm * (*)())dlsym(h, "evmc_create_evmone");
    evmc_vm* vm = create();
    evmc::address ADDR_B{}; ADDR_B.bytes[18] = 0xc0; ADDR_B.bytes[19] = 0xde;
    evmc::address ADDR_S{}; ADDR_S.bytes[18] = 0xc0; ADDR_S.bytes[19] = 0xdf;
    evmc::MockedHost host;
    auto keyOf = [](int sl) { evmc::bytes32 key{}; key.bytes[28] = (sl >> 24) & 0xFF; key.bytes[29] = (sl >> 16) & 0xFF; key.bytes[30] = (sl >> 8) & 0xFF; key.bytes[31] = sl & 0xFF; return key; };
    for (int sl = 0; sl < bslots; ++sl) { evmc::bytes32 val{}; for (int b = 0; b < 32; ++b) val.bytes[b] = BSLOT[sl][b]; host.accounts[ADDR_B].storage[keyOf(sl)] = evmc::StorageValue{val}; }
    for (int sl = 0; sl < sslots; ++sl) { evmc::bytes32 val{}; for (int b = 0; b < 32; ++b) val.bytes[b] = SSLOT[sl][b]; host.accounts[ADDR_S].storage[keyOf(sl)] = evmc::StorageValue{val}; }
    auto setAcc = [&](const evmc::address& a, int n, evmc_access_status st) { for (int sl = 0; sl < n; ++sl) host.accounts[a].storage[keyOf(sl)].access_status = st; };
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();
    const int64_t GBIG = 200000000000LL;
    auto run = [&](const evmc::address& a, std::vector<uint8_t>& kk, std::vector<uint8_t>& cd, std::vector<int>& preds, long long& w0) -> long long {
        evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GBIG; msg.recipient = a; msg.code_address = a;
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
    int ok = 0; long long w0; std::vector<int> pr;
    for (int im = 0; im < std::min(N_RUN, 20); ++im) {
        std::vector<uint8_t> a, b; cdB(im, a); cdS(im, b); int cp = cppPred(im);
        setAcc(ADDR_B, bslots, EVMC_ACCESS_WARM); run(ADDR_B, codeB, a, pr, w0); int pB = pr.empty() ? -1 : pr[0];
        setAcc(ADDR_S, sslots, EVMC_ACCESS_WARM); run(ADDR_S, codeS, b, pr, w0); int pS = pr.empty() ? -1 : pr[0];
        if (pB == cp && pS == cp) ok++; else printf("  [MISMATCH] im=%d C++=%d B2=%d S2=%d\n", im, cp, pB, pS);
    }
    printf("[CHECK 1] 20 images, C++ == B2 == S2: %d/20 %s\n", ok, ok == 20 ? "PASSED" : "FAILED");
    if (ok != 20) { printf("*** CHECK 1 FAILED -> no measurement ***\n"); vm->destroy(vm); dlclose(h); return 1; }
    bool fail = false;
    printf("[CHECK 5] EIP-170 (<24,576): B2 %zu B, S2 %zu B %s\n", codeB.size(), codeS.size(), (codeB.size() < 24576 && codeS.size() < 24576) ? "PASSED" : "FAILED");
    if (codeB.size() >= 24576 || codeS.size() >= 24576) fail = true;
    std::vector<int> Ns = {1, 2, 4, 8, 16};
    long long mB = 0, mS = 0, warmB1 = 0, warmS1 = 0, coldB1 = 0, coldS1 = 0;
    for (int which = 0; which < 2; ++which) {
        const char* nm = which ? "S2" : "B2"; const evmc::address& a = which ? ADDR_S : ADDR_B; int slots = which ? sslots : bslots;
        std::vector<uint8_t>& kk = which ? codeS : codeB; std::vector<uint8_t>& km = which ? codeSm : codeBm; int PR = which ? SPREDS : PREDS_B;
        long long Gw[5], MS[5];
        for (int ni = 0; ni < 5; ++ni) {
            int N = Ns[ni]; std::vector<uint8_t> cd; for (int r = 0; r < N; ++r) { if (which) cdS(0, cd); else cdB(0, cd); }
            setAcc(a, slots, EVMC_ACCESS_COLD); long long c1 = run(a, kk, cd, pr, w0);
            setAcc(a, slots, EVMC_ACCESS_COLD); long long c2 = run(a, kk, cd, pr, w0);
            setAcc(a, slots, EVMC_ACCESS_WARM); Gw[ni] = run(a, kk, cd, pr, w0);
            setAcc(a, slots, EVMC_ACCESS_WARM); run(a, km, cd, pr, MS[ni]);
            long long expMS = (((long long)PR + N * 32 + 31) / 32) * 32;
            bool g2 = (c1 - Gw[ni] == (long long)slots * 2000), g3 = (MS[ni] == expMS), g4 = (c1 == c2);
            printf("[%s N=%2d] cold/warm = %lld/%lld | cold-warm %s | MSIZE %lld (expected %lld) %s | repeat %s\n", nm, N, c1, Gw[ni], g2 ? "OK" : "FAILED", MS[ni], expMS, g3 ? "OK" : "FAILED", g4 ? "identical" : "DIFFERENT");
            if (!g2 || !g3 || !g4) fail = true;
            if (ni == 0) { if (which) { coldS1 = c1; warmS1 = Gw[0]; } else { coldB1 = c1; warmB1 = Gw[0]; } }
        }
        long long D[5]; for (int i = 0; i < 5; ++i) D[i] = Gw[i] - memgas(MS[i]);
        long long m01 = D[1] - D[0], m12 = (D[2] - D[1]) / 2, m23 = (D[3] - D[2]) / 4, m34 = (D[4] - D[3]) / 8;
        bool lin = (m01 == m12 && m12 == m23 && m23 == m34);
        printf("[CHECK 6] %s marginal per image (memory-stripped): %lld %lld %lld %lld -> %s\n", nm, m01, m12, m23, m34, lin ? "EXACTLY CONSTANT" : "FAILED");
        if (!lin) fail = true;
        if (which) mS = m01; else mB = m01;
    }
    printf("\n================ LOOP PER NEURON, UNROLLED INNER BODY ================\n");
    printf("  marginal cost per image: B2 (fixed-width XMACC) %lld | S2 (packed scalar) %lld\n", mB, mS);
    printf("  anchors (marginal, per-group loops): Path B 1,992,849 | byte-extraction scalar 20,038,957 | row-level Path N 199,911\n");
    printf("  anchors (unrolled, single image, incl. loader): Path B 446,833 | Path S 438,926 | Path N 284,875\n");
    printf("  RATIOS: S2/B2 = %.2fx | B2/N = %.2fx | S2/N = %.2fx | per-group B / B2 = %.2fx | byte-extraction scalar / S2 = %.2fx\n",
           (double)mS / mB, (double)mB / 199911.0, (double)mS / 199911.0, 1992849.0 / mB, 20038957.0 / mS);
    printf("  single image: B2 cold %lld, warm %lld | S2 cold %lld, warm %lld | code B2 %zu B, S2 %zu B\n", coldB1, warmB1, coldS1, warmS1, codeB.size(), codeS.size());
    if (fail) printf("\n*** AT LEAST ONE CHECK FAILED -> numbers are UNRELIABLE, do NOT record ***\n");
    else printf("\n*** ALL CHECKS PASSED -> numbers are valid ***\n");
    vm->destroy(vm); dlclose(h); return fail ? 1 : 0;
#endif
}
