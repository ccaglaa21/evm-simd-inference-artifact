// mnist_swar_codecopy.cpp — the packed-multiplication scalar baseline (Path S) with the weights stored as
// contract code and read with EXTCODECOPY, the same code-as-storage pattern that mnist_codecopy applies to the
// row-level path (Path N). It answers one question: how much of the cold single-image cost of the
// code-as-storage variant belongs to the storage pattern and how much to the proposed instructions.
//
// The contract is the S2 contract of mnist_loops (batch loop, loop per neuron, groups of a neuron unrolled,
// standard instructions only) with one change: the looped SLOAD loader is replaced by one EXTCODECOPY per data
// contract. The 14,052 weight and bias words of the Path S layout (449,664 bytes) are split into data contracts
// of at most 24,575 bytes, each prefixed with a STOP byte. Memory map, body and calldata are those of S2.
// CHECKS: (1) 20 images, C++ reference == contract (2) cold-warm == data contracts x 2,500 at every batch size
// (3) MSIZE == expected (4) repeat runs identical (5) code sizes under 24,576 (6) marginal cost per image
// constant over N = 1, 2, 4, 8, 16 and equal to the S2 marginal of mnist_loops (427,553, same body).
// Compile with -DLOCAL_TEST to dump the bytecode, data contracts and calldata instead of running evmone.
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

    // ======================= S2: packed-multiplication scalar, loop per neuron, unrolled groups =======================
    const int OFF = 32768;
    const int L1 = 8, LW1 = 32, G1 = I / L1, L2 = 7, LW2 = 36, G2 = (H1 + L2 - 1) / L2, L3 = 7, G3 = (H2 + L3 - 1) / L3, WIN1 = 32, WIN2 = 4;
    const int XIN = 0, W1S = XIN + G1 * 32, B1S = W1S + H1 * G1 * 32, W2S = B1S + H1 * 32, B2S = W2S + H2 * G2 * 32, W3S = B2S + H2 * 32;
    const int X2 = W3S + 10 * G3 * 32, X3 = X2 + G2 * 32, CORR = X3 + G3 * 32, SCOS = CORR + 3 * 32, SMAXV = SCOS + 10 * 32, SMAXI = SMAXV + 32;
    const int SCNT = SMAXI + 32, SNIMG = SCNT + 32, SICNT = SNIMG + 32, SCNT2 = SICNT + 32, SPREDS = SCNT2 + 32;
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
    // data region: the slots in loading order are one contiguous memory range [W1S, X2)
    { auto sg = segments(SADDR); if (sg.size() != 1 || sg[0].base != W1S || sg[0].cnt * 32 != X2 - W1S) { printf("*** data region is not contiguous, stop ***\n"); return 1; } }
    const int CHUNK = 24575;
    std::vector<uint8_t> data; for (auto& w : SSLOT) data.insert(data.end(), w.begin(), w.end());   // big-endian words, memory order
    std::vector<std::vector<uint8_t>> dcode; std::vector<int> dlen, ddest;
    for (size_t off = 0; off < data.size(); off += CHUNK) {
        int len = (int)std::min((size_t)CHUNK, data.size() - off);
        std::vector<uint8_t> cc; cc.push_back(0x00); cc.insert(cc.end(), data.begin() + off, data.begin() + off + len);
        dcode.push_back(cc); dlen.push_back(len); ddest.push_back(W1S + (int)off);
    }
    const int NDATA = (int)dcode.size();
    for (int dd = 0; dd < NDATA; ++dd) {   // EXTCODECOPY(addr_d, dest_d, 1, len_d)
        gs.PUSH(dlen[dd]); gs.P1(1); gs.PUSH(ddest[dd]);
        gs.B(0x73); for (int bb = 0; bb < 19; ++bb) gs.B(0x00); gs.B((uint8_t)(0xD0 + dd));
        gs.B(0x3C);
    }
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

    auto cdS = [&](int im, std::vector<uint8_t>& dst) { for (int g = 0; g < G1; ++g) { Word w{}; for (int k = 0; k < L1; ++k) or_bits(w, LW1 * (L1 - 1 - k), (uint64_t)imgs[im][g * L1 + k], LW1); dst.insert(dst.end(), w.begin(), w.end()); } };
    printf("Bytecode: Path S contract with EXTCODECOPY loader %zu B | %d data contracts of at most %d bytes (%zu data bytes, %d words)\n\n", codeS.size(), NDATA, CHUNK + 1, data.size(), sslots);

#ifdef LOCAL_TEST
    auto dump = [](const std::string& name, const std::vector<uint8_t>& v) { std::ofstream f(name, std::ios::binary); f.write((const char*)v.data(), (std::streamsize)v.size()); };
    dump("sc_code.bin", codeS); dump("sc_codem.bin", codeSm);
    for (int dd = 0; dd < NDATA; ++dd) dump("sc_data_" + std::to_string(dd) + ".bin", dcode[dd]);
    { std::ofstream f("sc_expected.txt"); f << NDATA << " " << sslots << " " << SPREDS << "\n";
      for (int im = 0; im < std::min(N_RUN, 20); ++im) { f << cppPred(im) << "\n"; std::vector<uint8_t> bb; cdS(im, bb); dump("sc_cd_" + std::to_string(im) + ".bin", bb); } }
    printf("LOCAL_TEST: bytecode, data contracts, calldata and expected predictions written.\n");
    return 0;
#else
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::cerr << "libevmone not found\n"; return 1; }
    auto create = (evmc_vm * (*)())dlsym(h, "evmc_create_evmone");
    evmc_vm* vm = create();
    evmc::address ADDR{}; ADDR.bytes[18] = 0xc0; ADDR.bytes[19] = 0xdf;
    evmc::MockedHost host;
    for (int dd = 0; dd < NDATA; ++dd) { evmc::address a{}; a.bytes[19] = (uint8_t)(0xD0 + dd); host.accounts[a].code.assign(dcode[dd].begin(), dcode[dd].end()); }
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
    // cold run: no account accessed yet in this transaction; warm run: the data contracts were accessed by a previous run
    auto runCold = [&](std::vector<uint8_t>& kk, std::vector<uint8_t>& cd, std::vector<int>& preds, long long& w0) { host.recorded_account_accesses.clear(); return run(kk, cd, preds, w0); };
    auto runWarm = [&](std::vector<uint8_t>& kk, std::vector<uint8_t>& cd, std::vector<int>& preds, long long& w0) { host.recorded_account_accesses.clear(); run(kk, cd, preds, w0); return run(kk, cd, preds, w0); };

    int ok = 0; long long w0; std::vector<int> pr;
    for (int im = 0; im < std::min(N_RUN, 20); ++im) {
        std::vector<uint8_t> cd; cdS(im, cd); int cp = cppPred(im);
        runCold(codeS, cd, pr, w0); int pS = pr.empty() ? -1 : pr[0];
        if (pS == cp) ok++; else printf("  [MISMATCH] im=%d C++=%d contract=%d\n", im, cp, pS);
    }
    printf("[CHECK 1] 20 images, C++ == contract (Path S, code-as-storage, EXTCODECOPY loader): %d/20 %s\n", ok, ok == 20 ? "PASSED" : "FAILED");
    if (ok != 20) { printf("*** CHECK 1 FAILED -> no measurement ***\n"); vm->destroy(vm); dlclose(h); return 1; }
    bool fail = false;
    printf("[CHECK 5] EIP-170 (<24,576): contract %zu B, largest data contract %zu B %s\n", codeS.size(), dcode[0].size(), (codeS.size() < 24576 && dcode[0].size() <= 24576) ? "PASSED" : "FAILED");
    if (codeS.size() >= 24576 || dcode[0].size() > 24576) fail = true;
    std::vector<int> Ns = {1, 2, 4, 8, 16};
    long long Gc[5], Gw[5], MS[5];
    const long long coldSurcharge = (long long)NDATA * 2500;
    for (int ni = 0; ni < 5; ++ni) {
        int N = Ns[ni]; std::vector<uint8_t> cd; for (int r = 0; r < N; ++r) cdS(0, cd);
        Gc[ni] = runCold(codeS, cd, pr, w0);
        long long r2 = runCold(codeS, cd, pr, w0);
        Gw[ni] = runWarm(codeS, cd, pr, w0);
        runWarm(codeSm, cd, pr, MS[ni]);
        long long expMS = (((long long)SPREDS + N * 32 + 31) / 32) * 32;
        bool g2 = (Gc[ni] - Gw[ni] == coldSurcharge), g3 = (MS[ni] == expMS), g4 = (Gc[ni] == r2);
        printf("[N=%2d] cold/warm = %lld/%lld | cold-warm %lld (expected %d x 2,500 = %lld) %s | MSIZE %lld (expected %lld) %s | repeat %s\n",
               N, Gc[ni], Gw[ni], Gc[ni] - Gw[ni], NDATA, coldSurcharge, g2 ? "OK" : "FAILED", MS[ni], expMS, g3 ? "OK" : "FAILED", g4 ? "identical" : "DIFFERENT");
        if (!g2 || !g3 || !g4) fail = true;
    }
    long long D[5]; for (int i = 0; i < 5; ++i) D[i] = Gw[i] - memgas(MS[i]);
    long long m01 = D[1] - D[0], m12 = (D[2] - D[1]) / 2, m23 = (D[3] - D[2]) / 4, m34 = (D[4] - D[3]) / 8;
    bool lin = (m01 == m12 && m12 == m23 && m23 == m34);
    printf("[CHECK 6] marginal per image (memory-stripped): %lld %lld %lld %lld -> %s | equals the S2 marginal of mnist_loops (427,553): %s\n", m01, m12, m23, m34, lin ? "EXACTLY CONSTANT" : "FAILED", m01 == 427553 ? "yes" : "NO");
    if (!lin || m01 != 427553) fail = true;

    long long copyWords = 0; for (int dd = 0; dd < NDATA; ++dd) copyWords += (dlen[dd] + 31) / 32;
    const long long readCold = (long long)NDATA * 2600 + 3 * copyWords;      // account access (cold) + copy cost
    long long mem1 = memgas(MS[0]);
    long long rest = Gc[0] - readCold - mem1;
    const long long CAP = 16777216;
    // anchors measured by mnist_codecopy and mnist_codecopy16 (Path N, weights as code) and by mnist_rowmacc (Path N, weights in storage)
    const double N8 = 715664.0, N16 = 368032.0, NSTORE = 32185375.0;
    printf("\n================ PATH S WITH THE WEIGHTS AS CONTRACT CODE (EXTCODECOPY loader, S2 body) ================\n");
    printf("  single image, cold: total %lld = weight read %lld (%d cold account accesses x 2,600 + %lld copied words x 3) + memory %lld (MSIZE %lld) + the rest %lld\n", Gc[0], readCold, NDATA, copyWords, mem1, MS[0], rest);
    printf("  single image, warm: %lld | batch of 16, cold: %lld | marginal per image %lld | code %zu B\n", Gw[0], Gc[4], m01, codeS.size());
    printf("  share of the EIP-7825 per-transaction cap (16,777,216): %.1f%%\n", 100.0 * Gc[0] / CAP);
    printf("  anchors, Path N cold single image: weights in storage 32,185,375 | weights as code, 8 lanes 715,664 | weights as code, 16 lanes 368,032\n");
    printf("  RATIOS (cold single image, weights as code): Path S / Path N (8 lanes) = %.2fx | Path S / Path N (16 lanes) = %.2fx\n", Gc[0] / N8, Gc[0] / N16);
    printf("  RATIOS (storage pattern alone): Path N in storage / Path S as code = %.2fx | Path N in storage / Path N as code (8 lanes) = %.2fx\n", NSTORE / Gc[0], NSTORE / N8);
    if (fail) printf("\n*** AT LEAST ONE CHECK FAILED -> numbers are UNRELIABLE, do NOT record ***\n");
    else printf("\n*** ALL CHECKS PASSED -> numbers are valid ***\n");
    vm->destroy(vm); dlclose(h); return fail ? 1 : 0;
#endif
}
