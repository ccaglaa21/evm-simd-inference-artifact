// mnist_codecopy16.cpp — the code-as-storage contract of mnist_codecopy with SIXTEEN 16-bit lanes per
// word and the row-level instruction XMACC_N16 (pairwise multiply-accumulate, 22 gas per 16-element group,
// the same price per element as XMACC_N). Weights, inputs and hidden activations occupy the whole 32-byte
// word, so the weight data halves (7,016 words, 224,512 bytes, 10 data contracts), and with it the memory
// expansion, the EXTCODECOPY copy cost and the number of cold account accesses. The accumulator keeps
// eight 32-bit lanes with pairwise sums, and XSUM32 gives the same dot product. Everything else (batch
// loop, loop per neuron, activation, argmax, compact memory map) is that of mnist_codecopy; the two
// packed 8-lane activation words of 16 consecutive neurons are joined into one 16-lane word.
// CHECKS: (1) 20 images, C++ reference == contract (2) cold-warm == data contracts * 2,500 for every batch
// size (3) MSIZE == expected (4) repeat run identical (5) code size under 24,576 (6) marginal cost per
// image constant over N = 1..16 and equal to the marginal cost of mnist_rowmacc (same body).
// Deployment costs are computed from the gas schedule and printed as such; they are not measured.
// Compile with -DLOCAL_TEST to dump the bytecode, data contracts and calldata instead of running evmone.
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

int main(int argc, char** argv) {
    // optional argument "fashion": the same contract generator on the Fashion-MNIST weight files (fashion_*.txt,
    // fashion_batch.txt); without it the MNIST files (deep_*.txt, mnist_batch.txt) are used
    const bool fashion = (argc > 1 && std::string(argv[1]) == "fashion");
    const std::string pfx = fashion ? "fashion_" : "deep_";
    const std::string batchFile = fashion ? "fashion_batch.txt" : "mnist_batch.txt";
    printf("dataset: %s (weights %s*.txt, images %s)\n", fashion ? "Fashion-MNIST" : "MNIST", pfx.c_str(), batchFile.c_str());
    int S, SHIFT1, SHIFT2, H1, H2;
    { std::ifstream f(pfx + "meta.txt"); f >> S >> SHIFT1 >> SHIFT2 >> H1 >> H2; }
    const int I = 784, L = 16, CDIMG = I / L * 32;
    std::vector<int> W1(H1 * I), W2(H2 * H1), W3(H2 * 10);
    { std::ifstream f(pfx + "W1.txt"); for (auto& x : W1) f >> x; }
    { std::ifstream f(pfx + "W2.txt"); for (auto& x : W2) f >> x; }
    { std::ifstream f(pfx + "W3.txt"); for (auto& x : W3) f >> x; }
    std::vector<long long> b1(H1), b2(H2);
    { std::ifstream f(pfx + "b1.txt"); for (auto& x : b1) f >> x; }
    { std::ifstream f(pfx + "b2.txt"); for (auto& x : b2) f >> x; }
    std::ifstream bf(batchFile); int Nb, wsc; bf >> Nb >> wsc;
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

    // memory map: compact, the data region is contiguous and follows the input, scratch follows the data
    const int INP = 0;
    const int W1M = INP + CDIMG;
    const int B1M = W1M + H1 * (I / L) * 32;
    const int W2M = B1M + H1 * 32;
    const int B2M = W2M + H2 * (H1 / L) * 32;
    const int W3M = B2M + H2 * 32;
    const int DATA_END = W3M + 10 * (H2 / L) * 32;
    const int H1OUT = DATA_END, H2OUT = H1OUT + (H1 / L) * 32, SCO = H2OUT + (H2 / L) * 32;
    const int SACC = SCO + 10 * 32, NCNT = SACC + (H1 / 8) * 32, MAXV = NCNT + 32, MAXI = MAXV + 32, CNT2 = MAXI + 32;
    const int NIMG = CNT2 + 32, ICNT = NIMG + 32, TMPW = ICNT + 32, PREDS = TMPW + 32;
    const int GCNT = 0; (void)GCNT;

    std::vector<std::pair<std::vector<uint8_t>, int>> WLOAD;
    for (int n = 0; n < H1; ++n) for (int gw = 0; gw < I / L; ++gw) { std::vector<uint8_t> w(32, 0); for (int l = 0; l < L; ++l) put16(w, l, W1[(gw * L + l) * H1 + n]); WLOAD.push_back({w, W1M + (n * (I / L) + gw) * 32}); }
    for (int n = 0; n < H1; ++n) { std::vector<uint8_t> w(32, 0); put32(w, 0, b1[n]); WLOAD.push_back({w, B1M + n * 32}); }
    for (int n = 0; n < H2; ++n) for (int gw = 0; gw < H1 / L; ++gw) { std::vector<uint8_t> w(32, 0); for (int l = 0; l < L; ++l) put16(w, l, W2[(gw * L + l) * H2 + n]); WLOAD.push_back({w, W2M + (n * (H1 / L) + gw) * 32}); }
    for (int n = 0; n < H2; ++n) { std::vector<uint8_t> w(32, 0); put32(w, 0, b2[n]); WLOAD.push_back({w, B2M + n * 32}); }
    for (int c = 0; c < 10; ++c) for (int gw = 0; gw < H2 / L; ++gw) { std::vector<uint8_t> w(32, 0); for (int l = 0; l < L; ++l) put16(w, l, W3[(gw * L + l) * 10 + c]); WLOAD.push_back({w, W3M + (c * (H2 / L) + gw) * 32}); }
    const int matWords = (int)WLOAD.size();
    struct Seg { int s0, cnt, base; };
    std::vector<Seg> segs;
    for (int sl = 0; sl < matWords; ++sl) { if (segs.empty() || WLOAD[sl].second != WLOAD[sl - 1].second + 32) segs.push_back({sl, 1, WLOAD[sl].second}); else segs.back().cnt++; }
    if (segs.size() != 1 || segs[0].base != W1M || segs[0].cnt * 32 != DATA_END - W1M) { printf("*** data region is not contiguous, stop ***\n"); return 1; }
    const long long MACS = (long long)I * H1 + (long long)H1 * H2 + (long long)H2 * 10;
    // data contracts: STOP byte + up to 24,575 data bytes each (EIP-170 limit 24,576)
    const int CHUNK = 24575;
    std::vector<uint8_t> data; for (int sl = 0; sl < matWords; ++sl) for (int b = 0; b < 32; ++b) data.push_back(WLOAD[sl].first[31 - b]);   // big-endian words, memory order
    std::vector<std::vector<uint8_t>> dcode; std::vector<int> dlen, ddest;
    for (size_t off = 0; off < data.size(); off += CHUNK) {
        int len = (int)std::min((size_t)CHUNK, data.size() - off);
        std::vector<uint8_t> c; c.push_back(0x00); c.insert(c.end(), data.begin() + off, data.begin() + off + len);
        dcode.push_back(c); dlen.push_back(len); ddest.push_back(W1M + (int)off);
    }
    const int NDATA = (int)dcode.size();

    // ================= N-deployable (looped loader, batch loop, looped neuron loop) =================
    std::vector<uint8_t> code;
    auto B_ = [&](uint8_t x) { code.push_back(x); };
    auto P1 = [&](uint8_t v) { B_(0x60); B_(v); };
    auto P3 = [&](int v) { B_(0x62); B_((v >> 16) & 0xFF); B_((v >> 8) & 0xFF); B_(v & 0xFF); };
    auto PUSH = [&](int v) { if (v < 256) P1((uint8_t)v); else if (v < 65536) { B_(0x61); B_((v >> 8) & 0xFF); B_(v & 0xFF); } else P3(v); };
    auto P4 = [&](uint32_t v) { B_(0x63); B_((v >> 24) & 0xFF); B_((v >> 16) & 0xFF); B_((v >> 8) & 0xFF); B_(v & 0xFF); };
    const uint8_t xgtcfg = (1 << 5) | 2;
    auto patch3 = [&](int ph, int target) { code[ph] = (target >> 16) & 0xFF; code[ph + 1] = (target >> 8) & 0xFF; code[ph + 2] = target & 0xFF; };
    auto emitCodeLoader = [&]() {      // EXTCODECOPY(addr_k, dest_k, 1, len_k) for every data contract
        for (int d = 0; d < NDATA; ++d) {
            PUSH(dlen[d]); P1(1); PUSH(ddest[d]);
            B_(0x73); for (int b = 0; b < 19; ++b) B_(0x00); B_((uint8_t)(0xD0 + d));   // PUSH20 address
            B_(0x3C);
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
        PUSH(NGin); B_(0xE7);                                                                     // XMACC_N16 -> [acc']
        B_(0xCD);                                                                                 // XSUM32 -> [sum]
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x06); P1(32); B_(0x02); B_(0x1B);              // sum << (NCNT%8)*32
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(SACC); B_(0x01);  // [v, saccaddr]
        B_(0x80); B_(0x51); B_(0x90); B_(0x91); B_(0x17); B_(0x90); B_(0x52);                     // sacc |= v
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x06); P1(7); B_(0x14); B_(0x15); P3(0); int skip_ph = (int)code.size() - 3; B_(0x57);
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(SACC); B_(0x01); B_(0x51);
        B_(0xCC); B_((uint8_t)SHIFT); B_(0xCE);                                                   // [v16]: 8 packed lanes (low 128 bits)
        PUSH(NCNT); B_(0x51); P1(16); B_(0x90); B_(0x06); P1(7); B_(0x14); B_(0x15); P3(0); int hi_ph = (int)code.size() - 3; B_(0x57);
        PUSH(TMPW); B_(0x52);                                                                    // NCNT%16 == 7: keep the low half
        P3(0); int end_ph = (int)code.size() - 3; B_(0x56);
        int hi = (int)code.size(); B_(0x5B); patch3(hi_ph, hi);                                   // NCNT%16 == 15: high half | low half
        P1(128); B_(0x1B); PUSH(TMPW); B_(0x51); B_(0x17);
        PUSH(NCNT); B_(0x51); P1(16); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(GOUT); B_(0x01); B_(0x52);
        int skip = (int)code.size(); B_(0x5B); patch3(skip_ph, skip); patch3(end_ph, skip);
        PUSH(NCNT); B_(0x51); P1(1); B_(0x01); PUSH(NCNT); B_(0x52);
        PUSH(outer); B_(0x56);
        int oe = (int)code.size(); B_(0x5B); patch3(oe_ph, oe);
    };
    auto emitBody = [&]() {
        looped_layer(H1, I / L, INP, W1M, B1M, H1OUT, SHIFT1);
        looped_layer(H2, H1 / L, H1OUT, W2M, B2M, H2OUT, SHIFT2);
        P1(0); PUSH(NCNT); B_(0x52);
        int co = (int)code.size(); B_(0x5B);
        PUSH(NCNT); B_(0x51); P1(10); B_(0x11); B_(0x15); P3(0); int co_ph = (int)code.size() - 3; B_(0x57);
        P1(0); PUSH(H2OUT); PUSH(NCNT); B_(0x51); PUSH((H2 / L) * 32); B_(0x02); PUSH(W3M); B_(0x01); PUSH(H2 / L); B_(0xE7);
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
    emitCodeLoader();
    B_(0x36); PUSH(CDIMG); B_(0x90); B_(0x04); PUSH(NIMG); B_(0x52);          // NIMG = CALLDATASIZE / 1568
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

    printf("Bytecode: inference contract %zu B | %d data contracts of at most %d bytes (%zu data bytes, %d words of 16 lanes) | XMACC_N16 priced at 22 gas per 16-element group\n\n", codeD.size(), NDATA, CHUNK + 1, data.size(), matWords);
    auto imgWords = [&](int im, std::vector<uint8_t>& dst) {
        for (int gw = 0; gw < I / L; ++gw) { std::vector<uint8_t> w(32, 0); for (int l = 0; l < L; ++l) { int val = imgs[im][gw * L + l]; uint16_t u = (uint16_t)(int16_t)val; w[l * 2] = u & 0xFF; w[l * 2 + 1] = (u >> 8) & 0xFF; } for (int i = 0; i < 32; ++i) dst.push_back(w[31 - i]); }
    };

#ifdef LOCAL_TEST
    auto dump = [](const char* name, const std::vector<uint8_t>& v) { std::ofstream f(name, std::ios::binary); f.write((const char*)v.data(), (std::streamsize)v.size()); };
    dump("c16_codeD.bin", codeD); dump("c16_codeDm.bin", codeDm);
    for (int d = 0; d < NDATA; ++d) dump(("c16_data_" + std::to_string(d) + ".bin").c_str(), dcode[d]);
    { std::ofstream f("c16_expected.txt"); f << NDATA << " " << matWords << " " << MACS << " " << SCO << " " << PREDS << " " << W1M << "\n";
      for (int im = 0; im < std::min(N_RUN, 20); ++im) { std::vector<long long> sc; int p = cppScores(im, sc); f << p; for (auto v : sc) f << " " << v; f << "\n";
          std::vector<uint8_t> cd; imgWords(im, cd); dump(("c16_calldata_" + std::to_string(im) + ".bin").c_str(), cd); } }
    printf("LOCAL_TEST: bytecode, data contracts, calldata and expected values written.\n");
    (void)cppPred;
    return 0;
#else
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::cerr << "libevmone not found\n"; return 1; }
    auto create = (evmc_vm * (*)())dlsym(h, "evmc_create_evmone");
    evmc_vm* vm = create();
    evmc::address ADDR{}; ADDR.bytes[18] = 0xc0; ADDR.bytes[19] = 0xde;
    evmc::MockedHost host;
    for (int d = 0; d < NDATA; ++d) { evmc::address a{}; a.bytes[19] = (uint8_t)(0xD0 + d); host.accounts[a].code.assign(dcode[d].begin(), dcode[d].end()); }
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

    // CHECK 1: 20 images, C++ == contract
    int ok = 0; long long w0; std::vector<int> pr;
    for (int im = 0; im < std::min(N_RUN, 20); ++im) {
        std::vector<uint8_t> cd; imgWords(im, cd); int cp = cppPred(im);
        runCold(codeD, cd, pr, w0); int pD = pr.empty() ? -1 : pr[0];
        if (pD == cp) ok++; else printf("  [MISMATCH] im=%d C++=%d contract=%d\n", im, cp, pD);
    }
    printf("[CHECK 1] 20 images, C++ == contract (code-as-storage, 16 lanes per word, XMACC_N16): %d/20 %s\n", ok, ok == 20 ? "PASSED" : "FAILED");
    if (ok != 20) { printf("*** CHECK 1 FAILED -> no measurement ***\n"); vm->destroy(vm); dlclose(h); return 1; }
    bool fail = false;
    printf("[CHECK 5] EIP-170 (<24,576): inference contract %zu B, largest data contract %zu B %s\n", codeD.size(), dcode[0].size(), (codeD.size() < 24576 && dcode[0].size() <= 24576) ? "PASSED" : "FAILED");
    if (codeD.size() >= 24576 || dcode[0].size() > 24576) fail = true;

    // batch N = 1, 2, 4, 8, 16: cold, cold repeat, warm, MSIZE
    std::vector<int> Ns = {1, 2, 4, 8, 16};
    long long GD_c[5], GD_w[5], MSD[5];
    const long long coldSurcharge = (long long)NDATA * 2500;
    for (int ni = 0; ni < 5; ++ni) {
        int N = Ns[ni]; std::vector<uint8_t> cd; for (int r = 0; r < N; ++r) imgWords(0, cd);
        GD_c[ni] = runCold(codeD, cd, pr, w0);
        long long r2 = runCold(codeD, cd, pr, w0);
        if (GD_c[ni] != r2) { printf("[CHECK 4] N=%d repeat MISMATCH\n", N); fail = true; }
        GD_w[ni] = runWarm(codeD, cd, pr, w0);
        runWarm(codeDm, cd, pr, MSD[ni]);
        long long expMS = (((long long)PREDS + N * 32 + 31) / 32) * 32;
        bool g2 = (GD_c[ni] - GD_w[ni] == coldSurcharge), g3 = (MSD[ni] == expMS);
        printf("[N=%2d] cold/warm = %lld/%lld | cold-warm %lld (expected %d x 2,500 = %lld) %s | MSIZE %lld (expected %lld) %s | repeat %s\n",
               N, GD_c[ni], GD_w[ni], GD_c[ni] - GD_w[ni], NDATA, coldSurcharge, g2 ? "OK" : "FAILED", MSD[ni], expMS, g3 ? "OK" : "FAILED", GD_c[ni] == r2 ? "identical" : "DIFFERENT");
        if (!g2 || !g3) fail = true;
    }
    printf("[CHECK 2] cold-warm == data contracts x 2,500 at every batch size: %s\n", fail ? "see above" : "PASSED");
    printf("[CHECK 3] MSIZE == expected at every batch size: %s\n", fail ? "see above" : "PASSED");
    printf("[CHECK 4] repeat runs identical: %s\n", fail ? "see above" : "PASSED");
    long long D[5]; for (int i = 0; i < 5; ++i) D[i] = GD_w[i] - memgas(MSD[i]);
    long long m01 = D[1] - D[0], m12 = (D[2] - D[1]) / 2, m23 = (D[3] - D[2]) / 4, m34 = (D[4] - D[3]) / 8;
    bool lin = (m01 == m12 && m12 == m23 && m23 == m34);
    printf("[CHECK 6] marginal per image (memory-stripped): %lld %lld %lld %lld -> %s\n", m01, m12, m23, m34, lin ? "EXACTLY CONSTANT" : "FAILED");
    if (!lin) fail = true;

    // components of the cold single-image run
    long long copyWords = 0; for (int d = 0; d < NDATA; ++d) copyWords += (dlen[d] + 31) / 32;
    const long long readCold = (long long)NDATA * 2600 + 3 * copyWords;      // account access (cold) + copy cost
    const long long readWarm = (long long)NDATA * 100 + 3 * copyWords;
    long long mem1 = memgas(MSD[0]);
    long long rest = GD_c[0] - readCold - mem1;                                 // compute + loop + calldata + loader pushes
    const long long CAP = 16777216;
    printf("\n================ CODE-AS-STORAGE WEIGHTS, 16 LANES PER WORD (EXTCODECOPY loader, XMACC_N16) ================\n");
    printf("  single image, cold: total %lld = weight read %lld (%d cold account accesses x 2,600 + %lld copied words x 3) + memory %lld (MSIZE %lld) + the rest %lld\n", GD_c[0], readCold, NDATA, copyWords, mem1, MSD[0], rest);
    printf("  single image, warm: total %lld (weight read %lld)\n", GD_w[0], readWarm);
    printf("  weight-read share of the cold total: %.2f%%\n", 100.0 * readCold / GD_c[0]);
    printf("  MNIST anchors (mnist_codecopy, 8 lanes per word, XMACC_N): cold 715,664 | warm 668,164 | weight read 90,923 | memory 424,524 | marginal 199,911 | 19 data contracts\n");
    printf("  MNIST anchors (mnist_rowmacc, SLOAD weights): cold 32,185,375 | marginal 199,911\n");
    printf("  cold single-image ratios: 8-lane code-as-storage / 16-lane = %.2fx | SLOAD / 16-lane = %.2fx | marginal 8-lane / 16-lane = %.2fx\n", 715664.0 / GD_c[0], 32185375.0 / GD_c[0], 199911.0 / m01);
    printf("  EIP-7825 cap %lld: single image %s (%.1f%% of the cap) | batch of 16 cold %lld %s\n", CAP, GD_c[0] < CAP ? "FITS in one transaction" : "does NOT fit", 100.0 * GD_c[0] / CAP, GD_c[4], GD_c[4] < CAP ? "fits" : "does not fit");
    printf("  marginal per image %lld | code %zu B | memory map compact (data at %d..%d, scratch to %d)\n", m01, codeD.size(), W1M, DATA_END, PREDS);
    // deployment, from the gas schedule (not measured): code deposit 200 gas per byte + 32,000 per creation
    long long depBytes = 0; for (int d = 0; d < NDATA; ++d) depBytes += (long long)dcode[d].size();
    long long depCode = 200 * depBytes + 32000LL * NDATA;
    long long depSlots = (long long)matWords * 22100;   // SSTORE of a fresh slot in a cold access: 20,000 + 2,100
    printf("  deployment (gas schedule, not measured): %d data contracts, %lld bytes x 200 + %d x 32,000 = %lld | SLOAD layout of the same words: %d x 22,100 = %lld\n", NDATA, depBytes, NDATA, depCode, matWords, depSlots);
    if (fail) printf("\n*** AT LEAST ONE CHECK FAILED -> numbers are UNRELIABLE, do NOT record ***\n");
    else printf("\n*** ALL CHECKS PASSED -> numbers are valid ***\n");
    vm->destroy(vm); dlclose(h); return fail ? 1 : 0;
#endif
}
