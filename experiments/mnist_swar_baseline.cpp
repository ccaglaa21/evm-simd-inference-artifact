// mnist_swar_baseline.cpp — Path S: a packed-multiplication scalar baseline for the MNIST network.
//
// The baseline uses only standard EVM instructions. Several 16-bit weights are packed into one
// 256-bit word with fixed lane spacing, the input values are packed in reverse lane order, and one
// 256-bit MUL then produces the dot product of the packed lanes in the top lane of the product
// (Kronecker substitution). Weights are stored with an offset of 2^15 so that every lane is
// non-negative; the offset is removed once per neuron with the lane sum of the inputs.
//   layer 1 (784 pixel inputs, 0..255):      8 lanes of 32 bits per word, extraction every 32 groups
//   layers 2 and 3 (inputs 0..32767):        7 lanes of 36 bits per word, extraction every 4 groups
// The program is straight-line (no loop, not deployable) and is measured exactly like the unrolled
// scalar Path C of mnist_e2e_cb: cold and warm storage, MSIZE variant, three-component split.
// CHECKS: (1) 20 images, prediction equality with the C++ integer reference (2) cold-warm == slots*2000
//         (3) compute_cold == compute_warm (4) static prediction == measured compute (5) MSIZE cold == warm
//         (6) repeat run identical.
// Compile with -DLOCAL_TEST to dump the bytecode, storage and calldata instead of running evmone.
#ifndef LOCAL_TEST
#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#endif
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Word = std::array<uint8_t, 32>;   // big-endian 256-bit value, as stored in storage and memory

static void or_bits(Word& w, int bitpos, uint64_t v, int nbits) {
    for (int b = 0; b < nbits; ++b)
        if ((v >> b) & 1ULL) { int bit = bitpos + b; w[31 - bit / 8] |= (uint8_t)(1u << (bit % 8)); }
}
static Word word_i256(long long v) {       // two's complement 256-bit encoding of a signed value
    Word w{}; uint64_t u = (uint64_t)v;
    for (int b = 0; b < 8; ++b) w[31 - b] = (uint8_t)((u >> (8 * b)) & 0xFF);
    if (v < 0) for (int i = 0; i < 24; ++i) w[i] = 0xFF;
    return w;
}

int main() {
    // ---------------- data (same files as mnist_e2e_cb) ----------------
    int S, SHIFT1, SHIFT2, H1, H2;
    { std::ifstream f("deep_meta.txt"); f >> S >> SHIFT1 >> SHIFT2 >> H1 >> H2; }
    const int I = 784;
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
    for (int im = 0; im < N_RUN; ++im) {
        int tlb, pfo; bf >> tlb >> pfo; double tmp; for (int i = 0; i < 10; ++i) bf >> tmp;
        for (int i = 0; i < I; ++i) bf >> imgs[im][i];
    }
    auto rss = [&](long long v, int sh) -> int { if (v < 0) v = 0; v >>= sh; if (v > 32767) v = 32767; return (int)v; };
    const int OFF = 32768;                      // weight offset: w' = w + 2^15, non-negative lanes

    // ---------------- lane geometry ----------------
    const int L1 = 8, LW1 = 32, G1 = I / L1;             // 98 groups of 8 pixels, 32-bit lanes, top lane at bit 224
    const int L2 = 7, LW2 = 36, G2 = (H1 + L2 - 1) / L2; // 19 groups, 36-bit lanes, top lane at bit 216
    const int L3 = 7, G3 = (H2 + L3 - 1) / L3;           // 10 groups
    const int WIN1 = 32, WIN2 = 4;                        // groups accumulated before one extraction
    const int MACS = I * H1 + H1 * H2 + H2 * 10;          // 109,184 multiply-accumulates

    // ---------------- storage words (slot order) and memory map ----------------
    std::vector<Word> SLOT;                               // slot k holds SLOT[k]
    std::vector<int> SLOTADDR;                            // memory address the loader writes slot k to
    const int XIN = 0;                                    // packed input, G1 words (from calldata)
    const int W1S = XIN + G1 * 32;                        // H1*G1 words
    const int B1S = W1S + H1 * G1 * 32;                   // H1 words (bias as signed 256-bit)
    const int W2S = B1S + H1 * 32;                        // H2*G2 words
    const int B2S = W2S + H2 * G2 * 32;                   // H2 words
    const int W3S = B2S + H2 * 32;                        // 10*G3 words
    const int X2 = W3S + 10 * G3 * 32;                    // packed layer-2 input, G2 words
    const int X3 = X2 + G2 * 32;                          // packed layer-3 input, G3 words
    const int CORR = X3 + G3 * 32;                        // three correction words
    const int SCO = CORR + 3 * 32;                        // 10 score words
    const int SMAXV = SCO + 10 * 32, SMAXI = SMAXV + 32;  // argmax scratch
    const int MEM_END = SMAXI + 32;                       // expected MSIZE
    for (int n = 0; n < H1; ++n) for (int g = 0; g < G1; ++g) {
        Word w{}; for (int k = 0; k < L1; ++k) or_bits(w, LW1 * k, (uint64_t)(W1[(g * L1 + k) * H1 + n] + OFF), LW1);
        SLOT.push_back(w); SLOTADDR.push_back(W1S + (n * G1 + g) * 32);
    }
    const int w1cnt = (int)SLOT.size();
    for (int n = 0; n < H1; ++n) { SLOT.push_back(word_i256(b1[n])); SLOTADDR.push_back(B1S + n * 32); }
    for (int n = 0; n < H2; ++n) for (int g = 0; g < G2; ++g) {
        Word w{}; for (int k = 0; k < L2; ++k) { int i = g * L2 + k; if (i < H1) or_bits(w, LW2 * k, (uint64_t)(W2[i * H2 + n] + OFF), LW2); }
        SLOT.push_back(w); SLOTADDR.push_back(W2S + (n * G2 + g) * 32);
    }
    for (int n = 0; n < H2; ++n) { SLOT.push_back(word_i256(b2[n])); SLOTADDR.push_back(B2S + n * 32); }
    for (int c = 0; c < 10; ++c) for (int g = 0; g < G3; ++g) {
        Word w{}; for (int k = 0; k < L3; ++k) { int i = g * L3 + k; if (i < H2) or_bits(w, LW2 * k, (uint64_t)(W3[i * 10 + c] + OFF), LW2); }
        SLOT.push_back(w); SLOTADDR.push_back(W3S + (c * G3 + g) * 32);
    }
    const int slots = (int)SLOT.size();
    Word ONES8{}; for (int k = 0; k < L1; ++k) or_bits(ONES8, LW1 * k, 1, LW1);   // lane sum helper, layer 1
    Word ONES7{}; for (int k = 0; k < L2; ++k) or_bits(ONES7, LW2 * k, 1, LW2);   // lane sum helper, layers 2 and 3

    // ---------------- bytecode emission with a static gas counter ----------------
    std::vector<uint8_t> code; long long sgas = 0;      // sgas: static cost of everything except SLOAD
    auto op = [&](uint8_t o, long long g) { code.push_back(o); sgas += g; };
    auto P1 = [&](uint8_t v) { op(0x60, 3); code.push_back(v); };
    auto PUSH = [&](long long v) {
        if (v < 256) { P1((uint8_t)v); return; }
        int n = 0; for (long long t = v; t > 0; t >>= 8) ++n;
        op((uint8_t)(0x5F + n), 3); for (int b = n - 1; b >= 0; --b) code.push_back((uint8_t)((v >> (8 * b)) & 0xFF));
    };
    auto PUSHW = [&](const Word& w) { op(0x7F, 3); for (int i = 0; i < 32; ++i) code.push_back(w[i]); };
    auto MLOAD = [&]() { op(0x51, 3); }; auto MSTORE = [&]() { op(0x52, 3); };
    auto MUL = [&]() { op(0x02, 5); };   auto ADD = [&]() { op(0x01, 3); };  auto SUB = [&]() { op(0x03, 3); };
    auto SHR = [&]() { op(0x1C, 3); };   auto SHL = [&]() { op(0x1B, 3); };  auto AND = [&]() { op(0x16, 3); };
    auto OR = [&]() { op(0x17, 3); };    auto DUP = [&](int n) { op((uint8_t)(0x7F + n), 3); };
    auto SWAP = [&](int n) { op((uint8_t)(0x8F + n), 3); }; auto POP = [&]() { op(0x50, 2); };
    auto SGT = [&]() { op(0x13, 3); };   auto ISZERO = [&]() { op(0x15, 3); };
    auto MASK36 = [&]() { PUSH((1LL << 36) - 1); };

    // prelude: CALLDATACOPY of the packed input, then the unrolled loader (PUSH32 slot, SLOAD, PUSH addr, MSTORE)
    PUSH(G1 * 32); P1(0); PUSH(XIN); op(0x37, 3 + 3 * G1);
    for (int sl = 0; sl < slots; ++sl) {
        op(0x7F, 3); for (int i = 0; i < 28; ++i) code.push_back(0);
        code.push_back((sl >> 24) & 0xFF); code.push_back((sl >> 16) & 0xFF); code.push_back((sl >> 8) & 0xFF); code.push_back(sl & 0xFF);
        op(0x54, 0);                                        // SLOAD, charged in the storage component
        PUSH(SLOTADDR[sl]); MSTORE();
    }
    const long long loaderGas = sgas;

    // lane sum of a packed input region: CORR[idx] = (sum of all lanes) << 15
    auto emitCorrection = [&](int xbase, int groups, const Word& ones, int topbit, bool mask, int corrAddr) {
        PUSHW(ones); P1(0);
        for (int g = 0; g < groups; ++g) { PUSH(xbase + g * 32); MLOAD(); DUP(3); MUL(); ADD(); }
        PUSH(topbit); SHR(); if (mask) { MASK36(); AND(); }
        P1(15); SHL(); PUSH(corrAddr); MSTORE(); POP();
    };
    // ReLU, right shift, clip to 0..32767 (branchless, same idiom as Path C)
    auto emitActivation = [&](int shift) {
        DUP(1); P1(0); SGT(); ISZERO(); MUL();                 // acc * (acc >= 0)
        P1((uint8_t)shift); SHR();                             // >> shift
        DUP(1); PUSH(32767); SWAP(1); SGT();                   // over = t > 32767
        PUSH(32767); DUP(3); SWAP(1); SUB(); MUL(); ADD();     // t + over * (32767 - t)
    };
    // one neuron: dot product over 'groups' packed words, window extraction, correction, bias
    auto emitDot = [&](int wbase, int xbase, int groups, int window, int topbit, bool mask, int corrAddr, int biasAddr) {
        P1(0);                                                   // T
        for (int g = 0; g < groups; ++g) {
            if (g % window == 0) P1(0);                          // P
            PUSH(wbase + g * 32); MLOAD(); PUSH(xbase + g * 32); MLOAD(); MUL(); ADD();
            if ((g + 1) % window == 0 || g == groups - 1) { PUSH(topbit); SHR(); if (mask) { MASK36(); AND(); } ADD(); }
        }
        PUSH(corrAddr); MLOAD(); SWAP(1); SUB();                 // T - correction
        if (biasAddr >= 0) { PUSH(biasAddr); MLOAD(); ADD(); }  // + bias
    };
    // layer 1
    emitCorrection(XIN, G1, ONES8, 224, false, CORR);
    for (int n = 0; n < H1; ++n) {
        emitDot(W1S + n * G1 * 32, XIN, G1, WIN1, 224, false, CORR, B1S + n * 32);
        emitActivation(SHIFT1);
        PUSH(LW2 * (L2 - 1 - n % L2)); SHL(); PUSH(X2 + (n / L2) * 32); MLOAD(); OR(); PUSH(X2 + (n / L2) * 32); MSTORE();
    }
    // layer 2
    emitCorrection(X2, G2, ONES7, 216, true, CORR + 32);
    for (int n = 0; n < H2; ++n) {
        emitDot(W2S + n * G2 * 32, X2, G2, WIN2, 216, true, CORR + 32, B2S + n * 32);
        emitActivation(SHIFT2);
        PUSH(LW2 * (L2 - 1 - n % L3)); SHL(); PUSH(X3 + (n / L3) * 32); MLOAD(); OR(); PUSH(X3 + (n / L3) * 32); MSTORE();
    }
    // layer 3 (no bias), scores as signed words
    emitCorrection(X3, G3, ONES7, 216, true, CORR + 64);
    for (int c = 0; c < 10; ++c) { emitDot(W3S + c * G3 * 32, X3, G3, WIN2, 216, true, CORR + 64, -1); PUSH(SCO + c * 32); MSTORE(); }
    // argmax (branchless, signed, first maximum wins) — same idiom as Path C
    PUSH(SCO); MLOAD(); PUSH(SMAXV); MSTORE(); P1(0); PUSH(SMAXI); MSTORE();
    for (int c = 1; c < 10; ++c) {
        PUSH(SMAXV); MLOAD(); PUSH(SCO + c * 32); MLOAD(); SGT(); DUP(1);
        PUSH(SCO + c * 32); MLOAD(); PUSH(SMAXV); MLOAD(); SWAP(1); SUB(); MUL();
        PUSH(SMAXV); MLOAD(); ADD(); PUSH(SMAXV); MSTORE();
        PUSH(SMAXI); MLOAD(); DUP(1); PUSH(c); SUB(); DUP(3); MUL(); ADD(); PUSH(SMAXI); MSTORE(); POP();
    }
    std::vector<uint8_t> body = code; const long long bodyGas = sgas;
    // tails: return the prediction, or return MSIZE
    std::vector<uint8_t> codeS = body;
    { auto& k = codeS; k.push_back(0x62); k.push_back((SMAXI >> 16) & 0xFF); k.push_back((SMAXI >> 8) & 0xFF); k.push_back(SMAXI & 0xFF);
      k.push_back(0x51); k.push_back(0x60); k.push_back(0x00); k.push_back(0x52); k.push_back(0x60); k.push_back(0x20); k.push_back(0x60); k.push_back(0x00); k.push_back(0xF3); }
    const long long tailGas = 3 + 3 + 3 + 3 + 3 + 3;      // PUSH3 MLOAD PUSH1 MSTORE PUSH1 PUSH1 RETURN(0)
    std::vector<uint8_t> codeSm = body;
    { uint8_t t[] = {0x59, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xF3}; for (uint8_t x : t) codeSm.push_back(x); }
    const long long predCompute = bodyGas + tailGas;        // static prediction of the compute component
    auto memgas = [](long long ms) -> long long { long long w = (ms + 31) / 32; return 3 * w + (w * w) / 512; };

    // packed calldata and reference prediction for one image
    auto buildCalldata = [&](int im, std::vector<uint8_t>& cd, int& cpp_pred) {
        std::vector<int>& x = imgs[im];
        std::vector<int> h1r(H1); for (int n = 0; n < H1; ++n) { long long s = b1[n]; for (int i = 0; i < I; ++i) s += (long long)x[i] * W1[i * H1 + n]; h1r[n] = rss(s, SHIFT1); }
        std::vector<int> h2r(H2); for (int n = 0; n < H2; ++n) { long long s = b2[n]; for (int i = 0; i < H1; ++i) s += (long long)h1r[i] * W2[i * H2 + n]; h2r[n] = rss(s, SHIFT2); }
        std::vector<long long> scr(10); for (int c = 0; c < 10; ++c) { long long s = 0; for (int i = 0; i < H2; ++i) s += (long long)h2r[i] * W3[i * 10 + c]; scr[c] = s; }
        cpp_pred = (int)(std::max_element(scr.begin(), scr.end()) - scr.begin());
        cd.assign(G1 * 32, 0);
        for (int g = 0; g < G1; ++g) { Word w{}; for (int k = 0; k < L1; ++k) or_bits(w, LW1 * (L1 - 1 - k), (uint64_t)x[g * L1 + k], LW1); std::memcpy(&cd[g * 32], w.data(), 32); }
    };

    printf("Bytecode: SWAR=%zu byte | slots=%d = W1:%d + b1:%d + W2:%d + b2:%d + W3:%d | lanes: layer1 %dx%d-bit, layers 2-3 %dx%d-bit\n",
           codeS.size(), slots, w1cnt, H1, H2 * G2, H2, 10 * G3, L1, LW1, L2, LW2);
    printf("[NOTE] Weights stored with offset 2^15 (non-negative lanes); offset removed once per neuron with the lane sum of the inputs.\n");
    printf("[NOTE] Static prediction of the compute component: %lld gas (loader %lld, expected MSIZE %d)\n\n", predCompute, loaderGas, MEM_END);

#ifdef LOCAL_TEST
    { std::ofstream f("swar_code.bin", std::ios::binary); f.write((const char*)codeS.data(), (std::streamsize)codeS.size()); }
    { std::ofstream f("swar_code_msize.bin", std::ios::binary); f.write((const char*)codeSm.data(), (std::streamsize)codeSm.size()); }
    { std::ofstream f("swar_storage.hex"); for (int sl = 0; sl < slots; ++sl) { for (int i = 0; i < 32; ++i) { char b[3]; std::snprintf(b, 3, "%02x", SLOT[sl][i]); f << b; } f << "\n"; } }
    { std::ofstream f("swar_expected.txt"); f << predCompute << " " << slots << " " << MEM_END << " " << MACS << " " << SCO << "\n";
      for (int im = 0; im < std::min(N_RUN, 20); ++im) { std::vector<uint8_t> cd; int p; buildCalldata(im, cd, p); f << p;
          { std::vector<int>& x = imgs[im]; std::vector<int> h1r(H1); for (int n = 0; n < H1; ++n) { long long s2 = b1[n]; for (int i = 0; i < I; ++i) s2 += (long long)x[i] * W1[i * H1 + n]; h1r[n] = rss(s2, SHIFT1); }
            std::vector<int> h2r(H2); for (int n = 0; n < H2; ++n) { long long s2 = b2[n]; for (int i = 0; i < H1; ++i) s2 += (long long)h1r[i] * W2[i * H2 + n]; h2r[n] = rss(s2, SHIFT2); }
            for (int c = 0; c < 10; ++c) { long long s2 = 0; for (int i = 0; i < H2; ++i) s2 += (long long)h2r[i] * W3[i * 10 + c]; f << " " << s2; } }
          f << "\n";
          std::ofstream c("swar_calldata_" + std::to_string(im) + ".bin", std::ios::binary); c.write((const char*)cd.data(), (std::streamsize)cd.size()); } }
    (void)memgas;
    printf("LOCAL_TEST: bytecode, storage, calldata and expected predictions written.\n");
    return 0;
#else
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::cerr << "libevmone not found\n"; return 1; }
    auto create = (evmc_vm * (*)())dlsym(h, "evmc_create_evmone");
    evmc_vm* vm = create();
    evmc::address ADDR{}; ADDR.bytes[18] = 0xc0; ADDR.bytes[19] = 0xde;
    evmc::MockedHost host;
    auto keyOf = [](int sl) { evmc::bytes32 key{}; key.bytes[28] = (sl >> 24) & 0xFF; key.bytes[29] = (sl >> 16) & 0xFF; key.bytes[30] = (sl >> 8) & 0xFF; key.bytes[31] = sl & 0xFF; return key; };
    for (int sl = 0; sl < slots; ++sl) { evmc::bytes32 val{}; for (int b = 0; b < 32; ++b) val.bytes[b] = SLOT[sl][b]; host.accounts[ADDR].storage[keyOf(sl)] = evmc::StorageValue{val}; }
    auto setAcc = [&](evmc_access_status st) { for (int sl = 0; sl < slots; ++sl) host.accounts[ADDR].storage[keyOf(sl)].access_status = st; };
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();
    const int64_t GBIG = 20000000000LL;
    auto run = [&](std::vector<uint8_t>& k, std::vector<uint8_t>& cd, int& pred, long long& word0) -> long long {
        evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GBIG; msg.recipient = ADDR; msg.code_address = ADDR;
        msg.input_data = cd.data(); msg.input_size = cd.size();
        auto res = vm->execute(vm, &IF, host.to_context(), EVMC_SHANGHAI, &msg, k.data(), k.size());
        long long g = GBIG - res.gas_left; pred = -1; word0 = -1;
        if (res.status_code == 0 && res.output_data && res.output_size >= 32) {
            pred = ((int32_t)res.output_data[28] << 24) | ((int32_t)res.output_data[29] << 16) | ((int32_t)res.output_data[30] << 8) | (int32_t)res.output_data[31];
            long long m = 0; for (int b = 24; b < 32; ++b) m = (m << 8) | res.output_data[b]; word0 = m;
        } else if (res.status_code != 0) printf("  [EXECUTE ERROR] status=%d\n", (int)res.status_code);
        if (res.release) res.release(&res);
        return g;
    };

    // CHECK 1: prediction equality on 20 images
    int ok = 0; int dummy; long long w0;
    for (int im = 0; im < std::min(N_RUN, 20); ++im) {
        std::vector<uint8_t> cd; int cpp_pred, pS; buildCalldata(im, cd, cpp_pred);
        setAcc(EVMC_ACCESS_WARM); run(codeS, cd, pS, w0);
        if (pS == cpp_pred) ok++; else printf("  [MISMATCH] im=%d C++=%d SWAR=%d\n", im, cpp_pred, pS);
    }
    printf("[CHECK 1] prediction equality (C++ reference == SWAR-EVM): %d/20 %s\n", ok, ok == 20 ? "PASSED" : "FAILED");
    if (ok != 20) { printf("*** CHECK 1 FAILED -> no measurement ***\n"); vm->destroy(vm); dlclose(h); return 1; }

    // MSIZE, gas, repeat
    std::vector<uint8_t> cd0; int cp0; buildCalldata(0, cd0, cp0);
    long long ms_c, ms_w;
    setAcc(EVMC_ACCESS_COLD); run(codeSm, cd0, dummy, ms_c);
    setAcc(EVMC_ACCESS_WARM); run(codeSm, cd0, dummy, ms_w);
    bool g5 = (ms_c == ms_w) && (ms_c == MEM_END);
    printf("[CHECK 5] MSIZE cold==warm==map: %lld/%lld/%d %s\n", ms_c, ms_w, MEM_END, g5 ? "PASSED" : "FAILED");
    long long mem = memgas(ms_c);
    long long cold1, cold2, warm;
    setAcc(EVMC_ACCESS_COLD); cold1 = run(codeS, cd0, dummy, w0);
    setAcc(EVMC_ACCESS_COLD); cold2 = run(codeS, cd0, dummy, w0);
    setAcc(EVMC_ACCESS_WARM); warm = run(codeS, cd0, dummy, w0);
    bool g6 = (cold1 == cold2);
    printf("[CHECK 6] repeat run identical: %s\n", g6 ? "OK" : "MISMATCH");
    long long sload_c = (long long)slots * 2100, sload_w = (long long)slots * 100;
    bool g2 = (cold1 - warm == (long long)slots * 2000);
    printf("[CHECK 2] cold-warm==slots*2000 (%lld): %lld %s\n", (long long)slots * 2000, cold1 - warm, g2 ? "OK" : "MISMATCH");
    long long h_c = cold1 - sload_c - mem, h_w = warm - sload_w - mem;
    bool g3 = (h_c == h_w);
    printf("[CHECK 3] compute_cold==compute_warm: %lld/%lld %s\n", h_c, h_w, g3 ? "OK" : "MISMATCH");
    bool g4 = (h_c == predCompute);
    printf("[CHECK 4] static prediction == measured compute: %lld/%lld diff=%lld %s\n", predCompute, h_c, h_c - predCompute, g4 ? "PASSED" : "FAILED");
    long long mn = -1, mx = -1;
    for (int im = 0; im < std::min(N_RUN, 5); ++im) { std::vector<uint8_t> cd; int cp; buildCalldata(im, cd, cp); setAcc(EVMC_ACCESS_WARM); long long g = run(codeS, cd, dummy, w0); if (mn < 0 || g < mn) mn = g; if (g > mx) mx = g; }
    printf("[INFO  ] gas spread (5 images, warm): %lld..%lld (diff %lld, branchless -> 0 expected)\n", mn, mx, mx - mn);

    printf("\n================ PACKED-MULTIPLICATION SCALAR BASELINE (Path S, unrolled, single image) ================\n");
    printf("  path     |  SLOAD        |  memory(MSIZE) | compute(pure) |  TOTAL\n");
    printf("  ---------+---------------+----------------+---------------+---------------\n");
    printf("  S  cold  | %13lld | %14lld | %13lld | %13lld\n", sload_c, mem, h_c, cold1);
    printf("  S  warm  | %13lld | %14lld | %13lld | %13lld\n", sload_w, mem, h_w, warm);
    printf("  compute per multiply-accumulate (%d MACs): %.2f gas including the loader | %.2f gas without the loader (%lld)\n",
           MACS, (double)h_c / MACS, (double)(h_c - loaderGas) / MACS, loaderGas);
    printf("  anchors from the other programs: Path C unrolled compute 10,408,685 | Path B unrolled compute 446,833 | Path B looped compute 2,117,326\n");
    printf("  RATIOS: C/S = %.2fx | B_unrolled/S = %.3fx | B_looped/S = %.2fx\n", 10408685.0 / h_c, 446833.0 / h_c, 2117326.0 / h_c);
    if (g2 && g3 && g4 && g5 && g6) printf("\n*** ALL CHECKS PASSED -> numbers are valid ***\n");
    else printf("\n*** AT LEAST ONE CHECK FAILED -> numbers are UNRELIABLE, do NOT record ***\n");
    vm->destroy(vm); dlclose(h); return 0;
#endif
}
