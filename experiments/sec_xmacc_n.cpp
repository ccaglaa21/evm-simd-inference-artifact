// sec_xmacc_n.cpp — SECURITY TESTS for the row-level instruction XMACC_N (0xe6).
//
// XMACC_N pops n (group count), off_w and off_x and accumulates n consecutive 8-lane groups into the
// accumulator word that stays on the stack. The tests check the guards of the instruction:
//   T1  n = 0            -> SUCCESS, accumulator unchanged, no memory expansion (MSIZE as expected)
//   T2  n = 2^31         -> clean OUT_OF_GAS (n limited to 31 bits, before any memory access)
//   T3  n = 2^255        -> clean OUT_OF_GAS (high limbs non-zero)
//   T4  gas < 11 n       -> clean OUT_OF_GAS (the per-group gas is charged before the memory check)
//   T5  off_w = 2^64 - 8 -> clean OUT_OF_GAS (offset + 32 n overflows the address space)
//   T6  off_x = 2^200    -> clean OUT_OF_GAS (offset outside the 64-bit range)
//   T7  n*32 beyond gas  -> clean OUT_OF_GAS from the memory-expansion charge, no host memory access
//   T8  off_w == off_x   -> SUCCESS, lanes equal the squares (overlapping operands are legal)
//   T9  cross-consistency -> XMACC_N over 40 groups of signed random data equals 40 XMACC steps
//                            (same lane arithmetic, 32-bit wrap included)
//   T10 gas accounting   -> gas of T9 equals the static count (11 per group plus the other opcodes)
//   T11 XMACC_N16 n=2^31 -> clean OUT_OF_GAS (same guards as XMACC_N)
//   T12 XMACC_N16 over 20 words equals the C++ reference (pairwise sums over the 16 lanes of a word, 32-bit wrap)
//       and its gas equals the static count (22 per 16-element group)
// A SUCCESS where OUT_OF_GAS is expected would mean a guard is missing. The program prints one line
// per test and a summary; the output is deterministic and is compared between the two builds.
#include <evmc/evmc.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static evmc_vm* VM;
static const int64_t GAS_BIG = 5000000000LL;
struct Code {
    std::vector<uint8_t> c; long long sgas = 0;
    void op(uint8_t o, long long g) { c.push_back(o); sgas += g; }
    void P1(uint8_t v) { op(0x60, 3); c.push_back(v); }
    void PUSH(uint64_t v) { if (v < 256) { P1((uint8_t)v); return; } int n = 0; for (uint64_t t = v; t; t >>= 8) ++n;
        op((uint8_t)(0x5F + n), 3); for (int b = n - 1; b >= 0; --b) c.push_back((uint8_t)((v >> (8 * b)) & 0xFF)); }
    void W32(const uint8_t* w) { op(0x7F, 3); for (int i = 0; i < 32; ++i) c.push_back(w[i]); }
    void MSTORE() { op(0x52, 3); } void MLOAD() { op(0x51, 3); }
    void XMACC_N(uint64_t n) { op(0xE6, 11 * (long long)n); }
    void XMACC() { op(0xCB, 11); }
    void RET(uint64_t off, uint64_t size) { PUSH(size); PUSH(off); op(0xF3, 0); }
};
struct Res { evmc_status_code st; long long gas; std::vector<uint8_t> out; };
#ifdef LOCAL_TEST
static int dumpno = 0;
static Res run(const Code& k, int64_t gas) {   // dump mode: write the bytecode, report nothing
    char f[32]; std::snprintf(f, sizeof f, "sx_%d.bin", ++dumpno); FILE* o = std::fopen(f, "wb"); std::fwrite(k.c.data(), 1, k.c.size(), o); std::fclose(o);
    std::printf("dumped %s sgas=%lld gas=%lld\n", f, k.sgas, (long long)gas); Res r; r.st = EVMC_FAILURE; r.gas = 0; return r;
}
#else
static Res run(const Code& k, int64_t gas) {
    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = gas;
    auto r = VM->execute(VM, nullptr, nullptr, EVMC_SHANGHAI, &msg, k.c.data(), k.c.size());
    Res o; o.st = r.status_code; o.gas = gas - r.gas_left;
    if (r.output_data) o.out.assign(r.output_data, r.output_data + r.output_size);
    if (r.release) r.release(&r);
    return o;
}
#endif
// word with eight int16 lanes (lane j at bytes 31-2j, 30-2j), as MLOAD reads it
static void lanes16(uint8_t* w, const int16_t* v) { std::memset(w, 0, 32); for (int j = 0; j < 8; ++j) { uint16_t u = (uint16_t)v[j]; w[31 - 2 * j] = u & 0xFF; w[30 - 2 * j] = (u >> 8) & 0xFF; } }
static int32_t lane32(const uint8_t* w, int j) { uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= (uint32_t)w[31 - 4 * j - k] << (8 * k); return (int32_t)v; }   // w: big-endian word bytes
static uint32_t rnd = 12345u; static int16_t rnd16() { rnd = rnd * 1664525u + 1013904223u; return (int16_t)(rnd >> 16); }

int main() {
#ifndef LOCAL_TEST
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== XMACC_N SECURITY TESTS ===\n");
    int pass = 0, total = 0;
    auto report = [&](const char* name, bool ok, const char* detail) { ++total; pass += ok; std::printf("[%s] %-4s %s\n", ok ? "PASS" : "FAIL", name, detail); };
    const int W = 0, X = 32 * 64, OUT = 32 * 200;       // memory map: 64 weight words, 64 input words
    // common prelude: 64 groups of random signed lanes for W and X, accumulator lanes = 1..8
    int16_t wl[64][8], xl[64][8]; uint8_t ww[64][32], xw[64][32], accw[32];
    for (int g = 0; g < 64; ++g) { for (int j = 0; j < 8; ++j) { wl[g][j] = rnd16(); xl[g][j] = rnd16(); } lanes16(ww[g], wl[g]); lanes16(xw[g], xl[g]);
        for (int b = 0; b < 16; ++b) { ww[g][b] = (uint8_t)rnd16(); xw[g][b] = (uint8_t)rnd16(); } }   // bytes 0..15 (lanes 8..15) are used by XMACC_N16 only
    std::memset(accw, 0, 32); for (int j = 0; j < 8; ++j) accw[31 - 4 * j] = (uint8_t)(j + 1);   // lane j = j + 1 (big-endian word bytes)
    auto prelude = [&](Code& k) { for (int g = 0; g < 64; ++g) { k.W32(ww[g]); k.PUSH(W + 32 * g); k.MSTORE(); k.W32(xw[g]); k.PUSH(X + 32 * g); k.MSTORE(); } };
    auto one = [&](uint64_t n, const uint8_t* n_word /*32 bytes or null*/, uint64_t offw, const uint8_t* offw_word, uint64_t offx, const uint8_t* offx_word) {
        Code k; prelude(k); k.W32(accw);
        if (offx_word) k.W32(offx_word); else k.PUSH(offx);
        if (offw_word) k.W32(offw_word); else k.PUSH(offw);
        if (n_word) { k.W32(n_word); k.op(0xE6, 0); } else { k.PUSH(n); k.XMACC_N(n); }
        k.PUSH(OUT); k.MSTORE(); k.op(0x59, 2); k.PUSH(OUT + 32); k.MSTORE(); k.RET(OUT, 64); return k; };
    auto bigword = [](int bit) { static uint8_t w[32]; std::memset(w, 0, 32); w[31 - bit / 8] = (uint8_t)(1u << (bit % 8)); return (const uint8_t*)w; };

    std::printf("running T1..T10\n");
    // T1: n = 0
    { Code k = one(0, nullptr, W, nullptr, X, nullptr); Res r = run(k, GAS_BIG);
      bool ok = r.st == EVMC_SUCCESS && r.out.size() == 64 && std::memcmp(r.out.data(), accw, 32) == 0;
      uint64_t ms = 0; if (r.out.size() == 64) for (int i = 24; i < 32; ++i) ms = (ms << 8) | r.out[32 + i];
      ok = ok && ms == (uint64_t)(OUT + 32);                     // MSIZE is read after the first store, before the second
      char d[160]; std::snprintf(d, sizeof d, "n=0: status=%d output words=%zu accumulator unchanged=%s MSIZE=%llu (expected %d)", (int)r.st, r.out.size() / 32, (r.out.size() == 64 && std::memcmp(r.out.data(), accw, 32) == 0) ? "yes" : "no", (unsigned long long)ms, OUT + 32); report("T1", ok, d); }
    // T2: n = 2^31
    { Code k = one(0x80000000ULL, nullptr, W, nullptr, X, nullptr); Res r = run(k, GAS_BIG); char d[96]; std::snprintf(d, sizeof d, "n=2^31: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("T2", r.st == EVMC_OUT_OF_GAS, d); }
    // T3: n = 2^255
    { Code k = one(0, bigword(255), W, nullptr, X, nullptr); Res r = run(k, GAS_BIG); char d[96]; std::snprintf(d, sizeof d, "n=2^255: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("T3", r.st == EVMC_OUT_OF_GAS, d); }
    // T4: gas smaller than 11 n (n = 10^8 groups -> 1.1e9 gas), budget 10^8
    { Code k = one(100000000ULL, nullptr, W, nullptr, X, nullptr); Res r = run(k, 100000000LL); char d[128]; std::snprintf(d, sizeof d, "n=1e8 with 1e8 gas: status=%d gas used=%lld (OUT_OF_GAS=3 expected, no memory touched)", (int)r.st, r.gas); report("T4", r.st == EVMC_OUT_OF_GAS, d); }
    // T5: off_w = 2^64 - 8
    { Code k = one(4, nullptr, 0xFFFFFFFFFFFFFFF8ULL, nullptr, X, nullptr); Res r = run(k, GAS_BIG); char d[96]; std::snprintf(d, sizeof d, "off_w=2^64-8, n=4: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("T5", r.st == EVMC_OUT_OF_GAS, d); }
    // T6: off_x = 2^200
    { Code k = one(4, nullptr, W, nullptr, 0, bigword(200)); Res r = run(k, GAS_BIG); char d[96]; std::snprintf(d, sizeof d, "off_x=2^200, n=4: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("T6", r.st == EVMC_OUT_OF_GAS, d); }
    // T7: n = 2^30 groups (32 GiB of memory expansion) with a 10^11 gas budget: the group gas fits, the expansion does not
    { Code k = one(1ULL << 30, nullptr, W, nullptr, X, nullptr); Res r = run(k, 100000000000LL); char d[128]; std::snprintf(d, sizeof d, "n=2^30 (32 GiB) with 1e11 gas: status=%d (OUT_OF_GAS=3 expected from the memory charge)", (int)r.st); report("T7", r.st == EVMC_OUT_OF_GAS, d); }
    // T8: overlapping operands off_w == off_x (squares)
    { Code k = one(8, nullptr, W, nullptr, W, nullptr); Res r = run(k, GAS_BIG); bool ok = r.st == EVMC_SUCCESS && r.out.size() == 64;
      if (ok) for (int j = 0; j < 8; ++j) { int32_t ref = j + 1; for (int g = 0; g < 8; ++g) ref = (int32_t)((uint32_t)ref + (uint32_t)((int32_t)wl[g][j] * (int32_t)wl[g][j])); if (lane32(r.out.data(), j) != ref) ok = false; }
      char d[96]; std::snprintf(d, sizeof d, "off_w == off_x, n=8: status=%d lanes == squares: %s", (int)r.st, ok ? "yes" : "no"); report("T8", ok, d); }
    // T9 + T10: 40 groups, XMACC_N against 40 XMACC steps and against the C++ reference; gas against the static count
    { Code kn = one(40, nullptr, W, nullptr, X, nullptr); Res rn = run(kn, GAS_BIG);
      Code kb; prelude(kb); kb.W32(accw); for (int g = 0; g < 40; ++g) { kb.PUSH(X + 32 * g); kb.MLOAD(); kb.PUSH(W + 32 * g); kb.MLOAD(); kb.XMACC(); }
      kb.PUSH(OUT); kb.MSTORE(); kb.op(0x59, 2); kb.PUSH(OUT + 32); kb.MSTORE(); kb.RET(OUT, 64); Res rb = run(kb, GAS_BIG);
      bool ok = rn.st == EVMC_SUCCESS && rb.st == EVMC_SUCCESS && rn.out.size() == 64 && rb.out.size() == 64 && std::memcmp(rn.out.data(), rb.out.data(), 32) == 0;
      bool okref = ok; if (ok) for (int j = 0; j < 8; ++j) { int32_t ref = j + 1; for (int g = 0; g < 40; ++g) ref = (int32_t)((uint32_t)ref + (uint32_t)((int32_t)wl[g][j] * (int32_t)xl[g][j])); if (lane32(rn.out.data(), j) != ref) okref = false; }
      char d[160]; std::snprintf(d, sizeof d, "40 groups: XMACC_N == 40 x XMACC: %s, == C++ reference (32-bit wrap): %s", ok ? "yes" : "no", okref ? "yes" : "no"); report("T9", ok && okref, d);
      long long memw = (OUT + 64 + 31) / 32; long long memgas = 3 * memw + (memw * memw) / 512;
      long long pred = kn.sgas + memgas; char e[160]; std::snprintf(e, sizeof e, "gas measured %lld, static count %lld (11 gas x 40 groups included): %s", rn.gas, pred, rn.gas == pred ? "equal" : "DIFFERENT"); report("T10", rn.gas == pred, e); }
    // T11, T12: the 16-lane variant XMACC_N16 (0xe7): guards and reference (pairwise sums, 22 gas per group)
    { Code k; prelude(k); k.W32(accw); k.PUSH(X); k.PUSH(W); k.PUSH(0x80000000ULL); k.op(0xE7, 0); k.PUSH(OUT); k.MSTORE(); k.RET(OUT, 32);
      Res r = run(k, GAS_BIG); char d[96]; std::snprintf(d, sizeof d, "XMACC_N16 n=2^31: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("T11", r.st == EVMC_OUT_OF_GAS, d); }
    { Code k; prelude(k); k.W32(accw); k.PUSH(X); k.PUSH(W); k.PUSH(20); k.op(0xE7, 22 * 20); k.PUSH(OUT); k.MSTORE(); k.RET(OUT, 32);
      Res r = run(k, GAS_BIG); bool ok = r.st == EVMC_SUCCESS && r.out.size() == 32;
      if (ok) for (int kk = 0; kk < 8; ++kk) { int32_t ref = kk + 1;
          for (int g = 0; g < 20; ++g) { uint32_t sum = 0; for (int h = 0; h < 2; ++h) { int j = 2 * kk + h;
              int16_t w = (int16_t)(ww[g][31 - 2 * j] | (ww[g][30 - 2 * j] << 8)); int16_t x = (int16_t)(xw[g][31 - 2 * j] | (xw[g][30 - 2 * j] << 8));
              sum += (uint32_t)((int32_t)w * (int32_t)x); } ref = (int32_t)((uint32_t)ref + sum); }
          if (lane32(r.out.data(), kk) != ref) ok = false; }
      long long memw = (OUT + 32 + 31) / 32; long long pred = k.sgas + 3 * memw + (memw * memw) / 512;
      char d[160]; std::snprintf(d, sizeof d, "XMACC_N16 over 20 words == C++ reference (pairwise sums over all 16 lanes, 32-bit wrap): %s | gas %lld == static %lld: %s", ok ? "yes" : "no", r.gas, pred, r.gas == pred ? "yes" : "no"); report("T12", ok && r.gas == pred, d); }
    std::printf("\n--- SUMMARY --- %d/%d passed%s\n", pass, total, pass == total ? "  *** ALL CHECKS PASSED ***" : "  *** INSPECT ***");
#ifndef LOCAL_TEST
    VM->destroy(VM); dlclose(h);
#endif
    return pass == total ? 0 : 1;
}
