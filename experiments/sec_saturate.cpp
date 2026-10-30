// sec_saturate.cpp — SATURATING ACCUMULATION TESTS of the five instructions that add products or lanes:
// XMACC (0xcb), XMACC_S (0xe5), XMACC_N (0xe6), XMACC_N16 (0xe7) and XSUM32 (0xcd).
//
// Specification under test: every product is added to its 32-bit lane with a saturating addition,
// the exact sum clamped to [-2^31, 2^31 - 1]; XSUM32 sums the eight lanes exactly and clamps the sum.
// A product of two int16 values fits in 32 bits, so the clamp is the only point at which a lane can
// leave the exact arithmetic. The tests drive every instruction into both clamps with hand-chosen
// lane values and compare the lanes with the expected constants:
//   T1  XMACC: lane at 2^31-5 plus a product of 100 -> 2^31-1; lane at -2^31+5 minus 100 -> -2^31;
//       a lane that stays in range is exact
//   T2  XMACC_S: the same three cases through the strided read (header 1 x 8, strides 8 and 1)
//   T3  XMACC_N: three groups of (-32768)*(-32768) = 2^30 from an accumulator of 0 -> 2^31-1 after the
//       second group and unchanged by the third; three groups of (-32768)*32767 -> -2^31
//   T4  XMACC_N16: one word whose pair products are both 2^30 (the one case in which the pairwise sum of
//       vpmaddwd would wrap) -> 2^31-1; a mixed pair (2^30 then -(2^30-2^15)) from 2^31-1 -> stays, then
//       the second product is applied to the clamped value
//   T5  XSUM32: eight lanes of 2^31-1 -> 2^31-1; eight lanes of -2^31 -> -2^31; four of each -> -4
//       (exact: 4*(2^31-1) + 4*(-2^31) = -4); lanes that sum inside the range are exact
//   T6  XMACC_N over 16 groups equals 16 XMACC steps on the same data (the clamp is reached in every lane)
//   T7  gas of every program equals the static count (saturation changes no price)
// The output is deterministic and is compared between the AVX2 build and the scalar-fallback build.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>   // a valid (empty) host interface; no host call is made by this program
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
static evmc::MockedHost g_host;

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
    void RET(uint64_t off, uint64_t size) { PUSH(size); PUSH(off); op(0xF3, 0); }
};
struct Res { evmc_status_code st; long long gas; std::vector<uint8_t> out; };
static Res run(const Code& k, int64_t gas) {
    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = gas;
    auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_SHANGHAI, &msg, k.c.data(), k.c.size());
    Res o; o.st = r.status_code; o.gas = gas - r.gas_left;
    if (r.output_data) o.out.assign(r.output_data, r.output_data + r.output_size);
    if (r.release) r.release(&r);
    return o;
}
static const int32_t MAXI = INT32_MAX, MINI = INT32_MIN;
// big-endian words as MLOAD reads them: int16 lane j at bytes 31-2j, 30-2j; int32 lane j at bytes 31-4j .. 28-4j
static void lanes16(uint8_t* w, const int16_t* v, int n = 8) { std::memset(w, 0, 32); for (int j = 0; j < n; ++j) { uint16_t u = (uint16_t)v[j]; w[31 - 2 * j] = u & 0xFF; w[30 - 2 * j] = (u >> 8) & 0xFF; } }
static void lanes32(uint8_t* w, const int32_t* v) { std::memset(w, 0, 32); for (int j = 0; j < 8; ++j) { uint32_t u = (uint32_t)v[j]; for (int k = 0; k < 4; ++k) w[31 - 4 * j - k] = (uint8_t)((u >> (8 * k)) & 0xFF); } }
static int32_t lane32(const uint8_t* w, int j) { uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= (uint32_t)w[31 - 4 * j - k] << (8 * k); return (int32_t)v; }
static long long memgas(uint64_t top) { long long w = (long long)((top + 31) / 32); return 3 * w + (w * w) / 512; }

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== SATURATING ACCUMULATION TESTS ===\n");
    int pass = 0, total = 0, gasok = 0, gastot = 0;
    auto report = [&](const char* name, bool ok, const char* detail) { ++total; pass += ok; std::printf("[%s] %-4s %s\n", ok ? "PASS" : "FAIL", name, detail); };
    auto lanes_ok = [&](const Res& r, const int32_t* exp, size_t off = 0) { if (r.st != EVMC_SUCCESS || r.out.size() < off + 32) return false; for (int j = 0; j < 8; ++j) if (lane32(r.out.data() + off, j) != exp[j]) return false; return true; };
    auto gascheck = [&](const Code& k, const Res& r, uint64_t top) { ++gastot; if (r.gas == k.sgas + memgas(top)) ++gasok; };
    const uint64_t OUT = 0x400;

    // ---------------- T1: XMACC on the stack ----------------
    {   int16_t a[8] = {10, -10, 7, 1, 2, 3, 4, 5}, b[8] = {10, 10, 7, 1, 2, 3, 4, 5};              // products 100, -100, 49, 1, 4, 9, 16, 25
        int32_t acc[8] = {MAXI - 5, MINI + 5, 123456, 0, -4, MAXI, MINI, 1000};
        int32_t exp[8] = {MAXI, MINI, 123505, 1, 0, MAXI, MINI + 16, 1025};
        uint8_t wa[32], wb[32], wacc[32]; lanes16(wa, a); lanes16(wb, b); lanes32(wacc, acc);
        Code k; k.W32(wacc); k.W32(wb); k.W32(wa); k.op(0xCB, 11); k.PUSH(OUT); k.MSTORE(); k.RET(OUT, 32);
        Res r = run(k, GAS_BIG); bool ok = lanes_ok(r, exp); gascheck(k, r, OUT + 32);
        char d[200]; std::snprintf(d, sizeof d, "XMACC: lanes clamp to 2^31-1 and -2^31, in-range lanes exact: %s (lane0=%d lane1=%d lane2=%d)", ok ? "yes" : "no", r.out.size() >= 32 ? lane32(r.out.data(), 0) : 0, r.out.size() >= 32 ? lane32(r.out.data(), 1) : 0, r.out.size() >= 32 ? lane32(r.out.data(), 2) : 0);
        report("T1", ok, d); }

    // ---------------- T2: XMACC_S through the strided read ----------------
    {   // memory: header at 0 (rows 1, cols 8, stride_row 8, stride_col 1), data at 32 (8 int16, little-endian
        // 16-bit elements as the instruction reads them), input at 0x100 (8 int16 LE), accumulator at 0x200 (8 int32 LE)
        int16_t m[8] = {10, -10, 7, 1, 2, 3, 4, 5}, in[8] = {10, 10, 7, 1, 2, 3, 4, 5};
        int32_t acc[8] = {MAXI - 5, MINI + 5, 123456, 0, -4, MAXI, MINI, 1000};
        int32_t exp[8] = {MAXI, MINI, 123505, 1, 0, MAXI, MINI + 16, 1025};
        uint8_t hdr[32]; std::memset(hdr, 0, 32); uint64_t f[4] = {1, 8, 8, 1}; for (int i = 0; i < 4; ++i) std::memcpy(hdr + 8 * i, &f[i], 8);
        uint8_t dat[32]; std::memset(dat, 0, 32); for (int j = 0; j < 8; ++j) std::memcpy(dat + 2 * j, &m[j], 2);
        uint8_t inw[32]; std::memset(inw, 0, 32); for (int j = 0; j < 8; ++j) std::memcpy(inw + 2 * j, &in[j], 2);
        uint8_t accw[32]; for (int j = 0; j < 8; ++j) std::memcpy(accw + 4 * j, &acc[j], 4);
        Code k; k.W32(hdr); k.PUSH(0); k.MSTORE(); k.W32(dat); k.PUSH(32); k.MSTORE(); k.W32(inw); k.PUSH(0x100); k.MSTORE(); k.W32(accw); k.PUSH(0x200); k.MSTORE();
        k.PUSH(0); k.PUSH(0); k.PUSH(0x200); k.PUSH(0x100); k.PUSH(0); k.op(0xE5, 14);   // start_col, row, off_acc, off_inp, off_mat
        k.RET(0x200, 32);
        Res r = run(k, GAS_BIG); bool ok = r.st == EVMC_SUCCESS && r.out.size() == 32;
        if (ok) for (int j = 0; j < 8; ++j) { int32_t v; std::memcpy(&v, r.out.data() + 4 * j, 4); if (v != exp[j]) ok = false; }
        gascheck(k, r, 0x220);
        char d[160]; std::snprintf(d, sizeof d, "XMACC_S: the same three cases through the header-mediated read: %s", ok ? "yes" : "no"); report("T2", ok, d); }

    // ---------------- T3: XMACC_N, three groups ----------------
    {   const uint64_t W = 0, X = 0x100;
        int16_t w[8], x[8]; for (int j = 0; j < 8; ++j) { w[j] = -32768; x[j] = (j < 4) ? -32768 : 32767; }   // lanes 0..3: +2^30 per group, lanes 4..7: -(2^30-2^15)
        uint8_t ww[32], xw[32], accw[32]; lanes16(ww, w); lanes16(xw, x); std::memset(accw, 0, 32);
        Code k; for (int g = 0; g < 3; ++g) { k.W32(ww); k.PUSH(W + 32 * g); k.MSTORE(); k.W32(xw); k.PUSH(X + 32 * g); k.MSTORE(); }
        k.W32(accw); k.PUSH(X); k.PUSH(W); k.PUSH(3); k.op(0xE6, 33); k.PUSH(OUT); k.MSTORE(); k.RET(OUT, 32);
        int32_t exp[8]; for (int j = 0; j < 8; ++j) exp[j] = (j < 4) ? MAXI : MINI;   // 3 * 2^30 clamps high; 3 * -(2^30-2^15) = -3221176320 clamps low
        Res r = run(k, GAS_BIG); bool ok = lanes_ok(r, exp); gascheck(k, r, OUT + 32);
        char d[160]; std::snprintf(d, sizeof d, "XMACC_N: 3 x (+2^30) -> 2^31-1 and 3 x -(2^30-2^15) -> -2^31: %s", ok ? "yes" : "no"); report("T3", ok, d); }

    // ---------------- T4: XMACC_N16, the pairwise case ----------------
    {   const uint64_t W = 0, X = 0x100;
        int16_t w[16], x[16];
        for (int j = 0; j < 16; ++j) { w[j] = -32768; x[j] = -32768; }                  // every product 2^30: pair sum 2^31 (would wrap in vpmaddwd)
        uint8_t ww[32], xw[32], accw[32]; lanes16(ww, w, 16); lanes16(xw, x, 16); std::memset(accw, 0, 32);
        Code k; k.W32(ww); k.PUSH(W); k.MSTORE(); k.W32(xw); k.PUSH(X); k.MSTORE();
        k.W32(accw); k.PUSH(X); k.PUSH(W); k.PUSH(1); k.op(0xE7, 22); k.PUSH(OUT); k.MSTORE(); k.RET(OUT, 32);
        int32_t exp[8]; for (int j = 0; j < 8; ++j) exp[j] = MAXI;
        Res r = run(k, GAS_BIG); bool ok1 = lanes_ok(r, exp); gascheck(k, r, OUT + 32);
        // mixed pair from a clamped accumulator: lane k gets +2^30 then -(2^30-2^15): sat(sat(MAXI + 2^30) - (2^30-2^15)) = MAXI - 2^30 + 2^15
        for (int j = 0; j < 16; ++j) { w[j] = -32768; x[j] = (j % 2 == 0) ? -32768 : 32767; }
        int32_t accv[8]; for (int j = 0; j < 8; ++j) accv[j] = MAXI;
        lanes16(ww, w, 16); lanes16(xw, x, 16); lanes32(accw, accv);
        Code k2; k2.W32(ww); k2.PUSH(W); k2.MSTORE(); k2.W32(xw); k2.PUSH(X); k2.MSTORE();
        k2.W32(accw); k2.PUSH(X); k2.PUSH(W); k2.PUSH(1); k2.op(0xE7, 22); k2.PUSH(OUT); k2.MSTORE(); k2.RET(OUT, 32);
        int32_t exp2[8]; for (int j = 0; j < 8; ++j) exp2[j] = MAXI - (1 << 30) + (1 << 15);
        Res r2 = run(k2, GAS_BIG); bool ok2 = lanes_ok(r2, exp2); gascheck(k2, r2, OUT + 32);
        char d[200]; std::snprintf(d, sizeof d, "XMACC_N16: pair of 2^30 products -> 2^31-1: %s; products of a pair applied in turn to a clamped lane: %s", ok1 ? "yes" : "no", ok2 ? "yes" : "no"); report("T4", ok1 && ok2, d); }

    // ---------------- T5: XSUM32 ----------------
    {   auto sum_of = [&](const int32_t* v, int32_t& got, Code& k) { uint8_t w[32]; lanes32(w, v); k.W32(w); k.op(0xCD, 6); k.PUSH(OUT); k.MSTORE(); k.RET(OUT, 32);
            Res r = run(k, GAS_BIG); gascheck(k, r, OUT + 32); if (r.st != EVMC_SUCCESS || r.out.size() != 32) return false;
            uint32_t u = 0; for (int i = 28; i < 32; ++i) u = (u << 8) | r.out[i]; got = (int32_t)u;
            for (int i = 0; i < 28; ++i) if (r.out[i] != 0) return false; return true; };
        int32_t hi[8], lo[8], mix[8], ex[8] = {MAXI, -MAXI, 1, 2, 3, -4, 100000, -100000};
        for (int j = 0; j < 8; ++j) { hi[j] = MAXI; lo[j] = MINI; mix[j] = (j < 4) ? MAXI : MINI; }
        int32_t g1 = 0, g2 = 0, g3 = 0, g4 = 0; Code k1, k2, k3, k4;
        bool ok = sum_of(hi, g1, k1) && g1 == MAXI && sum_of(lo, g2, k2) && g2 == MINI && sum_of(mix, g3, k3) && g3 == -4 && sum_of(ex, g4, k4) && g4 == 2;
        char d[200]; std::snprintf(d, sizeof d, "XSUM32: 8 x (2^31-1) -> %d, 8 x (-2^31) -> %d, 4 + 4 -> %d (exact -4), in-range sum -> %d (exact 2): %s", g1, g2, g3, g4, ok ? "yes" : "no"); report("T5", ok, d); }

    // ---------------- T6: XMACC_N == 16 x XMACC with the clamp reached ----------------
    {   const uint64_t W = 0, X = 0x400, OUT2 = 0x800;
        uint32_t rnd = 777u; auto rnd16 = [&]() { rnd = rnd * 1664525u + 1013904223u; return (int16_t)(rnd >> 16); };
        int16_t wl[16][8], xl[16][8]; uint8_t ww[16][32], xw[16][32], accw[32];
        for (int g = 0; g < 16; ++g) { for (int j = 0; j < 8; ++j) { wl[g][j] = (int16_t)((rnd16() & 0x3FFF) + 0x4000); const int16_t m = (int16_t)((rnd16() & 0x3FFF) + 0x4000); xl[g][j] = (j % 2 == 0) ? m : (int16_t)-m; } lanes16(ww[g], wl[g]); lanes16(xw[g], xl[g]); }   // products of at least 2^28: even lanes clamp high, odd lanes low
        std::memset(accw, 0, 32);
        Code kn; for (int g = 0; g < 16; ++g) { kn.W32(ww[g]); kn.PUSH(W + 32 * g); kn.MSTORE(); kn.W32(xw[g]); kn.PUSH(X + 32 * g); kn.MSTORE(); }
        Code kb = kn;
        kn.W32(accw); kn.PUSH(X); kn.PUSH(W); kn.PUSH(16); kn.op(0xE6, 11 * 16); kn.PUSH(OUT2); kn.MSTORE(); kn.RET(OUT2, 32);
        kb.W32(accw); for (int g = 0; g < 16; ++g) { kb.PUSH(X + 32 * g); kb.MLOAD(); kb.PUSH(W + 32 * g); kb.MLOAD(); kb.op(0xCB, 11); } kb.PUSH(OUT2); kb.MSTORE(); kb.RET(OUT2, 32);
        Res rn = run(kn, GAS_BIG), rb = run(kb, GAS_BIG); gascheck(kn, rn, OUT2 + 32); gascheck(kb, rb, OUT2 + 32);
        int32_t ref[8]; int clamped = 0; for (int j = 0; j < 8; ++j) { int32_t a = 0; for (int g = 0; g < 16; ++g) { int64_t s = (int64_t)a + (int32_t)wl[g][j] * (int32_t)xl[g][j]; a = s > MAXI ? MAXI : (s < MINI ? MINI : (int32_t)s); } ref[j] = a; if (a == MAXI || a == MINI) ++clamped; }
        bool ok = lanes_ok(rn, ref) && lanes_ok(rb, ref) && rn.out == rb.out;
        char d[200]; std::snprintf(d, sizeof d, "XMACC_N over 16 groups == 16 x XMACC == saturating reference (%d of 8 lanes at a clamp): %s", clamped, ok ? "yes" : "no"); report("T6", ok, d); }

    // ---------------- T7: gas ----------------
    { char d[120]; std::snprintf(d, sizeof d, "gas of every program equals the static count: %d of %d", gasok, gastot); report("T7", gasok == gastot, d); }

    std::printf("\n--- SUMMARY --- %d/%d passed%s\n", pass, total, pass == total ? "  *** ALL CHECKS PASSED ***" : "  *** INSPECT ***");
    return pass == total ? 0 : 1;
}
