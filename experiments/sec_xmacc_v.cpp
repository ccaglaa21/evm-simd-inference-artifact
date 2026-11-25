// sec_xmacc_v.cpp — TESTS for the view-based row-level instruction XMACC_V (0xea).
//
// XMACC_V pops off_a, row_a, off_b, row_b and accumulates the element-wise products of row_a of
// view A and row_b of view B into the accumulator word that stays on the stack. A view is a
// 32-byte MatrixHeader (rows, cols, stride_row, stride_col as little-endian 64-bit fields) followed
// by int16 elements, and row r is the sequence data[r*stride_row + t*stride_col], t = 0..cols-1.
//   R1  4000 random trials: shapes 1..40 columns, plain and transposed views, large accumulators
//       and extreme elements, against the C++ reference with the saturating lane addition
//   R2  XTRANSPOSE followed by XMACC_V reads a column of the stored matrix (no copy)
//   G1  cols of A != cols of B         -> clean OUT_OF_GAS (shape error)
//   G2  row_a == rows of A             -> clean OUT_OF_GAS
//   G3  row_b = 2^200                  -> clean OUT_OF_GAS
//   G4  cols = 2^31 in both headers    -> clean OUT_OF_GAS (count limited to 31 bits)
//   G5  stride_col = 2^62              -> clean OUT_OF_GAS (the highest element is outside the memory)
//   G6  off_a = 2^200                  -> clean OUT_OF_GAS
//   G7  gas below the group charge     -> clean OUT_OF_GAS
//   G8  cols = 0                       -> SUCCESS, accumulator unchanged, only the charge per call (50 gas)
//   G9  gas accounting                 -> measured gas equals the static count: 50 per call, 10 per group of eight (partial group
//                                         included) and 4 per element of a strided row
// The output is deterministic and is compared between the builds.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>
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
    void PUSH(uint64_t v) { int n = 0; for (uint64_t t = v; t; t >>= 8) ++n; if (n == 0) n = 1;
        op((uint8_t)(0x5F + n), 3); for (int b = n - 1; b >= 0; --b) c.push_back((uint8_t)((v >> (8 * b)) & 0xFF)); }
    void W32(const uint8_t* w) { op(0x7F, 3); for (int i = 0; i < 32; ++i) c.push_back(w[i]); }
    void MSTORE() { op(0x52, 3); }
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
static int32_t sat32(int32_t a, int32_t b) { int64_t s = (int64_t)a + b; return s > INT32_MAX ? INT32_MAX : (s < INT32_MIN ? INT32_MIN : (int32_t)s); }
static int32_t lane32(const uint8_t* w, int j) { uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= (uint32_t)w[31 - 4 * j - k] << (8 * k); return (int32_t)v; }
static uint32_t rnd = 20261124u; static uint32_t rnd32() { rnd = rnd * 1664525u + 1013904223u; return rnd >> 8; }
static int16_t rnd16() { uint32_t r = rnd32(); if (r % 16 == 0) return (r & 16) ? INT16_MAX : INT16_MIN; return (int16_t)(r >> 4); }

// a matrix image: header + elements, written into a byte vector at a word-aligned offset
struct Mat { uint64_t rows, cols, sr, sc; std::vector<int16_t> d; };
static void put64(std::vector<uint8_t>& m, size_t at, uint64_t v) { for (int k = 0; k < 8; ++k) m[at + k] = (uint8_t)(v >> (8 * k)); }
static size_t place(std::vector<uint8_t>& mem, const Mat& a) {
    size_t off = mem.size(); size_t bytes = 32 + 2 * a.d.size(); mem.resize(off + (bytes + 31) / 32 * 32, 0);
    put64(mem, off, a.rows); put64(mem, off + 8, a.cols); put64(mem, off + 16, a.sr); put64(mem, off + 24, a.sc);
    for (size_t i = 0; i < a.d.size(); ++i) { uint16_t u = (uint16_t)a.d[i]; mem[off + 32 + 2 * i] = u & 0xFF; mem[off + 33 + 2 * i] = u >> 8; }
    return off;
}
static void load(Code& k, const std::vector<uint8_t>& mem) { for (size_t o = 0; o < mem.size(); o += 32) { k.W32(&mem[o]); k.PUSH(o); k.MSTORE(); } }
static void accword(uint8_t* w, const int32_t* a) { for (int j = 0; j < 8; ++j) { uint32_t v = (uint32_t)a[j]; for (int k = 0; k < 4; ++k) w[31 - 4 * j - k] = (uint8_t)(v >> (8 * k)); } }
// stored matrix R x C in row-major order; plain view (R, C, C, 1) or transposed view (C, R, 1, C)
static Mat mk(uint64_t R, uint64_t C, bool transposed) { Mat m; m.d.resize(R * C); for (auto& v : m.d) v = rnd16();
    if (transposed) { m.rows = C; m.cols = R; m.sr = 1; m.sc = C; } else { m.rows = R; m.cols = C; m.sr = C; m.sc = 1; } return m; }
static int16_t el(const Mat& m, uint64_t r, uint64_t t) { return m.d[r * m.sr + t * m.sc]; }

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== XMACC_V TESTS ===\n");
    int pass = 0, total = 0;
    auto report = [&](const char* name, bool ok, const char* detail) { ++total; pass += ok; std::printf("[%s] %-3s %s\n", ok ? "PASS" : "FAIL", name, detail); };
    // one execution: memory image, accumulator, operands (as words when given), returns the accumulator
    auto exec = [&](const std::vector<uint8_t>& mem, const int32_t* acc0, uint64_t offa, uint64_t ra, uint64_t offb, uint64_t rb,
                    const uint8_t* offa_w, const uint8_t* rb_w, long long vgas, int64_t gas, Code* keep) {
        Code k; load(k, mem); uint8_t aw[32]; accword(aw, acc0); k.W32(aw);
        if (rb_w) k.W32(rb_w); else k.PUSH(rb);
        k.PUSH(offb); k.PUSH(ra);
        if (offa_w) k.W32(offa_w); else k.PUSH(offa);
        k.op(0xEA, vgas); size_t out = mem.size(); k.PUSH(out); k.MSTORE(); k.RET(out, 32);
        if (keep) *keep = k; return run(k, gas); };
    auto bigword = [](int bit) { static uint8_t w[32]; std::memset(w, 0, 32); w[31 - bit / 8] = (uint8_t)(1u << (bit % 8)); return (const uint8_t*)w; };

    // R1: random trials
    { int ok = 0, sat = 0; const int T = 4000; unsigned long long digest = 1469598103934665603ULL;
      for (int t = 0; t < T; ++t) {
          uint64_t n = 1 + rnd32() % 40, Ra = 1 + rnd32() % 6, Rb = 1 + rnd32() % 6; bool ta = rnd32() & 1, tb = rnd32() & 1;
          // plain: stored Ra x n, rows are rows. transposed: stored n x Ra, rows of the view are stored columns.
          Mat A = ta ? mk(n, Ra, true) : mk(Ra, n, false); Mat B = tb ? mk(n, Rb, true) : mk(Rb, n, false);
          std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B);
          uint64_t ra = rnd32() % A.rows, rb = rnd32() % B.rows; int32_t acc0[8], ref[8];
          for (int j = 0; j < 8; ++j) { uint32_t r = rnd32(); acc0[j] = (r % 4 == 0) ? (int32_t)(INT32_MAX - (r >> 8) % 1000000) : ((r % 4 == 1) ? (int32_t)(INT32_MIN + (r >> 8) % 1000000) : (int32_t)(r >> 6) - 100000); ref[j] = acc0[j]; }
          for (uint64_t e = 0; e < n; ++e) { int64_t s = (int64_t)ref[e % 8] + (int32_t)el(A, ra, e) * (int32_t)el(B, rb, e); if (s > INT32_MAX || s < INT32_MIN) ++sat; ref[e % 8] = sat32(ref[e % 8], (int32_t)el(A, ra, e) * (int32_t)el(B, rb, e)); }
          Res r = exec(mem, acc0, oa, ra, ob, rb, nullptr, nullptr, 0, GAS_BIG, nullptr);
          bool good = r.st == EVMC_SUCCESS && r.out.size() == 32; if (good) for (int j = 0; j < 8; ++j) if (lane32(r.out.data(), j) != ref[j]) good = false;
          ok += good; if (r.out.size() == 32) for (int i = 0; i < 32; ++i) { digest ^= r.out[i]; digest *= 1099511628211ULL; } }
      char d[200]; std::snprintf(d, sizeof d, "%d/%d random trials equal the C++ reference (%d saturating additions, output digest %016llx)", ok, T, sat, digest); report("R1", ok == T, d); }
    // R2: XTRANSPOSE then XMACC_V: column c of a stored 5 x 12 matrix against a plain 1 x 5 row
    { Mat S; S.rows = 5; S.cols = 12; S.sr = 12; S.sc = 1; S.d.resize(60); for (auto& v : S.d) v = rnd16(); Mat X = mk(1, 5, false);
      std::vector<uint8_t> mem; size_t os = place(mem, S), ox = place(mem, X); int32_t acc0[8] = {0, 0, 0, 0, 0, 0, 0, 0}; bool ok = true;
      for (uint64_t c = 0; c < 12; ++c) { Code k; load(k, mem); uint8_t aw[32]; accword(aw, acc0); k.W32(aw); k.PUSH(0); k.PUSH(ox); k.PUSH(c); k.PUSH(os);
          k.op(0xDD, 22);   /* XTRANSPOSE rewrites the header in place and leaves the offset on the stack */ k.op(0xEA, 14); size_t out = mem.size(); k.PUSH(out); k.MSTORE(); k.RET(out, 32); Res r = run(k, GAS_BIG);
          int32_t ref[8] = {0, 0, 0, 0, 0, 0, 0, 0}; for (int e = 0; e < 5; ++e) ref[e] = (int32_t)S.d[e * 12 + c] * (int32_t)X.d[e];
          if (r.st != EVMC_SUCCESS || r.out.size() != 32) ok = false; else for (int j = 0; j < 8; ++j) if (lane32(r.out.data(), j) != ref[j]) ok = false; }
      report("R2", ok, ok ? "XTRANSPOSE + XMACC_V reads each of the 12 stored columns without a copy: yes" : "XTRANSPOSE + XMACC_V column read: NO"); }
    // guards
    Mat A = mk(3, 20, false), B = mk(4, 20, false); int32_t z[8] = {1, 2, 3, 4, 5, 6, 7, 8}; char d[200];
    { Mat B2 = mk(4, 19, false); std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B2); Res r = exec(mem, z, oa, 0, ob, 0, nullptr, nullptr, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "cols 20 against cols 19: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G1", r.st == EVMC_OUT_OF_GAS, d); }
    { std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B); Res r = exec(mem, z, oa, 3, ob, 0, nullptr, nullptr, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "row_a = rows of A: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G2", r.st == EVMC_OUT_OF_GAS, d); }
    { std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B); Res r = exec(mem, z, oa, 0, ob, 0, nullptr, bigword(200), 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "row_b = 2^200: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G3", r.st == EVMC_OUT_OF_GAS, d); }
    { Mat A2 = A, B2 = B; A2.cols = B2.cols = 0x80000000ULL; std::vector<uint8_t> mem; size_t oa = place(mem, A2), ob = place(mem, B2); Res r = exec(mem, z, oa, 0, ob, 0, nullptr, nullptr, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "cols = 2^31: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G4", r.st == EVMC_OUT_OF_GAS, d); }
    { Mat A2 = A; A2.sc = 1ULL << 62; std::vector<uint8_t> mem; size_t oa = place(mem, A2), ob = place(mem, B); Res r = exec(mem, z, oa, 0, ob, 0, nullptr, nullptr, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "stride_col = 2^62: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G5", r.st == EVMC_OUT_OF_GAS, d); }
    { std::vector<uint8_t> mem; place(mem, A); size_t ob = place(mem, B); Res r = exec(mem, z, 0, 0, ob, 0, bigword(200), nullptr, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "off_a = 2^200: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G6", r.st == EVMC_OUT_OF_GAS, d); }
    { Mat A2 = A, B2 = B; A2.cols = B2.cols = 0x7fffffffULL; std::vector<uint8_t> mem; size_t oa = place(mem, A2), ob = place(mem, B2); Res r = exec(mem, z, oa, 0, ob, 0, nullptr, nullptr, 0, 1000000000LL, nullptr);
      std::snprintf(d, sizeof d, "cols = 2^31-1 with 1e9 gas: status=%d (OUT_OF_GAS=3 expected from the group charge)", (int)r.st); report("G7", r.st == EVMC_OUT_OF_GAS, d); }
    { Mat A2 = A, B2 = B; A2.cols = B2.cols = 0; std::vector<uint8_t> mem; size_t oa = place(mem, A2), ob = place(mem, B2); Code k; Res r = exec(mem, z, oa, 0, ob, 0, nullptr, nullptr, 50, GAS_BIG, &k);
      uint8_t aw[32]; accword(aw, z); long long mw = (long long)(mem.size() + 32) / 32; long long pred = k.sgas + 3 * mw + mw * mw / 512;
      bool ok = r.st == EVMC_SUCCESS && r.out.size() == 32 && std::memcmp(r.out.data(), aw, 32) == 0 && r.gas == pred;
      std::snprintf(d, sizeof d, "cols = 0: status=%d accumulator unchanged, gas %lld == static count with 50 for the call %lld: %s", (int)r.st, r.gas, pred, r.gas == pred ? "yes" : "no"); report("G8", ok, d); }
    { std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B); Code k; Res r = exec(mem, z, oa, 2, ob, 3, nullptr, nullptr, 50 + 10 * 3, GAS_BIG, &k);
      long long mw = (long long)(mem.size() + 32) / 32; long long pred = k.sgas + 3 * mw + mw * mw / 512;
      Mat T = mk(20, 3, true); std::vector<uint8_t> mem2; size_t ot = place(mem2, T), ob2 = place(mem2, B); Code k2; Res r2 = exec(mem2, z, ot, 2, ob2, 3, nullptr, nullptr, 50 + 10 * 3 + 4 * 20, GAS_BIG, &k2);
      long long mw2 = (long long)(mem2.size() + 32) / 32; long long pred2 = k2.sgas + 3 * mw2 + mw2 * mw2 / 512;
      std::snprintf(d, sizeof d, "20 elements, unit stride: gas %lld, static count %lld (50 + 10 x 3 groups) | one strided row: gas %lld, static count %lld (50 + 10 x 3 groups + 4 x 20 strided elements): %s", r.gas, pred, r2.gas, pred2, (r.gas == pred && r2.gas == pred2) ? "equal" : "DIFFERENT"); report("G9", r.st == EVMC_SUCCESS && r2.st == EVMC_SUCCESS && r.gas == pred && r2.gas == pred2, d); }
    std::printf("\n--- SUMMARY --- %d/%d passed%s\n", pass, total, pass == total ? "  *** ALL CHECKS PASSED ***" : "  *** INSPECT ***");
    VM->destroy(VM); dlclose(h);
    return pass == total ? 0 : 1;
}
