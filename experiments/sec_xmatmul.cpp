// sec_xmatmul.cpp — TESTS for the matrix-level instruction XMATMUL (0xe9).
//
// XMATMUL pops off_a, off_b, dst. A and B are matrix views with the same column count. For every row i of A
// and every row j of B it writes the word XSUM32(XMACC_V(0, row i of A, row j of B)) to dst + 32 (i rows_B + j).
//   R1  2000 random trials: 1..6 rows, 1..40 columns, plain and transposed views, extreme elements, against the
//       C++ reference (saturating lane additions, then the saturating sum of the eight lanes)
//   R2  the same memory image is processed by XMATMUL and by a loop of XMACC_V, XSUM32 and MSTORE: equal outputs
//   R3  XTRANSPOSE on both operands, then XMATMUL(A^T, A^T): the Gram matrix of the stored matrix, no copy
//   G1  cols of A != cols of B                 -> clean OUT_OF_GAS (shape error)
//   G2  rows of A = 2^31                       -> clean OUT_OF_GAS (rows and columns limited to 31 bits)
//   G3  the output range overlaps the elements of A -> clean OUT_OF_GAS
//   G4  the output range overlaps the header of B   -> clean OUT_OF_GAS
//   G5  dst = 2^200                            -> clean OUT_OF_GAS
//   G6  2^30 x 2^30 entries with 1e9 gas       -> clean OUT_OF_GAS from the charge, before any work
//   G7  stride_row = 2^62 with two rows        -> clean OUT_OF_GAS (the last row is outside the memory)
//   G8  rows of B = 0                          -> SUCCESS, nothing written, only the charge per call
//   G9  gas accounting                         -> measured gas equals the static count for unit-stride views and for a strided view
//   G10 packing rule                           -> a strided B whose rows share their elements is accepted while its packed form
//                                                 fits in the memory and halts when it is one row larger
//   G11 one view as both operands             -> the view is packed once: its elements are charged once, twice for two
//                                                 copies of the view, and the output is the same
//   R4  rows without elements (cols = 0)       -> every output word is zero
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

static const long long P_CALL = 43, P_ENTRY = 10, P_UNIT = 8, P_PACK = 3;   // must equal the values in the interpreter
static uint32_t refEntry(const Mat& A, uint64_t i, const Mat& B, uint64_t j, int* sat) {
    int32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint64_t e = 0; e < A.cols; ++e) { int64_t s = (int64_t)acc[e % 8] + (int32_t)el(A, i, e) * (int32_t)el(B, j, e); if ((s > INT32_MAX || s < INT32_MIN) && sat) ++*sat; acc[e % 8] = sat32(acc[e % 8], (int32_t)el(A, i, e) * (int32_t)el(B, j, e)); }
    int64_t t = 0; for (int k = 0; k < 8; ++k) t += acc[k]; if (t > INT32_MAX) { t = INT32_MAX; if (sat) ++*sat; } if (t < INT32_MIN) { t = INT32_MIN; if (sat) ++*sat; }
    return (uint32_t)(int32_t)t;
}
static uint32_t word32(const uint8_t* w, bool* hi_zero) { uint32_t v = 0; for (int i = 28; i < 32; ++i) v = (v << 8) | w[i]; if (hi_zero) for (int i = 0; i < 28; ++i) if (w[i]) *hi_zero = false; return v; }
int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== XMATMUL TESTS ===\n");
    int pass = 0, total = 0; char d[260];
    auto report = [&](const char* name, bool ok, const char* detail) { ++total; pass += ok; std::printf("[%s] %-3s %s\n", ok ? "PASS" : "FAIL", name, detail); };
    auto bigword = [](int bit) { static uint8_t w[32]; std::memset(w, 0, 32); w[31 - bit / 8] = (uint8_t)(1u << (bit % 8)); return (const uint8_t*)w; };
    // one execution of XMATMUL: returns the output region [dst, dst + outbytes)
    auto exec = [&](const std::vector<uint8_t>& mem, uint64_t offa, uint64_t offb, uint64_t dst, const uint8_t* dst_w, uint64_t outbytes, long long vgas, int64_t gas, Code* keep) {
        Code k; load(k, mem); if (dst_w) k.W32(dst_w); else k.PUSH(dst); k.PUSH(offb); k.PUSH(offa); k.op(0xE9, vgas); k.RET(dst, outbytes); if (keep) *keep = k; return run(k, gas); };
    // R1
    { int ok = 0, sat = 0; const int T = 2000; unsigned long long digest = 1469598103934665603ULL;
      for (int t = 0; t < T; ++t) {
          uint64_t n = 1 + rnd32() % 40, Ra = 1 + rnd32() % 6, Rb = 1 + rnd32() % 6; bool ta = rnd32() & 1, tb = rnd32() & 1;
          Mat A = ta ? mk(n, Ra, true) : mk(Ra, n, false); Mat B = tb ? mk(n, Rb, true) : mk(Rb, n, false);
          std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B); uint64_t dst = mem.size(), ob_bytes = 32 * Ra * Rb;
          Res r = exec(mem, oa, ob, dst, nullptr, ob_bytes, 0, GAS_BIG, nullptr);
          bool good = r.st == EVMC_SUCCESS && r.out.size() == ob_bytes; bool hz = true;
          if (good) for (uint64_t i = 0; i < Ra && good; ++i) for (uint64_t j = 0; j < Rb; ++j) if (word32(&r.out[32 * (i * Rb + j)], &hz) != refEntry(A, i, B, j, &sat)) { good = false; break; }
          good = good && hz; ok += good; for (uint8_t x : r.out) { digest ^= x; digest *= 1099511628211ULL; } }
      std::snprintf(d, sizeof d, "%d/%d random trials equal the C++ reference (%d saturating steps, output digest %016llx)", ok, T, sat, digest); report("R1", ok == T, d); }
    // R2: XMATMUL against the loop of XMACC_V + XSUM32 + MSTORE on the same memory image
    { int ok = 0; const int T = 60;
      for (int t = 0; t < T; ++t) { uint64_t n = 1 + rnd32() % 70, Ra = 1 + rnd32() % 5, Rb = 1 + rnd32() % 5; bool ta = rnd32() & 1, tb = rnd32() & 1;
          Mat A = ta ? mk(n, Ra, true) : mk(Ra, n, false); Mat B = tb ? mk(n, Rb, true) : mk(Rb, n, false);
          std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B); uint64_t dst = mem.size(), nb = 32 * Ra * Rb;
          Res r1 = exec(mem, oa, ob, dst, nullptr, nb, 0, GAS_BIG, nullptr);
          Code k; load(k, mem); for (uint64_t i = 0; i < Ra; ++i) for (uint64_t j = 0; j < Rb; ++j) { k.PUSH(0); k.PUSH(j); k.PUSH(ob); k.PUSH(i); k.PUSH(oa); k.op(0xEA, 0); k.op(0xCD, 6); k.PUSH(dst + 32 * (i * Rb + j)); k.MSTORE(); }
          k.RET(dst, nb); Res r2 = run(k, GAS_BIG);
          ok += (r1.st == EVMC_SUCCESS && r2.st == EVMC_SUCCESS && r1.out == r2.out && r1.out.size() == nb); }
      std::snprintf(d, sizeof d, "%d/%d shapes: XMATMUL equals the loop of XMACC_V, XSUM32 and MSTORE byte for byte", ok, T); report("R2", ok == T, d); }
    // R3: Gram matrix of a stored 9 x 7 matrix through two transposed views of the same data
    { Mat S; S.rows = 9; S.cols = 7; S.sr = 7; S.sc = 1; S.d.resize(63); for (auto& v : S.d) v = (int16_t)((int)(rnd32() % 4096) - 2048);
      std::vector<uint8_t> mem; size_t os = place(mem, S); uint64_t dst = mem.size();
      Code k; load(k, mem); k.PUSH(dst); k.PUSH(os); k.PUSH(os); k.op(0xDD, 10); k.op(0xE9, 0); k.RET(dst, 32 * 49); Res r = run(k, GAS_BIG);
      bool ok = r.st == EVMC_SUCCESS && r.out.size() == 32 * 49;
      if (ok) for (int i = 0; i < 7; ++i) for (int j = 0; j < 7; ++j) { int64_t g = 0; for (int t = 0; t < 9; ++t) g += (int64_t)S.d[t * 7 + i] * S.d[t * 7 + j]; if (word32(&r.out[32 * (i * 7 + j)], nullptr) != (uint32_t)g) ok = false; }
      report("R3", ok, ok ? "XTRANSPOSE + XMATMUL(A^T, A^T) equals the Gram matrix of the stored 9 x 7 matrix (49 entries): yes" : "Gram matrix through transposed views: NO"); }
    Mat A = mk(3, 20, false), B = mk(4, 20, false);
    { Mat B2 = mk(4, 19, false); std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B2); Res r = exec(mem, oa, ob, mem.size(), nullptr, 32, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "cols 20 against cols 19: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G1", r.st == EVMC_OUT_OF_GAS, d); }
    { Mat A2 = A; A2.rows = 0x80000000ULL; std::vector<uint8_t> mem; size_t oa = place(mem, A2), ob = place(mem, B); Res r = exec(mem, oa, ob, mem.size(), nullptr, 32, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "rows of A = 2^31: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G2", r.st == EVMC_OUT_OF_GAS, d); }
    { std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B); Res r = exec(mem, oa, ob, oa + 32 + 2 * 59, nullptr, 32, 0, GAS_BIG, nullptr);   // starts at the last element of A
      std::snprintf(d, sizeof d, "dst at the last element of A: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G3", r.st == EVMC_OUT_OF_GAS, d); }
    { std::vector<uint8_t> mem; size_t ob = place(mem, B); mem.resize(mem.size() + 32 * 12, 0); size_t oa = place(mem, A); (void)oa;
      // output of 12 words placed so that its last word covers the header of A
      std::vector<uint8_t> m2; size_t ob2 = place(m2, B); size_t gap = m2.size(); m2.resize(gap + 32 * 11, 0); size_t oa2 = place(m2, A); (void)ob;
      Res r = exec(m2, oa2, ob2, gap, nullptr, 32, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "last output word on the header of A: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G4", r.st == EVMC_OUT_OF_GAS, d); }
    { std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B); Res r = exec(mem, oa, ob, 0, bigword(200), 0, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "dst = 2^200: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G5", r.st == EVMC_OUT_OF_GAS, d); }
    { Mat A2 = A, B2 = B; A2.rows = B2.rows = 1ULL << 30; std::vector<uint8_t> mem; size_t oa = place(mem, A2), ob = place(mem, B2); Res r = exec(mem, oa, ob, mem.size(), nullptr, 32, 0, 1000000000LL, nullptr);
      std::snprintf(d, sizeof d, "2^30 x 2^30 entries with 1e9 gas: status=%d (OUT_OF_GAS=3 expected from the charge)", (int)r.st); report("G6", r.st == EVMC_OUT_OF_GAS, d); }
    { Mat A2 = mk(2, 20, false); A2.sr = 1ULL << 62; std::vector<uint8_t> mem; size_t oa = place(mem, A2), ob = place(mem, B); Res r = exec(mem, oa, ob, mem.size(), nullptr, 32, 0, GAS_BIG, nullptr);
      std::snprintf(d, sizeof d, "stride_row = 2^62 with two rows: status=%d (OUT_OF_GAS=3 expected)", (int)r.st); report("G7", r.st == EVMC_OUT_OF_GAS, d); }
    { Mat B2 = B; B2.rows = 0; std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B2); uint64_t dst = mem.size(); Code k; Res r = exec(mem, oa, ob, dst, nullptr, 32, P_CALL, GAS_BIG, &k);
      long long mw = (long long)(dst + 32) / 32, pred = k.sgas + 3 * mw + mw * mw / 512; bool z = r.out.size() == 32; if (z) for (uint8_t x : r.out) if (x) z = false;
      std::snprintf(d, sizeof d, "rows of B = 0: status=%d, output word untouched: %s, gas %lld == static count with %lld for the call %lld: %s", (int)r.st, z ? "yes" : "no", r.gas, P_CALL, pred, r.gas == pred ? "yes" : "no"); report("G8", r.st == EVMC_SUCCESS && z && r.gas == pred, d); }
    { std::vector<uint8_t> mem; size_t oa = place(mem, A), ob = place(mem, B); uint64_t dst = mem.size(); Code k; long long c1 = P_CALL + 12 * (P_ENTRY + 3 * P_UNIT); Res r = exec(mem, oa, ob, dst, nullptr, 32 * 12, c1, GAS_BIG, &k);
      long long mw = (long long)(dst + 32 * 12) / 32, pred = k.sgas + 3 * mw + mw * mw / 512;
      Mat T = mk(20, 3, true); std::vector<uint8_t> m2; size_t ot = place(m2, T), ob2 = place(m2, B); uint64_t d2 = m2.size(); Code k2; long long c2 = P_CALL + 12 * (P_ENTRY + 3 * P_UNIT) + 3 * 20 * P_PACK; Res r2 = exec(m2, ot, ob2, d2, nullptr, 32 * 12, c2, GAS_BIG, &k2);
      long long mw2 = (long long)(d2 + 32 * 12) / 32, pred2 = k2.sgas + 3 * mw2 + mw2 * mw2 / 512;
      std::snprintf(d, sizeof d, "3 x 4 entries of 20 elements, unit stride: gas %lld, static count %lld | one strided operand: gas %lld, static count %lld: %s", r.gas, pred, r2.gas, pred2, (r.gas == pred && r2.gas == pred2) ? "equal" : "DIFFERENT");
      report("G9", r.st == EVMC_SUCCESS && r2.st == EVMC_SUCCESS && r.gas == pred && r2.gas == pred2, d); }
    { // G10: B has column stride 2 and row stride 0, so all of its rows are the same 400 elements. Its packed form has rows * 800 bytes.
      // The output grows the memory by 32 bytes per row only, so the packed form passes the memory size after a few rows.
      Mat A2 = mk(1, 400, false); std::vector<uint8_t> mb; size_t oa2 = place(mb, A2); Mat B2; B2.rows = 1; B2.cols = 400; B2.sr = 0; B2.sc = 2; B2.d.resize(799); for (auto& v : B2.d) v = rnd16(); size_t ob2 = place(mb, B2); uint64_t d2 = mb.size();
      uint64_t lastOk = 0, firstBad = 0; bool cons = true;
      for (uint64_t r = 1; r <= 8; ++r) { std::vector<uint8_t> m2 = mb; for (int k = 0; k < 8; ++k) m2[ob2 + k] = (uint8_t)(r >> (8 * k));
          Code k; load(k, m2); k.PUSH(d2); k.PUSH(ob2); k.PUSH(oa2); k.op(0xE9, 0); k.RET(d2, 32 * r); Res q = run(k, GAS_BIG);
          const uint64_t memsize = d2 + 32 * r, packed = r * 400 * 2; const bool expect_ok = packed <= memsize;
          if ((q.st == EVMC_SUCCESS) != expect_ok) cons = false;
          if (q.st == EVMC_SUCCESS) { lastOk = r; for (uint64_t j = 0; j < r; ++j) if (word32(&q.out[32 * j], nullptr) != refEntry(A2, 0, B2, 0, nullptr)) cons = false; } else if (!firstBad) firstBad = r; }
      std::snprintf(d, sizeof d, "B with rows that share 400 elements: accepted up to %llu rows, halted from %llu rows (packed form above the memory size): %s", (unsigned long long)lastOk, (unsigned long long)firstBad, cons ? "as the rule says" : "NOT as the rule says");
      report("G10", cons && firstBad != 0 && lastOk + 1 == firstBad, d); }
    { // G11: the transposed view of a stored 20 x 3 matrix (3 rows of 20 strided elements) as both operands, and two copies of it
      Mat T = mk(20, 3, true); std::vector<uint8_t> m1; size_t o1 = place(m1, T); uint64_t d1 = m1.size();
      Code k1; long long c1 = P_CALL + 9 * (P_ENTRY + 3 * P_UNIT) + 3 * 20 * P_PACK; Res r1 = exec(m1, o1, o1, d1, nullptr, 32 * 9, c1, GAS_BIG, &k1);
      std::vector<uint8_t> m2; size_t oa2 = place(m2, T), ob2 = place(m2, T); uint64_t d2 = m2.size();
      Code k2; long long c2 = P_CALL + 9 * (P_ENTRY + 3 * P_UNIT) + 2 * 3 * 20 * P_PACK; Res r2 = exec(m2, oa2, ob2, d2, nullptr, 32 * 9, c2, GAS_BIG, &k2);
      long long w1 = (long long)(d1 + 32 * 9) / 32, p1 = k1.sgas + 3 * w1 + w1 * w1 / 512, w2 = (long long)(d2 + 32 * 9) / 32, p2 = k2.sgas + 3 * w2 + w2 * w2 / 512;
      bool ok = r1.st == EVMC_SUCCESS && r2.st == EVMC_SUCCESS && r1.out.size() == 32 * 9 && r2.out.size() == 32 * 9 && r1.gas == p1 && r2.gas == p2;
      if (ok) for (uint64_t i = 0; i < 3; ++i) for (uint64_t j = 0; j < 3; ++j) { const uint32_t e = refEntry(T, i, T, j, nullptr); if (word32(&r1.out[32 * (i * 3 + j)], nullptr) != e || word32(&r2.out[32 * (i * 3 + j)], nullptr) != e) ok = false; }
      std::snprintf(d, sizeof d, "one strided view as both operands: gas %lld, static count %lld (60 packed elements) | two copies of the view: gas %lld, static count %lld (120 packed elements) | both outputs equal the reference: %s", r1.gas, p1, r2.gas, p2, ok ? "yes" : "NO");
      report("G11", ok, d); }
    { // R4: 3 x 0 against 4 x 0
      Mat A0 = mk(3, 5, false), B0 = mk(4, 5, false); A0.cols = B0.cols = 0; std::vector<uint8_t> mem; size_t oa = place(mem, A0), ob = place(mem, B0); uint64_t dst = mem.size();
      Code k; load(k, mem); for (int w = 0; w < 12; ++w) { k.PUSH(0xFFFFFFFFFFFFFFFFULL); k.PUSH(dst + 32 * w); k.MSTORE(); }      // the output range holds non-zero words first
      k.PUSH(dst); k.PUSH(ob); k.PUSH(oa); k.op(0xE9, 0); k.RET(dst, 32 * 12); Res q = run(k, GAS_BIG); bool z = q.st == EVMC_SUCCESS && q.out.size() == 32 * 12; if (z) for (uint8_t x : q.out) if (x) z = false;
      report("R4", z, z ? "3 x 4 entries of rows without elements: twelve zero words written over non-zero words: yes" : "rows without elements: NO"); }
    std::printf("\n--- SUMMARY --- %d/%d passed%s\n", pass, total, pass == total ? "  *** ALL CHECKS PASSED ***" : "  *** INSPECT ***");
    VM->destroy(VM); dlclose(h);
    return pass == total ? 0 : 1;
}
