// sec_fuzz_views.cpp — DIFFERENTIAL FUZZ TEST of the view instructions against an independent reference model.
//
// Instructions under test: XMACC_V (0xea), XMATMUL (0xe9) and, in front of them, XTRANSPOSE (0xdd).
// The reference model below is written from the specification of the instructions and shares no code with
// the interpreter. It keeps its own memory image and its own gas counter, with the memory rules of the EVM
// (offsets above 2^32 - 1 are rejected, expansion costs 3 w + floor(w^2 / 512) for w words) and with the gas
// rule of the view instructions: a charge per call, 8 per group of eight elements, 3 per strided element
// (XMACC_V on every call, XMATMUL once per packed element, where a view that is both operands is packed
// once) and 10 per output entry of XMATMUL.
// Every trial builds a memory image, executes one instruction in the interpreter and in the model, and compares
//   (1) the status (SUCCESS or exceptional halt),
//   (2) on SUCCESS the gas used by the whole program,
//   (3) on SUCCESS every byte of the final memory, so a stray write anywhere in the memory is detected.
// Two thirds of the trials start from a well-formed pair of views (plain or transposed, rows of 0 to 40
// elements) and many of them receive one mutation. The others draw every header field and every operand from a
// distribution that favours boundary values: 0, 1, 2^31 - 1, 2^31, 2^32 - 1, 2^32, 2^62, 2^63, 2^64 - 1 and 2^200.
// Header offsets need not be word aligned, the two headers may overlap each other and their own elements, and
// the output range of XMATMUL may lie anywhere. The gas limit of a trial is 3,000,000, so a view that asks for
// more work or more memory than that must halt.
// The output is deterministic and is compared between the builds. The program is also run on the builds
// instrumented with AddressSanitizer and UndefinedBehaviorSanitizer.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
static evmc::MockedHost g_host;
static evmc_vm* VM;
static const long long GAS = 3000000;
typedef unsigned __int128 u128;
struct Val { bool huge; uint64_t v; };                 // huge stands for 2^200
static uint32_t rs = 0x9E3779B9u;
static uint32_t rnd() { rs = rs * 1664525u + 1013904223u; return rs >> 4; }
static const uint64_t BND[] = {0x7fffffffULL, 0x80000000ULL, 0xffffffffULL, 0x100000000ULL, 1ULL << 62, 1ULL << 63, ~0ULL, 4096, 65536, 1000, 2, 3};
static uint64_t wild() { uint32_t r = rnd() % 100; if (r < 55) return rnd() % 41; if (r < 65) return rnd() % 2; if (r < 78) return 41 + rnd() % 600; return BND[rnd() % (sizeof(BND) / sizeof(BND[0]))]; }

// ---------------- reference model ----------------
struct Model { std::vector<uint8_t> mem; long long gas; };
static long long memcost(u128 words) { return (long long)(3 * words + words * words / 512); }
static bool grow(Model& m, u128 new_size) {
    u128 w = (new_size + 31) / 32, c = m.mem.size() / 32;
    if (w <= c) return true;
    if (w > ((u128)1 << 40)) return false;
    long long cost = memcost(w) - memcost(c); m.gas -= cost; if (m.gas < 0) return false;
    m.mem.resize((size_t)(w * 32), 0); return true;
}
static bool chk(Model& m, Val off, u128 size) {
    if (off.huge || off.v > 0xffffffffULL) return false;
    u128 ns = (u128)off.v + size; if (ns > m.mem.size()) return grow(m, ns); return true;
}
static uint64_t rd64(const Model& m, uint64_t at) { uint64_t v = 0; for (int k = 0; k < 8; ++k) v |= (uint64_t)m.mem[at + k] << (8 * k); return v; }
static void wr64(Model& m, uint64_t at, uint64_t v) { for (int k = 0; k < 8; ++k) m.mem[at + k] = (uint8_t)(v >> (8 * k)); }
static int32_t elem(const Model& m, uint64_t off, u128 idx) { uint64_t a = off + 32 + 2 * (uint64_t)idx; return (int16_t)(uint16_t)(m.mem[a] | (m.mem[a + 1] << 8)); }
static int32_t sat(int64_t s) { return s > INT32_MAX ? INT32_MAX : (s < INT32_MIN ? INT32_MIN : (int32_t)s); }
static const u128 CAP = (u128)1 << 40;                 // any index at or above 2^40 is outside every legal memory
static u128 capmul(u128 a, u128 b) { if (a == 0 || b == 0) return 0; if (a >= CAP || b >= CAP || a * b >= CAP) return CAP; return a * b; }
static u128 capadd(u128 a, u128 b) { u128 s = a + b; return s >= CAP ? CAP : s; }
struct Hdr { uint64_t rows, cols, sr, sc; };
static Hdr hdr(const Model& m, uint64_t off) { return {rd64(m, off), rd64(m, off + 8), rd64(m, off + 16), rd64(m, off + 24)}; }
static long long satcount = 0;
// XTRANSPOSE: the offset stays on the stack. False on an exceptional halt.
static bool m_xtranspose(Model& m, Val off) {
    m.gas -= 10; if (m.gas < 0) return false;
    if (!chk(m, off, 32)) return false;
    Hdr h = hdr(m, off.v); wr64(m, off.v, h.cols); wr64(m, off.v + 8, h.rows); wr64(m, off.v + 16, h.sc); wr64(m, off.v + 24, h.sr); return true;
}
static bool m_xmacc_v(Model& m, Val offa, Val rowa, Val offb, Val rowb, int32_t* acc) {
    if (!chk(m, offa, 32) || !chk(m, offb, 32)) return false;
    Hdr A = hdr(m, offa.v), B = hdr(m, offb.v);
    if (A.cols != B.cols || A.cols > 0x7fffffffULL) return false;
    if (rowa.huge || rowb.huge || rowa.v >= A.rows || rowb.v >= B.rows) return false;
    uint64_t n = A.cols; long long strided = (A.sc != 1 ? 1 : 0) + (B.sc != 1 ? 1 : 0);
    m.gas -= 43 + (long long)((n + 7) / 8) * 8 + (long long)n * strided * 3; if (m.gas < 0) return false;
    if (n == 0) return true;
    u128 ba = capmul(rowa.v, A.sr), bb = capmul(rowb.v, B.sr);
    u128 la = capadd(ba, capmul(n - 1, A.sc)), lb = capadd(bb, capmul(n - 1, B.sc));
    if (la >= CAP || lb >= CAP) return false;
    if (!chk(m, Val{false, (uint64_t)((u128)offa.v + 32 + 2 * la)}, 2)) return false;
    if (!chk(m, Val{false, (uint64_t)((u128)offb.v + 32 + 2 * lb)}, 2)) return false;
    for (uint64_t t = 0; t < n; ++t) { int64_t p = (int64_t)elem(m, offa.v, ba + (u128)t * A.sc) * elem(m, offb.v, bb + (u128)t * B.sc); int64_t s = (int64_t)acc[t % 8] + p;
        if (s > INT32_MAX || s < INT32_MIN) ++satcount; acc[t % 8] = sat(s); }
    return true;
}
static long long sameStrided = 0;                     // successful XMATMUL trials with one strided view as both operands
static bool m_xmatmul(Model& m, Val offa, Val offb, Val dst) {
    if (!chk(m, offa, 32) || !chk(m, offb, 32)) return false;
    Hdr A = hdr(m, offa.v), B = hdr(m, offb.v); const uint64_t LIM = 0x7fffffffULL;
    if (A.cols != B.cols || A.cols > LIM || A.rows > LIM || B.rows > LIM) return false;
    uint64_t n = A.cols; u128 entries = (u128)A.rows * B.rows; const bool sA = A.sc != 1, sB = B.sc != 1;
    const bool sameView = offa.v == offb.v;                                  // one header: the view is packed once
    u128 packed = (sA && !sameView ? (u128)A.rows * n : 0) + (sB ? (u128)B.rows * n : 0);
    u128 charge = 43 + entries * (10 + (u128)((n + 7) / 8) * 8) + packed * 3;
    if (charge > (u128)m.gas) return false; m.gas -= (long long)charge;
    if (entries == 0) return true;
    u128 outb = entries * 32; if (outb > 0xffffffffULL) return false;
    if (!chk(m, dst, outb)) return false;
    u128 enda = (u128)offa.v + 32, endb = (u128)offb.v + 32;
    if (n != 0) {
        u128 la = capadd(capmul(A.rows - 1, A.sr), capmul(n - 1, A.sc)), lb = capadd(capmul(B.rows - 1, B.sr), capmul(n - 1, B.sc));
        if (la >= CAP || lb >= CAP) return false;
        enda += 2 * la + 2; endb += 2 * lb + 2;
        if (!chk(m, Val{false, (uint64_t)(enda - 2)}, 2) || !chk(m, Val{false, (uint64_t)(endb - 2)}, 2)) return false;
    }
    u128 d0 = dst.v, d1 = d0 + outb;
    if ((d0 < enda && offa.v < d1) || (d0 < endb && offb.v < d1)) return false;
    // the packed form of a strided B must not be larger than the memory
    if (n != 0 && sB && (u128)B.rows * n * 2 > m.mem.size()) return false;
    std::vector<uint32_t> res((size_t)entries);
    for (uint64_t i = 0; i < A.rows; ++i) for (uint64_t j = 0; j < B.rows; ++j) { int32_t l[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint64_t t = 0; t < n; ++t) { int64_t s = (int64_t)l[t % 8] + (int64_t)elem(m, offa.v, (u128)i * A.sr + (u128)t * A.sc) * elem(m, offb.v, (u128)j * B.sr + (u128)t * B.sc);
            if (s > INT32_MAX || s < INT32_MIN) ++satcount; l[t % 8] = sat(s); }
        int64_t tot = 0; for (int k = 0; k < 8; ++k) tot += l[k]; if (tot > INT32_MAX || tot < INT32_MIN) ++satcount; res[(size_t)(i * B.rows + j)] = (uint32_t)sat(tot); }
    for (size_t e = 0; e < res.size(); ++e) { uint8_t* w = &m.mem[(size_t)d0 + 32 * e]; std::memset(w, 0, 28); w[28] = res[e] >> 24; w[29] = res[e] >> 16; w[30] = res[e] >> 8; w[31] = res[e]; }
    if (sameView && sA && n != 0) ++sameStrided;
    return true;
}

// ---------------- program builder ----------------
struct Code { std::vector<uint8_t> c;
    void op(uint8_t o) { c.push_back(o); }
    void P(Val v) { if (v.huge) { c.push_back(0x7F); for (int i = 0; i < 32; ++i) c.push_back(i == 6 ? 1 : 0); return; }     // byte 6 of a big-endian word = 2^200
        int n = 0; for (uint64_t t = v.v; t; t >>= 8) ++n; if (n == 0) n = 1; c.push_back((uint8_t)(0x5F + n)); for (int k = n - 1; k >= 0; --k) c.push_back((uint8_t)((v.v >> (8 * k)) & 0xFF)); }
    void W32(const uint8_t* w) { c.push_back(0x7F); for (int i = 0; i < 32; ++i) c.push_back(w[i]); } };
static void put64(std::vector<uint8_t>& m, size_t at, uint64_t v) { for (int k = 0; k < 8; ++k) if (at + k < m.size()) m[at + k] = (uint8_t)(v >> (8 * k)); }
static uint64_t fnv = 1469598103934665603ULL;
static void dig(const uint8_t* p, size_t n) { for (size_t i = 0; i < n; ++i) { fnv ^= p[i]; fnv *= 1099511628211ULL; } }

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    std::printf("=== DIFFERENTIAL FUZZ TEST OF XMACC_V, XMATMUL AND XTRANSPOSE AGAINST A REFERENCE MODEL ===\n");
    const int T = 60000; long long nOK[2] = {0, 0}, nHALT[2] = {0, 0}, bad = 0, grown = 0, strided = 0, transposed = 0, wildn = 0, mutated = 0, bytes = 0; int shown = 0;
    // one trial: the program, the model and the comparison. Returns the model verdict (true = SUCCESS).
    auto trial = [&](int which, const std::vector<uint8_t>& img, Val offa, Val rowa, Val offb, Val rowb, Val dst, const int32_t* acc0, int ntr, const Val* troff, int t) -> bool {
        // ---- program ----
        Code k; for (size_t o = 0; o < img.size(); o += 32) { k.W32(&img[o]); k.P(Val{false, o}); k.op(0x52); }
        for (int q = 0; q < ntr; ++q) { k.P(troff[q]); k.op(0xDD); k.op(0x50); }
        uint8_t aw[32]; for (int j = 0; j < 8; ++j) { uint32_t v = (uint32_t)acc0[j]; for (int b = 0; b < 4; ++b) aw[31 - 4 * j - b] = (uint8_t)(v >> (8 * b)); }
        if (which == 0) { k.W32(aw); k.P(rowb); k.P(offb); k.P(rowa); k.P(offa); k.op(0xEA); k.op(0x59); k.op(0x52); }
        else { k.P(dst); k.P(offb); k.P(offa); k.op(0xE9); }
        k.op(0x59); k.P(Val{false, 0}); k.op(0xF3);
        evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS;
        auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, k.c.data(), k.c.size());
        // ---- model ----
        Model m; m.gas = GAS; bool ok = true; const size_t before = img.size() / 32;
        m.gas -= (long long)(img.size() / 32) * 9; ok = grow(m, img.size()); if (ok) std::memcpy(m.mem.data(), img.data(), img.size());
        for (int q = 0; q < ntr && ok; ++q) { m.gas -= 3; ok = m_xtranspose(m, troff[q]); m.gas -= 2; if (m.gas < 0) ok = false; }
        if (ok) { if (which == 0) { int32_t acc[8]; std::memcpy(acc, acc0, sizeof acc); m.gas -= 15; ok = m.gas >= 0 && m_xmacc_v(m, offa, rowa, offb, rowb, acc);
                if (ok) { m.gas -= 2 + 3; size_t at = m.mem.size(); ok = m.gas >= 0 && grow(m, at + 32);
                    if (ok) for (int j = 0; j < 8; ++j) { uint32_t v = (uint32_t)acc[j]; for (int b = 0; b < 4; ++b) m.mem[at + 31 - 4 * j - b] = (uint8_t)(v >> (8 * b)); } } }
            else { m.gas -= 9; ok = m.gas >= 0 && m_xmatmul(m, offa, offb, dst); } }
        if (ok) { m.gas -= 2 + 3; if (m.gas < 0) ok = false; }
        // ---- comparison ----
        const bool iok = r.status_code == EVMC_SUCCESS; bool same = (iok == ok) && (iok || r.status_code == EVMC_OUT_OF_GAS);
        if (same && ok) same = (GAS - r.gas_left) == (GAS - m.gas) && r.output_size == m.mem.size() && (r.output_size == 0 || std::memcmp(r.output_data, m.mem.data(), r.output_size) == 0);
        if (ok) { ++nOK[which]; if (m.mem.size() / 32 > before + (which == 0 ? 1 : 0)) ++grown; bytes += (long long)m.mem.size(); } else ++nHALT[which];
        uint8_t st = (uint8_t)r.status_code; dig(&st, 1); if (iok) { uint64_t gu = (uint64_t)(GAS - r.gas_left); dig((const uint8_t*)&gu, 8); dig(r.output_data, r.output_size); }
        if (!same) { ++bad; if (shown < 5) { ++shown; std::printf("  MISMATCH trial %d (%s): interpreter status %d gas %lld size %zu | model %s gas %lld size %zu\n", t, which ? "XMATMUL" : "XMACC_V", (int)r.status_code, (long long)(GAS - r.gas_left), (size_t)r.output_size, ok ? "SUCCESS" : "HALT", GAS - m.gas, m.mem.size()); } }
        if (r.release) r.release(&r);
        return ok;
    };
    for (int t = 0; t < T; ++t) {
        const int which = t & 1;                                    // 0: XMACC_V, 1: XMATMUL
        std::vector<uint8_t> img; Val offa, offb, rowa{false, 0}, rowb{false, 0}, dst{false, 0}; int32_t acc0[8];
        for (int j = 0; j < 8; ++j) { uint32_t r = rnd(); acc0[j] = (r % 5 == 0) ? (int32_t)(INT32_MAX - (rnd() % 100000)) : (r % 5 == 1) ? (int32_t)(INT32_MIN + (rnd() % 100000)) : (int32_t)(rnd() % 2000001) - 1000000; }
        const uint32_t kind = rnd() % 100; const bool extreme = (rnd() % 4 == 0);
        auto val16 = [&]() -> uint16_t { uint32_t r = rnd(); if (extreme && r % 3 == 0) return (r & 8) ? 0x7fff : 0x8000; return (uint16_t)(r >> 3); };
        if (kind < 67) {            // well-formed pair of views
            uint64_t n = rnd() % 41, Ra = 1 + rnd() % 6, Rb = 1 + rnd() % 6; bool ta = rnd() & 1, tb = rnd() & 1;
            auto mk = [&](uint64_t R, bool tr, Val& off) { size_t pad = (rnd() % 3 == 0) ? rnd() % 9 : 0; size_t o = img.size() + pad; img.resize(o + 32 + 2 * R * n, 0);
                if (tr) { put64(img, o, R); put64(img, o + 8, n); put64(img, o + 16, 1); put64(img, o + 24, R); } else { put64(img, o, R); put64(img, o + 8, n); put64(img, o + 16, n); put64(img, o + 24, 1); }
                for (size_t e = 0; e < R * n; ++e) { uint16_t u = val16(); img[o + 32 + 2 * e] = u & 0xFF; img[o + 33 + 2 * e] = u >> 8; } off = Val{false, o}; if (tr) ++transposed; };
            mk(Ra, ta, offa); mk(Rb, tb, offb); if (ta || tb) ++strided;
            rowa.v = rnd() % Ra; rowb.v = rnd() % Rb; img.resize((img.size() + 31) / 32 * 32, 0); dst.v = img.size() + 32 * (rnd() % 3);
            if (kind >= 30) { ++mutated;       // one mutation
                switch (rnd() % 12) {
                    case 0: { const uint64_t v = wild(); const uint64_t f = rnd() % 4; put64(img, offa.v + 8 * f, v); break; }
                    case 1: { const uint64_t v = wild(); const uint64_t f = rnd() % 4; put64(img, offb.v + 8 * f, v); break; }
                    case 2: rowa.v = wild(); break;
                    case 3: rowb = (rnd() & 1) ? Val{true, 0} : Val{false, wild()}; break;
                    case 4: dst.v = rnd() % (img.size() + 64); break;                       // output somewhere in the image
                    case 5: dst = (rnd() & 1) ? Val{true, 0} : Val{false, BND[rnd() % 8]}; break;
                    case 6: offa = (rnd() & 1) ? Val{true, 0} : Val{false, img.size() - (rnd() % 40)}; break;
                    case 7: offb.v = offa.v + rnd() % 40; break;                            // the views overlap
                    case 8: put64(img, offa.v + 16, 0); break;                              // all rows are the same row
                    case 9: put64(img, offb.v + 24, 0); break;                              // every element of a row is the same element
                    case 10: put64(img, offb.v + 8, rd64(Model{img, 0}, offa.v + 8) + 1 + rnd() % 3); break;   // shape error
                    default: offb = offa; break;                                            // the same view twice
                } }
        } else {                    // every field and operand from the boundary distribution
            ++wildn; img.resize(64 + rnd() % 2000); for (auto& b : img) b = (uint8_t)rnd(); img.resize((img.size() + 31) / 32 * 32, 0);
            auto anyoff = [&]() -> Val { uint32_t r = rnd() % 100; if (r < 70) return Val{false, rnd() % (img.size() - 31)}; if (r < 85) return Val{false, img.size() - 40 + rnd() % 3000}; if (r < 93) return Val{false, BND[rnd() % 8]}; return Val{true, 0}; };
            offa = anyoff(); offb = anyoff();
            if (!offa.huge && offa.v + 32 <= img.size()) for (int f = 0; f < 4; ++f) put64(img, offa.v + 8 * f, wild());
            if (!offb.huge && offb.v + 32 <= img.size()) for (int f = 0; f < 4; ++f) put64(img, offb.v + 8 * f, wild());
            if (rnd() % 2 && !offa.huge && !offb.huge && offa.v + 32 <= img.size() && offb.v + 32 <= img.size()) put64(img, offb.v + 8, rd64(Model{img, 0}, offa.v + 8));   // equal column counts in half of them
            rowa = (rnd() % 20 == 0) ? Val{true, 0} : Val{false, wild()}; rowb = Val{false, wild()};
            uint32_t r = rnd() % 100; dst = r < 60 ? Val{false, img.size() + 32 * (rnd() % 4)} : r < 85 ? Val{false, rnd() % (img.size() + 2000)} : r < 95 ? Val{false, BND[rnd() % 8]} : Val{true, 0};
        }
        const int ntr = (rnd() % 4 == 0) ? 1 + rnd() % 2 : 0; Val troff[2] = {(rnd() & 1) ? offa : offb, (rnd() & 1) ? offa : offb};
        trial(which, img, offa, rowa, offb, rowb, dst, acc0, ntr, troff, t);
    }
    const long long rOK[2] = {nOK[0], nOK[1]}, rHALT[2] = {nHALT[0], nHALT[1]}, rGrown = grown, rBytes = bytes, rSat = satcount, rSame = sameStrided;
    // ---- boundary sweeps: every byte position, so that each bound is hit exactly ----
    long long swOK = 0, swHALT = 0, swN = 0; const long long badBefore = bad;
    { // XMATMUL: the output range slides byte by byte across a plain view, a transposed view and the gaps between them
      for (int ncols : {5, 0}) { std::vector<uint8_t> img(96, 0); Val oa{false, 96}, ob; const uint64_t n = (uint64_t)ncols;
        img.resize(96 + 32 + 2 * 3 * n, 0); put64(img, 96, 3); put64(img, 104, n); put64(img, 112, n); put64(img, 120, 1);
        img.resize((img.size() + 31) / 32 * 32 + 224, 0); ob = Val{false, img.size()}; img.resize(ob.v + 32 + 2 * 2 * n, 0); put64(img, ob.v, 2); put64(img, ob.v + 8, n); put64(img, ob.v + 16, 1); put64(img, ob.v + 24, 2);
        for (size_t e = 0; e < 3 * n; ++e) { img[96 + 32 + 2 * e] = (uint8_t)rnd(); img[96 + 33 + 2 * e] = (uint8_t)rnd(); } for (size_t e = 0; e < 2 * n; ++e) { img[ob.v + 32 + 2 * e] = (uint8_t)rnd(); img[ob.v + 33 + 2 * e] = (uint8_t)rnd(); }
        img.resize((img.size() + 31) / 32 * 32, 0); int32_t z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint64_t d = 0; d <= img.size() + 40; ++d) { ++swN; if (trial(1, img, oa, Val{false, 0}, ob, Val{false, 0}, Val{false, d}, z, 0, nullptr, -1)) ++swOK; else ++swHALT; } } }
    const long long sw1N = swN, sw1OK = swOK, sw1HALT = swHALT; swN = swOK = swHALT = 0;
    { // XMATMUL: a strided view whose rows share their elements (row stride 0). Its packed form grows with the row count and
      // crosses the memory size, so the sweep hits the packing rule from both sides. Also with the roles of A and B exchanged.
      for (int role = 0; role < 2; ++role) for (uint64_t n : {17ULL, 40ULL}) { std::vector<uint8_t> img; Val ou{false, 0};      // unit view: 1 x n
        img.resize(32 + 2 * n, 0); put64(img, 0, 1); put64(img, 8, n); put64(img, 16, n); put64(img, 24, 1); for (size_t e = 32; e < img.size(); ++e) img[e] = (uint8_t)rnd();
        img.resize((img.size() + 31) / 32 * 32, 0); Val os{false, img.size()}; img.resize(os.v + 32 + 2 * (2 * (n - 1) + 1), 0);                // strided view: column stride 2, row stride 0
        put64(img, os.v + 8, n); put64(img, os.v + 16, 0); put64(img, os.v + 24, 2); for (size_t e = os.v + 32; e < img.size(); ++e) img[e] = (uint8_t)rnd();
        img.resize((img.size() + 31) / 32 * 32, 0); int32_t z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint64_t rows = 0; rows <= img.size() + 8; ++rows) { std::vector<uint8_t> im2 = img; put64(im2, os.v, rows); ++swN;
            bool ok = role == 0 ? trial(1, im2, ou, Val{false, 0}, os, Val{false, 0}, Val{false, im2.size()}, z, 0, nullptr, -1) : trial(1, im2, os, Val{false, 0}, ou, Val{false, 0}, Val{false, im2.size()}, z, 0, nullptr, -1);
            if (ok) ++swOK; else ++swHALT; } } }
    { // XMATMUL: two views in which every element of a row is the same element (column stride 0). The packed row has 2 n bytes
      // and the memory is small, so the column count crosses the rule.
      std::vector<uint8_t> base(128, 0); for (int q = 0; q < 2; ++q) { put64(base, 64 * q, 1); put64(base, 64 * q + 16, 0); put64(base, 64 * q + 24, 0); base[64 * q + 32] = (uint8_t)(3 + q); base[64 * q + 33] = 0; }
      int32_t z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (uint64_t n = 0; n <= 140; ++n) { std::vector<uint8_t> im2 = base; put64(im2, 8, n); put64(im2, 72, n); ++swN;
          if (trial(1, im2, Val{false, 0}, Val{false, 0}, Val{false, 64}, Val{false, 0}, Val{false, 128}, z, 0, nullptr, -1)) ++swOK; else ++swHALT; } }
    const long long sw2N = swN, sw2OK = swOK, sw2HALT = swHALT; swN = swOK = swHALT = 0;
    { // XMACC_V: every row index around the row count, every header offset around the end of the memory image, every column count around a group boundary
      for (uint64_t n = 0; n <= 18; ++n) { std::vector<uint8_t> img; Val oa{false, 0}; img.resize(32 + 2 * 4 * n, 0); put64(img, 0, 4); put64(img, 8, n); put64(img, 16, n); put64(img, 24, 1);
        img.resize((img.size() + 31) / 32 * 32, 0); Val ob{false, img.size()}; img.resize(ob.v + 32 + 2 * 3 * n, 0); put64(img, ob.v, n); put64(img, ob.v + 8, 3); put64(img, ob.v + 16, 3); put64(img, ob.v + 24, 1);
        for (size_t e = 32; e < ob.v; ++e) img[e] = (uint8_t)rnd(); for (size_t e = ob.v + 32; e < img.size(); ++e) img[e] = (uint8_t)rnd(); img.resize((img.size() + 31) / 32 * 32, 0);
        int32_t acc[8]; for (int j = 0; j < 8; ++j) acc[j] = (int32_t)(rnd() % 2001) - 1000; Val tr[1] = {ob};
        for (uint64_t ra = 0; ra <= 6; ++ra) for (uint64_t rb = 0; rb <= 5; ++rb) { ++swN; if (trial(0, img, oa, Val{false, ra}, ob, Val{false, rb}, Val{false, 0}, acc, 1, tr, -1)) ++swOK; else ++swHALT; }   // B is 3 x n after XTRANSPOSE
        for (uint64_t o = img.size() - 70; o <= img.size() + 70; ++o) { ++swN; if (trial(0, img, Val{false, o}, Val{false, 0}, Val{false, o}, Val{false, 0}, Val{false, 0}, acc, 0, nullptr, -1)) ++swOK; else ++swHALT; } } }
    std::printf("Trials: %d (%d per instruction). Well-formed start: %lld, of which mutated: %lld. Boundary-value trials: %lld. With XTRANSPOSE in front: about one in four.\n", T, T / 2, (long long)T - wildn, mutated, wildn);
    std::printf("XMACC_V : %lld SUCCESS, %lld exceptional halts\n", rOK[0], rHALT[0]);
    std::printf("XMATMUL : %lld SUCCESS, %lld exceptional halts\n", rOK[1], rHALT[1]);
    std::printf("Successful trials in which the instruction expanded the memory: %lld | trials with a strided view: %lld | saturating steps in the model: %lld\n", rGrown, strided, rSat);
    std::printf("Successful XMATMUL trials with one strided view as both operands (packed once): %lld\n", rSame);
    std::printf("Bytes of final memory compared in the random trials: %lld | digest of status, gas and memory of all trials and sweeps: %016llx\n", rBytes, (unsigned long long)fnv);
    std::printf("Boundary sweep of the XMATMUL output range (every byte position): %lld trials, %lld SUCCESS, %lld halts\n", sw1N, sw1OK, sw1HALT);
    std::printf("Sweep of the packing rule of XMATMUL (strided views whose rows share elements): %lld trials, %lld SUCCESS, %lld halts\n", sw2N, sw2OK, sw2HALT);
    std::printf("Boundary sweep of XMACC_V (row indices, header offsets, column counts 0 to 18): %lld trials, %lld SUCCESS, %lld halts\n", swN, swOK, swHALT);
    (void)badBefore;
    std::printf("Trials in which interpreter and model disagree: %lld\n", bad);
    std::printf("\n%s\n", bad == 0 ? "*** ALL CHECKS PASSED: interpreter and reference model agree in every trial ***" : "*** INSPECT: disagreement between interpreter and model ***");
    VM->destroy(VM); dlclose(h);
    return bad == 0 ? 0 : 1;
}
