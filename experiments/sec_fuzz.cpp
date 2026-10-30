// sec_fuzz.cpp -- randomized differential test of the memory-addressed instructions of the extension.
//
// Every trial builds a straight-line program: a few MSTOREs that place random words (including the
// MatrixHeader words) in memory, one instruction under test with random operands, then MSIZE and a RETURN
// of the whole memory. The operands and header fields are drawn from an edge-case set (0, 1, 7, 31, 32,
// 2^16, 2^31-1, 2^31, 2^32-1, 2^32, 2^63, 2^64-1, 2^255, 2^256-1, ...) and from uniform random values.
// A reference model of the specification (the memory bound rules of Section 4.7 and the arithmetic of
// Section 4.4) predicts, for every trial, whether the program succeeds or stops with a clean out-of-gas,
// the exact gas it consumes, the final memory size and the full content of the memory. The interpreter's
// result is compared with the prediction byte for byte.
//
// Instructions under test: XTRANSPOSE, XBCAST, XGATHER, XMACC_S, XMACC_N, XMACC_N16. The seed is fixed
// (deterministic), 6 x 4,000 trials. The same program runs on the AVX2 build and on the scalar-fallback
// build in run_all.sh, so the outputs of the two builds are also compared file by file.
//
// Exit code 0 and the line "ALL CHECKS PASSED" only when every trial matches the reference model.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------- 256-bit unsigned helper (only what the test needs) ----------------
struct U256 { uint64_t w[4]; };   // little-endian limbs
static U256 u256(uint64_t v) { return {{v, 0, 0, 0}}; }
static bool high_zero(const U256& a) { return (a.w[1] | a.w[2] | a.w[3]) == 0; }
static std::vector<uint8_t> be32(const U256& a) {  // big-endian 32-byte encoding (PUSH32 immediate)
    std::vector<uint8_t> b(32);
    for (int limb = 0; limb < 4; ++limb) for (int i = 0; i < 8; ++i) b[31 - (limb * 8 + i)] = (uint8_t)(a.w[limb] >> (8 * i));
    return b;
}

// ---------------- deterministic PRNG ----------------
static uint64_t rng_state = 0x5EED5EED5EED5EEDULL;
static uint64_t rnd() { uint64_t x = rng_state; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; rng_state = x; return x * 0x2545F4914F6CDD1DULL; }
static const uint64_t EDGE64[] = {0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 255, 256, 1023, 1024, 4095, 4096,
    0xFFFFULL, 0x10000ULL, 0x7FFFFFFFULL, 0x80000000ULL, 0xFFFFFFFFULL, 0x100000000ULL, 0x7FFFFFFFFFFFFFFFULL,
    0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
static uint64_t edge_or_random64() {
    switch (rnd() % 4) {
        case 0: return EDGE64[rnd() % (sizeof(EDGE64) / sizeof(EDGE64[0]))];
        case 1: return rnd() % 4096;          // small, in the region the program fills
        case 2: return rnd();                 // uniform 64-bit
        default: return rnd() % 64;           // very small
    }
}
static U256 edge_or_random256() {   // stack operand: usually a 64-bit value, sometimes with high limbs set
    U256 a = u256(edge_or_random64());
    const uint64_t k = rnd() % 16;
    if (k == 0) { a.w[3] = 0x8000000000000000ULL; }                       // 2^255 + low
    else if (k == 1) { a.w[0] = a.w[1] = a.w[2] = a.w[3] = ~0ULL; }      // 2^256 - 1
    else if (k == 2) { a.w[1] = 1; }                                       // 2^64 + low
    return a;
}

// ---------------- reference memory model ----------------
static const uint64_t MAX_BUFFER = 0xFFFFFFFFULL;   // evmone max_buffer_size (2^32 - 1)
static uint64_t words_of(uint64_t bytes) { return (bytes + 31) / 32; }
static int64_t mem_cost(uint64_t words) { return (int64_t)(3 * words + (words * words) / 512); }

struct Ref {
    std::vector<uint8_t> mem;   // current memory image (size is a multiple of 32)
    int64_t gas;                // gas left
    bool failed = false;
    // grow memory to cover [0, new_size); charge the expansion; false when gas runs out
    bool grow(uint64_t new_size) {
        if (new_size <= mem.size()) return true;
        const uint64_t nw = words_of(new_size);
        const int64_t cost = mem_cost(nw) - mem_cost(mem.size() / 32);
        gas -= cost;
        if (gas < 0) return false;
        mem.resize(nw * 32, 0);
        return true;
    }
    // check_memory with a 64-bit size (offset must be < 2^32)
    bool check64(const U256& off, uint64_t size) {
        if (!high_zero(off) || off.w[0] > MAX_BUFFER) return false;
        return grow(off.w[0] + size);
    }
    // check_memory with a 256-bit size: size 0 always valid, size >= 2^32 rejected
    bool check256(const U256& off, const U256& size) {
        if (size.w[0] == 0 && high_zero(size)) return true;
        if (!high_zero(size) || size.w[0] > MAX_BUFFER) return false;
        return check64(off, size.w[0]);
    }
    bool charge(int64_t g) { gas -= g; return gas >= 0; }
    uint64_t rd64(uint64_t off) { uint64_t v = 0; std::memcpy(&v, &mem[off], 8); return v; }
    uint16_t rd16(uint64_t off) { uint16_t v = 0; std::memcpy(&v, &mem[off], 2); return v; }
    int32_t rd32s(uint64_t off) { int32_t v = 0; std::memcpy(&v, &mem[off], 4); return v; }
};

// 256-bit product of two 64-bit values (exact), as a U256
static U256 mul64x64(uint64_t a, uint64_t b) {
    unsigned __int128 p = (unsigned __int128)a * b; return {{(uint64_t)p, (uint64_t)(p >> 64), 0, 0}};
}
// 32 + p*2 for a product p < 2^128: exact in U256
static U256 size_from_count(const U256& p) {
    unsigned __int128 v = ((unsigned __int128)p.w[1] << 64) | p.w[0];
    unsigned __int128 hi_carry = v >> 127;                 // p*2 overflows 128 bits?
    unsigned __int128 s = v * 2 + 32;
    U256 r = {{(uint64_t)s, (uint64_t)(s >> 64), (uint64_t)hi_carry, 0}};
    if (hi_carry == 0 && s < v) r.w[2] = 1;                 // carry from the +32
    return r;
}

// ---------------- program builder ----------------
struct Prog {
    std::vector<uint8_t> c;
    int64_t static_gas = 0;
    void op(uint8_t b) { c.push_back(b); }
    void push(const U256& v) { c.push_back(0x7F); auto b = be32(v); c.insert(c.end(), b.begin(), b.end()); static_gas += 3; }
    void push_small(uint64_t v) { if (v < 256) { c.push_back(0x60); c.push_back((uint8_t)v); } else { c.push_back(0x61); c.push_back((uint8_t)(v >> 8)); c.push_back((uint8_t)v); } static_gas += 3; }
};

enum Instr { T_XTRANSPOSE = 0, T_XBCAST, T_XGATHER, T_XMACC_S, T_XMACC_N, T_XMACC_N16, T_COUNT };
static const char* NAMES[] = {"XTRANSPOSE", "XBCAST", "XGATHER", "XMACC_S", "XMACC_N", "XMACC_N16"};
static const uint8_t OPC[] = {0xDD, 0xE4, 0xE3, 0xE5, 0xE6, 0xE7};
static const int STATIC_GAS[] = {10, 8, 20, 14, 0, 0};

static evmc_vm* VM;
static evmc::MockedHost host;
static const int64_t GAS_LIMIT = 2000000;   // bounds the memory a trial may legitimately expand to

struct Trial {
    Instr ins;
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> stores;   // (offset, 32 bytes) MSTOREs
    std::vector<U256> operands;                                       // pushed in this order (last push = stack top)
};

// write a random header word: four uint64 fields, little-endian, edge-biased
static std::vector<uint8_t> header_word(uint64_t rows, uint64_t cols, uint64_t sr, uint64_t sc) {
    std::vector<uint8_t> b(32); uint64_t f[4] = {rows, cols, sr, sc};
    for (int k = 0; k < 4; ++k) for (int i = 0; i < 8; ++i) b[k * 8 + i] = (uint8_t)(f[k] >> (8 * i));
    return b;
}
static std::vector<uint8_t> random_word() { std::vector<uint8_t> b(32); for (auto& x : b) x = (uint8_t)rnd(); return b; }
static uint64_t small_off() { return (rnd() % 128) * 32 + (rnd() % 4 == 0 ? rnd() % 32 : 0); }   // mostly aligned, sometimes not

static Trial make_trial(Instr ins) {
    Trial t; t.ins = ins;
    const int nstores = 1 + rnd() % 6;
    for (int i = 0; i < nstores; ++i) {
        uint64_t off = small_off();
        if (rnd() % 2) t.stores.push_back({off, header_word(edge_or_random64(), edge_or_random64(), edge_or_random64(), edge_or_random64())});
        else t.stores.push_back({off, random_word()});
    }
    auto off_operand = [&]() -> U256 {
        switch (rnd() % 8) {
            case 0: case 1: case 2: case 3: return u256(t.stores[rnd() % t.stores.size()].first);   // an offset we wrote
            case 4: return u256(small_off());
            default: return edge_or_random256();
        }
    };
    switch (ins) {
        case T_XTRANSPOSE: t.operands = {off_operand()}; break;
        case T_XBCAST:     t.operands = {edge_or_random256(), edge_or_random256(), off_operand()}; break;  // cols, rows, off (top)
        case T_XGATHER:    t.operands = {off_operand(), off_operand(), off_operand()}; break;              // res, ind, data (top)
        case T_XMACC_S:    t.operands = {edge_or_random256(), edge_or_random256(), off_operand(), off_operand(), off_operand()}; break; // start, row, acc, inp, mat (top)
        case T_XMACC_N: case T_XMACC_N16: {
            U256 n = (rnd() % 3 == 0) ? edge_or_random256() : u256(rnd() % 40);
            t.operands = {edge_or_random256(), off_operand(), off_operand(), n}; break;                    // acc word, x, w, n (top)
        }
        default: break;
    }
    return t;
}

static void build(const Trial& t, Prog& p) {
    for (auto& s : t.stores) { p.push(U256{{0, 0, 0, 0}}); std::copy(s.second.begin(), s.second.end(), p.c.end() - 32); p.push_small(s.first); p.op(0x52); p.static_gas += 3; }
    for (auto& o : t.operands) p.push(o);
    p.op(OPC[t.ins]); p.static_gas += STATIC_GAS[t.ins];
    p.op(0x59); p.static_gas += 2;        // MSIZE
    p.push_small(0);                      // offset 0
    p.op(0xF3);                           // RETURN(0, msize)
}

// reference execution: returns true when the program succeeds; fills r.mem and r.gas
static bool reference(const Trial& t, Ref& r, int64_t static_gas) {
    r.gas = GAS_LIMIT - static_gas;   // all static costs are charged in any case before the failing point matters
    if (r.gas < 0) return false;
    for (auto& s : t.stores) { if (!r.grow(s.first + 32)) return false; std::memcpy(&r.mem[s.first], s.second.data(), 32); }
    const auto& o = t.operands;
    switch (t.ins) {
        case T_XTRANSPOSE: {
            const U256 off = o[0];
            if (!r.check64(off, 32)) return false;
            uint64_t a = off.w[0]; uint8_t h[32]; std::memcpy(h, &r.mem[a], 32);
            uint8_t s[32]; std::memcpy(s, h + 8, 8); std::memcpy(s + 8, h, 8); std::memcpy(s + 16, h + 24, 8); std::memcpy(s + 24, h + 16, 8);
            std::memcpy(&r.mem[a], s, 32); return true;
        }
        case T_XBCAST: {
            const U256 off = o[2], rows = o[1], cols = o[0];
            if (!r.check64(off, 32)) return false;
            auto hw = header_word(rows.w[0], cols.w[0], 0, 0); std::memcpy(&r.mem[off.w[0]], hw.data(), 32); return true;
        }
        case T_XGATHER: {
            const U256 od = o[2], oi = o[1], orr = o[0];
            if (!r.check64(od, 32)) return false;
            if (!r.check64(oi, 32)) return false;
            const uint64_t ir = r.rd64(oi.w[0]), ic = r.rd64(oi.w[0] + 8), dr = r.rd64(od.w[0]), dc = r.rd64(od.w[0] + 8);
            const U256 n_idx = mul64x64(ir, ic), n_data = mul64x64(dr, dc);
            if (!r.check256(od, size_from_count(n_data))) return false;
            if (!r.check256(oi, size_from_count(n_idx))) return false;
            if (!r.check256(orr, size_from_count(n_idx))) return false;
            const uint64_t pd = od.w[0], pi = oi.w[0], pr = orr.w[0];
            const uint64_t ni = n_idx.w[0], nd = n_data.w[0];   // both < 2^31 after the checks
            std::memmove(&r.mem[pr], &r.mem[pi], 32);
            for (uint64_t i = 0; i < ni; ++i) {
                const uint16_t idx = r.rd16(pi + 32 + 2 * i);
                uint16_t v = 0; if (idx < nd) v = r.rd16(pd + 32 + 2 * (uint64_t)idx);
                std::memcpy(&r.mem[pr + 32 + 2 * i], &v, 2);
            }
            return true;
        }
        case T_XMACC_S: {
            const U256 om = o[4], oinp = o[3], oacc = o[2], row = o[1], start = o[0];
            if (!r.check64(om, 32)) return false;
            const uint64_t sr = r.rd64(om.w[0] + 16), sc = r.rd64(om.w[0] + 24);
            if (!r.check64(oinp, 16)) return false;
            if (!r.check64(oacc, 32)) return false;
            if (!high_zero(row) || !high_zero(start)) return false;
            // max_elem = row*sr + (start+7)*sc in 256-bit arithmetic (no overflow in the interpreter);
            // gather_top = om + 32 + max_elem*2 must be at most 2^32-1 and allow a 4-byte read.
            // Any term of 2^32 or more makes gather_top exceed the bound, so exact 128-bit sums with
            // overflow flags are enough.
            typedef unsigned __int128 u128;
            bool big = false;
            const u128 a = (u128)row.w[0] * sr;
            const u128 b1 = (u128)start.w[0] * sc, b2 = (u128)7 * sc;
            u128 b = b1 + b2; if (b < b1) big = true;
            u128 max_elem = a + b; if (max_elem < a) big = true;
            if (big || max_elem >= ((u128)1 << 32)) return false;
            const u128 top = (u128)om.w[0] + 32 + max_elem * 2;
            if (top > MAX_BUFFER) return false;
            if (!r.grow((uint64_t)top + 4)) return false;
            const uint64_t pm = om.w[0] + 32, pinp = oinp.w[0], pacc = oacc.w[0];
            const uint64_t base = row.w[0] * sr + start.w[0] * sc;      // <= max_elem <= 2^31-1
            int32_t mv[8], iv[8], av[8];
            for (int j = 0; j < 8; ++j) { mv[j] = (int16_t)r.rd16(pm + 2 * (base + (uint64_t)j * sc)); iv[j] = (int16_t)r.rd16(pinp + 2 * j); av[j] = r.rd32s(pacc + 4 * j); }
            for (int j = 0; j < 8; ++j) { int64_t s = (int64_t)av[j] + (int32_t)mv[j] * (int32_t)iv[j]; int32_t res = s > INT32_MAX ? INT32_MAX : (s < INT32_MIN ? INT32_MIN : (int32_t)s); std::memcpy(&r.mem[pacc + 4 * j], &res, 4); }   // saturating lane addition
            return true;
        }
        case T_XMACC_N: case T_XMACC_N16: {
            const U256 n = o[3], ow = o[2], ox = o[1];
            if (!high_zero(n) || n.w[0] > 0x7FFFFFFFULL) return false;
            const uint64_t nn = n.w[0];
            if (!r.charge((int64_t)nn * (t.ins == T_XMACC_N ? 11 : 22))) return false;
            if (nn == 0) return true;
            const uint64_t bytes = nn * 32;
            if (!r.check64(ow, bytes)) return false;
            if (!r.check64(ox, bytes)) return false;
            return true;   // the accumulator word stays on the stack; memory is only read
        }
        default: return false;
    }
}

int main() {
    const char* lib = getenv("EVMONE_LIB") ? getenv("EVMONE_LIB") : "build/lib/libevmone.so";
    void* h = dlopen(lib, RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 2; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=== RANDOMIZED DIFFERENTIAL TEST of the memory-addressed instructions (seed 0x5EED5EED5EED5EED) ===\n");
    printf("gas limit per trial %lld; every trial compared with the reference model: status, gas used, memory size, memory content\n", (long long)GAS_LIMIT);
    const int TRIALS = 4000;
    int total_fail = 0;
    for (int ins = 0; ins < T_COUNT; ++ins) {
        int ok_succ = 0, ok_rej = 0, fail = 0, shown = 0;
        for (int k = 0; k < TRIALS; ++k) {
            Trial t = make_trial((Instr)ins);
            Prog p; build(t, p);
            Ref r; const bool exp_ok = reference(t, r, p.static_gas);
            evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS_LIMIT;
            auto res = VM->execute(VM, &host.get_interface(), host.to_context(), EVMC_SHANGHAI, &msg, p.c.data(), p.c.size());
            bool good = true; std::string why;
            if (exp_ok) {
                if (res.status_code != EVMC_SUCCESS) { good = false; why = "expected success, got status " + std::to_string((int)res.status_code); }
                else {
                    const int64_t used = GAS_LIMIT - res.gas_left, exp_used = GAS_LIMIT - r.gas;
                    if (used != exp_used) { good = false; why = "gas used " + std::to_string(used) + " expected " + std::to_string(exp_used); }
                    else if (res.output_size != r.mem.size()) { good = false; why = "memory size " + std::to_string(res.output_size) + " expected " + std::to_string(r.mem.size()); }
                    else if (std::memcmp(res.output_data, r.mem.data(), r.mem.size()) != 0) {
                        size_t d = 0; while (d < r.mem.size() && res.output_data[d] == r.mem[d]) ++d;
                        good = false; why = "memory content differs at byte " + std::to_string(d);
                    }
                }
                if (good) ok_succ++;
            } else {
                if (res.status_code == EVMC_SUCCESS) { good = false; why = "expected a clean out-of-gas, got success"; }
                else if (res.status_code != EVMC_OUT_OF_GAS) { good = false; why = "expected out-of-gas, got status " + std::to_string((int)res.status_code); }
                else if (res.gas_left != 0) { good = false; why = "out-of-gas with gas left " + std::to_string((long long)res.gas_left); }
                if (good) ok_rej++;
            }
            if (!good) {
                fail++;
                if (shown < 5) { shown++; printf("  [%s trial %d] %s | operands:", NAMES[ins], k, why.c_str()); for (auto& o : t.operands) printf(" 0x%llx%s", (unsigned long long)o.w[0], high_zero(o) ? "" : "+hi"); printf("\n"); }
            }
            if (res.release) res.release(&res);
        }
        printf("[%-10s] %d trials: %d succeeded as predicted, %d rejected (clean out-of-gas) as predicted, %d mismatches %s\n",
               NAMES[ins], TRIALS, ok_succ, ok_rej, fail, fail == 0 ? "PASSED" : "FAILED");
        total_fail += fail;
    }
    if (total_fail == 0) printf("*** ALL CHECKS PASSED: %d random trials, every result equal to the reference model ***\n", TRIALS * T_COUNT);
    else printf("*** %d MISMATCHES ***\n", total_fail);
    VM->destroy(VM); dlclose(h);
    return total_fail == 0 ? 0 : 1;
}
