// sec_fuzz_lanes.cpp -- randomized differential test of the stack-only lane instructions of the extension:
// XADD_IMM, XSUB_IMM, XMUL_IMM, XGT_IMM (one immediate configuration byte each), XRELU (immediate shift),
// XSUM32, XPACK and XMACC. Together with sec_fuzz.cpp (the six memory-addressed instructions) this covers
// the whole instruction family.
//
// Every trial pushes random 256-bit words (lanes drawn from an edge-case set and from uniform random bits),
// executes one instruction with a random immediate byte (all 256 values are admissible), stores the result
// word to memory and returns it. A reference model of the lane semantics of Section 4.4 predicts the result
// word and the gas; the interpreter's output is compared byte for byte. The seed is fixed; 8 x 4,000 trials.
// The same program runs on the AVX2 build and on the scalar-fallback build in run_all.sh, so the two builds
// are also compared file by file.
//
// Lane layout: lane j of a word occupies bits [w*j, w*(j+1)) of the 256-bit value for a lane width of w bits,
// so lane 0 is the least significant part of the word (bytes 31 and 30 of the big-endian encoding for 16-bit
// lanes). XMACC multiplies the eight 16-bit lanes of the two top words (the low 128 bits of each) and adds
// the 32-bit products to the eight 32-bit lanes of the third word.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef uint8_t Word[32];   // little-endian: W[0] is the least significant byte (lane 0 side)

static uint64_t rng_state = 0x1A4E51A4E51A4E5ULL;
static uint64_t rnd() { uint64_t x = rng_state; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; rng_state = x; return x * 0x2545F4914F6CDD1DULL; }
static const uint64_t EDGE[] = {0, 1, 2, 7, 8, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x10000, 0x7FFFFFFFULL, 0x80000000ULL,
    0xFFFFFFFFULL, 0x100000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL};
static void random_word(Word w) {
    const int mode = rnd() % 4;
    if (mode == 0) { for (int i = 0; i < 32; ++i) w[i] = (uint8_t)rnd(); return; }          // uniform bytes
    std::memset(w, 0, 32);
    const int lanebytes = (mode == 1) ? 2 : (mode == 2) ? 4 : 8;                             // edge values per lane
    for (int off = 0; off + lanebytes <= 32; off += lanebytes) {
        uint64_t v = (rnd() % 3 == 0) ? rnd() : EDGE[rnd() % (sizeof(EDGE) / sizeof(EDGE[0]))];
        if (rnd() % 4 == 0) v = (uint64_t)(-(int64_t)(v & 0xFFFF));                           // small negative
        for (int b = 0; b < lanebytes; ++b) w[off + b] = (uint8_t)(v >> (8 * b));
    }
}
static uint64_t get_lane(const Word w, int lanebytes, int j) { uint64_t v = 0; for (int b = 0; b < lanebytes; ++b) v |= (uint64_t)w[j * lanebytes + b] << (8 * b); return v; }
static void set_lane(Word w, int lanebytes, int j, uint64_t v) { for (int b = 0; b < lanebytes; ++b) w[j * lanebytes + b] = (uint8_t)(v >> (8 * b)); }

enum Instr { T_XADD = 0, T_XSUB, T_XMUL, T_XGT, T_XRELU, T_XSUM32, T_XPACK, T_XMACC, T_COUNT };
static const char* NAMES[] = {"XADD_IMM", "XSUB_IMM", "XMUL_IMM", "XGT_IMM", "XRELU", "XSUM32", "XPACK", "XMACC"};
static const uint8_t OPC[] = {0xC7, 0xC9, 0xC8, 0xCA, 0xCC, 0xCD, 0xCE, 0xCB};
static const int GAS[] = {6, 6, 8, 6, 6, 6, 6, 11};
static const int NOPER[] = {2, 2, 2, 2, 1, 1, 1, 3};
static const bool HASIMM[] = {true, true, true, true, true, false, false, false};

// reference model: w1 = top of the stack, w2 = second, w3 = third; the result replaces the lowest operand
static void reference(Instr ins, uint8_t imm, const Word w1, const Word w2, const Word w3, Word out) {
    std::memset(out, 0, 32);
    const int width = imm & 3;                       // 0: 8-bit, 1: 16-bit, 2: 32-bit, 3: 64-bit lanes
    const int lb = 1 << width;                       // lane bytes
    const int nl = 32 / lb;
    switch (ins) {
        case T_XADD: for (int j = 0; j < nl; ++j) set_lane(out, lb, j, get_lane(w1, lb, j) + get_lane(w2, lb, j)); break;   // wrap in the lane
        case T_XSUB: for (int j = 0; j < nl; ++j) set_lane(out, lb, j, get_lane(w1, lb, j) - get_lane(w2, lb, j)); break;   // top minus second
        case T_XMUL: {
            const bool fixed = (imm >> 2) & 1; const int fs = (imm >> 3) & 3; const unsigned frac = (fs == 0) ? 4 : (fs == 1) ? 8 : (fs == 2) ? 12 : 4;
            const int mlb = (width >= 2) ? 4 : 2;   // widths 2 and 3 use 32-bit lanes, widths 0 and 1 use 16-bit lanes
            for (int j = 0; j < 32 / mlb; ++j) { uint64_t p = get_lane(w1, mlb, j) * get_lane(w2, mlb, j); if (fixed) p = (p & ((mlb == 4) ? 0xFFFFFFFFULL : 0xFFFFULL)) >> frac; set_lane(out, mlb, j, p); }
            break;
        }
        case T_XGT: {
            const bool sgn = (imm >> 5) & 1; const int glb = (width == 3) ? 2 : lb;   // width 3 is executed as 16-bit lanes
            for (int j = 0; j < 32 / glb; ++j) {
                uint64_t a = get_lane(w1, glb, j), b = get_lane(w2, glb, j); bool gt;
                if (sgn) { const int sh = 64 - 8 * glb; gt = ((int64_t)(a << sh) >> sh) > ((int64_t)(b << sh) >> sh); } else gt = a > b;
                set_lane(out, glb, j, gt ? ((glb == 8) ? ~0ULL : ((1ULL << (8 * glb)) - 1)) : 0);
            }
            break;
        }
        case T_XRELU: {
            unsigned sh = imm; if (sh > 31) sh = 31;   // shifts of 32 and more clear every lane (the lanes are non-negative after the ReLU)
            for (int j = 0; j < 8; ++j) { int32_t x = (int32_t)get_lane(w1, 4, j); if (x < 0) x = 0; x >>= sh; if (x > 32767) x = 32767; set_lane(out, 4, j, (uint32_t)x); }
            break;
        }
        case T_XSUM32: { int64_t s = 0; for (int j = 0; j < 8; ++j) s += (int32_t)get_lane(w1, 4, j); if (s > INT32_MAX) s = INT32_MAX; if (s < INT32_MIN) s = INT32_MIN; set_lane(out, 4, 0, (uint32_t)(int32_t)s); break; }   // exact sum clamped to int32, zero-extended to 256 bits
        case T_XPACK: for (int j = 0; j < 8; ++j) { int32_t x = (int32_t)get_lane(w1, 4, j); if (x > 32767) x = 32767; if (x < -32768) x = -32768; set_lane(out, 2, j, (uint16_t)(int16_t)x); } break;
        case T_XMACC: for (int j = 0; j < 8; ++j) { int32_t a = (int16_t)get_lane(w1, 2, j), b = (int16_t)get_lane(w2, 2, j); int64_t s = (int64_t)(int32_t)get_lane(w3, 4, j) + a * b; if (s > INT32_MAX) s = INT32_MAX; if (s < INT32_MIN) s = INT32_MIN; set_lane(out, 4, j, (uint32_t)(int32_t)s); } break;   // saturating lane addition
        default: break;
    }
}

int main() {
    const char* lib = getenv("EVMONE_LIB") ? getenv("EVMONE_LIB") : "build/lib/libevmone.so";
    void* h = dlopen(lib, RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 2; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); evmc_vm* VM = create();
    evmc::MockedHost host;
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=== RANDOMIZED DIFFERENTIAL TEST of the lane instructions (seed 0x1A4E51A4E51A4E5) ===\n");
    printf("every trial compared with the reference model: status, gas used, result word\n");
    const int TRIALS = 4000; const int64_t GAS_LIMIT = 100000;
    int total_fail = 0;
    for (int ins = 0; ins < T_COUNT; ++ins) {
        int fail = 0, shown = 0;
        for (int k = 0; k < TRIALS; ++k) {
            Word w1, w2, w3; random_word(w1); random_word(w2); random_word(w3);
            const uint8_t imm = (uint8_t)rnd();
            std::vector<uint8_t> p; int64_t sgas = 0;
            auto push_word = [&](const Word w) { p.push_back(0x7F); for (int i = 31; i >= 0; --i) p.push_back(w[i]); sgas += 3; };   // big-endian immediate
            if (NOPER[ins] == 3) push_word(w3);
            if (NOPER[ins] >= 2) push_word(w2);
            push_word(w1);
            p.push_back(OPC[ins]); if (HASIMM[ins]) p.push_back(imm); sgas += GAS[ins];
            p.push_back(0x60); p.push_back(0x00); p.push_back(0x52); sgas += 3 + 3 + 3;   // PUSH1 0, MSTORE (memory expansion to 32 bytes: 3 gas)
            p.push_back(0x60); p.push_back(0x20); p.push_back(0x60); p.push_back(0x00); p.push_back(0xF3); sgas += 3 + 3;   // RETURN(0, 32)
            Word exp; reference((Instr)ins, imm, w1, w2, w3, exp);
            evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS_LIMIT;
            auto res = VM->execute(VM, &host.get_interface(), host.to_context(), EVMC_SHANGHAI, &msg, p.data(), p.size());
            std::string why;
            if (res.status_code != EVMC_SUCCESS) why = "status " + std::to_string((int)res.status_code);
            else if (GAS_LIMIT - res.gas_left != sgas) why = "gas used " + std::to_string((long long)(GAS_LIMIT - res.gas_left)) + " expected " + std::to_string((long long)sgas);
            else if (res.output_size != 32) why = "output size " + std::to_string(res.output_size);
            else { for (int i = 0; i < 32; ++i) if (res.output_data[31 - i] != exp[i]) { why = "result differs at lane byte " + std::to_string(i); break; } }
            if (!why.empty()) {
                fail++;
                if (shown < 4) { shown++; printf("  [%s trial %d imm 0x%02x] %s\n", NAMES[ins], k, imm, why.c_str());
                    if (res.status_code == EVMC_SUCCESS && res.output_size == 32) { printf("    got:      "); for (int i = 31; i >= 0; --i) printf("%02x", res.output_data[31 - i]); printf("\n    expected: "); for (int i = 31; i >= 0; --i) printf("%02x", exp[i]); printf("\n"); } }
            }
            if (res.release) res.release(&res);
        }
        printf("[%-8s] %d trials, %d mismatches %s\n", NAMES[ins], TRIALS, fail, fail == 0 ? "PASSED" : "FAILED");
        total_fail += fail;
    }
    if (total_fail == 0) printf("*** ALL CHECKS PASSED: %d random trials, every result equal to the reference model ***\n", TRIALS * T_COUNT);
    else printf("*** %d MISMATCHES ***\n", total_fail);
    VM->destroy(VM); dlclose(h);
    return total_fail == 0 ? 0 : 1;
}
