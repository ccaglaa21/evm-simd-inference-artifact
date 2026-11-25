// price_bench.cpp — TIME OF EVERY ADDED INSTRUCTION, one instruction at a time, as the basis of the gas schedule.
//
// The gas of an instruction has to cover the time it takes. This program measures that time on the build it
// runs on, for every added instruction, next to the time of the standard instructions that serve as the
// reference. It makes no gas claim of its own: training/price_calibration.py reads its outputs from the four
// builds and derives the schedule from them.
// METHOD. A case is a straight-line program: one call-data copy, then N repetitions of
//     push the operands, execute the instruction (and remove its result where it leaves one),
// then a return. The time of one repetition is
//     ((total - analysis) of the program  minus  (total - analysis) of its copy-and-return program) / N
// with the three-part timing of gram_views.cpp: the program as it is (total), the same bytes behind a STOP
// (analysis and call setup, nothing executed), and the copy-and-return program timed in the same way. The
// pushes of the operands are counted in the time of the instruction and their gas is not credited to it,
// which is the strict direction for a price.
// REFERENCE. The scalar multiply-accumulate step of the packed scalar programs,
//     PUSH offset, MLOAD, PUSH offset, MLOAD, MUL, ADD     (20 gas, eight elements),
// is the standard-instruction work that the multiply-accumulate instructions replace. It is measured for
// three program lengths at the start of the run and once more at the end of the run, so that a drift of the
// machine during the run is visible. For context the program also measures four single standard instructions with the
// pushes of their operands (ADD, MUL, MULMOD, KECCAK256 of one word) and a walk PUSH address, MLOAD, POP
// (8 gas) over sequential addresses and over all addresses of the hostile view.
// CASES.
//   fixed price   XMACC, XMACC_S (on a transposed view and on the hostile view), XRELU, XSUM32, XPACK, XGT_IMM,
//                 XADD_IMM, XSUB_IMM, XMUL_IMM, XTRANSPOSE, XBCAST
//   XGATHER       8 to 4096 pseudo-random indices into an array of 65536 elements
//   XMACC_N       1 to 256 groups of eight elements;   XMACC_N16: 1 to 128 words of sixteen elements;
//                 for both, also rows of 16384 words (0.5 MB each, 1.5 MB of data in all)
//   XMACC_V       unit-stride rows of 8 to 2048 elements, and rows of 262144 elements (0.5 MB each) of a
//                 1.5 MB matrix; rows of a transposed view (both rows strided); rows of the hostile view
//                 (both rows strided, and one strided row against a unit-stride row)
//   XMATMUL       unit-stride operands in dense-layer shapes (rows x 1 row) and in Gram shapes (C x C rows),
//                 from one entry to 16384 entries; a transposed view as both operands (packed once); the
//                 hostile view against one unit-stride row and as both operands
// HOSTILE VIEW. A header is contract data, so a contract can choose any strides. In the hostile view two
// elements of a row are whole pages apart and two rows are 64 bytes apart, so every element of the view lies
// in a cache line of its own and a row touches as many pages as it has elements. The view has 28672 elements
// in every size (64 rows of 448 elements down to 512 rows of 56) and spans 1.84 MB of memory. A transaction
// can pay for this memory under the cap of 2^24 gas; the program prints the expansion gas.
// DATA. Elements are pseudo-random in [-128, 127]. The small values keep the accumulated sums small against the
// 32-bit range of a lane.
// CHECKS per case: the program ends with success, its gas equals the static count of the generator plus the
// memory expansion, and a fingerprint of the returned bytes is printed. The gas values and the fingerprints
// must be identical on every build. Timing depends on the machine and the build.
// With NO_TIMING=1 in the environment the programs are executed and checked but not timed, and the TIME tables
// are left out (used for the sanitizer builds and for emulated runs). The gas tables and check lines are the same.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
static evmc::MockedHost g_host;
static evmc_vm* VM;
static bool NOTIME = false;                    // set by NO_TIMING=1: execute and check only
static const int64_t GAS = 200000000000LL;
// gas of the added instructions; must equal the values in the interpreter
static const long long V_CALL = 50, V_GROUP = 10, V_STRIDED = 4;                   // XMACC_V
static const long long M_CALL = 270, M_ENTRY = 5, M_GROUP = 9, M_PACKED = 6;       // XMATMUL
static const long long N_CALL = 24, N_GROUP = 12, N16_CALL = 32, N16_WORD = 24;    // XMACC_N, XMACC_N16
static const long long P_XMACC = 23, P_XMACC_S = 74, P_XRELU = 14, P_XSUM32 = 14, P_XPACK = 15, P_XGT = 19, P_XADD = 17, P_XSUB = 17, P_XMUL = 21, P_XTRANSPOSE = 22, P_XBCAST = 18, P_XGATHER = 22, P_XGATHER_INDEX = 1;
struct Code {
    std::vector<uint8_t> c; long long sgas = 0;
    void b(uint8_t x, long long g) { c.push_back(x); sgas += g; }
    void PUSH(uint64_t v) { int n = 0; for (uint64_t t = v; t; t >>= 8) ++n; if (n == 0) n = 1;
        b((uint8_t)(0x5F + n), 3); for (int k = n - 1; k >= 0; --k) c.push_back((uint8_t)((v >> (8 * k)) & 0xFF)); }
    void W32(const uint8_t* w) { b(0x7F, 3); for (int i = 0; i < 32; ++i) c.push_back(w[i]); }
    void MLOAD() { b(0x51, 3); } void MSTORE() { b(0x52, 3); } void MUL() { b(0x02, 5); } void ADD() { b(0x01, 3); }
    void DUP(int n) { b((uint8_t)(0x7F + n), 3); } void POP() { b(0x50, 2); }
    void IMM(uint8_t op, uint8_t imm, long long g) { b(op, g); c.push_back(imm); }
    void CDCOPY(uint64_t bytes) { PUSH(bytes); PUSH(0); PUSH(0); b(0x37, 3 + 3 * (long long)((bytes + 31) / 32)); }
    void RET(uint64_t off, uint64_t size) { PUSH(size); PUSH(off); b(0xF3, 0); }
};
static long long memgas(uint64_t bytes) { long long w = (long long)((bytes + 31) / 32); return 3 * w + w * w / 512; }
static uint64_t pad32(uint64_t x) { return (x + 31) / 32 * 32; }
static uint32_t rnd = 2026u; static uint32_t rnd32() { rnd = rnd * 1664525u + 1013904223u; return rnd; }
static int small() { return (int)((rnd32() >> 8) % 256) - 128; }
static long long g8(long long n) { return (n + 7) / 8; }

struct Tm { double tot = 0, base = 0; };              // total time and time behind a STOP (analysis and call setup)
struct Case {
    std::string instr, variant; long long n = 0;       // instruction, variant, size parameter of the variant
    long long calls = 0;                               // repetitions in the program
    long long groups = 0, entries = 0, strided = 0, igas = 0;   // per repetition: groups of eight elements, output entries, strided elements, gas of the instruction
    Code k; std::vector<uint8_t> cd; uint64_t retOff = 0, retSize = 32, mend = 0;
    long long gas = 0, gasH = 0; uint64_t fp = 0; bool ok = false; Tm t, tH;
};
// ---------- call-data helpers ----------
static void put64(std::vector<uint8_t>& m, size_t at, uint64_t v) { for (int k = 0; k < 8; ++k) m[at + k] = (uint8_t)(v >> (8 * k)); }
static void hdr(std::vector<uint8_t>& m, size_t at, uint64_t rows, uint64_t cols, uint64_t sr, uint64_t sc) { put64(m, at, rows); put64(m, at + 8, cols); put64(m, at + 16, sr); put64(m, at + 24, sc); }
static void le16(std::vector<uint8_t>& m, size_t at, int v) { uint16_t u = (uint16_t)(int16_t)v; m[at] = (uint8_t)(u & 0xFF); m[at + 1] = (uint8_t)(u >> 8); }
// a memory word with int16 lanes as MLOAD sees them: lane j at bytes 31-2j (low) and 30-2j (high)
static void laneWord(uint8_t* w, int lanes) { std::memset(w, 0, 32); for (int j = 0; j < lanes; ++j) { uint16_t u = (uint16_t)(int16_t)small(); w[31 - 2 * j] = (uint8_t)(u & 0xFF); w[30 - 2 * j] = (uint8_t)(u >> 8); } }
// a unit-stride matrix with header at `at`: rows x cols elements; returns the padded size in bytes
static uint64_t unitMatrix(std::vector<uint8_t>& m, size_t at, uint64_t rows, uint64_t cols) {
    const uint64_t bytes = pad32(32 + 2 * rows * cols); if (m.size() < at + bytes) m.resize(at + bytes, 0);
    hdr(m, at, rows, cols, cols, 1); for (uint64_t e = 0; e < rows * cols; ++e) le16(m, at + 32 + 2 * e, small()); return bytes; }
// the hostile view with n elements per row: 28672 elements, one per cache line, column stride of 448/n pages
struct Hostile { uint64_t rows, cols, sr, sc, bytes; };
static Hostile hostileView(std::vector<uint8_t>& m, uint64_t n) {
    Hostile h; h.cols = n; h.rows = 64 * (448 / n); h.sr = 32; h.sc = 2048 * (448 / n);
    h.bytes = pad32(32 + 2 * (h.sr * (h.rows - 1) + h.sc * (h.cols - 1) + 1)); m.assign(h.bytes, 0); hdr(m, 0, h.rows, h.cols, h.sr, h.sc);
    for (uint64_t r = 0; r < h.rows; ++r) for (uint64_t t = 0; t < h.cols; ++t) le16(m, 32 + 2 * (h.sr * r + h.sc * t), small());
    return h; }
// repetitions for a program of about 0.3 ms, from a rough estimate of one repetition in picoseconds (integer arithmetic,
// so the count is the same on every machine)
static long long reps(long long est_ps, long long lo, long long hi) { long long N = 300000000LL / est_ps; return std::max(lo, std::min(hi, N)); }
static void finish(Case& c, uint64_t retOff, uint64_t retSize) {   // return, and the memory end for the static gas count
    c.retOff = retOff; c.retSize = retSize; c.k.RET(retOff, retSize); c.mend = std::max<uint64_t>(pad32(c.cd.size()), retOff + retSize); }
static void accTail(Case& c) { c.k.PUSH(0); c.k.MSTORE(); finish(c, 0, 32); }      // the word on top of the stack is the result

// ---------- reference cases (standard instructions only) ----------
static Case refStep(long long N) {
    Case c; c.instr = "REF_STEP"; c.variant = "PUSH MLOAD PUSH MLOAD MUL ADD"; c.n = N; c.calls = N; c.groups = 1; c.igas = 20;
    const int R = 128, CW = 32; const uint64_t SC = 0, PW = (uint64_t)R * 32; c.cd.assign(((size_t)R + (size_t)R * CW) * 32, 0);
    for (int t = 0; t < R; ++t) { uint32_t v = rnd32() % 4096; c.cd[SC + t * 32 + 30] = (uint8_t)(v >> 8); c.cd[SC + t * 32 + 31] = (uint8_t)(v & 0xFF); }
    for (size_t w = 0; w < (size_t)R * CW; ++w) for (int j = 0; j < 8; ++j) { uint32_t v = rnd32() % 4096; uint8_t* p = &c.cd[PW + w * 32]; p[31 - 4 * j] = (uint8_t)(v & 0xFF); p[30 - 4 * j] = (uint8_t)(v >> 8); }
    c.k.CDCOPY(c.cd.size()); c.k.PUSH(0);
    for (long long s = 0; s < N; ++s) { const uint64_t t = (uint64_t)(s % R), g = (uint64_t)((s / R) % CW);
        c.k.PUSH(SC + 32 * t); c.k.MLOAD(); c.k.PUSH(PW + (t * CW + g) * 32); c.k.MLOAD(); c.k.MUL(); c.k.ADD(); }
    accTail(c); return c; }
static Case refWalk(bool hostile) {
    Case c; c.instr = "REF_WALK"; c.variant = hostile ? "PUSH MLOAD POP, all addresses of the hostile view" : "PUSH MLOAD POP, sequential words"; c.n = 448; c.calls = 28672; c.igas = 8;
    if (hostile) { Hostile h = hostileView(c.cd, 448); (void)h; } else { c.cd.assign(28672 * 32 + 32, 0); for (auto& x : c.cd) x = (uint8_t)(rnd32() >> 24); }
    c.k.CDCOPY(c.cd.size());
    for (long long q = 0; q < c.calls; ++q) { const uint64_t r = (uint64_t)(q / 448), t = (uint64_t)(q % 448);      // row by row, as the instructions read the view
        c.k.PUSH(hostile ? 32 + 64 * r + 4096 * t : 32 + 32 * (uint64_t)q); c.k.MLOAD(); c.k.POP(); }
    finish(c, 0, 32); return c; }
// one standard instruction with the pushes of its operands, repeated; the result stays on the stack or is removed
static Case refOp(const std::string& name) {
    Case c; c.instr = "REF_OP"; c.calls = 16384; c.cd.assign(32, 0); for (auto& x : c.cd) x = (uint8_t)(rnd32() >> 24);
    uint8_t A[32], B[32], M[32]; for (int i = 0; i < 32; ++i) { A[i] = (uint8_t)(rnd32() >> 24); B[i] = (uint8_t)(rnd32() >> 24); M[i] = (uint8_t)(rnd32() >> 24); } A[31] |= 1; M[0] |= 0x80;
    Code& k = c.k; k.CDCOPY(32); k.W32(M); k.W32(A); k.W32(B);                                   // stack: M, A, B (B is the running value)
    if (name == "ADD") { c.variant = "DUP, ADD"; c.igas = 6; for (long long q = 0; q < c.calls; ++q) { k.DUP(2); k.ADD(); } }
    else if (name == "MUL") { c.variant = "DUP, MUL"; c.igas = 8; for (long long q = 0; q < c.calls; ++q) { k.DUP(2); k.MUL(); } }
    else if (name == "MULMOD") { c.variant = "DUP, DUP, DUP, MULMOD, POP"; c.igas = 19; for (long long q = 0; q < c.calls; ++q) { k.DUP(3); k.DUP(3); k.DUP(3); k.b(0x09, 8); k.POP(); } }
    else { c.variant = "PUSH, PUSH, KECCAK256 of one word, POP"; c.igas = 44; for (long long q = 0; q < c.calls; ++q) { k.PUSH(32); k.PUSH(0); k.b(0x20, 36); k.POP(); } }
    accTail(c); return c; }

// ---------- fixed-price instructions ----------
static Case fixedCase(const std::string& name) {
    Case c; c.instr = name; c.variant = "fixed price"; c.calls = 16384;
    uint8_t A[32], B[32], V[32]; laneWord(A, 8); laneWord(B, 8);
    for (int j = 0; j < 8; ++j) { uint32_t u = (uint32_t)(small() * 1000); V[31 - 4 * j] = (uint8_t)(u & 0xFF); V[30 - 4 * j] = (uint8_t)((u >> 8) & 0xFF); V[29 - 4 * j] = (uint8_t)((u >> 16) & 0xFF); V[28 - 4 * j] = (uint8_t)(u >> 24); }
    Code& k = c.k;
    if (name == "XMACC") { c.igas = P_XMACC; c.groups = 1; c.cd.assign(32, 0); k.CDCOPY(32); k.W32(A); k.W32(B); k.PUSH(0);
        for (long long q = 0; q < c.calls; ++q) { k.DUP(3); k.DUP(3); k.b(0xCB, P_XMACC); } accTail(c); }
    else if (name == "XRELU") { c.igas = P_XRELU; c.cd.assign(32, 0); k.CDCOPY(32); k.W32(V);
        for (long long q = 0; q < c.calls; ++q) { k.IMM(0xCC, 0, P_XRELU); }
        accTail(c); }
    else if (name == "XSUM32") { c.igas = P_XSUM32; c.cd.assign(32, 0); k.CDCOPY(32); k.W32(V);
        for (long long q = 0; q < c.calls; ++q) { k.DUP(1); k.b(0xCD, P_XSUM32); k.POP(); } accTail(c); }
    else if (name == "XPACK") { c.igas = P_XPACK; c.cd.assign(32, 0); k.CDCOPY(32); k.W32(V);
        for (long long q = 0; q < c.calls; ++q) { k.b(0xCE, P_XPACK); }
        accTail(c); }
    else if (name == "XGT_IMM" || name == "XADD_IMM" || name == "XSUB_IMM" || name == "XMUL_IMM") {
        const uint8_t op = name == "XGT_IMM" ? 0xCA : name == "XADD_IMM" ? 0xC7 : name == "XSUB_IMM" ? 0xC9 : 0xC8; c.igas = name == "XMUL_IMM" ? P_XMUL : name == "XGT_IMM" ? P_XGT : name == "XADD_IMM" ? P_XADD : P_XSUB;
        c.cd.assign(32, 0); k.CDCOPY(32); k.W32(A); k.W32(B);
        for (long long q = 0; q < c.calls; ++q) { k.DUP(2); k.IMM(op, 0x21, c.igas); } accTail(c); }     // 0x21: signed 16-bit lanes
    else if (name == "XTRANSPOSE") { c.igas = P_XTRANSPOSE; unitMatrix(c.cd, 0, 4, 12); k.CDCOPY(c.cd.size());
        for (long long q = 0; q < c.calls; ++q) { k.PUSH(0); k.b(0xDD, P_XTRANSPOSE); k.POP(); } finish(c, 0, 32); }
    else if (name == "XBCAST") { c.igas = P_XBCAST; c.cd.assign(64, 0); le16(c.cd, 32, 7); k.CDCOPY(64);
        for (long long q = 0; q < c.calls; ++q) { k.PUSH(5); k.PUSH(3); k.PUSH(0); k.b(0xE4, P_XBCAST); k.POP(); k.POP(); k.POP(); } finish(c, 0, 64); }
    else if (name == "XMACC_S" || name == "XMACC_S hostile") { c.instr = "XMACC_S"; c.igas = P_XMACC_S; c.groups = 1; c.strided = 8; c.calls = 8192;   // eight strided elements of one row
        uint64_t MB;
        if (name == "XMACC_S") { MB = unitMatrix(c.cd, 0, 448, 64); hdr(c.cd, 0, 64, 448, 1, 64); }                       // transposed view of a stored 448 x 64 matrix
        else { c.variant = "fixed price, hostile view"; Hostile h = hostileView(c.cd, 448); MB = h.bytes; }
        const uint64_t INP = MB, ACC = MB + 32; c.cd.resize(MB + 64, 0);
        for (int j = 0; j < 8; ++j) le16(c.cd, INP + 2 * j, small());
        k.CDCOPY(c.cd.size());
        for (long long q = 0; q < c.calls; ++q) { k.PUSH((uint64_t)((8 * q) % 440)); k.PUSH((uint64_t)(q % 64)); k.PUSH(ACC); k.PUSH(INP); k.PUSH(0); k.b(0xE5, P_XMACC_S); }
        finish(c, ACC, 32); }
    return c; }
static Case gatherCase(long long n) {
    Case c; c.instr = "XGATHER"; c.variant = "pseudo-random indices into 65536 elements"; c.n = n; c.igas = P_XGATHER + P_XGATHER_INDEX * n; c.calls = reps(20000 + n * 1500, 64, 8192);
    const uint64_t DATA = 0, DB = unitMatrix(c.cd, DATA, 1, 65536), IND = DB, IB = pad32(32 + 2 * (uint64_t)n), RES = IND + IB;      // every 16-bit index is valid
    c.cd.resize(IND + IB, 0); hdr(c.cd, IND, 1, (uint64_t)n, (uint64_t)n, 1); for (long long i = 0; i < n; ++i) { uint16_t u = (uint16_t)(rnd32() >> 8); c.cd[IND + 32 + 2 * i] = (uint8_t)(u & 0xFF); c.cd[IND + 33 + 2 * i] = (uint8_t)(u >> 8); }
    c.k.CDCOPY(c.cd.size());
    for (long long q = 0; q < c.calls; ++q) { c.k.PUSH(RES); c.k.PUSH(IND); c.k.PUSH(DATA); c.k.b(0xE3, c.igas); c.k.POP(); c.k.POP(); c.k.POP(); }
    finish(c, RES, IB); return c; }

// ---------- row-level instructions on contiguous words ----------
static Case maccN(long long n, bool sixteen) {
    Case c; c.instr = sixteen ? "XMACC_N16" : "XMACC_N"; c.variant = sixteen ? "words of sixteen elements" : "groups of eight elements"; c.n = n;
    c.groups = sixteen ? 2 * n : n; c.igas = sixteen ? N16_CALL + N16_WORD * n : N_CALL + N_GROUP * n; c.calls = reps(20000 + n * (sixteen ? 4000 : 2000), n > 1024 ? 12 : 256, 8192);
    const int RW = n > 1024 ? 2 : 32; const uint64_t X = 0, W = (uint64_t)n * 32; c.cd.assign((size_t)(n + (long long)RW * n) * 32, 0);      // long rows: 1.5 MB in all
    for (size_t w = 0; w < c.cd.size() / 32; ++w) laneWord(&c.cd[w * 32], sixteen ? 16 : 8);
    c.k.CDCOPY(c.cd.size()); c.k.PUSH(0);
    for (long long q = 0; q < c.calls; ++q) { c.k.PUSH(X); c.k.PUSH(W + (uint64_t)(q % RW) * (uint64_t)n * 32); c.k.PUSH((uint64_t)n); c.k.b(sixteen ? 0xE7 : 0xE6, c.igas); }
    accTail(c); return c; }

// ---------- XMACC_V ----------
enum { UNIT = 0, TRANSPOSED = 1, HOSTILE2 = 2, HOSTILE1 = 3 };
static Case maccV(long long n, int kind) {
    Case c; c.instr = "XMACC_V"; c.n = n; c.groups = g8(n); Code& k = c.k;
    if (kind == UNIT) { const long long RW = n > 8192 ? 3 : 64; c.variant = n > 8192 ? "two unit-stride rows of a 1.5 MB matrix" : "two unit-stride rows"; unitMatrix(c.cd, 0, (uint64_t)RW, (uint64_t)n);
        c.calls = reps(25000 + n * 250, n > 8192 ? 12 : 256, 8192); c.igas = V_CALL + V_GROUP * g8(n);
        k.CDCOPY(c.cd.size()); k.PUSH(0);
        for (long long q = 0; q < c.calls; ++q) { k.PUSH((uint64_t)((7 * q + 13) % RW)); k.PUSH(0); k.PUSH((uint64_t)((7 * q) % RW)); k.PUSH(0); k.b(0xEA, c.igas); } }
    else if (kind == TRANSPOSED) { c.variant = "two rows of a transposed view (stride 128 bytes)"; unitMatrix(c.cd, 0, (uint64_t)n, 64);   // n x 64 stored row by row
        c.calls = reps(25000 + n * 1500, 256, 8192); c.strided = 2 * n; c.igas = V_CALL + V_GROUP * g8(n) + V_STRIDED * 2 * n;
        k.CDCOPY(c.cd.size()); k.PUSH(0); k.b(0xDD, P_XTRANSPOSE); k.POP(); k.PUSH(0);
        for (long long q = 0; q < c.calls; ++q) { k.PUSH((uint64_t)((7 * q + 13) % 64)); k.PUSH(0); k.PUSH((uint64_t)((7 * q) % 64)); k.PUSH(0); k.b(0xEA, c.igas); } }
    else if (kind == HOSTILE2) { c.variant = "two rows of the hostile view"; Hostile h = hostileView(c.cd, (uint64_t)n);
        c.calls = std::max<long long>(2048, (long long)(4 * h.rows)); c.strided = 2 * n; c.igas = V_CALL + V_GROUP * g8(n) + V_STRIDED * 2 * n;
        k.CDCOPY(c.cd.size()); k.PUSH(0);
        for (long long q = 0; q < c.calls; ++q) { k.PUSH((uint64_t)((q + (long long)h.rows / 2) % (long long)h.rows)); k.PUSH(0); k.PUSH((uint64_t)(q % (long long)h.rows)); k.PUSH(0); k.b(0xEA, c.igas); } }
    else { c.variant = "one row of the hostile view and one unit-stride row"; Hostile h = hostileView(c.cd, (uint64_t)n); const uint64_t XA = h.bytes; unitMatrix(c.cd, XA, 1, (uint64_t)n);
        c.calls = std::max<long long>(2048, (long long)(4 * h.rows)); c.strided = n; c.igas = V_CALL + V_GROUP * g8(n) + V_STRIDED * n;
        k.CDCOPY(c.cd.size()); k.PUSH(0);
        for (long long q = 0; q < c.calls; ++q) { k.PUSH((uint64_t)(q % (long long)h.rows)); k.PUSH(0); k.PUSH(0); k.PUSH(XA); k.b(0xEA, c.igas); } }
    accTail(c); return c; }

// ---------- XMATMUL ----------
enum { M_DENSE = 0, M_GRAM = 1, M_TRANSPOSED = 2, M_HOSTILE1 = 3, M_HOSTILEG = 4 };
static Case matmul(long long ra, long long rb, long long n, int kind) {
    Case c; c.instr = "XMATMUL"; c.n = n; c.entries = ra * rb; c.groups = ra * rb * g8(n); Code& k = c.k; uint64_t offA = 0, offB = 0, dst = 0; long long packed = 0;
    char v[96];
    if (kind == M_DENSE) { std::snprintf(v, sizeof v, "unit stride, %lld row%s x 1 row", ra, ra == 1 ? "" : "s"); const uint64_t ab = unitMatrix(c.cd, 0, (uint64_t)ra, (uint64_t)n); offB = ab; dst = ab + unitMatrix(c.cd, ab, 1, (uint64_t)n); }
    else if (kind == M_GRAM) { std::snprintf(v, sizeof v, "unit stride, %lld x %lld rows (one matrix)", ra, rb); dst = unitMatrix(c.cd, 0, (uint64_t)ra, (uint64_t)n); }
    else if (kind == M_TRANSPOSED) { std::snprintf(v, sizeof v, "transposed view, %lld x %lld rows (one view)", ra, rb); dst = unitMatrix(c.cd, 0, (uint64_t)n, (uint64_t)ra); packed = ra * n; }   // n x C stored row by row; one view as both operands is packed once
    else if (kind == M_HOSTILE1) { std::snprintf(v, sizeof v, "1 unit-stride row x %lld rows of the hostile view", rb); Hostile h = hostileView(c.cd, (uint64_t)n); offA = h.bytes; offB = 0; dst = h.bytes + unitMatrix(c.cd, h.bytes, 1, (uint64_t)n); packed = rb * n; }
    else { std::snprintf(v, sizeof v, "hostile view, %lld x %lld rows (one view)", ra, rb); Hostile h = hostileView(c.cd, (uint64_t)n); dst = h.bytes; packed = ra * n; }
    c.variant = v; c.strided = packed; c.igas = M_CALL + c.entries * (M_ENTRY + M_GROUP * g8(n)) + M_PACKED * packed;
    c.calls = reps(60000 + c.entries * (8000 + n * 250) + packed * 3000, 1, 4096);
    if (kind == M_HOSTILE1) { c.calls = 12; }                                                  // several passes over the hostile view in one program
    if (kind == M_HOSTILEG) { c.calls = 2; }
    k.CDCOPY(c.cd.size()); if (kind == M_TRANSPOSED) { k.PUSH(0); k.b(0xDD, P_XTRANSPOSE); k.POP(); }
    for (long long q = 0; q < c.calls; ++q) { k.PUSH(dst); k.PUSH(offB); k.PUSH(offA); k.b(0xE9, c.igas); }
    finish(c, dst, (uint64_t)c.entries * 32); return c; }

// ---------- execution and timing ----------
static uint64_t fnv(const uint8_t* p, size_t n) { uint64_t h = 1469598103934665603ULL; for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; } return h; }
static double timeCode(const std::vector<uint8_t>& code, const std::vector<uint8_t>& cd, const char* nm, const char* what) {
    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS; msg.input_data = cd.data(); msg.input_size = cd.size();
    auto once = [&]() { auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, code.data(), code.size()); if (r.release) r.release(&r); };
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now(); once(); auto t1 = clk::now();
    double est = std::chrono::duration<double>(t1 - t0).count(); if (est <= 0) est = 1e-7;
    long long Rp = (long long)std::ceil(0.010 / est); if (Rp < 1) Rp = 1; if (Rp > 20000) Rp = 20000;
    std::vector<double> S; double tot = 0;
    while (true) { auto a = clk::now(); for (long long i = 0; i < Rp; ++i) once(); auto bb = clk::now();
        double dt = std::chrono::duration<double>(bb - a).count(); S.push_back(dt / Rp); tot += dt; size_t n = S.size();
        if (n >= 3 && ((n >= 7 && tot >= 1.0) || n >= 31 || tot >= 60.0)) break; }
    std::sort(S.begin(), S.end()); const double med = S[S.size() / 2];
    double mu = 0; for (double x : S) mu += x; mu /= S.size(); double sd = 0; for (double x : S) sd += (x - mu) * (x - mu); sd = std::sqrt(sd / S.size());
    std::printf("  [%s %s] med=%.9f s  min=%.9f  max=%.9f  CV=%.1f%%  (samples=%zu, inner-reps=%lld)\n", nm, what, med, S.front(), S.back(), 100.0 * sd / mu, S.size(), Rp);
    std::fflush(stdout);
    return med;
}
static void timeBoth(const std::vector<uint8_t>& code, const std::vector<uint8_t>& cd, const char* nm, Tm& t) {
    if (NOTIME) return;
    t.tot = timeCode(code, cd, nm, "total");
    std::vector<uint8_t> st; st.reserve(code.size() + 1); st.push_back(0x00); st.insert(st.end(), code.begin(), code.end());   // STOP, then the same bytes
    t.base = timeCode(st, cd, nm, "analysis");
}
struct Res { std::string instr, variant; long long n, calls, groups, entries, strided, igas, gas, gasH; double tot, base, htot, hbase; };
static std::vector<Res> RES;
static int runCase(Case c) {
    Code H; H.CDCOPY(c.cd.size()); H.RET(c.retOff, c.retSize);
    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = GAS; msg.input_data = c.cd.data(); msg.input_size = c.cd.size();
    auto r = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, c.k.c.data(), c.k.c.size());
    c.gas = GAS - r.gas_left; const bool okr = r.status_code == EVMC_SUCCESS && r.output_size == c.retSize; c.fp = okr ? fnv(r.output_data, r.output_size) : 0; if (r.release) r.release(&r);
    auto r2 = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, c.k.c.data(), c.k.c.size());
    const bool same = (GAS - r2.gas_left) == c.gas; if (r2.release) r2.release(&r2);
    auto rh = VM->execute(VM, &g_host.get_interface(), g_host.to_context(), EVMC_CANCUN, &msg, H.c.data(), H.c.size());
    c.gasH = GAS - rh.gas_left; const bool okh = rh.status_code == EVMC_SUCCESS && c.gasH == H.sgas + memgas(c.mend); if (rh.release) rh.release(&rh);
    const long long pred = c.k.sgas + memgas(c.mend);
    c.ok = okr && okh && same && c.gas == pred;
    char nm[160]; std::snprintf(nm, sizeof nm, "%s, %s, n=%lld", c.instr.c_str(), c.variant.c_str(), c.n);
    std::printf("[%s] repetitions %lld | per repetition: %lld groups of eight, %lld entries, %lld strided elements, instruction gas %lld | gas %lld | static count + memory %lld: %s | copy-and-return gas %lld | fingerprint %016llx | %s\n",
        nm, c.calls, c.groups, c.entries, c.strided, c.igas, c.gas, pred, c.gas == pred ? "equal" : "DIFFERENT", c.gasH, (unsigned long long)c.fp, c.ok ? "PASSED" : "FAILED");
    timeBoth(c.k.c, c.cd, nm, c.t); timeBoth(H.c, c.cd, "copy-and-return", c.tH);
    RES.push_back({c.instr, c.variant, c.n, c.calls, c.groups, c.entries, c.strided, c.igas, c.gas, c.gasH, c.t.tot, c.t.base, c.tH.tot, c.tH.base});
    return c.ok ? 0 : 1;
}

int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    NOTIME = std::getenv("NO_TIMING") != nullptr;
    int fail = 0;
    { std::vector<uint8_t> m; Hostile hv = hostileView(m, 448); const long long CAP = 1LL << 24;
      std::printf("hostile view: %llu elements, one per cache line, %llu bytes of memory, memory expansion gas %lld (%.1f%% of the per-transaction cap of 2^24 gas)\n",
          (unsigned long long)(hv.rows * hv.cols), (unsigned long long)hv.bytes, memgas(hv.bytes), 100.0 * memgas(hv.bytes) / CAP); }
    std::printf("---- reference: standard instructions ----\n");
    for (long long N : {512LL, 4096LL, 32768LL}) fail |= runCase(refStep(N));
    for (const char* nm : {"ADD", "MUL", "MULMOD", "KECCAK256"}) fail |= runCase(refOp(nm));
    fail |= runCase(refWalk(false)); fail |= runCase(refWalk(true));
    std::printf("---- fixed-price instructions ----\n");
    for (const char* nm : {"XMACC", "XMACC_S", "XMACC_S hostile", "XRELU", "XSUM32", "XPACK", "XGT_IMM", "XADD_IMM", "XSUB_IMM", "XMUL_IMM", "XTRANSPOSE", "XBCAST"}) fail |= runCase(fixedCase(nm));
    std::printf("---- XGATHER ----\n");
    for (long long n : {8LL, 64LL, 512LL, 4096LL}) fail |= runCase(gatherCase(n));
    std::printf("---- XMACC_N and XMACC_N16 ----\n");
    for (long long n : {1LL, 2LL, 4LL, 8LL, 16LL, 32LL, 64LL, 98LL, 128LL, 256LL, 16384LL}) fail |= runCase(maccN(n, false));
    for (long long n : {1LL, 2LL, 4LL, 8LL, 16LL, 32LL, 49LL, 64LL, 128LL, 16384LL}) fail |= runCase(maccN(n, true));
    std::printf("---- XMACC_V ----\n");
    for (long long n : {8LL, 16LL, 32LL, 64LL, 128LL, 256LL, 512LL, 784LL, 1024LL, 2048LL, 262144LL}) fail |= runCase(maccV(n, UNIT));
    for (long long n : {8LL, 16LL, 32LL, 64LL, 128LL, 256LL, 448LL, 1024LL}) fail |= runCase(maccV(n, TRANSPOSED));
    for (long long n : {56LL, 112LL, 224LL, 448LL}) fail |= runCase(maccV(n, HOSTILE2));
    fail |= runCase(maccV(448, HOSTILE1));
    std::printf("---- XMATMUL ----\n");
    { const long long D[][2] = {{1, 8}, {1, 64}, {4, 8}, {10, 32}, {32, 784}, {128, 64}, {128, 256}, {128, 784}, {128, 2048}};
      for (auto& d : D) fail |= runCase(matmul(d[0], 1, d[1], M_DENSE));
      const long long Gm[][2] = {{2, 8}, {8, 128}, {16, 128}, {32, 128}, {64, 128}, {128, 128}, {64, 8}, {32, 16}, {32, 64}, {32, 512}};
      for (auto& d : Gm) fail |= runCase(matmul(d[0], d[0], d[1], M_GRAM));
      const long long Tm_[][2] = {{8, 128}, {16, 128}, {32, 128}, {64, 128}, {32, 32}, {32, 512}};
      for (auto& d : Tm_) fail |= runCase(matmul(d[0], d[0], d[1], M_TRANSPOSED));
      for (long long n : {56LL, 112LL, 224LL, 448LL}) fail |= runCase(matmul(1, 64 * (448 / n), n, M_HOSTILE1));
      fail |= runCase(matmul(64, 64, 448, M_HOSTILEG)); }
    std::printf("---- reference again at the end of the run (a check for drift of the machine during the run) ----\n");
    { Case c = refStep(4096); c.variant += ", end of run"; fail |= runCase(c); }
    // times below are machine and build dependent; every table from here on starts with the word TIME
    if (NOTIME) std::printf("TIME tables are left out (NO_TIMING is set)\n");
    else {
    std::printf("TIME OF ONE REPETITION = ((total - analysis) of the program minus (total - analysis) of its copy-and-return program) / repetitions; ms columns are medians for one call of the whole program\n");
    std::printf("instruction | variant | n | repetitions | groups | entries | strided | instruction gas | total ms | analysis ms | copy-return ms | compute ms | ns per repetition | ns per instruction gas\n");
    for (auto& r : RES) { const double comp = (r.tot - r.base) - (r.htot - r.hbase), per = comp * 1e9 / r.calls;
        std::printf("%s | %s | %lld | %lld | %lld | %lld | %lld | %lld | %.5f | %.5f | %.5f | %.5f | %.3f | %.4f\n", r.instr.c_str(), r.variant.c_str(), r.n, r.calls, r.groups, r.entries, r.strided, r.igas,
            r.tot * 1e3, r.base * 1e3, (r.htot - r.hbase) * 1e3, comp * 1e3, per, per / r.igas); }
    }
    std::printf("\n%s\n", fail ? "*** A CHECK FAILED -> the numbers are not valid ***" : "*** ALL CHECKS PASSED -> gas is valid; timing numbers are valid for this build ***");
    VM->destroy(VM); dlclose(h);
    return fail;
}
