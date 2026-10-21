// sec_jumpdest.cpp — JUMPDEST ANALYSIS TEST for the five instructions that carry a one-byte immediate
// (XADD_IMM 0xc7, XMUL_IMM 0xc8, XSUB_IMM 0xc9, XGT_IMM 0xca, XRELU 0xcc).
//
// The jump-destination analysis of the baseline interpreter must skip the immediate byte of these
// instructions, exactly as it skips PUSH data. Otherwise an immediate equal to 0x5b (JUMPDEST) would be
// accepted as a jump target, which breaks the consensus rule that a jump may only land on a real
// JUMPDEST. For every instruction the test executes:
// Operands are zero words, so the lane results are zero on every backend whatever the immediate means.
//   A  the instruction with immediate 0x5b, followed by a JUMP whose target is the immediate's position
//      -> must fail with EVMC_BAD_JUMP_DESTINATION (status 7)
//   B  the same instruction with immediate 0x5b, then a JUMP to a real JUMPDEST placed after it
//      -> must succeed, which shows that the instruction itself still executes with that immediate
// A control program jumps into the data byte of a PUSH1 0x5b and must fail in the same way.
// The output is deterministic and is compared between the two builds.
#include <evmc/evmc.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <vector>

static evmc_vm* VM;
static evmc_status_code run(const std::vector<uint8_t>& code) {
    evmc_message msg{}; msg.kind = EVMC_CALL; msg.gas = 1000000;
    auto r = VM->execute(VM, nullptr, nullptr, EVMC_SHANGHAI, &msg, code.data(), code.size());
    auto st = r.status_code; if (r.release) r.release(&r); return st;
}
int main() {
    void* h = dlopen("build/lib/libevmone.so", RTLD_NOW);
    if (!h) { std::printf("libevmone not found\n"); return 1; }
    auto create = (evmc_vm * (*)()) dlsym(h, "evmc_create_evmone"); VM = create();
    std::printf("=== JUMPDEST ANALYSIS TEST (immediate bytes are not jump destinations) ===\n");
    struct Ins { uint8_t op; const char* name; int operands; };
    const Ins ins[] = {{0xC7, "XADD_IMM", 2}, {0xC8, "XMUL_IMM", 2}, {0xC9, "XSUB_IMM", 2}, {0xCA, "XGT_IMM", 2}, {0xCC, "XRELU", 1}};
    int pass = 0, total = 0;
    for (const Ins& in : ins) {
        // A: [PUSH1 1]*operands, OP 0x5b, PUSH1 <pos of 0x5b>, JUMP, STOP
        std::vector<uint8_t> a;
        for (int k = 0; k < in.operands; ++k) { a.push_back(0x60); a.push_back(0x00); }
        a.push_back(in.op); const uint8_t imm_pos = (uint8_t)a.size(); a.push_back(0x5B);
        a.push_back(0x60); a.push_back(imm_pos); a.push_back(0x56); a.push_back(0x00);
        auto sa = run(a);
        // B: same prefix, then PUSH1 <pos of JUMPDEST>, JUMP, JUMPDEST, STOP
        std::vector<uint8_t> b;
        for (int k = 0; k < in.operands; ++k) { b.push_back(0x60); b.push_back(0x00); }
        b.push_back(in.op); b.push_back(0x5B);
        const uint8_t jd_pos = (uint8_t)(b.size() + 3);
        b.push_back(0x60); b.push_back(jd_pos); b.push_back(0x56); b.push_back(0x5B); b.push_back(0x00);
        auto sb = run(b);
        bool ok = (sa == EVMC_BAD_JUMP_DESTINATION) && (sb == EVMC_SUCCESS); ++total; pass += ok;
        std::printf("[%s] %-9s immediate 0x5b: jump into the immediate -> status %d (expected %d BAD_JUMP_DESTINATION) | jump to a real JUMPDEST after it -> status %d (expected 0)\n",
                    ok ? "PASS" : "FAIL", in.name, (int)sa, (int)EVMC_BAD_JUMP_DESTINATION, (int)sb);
    }
    { std::vector<uint8_t> c = {0x60, 0x5B, 0x60, 0x01, 0x56, 0x00};   // PUSH1 0x5b, PUSH1 1, JUMP -> into PUSH data
      auto sc = run(c); bool ok = sc == EVMC_BAD_JUMP_DESTINATION; ++total; pass += ok;
      std::printf("[%s] control: jump into the data byte of PUSH1 0x5b -> status %d (expected %d)\n", ok ? "PASS" : "FAIL", (int)sc, (int)EVMC_BAD_JUMP_DESTINATION); }
    std::printf("\n--- SUMMARY --- %d/%d passed%s\n", pass, total, pass == total ? "  *** ALL CHECKS PASSED ***" : "  *** INSPECT ***");
    VM->destroy(VM); dlclose(h); return pass == total ? 0 : 1;
}
