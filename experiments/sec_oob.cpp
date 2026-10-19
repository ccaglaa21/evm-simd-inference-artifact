// SECURITY TEST: when XMACC_S is called with an out-of-bounds / int32-overflowing stride,
// it must NOT READ/WRITE host memory and must return a clean EVMC_OUT_OF_GAS (status=3).
// Setup identical to atv_k8_v2.cpp; the difference: we write a bad stride into the MatrixHeader.
#include <evmc/evmc.h>
#include <dlfcn.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdint>
static evmc_vm* VM;
static const int64_t GAS=5000000000LL;
static void u64le(std::vector<uint8_t>& b, uint64_t v){ for(int i=0;i<8;++i) b.push_back((v>>(8*i))&0xFF); }
struct Code {
    std::vector<uint8_t> c;
    void b(uint8_t x){ c.push_back(x); }
    void P1(uint8_t v){ b(0x60); b(v); }
    void P2(int v){ b(0x61); b((v>>8)&0xFF); b(v&0xFF); }
    void PUSH(int v){ if(v<256)P1((uint8_t)v); else P2(v); }
    void W32(std::vector<uint8_t> bytes){ while(bytes.size()<32)bytes.push_back(0); b(0x7F); for(int i=0;i<32;++i) b(bytes[i]); }
    void MS(int a){ PUSH(a); b(0x52); }
    void ML(int a){ PUSH(a); b(0x51); }
};
// Single XMACC_S call; hRow/hCol/hSr/hSc = MatrixHeader values (the bad value is injected here)
static evmc_status_code run_one(uint64_t hRows,uint64_t hCols,uint64_t hSr,uint64_t hSc,
                                int row,int start_col,const char* label){
    const int K=8;
    int MAT=0, VEC=512, ACC=576, OUTV=640;
    Code k;
    // --- MatrixHeader (including the bad stride) ---
    { std::vector<uint8_t> hb; u64le(hb,hRows);u64le(hb,hCols);u64le(hb,hSr);u64le(hb,hSc); k.W32(hb); k.MS(MAT); }
    // --- a few rows of real matrix data (so that at least the legitimate region after the header is filled) ---
    for(int w=0; w<(K*K+15)/16; ++w){ std::vector<uint8_t> d(32,0); for(int e=0;e<16;++e){ int idx=w*16+e; uint16_t val=(uint16_t)((idx%50)+1); d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } k.W32(d); k.MS(MAT+32+w*32); }
    // --- vector (input) ---
    { std::vector<uint8_t> dv(32,0); for(int e=0;e<8;++e){ uint16_t val=(uint16_t)((e%7)+1); dv[e*2]=val&0xFF; dv[e*2+1]=(val>>8)&0xFF; } k.W32(dv); k.MS(VEC); }
    // --- zero the accumulator ---
    { std::vector<uint8_t> da(32,0); k.W32(da); k.MS(ACC); }
    // --- single XMACC_S: push start_col,row,acc,inp,mat (pop order reversed) ---
    k.PUSH(start_col); k.PUSH(row); k.PUSH(ACC); k.PUSH(VEC); k.PUSH(MAT); k.b(0xE5);
    // try to return the result (status=0 would mean a vulnerability)
    k.ML(ACC); k.PUSH(OUTV); k.b(0x52);
    k.PUSH(32); k.PUSH(OUTV); k.b(0xF3);
    evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS;
    auto res=VM->execute(VM,nullptr,nullptr,EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
    auto st=res.status_code;
    printf("[%s] status=%d %s\n", label, (int)st,
        st==EVMC_OUT_OF_GAS ? "(EVMC_OUT_OF_GAS = guard triggered, clean stop)" :
        st==EVMC_SUCCESS     ? "(SUCCESS -- WARNING: if vulnerable, host memory was read!)" :
                               "(unexpected)");
    if(res.release)res.release(&res);
    return st;
}
int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone"); VM=create();
    std::cout<<"=== XMACC_S SECURITY TEST ===\n";
    // Control: normal stride (sr=K=8, sc=1), row=0 -> SUCCESS expected (sanity check)
    auto s0=run_one(8,8,8,1, 0,0, "CONTROL normal stride row=0");
    // Scenario A: huge stride_col=2^30 -> gather_top ~2^34 > 2^32-1 -> OOG expected
    auto sA=run_one(8,8,8, (uint64_t)0x40000000ULL, 0,0, "A  huge stride_col=2^30");
    // Scenario B: stride_col=2^29, start_col=8 -> base_idx exceeds int32 (used to wrap negative) -> OOG
    auto sB=run_one(8,8,8, (uint64_t)0x20000000ULL, 0,8, "B  int32-overflowing stride_col=2^29");
    // Scenario C: offset not far but stride_row huge, row=1 -> max_elem large -> OOG
    auto sC=run_one(8,8,(uint64_t)0x40000000ULL,1, 1,0, "C  huge stride_row=2^30 row=1");
    std::cout<<"\n--- SUMMARY ---\n";
    bool ok = (s0==EVMC_SUCCESS) && (sA==EVMC_OUT_OF_GAS) && (sB==EVMC_OUT_OF_GAS) && (sC==EVMC_OUT_OF_GAS);
    std::cout<<(ok ? "[PASSED] normal case ran; all 3 bad scenarios gave a clean OOG (no heap access).\n"
                   : "[INSPECT] expected pattern did not hold (see the statuses above).\n");
    VM->destroy(VM); dlclose(h); return 0;
}
