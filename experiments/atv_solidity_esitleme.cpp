// atv_solidity_esitleme.cpp — SCOPE MATCHING FOR THE SOLIDITY BASELINE (K=64)
// Goal: isolate the "inner-product loop only" gas from the Path C total (323,713).
// Method: the closed-form gas counter; snapshot before/after the loop emission.
// CHECKS: (1) bit-exact (2) predicted==measured DIFF 0 (3) total==323,713
// (atv_kapaliform K=64 anchor). Also prints ref[0..2] for the Remix comparison.
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cstdint>

static const int64_t GAS=200000000000LL;
static void u64le(std::vector<uint8_t>& b, uint64_t v){ for(int i=0;i<8;++i) b.push_back((v>>(8*i))&0xFF); }
static long long opGas(uint8_t op){
    if(op>=0x60&&op<=0x7F) return 3;
    switch(op){ case 0x01: return 3; case 0x02: return 5; case 0x1a: return 3;
        case 0x51: return 3; case 0x52: return 3; case 0x59: return 2; case 0xF3: return 0;
        default: std::fprintf(stderr,"opGas: unknown %02x\n",op); std::exit(2); }
}
struct Code {
    std::vector<uint8_t> c; long long g=0; long long peakB=0;
    void mem(long long e){ if(e>peakB) peakB=e; }
    void op(uint8_t x){ c.push_back(x); g+=opGas(x); }
    void raw(uint8_t x){ c.push_back(x); }
    void P1(uint8_t v){ op(0x60); raw(v); }
    void P2(int v){ op(0x61); raw((v>>8)&0xFF); raw(v&0xFF); }
    void PUSH(long long v){ if(v<256)P1((uint8_t)v); else if(v<65536)P2((int)v);
        else { op(0x62); raw((v>>16)&0xFF); raw((v>>8)&0xFF); raw(v&0xFF); } }
    void W32(std::vector<uint8_t> b){ while(b.size()<32)b.push_back(0); op(0x7F); for(int i=0;i<32;++i) raw(b[i]); }
    void MS(long long a){ PUSH(a); op(0x52); mem(a+32); }
};
static long long memgas(long long bytes){ long long a=(bytes+31)/32; return 3*a+(a*a)/512; }

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ std::cerr<<"libevmone not found\n"; return 1; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone");
    evmc_vm* vm=create();
    evmc::MockedHost host;
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();
    evmc::address ADDR{}; ADDR.bytes[19]=0x01;

    const int K=64;
    std::vector<int> M(K*K), v(K);
    for(int i=0;i<K*K;++i) M[i]=(i%50)+1;
    for(int c=0;c<K;++c) v[c]=(c%7)+1;
    std::vector<long long> ref(K);
    for(int r=0;r<K;++r){ long long s=0; for(int c=0;c<K;++c) s+=(long long)M[c*K+r]*v[c]; ref[r]=(long long)(uint32_t)s; }

    int matWords=(K*K+15)/16, vecWords=(K+7)/8, MAT=0;
    int VEC=(1+matWords)*32+256, ACC=VEC+vecWords*32+64, OUTV=ACC+64; (void)ACC;

    Code k;
    // data-write segment (matrix + vector into memory)
    { std::vector<uint8_t> hb; u64le(hb,K);u64le(hb,K);u64le(hb,K);u64le(hb,1); k.W32(hb); k.MS(MAT); }
    for(int w=0; w<matWords; ++w){ std::vector<uint8_t> d(32,0);
        for(int e=0;e<16;++e){ int idx=w*16+e; if(idx<K*K){ uint16_t val=(uint16_t)M[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
        k.W32(d); k.MS(MAT+32+w*32); }
    for(int blk=0; blk<vecWords; ++blk){ std::vector<uint8_t> d(32,0);
        for(int e=0;e<8;++e){ int idx=blk*8+e; if(idx<K){ uint16_t val=(uint16_t)v[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
        k.W32(d); k.MS(VEC+blk*32); }
    long long g_veri = k.g;
    // INNER-PRODUCT LOOP (identical to atv_3kova/kapaliform Path C, UNROLLED)
    for(int r=0;r<K;++r){ k.P1(0);
        for(int c=0;c<K;++c){
            long long eB=MAT+32+((long long)c*K+r)*2;
            k.PUSH((eB/32)*32); k.op(0x51); k.mem((eB/32)*32+32); k.P1((uint8_t)(eB%32)); k.op(0x1a);
            k.PUSH(((eB+1)/32)*32); k.op(0x51); k.mem(((eB+1)/32)*32+32); k.P1((uint8_t)((eB+1)%32)); k.op(0x1a);
            k.P2(256); k.op(0x02); k.op(0x01);
            long long vW=VEC+(c/8)*32; int vlo=(c%8)*2;
            k.PUSH(vW); k.op(0x51); k.mem(vW+32); k.P1((uint8_t)vlo); k.op(0x1a);
            k.PUSH(vW); k.op(0x51); k.mem(vW+32); k.P1((uint8_t)(vlo+1)); k.op(0x1a);
            k.P2(256); k.op(0x02); k.op(0x01); k.op(0x02); k.op(0x01); }
        k.MS(OUTV+(long long)r*32); }
    long long g_hesap = k.g - g_veri;
    // tail
    k.op(0x59); k.PUSH(0); k.op(0x52); k.mem(32);
    k.PUSH(OUTV+(long long)K*32); k.PUSH(0); k.op(0xF3); k.mem(OUTV+(long long)K*32);
    long long g_kuyruk = k.g - g_veri - g_hesap;
    long long tahmin = k.g + memgas(k.peakB);

    evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS; msg.recipient=ADDR; msg.code_address=ADDR;
    auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
    long long olcum=GAS-res.gas_left;
    bool be=true;
    if(res.status_code==0&&res.output_data&&(long long)res.output_size>=OUTV+(long long)K*32){
        for(int r=0;r<K;++r){ long long x=0; for(int j=28;j<32;++j) x=(x<<8)|res.output_data[OUTV+(long long)r*32+j];
            if(x!=ref[r]) be=false; }
    } else be=false;
    if(res.release)res.release(&res);

    printf("[CHECK 1] bit-exact: %s\n",be?"PASSED":"FAILED");
    printf("[CHECK 2] pred==meas: %lld/%lld diff=%lld %s\n",tahmin,olcum,olcum-tahmin,olcum==tahmin?"PASSED":"FAILED");
    printf("[CHECK 3] total==323,713 (closed-form K=64 anchor): %s\n",olcum==323713LL?"PASSED":"FAILED");
    printf("\n=== SCOPE MATCHING (K=64, Path C) ===\n");
    printf("data-write: %lld | INNER-PRODUCT LOOP (PURE): %lld | tail: %lld | memory: %lld | TOTAL: %lld\n",
        g_veri, g_hesap, g_kuyruk, memgas(k.peakB), olcum);
    printf("per element (4096 elements): %.2f gas\n",(double)g_hesap/(K*K));
    printf("\n=== REFERENCE VALUES FOR THE REMIX COMPARISON ===\n");
    printf("ref[0]=%lld  ref[1]=%lld  ref[2]=%lld\n",ref[0],ref[1],ref[2]);
    return (be&&olcum==tahmin&&olcum==323713LL)?0:1;
}
