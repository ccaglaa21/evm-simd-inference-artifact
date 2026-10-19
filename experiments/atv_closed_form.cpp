// atv_closed_form.cpp — CLOSED-FORM GAS MODEL: NO fitting, only COUNTING.
// Claim: G(K) = opcode_gas(from counting) + memory(3a+a^2/512), 0% error.
// This program reproduces and documents the claim with the CURRENT opcode version:
//   PREDICTION = count from the opcode-gas table during emission + canonical formula from the peak memory touched
//   MEASUREMENT = GAS - gas_left (and MSIZE)
// CHECKS (every K, every path): (1) bit-exact  (2) predicted==measured GAS (diff 0)
//   (3) predicted==measured MSIZE  (4) the K^2 coefficient of the opcode gas is EXACTLY CONSTANT (polynomial)
//   (5) the coefficients solved as integers from K=8/16/32 hold EXACTLY at the remaining 7 points (not a fit).
// The opcode-gas table must match instructions_traits.hpp; if it is wrong, check 2 catches it.
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdint>

static evmc_vm* VM;
static const int64_t GAS=200000000000LL;
static void u64le(std::vector<uint8_t>& b, uint64_t v){ for(int i=0;i<8;++i) b.push_back((v>>(8*i))&0xFF); }

// opcode -> static gas (Shanghai + custom opcodes; must match the traits)
static long long opGas(uint8_t op){
    if(op>=0x60&&op<=0x7F) return 3;            // PUSH1..32
    switch(op){
        case 0x01: return 3;  // ADD
        case 0x02: return 5;  // MUL
        case 0x1a: return 3;  // BYTE
        case 0x51: return 3;  // MLOAD
        case 0x52: return 3;  // MSTORE
        case 0x53: return 3;  // MSTORE8
        case 0x59: return 2;  // MSIZE
        case 0xF3: return 0;  // RETURN
        case 0xDD: return 10; // XTRANSPOSE
        case 0xE5: return 14; // XMACC_S
        default: std::fprintf(stderr,"opGas: unknown opcode %02x\n",op); std::exit(2);
    }
}

struct Code {
    std::vector<uint8_t> c;
    long long g=0;        // opcode gas counter (excluding memory)
    long long peakB=0;    // peak memory touched (bytes)
    void mem(long long endB){ if(endB>peakB) peakB=endB; }
    void op(uint8_t x){ c.push_back(x); g+=opGas(x); }
    void raw(uint8_t x){ c.push_back(x); }               // immediate byte (no gas)
    void P1(uint8_t v){ op(0x60); raw(v); }
    void P2(int v){ op(0x61); raw((v>>8)&0xFF); raw(v&0xFF); }
    void P3(int v){ op(0x62); raw((v>>16)&0xFF); raw((v>>8)&0xFF); raw(v&0xFF); }
    void P4(unsigned v){ op(0x63); raw((v>>24)&0xFF); raw((v>>16)&0xFF); raw((v>>8)&0xFF); raw(v&0xFF); }
    void PUSH(long long v){ if(v<256)P1((uint8_t)v); else if(v<65536)P2((int)v); else if(v<16777216)P3((int)v); else P4((unsigned)v); }
    void W32(std::vector<uint8_t> bytes){ while(bytes.size()<32)bytes.push_back(0); op(0x7F); for(int i=0;i<32;++i) raw(bytes[i]); }
    void MS(long long a){ PUSH(a); op(0x52); mem(a+32); }
    void ML(long long a){ PUSH(a); op(0x51); mem(a+32); }
};

static long long memgas(long long bytes){ long long a=(bytes+31)/32; return 3*a+(a*a)/512; }

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ std::cerr<<"libevmone not found\n"; return 1; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone"); VM=create();
    evmc::MockedHost host;
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();
    evmc::address ADDR{}; ADDR.bytes[19]=0x01;

    std::vector<int> Ks={8,16,32,64,128,256,512,1024,2048,4096};
    const int NK=(int)Ks.size();
    // records: [Kidx][path 0=C 1=B]
    long long OG[10][2]={}, PMEM[10][2]={}, PTOT[10][2]={}, MTOT[10][2]={}, MMS[10][2]={}, PMS[10][2]={};
    bool BE[10]={};
    int fail=0;

    for(int ki=0; ki<NK; ++ki){
        int K=Ks[ki];
        std::vector<int> M(K*K), v(K);
        for(int i=0;i<K*K;++i) M[i]=(i%50)+1;
        for(int c=0;c<K;++c) v[c]=(c%7)+1;
        std::vector<long long> ref(K);
        for(int r=0;r<K;++r){ long long s=0; for(int c=0;c<K;++c) s+=(long long)M[c*K+r]*v[c]; ref[r]=(long long)(uint32_t)s; }

        int matWords=(K*K+15)/16, vecWords=(K+7)/8, MAT=0;
        int VEC=(1+matWords)*32+256, ACC=VEC+vecWords*32+64, OUTV=ACC+64;

        auto emitMatMem=[&](Code& k){   // header + matrix INTO MEMORY (PUSH32+MSTORE), NO SLOAD
            std::vector<uint8_t> hb; u64le(hb,K);u64le(hb,K);u64le(hb,K);u64le(hb,1); k.W32(hb); k.MS(MAT);
            for(int w=0; w<matWords; ++w){ std::vector<uint8_t> d(32,0);
                for(int e=0;e<16;++e){ int idx=w*16+e; if(idx<K*K){ uint16_t val=(uint16_t)M[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
                k.W32(d); k.MS(MAT+32+w*32); } };
        auto emitVec=[&](Code& k){
            for(int blk=0; blk<vecWords; ++blk){ std::vector<uint8_t> d(32,0);
                for(int e=0;e<8;++e){ int idx=blk*8+e; if(idx<K){ uint16_t val=(uint16_t)v[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
                k.W32(d); k.MS(VEC+blk*32); } };
        auto emitTail=[&](Code& k){     // MSIZE into word0 + return together with the result region
            k.op(0x59); k.PUSH(0); k.op(0x52); k.mem(32);
            k.PUSH(OUTV+(long long)K*32); k.PUSH(0); k.op(0xF3); k.mem(OUTV+(long long)K*32); };

        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS; msg.recipient=ADDR; msg.code_address=ADDR;
        auto run=[&](Code& k, std::vector<long long>& out, int simd, long long& msv)->long long{
            auto res=VM->execute(VM,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
            long long g=GAS-res.gas_left; msv=-1;
            if(res.status_code==0 && res.output_data && (long long)res.output_size >= OUTV+(long long)K*32){
                unsigned long long m=0; for(int j=24;j<32;++j) m=(m<<8)|res.output_data[j]; msv=(long long)m;
                if(simd){ for(int r=0;r<K;++r){ long long s=0; for(int i=0;i<8;++i){ uint32_t x=0; for(int kk=0;kk<4;++kk) x|=((uint32_t)res.output_data[OUTV+(long long)r*32+i*4+kk])<<(8*kk); s+=(int32_t)x; } out[r]=(long long)(uint32_t)s; } }
                else { for(int r=0;r<K;++r){ long long x=0; for(int j=28;j<32;++j) x=(x<<8)|res.output_data[OUTV+(long long)r*32+j]; out[r]=x; } }
            } else { std::printf("K=%d: EXECUTE ERROR status=%d\n",K,(int)res.status_code); }
            if(res.release)res.release(&res); return g;
        };

        // ---- PATH C (fair scalar, identical to the atv_cost_split idiom) ----
        std::vector<long long> rC(K,-1); long long msC=-1;
        {
            Code k; emitMatMem(k); emitVec(k);
            for(int r=0;r<K;++r){ k.P1(0);
                for(int c=0;c<K;++c){
                    long long elemB=MAT+32+((long long)c*K+r)*2;
                    k.PUSH((elemB/32)*32); k.op(0x51); k.mem((elemB/32)*32+32); k.P1((uint8_t)(elemB%32)); k.op(0x1a);
                    k.PUSH(((elemB+1)/32)*32); k.op(0x51); k.mem(((elemB+1)/32)*32+32); k.P1((uint8_t)((elemB+1)%32)); k.op(0x1a);
                    k.P2(256); k.op(0x02); k.op(0x01);
                    long long vW=VEC+(c/8)*32; int vlo=(c%8)*2;
                    k.PUSH(vW); k.op(0x51); k.mem(vW+32); k.P1((uint8_t)vlo); k.op(0x1a);
                    k.PUSH(vW); k.op(0x51); k.mem(vW+32); k.P1((uint8_t)(vlo+1)); k.op(0x1a);
                    k.P2(256); k.op(0x02); k.op(0x01); k.op(0x02); k.op(0x01); }
                k.MS(OUTV+(long long)r*32); }
            long long ogC=k.g;                       // opcode gas before the tail (for the report)
            emitTail(k);
            OG[ki][0]=k.g; PMS[ki][0]=k.peakB; PMEM[ki][0]=memgas(k.peakB); PTOT[ki][0]=k.g+PMEM[ki][0];
            MTOT[ki][0]=run(k,rC,0,msC); MMS[ki][0]=msC; (void)ogC;
        }
        // ---- PATH B (zero-copy + SIMD, identical to atv_cost_split) ----
        std::vector<long long> rB(K,-1); long long msB=-1;
        {
            Code k; emitMatMem(k); emitVec(k);
            k.PUSH(MAT); k.op(0xDD);
            for(int r=0;r<K;++r){ { std::vector<uint8_t> da(32,0); k.W32(da); k.MS(ACC); }
                for(int blk=0; blk<vecWords; ++blk){ k.PUSH(blk*8); k.PUSH(r); k.PUSH(ACC); k.PUSH(VEC+blk*32); k.PUSH(MAT); k.op(0xE5); }
                k.ML(ACC); k.MS(OUTV+(long long)r*32); }
            emitTail(k);
            OG[ki][1]=k.g; PMS[ki][1]=k.peakB; PMEM[ki][1]=memgas(k.peakB); PTOT[ki][1]=k.g+PMEM[ki][1];
            MTOT[ki][1]=run(k,rB,1,msB); MMS[ki][1]=msB;
        }
        bool ok=true; for(int r=0;r<K;++r) if(rC[r]!=ref[r]||rB[r]!=ref[r]) ok=false;
        BE[ki]=ok; if(!ok) fail=1;
        std::printf("[K=%4d] bit-exact:%s | C: pred=%lld meas=%lld diff=%lld | B: pred=%lld meas=%lld diff=%lld | MSIZE C:%lld/%lld B:%lld/%lld\n",
            K, ok?"OK":"FAIL", PTOT[ki][0],MTOT[ki][0],MTOT[ki][0]-PTOT[ki][0],
            PTOT[ki][1],MTOT[ki][1],MTOT[ki][1]-PTOT[ki][1], PMS[ki][0],MMS[ki][0],PMS[ki][1],MMS[ki][1]);
        std::fflush(stdout);
        if(MTOT[ki][0]!=PTOT[ki][0]||MTOT[ki][1]!=PTOT[ki][1]||PMS[ki][0]!=MMS[ki][0]||PMS[ki][1]!=MMS[ki][1]) fail=1;
    }

    // ---- closed-form coefficients: INTEGER solution from K=8,16,32, verification at the remaining points (NO fit) ----
    std::printf("\n================ CLOSED FORM (opcode gas, excluding memory) ================\n");
    for(int p=0;p<2;++p){
        const char* pn = p? "B (SIMD)":"C (scalar)";
        // OG = a*K^2 + b*K + c ; K=8,16,32 -> integer solution
        long long K1=8,K2=16,K3=32, y1=OG[0][p],y2=OG[1][p],y3=OG[2][p];
        // method of differences (integer): d1=(y2-y1)/(K2-K1), d2=(y3-y2)/(K3-K2), a=(d2-d1)/(K3-K1)
        long long d1n=y2-y1, d2n=y3-y2;
        double d1=(double)d1n/(K2-K1), d2=(double)d2n/(K3-K2);
        double a=(d2-d1)/(K3-K1), b=d1-a*(K1+K2), c=(double)y1-a*K1*K1-b*K1;
        std::printf("%s: OG(K) = %.6f*K^2 + %.6f*K + %.1f\n",pn,a,b,c);
        int bad=0;
        for(int ki=3;ki<NK;++ki){ long long K=Ks[ki];
            long long pred=(long long)(a*K*K+b*K+c+0.5);
            if(pred!=OG[ki][p]){ std::printf("   K=%lld: polynomial=%lld count=%lld DIFF=%lld\n",K,pred,OG[ki][p],OG[ki][p]-pred); bad=1; } }
        std::printf("   polynomial check at the remaining 7 points: %s\n", bad?"FAILED":"EXACT (diff 0)");
        if(bad) fail=1;
    }
    double aC,aB;
    { long long y1=OG[0][0],y2=OG[1][0],y3=OG[2][0]; double d1=(y2-y1)/8.0,d2=(y3-y2)/16.0; aC=(d2-d1)/24.0; }
    { long long y1=OG[0][1],y2=OG[1][1],y3=OG[2][1]; double d1=(y2-y1)/8.0,d2=(y3-y2)/16.0; aB=(d2-d1)/24.0; }
    std::printf("\nASYMPTOTIC PURE-COMPUTE C/B (K->inf) = aC/aB = %.6f/%.6f = %.2fx\n",aC,aB,aC/aB);

    // ---- per-K summary: pure compute = measured total - memory(measured MSIZE); ratio C/B ----
    std::printf("\n================ PURE COMPUTE PER K (measured total - memory gas from measured MSIZE) ================\n");
    std::printf("     K |   pure compute C |   pure compute B |    C/B |     memory C |     memory B | pred-meas C | pred-meas B\n");
    std::printf("-------+------------------+------------------+--------+--------------+--------------+-------------+------------\n");
    for(int ki=0; ki<NK; ++ki){
        long long memC=memgas(MMS[ki][0]), memB=memgas(MMS[ki][1]);
        long long pcC=MTOT[ki][0]-memC, pcB=MTOT[ki][1]-memB;
        std::printf(" %5d | %16lld | %16lld | %6.2f | %12lld | %12lld | %11lld | %10lld\n",
            Ks[ki], pcC, pcB, (double)pcC/(double)pcB, memC, memB, PTOT[ki][0]-MTOT[ki][0], PTOT[ki][1]-MTOT[ki][1]);
    }
    std::printf("(compare with the MSIZE-measured 18.39/18.67/18.74x of atv_cost_split: must agree in direction)\n");

    if(fail) std::printf("\n*** AT LEAST ONE CHECK FAILED -> closed-form claim NOT DOCUMENTED, do NOT record ***\n");
    else     std::printf("\n*** ALL CHECKS PASSED -> closed form DOCUMENTED: gas = count + canonical memory, diff 0 ***\n");
    VM->destroy(VM); dlclose(h); return fail;
}
