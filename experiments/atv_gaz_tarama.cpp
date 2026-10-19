// atv_gaz_tarama.cpp — TEMPLATE: atv_3kova.cpp. SWEEPS the pure-compute C/B over the XMACC_S gas price.
// atv_3kova.cpp is UNTOUCHED. Same 3-bucket split; in addition:
//   (1) the number of XMACC_S (0xE5) calls on path B is COUNTED (nXmacc).
//   (2) DERIVATION: computeB(g)=computeB(14)+(g-14)*nXmacc, C/B(g)=computeC/computeB(g), g in {14..84}.
// The DERIVATION IS NOT A MEASUREMENT (constant-gas-per-opcode assumption). Cross-checked at g=84 with a real rebuild.
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdint>

static evmc_vm* VM;
static const int64_t GAS=20000000000LL;
static void u64le(std::vector<uint8_t>& b, uint64_t v){ for(int i=0;i<8;++i) b.push_back((v>>(8*i))&0xFF); }

struct Code {
    std::vector<uint8_t> c;
    long long xmacc=0;   // <<< XMACC_S opcode counter (incremented only where the opcode is emitted, not for immediate bytes)
    void b(uint8_t x){ c.push_back(x); }
    void P1(uint8_t v){ b(0x60); b(v); }
    void P2(int v){ b(0x61); b((v>>8)&0xFF); b(v&0xFF); }
    void P3(int v){ b(0x62); b((v>>16)&0xFF); b((v>>8)&0xFF); b(v&0xFF); }
    void P4(unsigned v){ b(0x63); b((v>>24)&0xFF); b((v>>16)&0xFF); b((v>>8)&0xFF); b(v&0xFF); }
    void PUSH(int v){ if(v<256)P1((uint8_t)v); else if(v<65536)P2(v); else if(v<16777216)P3(v); else P4((unsigned)v); }
    void W32(std::vector<uint8_t> bytes){ while(bytes.size()<32)bytes.push_back(0); b(0x7F); for(int i=0;i<32;++i) b(bytes[i]); }
    void MS(int a){ PUSH(a); b(0x52); }
    void ML(int a){ PUSH(a); b(0x51); }
    void PUSHslot(uint32_t s){ std::vector<uint8_t> k(32,0); k[28]=(s>>24)&0xFF;k[29]=(s>>16)&0xFF;k[30]=(s>>8)&0xFF;k[31]=s&0xFF; W32(k); }
};

static evmc::address ADDR{};

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ std::cerr<<"libevmone not found\n"; return 1; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone"); VM=create();
    ADDR.bytes[18]=0xc0; ADDR.bytes[19]=0xde;
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();

    std::vector<int> Ks={256,1024,4096};
    const int NK=(int)Ks.size();

    long long G[2][8][3]={};
    long long MSv[2][8][3]={};
    bool      OKf[2][8]={};
    long long MW[8]={};
    long long NX[8]={};   // <<< number of XMACC_S on path B (per Kidx)

    for(int phase=0; phase<2; ++phase){
        bool WARM=(phase==1);
        for(int ki=0; ki<NK; ++ki){
            int K=Ks[ki];
            std::vector<int> M(K*K), v(K);
            for(int i=0;i<K*K;++i) M[i]=(i%50)+1;
            for(int c=0;c<K;++c) v[c]=(c%7)+1;
            std::vector<long long> ref(K);
            for(int r=0;r<K;++r){ long long s=0; for(int c=0;c<K;++c) s+=(long long)M[c*K+r]*v[c]; ref[r]=(long long)(uint32_t)s; }

            int matWords=(K*K+15)/16, vecWords=(K+7)/8; int MAT=0;
            MW[ki]=matWords;

            evmc::MockedHost host;
            for(int w=0; w<matWords; ++w){
                std::vector<uint8_t> d(32,0);
                for(int e=0;e<16;++e){ int idx=w*16+e; if(idx<K*K){ uint16_t val=(uint16_t)M[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
                evmc::bytes32 key{}, val{};
                key.bytes[28]=(w>>24)&0xFF;key.bytes[29]=(w>>16)&0xFF;key.bytes[30]=(w>>8)&0xFF;key.bytes[31]=w&0xFF;
                for(int i=0;i<32;++i) val.bytes[i]=d[i];
                host.accounts[ADDR].storage[key] = evmc::StorageValue{ val, WARM?EVMC_ACCESS_WARM:EVMC_ACCESS_COLD };
            }
            auto reset_access=[&](){ for(int w=0; w<matWords; ++w){ evmc::bytes32 key{}; key.bytes[28]=(w>>24)&0xFF;key.bytes[29]=(w>>16)&0xFF;key.bytes[30]=(w>>8)&0xFF;key.bytes[31]=w&0xFF; host.accounts[ADDR].storage[key].access_status = WARM?EVMC_ACCESS_WARM:EVMC_ACCESS_COLD; } };

            auto emitLoad=[&](Code& k){
                std::vector<uint8_t> hb; u64le(hb,K);u64le(hb,K);u64le(hb,K);u64le(hb,1); k.W32(hb); k.MS(MAT);
                for(int w=0; w<matWords; ++w){ k.PUSHslot((uint32_t)w); k.b(0x54); k.MS(MAT+32+w*32); }
            };
            auto writeVecBlocks=[&](Code& k, int VEC){
                for(int blk=0; blk<vecWords; ++blk){ std::vector<uint8_t> d(32,0); for(int e=0;e<8;++e){ int idx=blk*8+e; if(idx<K){ uint16_t val=(uint16_t)v[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } } k.W32(d); k.MS(VEC+blk*32); }
            };

            evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS; msg.recipient=ADDR; msg.code_address=ADDR;

            auto emitMsizeReturn=[&](Code& k, int OUTV){
                k.b(0x59); k.PUSH(0); k.b(0x52); k.PUSH(OUTV+K*32); k.PUSH(0); k.b(0xF3);
            };
            auto runWO=[&](Code& k, std::vector<long long>& out, int simd, int OUTV, long long& msv)->long long{
                reset_access(); auto res=VM->execute(VM,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
                long long g=GAS-res.gas_left; msv=-1;
                if(res.status_code==0 && res.output_data && (long long)res.output_size >= (long long)OUTV+(long long)K*32){
                    unsigned long long m=0; for(int j=24;j<32;++j) m=(m<<8)|res.output_data[j]; msv=(long long)m;
                    if(simd){ for(int r=0;r<K;++r){ long long s=0; for(int i=0;i<8;++i){ uint32_t x=0; for(int kk=0;kk<4;++kk) x|=((uint32_t)res.output_data[OUTV+r*32+i*4+kk])<<(8*kk); s+=(int32_t)x; } out[r]=(long long)(uint32_t)s; } }
                    else { for(int r=0;r<K;++r){ long long x=0; for(int j=28;j<32;++j) x=(x<<8)|res.output_data[OUTV+r*32+j]; out[r]=x; } }
                }
                if(res.release)res.release(&res); return g;
            };

            std::vector<long long> rA(K,-1); long long msA=-1;
            {
                int TMP=(1+matWords)*32, VEC=(1+matWords+1+matWords)*32+256, ACC=VEC+vecWords*32+64, OUTV=ACC+64; (void)ACC;
                Code k; emitLoad(k);
                { std::vector<uint8_t> hb; u64le(hb,K);u64le(hb,K);u64le(hb,K);u64le(hb,1); k.W32(hb); k.MS(TMP); }
                for(int r=0;r<K;++r) for(int c=0;c<K;++c){ int srcB=MAT+32+(c*K+r)*2, dstB=TMP+32+(r*K+c)*2;
                    k.PUSH((srcB/32)*32); k.b(0x51); k.P1(srcB%32); k.b(0x1a); k.PUSH(dstB); k.b(0x53);
                    k.PUSH(((srcB+1)/32)*32); k.b(0x51); k.P1((srcB+1)%32); k.b(0x1a); k.PUSH(dstB+1); k.b(0x53); }
                for(int r=0;r<K;++r){ k.P1(0); for(int c=0;c<K;++c){ int elemB=TMP+32+(r*K+c)*2;
                    k.PUSH((elemB/32)*32); k.b(0x51); k.P1(elemB%32); k.b(0x1a);
                    k.PUSH(((elemB+1)/32)*32); k.b(0x51); k.P1((elemB+1)%32); k.b(0x1a);
                    k.P2(256); k.b(0x02); k.b(0x01); k.PUSH(v[c]); k.b(0x02); k.b(0x01); }
                    k.PUSH(OUTV+r*32); k.b(0x52); }
                emitMsizeReturn(k,OUTV);
                G[phase][ki][0]=runWO(k,rA,0,OUTV,msA);
            }
            MSv[phase][ki][0]=msA;

            std::vector<long long> rC(K,-1); long long msC=-1;
            {
                int VEC=(1+matWords)*32+256, ACC=VEC+vecWords*32+64, OUTV=ACC+64; (void)ACC;
                Code k; emitLoad(k); writeVecBlocks(k,VEC);
                for(int r=0;r<K;++r){ k.P1(0); for(int c=0;c<K;++c){
                    int elemB=MAT+32+(c*K+r)*2;
                    k.PUSH((elemB/32)*32); k.b(0x51); k.P1(elemB%32); k.b(0x1a);
                    k.PUSH(((elemB+1)/32)*32); k.b(0x51); k.P1((elemB+1)%32); k.b(0x1a);
                    k.P2(256); k.b(0x02); k.b(0x01);
                    int vWord=VEC+(c/8)*32, vlo=(c%8)*2;
                    k.PUSH(vWord); k.b(0x51); k.P1(vlo); k.b(0x1a);
                    k.PUSH(vWord); k.b(0x51); k.P1(vlo+1); k.b(0x1a);
                    k.P2(256); k.b(0x02); k.b(0x01); k.b(0x02); k.b(0x01); }
                    k.PUSH(OUTV+r*32); k.b(0x52); }
                emitMsizeReturn(k,OUTV);
                G[phase][ki][1]=runWO(k,rC,0,OUTV,msC);
            }
            MSv[phase][ki][1]=msC;

            std::vector<long long> rB(K,-1); long long msB=-1;
            {
                int VEC=(1+matWords)*32+256, ACC=VEC+vecWords*32+64, OUTV=ACC+64;
                Code k; emitLoad(k); writeVecBlocks(k,VEC);
                k.PUSH(MAT); k.b(0xDD);
                for(int r=0;r<K;++r){ { std::vector<uint8_t> da(32,0); k.W32(da); k.MS(ACC); }
                    for(int blk=0; blk<vecWords; ++blk){ k.PUSH(blk*8); k.PUSH(r); k.PUSH(ACC); k.PUSH(VEC+blk*32); k.PUSH(MAT); k.b(0xE5); ++k.xmacc; }
                    k.ML(ACC); k.PUSH(OUTV+r*32); k.b(0x52); }
                emitMsizeReturn(k,OUTV);
                G[phase][ki][2]=runWO(k,rB,1,OUTV,msB);
                if(phase==0) NX[ki]=k.xmacc;   // <<< record the XMACC_S count (from the cold build)
            }
            MSv[phase][ki][2]=msB;

            bool ok=true; for(int r=0;r<K;++r) if(rA[r]!=ref[r]||rC[r]!=ref[r]||rB[r]!=ref[r]) ok=false;
            OKf[phase][ki]=ok;
            printf("[meas] %s K=%d : gA=%lld gC=%lld gB=%lld | msA=%lld msC=%lld msB=%lld | matWords=%d nXmacc=%lld %s\n",
                   WARM?"WARM":"COLD",K,G[phase][ki][0],G[phase][ki][1],G[phase][ki][2],msA,msC,msB,matWords,NX[ki], ok?"OK":"BIT-FAIL");
            if(!ok){ for(int r=0;r<3;++r) printf("   r=%d ref=%lld A=%lld C=%lld B=%lld\n",r,ref[r],rA[r],rC[r],rB[r]); }
            fflush(stdout);
        }
    }

    auto memgas=[](long long ms)->long long{ long long w=ms/32; return 3*w + (w*w)/512; };
    const char* pn[3]={"A","C","B"};
    int hard_fail=0;
    printf("\n========== CLEAN 3-BUCKET SPLIT + GAS SWEEP ==========\n");
    for(int ki=0; ki<NK; ++ki){
        int K=Ks[ki]; long long matWords=MW[ki];
        if(!OKf[0][ki] || !OKf[1][ki]){ printf("K=%d: BIT-EXACT FAIL -> SKIPPED\n",K); hard_fail=1; continue; }

        long long sload_cold=matWords*2100, sload_warm=matWords*100;
        bool g3=true;
        for(int p=0;p<3;++p){ long long d=G[0][ki][p]-G[1][ki][p]; if(d!=matWords*2000){ printf("K=%d path %s: cold-warm=%lld != matWords*2000=%lld\n",K,pn[p],d,matWords*2000); g3=false; } }
        if(!g3){ hard_fail=1; continue; }
        for(int p=0;p<3;++p) if(MSv[0][ki][p]!=MSv[1][ki][p]){ printf("K=%d path %s: MSIZE cold!=warm\n",K,pn[p]); hard_fail=1; }

        long long memv[3], hcold[3], hwarm[3];
        for(int p=0;p<3;++p){
            memv[p]=memgas(MSv[0][ki][p]);
            hcold[p]=G[0][ki][p]-sload_cold-memv[p];
            hwarm[p]=G[1][ki][p]-sload_warm-memgas(MSv[1][ki][p]);
        }
        for(int p=0;p<3;++p){
            if(hcold[p]<=0){ printf("K=%d path %s: compute_cold<=0\n",K,pn[p]); hard_fail=1; }
            if(hcold[p]!=hwarm[p]){ printf("K=%d path %s: compute_cold!=compute_warm\n",K,pn[p]); hard_fail=1; }
        }

        long long hesapC=hcold[1], hesapB14=hcold[2], N=NX[ki];
        double cb14=(double)hesapC/(double)hesapB14;
        printf("\nK=%d (matWords=%lld, nXmacc=%lld)\n",K,matWords,N);
        printf("  MEASURED (XMACC_S=14): computeC=%lld computeB=%lld -> pure C/B=%.2fx\n",hesapC,hesapB14,cb14);
        if(memv[1]==memv[2]) printf("  [memory C==B -> ratio is pure compute]\n");
        // --- DERIVATION (NOT A MEASUREMENT): constant gas per opcode. computeB(g)=computeB14+(g-14)*N ---
        printf("  --- DERIVATION (not a measurement; to be cross-checked at g=84 with a real measurement) ---\n");
        printf("  g(XMACC_S) | computeB(g)_derived |  C/B(g)_derived\n");
        printf("  -----------+---------------------+------------------\n");
        int gs[]={14,26,28,32,42,56,70,84};   // 26 and 84: measured by rebuilding the interpreter (run_all.sh RUN_PRICE=1); 32: upper end of the time-neutral range
        for(int gi=0; gi<(int)(sizeof(gs)/sizeof(gs[0])); ++gi){ int g=gs[gi]; long long hb=hesapB14+(long long)(g-14)*N; double cb=(double)hesapC/(double)hb;
            printf("  %10d | %19lld | %14.2fx%s\n",g,hb,cb, g==14?"  <- MEASURED":""); }
    }
    if(hard_fail) printf("\n*** CHECK FAILED -> UNRELIABLE ***\n");
    else          printf("\n*** ALL CHECKS PASSED. g=14 MEASURED; g>14 DERIVED (to be verified at 84) ***\n");

    VM->destroy(VM); dlclose(h); return 0;
}
