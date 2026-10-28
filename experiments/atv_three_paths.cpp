// FULL A^T v COMPARISON: K=8..128. Real, complete, no shortcuts.
#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>   // a valid (empty) host interface; no host call is made by this program
#include <dlfcn.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdint>
static evmc::MockedHost g_host;

static evmc_vm* VM;
static const int64_t GAS=20000000000LL;
static void u64le(std::vector<uint8_t>& b, uint64_t v){ for(int i=0;i<8;++i) b.push_back((v>>(8*i))&0xFF); }

struct Code {
    std::vector<uint8_t> c;
    void b(uint8_t x){ c.push_back(x); }
    void P1(uint8_t v){ b(0x60); b(v); }
    void P2(int v){ b(0x61); b((v>>8)&0xFF); b(v&0xFF); }
    void P3(int v){ b(0x62); b((v>>16)&0xFF); b((v>>8)&0xFF); b(v&0xFF); }
    void P4(unsigned v){ b(0x63); b((v>>24)&0xFF); b((v>>16)&0xFF); b((v>>8)&0xFF); b(v&0xFF); }
    void PUSH(int v){ if(v<256)P1((uint8_t)v); else if(v<65536)P2(v); else if(v<16777216)P3(v); else P4((unsigned)v); }
    void W32(std::vector<uint8_t> bytes){ while(bytes.size()<32)bytes.push_back(0); b(0x7F); for(int i=0;i<32;++i) b(bytes[i]); }
    void MS(int a){ PUSH(a); b(0x52); }
    void ML(int a){ PUSH(a); b(0x51); }
};

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ std::cerr<<"libevmone not found\n"; return 1; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone"); VM=create();

    std::vector<int> Ks={256,1024,4096};
    std::cout<<"=== A^T v 3 PATHS (FAIR): A=naive(2 buffers) | C=fair scalar(1 buffer,strided) | B=zero-copy+SIMD | C/B = SIMD gain ===\n";
    std::cout<<"The whole matrix is processed. K-element result, bit-exact + end-to-end gas.\n";
    std::cout<<"  K  | PathA(naive) | PathC(fair scalar) | PathB(SIMD) |  C/B(FAIR) |  A/B  | exact?\n";
    std::cout<<"-----+--------------+--------------------+-------------+------------+-------+-------\n";

    for(int K : Ks){
        std::vector<int> M(K*K), v(K);
        for(int i=0;i<K*K;++i) M[i]=(i%50)+1;
        for(int c=0;c<K;++c) v[c]=(c%7)+1;
        std::vector<long long> ref(K);
        for(int r=0;r<K;++r){ long long s=0; for(int c=0;c<K;++c) s+=(long long)M[c*K+r]*v[c]; ref[r]=(long long)(uint32_t)s; }

        int matWords=(K*K+15)/16;
        int vecWords=(K+7)/8;
        int MAT=0;
        int TMP=(1+matWords)*32;
        int VEC=(1+matWords+1+matWords)*32+256;
        int ACC=VEC+vecWords*32+64;
        int OUTV=ACC+64;

        auto writeMat=[&](Code& k){
            std::vector<uint8_t> hb; u64le(hb,K);u64le(hb,K);u64le(hb,K);u64le(hb,1); k.W32(hb); k.MS(MAT);
            for(int w=0; w<matWords; ++w){ std::vector<uint8_t> d(32,0); for(int e=0;e<16;++e){ int idx=w*16+e; if(idx<K*K){ uint16_t val=(uint16_t)M[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } } k.W32(d); k.MS(MAT+32+w*32); }
        };
        auto writeVecBlocks=[&](Code& k){
            for(int blk=0; blk<vecWords; ++blk){
                std::vector<uint8_t> d(32,0);
                for(int e=0;e<8;++e){ int idx=blk*8+e; if(idx<K){ uint16_t val=(uint16_t)v[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
                k.W32(d); k.MS(VEC+blk*32);
            }
        };

        long long gA; std::vector<long long> resA(K,-1); int stA=0;
        {
            Code k; writeMat(k);
            { std::vector<uint8_t> hb; u64le(hb,K);u64le(hb,K);u64le(hb,K);u64le(hb,1); k.W32(hb); k.MS(TMP); }
            for(int r=0;r<K;++r) for(int c=0;c<K;++c){
                int srcB=MAT+32+(c*K+r)*2, dstB=TMP+32+(r*K+c)*2;
                k.PUSH((srcB/32)*32); k.b(0x51); k.P1(srcB%32); k.b(0x1a); k.PUSH(dstB); k.b(0x53);
                k.PUSH(((srcB+1)/32)*32); k.b(0x51); k.P1((srcB+1)%32); k.b(0x1a); k.PUSH(dstB+1); k.b(0x53);
            }
            for(int r=0;r<K;++r){
                k.P1(0);
                for(int c=0;c<K;++c){ int elemB=TMP+32+(r*K+c)*2;
                    k.PUSH((elemB/32)*32); k.b(0x51); k.P1(elemB%32); k.b(0x1a);
                    k.PUSH(((elemB+1)/32)*32); k.b(0x51); k.P1((elemB+1)%32); k.b(0x1a);
                    k.P2(256); k.b(0x02); k.b(0x01); k.PUSH(v[c]); k.b(0x02); k.b(0x01);
                }
                k.PUSH(OUTV+r*32); k.b(0x52);
            }
            k.PUSH(K*32); k.PUSH(OUTV); k.b(0xF3);
            evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS;
            auto res=VM->execute(VM, &g_host.get_interface(), g_host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
            gA=GAS-res.gas_left; stA=res.status_code;
            if(res.status_code==0&&res.output_data&&(int)res.output_size>=K*32)
                for(int r=0;r<K;++r){ long long x=0; for(int j=28;j<32;++j) x=(x<<8)|res.output_data[r*32+j]; resA[r]=x; }
            if(res.release)res.release(&res);
        }

        // ----- PATH C: FAIR scalar (strided reads, NO transpose). SAME memory plan AS B. -----
        long long gC; std::vector<long long> resC(K,-1); int stC=0;
        {
            int VEC=(1+matWords)*32+256;   // IDENTICAL offsets to B
            int ACC=VEC+vecWords*32+64;
            int OUTV=ACC+64;
            Code k; writeMat(k);
            for(int blk=0; blk<vecWords; ++blk){            // write the vector to memory the SAME way as B
                std::vector<uint8_t> d(32,0);
                for(int e=0;e<8;++e){ int idx=blk*8+e; if(idx<K){ uint16_t val=(uint16_t)v[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
                k.W32(d); k.MS(VEC+blk*32);
            }
            for(int r=0;r<K;++r){
                k.P1(0);                                     // acc=0
                for(int c=0;c<K;++c){
                    // --- A[c*K+r]: STRIDED read, 16-bit (transpose is not materialized) ---
                    int elemB=MAT+32+(c*K+r)*2;
                    k.PUSH((elemB/32)*32); k.b(0x51); k.P1(elemB%32); k.b(0x1a);          // low byte
                    k.PUSH(((elemB+1)/32)*32); k.b(0x51); k.P1((elemB+1)%32); k.b(0x1a);  // high byte (aligned)
                    k.P2(256); k.b(0x02); k.b(0x01);                                       // combine with <<8
                    // --- v[c]: FROM MEMORY, two bytes of the same word (aligned) ---
                    int vWord=VEC+(c/8)*32, vlo=(c%8)*2;
                    k.PUSH(vWord); k.b(0x51); k.P1(vlo); k.b(0x1a);                         // v low byte
                    k.PUSH(vWord); k.b(0x51); k.P1(vlo+1); k.b(0x1a);                       // v high byte (SAME word, aligned)
                    k.P2(256); k.b(0x02); k.b(0x01);                                        // combine with <<8
                    k.b(0x02); k.b(0x01);                                                   // element*v[c] + acc
                }
                k.PUSH(OUTV+r*32); k.b(0x52);
            }
            k.PUSH(K*32); k.PUSH(OUTV); k.b(0xF3);
            evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS;
            auto res=VM->execute(VM, &g_host.get_interface(), g_host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
            gC=GAS-res.gas_left; stC=res.status_code;
            if(res.status_code==0&&res.output_data&&(int)res.output_size>=K*32)
                for(int r=0;r<K;++r){ long long x=0; for(int j=28;j<32;++j) x=(x<<8)|res.output_data[r*32+j]; resC[r]=x; }
            if(res.release)res.release(&res);
        }

        long long gB; std::vector<long long> resB(K,-1); int stB=0;
        {
            // FAIR PLAN: B uses its own real memory; NO unused TMP gap.
            int VEC=(1+matWords)*32+256;     // place VEC/ACC/OUTV right after MAT (local shadow)
            int ACC=VEC+vecWords*32+64;
            int OUTV=ACC+64;
            Code k; writeMat(k);
            for(int blk=0; blk<vecWords; ++blk){              // write the vector at the FAIR offset
                std::vector<uint8_t> d(32,0);
                for(int e=0;e<8;++e){ int idx=blk*8+e; if(idx<K){ uint16_t val=(uint16_t)v[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
                k.W32(d); k.MS(VEC+blk*32);
            }
            k.PUSH(MAT); k.b(0xDD);
            for(int r=0;r<K;++r){
                { std::vector<uint8_t> da(32,0); k.W32(da); k.MS(ACC); }
                for(int blk=0; blk<vecWords; ++blk){
                    k.PUSH(blk*8);
                    k.PUSH(r);
                    k.PUSH(ACC);
                    k.PUSH(VEC+blk*32);
                    k.PUSH(MAT);
                    k.b(0xE5);
                }
                k.ML(ACC); k.PUSH(OUTV+r*32); k.b(0x52);
            }
            k.PUSH(K*32); k.PUSH(OUTV); k.b(0xF3);
            evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS;
            auto res=VM->execute(VM, &g_host.get_interface(), g_host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
            gB=GAS-res.gas_left; stB=res.status_code;
            if(res.status_code==0&&res.output_data&&(int)res.output_size>=K*32)
                for(int r=0;r<K;++r){ long long s=0; for(int i=0;i<8;++i){ uint32_t x=0; for(int kk=0;kk<4;++kk) x|=((uint32_t)res.output_data[r*32+i*4+kk])<<(8*kk); s+=(int32_t)x; } resB[r]=(long long)(uint32_t)s; }
            if(res.release)res.release(&res);
        }

        bool ok=true; for(int r=0;r<K;++r) if(resA[r]!=ref[r]||resB[r]!=ref[r]||resC[r]!=ref[r]) ok=false;
        if(stA!=0||stB!=0||stC!=0){ printf("%4d | ERROR stA=%d stC=%d stB=%d\n",K,stA,stC,stB); continue; }
        printf("%4d | %12lld | %18lld | %11lld | %9.2fx | %.2fx | %s\n", K, gA, gC, gB, (double)gC/gB, (double)gA/gB, ok?"YES":"NO");
        if(!ok){ for(int r=0;r<3;++r) printf("      r=%d ref=%lld A=%lld B=%lld\n",r,ref[r],resA[r],resB[r]); }
    }
    VM->destroy(VM); dlclose(h); return 0;
}
