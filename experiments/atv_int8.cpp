// int8 storage + in-EVM unpack vs int16 storage | Path B (zero-copy+SIMD)
// int8: SLOAD halves but the unpack adds compute cost. 3 buckets: SLOAD | compute | memory.
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

    printf("=== int8 storage + in-EVM unpack vs int16 | Path B (zero-copy+SIMD) ===\n");
    printf("Same compute (XTRANSPOSE+XMACC_S). int8: half the SLOADs + unpack. Result must equal int16.\n\n");

    for(int K : {256,1024,4096}){
        std::vector<int> M(K*K), v(K);
        for(int i=0;i<K*K;++i) M[i]=(i%50)+1;
        for(int c=0;c<K;++c) v[c]=(c%7)+1;
        std::vector<long long> ref(K);
        for(int r=0;r<K;++r){ long long s=0; for(int c=0;c<K;++c) s+=(long long)M[c*K+r]*v[c]; ref[r]=(long long)(uint32_t)s; }

        int matWords16=(K*K+15)/16, matWords8=(K*K+31)/32, vecWords=(K+7)/8, MAT=0;
        int VEC=(1+matWords16)*32+256, ACC=VEC+vecWords*32+64, OUTV=ACC+64;
        long long peakByte=(long long)OUTV+(long long)K*32; long long mtop=32+2LL*(K*K-1)+1; if(mtop>peakByte)peakByte=mtop;
        long long aPeak=(peakByte+31)/32, bellek=3*aPeak+(aPeak*aPeak)/512;

        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS; msg.recipient=ADDR; msg.code_address=ADDR;

        auto emitHeader=[&](Code& k){ std::vector<uint8_t> hb; u64le(hb,K);u64le(hb,K);u64le(hb,K);u64le(hb,1); k.W32(hb); k.MS(MAT); };
        auto emitCompute=[&](Code& k){
            for(int blk=0; blk<vecWords; ++blk){ std::vector<uint8_t> d(32,0); for(int e=0;e<8;++e){ int idx=blk*8+e; if(idx<K){ uint16_t val=(uint16_t)v[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } } k.W32(d); k.MS(VEC+blk*32); }
            k.PUSH(MAT); k.b(0xDD);
            for(int r=0;r<K;++r){ { std::vector<uint8_t> da(32,0); k.W32(da); k.MS(ACC); }
                for(int blk=0; blk<vecWords; ++blk){ k.PUSH(blk*8); k.PUSH(r); k.PUSH(ACC); k.PUSH(VEC+blk*32); k.PUSH(MAT); k.b(0xE5); }
                k.ML(ACC); k.PUSH(OUTV+r*32); k.b(0x52); }
            k.PUSH(K*32); k.PUSH(OUTV); k.b(0xF3);
        };

        auto measure=[&](int fmt, long long& tc, long long& tw, bool& ok){
            evmc::MockedHost host;
            int matWords=(fmt==16)?matWords16:matWords8;
            for(int w=0; w<matWords; ++w){ evmc::bytes32 key{}, val{};
                key.bytes[28]=(w>>24)&0xFF;key.bytes[29]=(w>>16)&0xFF;key.bytes[30]=(w>>8)&0xFF;key.bytes[31]=w&0xFF;
                if(fmt==16){ for(int e=0;e<16;++e){ int idx=w*16+e; if(idx<K*K){ uint16_t u=(uint16_t)M[idx]; val.bytes[e*2]=u&0xFF; val.bytes[e*2+1]=(u>>8)&0xFF; } } }
                else { for(int p=0;p<32;++p){ int idx=w*32+p; if(idx<K*K) val.bytes[p]=(uint8_t)M[idx]; } }
                host.accounts[ADDR].storage[key]=evmc::StorageValue{val}; }
            auto setAcc=[&](evmc_access_status st){ for(int w=0;w<matWords;++w){ evmc::bytes32 key{}; key.bytes[28]=(w>>24)&0xFF;key.bytes[29]=(w>>16)&0xFF;key.bytes[30]=(w>>8)&0xFF;key.bytes[31]=w&0xFF; host.accounts[ADDR].storage[key].access_status=st; } };

            Code k; emitHeader(k);
            if(fmt==16){ for(int w=0; w<matWords16; ++w){ k.PUSHslot((uint32_t)w); k.b(0x54); k.MS(MAT+32+w*32); } }
            else { for(int w8=0; w8<matWords8; ++w8){ k.PUSHslot((uint32_t)w8); k.b(0x54);
                    for(int p=0;p<32;++p){ int idx=w8*32+p; if(idx>=K*K) break; int w16=idx/16,e=idx%16,addr=MAT+32+w16*32+e*2;
                        k.b(0x80); k.P1((uint8_t)p); k.b(0x1a); k.PUSH(addr); k.b(0x53); }   // DUP1,PUSH p,BYTE,PUSH addr,MSTORE8
                    k.b(0x50); } }                                                            // POP word
            emitCompute(k);

            std::vector<long long> got(K,-1);
            setAcc(EVMC_ACCESS_COLD);
            auto rc=VM->execute(VM,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
            tc=GAS-rc.gas_left; int sc=rc.status_code;
            if(sc==0&&rc.output_data&&(int)rc.output_size>=K*32) for(int r=0;r<K;++r){ long long s=0; for(int i=0;i<8;++i){ uint32_t x=0; for(int kk=0;kk<4;++kk) x|=((uint32_t)rc.output_data[r*32+i*4+kk])<<(8*kk); s+=(int32_t)x; } got[r]=(long long)(uint32_t)s; }
            if(rc.release)rc.release(&rc);
            ok=(sc==0); for(int r=0;r<K&&ok;++r) if(got[r]!=ref[r]) ok=false;
            setAcc(EVMC_ACCESS_WARM);
            auto rw=VM->execute(VM,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
            tw=GAS-rw.gas_left; if(rw.release)rw.release(&rw);
        };

        long long t16c,t16w,t8c,t8w; bool ok16,ok8;
        measure(16,t16c,t16w,ok16); measure(8,t8c,t8w,ok8);
        auto show=[&](const char* nm,int fmt,long long tc,long long tw,bool ok){
            int mw=(fmt==16)?matWords16:matWords8;
            printf("K=%d %s exact?=%s\n",K,nm,ok?"YES":"NO");
            printf("   COLD: SLOAD=%lld compute=%lld memory=%lld TOTAL=%lld\n",(long long)mw*2100, tc-(long long)mw*2100-bellek, bellek, tc);
            printf("   WARM: SLOAD=%lld compute=%lld memory=%lld TOTAL=%lld\n",(long long)mw*100, tw-(long long)mw*100-bellek, bellek, tw);
        };
        show("int16",16,t16c,t16w,ok16); show("int8 ",8,t8c,t8w,ok8);
        printf("   >>> int8/int16 TOTAL:  COLD %.3f | WARM %.3f  (<1 = int8 wins)\n",(double)t8c/t16c,(double)t8w/t16w);
        printf("------------------------------------------------------------\n");
    }
    VM->destroy(VM); dlclose(h); return 0;
}
