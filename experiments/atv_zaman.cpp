// atv_zaman.cpp — WALL-CLOCK MEASUREMENT (answer to the "where is the time?" objection)
// Path C (fair scalar) and Path B (zero-copy+SIMD), IDENTICAL to the atv_kapaliform emitters.
// Question: does the time ratio C/B track the gas ratio? (gas = proxy for per-node compute argument)
// Method: steady_clock only around vm->execute; adaptive inner repetition (sample >= 10ms),
//   7..31 samples (budget 90s/cell, min 3); MEDIAN + min-max + CV% are reported.
// CHECKS: (1) bit-exact  (2) measured gas == atv_kapaliform 2026-07-06 measurements (20 values, identical)
//   (3) gas is identical across repeated runs.
// NOTE: runs under VirtualBox; CV% is visible in the output, the paper notes the environment.
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <cmath>

static evmc_vm* VM;
static const int64_t GAS=200000000000LL;
static void u64le(std::vector<uint8_t>& b, uint64_t v){ for(int i=0;i<8;++i) b.push_back((v>>(8*i))&0xFF); }

struct Code {
    std::vector<uint8_t> c;
    void b(uint8_t x){ c.push_back(x); }
    void P1(uint8_t v){ b(0x60); b(v); }
    void P2(int v){ b(0x61); b((v>>8)&0xFF); b(v&0xFF); }
    void P3(int v){ b(0x62); b((v>>16)&0xFF); b((v>>8)&0xFF); b(v&0xFF); }
    void P4(unsigned v){ b(0x63); b((v>>24)&0xFF); b((v>>16)&0xFF); b((v>>8)&0xFF); b(v&0xFF); }
    void PUSH(long long v){ if(v<256)P1((uint8_t)v); else if(v<65536)P2((int)v); else if(v<16777216)P3((int)v); else P4((unsigned)v); }
    void W32(std::vector<uint8_t> bytes){ while(bytes.size()<32)bytes.push_back(0); b(0x7F); for(int i=0;i<32;++i) b(bytes[i]); }
    void MS(long long a){ PUSH(a); b(0x52); }
    void ML(long long a){ PUSH(a); b(0x51); }
};

// atv_kapaliform 2026-07-06 MEASURED total gas (cross-run anchor)
static const long long EXPC[10]={5211,20442,81158,323713,1294754,5202238,21213475,91278744,466929971,3487714088LL};
static const long long EXPB[10]={560,1607,5395,19854,77743,331083,1722672,13303205,155003200,2239957813LL};

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ std::cerr<<"libevmone not found\n"; return 1; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone"); VM=create();
    evmc::MockedHost host;
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();
    evmc::address ADDR{}; ADDR.bytes[19]=0x01;

    std::vector<int> Ks={8,16,32,64,128,256,512,1024,2048,4096};
    const int NK=(int)Ks.size();
    double medC[10],medB[10],mnC[10],mxC[10],mnB[10],mxB[10],cvC[10],cvB[10];
    long long GC[10],GB[10];
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

        auto emitMatMem=[&](Code& k){
            std::vector<uint8_t> hb; u64le(hb,K);u64le(hb,K);u64le(hb,K);u64le(hb,1); k.W32(hb); k.MS(MAT);
            for(int w=0; w<matWords; ++w){ std::vector<uint8_t> d(32,0);
                for(int e=0;e<16;++e){ int idx=w*16+e; if(idx<K*K){ uint16_t val=(uint16_t)M[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
                k.W32(d); k.MS(MAT+32+w*32); } };
        auto emitVec=[&](Code& k){
            for(int blk=0; blk<vecWords; ++blk){ std::vector<uint8_t> d(32,0);
                for(int e=0;e<8;++e){ int idx=blk*8+e; if(idx<K){ uint16_t val=(uint16_t)v[idx]; d[e*2]=val&0xFF; d[e*2+1]=(val>>8)&0xFF; } }
                k.W32(d); k.MS(VEC+blk*32); } };
        auto emitTail=[&](Code& k){
            k.b(0x59); k.PUSH(0); k.b(0x52);
            k.PUSH(OUTV+(long long)K*32); k.PUSH(0); k.b(0xF3); };

        Code kC; emitMatMem(kC); emitVec(kC);
        for(int r=0;r<K;++r){ kC.P1(0);
            for(int c=0;c<K;++c){
                long long eB=MAT+32+((long long)c*K+r)*2;
                kC.PUSH((eB/32)*32); kC.b(0x51); kC.P1((uint8_t)(eB%32)); kC.b(0x1a);
                kC.PUSH(((eB+1)/32)*32); kC.b(0x51); kC.P1((uint8_t)((eB+1)%32)); kC.b(0x1a);
                kC.P2(256); kC.b(0x02); kC.b(0x01);
                long long vW=VEC+(c/8)*32; int vlo=(c%8)*2;
                kC.PUSH(vW); kC.b(0x51); kC.P1((uint8_t)vlo); kC.b(0x1a);
                kC.PUSH(vW); kC.b(0x51); kC.P1((uint8_t)(vlo+1)); kC.b(0x1a);
                kC.P2(256); kC.b(0x02); kC.b(0x01); kC.b(0x02); kC.b(0x01); }
            kC.MS(OUTV+(long long)r*32); }
        emitTail(kC);

        Code kB; emitMatMem(kB); emitVec(kB);
        kB.PUSH(MAT); kB.b(0xDD);
        for(int r=0;r<K;++r){ { std::vector<uint8_t> da(32,0); kB.W32(da); kB.MS(ACC); }
            for(int blk=0; blk<vecWords; ++blk){ kB.PUSH(blk*8); kB.PUSH(r); kB.PUSH(ACC); kB.PUSH(VEC+blk*32); kB.PUSH(MAT); kB.b(0xE5); }
            kB.ML(ACC); kB.MS(OUTV+(long long)r*32); }
        emitTail(kB);

        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS; msg.recipient=ADDR; msg.code_address=ADDR;
        auto runOnce=[&](Code& k, bool check, int simd)->long long{
            auto res=VM->execute(VM,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.c.data(),k.c.size());
            long long g=GAS-res.gas_left;
            if(check){
                bool ok = res.status_code==0 && res.output_data && (long long)res.output_size>=OUTV+(long long)K*32;
                if(ok){ for(int r=0;r<K && ok;++r){
                    long long got;
                    if(simd){ long long s=0; for(int i=0;i<8;++i){ uint32_t x=0; for(int kk=0;kk<4;++kk) x|=((uint32_t)res.output_data[OUTV+(long long)r*32+i*4+kk])<<(8*kk); s+=(int32_t)x; } got=(long long)(uint32_t)s; }
                    else { long long x=0; for(int j=28;j<32;++j) x=(x<<8)|res.output_data[OUTV+(long long)r*32+j]; got=x; }
                    if(got!=ref[r]) ok=false; } }
                if(!ok){ std::printf("K=%d simd=%d BIT-EXACT FAIL\n",K,simd); fail=1; }
            }
            if(res.release)res.release(&res); return g;
        };
        auto timePath=[&](Code& k,int simd,long long expg,double& med,double& mn,double& mx,double& cv,long long& gout){
            long long g1=runOnce(k,true,simd);                  // warm-up + bit-exact + gas
            gout=g1;
            if(g1!=expg){ std::printf("K=%d %s: gas=%lld != closed-form=%lld [CHECK 2 FAILED]\n",K,simd?"B":"C",g1,expg); fail=1; }
            using clk=std::chrono::steady_clock;
            auto t0=clk::now(); long long g2=runOnce(k,false,simd); auto t1=clk::now();
            if(g2!=g1){ std::printf("K=%d: repeated-run gas differs [CHECK 3 FAILED]\n",K); fail=1; }
            double est=std::chrono::duration<double>(t1-t0).count(); if(est<=0) est=1e-7;
            long long R=(long long)std::ceil(0.010/est); if(R<1)R=1; if(R>20000)R=20000;
            std::vector<double> S; double tot=0;
            while(true){
                auto a=clk::now();
                for(long long i=0;i<R;++i) runOnce(k,false,simd);
                auto b=clk::now();
                double dt=std::chrono::duration<double>(b-a).count();
                S.push_back(dt/R); tot+=dt;
                size_t n=S.size();
                if(n>=3 && ((n>=7 && tot>=1.0) || n>=31 || tot>=90.0)) break;
            }
            std::sort(S.begin(),S.end());
            med=S[S.size()/2]; mn=S.front(); mx=S.back();
            double mu=0; for(double x:S)mu+=x; mu/=S.size();
            double sd=0; for(double x:S)sd+=(x-mu)*(x-mu); sd=std::sqrt(sd/S.size());
            cv=100.0*sd/mu;
            std::printf("  [K=%4d %s] med=%.6f s  min=%.6f  max=%.6f  CV=%.1f%%  (samples=%zu, inner-reps=%lld)\n",
                K,simd?"B":"C",med,mn,mx,cv,S.size(),R);
            std::fflush(stdout);
        };
        std::printf("[K=%4d] bytecode C=%zu B=%zu bytes, measuring...\n",K,kC.c.size(),kB.c.size()); std::fflush(stdout);
        timePath(kC,0,EXPC[ki],medC[ki],mnC[ki],mxC[ki],cvC[ki],GC[ki]);
        timePath(kB,1,EXPB[ki],medB[ki],mnB[ki],mxB[ki],cvB[ki],GB[ki]);
    }

    std::printf("\n================ WALL-CLOCK vs GAS (median; VirtualBox environment) ================\n");
    std::printf("   K   |   tC (ms)   |   tB (ms)   | TIME C/B  |  GAS C/B  | MGas/s C | MGas/s B | CV_C%% | CV_B%% |   g*\n");
    std::printf("-------+-------------+-------------+-----------+-----------+----------+----------+-------+-------+------\n");
    for(int ki=0; ki<NK; ++ki){
        double tr=medC[ki]/medB[ki], gr=(double)GC[ki]/(double)GB[ki];
        // g* = time-neutral XMACC_S price: the price at which the end-to-end gas ratio equals the time ratio,
        // GC / (GB + (g*-14)*K^2/8) == TIME C/B  (only path B contains XMACC_S; K^2/8 instructions per run)
        double nx=(double)Ks[ki]*(double)Ks[ki]/8.0;
        double gstar=14.0+((double)GC[ki]/tr-(double)GB[ki])/nx;
        std::printf(" %5d | %11.4f | %11.4f | %9.2f | %9.2f | %8.1f | %8.1f | %5.1f | %5.1f | %5.1f\n",
            Ks[ki], medC[ki]*1e3, medB[ki]*1e3, tr, gr,
            (double)GC[ki]/medC[ki]/1e6, (double)GB[ki]/medB[ki]/1e6, cvC[ki], cvB[ki], gstar);
    }
    std::printf("\nReading: if TIME C/B and GAS C/B are of the same order, gas is a valid proxy for per-node compute time.\n");
    std::printf("The MGas/s columns give the real speed per gas of each path (SIMD opcodes do 'denser' work).\n");
    std::printf("g* = time-neutral XMACC_S price per K: 14 + (GC/(TIME C/B) - GB) / (K^2/8); the deployed price is 14.\n");
    if(fail) std::printf("\n*** AT LEAST ONE CHECK FAILED -> timings cannot be reported ***\n");
    else     std::printf("\n*** ALL CHECKS PASSED (bit-exact + gas==closed-form 20/20 + repeat stable) ***\n");
    VM->destroy(VM); dlclose(h); return fail;
}
