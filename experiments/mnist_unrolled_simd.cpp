// mnist_unrolled_simd.cpp — UNROLLED SIMD: turns the claim "the gap between 4.92x and the kernel
// ratio is loop overhead" into a measurement. The unrolled body is STRAIGHT-LINE (zero jumps), keeps the
// accumulator on the stack (symmetric with the scalar baseline); argmax is branchless (deterministic
// gas) -> a CLOSED-FORM prediction can be written, check: predicted==measured DIFF 0.
// NOTE (revised expectation): the MNIST SIMD kernel is XMACC (pre-transposed, gas 11),
// the A^Tv kernel is XMACC_S (gather, gas 14) — the ratio MAY EXCEED 18.4; let the measurement decide.
// CHECKS: (1) 20 images three-way bit-exact (2) unrolled warm predicted==measured diff 0
// (including MSIZE) (3) cold-warm==matWords*2000 x2 (4) compute_c==compute_w x2
// (5) looped warm == 5,134,719 anchor (6) repeat run identical.
#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <algorithm>

static void put16(std::vector<uint8_t>& w,int lane,int val){
    uint16_t u=(uint16_t)(int16_t)val; w[lane*2]=u&0xFF; w[lane*2+1]=(u>>8)&0xFF;
}
static void put32(std::vector<uint8_t>& w,int lane,long long val){
    uint32_t u=(uint32_t)(int32_t)val; for(int b=0;b<4;++b) w[lane*4+b]=(u>>(8*b))&0xFF;
}
static long long opGas(uint8_t op){
    if(op>=0x60&&op<=0x7F) return 3;                    // PUSH1..32
    if(op>=0x80&&op<=0x8F) return 3;                    // DUPx
    if(op>=0x90&&op<=0x9F) return 3;                    // SWAPx
    switch(op){
        case 0x01: case 0x03: return 3;                 // ADD SUB
        case 0x02: return 5;                            // MUL
        case 0x0B: return 5;                            // SIGNEXTEND
        case 0x13: case 0x15: case 0x16: case 0x17: return 3; // SGT ISZERO AND OR
        case 0x1B: case 0x1C: return 3;                 // SHL SHR
        case 0x50: return 2;                            // POP
        case 0x51: case 0x52: case 0x53: return 3;      // MLOAD MSTORE MSTORE8
        case 0x54: return 100;                          // SLOAD (WARM; prediction is for the warm run)
        case 0x59: return 2;                            // MSIZE
        case 0xF3: return 0;                            // RETURN
        case 0xCB: return 11;                           // XMACC
        case 0xCC: case 0xCD: case 0xCE: return 6;      // XRELU XSUM32 XPACK
        default: std::fprintf(stderr,"opGas: unknown %02x\n",op); std::exit(2);
    }
}
struct Code {
    std::vector<uint8_t> c; long long g=0; long long peakB=0; bool say=true;
    void mem(long long e){ if(e>peakB) peakB=e; }
    void op(uint8_t x){ c.push_back(x); if(say) g+=opGas(x); }
    void raw(uint8_t x){ c.push_back(x); }
    void P1(uint8_t v){ op(0x60); raw(v); }
    void P2(int v){ op(0x61); raw((v>>8)&0xFF); raw(v&0xFF); }
    void PUSH(long long v){ if(v<256)P1((uint8_t)v); else if(v<65536)P2((int)v);
        else { op(0x62); raw((v>>16)&0xFF); raw((v>>8)&0xFF); raw(v&0xFF); } }
    void W32(std::vector<uint8_t> b){ while(b.size()<32)b.push_back(0); op(0x7F); for(int i=0;i<32;++i) raw(b[i]); }
};
static long long memgas(long long bytes){ long long a=(bytes+31)/32; return 3*a+(a*a)/512; }

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ std::cerr<<"libevmone not found\n"; return 1; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone");
    evmc_vm* vm=create();

    int S,SHIFT1,SHIFT2,H1,H2;
    { std::ifstream f("deep_meta.txt"); f>>S>>SHIFT1>>SHIFT2>>H1>>H2; }
    const int I=784, CDIMG=I/8*32;
    std::vector<int> W1(H1*I), W2(H2*H1), W3(H2*10);
    { std::ifstream f("deep_W1.txt"); for(auto&x:W1) f>>x; }
    { std::ifstream f("deep_W2.txt"); for(auto&x:W2) f>>x; }
    { std::ifstream f("deep_W3.txt"); for(auto&x:W3) f>>x; }
    std::vector<long long> b1(H1), b2(H2);
    { std::ifstream f("deep_b1.txt"); for(auto&x:b1) f>>x; }
    { std::ifstream f("deep_b2.txt"); for(auto&x:b2) f>>x; }
    std::ifstream bf("mnist_batch.txt"); int Nb,wsc; bf>>Nb>>wsc;
    int N_RUN=500;
    std::vector<std::vector<int>> imgs(N_RUN, std::vector<int>(I));
    for(int im=0; im<N_RUN; ++im){
        int tlb,pfo; bf>>tlb>>pfo; double tmp; for(int i=0;i<10;++i) bf>>tmp;
        for(int i=0;i<I;++i) bf>>imgs[im][i];
    }
    auto rss=[&](long long v,int sh)->int{ if(v<0)v=0; v>>=sh; if(v>32767)v=32767; return (int)v; };
    auto cppPred=[&](int im)->int{
        std::vector<int>& g=imgs[im];
        std::vector<int> h1r(H1); for(int n=0;n<H1;++n){ long long s=b1[n]; for(int i=0;i<I;++i) s+=(long long)g[i]*W1[i*H1+n]; h1r[n]=rss(s,SHIFT1);}
        std::vector<int> h2r(H2); for(int n=0;n<H2;++n){ long long s=b2[n]; for(int i=0;i<H1;++i) s+=(long long)h1r[i]*W2[i*H2+n]; h2r[n]=rss(s,SHIFT2);}
        std::vector<long long> sc(10); for(int c=0;c<10;++c){ long long s=0; for(int i=0;i<H2;++i) s+=(long long)h2r[i]*W3[i*10+c]; sc[c]=s;}
        return (int)(std::max_element(sc.begin(),sc.end())-sc.begin());
    };

    int INP=0, W1M=10000, B1M=420000, H1OUT=430000, W2M=440000, B2M=480000, H2OUT=490000;
    int W3M=495000, SCO=500000;
    int ACC=900000, SACC=900100, NCNT=901000, GCNT=901100;
    int MAXV=900200, MAXI=900232, CNT2=900264;

    std::vector<std::pair<std::vector<uint8_t>,int>> WLOAD;
    for(int n=0;n<H1;++n) for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W1[(gw*8+l)*H1+n]); WLOAD.push_back({w, W1M+(n*(I/8)+gw)*32}); }
    for(int n=0;n<H1;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b1[n]); WLOAD.push_back({w, B1M+n*32}); }
    for(int n=0;n<H2;++n) for(int gw=0; gw<H1/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W2[(gw*8+l)*H2+n]); WLOAD.push_back({w, W2M+(n*(H1/8)+gw)*32}); }
    for(int n=0;n<H2;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b2[n]); WLOAD.push_back({w, B2M+n*32}); }
    for(int c=0;c<10;++c) for(int gw=0; gw<H2/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W3[(gw*8+l)*10+c]); WLOAD.push_back({w, W3M+(c*(H2/8)+gw)*32}); }
    int matWords=(int)WLOAD.size();

    // ================= UNROLLED SIMD (straight-line, with closed-form prediction) =================
    Code k;
    // CALLDATACOPY(INP,0,3136): 3 PUSH + (3 + 3*98) dynamic
    k.PUSH(CDIMG); k.P1(0); k.PUSH(INP); k.raw(0x37); k.g += 3 + 3*(CDIMG/32); k.mem(INP+CDIMG);
    // loader (unrolled; SLOAD warm=100 in the count)
    for(int sl=0; sl<matWords; ++sl){
        std::vector<uint8_t> key(32,0); key[28]=(sl>>24)&0xFF;key[29]=(sl>>16)&0xFF;key[30]=(sl>>8)&0xFF;key[31]=sl&0xFF;
        k.W32(key); k.op(0x54); k.PUSH(WLOAD[sl].second); k.op(0x52); k.mem(WLOAD[sl].second+32);
    }
    auto unrolled_layer=[&](int M,int NGin,int GINP,int GWGT,int GBIA,int GOUT,int SHIFT){
        for(int s=0;s<M/8;++s){ k.P1(0); k.PUSH(SACC+s*32); k.op(0x52); k.mem(SACC+s*32+32); }
        for(int n=0;n<M;++n){
            k.PUSH(GBIA+n*32); k.op(0x51); k.mem(GBIA+n*32+32);                 // [acc]=bias
            for(int g=0; g<NGin; ++g){
                k.PUSH(GINP+g*32); k.op(0x51); k.mem(GINP+g*32+32);             // [acc,inp]
                k.PUSH(GWGT+(n*NGin+g)*32); k.op(0x51); k.mem(GWGT+(n*NGin+g)*32+32); // [acc,inp,w]
                k.op(0xCB);                                                     // XMACC -> [acc']
            }
            k.op(0xCD);                                                         // XSUM32 -> [sum]
            int sh=(n%8)*32;
            if(sh){ k.PUSH(sh); k.op(0x1B); }                                   // sum<<sh
            int sw=SACC+(n/8)*32;
            k.PUSH(sw); k.op(0x51); k.op(0x17); k.PUSH(sw); k.op(0x52); k.mem(sw+32); // sacc|=
            if(n%8==7){
                k.PUSH(sw); k.op(0x51); k.op(0xCC); k.raw((uint8_t)SHIFT); k.op(0xCE);
                k.PUSH(GOUT+(n/8)*32); k.op(0x52); k.mem(GOUT+(n/8)*32+32);
            }
        }
    };
    unrolled_layer(H1, I/8, INP,   W1M, B1M, H1OUT, SHIFT1);
    long long g_l1=k.g;
    unrolled_layer(H2, H1/8, H1OUT, W2M, B2M, H2OUT, SHIFT2);
    for(int c=0;c<10;++c){                                                       // output
        k.P1(0);
        for(int g=0; g<H2/8; ++g){
            k.PUSH(H2OUT+g*32); k.op(0x51);
            k.PUSH(W3M+(c*(H2/8)+g)*32); k.op(0x51);
            k.op(0xCB);
        }
        k.op(0xCD); k.P1(3); k.op(0x0B);                                        // SIGNEXTEND(3): fully signed
        k.PUSH(SCO+c*32); k.op(0x52); k.mem(SCO+c*32+32);
    }
    // branchless argmax (identical to the mnist_e2e_cb scalar argmax, with MAXV/MAXI)
    k.PUSH(SCO); k.op(0x51); k.PUSH(MAXV); k.op(0x52); k.mem(MAXV+32);
    k.P1(0); k.PUSH(MAXI); k.op(0x52); k.mem(MAXI+32);
    for(int c=1;c<10;++c){
        k.PUSH(MAXV); k.op(0x51); k.PUSH(SCO+c*32); k.op(0x51); k.op(0x13);
        k.op(0x80);
        k.PUSH(SCO+c*32); k.op(0x51); k.PUSH(MAXV); k.op(0x51);
        k.op(0x90); k.op(0x03); k.op(0x02);
        k.PUSH(MAXV); k.op(0x51); k.op(0x01); k.PUSH(MAXV); k.op(0x52);
        k.PUSH(MAXI); k.op(0x51); k.op(0x80); k.PUSH(c); k.op(0x03);
        k.op(0x82); k.op(0x02); k.op(0x01); k.PUSH(MAXI); k.op(0x52);
        k.op(0x50);
    }
    long long g_body=k.g;
    k.PUSH(MAXI); k.op(0x51); k.P1(0); k.op(0x52); k.PUSH(0x20); k.P1(0); k.op(0xF3);
    long long predMS=((k.peakB+31)/32)*32;
    long long predWarm = k.g + memgas(k.peakB);
    std::vector<uint8_t> codeU=k.c;
    std::vector<uint8_t> codeUm(codeU.begin(),codeU.end()-13);
    { uint8_t t[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3}; for(uint8_t x:t) codeUm.push_back(x); }

    // ================= LOOPED SIMD (mnist_e2e_cb body, for the anchor) =================
    std::vector<uint8_t> code;
    auto P1=[&](uint8_t v){ code.push_back(0x60); code.push_back(v); };
    auto P3=[&](int v){ code.push_back(0x62); code.push_back((v>>16)&0xFF); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); };
    auto PUSH=[&](int v){ if(v<256)P1((uint8_t)v); else if(v<65536){code.push_back(0x61);code.push_back((v>>8)&0xFF);code.push_back(v&0xFF);} else P3(v); };
    auto P4=[&](uint32_t v){ code.push_back(0x63); code.push_back((v>>24)&0xFF); code.push_back((v>>16)&0xFF); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); };
    uint8_t xgtcfg=(1<<5)|2;
    auto emit_layer=[&](int M, int NGin, int GINP, int GWGT, int GBIA, int GOUT,
                        int ACC, int SACC, int NCNT, int GCNT, int SHIFT){
        for(int s=0;s<M/8;++s){ P1(0); PUSH(SACC+s*32); code.push_back(0x52); }
        P1(0); PUSH(NCNT); code.push_back(0x52);
        int outer_start=(int)code.size(); code.push_back(0x5B);
        PUSH(NCNT); code.push_back(0x51); PUSH(M); code.push_back(0x11); code.push_back(0x15);
        P3(0); int oe_ph=(int)code.size()-3; code.push_back(0x57);
        PUSH(NCNT); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(GBIA); code.push_back(0x01); code.push_back(0x51);
        PUSH(ACC); code.push_back(0x52);
        P1(0); PUSH(GCNT); code.push_back(0x52);
        int inner_start=(int)code.size(); code.push_back(0x5B);
        PUSH(GCNT); code.push_back(0x51); PUSH(NGin); code.push_back(0x11); code.push_back(0x15);
        P3(0); int ie_ph=(int)code.size()-3; code.push_back(0x57);
        PUSH(GCNT); code.push_back(0x51); P1(32); code.push_back(0x02); code.push_back(0x80); PUSH(GINP); code.push_back(0x01); code.push_back(0x51);
        code.push_back(0x90); PUSH(NCNT); code.push_back(0x51); PUSH(NGin*32); code.push_back(0x02); code.push_back(0x01); PUSH(GWGT); code.push_back(0x01); code.push_back(0x51);
        PUSH(ACC); code.push_back(0x51); code.push_back(0x91); code.push_back(0x90); code.push_back(0xCB); PUSH(ACC); code.push_back(0x52);
        PUSH(GCNT); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(GCNT); code.push_back(0x52);
        PUSH(inner_start); code.push_back(0x56);
        int inner_end=(int)code.size(); code.push_back(0x5B);
        code[ie_ph]=(inner_end>>16)&0xFF; code[ie_ph+1]=(inner_end>>8)&0xFF; code[ie_ph+2]=inner_end&0xFF;
        PUSH(ACC); code.push_back(0x51); code.push_back(0xCD);
        PUSH(NCNT); code.push_back(0x51); P1(8); code.push_back(0x90); code.push_back(0x06); P1(32); code.push_back(0x02); code.push_back(0x1B);
        PUSH(NCNT); code.push_back(0x51); P1(8); code.push_back(0x90); code.push_back(0x04); P1(32); code.push_back(0x02); PUSH(SACC); code.push_back(0x01);
        code.push_back(0x80); code.push_back(0x51); code.push_back(0x90); code.push_back(0x91); code.push_back(0x17); code.push_back(0x90); code.push_back(0x52);
        PUSH(NCNT); code.push_back(0x51); P1(8); code.push_back(0x90); code.push_back(0x06); P1(7); code.push_back(0x14); code.push_back(0x15);
        P3(0); int skip_ph=(int)code.size()-3; code.push_back(0x57);
        PUSH(NCNT); code.push_back(0x51); P1(8); code.push_back(0x90); code.push_back(0x04); P1(32); code.push_back(0x02); PUSH(SACC); code.push_back(0x01); code.push_back(0x51);
        code.push_back(0xCC); code.push_back((uint8_t)SHIFT); code.push_back(0xCE);
        PUSH(NCNT); code.push_back(0x51); P1(8); code.push_back(0x90); code.push_back(0x04); P1(32); code.push_back(0x02); PUSH(GOUT); code.push_back(0x01); code.push_back(0x52);
        int skip_target=(int)code.size(); code.push_back(0x5B);
        code[skip_ph]=(skip_target>>16)&0xFF; code[skip_ph+1]=(skip_target>>8)&0xFF; code[skip_ph+2]=skip_target&0xFF;
        PUSH(NCNT); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(NCNT); code.push_back(0x52);
        PUSH(outer_start); code.push_back(0x56);
        int outer_end=(int)code.size(); code.push_back(0x5B);
        code[oe_ph]=(outer_end>>16)&0xFF; code[oe_ph+1]=(outer_end>>8)&0xFF; code[oe_ph+2]=outer_end&0xFF;
    };
    auto PUSHslot=[&](uint32_t sl){ code.push_back(0x7F); for(int i=0;i<28;++i) code.push_back(0); code.push_back((sl>>24)&0xFF); code.push_back((sl>>16)&0xFF); code.push_back((sl>>8)&0xFF); code.push_back(sl&0xFF); };
    PUSH(CDIMG); P1(0); PUSH(INP); code.push_back(0x37);
    for(int sl=0; sl<matWords; ++sl){ PUSHslot((uint32_t)sl); code.push_back(0x54); PUSH(WLOAD[sl].second); code.push_back(0x52); }
    emit_layer(H1, I/8, INP, W1M, B1M, H1OUT, ACC, SACC, NCNT, GCNT, SHIFT1);
    emit_layer(H2, H1/8, H1OUT, W2M, B2M, H2OUT, ACC, SACC, NCNT, GCNT, SHIFT2);
    P1(0); PUSH(NCNT); code.push_back(0x52);
    int co_start=(int)code.size(); code.push_back(0x5B);
    PUSH(NCNT); code.push_back(0x51); P1(10); code.push_back(0x11); code.push_back(0x15);
    P3(0); int co_ph=(int)code.size()-3; code.push_back(0x57);
    P1(0); PUSH(ACC); code.push_back(0x52);
    P1(0); PUSH(GCNT); code.push_back(0x52);
    int ci_start=(int)code.size(); code.push_back(0x5B);
    PUSH(GCNT); code.push_back(0x51); PUSH(H2/8); code.push_back(0x11); code.push_back(0x15);
    P3(0); int ci_ph=(int)code.size()-3; code.push_back(0x57);
    PUSH(GCNT); code.push_back(0x51); P1(32); code.push_back(0x02); code.push_back(0x80); PUSH(H2OUT); code.push_back(0x01); code.push_back(0x51);
    code.push_back(0x90); PUSH(NCNT); code.push_back(0x51); PUSH((H2/8)*32); code.push_back(0x02); code.push_back(0x01); PUSH(W3M); code.push_back(0x01); code.push_back(0x51);
    PUSH(ACC); code.push_back(0x51); code.push_back(0x91); code.push_back(0x90); code.push_back(0xCB); PUSH(ACC); code.push_back(0x52);
    PUSH(GCNT); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(GCNT); code.push_back(0x52);
    PUSH(ci_start); code.push_back(0x56);
    int ci_end=(int)code.size(); code.push_back(0x5B);
    code[ci_ph]=(ci_end>>16)&0xFF; code[ci_ph+1]=(ci_end>>8)&0xFF; code[ci_ph+2]=ci_end&0xFF;
    PUSH(ACC); code.push_back(0x51); code.push_back(0xCD);
    PUSH(NCNT); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(SCO); code.push_back(0x01);
    code.push_back(0x52);
    PUSH(NCNT); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(NCNT); code.push_back(0x52);
    PUSH(co_start); code.push_back(0x56);
    int co_end=(int)code.size(); code.push_back(0x5B);
    code[co_ph]=(co_end>>16)&0xFF; code[co_ph+1]=(co_end>>8)&0xFF; code[co_ph+2]=co_end&0xFF;
    PUSH(SCO); code.push_back(0x51); PUSH(MAXV); code.push_back(0x52);
    P1(0); PUSH(MAXI); code.push_back(0x52);
    P1(1); PUSH(CNT2); code.push_back(0x52);
    int amloop=(int)code.size(); code.push_back(0x5B);
    PUSH(CNT2); code.push_back(0x51); P1(10); code.push_back(0x11); code.push_back(0x15);
    P3(0); int amend_ph=(int)code.size()-3; code.push_back(0x57);
    PUSH(MAXV); code.push_back(0x51);
    PUSH(CNT2); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(SCO); code.push_back(0x01); code.push_back(0x51);
    code.push_back(0xCA); code.push_back(xgtcfg);
    P4(0xFFFFFFFF); code.push_back(0x16); code.push_back(0x15);
    P3(0); int amskip_ph=(int)code.size()-3; code.push_back(0x57);
    PUSH(CNT2); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(SCO); code.push_back(0x01); code.push_back(0x51);
    PUSH(MAXV); code.push_back(0x52);
    PUSH(CNT2); code.push_back(0x51); PUSH(MAXI); code.push_back(0x52);
    int amskip=(int)code.size(); code.push_back(0x5B);
    code[amskip_ph]=(amskip>>16)&0xFF; code[amskip_ph+1]=(amskip>>8)&0xFF; code[amskip_ph+2]=amskip&0xFF;
    PUSH(CNT2); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(CNT2); code.push_back(0x52);
    PUSH(amloop); code.push_back(0x56);
    int amend=(int)code.size(); code.push_back(0x5B);
    code[amend_ph]=(amend>>16)&0xFF; code[amend_ph+1]=(amend>>8)&0xFF; code[amend_ph+2]=amend&0xFF;
    PUSH(MAXI); code.push_back(0x51); P1(0); code.push_back(0x52);
    PUSH(0x20); P1(0); code.push_back(0xF3);
    std::vector<uint8_t> codeL=code;
    std::vector<uint8_t> codeLm(codeL.begin(),codeL.end()-13);
    { uint8_t t[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3}; for(uint8_t x:t) codeLm.push_back(x); }

    printf("Bytecode: UNROLLED-SIMD=%zu B | LOOPED-SIMD=%zu B | matWords=%d\n",codeU.size(),codeL.size(),matWords);

    evmc::address ADDR{}; ADDR.bytes[18]=0xc0; ADDR.bytes[19]=0xde;
    evmc::MockedHost host;
    for(int sl=0; sl<matWords; ++sl){
        evmc::bytes32 key{}, val{};
        key.bytes[28]=(sl>>24)&0xFF; key.bytes[29]=(sl>>16)&0xFF; key.bytes[30]=(sl>>8)&0xFF; key.bytes[31]=sl&0xFF;
        for(int b=0;b<32;++b) val.bytes[b]=WLOAD[sl].first[31-b];
        host.accounts[ADDR].storage[key]=evmc::StorageValue{val};
    }
    auto setAcc=[&](evmc_access_status st){ for(int sl=0;sl<matWords;++sl){ evmc::bytes32 key{}; key.bytes[28]=(sl>>24)&0xFF;key.bytes[29]=(sl>>16)&0xFF;key.bytes[30]=(sl>>8)&0xFF;key.bytes[31]=sl&0xFF; host.accounts[ADDR].storage[key].access_status=st; } };
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();
    const int64_t GBIG=200000000000LL;
    auto buildCalldata=[&](int im, std::vector<uint8_t>& cd){
        cd.assign(CDIMG,0);
        for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0);
            for(int l=0;l<8;++l){ int val=imgs[im][gw*8+l]; uint16_t u=(uint16_t)(int16_t)val; w[l*2]=u&0xFF; w[l*2+1]=(u>>8)&0xFF; }
            for(int i=0;i<32;++i) cd[gw*32+i]=w[31-i]; }
    };
    auto run=[&](std::vector<uint8_t>& kk, std::vector<uint8_t>& cd, int& pred, long long& w0)->long long{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GBIG; msg.recipient=ADDR; msg.code_address=ADDR;
        msg.input_data=cd.data(); msg.input_size=cd.size();
        auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,kk.data(),kk.size());
        long long g=GBIG-res.gas_left; pred=-1; w0=-1;
        if(res.status_code==0&&res.output_data&&res.output_size>=32){
            pred=((int32_t)res.output_data[28]<<24)|((int32_t)res.output_data[29]<<16)|((int32_t)res.output_data[30]<<8)|(int32_t)res.output_data[31];
            long long m=0; for(int b=24;b<32;++b) m=(m<<8)|res.output_data[b]; w0=m;
        }
        if(res.release)res.release(&res); return g;
    };

    // CHECK 1
    int ok=0; long long w0;
    for(int im=0; im<20; ++im){
        std::vector<uint8_t> cd; buildCalldata(im,cd); int cp=cppPred(im),pU,pL;
        setAcc(EVMC_ACCESS_WARM); run(codeU,cd,pU,w0);
        setAcc(EVMC_ACCESS_WARM); run(codeL,cd,pL,w0);
        if(pU==cp&&pL==cp) ok++;
        else printf("  [MISMATCH] im=%d C++=%d UNROLLED=%d LOOPED=%d\n",im,cp,pU,pL);
    }
    printf("[CHECK 1] 20 images three-way bit-exact: %d/20 %s\n",ok,ok==20?"PASSED":"FAILED");
    if(ok!=20){ vm->destroy(vm); dlclose(h); return 1; }

    std::vector<uint8_t> cd0; buildCalldata(0,cd0); int dummy;
    long long cU1,cU2,wU,cL1,wL,msU,msL;
    setAcc(EVMC_ACCESS_WARM); wU=run(codeU,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); cU1=run(codeU,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); cU2=run(codeU,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_WARM); wL=run(codeL,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); cL1=run(codeL,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_WARM); run(codeUm,cd0,dummy,msU);
    setAcc(EVMC_ACCESS_WARM); run(codeLm,cd0,dummy,msL);

    printf("[CHECK 2] UNROLLED warm pred==meas: %lld/%lld diff=%lld %s | MSIZE pred/meas: %lld/%lld %s\n",
        predWarm,wU,wU-predWarm,wU==predWarm?"PASSED":"FAILED",predMS,msU,predMS==msU?"OK":"FAILED");
    printf("[CHECK 3] cold-warm==%lld: UNROLLED %lld %s | LOOPED %lld %s\n",
        (long long)matWords*2000, cU1-wU,(cU1-wU==(long long)matWords*2000)?"OK":"FAILED",
        cL1-wL,(cL1-wL==(long long)matWords*2000)?"OK":"FAILED");
    long long memU=memgas(msU), memL=memgas(msL);
    long long hU_c=cU1-(long long)matWords*2100-memU, hU_w=wU-(long long)matWords*100-memU;
    long long hL_w=wL-(long long)matWords*100-memL;
    printf("[CHECK 4] compute_cold==compute_warm (UNROLLED): %lld/%lld %s\n",hU_c,hU_w,hU_c==hU_w?"PASSED":"FAILED");
    printf("[CHECK 5] LOOPED warm == 5,134,719 anchor: %lld %s (compute bucket: %lld, anchor 2,117,326 %s)\n",
        wL,wL==5134719LL?"PASSED":"FAILED",hL_w,hL_w==2117326LL?"OK":"FAILED");
    printf("[CHECK 6] repeat run identical (UNROLLED cold): %s\n",cU1==cU2?"PASSED":"FAILED");

    long long SCALAR_COMPUTE=10408685LL; // constant verified three times (mnist_e2e_cb), not re-run here
    printf("\n================ UNROLLED-SIMD RESULT ================\n");
    printf("  UNROLLED-SIMD pure compute: %lld  (layer-1 share ~%lld)\n",hU_w,g_l1-(long long)matWords*100-3-3-3-3-3*(CDIMG/32)-3-3);
    printf("  LOOPED-SIMD pure compute: %lld  ->  loop-overhead factor: %.2fx\n",hL_w,(double)hL_w/hU_w);
    printf("  SCALAR (unrolled, constant): %lld\n",SCALAR_COMPUTE);
    printf("  PURE COMPUTE C/B: looped %.2fx  ->  UNROLLED %.2fx\n",(double)SCALAR_COMPUTE/hL_w,(double)SCALAR_COMPUTE/hU_w);
    printf("  per group (8 elements) unrolled kernel: PUSH+MLOAD+PUSH+MLOAD+XMACC = 23 gas (from the count)\n");
    { long long nmac=784LL*128+128LL*64+64LL*10;   // multiply-accumulate operations per inference (784->128->64->10)
      printf("  per multiply-accumulate (%lld MACs): SCALAR unrolled %.2f gas | UNROLLED-SIMD %.2f gas | 8-way core block 23/8 = %.3f gas\n",
          nmac,(double)SCALAR_COMPUTE/nmac,(double)hU_w/nmac,23.0/8.0); }
    bool all= ok==20 && wU==predWarm && predMS==msU && (cU1-wU==(long long)matWords*2000)
        && (cL1-wL==(long long)matWords*2000) && hU_c==hU_w && wL==5134719LL && cU1==cU2;
    printf("\n%s\n", all?"*** ALL CHECKS PASSED ***":"*** AT LEAST ONE CHECK FAILED -> do NOT record ***");
    vm->destroy(vm); dlclose(h); return all?0:1;
}
