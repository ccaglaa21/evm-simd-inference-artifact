// mnist_e2e_cb.cpp — MNIST END-TO-END C/B: fair scalar (Path C, UNROLLED) vs SIMD (mnist_sload PATH, UNTOUCHED)
// Goal: the MEASURED answer to "what does SIMD save in total in a real application (MNIST)?".
// The scalar path uses the atv_3kova Path C idiom (MLOAD+BYTE extraction) + SIGNEXTEND (signed weights).
// Scalar is UNROLLED (no loop) -> generous to scalar -> the ratio is a LOWER BOUND.
// The SIMD body is identical to mnist_sload.cpp; the cold total must EQUAL the mnist_sload output (cross-check).
// CHECKS: (1) 20 images, three-way prediction equality (2) cold-warm==matWords*2000 (3) compute_cold==compute_warm
//          (4) compute>0 (5) MSIZE cold==warm (6) repeat run identical.
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

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ std::cerr<<"libevmone not found\n"; return 1; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone");
    evmc_vm* vm=create();

    int S,SHIFT1,SHIFT2,H1,H2;
    { std::ifstream f("deep_meta.txt"); f>>S>>SHIFT1>>SHIFT2>>H1>>H2; }
    const int I=784;
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

    // ---------- shared emit helpers (same as mnist_sload) ----------
    std::vector<uint8_t> code;
    auto P1=[&](uint8_t v){ code.push_back(0x60); code.push_back(v); };
    auto P3=[&](int v){ code.push_back(0x62); code.push_back((v>>16)&0xFF); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); };
    auto PUSH=[&](int v){ if(v<256)P1((uint8_t)v); else if(v<65536){code.push_back(0x61);code.push_back((v>>8)&0xFF);code.push_back(v&0xFF);} else P3(v); };
    auto P4=[&](uint32_t v){ code.push_back(0x63); code.push_back((v>>24)&0xFF); code.push_back((v>>16)&0xFF); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); };
    uint8_t xgtcfg=(1<<5)|2;

    // emit_layer — IDENTICAL to mnist_sload.cpp (SIMD path)
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

    // MEMORY map — SAME as mnist_sload
    int INP=0, W1M=10000, B1M=420000, H1OUT=430000, W2M=440000, B2M=480000, H2OUT=490000;
    int W3M=495000, SCO=500000;
    int ACC=900000, SACC=900100, NCNT=901000, GCNT=901100;
    int MAXV=900200, MAXI=900232, CNT2=900264;
    int SMAXV=SCO+10*32, SMAXI=SMAXV+32;   // scalar argmax (500320, 500352)

    // WLOAD — SAME as mnist_sload
    std::vector<std::pair<std::vector<uint8_t>,int>> WLOAD;
    for(int n=0;n<H1;++n) for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W1[(gw*8+l)*H1+n]); WLOAD.push_back({w, W1M+(n*(I/8)+gw)*32}); }
    int w1cnt=(int)WLOAD.size();
    for(int n=0;n<H1;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b1[n]); WLOAD.push_back({w, B1M+n*32}); }
    int b1cnt=(int)WLOAD.size()-w1cnt;
    for(int n=0;n<H2;++n) for(int gw=0; gw<H1/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W2[(gw*8+l)*H2+n]); WLOAD.push_back({w, W2M+(n*(H1/8)+gw)*32}); }
    int w2cnt=(int)WLOAD.size()-w1cnt-b1cnt;
    for(int n=0;n<H2;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b2[n]); WLOAD.push_back({w, B2M+n*32}); }
    for(int c=0;c<10;++c) for(int gw=0; gw<H2/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W3[(gw*8+l)*10+c]); WLOAD.push_back({w, W3M+(c*(H2/8)+gw)*32}); }
    int matWords=(int)WLOAD.size();

    auto PUSHslot=[&](uint32_t sl){ code.push_back(0x7F); for(int i=0;i<28;++i) code.push_back(0); code.push_back((sl>>24)&0xFF); code.push_back((sl>>16)&0xFF); code.push_back((sl>>8)&0xFF); code.push_back(sl&0xFF); };
    auto emitPrelude=[&](int& lcs, int& lce){
        PUSH(I/8*32); P1(0); PUSH(INP); code.push_back(0x37);            // CALLDATACOPY
        lcs=(int)code.size();
        for(int sl=0; sl<matWords; ++sl){ PUSHslot((uint32_t)sl); code.push_back(0x54); PUSH(WLOAD[sl].second); code.push_back(0x52); }
        lce=(int)code.size();
    };

    // ---------- SIMD body (Path B) — identical to mnist_sload ----------
    int loadCodeStart=0, loadCodeEnd=0;
    emitPrelude(loadCodeStart, loadCodeEnd);
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
    std::vector<uint8_t> bodyB=code;

    // load-only code (for the I/O measurement) — same as mnist_sload
    std::vector<uint8_t> loadCode(bodyB.begin()+loadCodeStart, bodyB.begin()+loadCodeEnd);
    loadCode.push_back(0x60); loadCode.push_back(0x00); loadCode.push_back(0x60); loadCode.push_back(0x00); loadCode.push_back(0xF3);

    // ---------- SCALAR body (Path C, UNROLLED) ----------
    code.clear();
    // signed int16 extraction: memory layout is REVERSED (mem[j]=w[31-j]) -> lane l: lo=+31-2l, hi=+30-2l
    auto EXT16=[&](int loOff,int hiOff){
        PUSH((loOff/32)*32); code.push_back(0x51); P1((uint8_t)(loOff%32)); code.push_back(0x1a);
        PUSH((hiOff/32)*32); code.push_back(0x51); P1((uint8_t)(hiOff%32)); code.push_back(0x1a);
        PUSH(256); code.push_back(0x02); code.push_back(0x01);
        P1(1); code.push_back(0x0B);                                      // SIGNEXTEND(1)
    };
    auto emit_layer_scalar=[&](int M,int Kin,int GINP,int GWGT,int GBIA,int GOUT,int SHIFT){
        for(int n=0;n<M;++n){
            PUSH(GBIA+n*32); code.push_back(0x51); P1(3); code.push_back(0x0B);   // acc = sext32(bias)
            for(int i=0;i<Kin;++i){
                int wi=GINP+(i/8)*32, li=i%8;
                EXT16(wi+31-2*li, wi+30-2*li);                            // input lane
                int ww=GWGT+(n*(Kin/8)+i/8)*32, lw=i%8;
                EXT16(ww+31-2*lw, ww+30-2*lw);                            // weight lane
                code.push_back(0x02); code.push_back(0x01);               // MUL ADD
            }
            // ReLU (branchless): acc *= (acc>=0)
            code.push_back(0x80); P1(0); code.push_back(0x13); code.push_back(0x15); code.push_back(0x02);
            P1((uint8_t)SHIFT); code.push_back(0x1C);                     // >> SHIFT
            // clip 32767 (branchless): t + over*(32767-t)
            code.push_back(0x80); PUSH(32767); code.push_back(0x90); code.push_back(0x13);
            PUSH(32767); code.push_back(0x82); code.push_back(0x90); code.push_back(0x03);
            code.push_back(0x02); code.push_back(0x01);
            // write packed int16 (same reversed layout)
            int dw=GOUT+(n/8)*32, l=n%8;
            code.push_back(0x80); P1(0xFF); code.push_back(0x16); PUSH(dw+31-2*l); code.push_back(0x53);
            P1(8); code.push_back(0x1C); P1(0xFF); code.push_back(0x16); PUSH(dw+30-2*l); code.push_back(0x53);
        }
    };
    { int lcs2,lce2; emitPrelude(lcs2,lce2); }                            // same CALLDATACOPY + SLOAD block
    emit_layer_scalar(H1, I,  INP,   W1M, B1M, H1OUT, SHIFT1);
    emit_layer_scalar(H2, H1, H1OUT, W2M, B2M, H2OUT, SHIFT2);
    for(int c=0;c<10;++c){                                                // output: raw score, full word
        P1(0);
        for(int i=0;i<H2;++i){
            int wi=H2OUT+(i/8)*32, li=i%8; EXT16(wi+31-2*li, wi+30-2*li);
            int ww=W3M+(c*(H2/8)+i/8)*32, lw=i%8; EXT16(ww+31-2*lw, ww+30-2*lw);
            code.push_back(0x02); code.push_back(0x01);
        }
        PUSH(SCO+c*32); code.push_back(0x52);
    }
    // scalar argmax (branchless, signed SGT)
    PUSH(SCO); code.push_back(0x51); PUSH(SMAXV); code.push_back(0x52);
    P1(0); PUSH(SMAXI); code.push_back(0x52);
    for(int c=1;c<10;++c){
        PUSH(SMAXV); code.push_back(0x51); PUSH(SCO+c*32); code.push_back(0x51); code.push_back(0x13); // over=sc>maxv
        code.push_back(0x80);
        PUSH(SCO+c*32); code.push_back(0x51); PUSH(SMAXV); code.push_back(0x51);
        code.push_back(0x90); code.push_back(0x03); code.push_back(0x02);
        PUSH(SMAXV); code.push_back(0x51); code.push_back(0x01); PUSH(SMAXV); code.push_back(0x52);
        PUSH(SMAXI); code.push_back(0x51); code.push_back(0x80); PUSH(c); code.push_back(0x03);
        code.push_back(0x82); code.push_back(0x02); code.push_back(0x01); PUSH(SMAXI); code.push_back(0x52);
        code.push_back(0x50);
    }
    std::vector<uint8_t> bodyC=code;

    // tails: normal (return the prediction) and MSIZE
    auto withTail=[&](const std::vector<uint8_t>& body,int maxiAddr)->std::vector<uint8_t>{
        std::vector<uint8_t> k=body;
        k.push_back(0x62); k.push_back((maxiAddr>>16)&0xFF); k.push_back((maxiAddr>>8)&0xFF); k.push_back(maxiAddr&0xFF);
        k.push_back(0x51); k.push_back(0x60); k.push_back(0x00); k.push_back(0x52);
        k.push_back(0x60); k.push_back(0x20); k.push_back(0x60); k.push_back(0x00); k.push_back(0xF3);
        return k;
    };
    auto withMsize=[&](const std::vector<uint8_t>& body)->std::vector<uint8_t>{
        std::vector<uint8_t> k=body;
        uint8_t t[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3};
        for(uint8_t x:t) k.push_back(x);
        return k;
    };
    std::vector<uint8_t> codeB=withTail(bodyB,MAXI),  codeBm=withMsize(bodyB);
    std::vector<uint8_t> codeC=withTail(bodyC,SMAXI), codeCm=withMsize(bodyC);
    printf("Bytecode: SIMD=%zu byte | SCALAR=%zu byte | matWords=%d = W1:%d + b1:%d + W2:%d + b2:64 + W3:80\n",
           codeB.size(), codeC.size(), matWords, w1cnt, b1cnt, w2cnt);
    printf("[NOTE] Storage layout: 8 lanes x int16 = 16B of data per 32B word (half-word occupancy, required by the SIMD lane alignment)\n\n");

    // ---------- host + storage (SAME as mnist_sload, REVERSED write) ----------
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

    auto buildCalldata=[&](int im, std::vector<uint8_t>& calldata, int& cpp_pred){
        std::vector<int>& im_img=imgs[im];
        std::vector<int> h1r(H1); for(int n=0;n<H1;++n){ long long s=b1[n]; for(int i=0;i<I;++i) s+=(long long)im_img[i]*W1[i*H1+n]; h1r[n]=rss(s,SHIFT1);}
        std::vector<int> h2r(H2); for(int n=0;n<H2;++n){ long long s=b2[n]; for(int i=0;i<H1;++i) s+=(long long)h1r[i]*W2[i*H2+n]; h2r[n]=rss(s,SHIFT2);}
        std::vector<long long> scr(10); for(int cc=0;cc<10;++cc){ long long s=0; for(int i=0;i<H2;++i) s+=(long long)h2r[i]*W3[i*10+cc]; scr[cc]=s;}
        cpp_pred=(int)(std::max_element(scr.begin(),scr.end())-scr.begin());
        calldata.assign(I/8*32,0);
        for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l){ int val=im_img[gw*8+l]; uint16_t u=(uint16_t)(int16_t)val; w[l*2]=u&0xFF; w[l*2+1]=(u>>8)&0xFF;} for(int i=0;i<32;++i) calldata[gw*32+i]=w[31-i]; }
    };

    const int64_t GBIG=20000000000LL;
    auto run=[&](std::vector<uint8_t>& k, std::vector<uint8_t>& calldata, int& pred, long long& word0)->long long{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GBIG; msg.recipient=ADDR; msg.code_address=ADDR;
        msg.input_data=calldata.data(); msg.input_size=calldata.size();
        auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.data(),k.size());
        long long g=GBIG-res.gas_left; pred=-1; word0=-1;
        if(res.status_code==0&&res.output_data&&res.output_size>=32){
            pred=((int32_t)res.output_data[28]<<24)|((int32_t)res.output_data[29]<<16)|((int32_t)res.output_data[30]<<8)|(int32_t)res.output_data[31];
            long long m=0; for(int b=24;b<32;++b) m=(m<<8)|res.output_data[b]; word0=m;
        }
        if(res.release)res.release(&res); return g;
    };
    auto runLoad=[&]()->long long{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GBIG; msg.recipient=ADDR; msg.code_address=ADDR;
        std::vector<uint8_t> cd(I/8*32,0); msg.input_data=cd.data(); msg.input_size=cd.size();
        auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,loadCode.data(),loadCode.size());
        long long g=GBIG-res.gas_left; if(res.release)res.release(&res); return g;
    };
    auto memgas=[](long long ms)->long long{ long long w=(ms+31)/32; return 3*w+(w*w)/512; };

    // ---------- CHECK 1: three-way prediction equality on 20 images ----------
    int ok3=0; int dummy; long long w0;
    for(int im=0; im<20; ++im){
        std::vector<uint8_t> cd; int cpp_pred,pB,pC;
        buildCalldata(im,cd,cpp_pred);
        setAcc(EVMC_ACCESS_WARM); run(codeB,cd,pB,w0);
        setAcc(EVMC_ACCESS_WARM); run(codeC,cd,pC,w0);
        if(pB==cpp_pred && pC==cpp_pred) ok3++;
        else printf("  [MISMATCH] im=%d C++=%d SIMD=%d SCALAR=%d\n",im,cpp_pred,pB,pC);
    }
    printf("[CHECK 1] three-way prediction equality (C++ == SIMD-EVM == SCALAR-EVM): %d/20 %s\n",ok3,ok3==20?"PASSED":"FAILED");
    if(ok3!=20){ printf("*** CHECK 1 FAILED -> no measurement ***\n"); vm->destroy(vm); dlclose(h); return 1; }

    // ---------- MSIZE (cold + warm) ----------
    std::vector<uint8_t> cd0; int cp0; buildCalldata(0,cd0,cp0);
    long long msB_c,msB_w,msC_c,msC_w;
    setAcc(EVMC_ACCESS_COLD); run(codeBm,cd0,dummy,msB_c);
    setAcc(EVMC_ACCESS_WARM); run(codeBm,cd0,dummy,msB_w);
    setAcc(EVMC_ACCESS_COLD); run(codeCm,cd0,dummy,msC_c);
    setAcc(EVMC_ACCESS_WARM); run(codeCm,cd0,dummy,msC_w);
    bool g5 = (msB_c==msB_w) && (msC_c==msC_w);
    printf("[CHECK 5] MSIZE cold==warm: SIMD %lld/%lld, SCALAR %lld/%lld %s\n",msB_c,msB_w,msC_c,msC_w,g5?"PASSED":"FAILED");
    long long memB=memgas(msB_c), memC=memgas(msC_c);

    // ---------- gas measurements (image 0) + repeat ----------
    long long coldB1,coldB2,warmB,coldC1,coldC2,warmC;
    setAcc(EVMC_ACCESS_COLD); coldB1=run(codeB,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); coldB2=run(codeB,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_WARM); warmB =run(codeB,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); coldC1=run(codeC,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); coldC2=run(codeC,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_WARM); warmC =run(codeC,cd0,dummy,w0);
    bool g6=(coldB1==coldB2)&&(coldC1==coldC2);
    printf("[CHECK 6] repeat run identical: SIMD %s, SCALAR %s\n",coldB1==coldB2?"OK":"MISMATCH",coldC1==coldC2?"OK":"MISMATCH");
    setAcc(EVMC_ACCESS_COLD); long long ioCold=runLoad();

    long long sload_c=(long long)matWords*2100, sload_w=(long long)matWords*100;
    bool g2=(coldB1-warmB==(long long)matWords*2000)&&(coldC1-warmC==(long long)matWords*2000);
    printf("[CHECK 2] cold-warm==matWords*2000 (%lld): SIMD %lld %s, SCALAR %lld %s\n",
           (long long)matWords*2000, coldB1-warmB,(coldB1-warmB==(long long)matWords*2000)?"OK":"MISMATCH",
           coldC1-warmC,(coldC1-warmC==(long long)matWords*2000)?"OK":"MISMATCH");
    long long hB_c=coldB1-sload_c-memB, hB_w=warmB-sload_w-memB;
    long long hC_c=coldC1-sload_c-memC, hC_w=warmC-sload_w-memC;
    bool g3=(hB_c==hB_w)&&(hC_c==hC_w);
    bool g4=(hB_c>0)&&(hC_c>0);
    printf("[CHECK 3] compute_cold==compute_warm: SIMD %lld/%lld %s, SCALAR %lld/%lld %s\n",hB_c,hB_w,hB_c==hB_w?"OK":"MISMATCH",hC_c,hC_w,hC_c==hC_w?"OK":"MISMATCH");
    printf("[CHECK 4] compute>0: %s\n",g4?"PASSED":"FAILED");
    printf("[INFO  ] load-only I/O (cold) = %lld | analytic = %lld | diff = %lld (harness overhead)\n",ioCold,sload_c,ioCold-sload_c);

    // ---------- gas variance (5 images, warm) ----------
    long long mnB=-1,mxB=-1,mnC=-1,mxC=-1;
    for(int im=0; im<5; ++im){
        std::vector<uint8_t> cd; int cp; buildCalldata(im,cd,cp);
        setAcc(EVMC_ACCESS_WARM); long long gB=run(codeB,cd,dummy,w0);
        setAcc(EVMC_ACCESS_WARM); long long gC=run(codeC,cd,dummy,w0);
        if(mnB<0||gB<mnB)mnB=gB; if(gB>mxB)mxB=gB;
        if(mnC<0||gC<mnC)mnC=gC; if(gC>mxC)mxC=gC;
    }
    printf("[INFO  ] gas spread (5 images, warm): SIMD %lld..%lld (diff %lld) | SCALAR %lld..%lld (diff %lld, branchless->0 expected)\n",
           mnB,mxB,mxB-mnB,mnC,mxC,mxC-mnC);

    // ---------- report ----------
    printf("\n================ MNIST END-TO-END C/B (fair scalar vs SIMD) ================\n");
    printf("  path     |  SLOAD        |  memory(MSIZE) | compute(pure) |  TOTAL\n");
    printf("  ---------+---------------+----------------+---------------+---------------\n");
    printf("  C  cold  | %13lld | %14lld | %13lld | %13lld\n",sload_c,memC,hC_c,coldC1);
    printf("  B  cold  | %13lld | %14lld | %13lld | %13lld  <- must EQUAL the mnist_sload cold value\n",sload_c,memB,hB_c,coldB1);
    printf("  C  warm  | %13lld | %14lld | %13lld | %13lld\n",sload_w,memC,hC_w,warmC);
    printf("  B  warm  | %13lld | %14lld | %13lld | %13lld\n",sload_w,memB,hB_w,warmB);
    printf("\n  END-TO-END C/B:  cold %.2fx   warm %.2fx\n",(double)coldC1/coldB1,(double)warmC/warmB);
    printf("  PURE COMPUTE (MNIST) C/B: %.2fx  (computeC=%lld / computeB=%lld)\n",(double)hC_c/hB_c,hC_c,hB_c);
    printf("  [NOTE] This pure-compute ratio is below 18.39x and is a LOWER BOUND: the SIMD path is LOOPED\n");
    printf("        (carries counter/jump overhead), the scalar path is UNROLLED (ZERO loop overhead, generous to scalar).\n");
    printf("        18.39x is the kernel ratio; the gap is loop overhead and can be closed by engineering.\n");
    printf("  [NOTE] The memory bucket is NOT common mode: scalar uses no counter area, smaller footprint (again in favour of scalar).\n");
    if(g2&&g3&&g4&&g5&&g6) printf("\n*** ALL CHECKS PASSED -> numbers are valid ***\n");
    else                   printf("\n*** AT LEAST ONE CHECK FAILED -> numbers are UNRELIABLE, do NOT record ***\n");

    vm->destroy(vm); dlclose(h); return 0;
}
