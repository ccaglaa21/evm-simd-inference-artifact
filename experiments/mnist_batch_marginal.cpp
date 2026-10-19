// mnist_batch_marginal.cpp — BATCH / MARGINAL COST
// Question: the saving a user ACTUALLY pays for = marginal cost per image.
// Single call, N images in calldata (N=CALLDATASIZE/3136, code fixed); the weight-SLOAD
// block runs once OUTSIDE the loop; each iteration: CALLDATACOPY -> inference -> prediction into PREDS.
// Linearity is measured with the SAME image (SIMD argmax gas is image-dependent, ~123 gas);
// bit-exact correctness in a SEPARATE run with 16 DIFFERENT images.
// CHECKS: (1) 16 different images, both paths bit-exact (2) memory-stripped gas
// EXACTLY linear in N (cold+warm, both paths) (3) cold-warm==matWords*2000 for every N
// (4) MSIZE == PREDS+N*32 exactly (5) repeat run identical.
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
    const int I=784, CDIMG=I/8*32; // 3136
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
    auto imgWords=[&](int im, std::vector<uint8_t>& dst){ // 3136B, reversed layout
        for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0);
            for(int l=0;l<8;++l){ int val=imgs[im][gw*8+l]; uint16_t u=(uint16_t)(int16_t)val; w[l*2]=u&0xFF; w[l*2+1]=(u>>8)&0xFF; }
            for(int i=0;i<32;++i) dst.push_back(w[31-i]); }
    };

    // MEMORY map (same kernel as mnist_e2e_cb) + batch cells
    int INP=0, W1M=10000, B1M=420000, H1OUT=430000, W2M=440000, B2M=480000, H2OUT=490000;
    int W3M=495000, SCO=500000;
    int ACC=900000, SACC=900100, NCNT=901000, GCNT=901100;
    int MAXV=900200, MAXI=900232, CNT2=900264;
    int SMAXV=SCO+10*32, SMAXI=SMAXV+32;                 // 500320, 500352
    int NIMGc=500384, ICNTc=500416, PREDSc=500448;       // scalar batch cells
    int NIMGb=901152, ICNTb=901184, PREDSb=901216;       // SIMD batch cells

    std::vector<std::pair<std::vector<uint8_t>,int>> WLOAD;
    for(int n=0;n<H1;++n) for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W1[(gw*8+l)*H1+n]); WLOAD.push_back({w, W1M+(n*(I/8)+gw)*32}); }
    for(int n=0;n<H1;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b1[n]); WLOAD.push_back({w, B1M+n*32}); }
    for(int n=0;n<H2;++n) for(int gw=0; gw<H1/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W2[(gw*8+l)*H2+n]); WLOAD.push_back({w, W2M+(n*(H1/8)+gw)*32}); }
    for(int n=0;n<H2;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b2[n]); WLOAD.push_back({w, B2M+n*32}); }
    for(int c=0;c<10;++c) for(int gw=0; gw<H2/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W3[(gw*8+l)*10+c]); WLOAD.push_back({w, W3M+(c*(H2/8)+gw)*32}); }
    int matWords=(int)WLOAD.size();

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

    auto emitSimdBody=[&](){   // mnist_e2e_cb SIMD body (INP -> MAXI)
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
    };

    auto EXT16=[&](int loOff,int hiOff){
        PUSH((loOff/32)*32); code.push_back(0x51); P1((uint8_t)(loOff%32)); code.push_back(0x1a);
        PUSH((hiOff/32)*32); code.push_back(0x51); P1((uint8_t)(hiOff%32)); code.push_back(0x1a);
        PUSH(256); code.push_back(0x02); code.push_back(0x01);
        P1(1); code.push_back(0x0B);
    };
    auto emit_layer_scalar=[&](int M,int Kin,int GINP,int GWGT,int GBIA,int GOUT,int SHIFT){
        for(int n=0;n<M;++n){
            PUSH(GBIA+n*32); code.push_back(0x51); P1(3); code.push_back(0x0B);
            for(int i=0;i<Kin;++i){
                int wi=GINP+(i/8)*32, li=i%8;
                EXT16(wi+31-2*li, wi+30-2*li);
                int ww=GWGT+(n*(Kin/8)+i/8)*32, lw=i%8;
                EXT16(ww+31-2*lw, ww+30-2*lw);
                code.push_back(0x02); code.push_back(0x01);
            }
            code.push_back(0x80); P1(0); code.push_back(0x13); code.push_back(0x15); code.push_back(0x02);
            P1((uint8_t)SHIFT); code.push_back(0x1C);
            code.push_back(0x80); PUSH(32767); code.push_back(0x90); code.push_back(0x13);
            PUSH(32767); code.push_back(0x82); code.push_back(0x90); code.push_back(0x03);
            code.push_back(0x02); code.push_back(0x01);
            int dw=GOUT+(n/8)*32, l=n%8;
            code.push_back(0x80); P1(0xFF); code.push_back(0x16); PUSH(dw+31-2*l); code.push_back(0x53);
            P1(8); code.push_back(0x1C); P1(0xFF); code.push_back(0x16); PUSH(dw+30-2*l); code.push_back(0x53);
        }
    };
    auto emitScalarBody=[&](){   // mnist_e2e_cb scalar body (INP -> SMAXI)
        emit_layer_scalar(H1, I,  INP,   W1M, B1M, H1OUT, SHIFT1);
        emit_layer_scalar(H2, H1, H1OUT, W2M, B2M, H2OUT, SHIFT2);
        for(int c=0;c<10;++c){
            P1(0);
            for(int i=0;i<H2;++i){
                int wi=H2OUT+(i/8)*32, li=i%8; EXT16(wi+31-2*li, wi+30-2*li);
                int ww=W3M+(c*(H2/8)+i/8)*32, lw=i%8; EXT16(ww+31-2*lw, ww+30-2*lw);
                code.push_back(0x02); code.push_back(0x01);
            }
            PUSH(SCO+c*32); code.push_back(0x52);
        }
        PUSH(SCO); code.push_back(0x51); PUSH(SMAXV); code.push_back(0x52);
        P1(0); PUSH(SMAXI); code.push_back(0x52);
        for(int c=1;c<10;++c){
            PUSH(SMAXV); code.push_back(0x51); PUSH(SCO+c*32); code.push_back(0x51); code.push_back(0x13);
            code.push_back(0x80);
            PUSH(SCO+c*32); code.push_back(0x51); PUSH(SMAXV); code.push_back(0x51);
            code.push_back(0x90); code.push_back(0x03); code.push_back(0x02);
            PUSH(SMAXV); code.push_back(0x51); code.push_back(0x01); PUSH(SMAXV); code.push_back(0x52);
            PUSH(SMAXI); code.push_back(0x51); code.push_back(0x80); PUSH(c); code.push_back(0x03);
            code.push_back(0x82); code.push_back(0x02); code.push_back(0x01); PUSH(SMAXI); code.push_back(0x52);
            code.push_back(0x50);
        }
    };

    auto PUSHslot=[&](uint32_t sl){ code.push_back(0x7F); for(int i=0;i<28;++i) code.push_back(0); code.push_back((sl>>24)&0xFF); code.push_back((sl>>16)&0xFF); code.push_back((sl>>8)&0xFF); code.push_back(sl&0xFF); };
    auto buildProgram=[&](bool simd)->std::vector<uint8_t>{
        code.clear();
        int NIMG = simd? NIMGb:NIMGc, ICNT = simd? ICNTb:ICNTc, PREDS = simd? PREDSb:PREDSc;
        int PSRC = simd? MAXI:SMAXI;
        for(int sl=0; sl<matWords; ++sl){ PUSHslot((uint32_t)sl); code.push_back(0x54); PUSH(WLOAD[sl].second); code.push_back(0x52); }  // weights: ONCE
        code.push_back(0x36); PUSH(CDIMG); code.push_back(0x90); code.push_back(0x04); PUSH(NIMG); code.push_back(0x52); // NIMG=CALLDATASIZE/3136
        P1(0); PUSH(ICNT); code.push_back(0x52);
        int lp=(int)code.size(); code.push_back(0x5B);
        PUSH(ICNT); code.push_back(0x51); PUSH(NIMG); code.push_back(0x51); code.push_back(0x11); code.push_back(0x15);
        P3(0); int eph=(int)code.size()-3; code.push_back(0x57);
        PUSH(CDIMG); PUSH(ICNT); code.push_back(0x51); PUSH(CDIMG); code.push_back(0x02); PUSH(INP); code.push_back(0x37); // CALLDATACOPY(INP, i*3136, 3136)
        if(simd) emitSimdBody(); else emitScalarBody();
        PUSH(PSRC); code.push_back(0x51);
        PUSH(ICNT); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(PREDS); code.push_back(0x01); code.push_back(0x52); // preds[i]=prediction
        PUSH(ICNT); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(ICNT); code.push_back(0x52);
        PUSH(lp); code.push_back(0x56);
        int ed=(int)code.size(); code.push_back(0x5B);
        code[eph]=(ed>>16)&0xFF; code[eph+1]=(ed>>8)&0xFF; code[eph+2]=ed&0xFF;
        PUSH(NIMG); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(PREDS); code.push_back(0xF3); // RETURN(PREDS, N*32)
        return code;
    };
    auto withMsize=[&](std::vector<uint8_t> k)->std::vector<uint8_t>{
        // tail: PUSH3(NIMG)[4]+MLOAD[1]+PUSH1_32[2]+MUL[1]+PUSH3(PREDS)[4]+RETURN[1]=13B -> replace with the MSIZE tail
        for(int t=0;t<13;++t) k.pop_back();
        uint8_t t2[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3};
        for(uint8_t x:t2) k.push_back(x);
        return k;
    };
    std::vector<uint8_t> codeB=buildProgram(true),  codeBm=withMsize(codeB);
    std::vector<uint8_t> codeC=buildProgram(false), codeCm=withMsize(codeC);
    printf("Bytecode: SIMD=%zu B | SCALAR=%zu B | matWords=%d\n",codeB.size(),codeC.size(),matWords);

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
    auto run=[&](std::vector<uint8_t>& k, std::vector<uint8_t>& cd, std::vector<int>& preds, long long& w0)->long long{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GBIG; msg.recipient=ADDR; msg.code_address=ADDR;
        msg.input_data=cd.data(); msg.input_size=cd.size();
        auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.data(),k.size());
        long long g=GBIG-res.gas_left; preds.clear(); w0=-1;
        if(res.status_code==0&&res.output_data&&res.output_size>=32){
            long long m=0; for(int b=24;b<32;++b) m=(m<<8)|res.output_data[b]; w0=m;
            int n=(int)(res.output_size/32);
            for(int i=0;i<n;++i){ int p=((int32_t)res.output_data[i*32+28]<<24)|((int32_t)res.output_data[i*32+29]<<16)|((int32_t)res.output_data[i*32+30]<<8)|(int32_t)res.output_data[i*32+31]; preds.push_back(p); }
        }
        if(res.release)res.release(&res); return g;
    };
    auto memgas=[](long long ms)->long long{ long long w=(ms+31)/32; return 3*w+(w*w)/512; };

    // ---- CHECK 1: 16 DIFFERENT images, both paths bit-exact ----
    std::vector<uint8_t> cd16; for(int im=0;im<16;++im) imgWords(im,cd16);
    std::vector<int> pB,pC; long long w0;
    setAcc(EVMC_ACCESS_WARM); run(codeB,cd16,pB,w0);
    setAcc(EVMC_ACCESS_WARM); run(codeC,cd16,pC,w0);
    int ok=0;
    for(int im=0;im<16;++im){ int cp=cppPred(im);
        if((int)pB.size()>im&&(int)pC.size()>im&&pB[im]==cp&&pC[im]==cp) ok++;
        else printf("  [MISMATCH] im=%d C++=%d SIMD=%d SCALAR=%d\n",im,cp,(int)pB.size()>im?pB[im]:-1,(int)pC.size()>im?pC[im]:-1); }
    printf("[CHECK 1] 16 different images, both paths bit-exact: %d/16 %s\n",ok,ok==16?"PASSED":"FAILED");
    if(ok!=16){ vm->destroy(vm); dlclose(h); return 1; }

    // ---- N sweep (same image 0, calldata with N copies) ----
    std::vector<int> Ns={1,2,4,8,16};
    long long GC_c[5],GC_w[5],GB_c[5],GB_w[5],MSC[5],MSB[5];
    bool fail=false;
    for(int ni=0;ni<5;++ni){
        int N=Ns[ni];
        std::vector<uint8_t> cd; for(int r=0;r<N;++r) imgWords(0,cd);
        std::vector<int> pr;
        setAcc(EVMC_ACCESS_COLD); GB_c[ni]=run(codeB,cd,pr,w0);
        setAcc(EVMC_ACCESS_COLD); long long r2=run(codeB,cd,pr,w0);
        if(GB_c[ni]!=r2){ printf("[CHECK 5] SIMD N=%d repeat MISMATCH\n",N); fail=true; }
        setAcc(EVMC_ACCESS_WARM); GB_w[ni]=run(codeB,cd,pr,w0);
        setAcc(EVMC_ACCESS_COLD); GC_c[ni]=run(codeC,cd,pr,w0);
        setAcc(EVMC_ACCESS_COLD); r2=run(codeC,cd,pr,w0);
        if(GC_c[ni]!=r2){ printf("[CHECK 5] SCALAR N=%d repeat MISMATCH\n",N); fail=true; }
        setAcc(EVMC_ACCESS_WARM); GC_w[ni]=run(codeC,cd,pr,w0);
        setAcc(EVMC_ACCESS_WARM); run(codeBm,cd,pr,MSB[ni]);
        setAcc(EVMC_ACCESS_WARM); run(codeCm,cd,pr,MSC[ni]);
        bool g3=(GB_c[ni]-GB_w[ni]==(long long)matWords*2000)&&(GC_c[ni]-GC_w[ni]==(long long)matWords*2000);
        bool g4=(MSB[ni]==(long long)PREDSb+N*32)&&(MSC[ni]==(long long)PREDSc+N*32);
        printf("[N=%2d] SIMD cold=%lld warm=%lld | SCALAR cold=%lld warm=%lld | K3(cold-warm==27.68M):%s K4(MSIZE==PREDS+N*32):%s\n",
            N,GB_c[ni],GB_w[ni],GC_c[ni],GC_w[ni],g3?"OK":"FAILED",g4?"OK":"FAILED");
        if(!g3||!g4) fail=true;
    }

    // ---- CHECK 2: memory-stripped gas EXACTLY linear ----
    auto linCheck=[&](const char* ad, long long G[5], long long MS[5])->bool{
        long long D[5]; for(int i=0;i<5;++i) D[i]=G[i]-memgas(MS[i]);
        long long m01=(D[1]-D[0])/1, m12=(D[2]-D[1])/2, m23=(D[3]-D[2])/4, m34=(D[4]-D[3])/8;
        bool ok=(m01==m12&&m12==m23&&m23==m34)
            &&((D[1]-D[0])%1==0)&&((D[2]-D[1])%2==0)&&((D[3]-D[2])%4==0)&&((D[4]-D[3])%8==0);
        printf("[CHECK 2] %s marginal (memory-stripped): %lld %lld %lld %lld -> %s\n",ad,m01,m12,m23,m34,ok?"EXACTLY CONSTANT, PASSED":"FAILED");
        return ok;
    };
    bool l1=linCheck("SIMD   warm",GB_w,MSB), l2=linCheck("SCALAR warm",GC_w,MSC);
    bool l3=linCheck("SIMD   cold",GB_c,MSB), l4=linCheck("SCALAR cold",GC_c,MSC);
    if(!(l1&&l2&&l3&&l4)) fail=true;

    long long margB=(GB_w[4]-GB_w[0])/15, margC=(GC_w[4]-GC_w[0])/15;
    printf("\n================ BATCH / MARGINAL COST (per image) ================\n");
    printf("  N=1 warm:   SIMD %lld | SCALAR %lld  (previous anchors 5,134,719 / 12,317,165; diff=outer-loop overhead: %+lld / %+lld)\n",
        GB_w[0],GC_w[0],GB_w[0]-5134719LL,GC_w[0]-12317165LL);
    printf("  MARGINAL:   SIMD %lld | SCALAR %lld  ->  RATIO C/B = %.2fx\n",margB,margC,(double)margC/margB);
    printf("  (pure-compute buckets: 2,117,326 / 10,408,685 -> marginal-minus-compute = per-image harness: %+lld / %+lld)\n",
        margB-2117326LL,margC-10408685LL);
    printf("  N=16 amortized total/image: SIMD warm %lld | cold %lld  (cold weight load divided by 16)\n",
        GB_w[4]/16, GB_c[4]/16);
    printf("  N=16 end-to-end C/B: cold %lld/%lld = %.2fx | warm %lld/%lld = %.2fx\n",
        GC_c[4],GB_c[4],(double)GC_c[4]/GB_c[4], GC_w[4],GB_w[4],(double)GC_w[4]/GB_w[4]);
    { long long mBc=(GB_c[4]-GB_c[0])/15, mCc=(GC_c[4]-GC_c[0])/15;   // cold marginal (memory included)
      printf("  linear model G_tx(N) = G_0 + N*marginal (memory included): intercept G_0 = G_tx(1) - marginal\n");
      printf("    SIMD   : cold G_0 %lld (marginal %lld) | warm G_0 %lld (marginal %lld) | cold-warm G_0 diff %lld (= 13,840 slots x 2,000)\n",
          GB_c[0]-mBc, mBc, GB_w[0]-margB, margB, (GB_c[0]-mBc)-(GB_w[0]-margB));
      printf("    SCALAR : cold G_0 %lld (marginal %lld) | warm G_0 %lld (marginal %lld)\n",
          GC_c[0]-mCc, mCc, GC_w[0]-margC, margC); }
    printf("\n%s\n", fail?"*** AT LEAST ONE CHECK FAILED -> do NOT record ***":"*** ALL CHECKS PASSED ***");
    vm->destroy(vm); dlclose(h); return fail?1:0;
}
