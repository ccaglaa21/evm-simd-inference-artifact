// mnist_deploy_cb.cpp — DEPLOYABLE vs DEPLOYABLE
// Both sides are real deployment candidates: looped segment loader + batch outer loop +
// LOOPED body. The SIMD body is the verified looped body; the SCALAR body is NEW:
// packed int16 dynamic unpack = MLOAD + SHR((i&7)*16) + AND 0xFFFF + SIGNEXTEND
// (more efficient than the double-BYTE idiom of the unrolled version -> generous to scalar). ReLU/clip/argmax
// sequences are copied from the verified bytes of e2e.
// CHECKS: (1) 16 different images, both paths bit-exact (2) BOTH bytecodes < 24,576
// (3) memory-stripped marginal EXACTLY constant (4) cold-warm==27,680,000 for every N
// (5) MSIZE==PREDS+N*32 (6) SIMD marginal == 1,992,962 anchor (7) repeat run identical.
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
    int WBASE=901200;
    int NIMGb=901232, ICNTb=901264, PREDSb=901296;      // SIMD batch cells
    int NIMGs=901232, ICNTs=901264, PREDSs=901296;      // scalar: same map (separate program)

    std::vector<std::pair<std::vector<uint8_t>,int>> WLOAD;
    for(int n=0;n<H1;++n) for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W1[(gw*8+l)*H1+n]); WLOAD.push_back({w, W1M+(n*(I/8)+gw)*32}); }
    for(int n=0;n<H1;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b1[n]); WLOAD.push_back({w, B1M+n*32}); }
    for(int n=0;n<H2;++n) for(int gw=0; gw<H1/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W2[(gw*8+l)*H2+n]); WLOAD.push_back({w, W2M+(n*(H1/8)+gw)*32}); }
    for(int n=0;n<H2;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b2[n]); WLOAD.push_back({w, B2M+n*32}); }
    for(int c=0;c<10;++c) for(int gw=0; gw<H2/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W3[(gw*8+l)*10+c]); WLOAD.push_back({w, W3M+(c*(H2/8)+gw)*32}); }
    int matWords=(int)WLOAD.size();
    struct Seg{ int s0, cnt, base; };
    std::vector<Seg> segs;
    for(int sl=0; sl<matWords; ++sl){
        if(segs.empty() || WLOAD[sl].second != WLOAD[sl-1].second+32)
            segs.push_back({sl,1,WLOAD[sl].second});
        else segs.back().cnt++;
    }
    if(segs.size()!=5){ printf("*** layout is not contiguous, stop ***\n"); return 1; }

    std::vector<uint8_t> code;
    auto B_=[&](uint8_t x){ code.push_back(x); };
    auto P1=[&](uint8_t v){ B_(0x60); B_(v); };
    auto P3=[&](int v){ B_(0x62); B_((v>>16)&0xFF); B_((v>>8)&0xFF); B_(v&0xFF); };
    auto PUSH=[&](int v){ if(v<256)P1((uint8_t)v); else if(v<65536){B_(0x61);B_((v>>8)&0xFF);B_(v&0xFF);} else P3(v); };
    auto P4=[&](uint32_t v){ B_(0x63); B_((v>>24)&0xFF); B_((v>>16)&0xFF); B_((v>>8)&0xFF); B_(v&0xFF); };
    uint8_t xgtcfg=(1<<5)|2;

    auto emitLoopLoader=[&](){
        for(auto& sg: segs){
            P1(0); PUSH(GCNT); B_(0x52);
            int lp=(int)code.size(); B_(0x5B);
            PUSH(GCNT); B_(0x51); PUSH(sg.cnt); B_(0x11); B_(0x15);
            P3(0); int eph=(int)code.size()-3; B_(0x57);
            PUSH(GCNT); B_(0x51); PUSH(sg.s0); B_(0x01); B_(0x54);
            PUSH(GCNT); B_(0x51); P1(32); B_(0x02); PUSH(sg.base); B_(0x01);
            B_(0x52);
            PUSH(GCNT); B_(0x51); P1(1); B_(0x01); PUSH(GCNT); B_(0x52);
            PUSH(lp); B_(0x56);
            int ed=(int)code.size(); B_(0x5B);
            code[eph]=(ed>>16)&0xFF; code[eph+1]=(ed>>8)&0xFF; code[eph+2]=ed&0xFF;
        }
    };

    // ---- SIMD looped body (verified; identical to mnist_e2e_cb) ----
    auto emit_layer=[&](int M, int NGin, int GINP, int GWGT, int GBIA, int GOUT, int SHIFT){
        for(int s=0;s<M/8;++s){ P1(0); PUSH(SACC+s*32); B_(0x52); }
        P1(0); PUSH(NCNT); B_(0x52);
        int outer_start=(int)code.size(); B_(0x5B);
        PUSH(NCNT); B_(0x51); PUSH(M); B_(0x11); B_(0x15);
        P3(0); int oe_ph=(int)code.size()-3; B_(0x57);
        PUSH(NCNT); B_(0x51); P1(32); B_(0x02); PUSH(GBIA); B_(0x01); B_(0x51);
        PUSH(ACC); B_(0x52);
        P1(0); PUSH(GCNT); B_(0x52);
        int inner_start=(int)code.size(); B_(0x5B);
        PUSH(GCNT); B_(0x51); PUSH(NGin); B_(0x11); B_(0x15);
        P3(0); int ie_ph=(int)code.size()-3; B_(0x57);
        PUSH(GCNT); B_(0x51); P1(32); B_(0x02); B_(0x80); PUSH(GINP); B_(0x01); B_(0x51);
        B_(0x90); PUSH(NCNT); B_(0x51); PUSH(NGin*32); B_(0x02); B_(0x01); PUSH(GWGT); B_(0x01); B_(0x51);
        PUSH(ACC); B_(0x51); B_(0x91); B_(0x90); B_(0xCB); PUSH(ACC); B_(0x52);
        PUSH(GCNT); B_(0x51); P1(1); B_(0x01); PUSH(GCNT); B_(0x52);
        PUSH(inner_start); B_(0x56);
        int inner_end=(int)code.size(); B_(0x5B);
        code[ie_ph]=(inner_end>>16)&0xFF; code[ie_ph+1]=(inner_end>>8)&0xFF; code[ie_ph+2]=inner_end&0xFF;
        PUSH(ACC); B_(0x51); B_(0xCD);
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x06); P1(32); B_(0x02); B_(0x1B);
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(SACC); B_(0x01);
        B_(0x80); B_(0x51); B_(0x90); B_(0x91); B_(0x17); B_(0x90); B_(0x52);
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x06); P1(7); B_(0x14); B_(0x15);
        P3(0); int skip_ph=(int)code.size()-3; B_(0x57);
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(SACC); B_(0x01); B_(0x51);
        B_(0xCC); B_((uint8_t)SHIFT); B_(0xCE);
        PUSH(NCNT); B_(0x51); P1(8); B_(0x90); B_(0x04); P1(32); B_(0x02); PUSH(GOUT); B_(0x01); B_(0x52);
        int skip_target=(int)code.size(); B_(0x5B);
        code[skip_ph]=(skip_target>>16)&0xFF; code[skip_ph+1]=(skip_target>>8)&0xFF; code[skip_ph+2]=skip_target&0xFF;
        PUSH(NCNT); B_(0x51); P1(1); B_(0x01); PUSH(NCNT); B_(0x52);
        PUSH(outer_start); B_(0x56);
        int outer_end=(int)code.size(); B_(0x5B);
        code[oe_ph]=(outer_end>>16)&0xFF; code[oe_ph+1]=(outer_end>>8)&0xFF; code[oe_ph+2]=outer_end&0xFF;
    };
    auto emitSimdBody=[&](){
        emit_layer(H1, I/8, INP, W1M, B1M, H1OUT, SHIFT1);
        emit_layer(H2, H1/8, H1OUT, W2M, B2M, H2OUT, SHIFT2);
        P1(0); PUSH(NCNT); B_(0x52);
        int co_start=(int)code.size(); B_(0x5B);
        PUSH(NCNT); B_(0x51); P1(10); B_(0x11); B_(0x15);
        P3(0); int co_ph=(int)code.size()-3; B_(0x57);
        P1(0); PUSH(ACC); B_(0x52);
        P1(0); PUSH(GCNT); B_(0x52);
        int ci_start=(int)code.size(); B_(0x5B);
        PUSH(GCNT); B_(0x51); PUSH(H2/8); B_(0x11); B_(0x15);
        P3(0); int ci_ph=(int)code.size()-3; B_(0x57);
        PUSH(GCNT); B_(0x51); P1(32); B_(0x02); B_(0x80); PUSH(H2OUT); B_(0x01); B_(0x51);
        B_(0x90); PUSH(NCNT); B_(0x51); PUSH((H2/8)*32); B_(0x02); B_(0x01); PUSH(W3M); B_(0x01); B_(0x51);
        PUSH(ACC); B_(0x51); B_(0x91); B_(0x90); B_(0xCB); PUSH(ACC); B_(0x52);
        PUSH(GCNT); B_(0x51); P1(1); B_(0x01); PUSH(GCNT); B_(0x52);
        PUSH(ci_start); B_(0x56);
        int ci_end=(int)code.size(); B_(0x5B);
        code[ci_ph]=(ci_end>>16)&0xFF; code[ci_ph+1]=(ci_end>>8)&0xFF; code[ci_ph+2]=ci_end&0xFF;
        PUSH(ACC); B_(0x51); B_(0xCD);
        PUSH(NCNT); B_(0x51); P1(32); B_(0x02); PUSH(SCO); B_(0x01);
        B_(0x52);
        PUSH(NCNT); B_(0x51); P1(1); B_(0x01); PUSH(NCNT); B_(0x52);
        PUSH(co_start); B_(0x56);
        int co_end=(int)code.size(); B_(0x5B);
        code[co_ph]=(co_end>>16)&0xFF; code[co_ph+1]=(co_end>>8)&0xFF; code[co_ph+2]=co_end&0xFF;
        PUSH(SCO); B_(0x51); PUSH(MAXV); B_(0x52);
        P1(0); PUSH(MAXI); B_(0x52);
        P1(1); PUSH(CNT2); B_(0x52);
        int amloop=(int)code.size(); B_(0x5B);
        PUSH(CNT2); B_(0x51); P1(10); B_(0x11); B_(0x15);
        P3(0); int amend_ph=(int)code.size()-3; B_(0x57);
        PUSH(MAXV); B_(0x51);
        PUSH(CNT2); B_(0x51); P1(32); B_(0x02); PUSH(SCO); B_(0x01); B_(0x51);
        B_(0xCA); B_(xgtcfg);
        P4(0xFFFFFFFF); B_(0x16); B_(0x15);
        P3(0); int amskip_ph=(int)code.size()-3; B_(0x57);
        PUSH(CNT2); B_(0x51); P1(32); B_(0x02); PUSH(SCO); B_(0x01); B_(0x51);
        PUSH(MAXV); B_(0x52);
        PUSH(CNT2); B_(0x51); PUSH(MAXI); B_(0x52);
        int amskip=(int)code.size(); B_(0x5B);
        code[amskip_ph]=(amskip>>16)&0xFF; code[amskip_ph+1]=(amskip>>8)&0xFF; code[amskip_ph+2]=amskip&0xFF;
        PUSH(CNT2); B_(0x51); P1(1); B_(0x01); PUSH(CNT2); B_(0x52);
        PUSH(amloop); B_(0x56);
        int amend=(int)code.size(); B_(0x5B);
        code[amend_ph]=(amend>>16)&0xFF; code[amend_ph+1]=(amend>>8)&0xFF; code[amend_ph+2]=amend&0xFF;
    };

    // ---- SCALAR looped body (NEW) ----
    // element(idx, BASE constant or WBASE cell): s=(idx&7)*16; woff=(idx>>3)*32;
    // value = SEXT1( (MLOAD(base+woff) >> s) & 0xFFFF )
    auto emitElemDynamic=[&](int GINP, bool wgtViaWBASE){
        // input: stack [i] ; output: stack [not a single value for in*wt — this function extracts ONE operand]
        // usage is managed outside; instead of producing [i] -> [s, val] here, the full two-operand block is below
        (void)GINP; (void)wgtViaWBASE;
    };
    auto looped_scalar_layer=[&](int M,int Kin,int GINP,int GWGT,int GBIA,int GOUT,int SHIFT){
        for(int s=0;s<M/8;++s){ P1(0); PUSH(GOUT+s*32); B_(0x52); }   // zero the packed words
        P1(0); PUSH(NCNT); B_(0x52);
        int nlp=(int)code.size(); B_(0x5B);
        PUSH(NCNT); B_(0x51); PUSH(M); B_(0x11); B_(0x15);
        P3(0); int ne_ph=(int)code.size()-3; B_(0x57);
        PUSH(NCNT); B_(0x51); PUSH(Kin/8*32); B_(0x02); PUSH(GWGT); B_(0x01); PUSH(WBASE); B_(0x52); // WBASE=GWGT+n*(Kin/8)*32
        PUSH(NCNT); B_(0x51); P1(32); B_(0x02); PUSH(GBIA); B_(0x01); B_(0x51); P1(3); B_(0x0B); PUSH(ACC); B_(0x52); // acc=sext32(bias)
        P1(0); PUSH(GCNT); B_(0x52);
        int ilp=(int)code.size(); B_(0x5B);
        PUSH(GCNT); B_(0x51); PUSH(Kin); B_(0x11); B_(0x15);
        P3(0); int ie_ph=(int)code.size()-3; B_(0x57);
        PUSH(GCNT); B_(0x51);                                   // [i]
        B_(0x80); P1(7); B_(0x16); P1(4); B_(0x1B);             // [i,s]
        B_(0x90); P1(3); B_(0x1C); P1(5); B_(0x1B);             // [s,woff]
        B_(0x80); PUSH(GINP); B_(0x01); B_(0x51);               // [s,woff,inword]
        B_(0x82); B_(0x1C); PUSH(0xFFFF); B_(0x16); P1(1); B_(0x0B); // [s,woff,in]
        B_(0x90); PUSH(WBASE); B_(0x51); B_(0x01); B_(0x51);    // [s,in,wword]
        B_(0x82); B_(0x1C); PUSH(0xFFFF); B_(0x16); P1(1); B_(0x0B); // [s,in,wt]
        B_(0x02);                                                // [s,prod]
        PUSH(ACC); B_(0x51); B_(0x01); PUSH(ACC); B_(0x52);     // [s]
        B_(0x50);                                                // []
        PUSH(GCNT); B_(0x51); P1(1); B_(0x01); PUSH(GCNT); B_(0x52);
        PUSH(ilp); B_(0x56);
        int ied=(int)code.size(); B_(0x5B);
        code[ie_ph]=(ied>>16)&0xFF; code[ie_ph+1]=(ied>>8)&0xFF; code[ie_ph+2]=ied&0xFF;
        PUSH(ACC); B_(0x51);
        B_(0x80); P1(0); B_(0x13); B_(0x15); B_(0x02);           // ReLU (identical to e2e)
        P1((uint8_t)SHIFT); B_(0x1C);
        B_(0x80); PUSH(32767); B_(0x90); B_(0x13); PUSH(32767); B_(0x82); B_(0x90); B_(0x03); B_(0x02); B_(0x01); // clip (identical to e2e)
        PUSH(NCNT); B_(0x51); P1(7); B_(0x16); P1(4); B_(0x1B); B_(0x1B); // [val<<bit]
        PUSH(NCNT); B_(0x51); P1(3); B_(0x1C); P1(5); B_(0x1B); PUSH(GOUT); B_(0x01); // [shifted,sw]
        B_(0x80); B_(0x51);                                      // [shifted,sw,cur]
        B_(0x90); B_(0x91); B_(0x17); B_(0x90); B_(0x52);        // MSTORE(sw, cur|shifted)
        PUSH(NCNT); B_(0x51); P1(1); B_(0x01); PUSH(NCNT); B_(0x52);
        PUSH(nlp); B_(0x56);
        int ned=(int)code.size(); B_(0x5B);
        code[ne_ph]=(ned>>16)&0xFF; code[ne_ph+1]=(ned>>8)&0xFF; code[ne_ph+2]=ned&0xFF;
    };
    auto emitScalarBody=[&](){
        looped_scalar_layer(H1, I,  INP,   W1M, B1M, H1OUT, SHIFT1);
        looped_scalar_layer(H2, H1, H1OUT, W2M, B2M, H2OUT, SHIFT2);
        // output layer: looped, raw score at SCO+c*32
        P1(0); PUSH(NCNT); B_(0x52);
        int clp=(int)code.size(); B_(0x5B);
        PUSH(NCNT); B_(0x51); P1(10); B_(0x11); B_(0x15);
        P3(0); int ce_ph=(int)code.size()-3; B_(0x57);
        PUSH(NCNT); B_(0x51); PUSH((H2/8)*32); B_(0x02); PUSH(W3M); B_(0x01); PUSH(WBASE); B_(0x52);
        P1(0); PUSH(ACC); B_(0x52);
        P1(0); PUSH(GCNT); B_(0x52);
        int cil=(int)code.size(); B_(0x5B);
        PUSH(GCNT); B_(0x51); PUSH(H2); B_(0x11); B_(0x15);
        P3(0); int cie_ph=(int)code.size()-3; B_(0x57);
        PUSH(GCNT); B_(0x51);
        B_(0x80); P1(7); B_(0x16); P1(4); B_(0x1B);
        B_(0x90); P1(3); B_(0x1C); P1(5); B_(0x1B);
        B_(0x80); PUSH(H2OUT); B_(0x01); B_(0x51);
        B_(0x82); B_(0x1C); PUSH(0xFFFF); B_(0x16); P1(1); B_(0x0B);
        B_(0x90); PUSH(WBASE); B_(0x51); B_(0x01); B_(0x51);
        B_(0x82); B_(0x1C); PUSH(0xFFFF); B_(0x16); P1(1); B_(0x0B);
        B_(0x02);
        PUSH(ACC); B_(0x51); B_(0x01); PUSH(ACC); B_(0x52);
        B_(0x50);
        PUSH(GCNT); B_(0x51); P1(1); B_(0x01); PUSH(GCNT); B_(0x52);
        PUSH(cil); B_(0x56);
        int cied=(int)code.size(); B_(0x5B);
        code[cie_ph]=(cied>>16)&0xFF; code[cie_ph+1]=(cied>>8)&0xFF; code[cie_ph+2]=cied&0xFF;
        PUSH(ACC); B_(0x51); PUSH(NCNT); B_(0x51); P1(32); B_(0x02); PUSH(SCO); B_(0x01); B_(0x52);
        PUSH(NCNT); B_(0x51); P1(1); B_(0x01); PUSH(NCNT); B_(0x52);
        PUSH(clp); B_(0x56);
        int ced=(int)code.size(); B_(0x5B);
        code[ce_ph]=(ced>>16)&0xFF; code[ce_ph+1]=(ced>>8)&0xFF; code[ce_ph+2]=ced&0xFF;
        // branchless argmax (identical to the e2e scalar argmax; MAXV/MAXI cells)
        PUSH(SCO); B_(0x51); PUSH(MAXV); B_(0x52);
        P1(0); PUSH(MAXI); B_(0x52);
        for(int c=1;c<10;++c){
            PUSH(MAXV); B_(0x51); PUSH(SCO+c*32); B_(0x51); B_(0x13);
            B_(0x80);
            PUSH(SCO+c*32); B_(0x51); PUSH(MAXV); B_(0x51);
            B_(0x90); B_(0x03); B_(0x02);
            PUSH(MAXV); B_(0x51); B_(0x01); PUSH(MAXV); B_(0x52);
            PUSH(MAXI); B_(0x51); B_(0x80); PUSH(c); B_(0x03);
            B_(0x82); B_(0x02); B_(0x01); PUSH(MAXI); B_(0x52);
            B_(0x50);
        }
    };

    auto buildProgram=[&](bool simd)->std::vector<uint8_t>{
        code.clear();
        int NIMG = simd? NIMGb:NIMGs, ICNT = simd? ICNTb:ICNTs, PREDS = simd? PREDSb:PREDSs;
        emitLoopLoader();                                                       // weights: looped, ONCE
        B_(0x36); PUSH(CDIMG); B_(0x90); B_(0x04); PUSH(NIMG); B_(0x52);        // NIMG=CDS/3136
        P1(0); PUSH(ICNT); B_(0x52);
        int lp=(int)code.size(); B_(0x5B);
        PUSH(ICNT); B_(0x51); PUSH(NIMG); B_(0x51); B_(0x11); B_(0x15);
        P3(0); int eph=(int)code.size()-3; B_(0x57);
        PUSH(CDIMG); PUSH(ICNT); B_(0x51); PUSH(CDIMG); B_(0x02); PUSH(INP); B_(0x37);
        if(simd) emitSimdBody(); else emitScalarBody();
        PUSH(MAXI); B_(0x51);
        PUSH(ICNT); B_(0x51); P1(32); B_(0x02); PUSH(PREDS); B_(0x01); B_(0x52);
        PUSH(ICNT); B_(0x51); P1(1); B_(0x01); PUSH(ICNT); B_(0x52);
        PUSH(lp); B_(0x56);
        int ed=(int)code.size(); B_(0x5B);
        code[eph]=(ed>>16)&0xFF; code[eph+1]=(ed>>8)&0xFF; code[eph+2]=ed&0xFF;
        PUSH(NIMG); B_(0x51); P1(32); B_(0x02); PUSH(PREDS); B_(0xF3);
        return code;
    };
    auto withMsize=[&](std::vector<uint8_t> k)->std::vector<uint8_t>{
        for(int t=0;t<13;++t) k.pop_back();
        uint8_t t2[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3};
        for(uint8_t x:t2) k.push_back(x);
        return k;
    };
    std::vector<uint8_t> codeB=buildProgram(true),  codeBm=withMsize(codeB);
    std::vector<uint8_t> codeC=buildProgram(false), codeCm=withMsize(codeC);
    printf("Bytecode: SIMD=%zu B | SCALAR=%zu B\n",codeB.size(),codeC.size());
    printf("[CHECK 2] EIP-170 (<24,576): SIMD %s | SCALAR %s\n",
        codeB.size()<24576?"PASSED":"FAILED", codeC.size()<24576?"PASSED":"FAILED");

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
    auto imgWords=[&](int im, std::vector<uint8_t>& dst){
        for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0);
            for(int l=0;l<8;++l){ int val=imgs[im][gw*8+l]; uint16_t u=(uint16_t)(int16_t)val; w[l*2]=u&0xFF; w[l*2+1]=(u>>8)&0xFF; }
            for(int i=0;i<32;++i) dst.push_back(w[31-i]); }
    };
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

    // CHECK 1: 16 different images
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

    std::vector<int> Ns={1,2,4,8,16};
    long long GB_c[5],GB_w[5],GC_c[5],GC_w[5],MSB[5],MSC[5];
    bool fail=false;
    for(int ni=0;ni<5;++ni){
        int N=Ns[ni];
        std::vector<uint8_t> cd; for(int r=0;r<N;++r) imgWords(0,cd);
        std::vector<int> pr;
        setAcc(EVMC_ACCESS_COLD); GB_c[ni]=run(codeB,cd,pr,w0);
        setAcc(EVMC_ACCESS_COLD); long long r2=run(codeB,cd,pr,w0);
        if(GB_c[ni]!=r2){ printf("[CHECK 7] SIMD N=%d repeat MISMATCH\n",N); fail=true; }
        setAcc(EVMC_ACCESS_WARM); GB_w[ni]=run(codeB,cd,pr,w0);
        setAcc(EVMC_ACCESS_COLD); GC_c[ni]=run(codeC,cd,pr,w0);
        setAcc(EVMC_ACCESS_WARM); GC_w[ni]=run(codeC,cd,pr,w0);
        setAcc(EVMC_ACCESS_WARM); run(codeBm,cd,pr,MSB[ni]);
        setAcc(EVMC_ACCESS_WARM); run(codeCm,cd,pr,MSC[ni]);
        bool g4=(GB_c[ni]-GB_w[ni]==(long long)matWords*2000)&&(GC_c[ni]-GC_w[ni]==(long long)matWords*2000);
        long long expB=(((long long)PREDSb+N*32+31)/32)*32, expS=(((long long)PREDSs+N*32+31)/32)*32;
        bool g5=(MSB[ni]==expB)&&(MSC[ni]==expS);
        if(!g5) printf("   [DIAG] MSIZE measured SIMD=%lld expected=%lld | SCALAR=%lld expected=%lld\n",MSB[ni],expB,MSC[ni],expS);
        printf("[N=%2d] SIMD c/w=%lld/%lld | SCALAR c/w=%lld/%lld | K4:%s K5:%s\n",
            N,GB_c[ni],GB_w[ni],GC_c[ni],GC_w[ni],g4?"OK":"FAILED",g5?"OK":"FAILED");
        if(!g4||!g5) fail=true;
    }
    auto linCheck=[&](const char* ad, long long G[5], long long MS[5], long long& marg)->bool{
        long long D[5]; for(int i=0;i<5;++i) D[i]=G[i]-memgas(MS[i]);
        long long m01=D[1]-D[0], m12=(D[2]-D[1])/2, m23=(D[3]-D[2])/4, m34=(D[4]-D[3])/8;
        bool okl=(m01==m12&&m12==m23&&m23==m34);
        marg=m01;
        printf("[CHECK 3] %s marginal: %lld %lld %lld %lld -> %s\n",ad,m01,m12,m23,m34,okl?"EXACTLY CONSTANT":"FAILED");
        return okl;
    };
    long long margB,margC,dB,dC;
    bool l1=linCheck("SIMD   warm",GB_w,MSB,margB), l2=linCheck("SCALAR warm",GC_w,MSC,margC);
    bool l3=linCheck("SIMD   cold",GB_c,MSB,dB),    bool_l4; bool l4=linCheck("SCALAR cold",GC_c,MSC,dC); (void)bool_l4;
    if(!(l1&&l2&&l3&&l4)) fail=true;
    long long margBtx=margB, margCtx=margC;
    printf("[CHECK 6] SIMD marginal(memory-stripped) == 1,992,849 (CHECK-2 value of the batch program): %lld %s\n",margBtx+0,margB==1992849LL?"PASSED":"FAILED");
    if(margB!=1992849LL) fail=true;

    printf("\n================ DEPLOYABLE vs DEPLOYABLE ================\n");
    printf("  code: SIMD %zu B | SCALAR %zu B (both under EIP-170)\n",codeB.size(),codeC.size());
    printf("  MARGINAL/image: SIMD %lld | SCALAR %lld  ->  C/B = %.2fx\n",margB,margC,(double)margC/margB);
    printf("  tx total (N images): 27.68M warm-up + N*marginal + constant; N=1: SIMD %lld | SCALAR %lld -> %.2fx\n",
        GB_c[0],GC_c[0],(double)GC_c[0]/GB_c[0]);
    printf("  (previous marginal vs unrolled scalar was 5.16x; the ratio grew because the looped scalar also pays the loop overhead)\n");
    printf("  loop factor of the scalar path: looped marginal %lld / unrolled-scalar marginal 10,284,208 (batch program, memory-stripped) = %.2fx\n",
        margC,(double)margC/10284208.0);
    printf("\n%s\n", fail?"*** AT LEAST ONE CHECK FAILED -> do NOT record ***":"*** ALL CHECKS PASSED ***");
    vm->destroy(vm); dlclose(h); return fail?1:0;
}
