// mnist_loader_loop.cpp — LOOPED WEIGHT LOADER vs UNROLLED (answer to the EIP-170 gap)
// Question: instead of the unrolled 13840x(PUSH32+SLOAD+PUSH+MSTORE) loader (~537KB), a looped
// loader: does the code size drop below 24,576B (EIP-170), and what is the gas price?
// Design: contiguous segments are DERIVED by scanning the WLOAD layout (not assumed);
// the program stops unless there are exactly 5 contiguous segments. Counter=GCNT (reused) -> same MSIZE.
// CHECKS: (1) 20/20 three-way prediction (both variants) (2) coldU==32,814,719 (mnist_e2e_cb anchor)
//   (3) cold-warm==matWords*2000 (both variants; proof that the SLOAD COUNT is preserved)
//   (4) compute_cold==compute_warm (5) MSIZE U==L (6) repeat run identical.
// BONUS: tx-level accounting — intrinsic 21000 + calldata gas from the REAL bytes
// (EIP-2028: zero byte 4, non-zero 16; protocol constants).
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

    // MEMORY map (SAME as mnist_e2e_cb)
    int INP=0, W1M=10000, B1M=420000, H1OUT=430000, W2M=440000, B2M=480000, H2OUT=490000;
    int W3M=495000, SCO=500000;
    int ACC=900000, SACC=900100, NCNT=901000, GCNT=901100;
    int MAXV=900200, MAXI=900232, CNT2=900264;

    // WLOAD (identical to mnist_sload/e2e)
    std::vector<std::pair<std::vector<uint8_t>,int>> WLOAD;
    for(int n=0;n<H1;++n) for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W1[(gw*8+l)*H1+n]); WLOAD.push_back({w, W1M+(n*(I/8)+gw)*32}); }
    for(int n=0;n<H1;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b1[n]); WLOAD.push_back({w, B1M+n*32}); }
    for(int n=0;n<H2;++n) for(int gw=0; gw<H1/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W2[(gw*8+l)*H2+n]); WLOAD.push_back({w, W2M+(n*(H1/8)+gw)*32}); }
    for(int n=0;n<H2;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b2[n]); WLOAD.push_back({w, B2M+n*32}); }
    for(int c=0;c<10;++c) for(int gw=0; gw<H2/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W3[(gw*8+l)*10+c]); WLOAD.push_back({w, W3M+(c*(H2/8)+gw)*32}); }
    int matWords=(int)WLOAD.size();

    // SEGMENT DERIVATION (scan, not assumption): contiguous dest runs
    struct Seg{ int s0, cnt, base; };
    std::vector<Seg> segs;
    for(int sl=0; sl<matWords; ++sl){
        if(segs.empty() || WLOAD[sl].second != WLOAD[sl-1].second+32)
            segs.push_back({sl,1,WLOAD[sl].second});
        else segs.back().cnt++;
    }
    printf("Segment scan: %zu segments found (expected 5)\n",segs.size());
    for(auto&s:segs) printf("  slot %5d..%5d -> mem %d..%d\n",s.s0,s.s0+s.cnt-1,s.base,s.base+(s.cnt-1)*32);
    if(segs.size()!=5){ printf("*** LAYOUT NOT CONTIGUOUS -> looped loader INVALID as is, stop ***\n"); return 1; }

    // ---- shared emit machinery ----
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

    auto emitBody=[&](bool loopLoader)->std::pair<size_t,size_t>{  // (loaderStart,loaderEnd)
        PUSH(I/8*32); P1(0); PUSH(INP); code.push_back(0x37);      // CALLDATACOPY
        size_t ls=code.size();
        if(!loopLoader){
            auto PUSHslot=[&](uint32_t sl){ code.push_back(0x7F); for(int i=0;i<28;++i) code.push_back(0); code.push_back((sl>>24)&0xFF); code.push_back((sl>>16)&0xFF); code.push_back((sl>>8)&0xFF); code.push_back(sl&0xFF); };
            for(int sl=0; sl<matWords; ++sl){ PUSHslot((uint32_t)sl); code.push_back(0x54); PUSH(WLOAD[sl].second); code.push_back(0x52); }
        } else {
            for(auto& sg: segs){
                P1(0); PUSH(GCNT); code.push_back(0x52);                        // i=0
                int lp=(int)code.size(); code.push_back(0x5B);
                PUSH(GCNT); code.push_back(0x51); PUSH(sg.cnt); code.push_back(0x11); code.push_back(0x15);
                P3(0); int eph=(int)code.size()-3; code.push_back(0x57);
                PUSH(GCNT); code.push_back(0x51); PUSH(sg.s0); code.push_back(0x01); code.push_back(0x54);   // SLOAD(s0+i)
                PUSH(GCNT); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(sg.base); code.push_back(0x01);
                code.push_back(0x52);                                            // MSTORE
                PUSH(GCNT); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(GCNT); code.push_back(0x52);
                PUSH(lp); code.push_back(0x56);
                int ed=(int)code.size(); code.push_back(0x5B);
                code[eph]=(ed>>16)&0xFF; code[eph+1]=(ed>>8)&0xFF; code[eph+2]=ed&0xFF;
            }
        }
        size_t le=code.size();
        emit_layer(H1, I/8, INP, W1M, B1M, H1OUT, ACC, SACC, NCNT, GCNT, SHIFT1);
        emit_layer(H2, H1/8, H1OUT, W2M, B2M, H2OUT, ACC, SACC, NCNT, GCNT, SHIFT2);
        // output + argmax (identical to mnist_e2e_cb)
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
        return {ls,le};
    };

    code.clear(); auto luU=emitBody(false); std::vector<uint8_t> codeU=code;
    code.clear(); auto luL=emitBody(true);  std::vector<uint8_t> codeL=code;
    auto withMsize=[&](const std::vector<uint8_t>& body)->std::vector<uint8_t>{
        std::vector<uint8_t> k(body.begin(),body.end()-13);
        uint8_t t[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3};
        for(uint8_t x:t) k.push_back(x); return k;
    };
    std::vector<uint8_t> codeUm=withMsize(codeU), codeLm=withMsize(codeL);
    printf("\nBytecode: UNROLLED=%zu bytes (loader %zu) | LOOPED=%zu bytes (loader %zu)\n",
        codeU.size(), luU.second-luU.first, codeL.size(), luL.second-luL.first);
    printf("[EIP-170] looped variant under 24,576B: %s (%zu bytes)\n",
        codeL.size()<24576?"YES":"NO", codeL.size());

    // host + storage
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
    const int64_t GBIG=20000000000LL;

    auto buildCalldata=[&](int im, std::vector<uint8_t>& calldata, int& cpp_pred){
        std::vector<int>& img=imgs[im];
        std::vector<int> h1r(H1); for(int n=0;n<H1;++n){ long long s=b1[n]; for(int i=0;i<I;++i) s+=(long long)img[i]*W1[i*H1+n]; h1r[n]=rss(s,SHIFT1);}
        std::vector<int> h2r(H2); for(int n=0;n<H2;++n){ long long s=b2[n]; for(int i=0;i<H1;++i) s+=(long long)h1r[i]*W2[i*H2+n]; h2r[n]=rss(s,SHIFT2);}
        std::vector<long long> scr(10); for(int cc=0;cc<10;++cc){ long long s=0; for(int i=0;i<H2;++i) s+=(long long)h2r[i]*W3[i*10+cc]; scr[cc]=s;}
        cpp_pred=(int)(std::max_element(scr.begin(),scr.end())-scr.begin());
        calldata.assign(I/8*32,0);
        for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l){ int val=img[gw*8+l]; uint16_t u=(uint16_t)(int16_t)val; w[l*2]=u&0xFF; w[l*2+1]=(u>>8)&0xFF;} for(int i=0;i<32;++i) calldata[gw*32+i]=w[31-i]; }
    };
    auto run=[&](std::vector<uint8_t>& k, std::vector<uint8_t>& cd, int& pred, long long& w0)->long long{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GBIG; msg.recipient=ADDR; msg.code_address=ADDR;
        msg.input_data=cd.data(); msg.input_size=cd.size();
        auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,k.data(),k.size());
        long long g=GBIG-res.gas_left; pred=-1; w0=-1;
        if(res.status_code==0&&res.output_data&&res.output_size>=32){
            pred=((int32_t)res.output_data[28]<<24)|((int32_t)res.output_data[29]<<16)|((int32_t)res.output_data[30]<<8)|(int32_t)res.output_data[31];
            long long m=0; for(int b=24;b<32;++b) m=(m<<8)|res.output_data[b]; w0=m;
        }
        if(res.release)res.release(&res); return g;
    };
    auto memgas=[](long long ms)->long long{ long long w=(ms+31)/32; return 3*w+(w*w)/512; };

    // CHECK 1
    int ok=0; long long w0;
    for(int im=0; im<20; ++im){
        std::vector<uint8_t> cd; int cp,pU,pL;
        buildCalldata(im,cd,cp);
        setAcc(EVMC_ACCESS_WARM); run(codeU,cd,pU,w0);
        setAcc(EVMC_ACCESS_WARM); run(codeL,cd,pL,w0);
        if(pU==cp&&pL==cp) ok++;
        else printf("  [MISMATCH] im=%d C++=%d U=%d L=%d\n",im,cp,pU,pL);
    }
    printf("[CHECK 1] three-way prediction (C++ == UNROLLED == LOOPED): %d/20 %s\n",ok,ok==20?"PASSED":"FAILED");
    if(ok!=20){ vm->destroy(vm); dlclose(h); return 1; }

    std::vector<uint8_t> cd0; int cp0,dummy; buildCalldata(0,cd0,cp0);
    long long msU,msL;
    setAcc(EVMC_ACCESS_COLD); run(codeUm,cd0,dummy,msU);
    setAcc(EVMC_ACCESS_COLD); run(codeLm,cd0,dummy,msL);
    printf("[CHECK 5] MSIZE U==L: %lld/%lld %s\n",msU,msL,msU==msL?"PASSED":"FAILED");
    long long mem=memgas(msU);

    long long cU1,cU2,wU,cL1,cL2,wL;
    setAcc(EVMC_ACCESS_COLD); cU1=run(codeU,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); cU2=run(codeU,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_WARM); wU =run(codeU,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); cL1=run(codeL,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_COLD); cL2=run(codeL,cd0,dummy,w0);
    setAcc(EVMC_ACCESS_WARM); wL =run(codeL,cd0,dummy,w0);
    printf("[CHECK 6] repeat run identical: U %s, L %s\n",cU1==cU2?"OK":"MISMATCH",cL1==cL2?"OK":"MISMATCH");
    printf("[CHECK 2] coldU == 32,814,719 (mnist_e2e_cb anchor): %lld %s\n",cU1,cU1==32814719LL?"PASSED":"FAILED");
    printf("[CHECK 3] cold-warm==matWords*2000 (%lld): U %lld %s | L %lld %s\n",
        (long long)matWords*2000, cU1-wU,(cU1-wU==(long long)matWords*2000)?"OK":"MISMATCH",
        cL1-wL,(cL1-wL==(long long)matWords*2000)?"OK":"MISMATCH");
    long long sc=(long long)matWords*2100, sw=(long long)matWords*100;
    long long hU=cU1-sc-mem, hUw=wU-sw-mem, hL=cL1-sc-mem, hLw=wL-sw-mem;
    printf("[CHECK 4] compute_cold==compute_warm: U %lld/%lld %s | L %lld/%lld %s\n",
        hU,hUw,hU==hUw?"OK":"MISMATCH",hL,hLw,hL==hLw?"OK":"MISMATCH");

    // tx-level accounting (EIP-2028: zero 4 / non-zero 16; intrinsic 21000)
    long long zb=0,nzb=0; for(uint8_t b:cd0){ if(b==0)zb++; else nzb++; }
    long long cdgas=zb*4+nzb*16, intr=21000;
    printf("\n================ UNROLLED vs LOOPED LOADER ================\n");
    printf("  variant  |  code size  |  cold TOTAL  |  warm TOTAL  |  compute\n");
    printf(" UNROLLED  | %10zu | %12lld | %12lld | %10lld\n",codeU.size(),cU1,wU,hU);
    printf("  LOOPED   | %10zu | %12lld | %12lld | %10lld\n",codeL.size(),cL1,wL,hL);
    printf("  DIFF     | %+10lld | %+12lld | %+12lld | %+10lld  (loop overhead)\n",
        (long long)codeL.size()-(long long)codeU.size(), cL1-cU1, wL-wU, hL-hU);
    printf("  loop overhead per slot: %.2f gas | share of the cold total: %.2f%%\n",
        (double)(cL1-cU1)/matWords, 100.0*(cL1-cU1)/cL1);
    printf("\n  TX LEVEL (image 0 calldata: %lld zero + %lld non-zero bytes):\n",zb,nzb);
    printf("  intrinsic 21,000 + calldata %lld = %lld extra; LOOPED tx total: cold %lld | warm %lld\n",
        cdgas, intr+cdgas, cL1+intr+cdgas, wL+intr+cdgas);
    bool all = ok==20 && msU==msL && cU1==cU2 && cL1==cL2 && cU1==32814719LL
        && (cU1-wU==(long long)matWords*2000) && (cL1-wL==(long long)matWords*2000)
        && hU==hUw && hL==hLw;
    printf("\n%s\n", all?"*** ALL CHECKS PASSED ***":"*** AT LEAST ONE CHECK FAILED -> do NOT record ***");
    vm->destroy(vm); dlclose(h); return all?0:1;
}
