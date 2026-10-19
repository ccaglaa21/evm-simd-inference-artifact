// mnist_layers.cpp — MNIST layer-by-layer COMPUTE gas measurement (prefix method).
// mnist_sload.cpp was used as the TEMPLATE, the ORIGINAL is UNTOUCHED. emit_layer + argmax are IDENTICAL copies.
// 4 prefixes: P1=load-only | P2=+layer1 | P3=+layer2 | P4=full(argmax).
// Each prefix measures its own MSIZE (0x59) separately. compute=total-SLOAD-memgas(MSIZE).
// layer isolation by subtraction: k1=hP2-hP1, k2=hP3-hP2, output=hP4-hP3.
// CHECKS: (1)P4 prediction==ref (2)intermediate h1/h2==ref (3)layer>0 (4)cold-warm==mW*2000 (5)h_cold==h_warm.
#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>

static void put16(std::vector<uint8_t>& w,int lane,int val){ uint16_t u=(uint16_t)(int16_t)val; w[lane*2]=u&0xFF; w[lane*2+1]=(u>>8)&0xFF; }
static void put32(std::vector<uint8_t>& w,int lane,long long val){ uint32_t u=(uint32_t)(int32_t)val; for(int b=0;b<4;++b) w[lane*4+b]=(u>>(8*b))&0xFF; }

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
    int N_ALL=500;
    std::vector<std::vector<int>> imgs(N_ALL, std::vector<int>(I));
    for(int im=0; im<N_ALL; ++im){ int tlb,pfo; bf>>tlb>>pfo; double tmp; for(int i=0;i<10;++i) bf>>tmp; for(int i=0;i<I;++i) bf>>imgs[im][i]; }
    std::ifstream rf("deep_ref.txt"); int Nr; rf>>Nr;
    std::vector<int> ref_pred(N_ALL), true_label(N_ALL);
    for(int im=0; im<N_ALL; ++im){ int tl,pd; rf>>tl>>pd; ref_pred[im]=pd; true_label[im]=tl; }

    auto rss=[&](long long v,int sh)->int{ if(v<0)v=0; v>>=sh; if(v>32767)v=32767; return (int)v; };

    std::vector<uint8_t> code;
    auto P32=[&](const std::vector<uint8_t>& w){ code.push_back(0x7F); for(int i=31;i>=0;--i) code.push_back(w[i]); };
    auto P1=[&](uint8_t v){ code.push_back(0x60); code.push_back(v); };
    auto P3=[&](int v){ code.push_back(0x62); code.push_back((v>>16)&0xFF); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); };
    auto PUSH=[&](int v){ if(v<256)P1((uint8_t)v); else if(v<65536){code.push_back(0x61);code.push_back((v>>8)&0xFF);code.push_back(v&0xFF);} else P3(v); };
    auto P4=[&](uint32_t v){ code.push_back(0x63); code.push_back((v>>24)&0xFF); code.push_back((v>>16)&0xFF); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); };
    uint8_t xgtcfg=(1<<5)|2;

    // ===== emit_layer: IDENTICAL to mnist_sload.cpp =====
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

    // ===== MEMORY ADDRESSES (same as mnist_sload) =====
    int INP=0, W1M=10000, B1M=420000, H1OUT=430000, W2M=440000, B2M=480000, H2OUT=490000;
    int W3M=495000, SCO=500000;
    int ACC=900000, SACC=900100, NCNT=901000, GCNT=901100;

    PUSH(I/8*32); P1(0); PUSH(INP); code.push_back(0x37); // CALLDATACOPY input

    std::vector<std::pair<std::vector<uint8_t>,int>> WLOAD;
    for(int n=0;n<H1;++n) for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W1[(gw*8+l)*H1+n]); WLOAD.push_back({w, W1M+(n*(I/8)+gw)*32}); }
    for(int n=0;n<H1;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b1[n]); WLOAD.push_back({w, B1M+n*32}); }
    for(int n=0;n<H2;++n) for(int gw=0; gw<H1/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W2[(gw*8+l)*H2+n]); WLOAD.push_back({w, W2M+(n*(H1/8)+gw)*32}); }
    for(int n=0;n<H2;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b2[n]); WLOAD.push_back({w, B2M+n*32}); }
    for(int c=0;c<10;++c) for(int gw=0; gw<H2/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W3[(gw*8+l)*10+c]); WLOAD.push_back({w, W3M+(c*(H2/8)+gw)*32}); }
    int matWords=(int)WLOAD.size();

    auto PUSHslot=[&](uint32_t sl){ code.push_back(0x7F); for(int i=0;i<28;++i) code.push_back(0); code.push_back((sl>>24)&0xFF); code.push_back((sl>>16)&0xFF); code.push_back((sl>>8)&0xFF); code.push_back(sl&0xFF); };
    for(int sl=0; sl<matWords; ++sl){ PUSHslot((uint32_t)sl); code.push_back(0x54); PUSH(WLOAD[sl].second); code.push_back(0x52); }
    int cp_load=(int)code.size();   // <<< CHECKPOINT: end of load

    emit_layer(H1, I/8, INP, W1M, B1M, H1OUT, ACC, SACC, NCNT, GCNT, SHIFT1);
    int cp_l1=(int)code.size();     // <<< CHECKPOINT: end of layer1

    emit_layer(H2, H1/8, H1OUT, W2M, B2M, H2OUT, ACC, SACC, NCNT, GCNT, SHIFT2);
    int cp_l2=(int)code.size();     // <<< CHECKPOINT: end of layer2

    // ===== OUTPUT LAYER + ARGMAX: IDENTICAL to mnist_sload =====
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
    int MAXV=900200, MAXI=900232, CNT2=900264;
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
    int cp_end=(int)code.size();    // <<< CHECKPOINT: end of full code (including RETURN)

    // ===== STORAGE =====
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

    // prefix bytecode generator
    auto pushInto=[](std::vector<uint8_t>& v, int val){
        if(val<256){ v.push_back(0x60); v.push_back((uint8_t)val); }
        else if(val<65536){ v.push_back(0x61); v.push_back((val>>8)&0xFF); v.push_back(val&0xFF); }
        else { v.push_back(0x62); v.push_back((val>>16)&0xFF); v.push_back((val>>8)&0xFF); v.push_back(val&0xFF); }
    };
    auto makeReturn=[&](int upto, int off, int size)->std::vector<uint8_t>{
        std::vector<uint8_t> v(code.begin(), code.begin()+upto);
        pushInto(v,size); pushInto(v,off); v.push_back(0xF3); return v;
    };
    auto makeMsize=[&](int upto)->std::vector<uint8_t>{
        std::vector<uint8_t> v(code.begin(), code.begin()+upto);
        uint8_t tail[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3}; for(uint8_t x:tail) v.push_back(x); return v;
    };
    // P4 full code; MSIZE version = MSIZE tail instead of the last 13 bytes (prediction return)
    std::vector<uint8_t> bcP1=makeReturn(cp_load,0,32);
    std::vector<uint8_t> bcP2=makeReturn(cp_l1,H1OUT,(H1/8)*32);
    std::vector<uint8_t> bcP3=makeReturn(cp_l2,H2OUT,(H2/8)*32);
    std::vector<uint8_t> bcP4(code.begin(), code.begin()+cp_end);
    std::vector<uint8_t> msP1=makeMsize(cp_load), msP2=makeMsize(cp_l1), msP3=makeMsize(cp_l2);
    std::vector<uint8_t> msP4(code.begin(), code.begin()+cp_end);
    for(int t=0;t<13;++t) msP4.pop_back();
    { uint8_t tail[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3}; for(uint8_t x:tail) msP4.push_back(x); }

    auto buildCalldata=[&](int im, std::vector<uint8_t>& cd, int& cpp_pred, std::vector<int>& h1r, std::vector<int>& h2r){
        std::vector<int> im_img=imgs[im];
        h1r.assign(H1,0); for(int n=0;n<H1;++n){ long long s=b1[n]; for(int i=0;i<I;++i) s+=(long long)im_img[i]*W1[i*H1+n]; h1r[n]=rss(s,SHIFT1);}
        h2r.assign(H2,0); for(int n=0;n<H2;++n){ long long s=b2[n]; for(int i=0;i<H1;++i) s+=(long long)h1r[i]*W2[i*H2+n]; h2r[n]=rss(s,SHIFT2);}
        std::vector<long long> scr(10); for(int cc=0;cc<10;++cc){ long long s=0; for(int i=0;i<H2;++i) s+=(long long)h2r[i]*W3[i*10+cc]; scr[cc]=s;}
        cpp_pred=std::max_element(scr.begin(),scr.end())-scr.begin();
        cd.assign(I/8*32,0);
        for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l){ int val=im_img[gw*8+l]; uint16_t u=(uint16_t)(int16_t)val; w[l*2]=u&0xFF; w[l*2+1]=(u>>8)&0xFF;} for(int i=0;i<32;++i) cd[gw*32+i]=w[31-i]; }
    };
    auto runCode=[&](std::vector<uint8_t>& bc, std::vector<uint8_t>& cd, std::vector<uint8_t>& ob)->long long{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GBIG; msg.recipient=ADDR; msg.code_address=ADDR;
        msg.input_data=cd.data(); msg.input_size=cd.size();
        auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,bc.data(),bc.size());
        long long g=GBIG-res.gas_left; ob.clear();
        if(res.status_code==0 && res.output_data) ob.assign(res.output_data,res.output_data+res.output_size);
        else ob.clear();
        if(res.release)res.release(&res); return g;
    };
    auto readMsize=[&](std::vector<uint8_t>& ob)->long long{ long long ms=0; if((int)ob.size()>=32) for(int b=0;b<32;++b) ms=(ms<<8)|ob[b]; return ms; };
    auto memgas=[&](long long ms)->long long{ long long w=(ms+31)/32; return 3*w + (w*w)/512; };
    // intermediate activation parse: same format as INP (word reversed, lane int16 LE). RISKY PART.
    auto parseAct=[&](std::vector<uint8_t>& ob, int M, std::vector<int>& act)->bool{
        act.assign(M,0); if((int)ob.size() < (M/8)*32) return false;
        for(int g=0; g<M/8; ++g){ uint8_t wt[32]; for(int i=0;i<32;++i) wt[i]=ob[g*32+(31-i)];
            for(int l=0;l<8;++l){ int n=g*8+l; if(n>=M) break; uint16_t u=wt[l*2]|((uint16_t)wt[l*2+1]<<8); act[n]=(int)(int16_t)u; } }
        return true;
    };

    std::cout<<"Bytecode(full)="<<cp_end<<" byte | matWords="<<matWords<<" | cp_load="<<cp_load<<" cp_l1="<<cp_l1<<" cp_l2="<<cp_l2<<"\n";

    int N_MEASURE=32;
    std::vector<long long> k1v,k2v,koutv,loadv; long long ref_p4=0; int hard_fail=0;
    long long need=(long long)matWords*2000, sc=(long long)matWords*2100, sw=(long long)matWords*100;

    for(int im=0; im<N_MEASURE; ++im){
        std::vector<uint8_t> cd; int cpp; std::vector<int> h1r,h2r; buildCalldata(im,cd,cpp,h1r,h2r);
        std::vector<uint8_t> ob;
        setAcc(EVMC_ACCESS_COLD); runCode(msP1,cd,ob); long long m1=readMsize(ob);
        setAcc(EVMC_ACCESS_COLD); runCode(msP2,cd,ob); long long m2=readMsize(ob);
        setAcc(EVMC_ACCESS_COLD); runCode(msP3,cd,ob); long long m3=readMsize(ob);
        setAcc(EVMC_ACCESS_COLD); runCode(msP4,cd,ob); long long m4=readMsize(ob);
        setAcc(EVMC_ACCESS_COLD); long long gc1=runCode(bcP1,cd,ob);
        setAcc(EVMC_ACCESS_COLD); long long gc2=runCode(bcP2,cd,ob); std::vector<int> a2; bool pa2=parseAct(ob,H1,a2);
        setAcc(EVMC_ACCESS_COLD); long long gc3=runCode(bcP3,cd,ob); std::vector<int> a3; bool pa3=parseAct(ob,H2,a3);
        setAcc(EVMC_ACCESS_COLD); long long gc4=runCode(bcP4,cd,ob); int p4=-1; if((int)ob.size()>=32) p4=((int32_t)ob[28]<<24)|((int32_t)ob[29]<<16)|((int32_t)ob[30]<<8)|(int32_t)ob[31];
        setAcc(EVMC_ACCESS_WARM); long long gw1=runCode(bcP1,cd,ob);
        setAcc(EVMC_ACCESS_WARM); long long gw2=runCode(bcP2,cd,ob);
        setAcc(EVMC_ACCESS_WARM); long long gw3=runCode(bcP3,cd,ob);
        setAcc(EVMC_ACCESS_WARM); long long gw4=runCode(bcP4,cd,ob);

        long long hc1=gc1-sc-memgas(m1), hc2=gc2-sc-memgas(m2), hc3=gc3-sc-memgas(m3), hc4=gc4-sc-memgas(m4);
        long long hw1=gw1-sw-memgas(m1), hw2=gw2-sw-memgas(m2), hw3=gw3-sw-memgas(m3), hw4=gw4-sw-memgas(m4);

        if(gc1-gw1!=need||gc2-gw2!=need||gc3-gw3!=need||gc4-gw4!=need){ printf("im=%d CHECK4 SLOAD FAIL: diffs %lld/%lld/%lld/%lld need=%lld\n",im,gc1-gw1,gc2-gw2,gc3-gw3,gc4-gw4,need); hard_fail=1; }
        if(hc1!=hw1||hc2!=hw2||hc3!=hw3||hc4!=hw4){ printf("im=%d CHECK5 h_cold!=h_warm FAIL\n",im); hard_fail=1; }
        if(p4!=ref_pred[im]){ printf("im=%d CHECK1 prediction FAIL: EVM=%d ref=%d\n",im,p4,ref_pred[im]); hard_fail=1; }
        bool a2ok=pa2&&(a2==h1r), a3ok=pa3&&(a3==h2r);
        if(!a2ok){ hard_fail=1; printf("im=%d CHECK2 h1 FAIL (suspect the PARSE first). first 8 EVM vs ref: ",im); for(int q=0;q<8&&q<H1;++q) printf("%d/%d ",a2.size()>(size_t)q?a2[q]:-999,h1r[q]); printf("\n"); }
        if(!a3ok){ hard_fail=1; printf("im=%d CHECK2 h2 FAIL (suspect the PARSE first). first 8 EVM vs ref: ",im); for(int q=0;q<8&&q<H2;++q) printf("%d/%d ",a3.size()>(size_t)q?a3[q]:-999,h2r[q]); printf("\n"); }
        long long k1=hc2-hc1, k2=hc3-hc2, kout=hc4-hc3;
        if(k1<=0||k2<=0||kout<=0){ printf("im=%d CHECK3 layer<=0 FAIL: k1=%lld k2=%lld kout=%lld\n",im,k1,k2,kout); hard_fail=1; }

        loadv.push_back(hc1); k1v.push_back(k1); k2v.push_back(k2); koutv.push_back(kout);
        if(im==0){ ref_p4=hc4; printf("[im=0] load=%lld k1=%lld k2=%lld kout=%lld | P4total=%lld | MSIZE m1/m2/m3/m4=%lld/%lld/%lld/%lld | prediction EVM=%d ref=%d | intermediate h1=%s h2=%s\n",hc1,k1,k2,kout,hc4,m1,m2,m3,m4,p4,ref_pred[im],a2ok?"OK":"FAIL",a3ok?"OK":"FAIL"); }
    }

    auto stats=[&](std::vector<long long>& v,const char* nm){ long long mn=v[0],mx=v[0]; double s=0; for(auto x:v){ if(x<mn)mn=x; if(x>mx)mx=x; s+=x; } printf("  %-14s mean=%12.1f  min=%12lld  max=%12lld  spread=%lld\n",nm,s/v.size(),mn,mx,mx-mn); };
    printf("\n========== MNIST PER-LAYER COMPUTE PROFILE (%d images, MSIZE-measured) ==========\n",N_MEASURE);
    stats(loadv,"Load-ovh"); stats(k1v,"Layer1(128x784)"); stats(k2v,"Layer2(64x128)"); stats(koutv,"Output+argmax");
    { double t=0; long long a=k1v[0]+k2v[0]+koutv[0]; for(size_t i=0;i<k1v.size();++i){} (void)t;
      printf("  first image: k1+k2+kout=%lld | compute[P4]=%lld | mnist_sload ref ~2,117,326 (cross-check)\n", a, ref_p4);
      printf("  share of compute[P4] (first image): load-ovh %.2f%% | layer1 %.2f%% | layer2 %.2f%% | output+argmax %.2f%%\n",
          100.0*loadv[0]/ref_p4, 100.0*k1v[0]/ref_p4, 100.0*k2v[0]/ref_p4, 100.0*koutv[0]/ref_p4); }
    if(hard_fail) printf("\n*** AT LEAST ONE CHECK FAILED -> profile UNRELIABLE ***\n");
    else printf("\n*** ALL CHECKS PASSED -> per-layer compute profile is valid ***\n");

    vm->destroy(vm); dlclose(h); return 0;
}
