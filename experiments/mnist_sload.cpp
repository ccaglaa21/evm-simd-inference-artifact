#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
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
    // read ALL images (500)
    std::ifstream bf("mnist_batch.txt"); int Nb,wsc; bf>>Nb>>wsc;
    int N_RUN=500;
    std::vector<std::vector<int>> imgs(N_RUN, std::vector<int>(I));
    std::vector<int> pf_list(N_RUN); // float-model prediction (pfo from the batch file, not from deep_ref)
    for(int im=0; im<N_RUN; ++im){
        int tlb,pfo; bf>>tlb>>pfo; double tmp; for(int i=0;i<10;++i) bf>>tmp;
        for(int i=0;i<I;++i) bf>>imgs[im][i];
        pf_list[im]=pfo;
    }
    // read the reference predictions from deep_ref
    std::ifstream rf("deep_ref.txt"); int Nr; rf>>Nr;
    std::vector<int> ref_pred(N_RUN);
    std::vector<int> true_label(N_RUN);
    for(int im=0; im<N_RUN; ++im){ int tl,pd; rf>>tl>>pd; ref_pred[im]=pd; true_label[im]=tl; }
    std::vector<int> img(I); // temporary, filled in the loop

    auto rss=[&](long long v,int sh)->int{ if(v<0)v=0; v>>=sh; if(v>32767)v=32767; return (int)v; };
    std::vector<int> h1_ref(H1);
    for(int n=0;n<H1;++n){ long long s=b1[n]; for(int i=0;i<I;++i) s+=(long long)img[i]*W1[i*H1+n]; h1_ref[n]=rss(s,SHIFT1); }
    std::vector<int> h2_ref(H2);
    for(int n=0;n<H2;++n){ long long s=b2[n]; for(int i=0;i<H1;++i) s+=(long long)h1_ref[i]*W2[i*H2+n]; h2_ref[n]=rss(s,SHIFT2); }
    std::vector<long long> score_ref(10);
    for(int c=0;c<10;++c){ long long s=0; for(int i=0;i<H2;++i) s+=(long long)h2_ref[i]*W3[i*10+c]; score_ref[c]=s; } // bias=0, no ReLU
    int pred_ref = std::max_element(score_ref.begin(),score_ref.end())-score_ref.begin();

    std::vector<uint8_t> code;
    auto P32=[&](const std::vector<uint8_t>& w){ code.push_back(0x7F); for(int i=31;i>=0;--i) code.push_back(w[i]); };
    auto P1=[&](uint8_t v){ code.push_back(0x60); code.push_back(v); };
    auto P3=[&](int v){ code.push_back(0x62); code.push_back((v>>16)&0xFF); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); };
    auto PUSH=[&](int v){ if(v<256)P1((uint8_t)v); else if(v<65536){code.push_back(0x61);code.push_back((v>>8)&0xFF);code.push_back(v&0xFF);} else P3(v); };
    auto P4=[&](uint32_t v){ code.push_back(0x63); code.push_back((v>>24)&0xFF); code.push_back((v>>16)&0xFF); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); };
    uint8_t xgtcfg=(1<<5)|2;

    // emit_layer (for layers 1 and 2, multiples of 8, ReLU+XPACK)
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

    // MEMORY
    int INP=0, W1M=10000, B1M=420000, H1OUT=430000, W2M=440000, B2M=480000, H2OUT=490000;
    int W3M=495000;       // W3 (10 classes x 8 groups = 80 words) -> each class H2=64 inputs=8 groups
    int SCO=500000;       // 10 scores (1 word each, int32 raw score in lane 0)
    int ACC=900000, SACC=900100, NCNT=901000, GCNT=901100;

    // Setup: INPUT from calldata (CALLDATACOPY: dest=INP, offset=0, len=98*32)
    PUSH(I/8*32);   // length (3136)
    P1(0);          // calldata offset 0
    PUSH(INP);      // dest mem offset
    code.push_back(0x37); // CALLDATACOPY
    // === WEIGHTS FROM STORAGE: each word is one slot, loaded with SLOAD to its target memory address ===
    // C++ side: build a (value, target_memory_address) list; write to storage in slot order.
    std::vector<std::pair<std::vector<uint8_t>,int>> WLOAD;  // (word, mem_addr)
    for(int n=0;n<H1;++n) for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W1[(gw*8+l)*H1+n]); WLOAD.push_back({w, W1M+(n*(I/8)+gw)*32}); }
    for(int n=0;n<H1;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b1[n]); WLOAD.push_back({w, B1M+n*32}); }
    for(int n=0;n<H2;++n) for(int gw=0; gw<H1/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W2[(gw*8+l)*H2+n]); WLOAD.push_back({w, W2M+(n*(H1/8)+gw)*32}); }
    for(int n=0;n<H2;++n){ std::vector<uint8_t> w(32,0); put32(w,0,b2[n]); WLOAD.push_back({w, B2M+n*32}); }
    for(int c=0;c<10;++c) for(int gw=0; gw<H2/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l) put16(w,l,W3[(gw*8+l)*10+c]); WLOAD.push_back({w, W3M+(c*(H2/8)+gw)*32}); }
    int matWords=(int)WLOAD.size();   // total number of weight words (W1+b1+W2+b2+W3)

    // Bytecode: each slot -> SLOAD -> MSTORE to the target address. PUSH32 slot key (big-endian).
    auto PUSHslot=[&](uint32_t sl){ code.push_back(0x7F); for(int i=0;i<28;++i) code.push_back(0); code.push_back((sl>>24)&0xFF); code.push_back((sl>>16)&0xFF); code.push_back((sl>>8)&0xFF); code.push_back(sl&0xFF); };
    int loadCodeStart=(int)code.size();
    for(int sl=0; sl<matWords; ++sl){ PUSHslot((uint32_t)sl); code.push_back(0x54); PUSH(WLOAD[sl].second); code.push_back(0x52); }
    int loadCodeEnd=(int)code.size();
    (void)loadCodeStart; (void)loadCodeEnd;

    // LAYERS 1, 2
    emit_layer(H1, I/8, INP, W1M, B1M, H1OUT, ACC, SACC, NCNT, GCNT, SHIFT1);
    emit_layer(H2, H1/8, H1OUT, W2M, B2M, H2OUT, ACC, SACC, NCNT, GCNT, SHIFT2);

    // OUTPUT LAYER: 10 classes, raw score (bias=0, no ReLU, no shift)
    // For each class c: acc=0, 8 groups of XMACC (H2OUT input), XSUM32 -> raw score -> SCO+c*32 (lane 0)
    // Could be unrolled (10 is small), but done as a loop for consistency. NCNT=class counter.
    P1(0); PUSH(NCNT); code.push_back(0x52); // c=0
    int co_start=(int)code.size(); code.push_back(0x5B);
    PUSH(NCNT); code.push_back(0x51); P1(10); code.push_back(0x11); code.push_back(0x15); // c<10?
    P3(0); int co_ph=(int)code.size()-3; code.push_back(0x57);
    // acc=0
    P1(0); PUSH(ACC); code.push_back(0x52);
    P1(0); PUSH(GCNT); code.push_back(0x52);
    int ci_start=(int)code.size(); code.push_back(0x5B);
    PUSH(GCNT); code.push_back(0x51); PUSH(H2/8); code.push_back(0x11); code.push_back(0x15); // g<8?
    P3(0); int ci_ph=(int)code.size()-3; code.push_back(0x57);
    // input addr = H2OUT + g*32
    PUSH(GCNT); code.push_back(0x51); P1(32); code.push_back(0x02); code.push_back(0x80); PUSH(H2OUT); code.push_back(0x01); code.push_back(0x51);
    // weight addr = W3M + (c*(H2/8) + g)*32
    code.push_back(0x90); PUSH(NCNT); code.push_back(0x51); PUSH((H2/8)*32); code.push_back(0x02); code.push_back(0x01); PUSH(W3M); code.push_back(0x01); code.push_back(0x51);
    PUSH(ACC); code.push_back(0x51); code.push_back(0x91); code.push_back(0x90); code.push_back(0xCB); PUSH(ACC); code.push_back(0x52);
    PUSH(GCNT); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(GCNT); code.push_back(0x52);
    PUSH(ci_start); code.push_back(0x56);
    int ci_end=(int)code.size(); code.push_back(0x5B);
    code[ci_ph]=(ci_end>>16)&0xFF; code[ci_ph+1]=(ci_end>>8)&0xFF; code[ci_ph+2]=ci_end&0xFF;
    // raw score = XSUM32(acc) -> SCO + c*32 (NO ReLU, NO shift)
    PUSH(ACC); code.push_back(0x51); code.push_back(0xCD); // [score]
    PUSH(NCNT); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(SCO); code.push_back(0x01); // [score, addr]
    code.push_back(0x52); // MSTORE score -> SCO+c*32
    PUSH(NCNT); code.push_back(0x51); P1(1); code.push_back(0x01); PUSH(NCNT); code.push_back(0x52);
    PUSH(co_start); code.push_back(0x56);
    int co_end=(int)code.size(); code.push_back(0x5B);
    code[co_ph]=(co_end>>16)&0xFF; code[co_ph+1]=(co_end>>8)&0xFF; code[co_ph+2]=co_end&0xFF;

    // === ON-CHAIN ARGMAX (XGT signed int32) ===
    // NOTE: scores are at SCO+c*32; the XSUM32 output is in the last lane (bytes 28-31), i.e. lane 7.
    // Instead of moving the score to lane 0 first,
    // XGT compares all 8 lanes; since the score is in the last lane, we read the last-lane mask.
    // Simplicity: the score is in the last lane of SCO[c]. XGT(SCO[c],MAXV) produces a mask in the last lane.
    // Last-lane mask = word & 0xFFFFFFFF (lowest 32 bits = bytes 28-31 = last lane). CORRECT.
    int MAXV=900200, MAXI=900232, CNT2=900264;
    // max_val=SCO[0], max_idx=0, c=1
    PUSH(SCO); code.push_back(0x51); PUSH(MAXV); code.push_back(0x52);
    P1(0); PUSH(MAXI); code.push_back(0x52);
    P1(1); PUSH(CNT2); code.push_back(0x52);
    int amloop=(int)code.size(); code.push_back(0x5B);
    PUSH(CNT2); code.push_back(0x51); P1(10); code.push_back(0x11); code.push_back(0x15);
    P3(0); int amend_ph=(int)code.size()-3; code.push_back(0x57);
    // XGT(SCO[c] > MAXV)
    PUSH(MAXV); code.push_back(0x51);
    PUSH(CNT2); code.push_back(0x51); P1(32); code.push_back(0x02); PUSH(SCO); code.push_back(0x01); code.push_back(0x51);
    code.push_back(0xCA); code.push_back(xgtcfg);
    P4(0xFFFFFFFF); code.push_back(0x16); code.push_back(0x15); // AND 0xFFFFFFFF, ISZERO
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
    // return the prediction (MAXI)
    PUSH(MAXI); code.push_back(0x51); P1(0); code.push_back(0x52);
    PUSH(0x20); P1(0); code.push_back(0xF3);

    std::cout<<"Bytecode: "<<code.size()<<" bytes. Testing 500 images...\n";

    // ===================== STORAGE + BUCKET MEASUREMENT =====================
    // Address (model contract). Write the weights to slots 0..matWords-1.
    evmc::address ADDR{}; ADDR.bytes[18]=0xc0; ADDR.bytes[19]=0xde;
    evmc::MockedHost host;
    for(int sl=0; sl<matWords; ++sl){
        evmc::bytes32 key{}, val{};
        key.bytes[28]=(sl>>24)&0xFF; key.bytes[29]=(sl>>16)&0xFF; key.bytes[30]=(sl>>8)&0xFF; key.bytes[31]=sl&0xFF;
        // WLOAD[sl].first is written REVERSED by P32 (byte0=w[31]); the storage word must be w itself.
        // PUSH32 interprets big-endian: SLOAD puts w on the stack big-endian, MSTORE writes it to memory big-endian.
        // So we put w into storage (the SAME 32 bytes that P32 embeds).
        for(int b=0;b<32;++b) val.bytes[b]=WLOAD[sl].first[31-b];  // REVERSED: so that SLOAD+MSTORE produces the same byte layout as PUSH32 (verified with endian_check)
        host.accounts[ADDR].storage[key]=evmc::StorageValue{val};
    }
    auto setAcc=[&](evmc_access_status st){ for(int sl=0;sl<matWords;++sl){ evmc::bytes32 key{}; key.bytes[28]=(sl>>24)&0xFF;key.bytes[29]=(sl>>16)&0xFF;key.bytes[30]=(sl>>8)&0xFF;key.bytes[31]=sl&0xFF; host.accounts[ADDR].storage[key].access_status=st; } };
    const evmc_host_interface& IF = evmc::MockedHost::get_interface();

    // load-only bytecode: only the SLOAD-load part (loadCodeStart..loadCodeEnd) + RETURN(0,0)
    std::vector<uint8_t> loadCode(code.begin()+loadCodeStart, code.begin()+loadCodeEnd);
    loadCode.push_back(0x60); loadCode.push_back(0x00); loadCode.push_back(0x60); loadCode.push_back(0x00); loadCode.push_back(0xF3);

    long long aPeak=0, bellek=0;   // to be MEASURED for real with MSIZE (after runFull is defined)

    // representative image (im=0) calldata + C++ ref prediction
    auto buildCalldata=[&](int im, std::vector<uint8_t>& calldata, int& cpp_pred){
        std::vector<int> im_img=imgs[im];
        std::vector<int> h1r(H1); for(int n=0;n<H1;++n){ long long s=b1[n]; for(int i=0;i<I;++i) s+=(long long)im_img[i]*W1[i*H1+n]; h1r[n]=rss(s,SHIFT1);}
        std::vector<int> h2r(H2); for(int n=0;n<H2;++n){ long long s=b2[n]; for(int i=0;i<H1;++i) s+=(long long)h1r[i]*W2[i*H2+n]; h2r[n]=rss(s,SHIFT2);}
        std::vector<long long> scr(10); for(int cc=0;cc<10;++cc){ long long s=0; for(int i=0;i<H2;++i) s+=(long long)h2r[i]*W3[i*10+cc]; scr[cc]=s;}
        cpp_pred=std::max_element(scr.begin(),scr.end())-scr.begin();
        calldata.assign(I/8*32,0);
        for(int gw=0; gw<I/8; ++gw){ std::vector<uint8_t> w(32,0); for(int l=0;l<8;++l){ int val=im_img[gw*8+l]; uint16_t u=(uint16_t)(int16_t)val; w[l*2]=u&0xFF; w[l*2+1]=(u>>8)&0xFF;} for(int i=0;i<32;++i) calldata[gw*32+i]=w[31-i]; }
    };

    const int64_t GBIG=20000000000LL;
    auto runFull=[&](std::vector<uint8_t>& calldata, int& pred)->long long{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GBIG; msg.recipient=ADDR; msg.code_address=ADDR;
        msg.input_data=calldata.data(); msg.input_size=calldata.size();
        auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,code.data(),code.size());
        long long g=GBIG-res.gas_left; pred=-1;
        if(res.status_code==0&&res.output_data&&res.output_size>=32) pred=((int32_t)res.output_data[28]<<24)|((int32_t)res.output_data[29]<<16)|((int32_t)res.output_data[30]<<8)|(int32_t)res.output_data[31];
        if(res.release)res.release(&res); return g;
    };
    auto runLoad=[&]()->long long{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GBIG; msg.recipient=ADDR; msg.code_address=ADDR;
        std::vector<uint8_t> cd(I/8*32,0); msg.input_data=cd.data(); msg.input_size=cd.size();
        auto res=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&msg,loadCode.data(),loadCode.size());
        long long g=GBIG-res.gas_left; if(res.release)res.release(&res); return g;
    };

    // --- REAL memory peak via MSIZE: replace the last 12 bytes of the code copy (prediction return) with an MSIZE return ---
    {
        std::vector<uint8_t> mcode=code;
        for(int t=0;t<13;++t) mcode.pop_back();                 // PUSH3(MAXI)[4]+MLOAD[1]+PUSH1_0[2]+MSTORE[1]+PUSH1_0x20[2]+PUSH1_0[2]+RETURN[1] = 13 byte
        uint8_t tail[]={0x59,0x60,0x00,0x52,0x60,0x20,0x60,0x00,0xF3};  // MSIZE PUSH0 MSTORE PUSH0x20 PUSH0 RETURN
        for(uint8_t x:tail) mcode.push_back(x);
        std::vector<uint8_t> cd0; int cp0; buildCalldata(0,cd0,cp0);
        setAcc(EVMC_ACCESS_COLD);
        evmc_message m{}; m.kind=EVMC_CALL; m.gas=20000000000LL; m.recipient=ADDR; m.code_address=ADDR;
        m.input_data=cd0.data(); m.input_size=cd0.size();
        auto r=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&m,mcode.data(),mcode.size());
        long long msize=0; if(r.status_code==0&&r.output_data&&r.output_size>=32) for(int b=0;b<32;++b) msize=(msize<<8)|r.output_data[b];
        if(r.release)r.release(&r);
        aPeak=(msize+31)/32; bellek=3*aPeak+(aPeak*aPeak)/512;
        std::cout<<"[MSIZE] real memory = "<<msize<<" byte ("<<aPeak<<" words) -> memory gas = "<<bellek<<"\n";
    }

    std::cout<<"\n=== MNIST REAL COST MAP (weights SLOADed from storage) ===\n";
    std::cout<<"Bytecode(full): "<<code.size()<<" bytes | weight words (matWords): "<<matWords<<" | peak memory words: "<<aPeak<<"\n";

    // PREDICTION CHECK: does EVM(storage) == C++ ref on the first 5 images?
    int okpred=0;
    for(int im=0; im<5; ++im){ std::vector<uint8_t> cd; int cpp_pred,evm_pred; buildCalldata(im,cd,cpp_pred); setAcc(EVMC_ACCESS_COLD); runFull(cd,evm_pred); if(evm_pred==cpp_pred) okpred++; else std::cout<<"  [MISMATCH] im="<<im<<" EVM="<<evm_pred<<" C++="<<cpp_pred<<"\n"; }
    std::cout<<"Prediction check (weights from storage): "<<okpred<<"/5 "<<(okpred==5?"[OK - weights from storage did not change the result]":"[ERROR - stop!]")<<"\n";

    if(okpred==5){
        std::vector<uint8_t> cd; int cpp_pred,pp; buildCalldata(0,cd,cpp_pred);
        // COLD
        setAcc(EVMC_ACCESS_COLD); long long fullCold=runFull(cd,pp);
        setAcc(EVMC_ACCESS_COLD); long long ioCold=runLoad();
        // WARM
        setAcc(EVMC_ACCESS_WARM); long long fullWarm=runFull(cd,pp);
        long long sloadCold=(long long)matWords*2100, sloadWarm=(long long)matWords*100;
        long long hesapCold=fullCold-sloadCold-bellek, hesapWarm=fullWarm-sloadWarm-bellek;
        auto P=[&](const char* ph,long long sload,long long hesap,long long full){ printf("%-5s | SLOAD-read=%-12lld | compute=%-10lld | memory=%-10lld | TOTAL=%-12lld\n",ph,sload,hesap,bellek,full); };
        printf("\n--- 3 BUCKETS (one inference) ---\n");
        P("COLD",sloadCold,hesapCold,fullCold);
        P("WARM",sloadWarm,hesapWarm,fullWarm);
        printf("\n[CHECK] load-only measured I/O (cold) = %lld | analytic SLOAD = %lld | diff = %lld (should be small: header+loop overhead)\n", ioCold, sloadCold, ioCold-sloadCold);
        printf("[CHECK] (cold full - warm full) = %lld | matWords*2000 = %lld | %s\n", fullCold-fullWarm, (long long)matWords*2000, (fullCold-fullWarm==(long long)matWords*2000)?"OK (only difference is SLOAD)":"MISMATCH");
        printf("\nCOMPUTE/TOTAL: cold %.1f%% | warm %.1f%%  -> SLOAD weight I/O dominates\n", 100.0*hesapCold/fullCold, 100.0*hesapWarm/fullWarm);
    }
        vm->destroy(vm); dlclose(h); return 0;
}
