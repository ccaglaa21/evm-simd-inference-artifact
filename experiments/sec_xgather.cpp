// XGATHER (0xE3) OVERFLOW SECURITY TEST. Setup identical to test_xgather.cpp.
// S1 sanity: normal header -> correct result.
// S2 max_data_idx overflow: when DATA header rows*cols wraps in uint64, is the index check bypassed?
// S3 total_indices overflow: when IND header rows*cols wraps, does dRes[] exceed the Res region (OOB write)?
#include <evmc/evmc.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <cstdint>
static evmc_vm* VM;
static const int64_t GAS=2000000;

struct Builder {
    std::vector<uint8_t> code;
    void u64le(std::vector<uint8_t>& b, uint64_t v){ for(int i=0;i<8;++i) b.push_back((v>>(8*i))&0xFF); }
    void P1(uint8_t v){ code.push_back(0x60); code.push_back(v); }
    void P2(int v){ code.push_back(0x61); code.push_back((v>>8)&0xFF); code.push_back(v&0xFF); }
    void W32(std::vector<uint8_t> b){ while(b.size()<32)b.push_back(0); code.push_back(0x7F); for(int i=0;i<32;++i) code.push_back(b[i]); }
    void MS(int off){ P2(off); code.push_back(0x52); }
    void hdr(int off, uint64_t r,uint64_t c,uint64_t sr,uint64_t sc){ std::vector<uint8_t> hb; u64le(hb,r);u64le(hb,c);u64le(hb,sr);u64le(hb,sc); W32(hb); MS(off); }
};

// XGATHER call: DATA/IND/RES regions, chosen header values + index list.
static evmc_status_code run(const char* name,
    uint64_t dRows,uint64_t dCols,            // DATA header (may be huge to force overflow)
    uint64_t iRows,uint64_t iCols,            // IND header (may be huge to force overflow)
    const std::vector<uint16_t>& idxList,     // index values (IND data)
    const std::vector<uint16_t>& dataVals,    // DATA contents (real elements)
    int retLen, const std::vector<uint16_t>* expect)
{
    Builder k;
    int DATA=0, IND=256, RES=512;             // wide spacing, no overlap
    k.hdr(DATA, dRows,dCols,dCols,1);
    { std::vector<uint8_t> d(64,0); for(size_t i=0;i<dataVals.size() && i<32;++i){ d[i*2]=dataVals[i]&0xFF; d[i*2+1]=(dataVals[i]>>8)&0xFF; } 
      // write 64 bytes of data (2 words) so that the normal case has enough
      k.code.push_back(0x7F); for(int i=0;i<32;++i) k.code.push_back(d[i]); k.MS(DATA+32);
      k.code.push_back(0x7F); for(int i=0;i<32;++i) k.code.push_back(d[32+i]); k.MS(DATA+64); }
    k.hdr(IND, iRows,iCols,iCols,1);
    { std::vector<uint8_t> d(64,0); for(size_t i=0;i<idxList.size() && i<32;++i){ d[i*2]=idxList[i]&0xFF; d[i*2+1]=(idxList[i]>>8)&0xFF; }
      k.code.push_back(0x7F); for(int i=0;i<32;++i) k.code.push_back(d[i]); k.MS(IND+32);
      k.code.push_back(0x7F); for(int i=0;i<32;++i) k.code.push_back(d[32+i]); k.MS(IND+64); }
    // XGATHER: push RES, IND, DATA
    k.P2(RES); k.P2(IND); k.P2(DATA); k.code.push_back(0xE3);
    // return retLen bytes from RES+32
    k.P2(RES+32); k.code.push_back(0x51); k.P1(0); k.code.push_back(0x52);
    k.P1((uint8_t)retLen); k.P1(0); k.code.push_back(0xF3);

    evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS;
    auto res=VM->execute(VM,nullptr,nullptr,EVMC_SHANGHAI,&msg,k.code.data(),k.code.size());
    auto st=res.status_code;
    printf("[%-34s] status=%d gas=%lld", name, (int)st, (long long)(GAS-res.gas_left));
    if(st==EVMC_SUCCESS && expect && res.output_data && (int)res.output_size>=retLen){
        bool ok=true; printf("  result=[");
        for(size_t i=0;i<expect->size();++i){ uint16_t v=(uint16_t)res.output_data[i*2]|((uint16_t)res.output_data[i*2+1]<<8); printf("%u%s",v,i+1<expect->size()?",":""); if(v!=(*expect)[i])ok=false; }
        printf("] %s", ok?"OK":"MISMATCH!");
    } else if(st==EVMC_OUT_OF_GAS){ printf("  (OUT_OF_GAS = caught by check_memory, safe)"); }
    else if(st==EVMC_SUCCESS){ printf("  (SUCCESS)"); }
    printf("\n");
    if(res.release)res.release(&res);
    return st;
}

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ printf("dlopen FAIL\n"); return 2; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone"); VM=create();
    printf("=== XGATHER OVERFLOW SECURITY TEST ===\n");

    // S1: SANITY — data=[10..80], idx=[0,2,4,6] -> [10,30,50,70]
    std::vector<uint16_t> data8={10,20,30,40,50,60,70,80};
    std::vector<uint16_t> exp={10,30,50,70};
    run("S1 sanity (normal header)", 1,8, 1,4, {0,2,4,6}, data8, 8, &exp);

    // S2: max_data_idx OVERFLOW — DATA rows*cols wraps in uint64; index far beyond the real data (60000)
    //     Expected SAFE: if target_idx(60000) < max_data_idx(wrapped) does not hold, dRes=0 is returned (NO OOB READ)
    run("S2 max_data_idx overflow (huge rows*cols)", 
        (uint64_t)0x100000001ULL, (uint64_t)0x100000000ULL,   // rows*cols exceeds 2^64 -> wraps
        1,4, {60000,60001,60002,60003}, data8, 8, nullptr);

    // S3: total_indices OVERFLOW — IND rows*cols wraps; loop count vs Res map inconsistency
    //     Expected SAFE: OOG via check_memory(off_res,32+n_idx*2), or controlled behaviour
    run("S3 total_indices overflow (huge IND rows*cols)",
        1,8,
        (uint64_t)0x100000001ULL, (uint64_t)0x100000000ULL,   // total_indices wraps
        {0,2,4,6}, data8, 8, nullptr);

    printf("\nNOTE: This test MEASURES the behaviour of the 3 scenarios. 'OK/safe' is expected; if we see SUCCESS+wrong\n");
    printf("values or a segfault, there is a vulnerability, to be closed with a uint256/overflow-safe check as in XMACC_S.\n");
    VM->destroy(VM); dlclose(h); return 0;
}
