// XGATHER small-n OVERFLOW test (S4/S5): headers whose rows*cols product wraps to a SMALL value in
// 64-bit arithmetic. The length check of XGATHER forms the products in 256-bit arithmetic, so such a
// header (more than 2^64 elements claimed) must be rejected with a clean out-of-gas before the opcode
// body is entered; the reference case (a normal header with one out-of-range index) must succeed and
// return 0 for that index. An earlier version of the check used 64-bit products and entered the body.
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
static evmc_status_code run(const char* name,
    uint64_t dRows,uint64_t dCols, uint64_t iRows,uint64_t iCols,
    const std::vector<uint16_t>& idxList, const std::vector<uint16_t>& dataVals, int retLen)
{
    Builder k;
    int DATA=0, IND=256, RES=512;
    k.hdr(DATA, dRows,dCols,dCols?dCols:1,1);
    { std::vector<uint8_t> d(64,0); for(size_t i=0;i<dataVals.size() && i<32;++i){ d[i*2]=dataVals[i]&0xFF; d[i*2+1]=(dataVals[i]>>8)&0xFF; }
      k.code.push_back(0x7F); for(int i=0;i<32;++i) k.code.push_back(d[i]); k.MS(DATA+32);
      k.code.push_back(0x7F); for(int i=0;i<32;++i) k.code.push_back(d[32+i]); k.MS(DATA+64); }
    k.hdr(IND, iRows,iCols,iCols?iCols:1,1);
    { std::vector<uint8_t> d(64,0); for(size_t i=0;i<idxList.size() && i<32;++i){ d[i*2]=idxList[i]&0xFF; d[i*2+1]=(idxList[i]>>8)&0xFF; }
      k.code.push_back(0x7F); for(int i=0;i<32;++i) k.code.push_back(d[i]); k.MS(IND+32);
      k.code.push_back(0x7F); for(int i=0;i<32;++i) k.code.push_back(d[32+i]); k.MS(IND+64); }
    k.P2(RES); k.P2(IND); k.P2(DATA); k.code.push_back(0xE3);
    k.P2(RES+32); k.code.push_back(0x51); k.P1(0); k.code.push_back(0x52);
    k.P1((uint8_t)retLen); k.P1(0); k.code.push_back(0xF3);
    evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=GAS;
    auto res=VM->execute(VM,nullptr,nullptr,EVMC_SHANGHAI,&msg,k.code.data(),k.code.size());
    auto st=res.status_code;
    printf("[%-40s] status=%d gas=%lld", name, (int)st, (long long)(GAS-res.gas_left));
    if(st==EVMC_SUCCESS && res.output_data && (int)res.output_size>=retLen){
        printf("  RES=[");
        for(int i=0;i<retLen/2;++i){ uint16_t v=(uint16_t)res.output_data[i*2]|((uint16_t)res.output_data[i*2+1]<<8); printf("%u%s",v,i+1<retLen/2?",":""); }
        printf("]");
    } else if(st==EVMC_OUT_OF_GAS){ printf("  (OOG = caught by check_memory)"); }
    printf("\n");
    if(res.release)res.release(&res);
    return st;
}
int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ printf("dlopen FAIL\n"); return 2; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone"); VM=create();
    printf("=== XGATHER SMALL-N OVERFLOW TEST (S4/S5) ===\n");
    std::vector<uint16_t> data8={10,20,30,40,50,60,70,80};

    // Reference: normal, idx exceeds the real data (does the check really work?)
    printf("-- ref: normal header, idx=[0,2,100,6] (100 out of bounds -> must return 0) --\n");
    auto sRef = run("REF idx=100 normal(rows=1,cols=8)", 1,8, 1,4, {0,2,100,6}, data8, 8);

    // S4: rows*cols wraps to 8 in 64 bits (rows = 2^62+1, cols = 8); must be rejected
    auto s4a = run("S4a max_data_idx->8, idx=[0,2,4,6]", (uint64_t)((1ULL<<62)+1), 8, 1,4, {0,2,4,6}, data8, 8);
    // S4b: max_data_idx SMALL wrap (=8). idx=[6,7,5,4] all <8 but data has 8 elements -> boundary test
    auto s4b = run("S4b max_data_idx->8, idx=[6,7,5,4]", (uint64_t)((1ULL<<62)+1), 8, 1,4, {6,7,5,4}, data8, 8);
    // S4c: max_data_idx->4 (rows=2^63+2,cols=2). idx=[5,6,7,3]: 5,6,7 >= 4 -> must return 0 (check became stricter)
    auto s4c = run("S4c max_data_idx->4, idx=[5,6,7,3]", (uint64_t)((1ULL<<63)+2), 2, 1,4, {5,6,7,3}, data8, 8);

    // S5: the index header wraps to 8 elements in 64 bits; must be rejected
    auto s5a = run("S5a total_indices->8, idx[0..7]", 1,8, (uint64_t)((1ULL<<62)+1), 8, {0,1,2,3,4,5,6,7}, data8, 16);
    // S5b: total_indices->4. loop runs 4 times
    auto s5b = run("S5b total_indices->4, idx[0,2,4,6]", 1,8, (uint64_t)((1ULL<<63)+2), 2, {0,2,4,6}, data8, 8);

    printf("\nExpected: REF succeeds with RES=[10,30,0,70]; S4a-S5b stop with a clean out-of-gas (status=3), because a\n");
    printf("header that claims more than 2^64 elements is rejected by the 256-bit length check before the opcode body.\n");
    const bool ok = (sRef == EVMC_SUCCESS) && (s4a == EVMC_OUT_OF_GAS) && (s4b == EVMC_OUT_OF_GAS) && (s4c == EVMC_OUT_OF_GAS)
                 && (s5a == EVMC_OUT_OF_GAS) && (s5b == EVMC_OUT_OF_GAS);
    printf("[CHECK] 1 reference success + 5 wrapped headers rejected: %s\n", ok ? "PASSED" : "FAILED");
    VM->destroy(VM); dlclose(h); return ok ? 0 : 1;
}
