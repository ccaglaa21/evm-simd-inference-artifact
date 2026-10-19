// sec_xtranspose_big.cpp — closing gap D: XTRANSPOSE with a HUGE offset.
// Expectation: the header read incurs the memory-expansion charge; at a huge offset
// the quadratic charge -> clean OOG (status=3). SUCCESS or a crash = VULNERABILITY.
#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <vector>
#include <cstdint>
int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    if(!h){ std::printf("libevmone not found\n"); return 1; }
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone");
    evmc_vm* vm=create();
    evmc::MockedHost host;
    const evmc_host_interface& IF=evmc::MockedHost::get_interface();
    evmc::address ADDR{}; ADDR.bytes[19]=0x01;
    struct Cs{ const char* ad; int pushBytes; unsigned long long off; };
    std::vector<Cs> cs={
        {"CONTROL offset=0 (normal)",1,0},
        {"offset=2^32",5,1ULL<<32},
        {"offset=2^48",7,1ULL<<48},
        {"offset=2^60",8,1ULL<<60},
    };
    int fail=0;
    for(auto&c:cs){
        std::vector<uint8_t> k;
        k.push_back((uint8_t)(0x60+c.pushBytes-1));                 // PUSHn
        for(int b=c.pushBytes-1;b>=0;--b) k.push_back((uint8_t)((c.off>>(8*b))&0xFF));
        k.push_back(0xDD);                                          // XTRANSPOSE
        k.push_back(0x00);                                          // STOP
        evmc_message m{}; m.kind=EVMC_CALL; m.gas=1000000; m.recipient=ADDR; m.code_address=ADDR;
        auto r=vm->execute(vm,&IF,host.to_context(),EVMC_SHANGHAI,&m,k.data(),k.size());
        long long g=1000000-r.gas_left;
        bool ok = (c.off==0)? (r.status_code==0) : (r.status_code==3);
        std::printf("[%s] status=%d gas=%lld -> %s\n",c.ad,(int)r.status_code,g,
            ok?(c.off==0?"ran (expected)":"clean OOG (expected)"):"UNEXPECTED — POSSIBLE VULNERABILITY");
        if(!ok) fail=1;
        if(r.release)r.release(&r);
    }
    std::printf(fail?"*** AT LEAST ONE SCENARIO UNEXPECTED ***\n":"*** PASSED: huge offsets give a clean OOG, no crash ***\n");
    vm->destroy(vm); dlclose(h); return fail;
}
