#include <evmc/evmc.h>
#include <evmc/mocked_host.hpp>   // a valid (empty) host interface; no host call is made by this program
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <cstdint>
static evmc::MockedHost g_host;

int main(){
    void* h=dlopen("build/lib/libevmone.so",RTLD_NOW);
    auto create=(evmc_vm*(*)())dlsym(h,"evmc_create_evmone");
    evmc_vm* vm=create();

    auto run=[&](const std::string& ad, std::vector<uint8_t> code)->void{
        evmc_message msg{}; msg.kind=EVMC_CALL; msg.gas=1000000;
        auto res=vm->execute(vm, &g_host.get_interface(), g_host.to_context(),EVMC_SHANGHAI,&msg,code.data(),code.size());
        std::cout<<ad<<": status="<<res.status_code<<" gas="<<(1000000-res.gas_left)<<"\n";
        if(res.release)res.release(&res);
    };

    std::cout<<"=== MATRIXHEADER / XTRANSPOSE SECURITY ===\n\n";

    // 1. XTRANSPOSE normal (small offset, expand memory first)
    {
        std::vector<uint8_t> code;
        // first write a word at offset 0 (so that memory exists)
        code.push_back(0x7F); for(int i=0;i<32;++i) code.push_back(0x01);
        code.push_back(0x60);code.push_back(0x00);code.push_back(0x52); // MSTORE 0
        // XTRANSPOSE offset 0
        code.push_back(0x60);code.push_back(0x00); // PUSH1 0
        code.push_back(0xDD); // XTRANSPOSE
        run("1. XTRANSPOSE offset=0 (normal, memory present)", code);
    }

    // 2. XTRANSPOSE without expanding memory (offset 0 but memory empty)
    {
        std::vector<uint8_t> code;
        code.push_back(0x60);code.push_back(0x00); // PUSH1 0
        code.push_back(0xDD); // XTRANSPOSE, memory never expanded
        run("2. XTRANSPOSE empty memory offset=0", code);
    }

    // 3. XTRANSPOSE medium offset (memory not expanded, 1000)
    {
        std::vector<uint8_t> code;
        code.push_back(0x61);code.push_back(0x10);code.push_back(0x00); // PUSH2 4096
        code.push_back(0xDD);
        run("3. XTRANSPOSE offset=4096 (no memory)", code);
    }

    std::cout<<"\n(Note: the huge-offset test risks a segfault, done separately/carefully)\n";
    std::cout<<"\n=== FINDING ===\n";
    std::cout<<"Testing the XTRANSPOSE offset bounds check.\n";
    vm->destroy(vm); dlclose(h); return 0;
}
