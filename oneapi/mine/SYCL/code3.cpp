#include <CL/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;
using namespace std;
constexpr size_t array_size = 10000;
int main(){
    constexpr int value = 100000;
    try{
        //
        // The default device selector will select the most performant device.
        default_selector d_selector;
        queue q(d_selector);
        //Allocating shared memory using USM.
        int *sequential = malloc_shared<int>(array_size, q);
        int *parallel = malloc_shared<int>(array_size, q);
        //Sequential iota
        for (size_t i = 0; i < array_size; i++) sequential[i] = value + i;
        //Parallel iota in SYCL
        auto e = q.parallel_for(range{array_size}, [=](auto i) { parallel[i] = value + i; });
        e.wait();
        // Verify two results are equal.
        for (size_t i = 0; i < array_size; i++) {
            if (parallel[i] != sequential[i]) {
            cout << "Failed on device.\n";
            return -1;
            }
        }
        free(sequential, q);
        free(parallel, q);
    }catch (std::exception const &e) {
    cout << "An exception is caught while computing on device.\n";
    terminate();
    }
    cout << "Successfully completed on device.\n";
    return 0;
}