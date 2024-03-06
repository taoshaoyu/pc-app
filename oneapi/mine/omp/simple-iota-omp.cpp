#include <stdlib.h>
#include <omp.h>
#include <iostream>
constexpr size_t array_size = 10000;
#pragma omp requires unified_shared_memory
int main(){
constexpr int value = 100000;
// Returns the default target device.
int deviceId = (omp_get_num_devices() > 0) ? omp_get_default_device() : omp_get_initial_device();
int *sequential = (int *)omp_target_alloc_host(array_size, deviceId);
int *parallel = (int *)omp_target_alloc(array_size, deviceId);
        for (size_t i = 0; i < array_size; i++)
                sequential[i] = value + i;
        #pragma omp target parallel for
        for (size_t i = 0; i < array_size; i++)
                parallel[i] = value + i;
        for (size_t i = 0; i < array_size; i++) {
         if (parallel[i] != sequential[i]) {
           std::cout << "Failed on device.\n";
           return -1;
         }
        }
        omp_target_free(sequential, deviceId);
        omp_target_free(parallel, deviceId);
        std::cout << "Successfully completed on device.\n";
         return 0;
}