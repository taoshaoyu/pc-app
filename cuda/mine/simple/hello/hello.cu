#include <stdio.h>

__global__ void helloFromGPU(void){
    printf("hello world from GPU\n");
}

int main(void){
    printf("hello world from CPU\n");
    helloFromGPU <<<1,3>>>();
    cudaDeviceReset();
    return 0;
}