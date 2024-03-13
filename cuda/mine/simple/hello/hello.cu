#include <stdio.h>

__global__ void helloFromGPU(void){
    printf("hello world from GPU, blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
}

__global__ void dimTest(void){
    int block_idx = blockIdx.x + gridDim.x * gridDim.y * blockIdx.y; 
    int thread_idx = threadIdx.x + blockDim.x * blockDim.y * threadIdx.y;
//    printf("block_idx, thread_idx (%d, %d)\n", block_idx, thread_idx);
    printf("blockIdx, threadIdx (%d, %d, %d, %d) gridm(%d, %d, %d), blockDim(%d, %d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
}


int main(void){
    printf("hello world from CPU\n");
    helloFromGPU <<<2,3>>>();
    dim3 grid(3,2);
    dim3 blockdim(2,3);
    dimTest<<<grid,blockdim>>>();
    
    printf("%d\n",grid);

    cudaDeviceReset();
    return 0;
}