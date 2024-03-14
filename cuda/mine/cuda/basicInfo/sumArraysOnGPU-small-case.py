import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import random
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule

def initialData(size):
    return np.random.rand(size).astype(np.float32)

def sumArraysOnHost(A, B, N):
    return [A[idx]+B[idx] for idx in range(N) ]

def checkResult(hostRef, gpuRef, N):
    epsilon = 1.0E-8
    match = 1

    for i in range(N):
        if abs(hostRef[i] - gpuRef[i]) > 0: #epsilon:
            match = 0
            print("Arrays do not match!\n", end="");
            print("host %5.2f gpu %5.2f at current %d\n"%(
                hostRef[i],
                gpuRef[i], i))
            break
    if match:
        print("Arrays match.\n\n", end="")
    return


mod = SourceModule(
r"""
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}
"""
)

print("Starting...")

drv.init()
deviceCount = drv.Device.count()
dev = drv.Device(0)

nElem = 1 << 5
print("Vector size %d"%(nElem))

h_A = initialData(nElem)
h_B = initialData(nElem)

hostRef = np.zeros(nElem).astype(np.float32)
gpuRef = np.zeros(nElem).astype(np.float32)

d_A = drv.mem_alloc( h_A.nbytes)
d_B = drv.mem_alloc( h_A.nbytes)
d_C = drv.mem_alloc( h_A.nbytes)

drv.memcpy_htod(d_A, h_A)
drv.memcpy_htod(d_B, h_B)
drv.memcpy_htod(d_C, gpuRef)

block = (nElem,1,1)
grid = (1,1)

func = mod.get_function("sumArraysOnGPU")

#func(d_A, d_B, d_C, np.uint32(nElem), block=block, grid=grid)
#drv.memcpy_dtoh(gpuRef, d_C)

func(drv.In(h_A), drv.In(h_B), drv.Out(gpuRef), np.uint32(nElem), block=block, grid=grid)

hostRef = sumArraysOnHost(h_A, h_B, nElem)

checkResult(hostRef, gpuRef, nElem)




