import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import random
import time
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule

def initialData(size):
    return np.random.rand(size).astype(np.float32)

def sumMatrixOnHost(A, B, nx, ny):
    return A+B

def checkResult(hostRef, gpuRef, N):
    if (hostRef-gpuRef).any():
        print("Arrays do not match!\n", end="")
    else:
        print("Arrays match.\n\n", end="")


mod = SourceModule(
r"""
__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx )
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }


}
"""
)

print("Starting...")

drv.init()
deviceCount = drv.Device.count()
dev = drv.Device(0)
print("Using Device %d: %s"%(0, dev.name()))

nx = 1 << 14
ny = 1 << 14
nxy = nx * ny
nBytes = nxy * 4
print("Matrix size: nx %d ny %d\n"%(nx, ny), end="")


iStart=time.time()
h_A= initialData(nx * ny)
h_B= initialData(nx * ny)
iElaps = time.time() - iStart
print("initialize matrix elapsed %f sec\n"%(iElaps), end="")

ostRef = np.zeros(nxy).astype(np.float32)
gpuRef = np.zeros(nxy).astype(np.float32)

iStart = time.time()
hostRef = sumMatrixOnHost(h_A, h_B, nx, ny)
iElaps = time.time() - iStart
print("sumMatrixOnHost elapsed %f sec\n"%(iElaps),end="")

d_MatA = drv.mem_alloc( h_A.nbytes)
d_MatB = drv.mem_alloc( h_A.nbytes)
d_MatC = drv.mem_alloc( h_A.nbytes)

drv.memcpy_htod(d_MatA, h_A)
drv.memcpy_htod(d_MatB, h_B)

dimx = 32
block = (dimx, 1, 1)
grid = (int((nx + block[0] - 1) / block[0]), 1)



func = mod.get_function("sumMatrixOnGPU1D")
iStart=time.time()
func(d_MatA, d_MatB, d_MatC, np.uint32(nx), np.uint32(ny), block=block, grid=grid)
#func(drv.In(h_A), drv.In(h_B), drv.Out(gpuRef), np.uint32(nx), np.uint32(ny), block=block, grid=grid)
iElaps = time.time() - iStart
print("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n"%( 
        grid[0],
        grid[1],
        block[0], block[1], iElaps), end="")


drv.memcpy_dtoh(gpuRef, d_MatC)

checkResult(hostRef, gpuRef, nxy)




