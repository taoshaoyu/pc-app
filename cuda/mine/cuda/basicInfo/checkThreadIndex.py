import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule


def printMatrix(arr, nx, ny):
    print("\nMatrix: (%d.%d)"%(nx, ny))
    for i in range(nx):
        for j in range(ny):
            print("%3d"%(arr[i*ny+j]), end=" ")
        print("")


mod = SourceModule(
r"""
__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index"
           " %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
           ix, iy, idx, A[idx]);
}
"""
)

print("Starting...")

drv.init()
deviceCount = drv.Device.count()
dev = drv.Device(0)
print("Using Device %d: %s"%(0, dev.name()))

nx = 8
ny = 6
nxy = nx * ny
nBytes = nxy * 4

h_A = np.arange(nxy).astype(np.int32)

printMatrix(h_A, nx, ny)

#d_MatA = drv.mem_alloc( h_A.nbytes )
#drv.memcpy_htod(d_MatA, h_A)

block = (4,2,1)
grid = (int((nx + block[0] -1) / block[0]) , int((ny + block[1] - 1) / block[1]),1)

func = mod.get_function("printThreadIndex")

# func(d_MatA, np.uint32(nx), np.uint32(ny), block=block, grid=grid)
func(drv.In(h_A), np.uint32(nx), np.uint32(ny), block=block, grid=grid)




