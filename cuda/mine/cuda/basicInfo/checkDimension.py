import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule

mod = SourceModule(
r"""
__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}
"""
)

nElem = 6
block = (3,1,1)
grid = (int((nElem + block[0] - 1) / block[0]), 1,1)

print("grid.x %d grid.y %d grid.z %d"%(grid[0], grid[1], grid[2]))
print("block.x %d block.y %d block.z %d"%(block[0], block[1], block[2]))

checkIndex = mod.get_function("checkIndex")

checkIndex(grid=grid,block=block)