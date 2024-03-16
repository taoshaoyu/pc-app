import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule

mod = DynamicSourceModule(r"""
/*
 * A simple example of nested kernel launches from the GPU. Each thread displays
 * its information when execution begins, and also diagnostics when the next
 * lowest nesting layer completes.
 */

__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid,
           blockIdx.x);

    // condition to stop recursive execution
    if (iSize == 1) return;

    // reduce block size to half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}
""", 
#options=["--relocatable-device-code", "true", "-lcudadevrt" ],
keep=True)

size = 8
blocksize = 8
igrid = 1

block = (blocksize, 1, 1)
grid = (int((size + block[0] - 1) / block[0]), 1)
print("%s Execution Configuration: grid %d block %d\n"%(
        "nestedHelloWorld", grid[0], block[0]))

func = mod.get_function("nestedHelloWorld")
func(np.uint32(block[0]), np.uint32(0), grid=grid,block=block)


