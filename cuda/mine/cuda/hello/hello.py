import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule

mod = SourceModule(r"""
__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}
""")

print("Hello World from CPU!")
helloFromGPU = mod.get_function("helloFromGPU")


helloFromGPU(grid=(1,1),block=(10,1,1))