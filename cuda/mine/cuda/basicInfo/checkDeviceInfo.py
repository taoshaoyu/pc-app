import pycuda.driver as drv
import math

print("Starting...")

drv.init()
deviceCount = drv.Device.count()
if deviceCount == 0:
    print("There are no available device(s) that support CUDA");
else:
    print("Detected %d CUDA Capable device(s)"%(drv.Device.count()))

device=drv.Device(0)
print("Device %d: \"%s\""%(0, device.name()))

drv_vesion = drv.get_driver_version()
print("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d"%(
           drv_vesion / 1000, (drv_vesion % 100) / 10,
           drv_vesion / 1000, (drv_vesion % 100) / 10))   #FIXME: runtimeVersion ?

print("  CUDA Capability Major/Minor version number:    %d.%d"%(
           device.get_attribute(drv.device_attribute.COMPUTE_CAPABILITY_MAJOR), 
           device.get_attribute(drv.device_attribute.COMPUTE_CAPABILITY_MINOR)))

print("  Total amount of global memory:                 %5.2f MBytes (%d "
           "bytes)"%( device.total_memory() / math.pow(1024.0,3),
           device.total_memory()))

print("  GPU Clock rate:                                %.0f MHz (%0.2f "
           "GHz)"%(
            device.get_attribute(drv.device_attribute.CLOCK_RATE) * 0.001,
            device.get_attribute(drv.device_attribute.CLOCK_RATE) * 0.000001))

print("  Memory Clock rate:                             %.0f Mhz"%(
           device.get_attribute(drv.device_attribute.MEMORY_CLOCK_RATE) * 0.001))

print("  Memory Bus Width:                              %d-bit"%(
           device.get_attribute(drv.device_attribute.GLOBAL_MEMORY_BUS_WIDTH)))

if device.get_attribute(drv.device_attribute.L2_CACHE_SIZE) != 0:
    print("  L2 Cache Size:                                 %d bytes"%(
               device.get_attribute(drv.device_attribute.L2_CACHE_SIZE)))

print("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
           "2D=(%d,%d), 3D=(%d,%d,%d)"%(
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE1D_WIDTH),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE2D_WIDTH),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE2D_HEIGHT),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE3D_WIDTH),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE3D_HEIGHT),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE3D_DEPTH)
    ))

print("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
           "2D=(%d,%d) x %d"%(
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE1D_LAYERED_WIDTH),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE1D_LAYERED_LAYERS),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE2D_LINEAR_WIDTH),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE2D_HEIGHT),
            device.get_attribute(drv.device_attribute.MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES),
        ))

print("  Total amount of constant memory:               %lu bytes"%(
           device.get_attribute(drv.device_attribute.TOTAL_CONSTANT_MEMORY)))

print("  Total amount of shared memory per block:       %lu bytes"%(
           device.get_attribute(drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)))

print("  Total number of registers available per block: %d"%(
           device.get_attribute(drv.device_attribute.MAX_REGISTERS_PER_BLOCK)))

print("  Warp size:                                     %d"%(
           device.get_attribute(drv.device_attribute.WARP_SIZE)))

print("  Maximum number of threads per multiprocessor:  %d"%(
           device.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)))

print("  Maximum sizes of each dimension of a block:    %d x %d x %d"%(
           device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X),
           device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_Y),
           device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_Z)))

print("  Maximum sizes of each dimension of a grid:     %d x %d x %d"%(
           device.get_attribute(drv.device_attribute.MAX_GRID_DIM_X),
           device.get_attribute(drv.device_attribute.MAX_GRID_DIM_Y),
           device.get_attribute(drv.device_attribute.MAX_GRID_DIM_Z)))

print("  Maximum memory pitch:                          %lu bytes"%(
           device.get_attribute(drv.device_attribute.MAX_PITCH)))