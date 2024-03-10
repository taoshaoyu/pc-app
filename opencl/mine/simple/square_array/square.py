import numpy as np
import pyopencl as cl

ARRAY_SIZE = 100000000

if __name__ == '__main__':

    f = open('square.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Prepare input
    hdata = np.arange(ARRAY_SIZE).astype(np.float32)

    # Input -> Input buffer
    mf = cl.mem_flags
    ddata = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hdata)

    # Prepare output
    doutput = cl.Buffer(ctx, mf.WRITE_ONLY, ARRAY_SIZE*4)

    # Run program
    prg = cl.Program(ctx, kernels).build()
    kernel = prg.square
    evt = kernel(queue, (ARRAY_SIZE,), None, ddata, doutput, np.int32(ARRAY_SIZE))
    evt.wait()

    # Output buffer -> Output
    houtput = np.empty_like(hdata)
    cl.enqueue_copy(queue, houtput, doutput).wait()

#    print(np.sum(houtput))