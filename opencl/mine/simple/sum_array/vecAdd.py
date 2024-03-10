import numpy as np
import pyopencl as cl

N = 100000

if __name__ == '__main__':

    f = open('vecAdd.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Prepare input
    h_a = np.power(np.sin(np.arange(N)).astype(np.float64),2)
    h_b = np.power(np.sin(np.arange(N)).astype(np.float64),2)

    # Input -> Input buffer
    mf = cl.mem_flags
    d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
    d_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)

    # Prepare output
    d_c = cl.Buffer(ctx, mf.WRITE_ONLY, N*8)

    # Run program
    prg = cl.Program(ctx, kernels).build()
    kernel = prg.vecAdd
    evt = kernel(queue, (N,), None, d_a, d_b, d_c, np.int32(N))
    evt.wait()

    # Output buffer -> Output
    h_c = np.empty_like(h_a)
    cl.enqueue_copy(queue, h_c, d_c).wait()

    print(np.sum(h_c))