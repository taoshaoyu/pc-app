import numpy as np
import pyopencl as cl

TASKS = 64

if __name__ == '__main__':

    f = open('add_numbers.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    data = np.arange(TASKS).astype(np.float32)

    # Input -> Input buffer
    mf = cl.mem_flags
    input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    local_buf=cl.LocalMemory(4*4)
    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, 4*4)

    prg = cl.Program(ctx, kernels).build()
    kernel = prg.add_numbers
    evt = kernel(queue, (TASKS,), None, input_buf, local_buf, output_buf)
    evt.wait()

    # Output buffer -> Output
    output_var = np.empty((4,), dtype=np.float32)
    cl.enqueue_copy(queue, output_var, output_buf).wait()

    print(np.sum(output_var))
