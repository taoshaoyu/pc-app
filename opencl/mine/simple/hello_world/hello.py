import numpy as np
import pyopencl as cl

DATA_SIZE = 1024
PROGRAM_FILE = "hello.cl"
KERNEL_FUNC = "square"

if __name__ == '__main__':

    f = open(PROGRAM_FILE, 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Prepare input
    data = np.random.rand(DATA_SIZE).astype(np.float32)

    # Input -> Input buffer
    mf = cl.mem_flags
    input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)

    # Prepare output
    output = cl.Buffer(ctx, mf.WRITE_ONLY, DATA_SIZE*4)

    # Run program
    prg = cl.Program(ctx, kernels).build()
    kernel = prg.square
    evt = kernel(queue, (DATA_SIZE,), None, input, output, np.int32(DATA_SIZE))
    evt.wait()

    # Output buffer -> Output
    results = np.empty_like(data)
    cl.enqueue_copy(queue, results, output).wait()

    correct = 0
    for i in range(DATA_SIZE):
        if(results[i] == data[i] * data[i]):
            correct += 1
    print("Computed '%d/%d' correct values!\n" %(correct, DATA_SIZE));

#    print(np.sum(h_c))