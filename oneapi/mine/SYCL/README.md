# Sample code

- simple-sycl-app
  - From: <https://developer.codeplay.com/products/oneapi/nvidia/2024.0.2/guides/get-started-guide-nvidia#use-dpc-to-target-nvidia-gpus>
  
  - Build cmd

  ~~~plaintext
  icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda simple-sycl-app.cpp -o simple-sycl-app
  ~~~

  - Run cmd

  ~~~plaintext
  ONEAPI_DEVICE_SELECTOR="ext_oneapi_cuda:*" SYCL_PI_TRACE=1 ./simple-sycl-app
  ~~~

- axpy.cpp
  - From: <https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-0/example-compilation.html>
  - Build cmd

  ~~~plaintext
  icpx -fsycl -I${MKLROOT}/include /EHsc -c axpy.cpp /Foaxpy.obj
  $icpx -fsycl axpy.o -fsycl-device-code-split=per_kernel \
  ${MKLROOT}/lib/intel64"/libmkl_sycl.a -Wl,-export-dynamic -Wl,--start-group \
  ${MKLROOT}/lib/intel64"/libmkl_intel_ilp64.a \
  ${MKLROOT}/lib/intel64"/libmkl_sequential.a \
  ${MKLROOT}/lib/intel64"/libmkl_core.a -Wl,--end-group -lsycl -lOpenCL \
  -lpthread -lm -ldl -o axpy.out
  ~~~

