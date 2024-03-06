# Sample code
- simple-sycl-app
  - From: https://developer.codeplay.com/products/oneapi/nvidia/2024.0.2/guides/get-started-guide-nvidia#use-dpc-to-target-nvidia-gpus
~~~
$ icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda simple-sycl-app.cpp -o simple-sycl-app
~~~
