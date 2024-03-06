# Sample code

- simple-iota-omp.cpp
  - From: <https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-0/example-compilation.html>
  
  - Build cmd

  ~~~plaintext
  icpx -fsycl simple-iota-omp.cpp -fiopenmp -fopenmp-targets=spir64 -o simple-iota
  ~~~
