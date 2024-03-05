#include <vector> // std::vector()
#include <cstdlib> // std::rand()
#include <CL/sycl.hpp>
#include "oneapi/mkl/blas.hpp"

int main(int argc, char* argv[]) {
    double alpha = 2.0;
    int n_elements = 1024;
    int incx = 1;
    std::vector<double> x;
    x.resize(incx * n_elements);
    for (int i=0; i<n_elements; i++)
    x[i*incx] = 4.0 * double(std::rand()) / RAND_MAX - 2.0;
    // rand value between -2.0 and 2.0
    int incy = 3;
    std::vector<double> y;
    y.resize(incy * n_elements);
    for (int i=0; i<n_elements; i++)
    y[i*incy] = 4.0 * double(std::rand()) / RAND_MAX - 2.0;
    // rand value between -2.0 and 2.0
    cl::sycl::device my_dev;
    try {
    my_dev = cl::sycl::device(cl::sycl::gpu_selector());
    } catch (...) {
        std::cout << "Warning, failed at selecting gpu device. Continuing on default(host) device.\n";
    }
    // Catch asynchronous exceptions
    auto exception_handler = [] (cl::sycl::exception_list
    exceptions) {
    for (std::exception_ptr const& e : exceptions) {
    try {
    std::rethrow_exception(e);
    } catch(cl::sycl::exception const& e) {
    std::cout << "Caught asynchronous SYCL exception:\n";
    std::cout << e.what() << std::endl;
    }
    }
    };
    cl::sycl::queue my_queue(my_dev, exception_handler);
    cl::sycl::buffer<double, 1> x_buffer(x.data(), x.size());
    cl::sycl::buffer<double, 1> y_buffer(y.data(), y.size());
    // perform y = alpha*x + y
    try {
    oneapi::mkl::blas::axpy(my_queue, n_elements, alpha, x_buffer,
    incx, y_buffer, incy);
    }
    catch(cl::sycl::exception const& e) {
    std::cout << "\t\tCaught synchronous SYCL exception:\n"
    << e.what() << std::endl;
    }
    std::cout << "The axpy (y = alpha * x + y) computation is complete!" << std::endl;
    // print y_buffer
    auto y_accessor = y_buffer.template
    get_access<cl::sycl::access::mode::read>();
    std::cout << std::endl;
    std::cout << "y" << " = [ " << y_accessor[0] << " ]\n";
    std::cout << " [ " << y_accessor[1*incy] << " ]\n";
    std::cout << " [ " << "... ]\n";
    std::cout << std::endl;
    return 0;
}