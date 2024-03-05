#include <CL/sycl.hpp>
#include <iostream>
constexpr int num=16;
using namespace sycl;
int main() {
    auto r = range{num};
    buffer<int> a{r};
    queue{}.submit([&](handler& h) {
        accessor out{a, h};
        h.parallel_for(r, [=](item<1> idx) {
            out[idx] = idx;
        });
    });
    host_accessor result{a};
    for (int i=0; i<num; ++i)
    std::cout << result[i] << "\n";
}

