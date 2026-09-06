#ifndef UTILS
#define UTILS

#include <cxxabi.h>
#include <cstdlib>
#include <typeinfo>
#include <vector>
#include <tuple>
#include <filesystem>
#include <cnpy.h> // read from npy

namespace fs = std::filesystem;

template <typename T>
void print_type(const T& value)
{
    int status;

    char* name = abi::__cxa_demangle(
        typeid(value).name(),
        nullptr,
        nullptr,
        &status
    );

    if (status == 0) {
        std::cout << name << '\n';
    }

    std::free(name);
}

template <typename indtype, typename outdtype>
std::vector<std::vector<outdtype>> read(const fs::path& input) {
    cnpy::NpyArray arr = cnpy::npy_load(input);
    indtype* data = arr.data<indtype>();
    std::vector<outdtype> vals(arr.num_vals);
    for (size_t i = 0; i < arr.num_vals; ++i) {
      vals[i] = static_cast<outdtype>(data[i]);
    }
    std::vector<std::vector<outdtype>> matrix(
        arr.shape[0],
        std::vector<outdtype>(arr.shape[1])
    );
    for (size_t i = 0; i < arr.shape[0]; ++i) {
        for (size_t j = 0; j < arr.shape[1]; ++j) {
            matrix[i][j] = vals[i*arr.shape[1]+j];
        }
    }
    return matrix;
}

#endif
