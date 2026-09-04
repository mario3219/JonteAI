#ifndef UTILS
#define UTILS

#include <cxxabi.h>
#include <cstdlib>
#include <typeinfo>
#include <vector>
#include <tuple>
#include <filesystem> // paths

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

template <typename T>
void print_matrix(
    const std::vector<T>& matrix,
    size_t rows,
    size_t cols
) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << '\t';
        }
        std::cout << '\n';
    }
}

template <typename indtype, typename outdtype>
std::tuple<
  std::vector<outdtype>, std::vector<unsigned long, std::allocator<unsigned long>>
> read(const fs::path& input) {
    cnpy::NpyArray arr = cnpy::npy_load(input);
    indtype* data = arr.data<indtype>();
    std::vector<outdtype> vals(arr.num_vals);
    for (size_t i = 0; i < arr.num_vals; ++i) {
      vals[i] = static_cast<outdtype>(data[i]);
    }
    return {vals, arr.shape};
}

#endif
