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

    // Checks wether the input data is a matrix or a vector
    // If vector, creates a matrix with only 1 row
    std::size_t rows;
    std::size_t cols;

    if (arr.shape.size() == 1) {
        rows = 1;
        cols = arr.shape[0];
    } else {
        rows = arr.shape[0];
        cols = arr.shape[1];
    }

    std::vector<std::vector<outdtype>> matrix(
        rows,
        std::vector<outdtype>(cols)
    );

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix[i][j] = vals[i * cols + j];
        }
    }
    return matrix;
}

#endif
