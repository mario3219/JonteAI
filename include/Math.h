#ifndef MATH
#define MATH

#include <cstddef> // for matmul function
#include <vector>
#include <cmath>
#include <cassert>

std::vector<std::vector<double>> matmul(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::size_t p = B[0].size();

    assert(n == B.size());

    std::vector<std::vector<double>> C(m, std::vector<double>(p,0.0));

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

std::vector<std::vector<double>> add(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
) {
  std::size_t m = A.size();
  std::size_t n = A[0].size();
  assert(n == B[0].size());
  std::vector<std::vector<double>> C(m, std::vector<double>(n));
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      C[i][k] = A[i][k]+B[0][k];
    }
  }
  return C;
}

std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& A
) {
    std::size_t rows = A.size();
    std::size_t cols = A[0].size();
    std::vector<std::vector<double>> AT(cols, std::vector<double>(rows));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

#endif
