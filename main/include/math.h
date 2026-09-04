#ifndef MATH
#define MATH

#include <cstddef> // for matmul function
#include <vector>
#include <cmath>

std::vector<double> matmul(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::size_t m,
    std::size_t n,
    std::size_t p
) {
    assert(A.size() == m * n);
    assert(B.size() == n * p);
    std::vector<double> C(m * p, 0.0);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {

            double sum = 0.0;

            for (std::size_t k = 0; k < n; ++k) {
                sum += A[i * n + k]
                     * B[k * p + j];
            }

            C[i * p + j] = sum;
        }
    }
    return C;
}

std::vector<double> add(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::size_t m,
    std::size_t n
) {
  assert(A.size() == m * n);
  assert(B.size() == n);
  std::vector<double> C(m*n);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      C[i*n+k] = A[i*n+k]+B[k];
    }
  }
  return C;
}

std::vector<double> transpose(
    const std::vector<double>& A,
    std::size_t rows,
    std::size_t cols
) {
    std::vector<double> AT(cols * rows);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {

            AT[j * rows + i] =
                A[i * cols + j];
        }
    }
    return AT;
}

#endif
