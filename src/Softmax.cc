#include "Softmax.h"

#include <vector>
#include <cmath>

std::vector<std::vector<double>> Softmax::forward(
    const std::vector<std::vector<double>>& input
) {
  std::size_t rows = input.size();
  std::size_t cols = input[0].size();

  std::vector<std::vector<double>> C(rows, std::vector<double>(cols));

  // Exponentiate and calculate row sum
  for (std::size_t i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
      C[i][j] = std::exp(input[i][j]);
      sum += C[i][j];
    }

    // Divide every element by row sum
    for (std::size_t j = 0; j < cols; ++j) {
      C[i][j] /= sum;
    }
  }
  return C;
}

std::vector<std::vector<double>> Softmax::backward(
    const std::vector<std::vector<double>>& Z,
    double /* eta */
) {
    // The softmax backwards is already implemented in CrossEntropy backwards
    return Z;
}
