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
    double eta
) {
    std::size_t rows = y_hat.size();
    std::size_t cols = y_hat[0].size();

    std::vector<std::vector<double>> gradient(
        rows,
        std::vector<double>(cols, 0.0)
    );

    for (std::size_t i = 0; i < rows; ++i) {
        std::size_t label = static_cast<std::size_t>(y[0][i]);
        double p = std::max(y_hat[i][label], 0.0);
        gradient[i][label] = -1.0 / (p * static_cast<double>(rows));
    }
    return gradient;
}
