#include "ReLu.h"

#include <vector>

std::vector<std::vector<double>> ReLu::forward(
    const std::vector<std::vector<double>>& input
) {
  std::size_t rows = input.size();
  std::size_t cols = input[0].size();
  std::vector<std::vector<double>> output(rows, std::vector<double>(cols));
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      output[i][j] = std::max(double(0.0), input[i][j]);
    }
  }
  Z_prev = input;
  return output;
}


std::vector<std::vector<double>> ReLu::backward(
    const std::vector<std::vector<double>>& gradient,
    double /* eta */
) {
  auto dZ = gradient;
  for (std::size_t i = 0; i < dZ.size(); ++i) {
    for (std::size_t j = 0; j < dZ[0].size(); ++j) {
      if (Z_prev[i][j] <= 0.0) {
        dZ[i][j] = 0.0;
      }
    }
  }
  return dZ;
}
