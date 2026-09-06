#include <vector>
#include <random>

#include "Dense.h"

Dense::Dense(std::size_t input_dim, std::size_t hidden_layer_size): 
  W(input_dim,
  std::vector<double>(hidden_layer_size)),
  b(hidden_layer_size, 0.0)
  {
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 0.01);
    for (std::size_t i = 0; i < W.size(); i++) {
      for (std::size_t k = 0; k < W[0].size(); k++) {
        W[i][k] = dist(gen);
      }
    }
}

std::vector<std::vector<double>> Dense::forward(
    const std::vector<std::vector<double>>& input
) {
    return input;
}

std::vector<std::vector<double>> Dense::backward(
    const std::vector<std::vector<double>>& gradient,
    double eta
) {
    return gradient;
}
