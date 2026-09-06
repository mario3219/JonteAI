#include <vector>
#include <random>

#include "Dense.h"
#include "Math.h"

Dense::Dense(std::size_t input_dim, std::size_t hidden_layer_size): 
  W(input_dim,
  std::vector<double>(hidden_layer_size)),
  b(1,
  std::vector<double>(hidden_layer_size))
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
    Q_prev = input;
    return add(matmul(input,W), b);
}

std::vector<std::vector<double>> Dense::backward(
    const std::vector<std::vector<double>>& dZ,
    double eta
) {
    std::size_t num_examples = dZ.size();
    std::size_t hidden_layer_size = dZ[0].size();

    std::vector<std::vector<double>> dW = matmul(
        transpose(Q_prev),
        dZ
    );
    std::vector<std::vector<double>> db(1,std::vector<double>(b[0].size()));
    for (std::size_t i = 0; i < num_examples; ++i) {
      for (std::size_t j = 0; j < hidden_layer_size; ++j) {
        db[0][j] += dZ[i][j];
      }
    }

    std::vector<std::vector<double>> dZ_next = matmul(
      dZ,
      transpose(W)
    );

    for (std::size_t i = 0; i < W.size(); ++i) {
      for (std::size_t j = 0; j < W[0].size(); ++j) {
        W[i][j] -= eta*dW[i][j];
      }
    }
    for (std::size_t i = 0; i < b[0].size(); ++i) {
      b[0][i] -= eta*db[0][i];
    }

    return dZ_next;
}
