#include "layer.h"
#include <vector>
#include <random>

Layer::Layer(int input_dim, int hidden_layer_size): 
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

std::vector<double> Layer::operator[](size_t row) {
  return W[row];
}
