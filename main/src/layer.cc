#include "layer.h"
#include "matrix.h"

#include <random>
#include <utility>

Layer::Layer(int input_dim, int hidden_layer_size):
  W(input_dim,hidden_layer_size),
  b(hidden_layer_size, 0.0)
{
  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 0.01);
  for (double& w: W) {
    w = dist(gen);
  }
}

void Layer::forward() {}

void Layer::backward() {}
