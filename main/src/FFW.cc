#include "FFW.h"

#include <vector>

void FFW::train(const int& iterations, const double& eta) {
  return;
}

std::vector<std::vector<double>> FFW::operator()(
    const std::vector<std::vector<double>>& input) 
{
  auto output = input;
  for (auto& layer: layers) {
    output = layer->forward(output);
  }
  return output;
}
