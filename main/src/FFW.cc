#include "FFW.h"

#include <vector>
#include <iostream>

void FFW::fit(
    const std::vector<std::vector<double>>& x,
    const std::vector<std::vector<double>>& y,
    const int& iterations,
    const double& eta) {

  loss_arr.resize(iterations, 0.0);
  std::vector<std::vector<double>> y_hat = x;

  int progress_step = std::max(1, iterations / 20);
  std::cout << "0% complete\n";

  for (int i = 0; i < iterations; i++) {

    if ((i + 1) % progress_step == 0 || i + 1 == iterations) {
        int progress = (i + 1) * 100 / iterations;
        std::cout << progress << "% complete\n";
    }

    std::vector<std::vector<double>> y_hat = x;

    for (auto& layer: layers) {
      y_hat = layer->forward(y_hat);
    }
    loss_arr[i] = (*loss_function)(y_hat,y);
  }
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
