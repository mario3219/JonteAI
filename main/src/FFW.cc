#include "FFW.h"

void FFW::add(std::unique_ptr<Layer> layer)
{
    layers.push_back(std::move(layer));
}

void FFW::train(const int& iterations, const double& eta) {
  return;
}
