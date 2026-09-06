#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "Layer.h"

#include <vector>

class Softmax : public Layer {
  public:
    std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& input
    ) override;

    std::vector<std::vector<double>> backward(
        const std::vector<std::vector<double>>& Z
    ) override;
  private:
    std::vector<std::vector<double>> dZ;
};

#endif
