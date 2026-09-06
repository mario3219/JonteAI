#ifndef RELU_H
#define RELU_H

#include "Layer.h"

class ReLu : public Layer {
  public:
    std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& input
    ) override;

    std::vector<std::vector<double>> backward(
        const std::vector<std::vector<double>>& gradient,
        double /* eta */
    ) override;
  private:
    std::vector<std::vector<double>> Z_prev;
};

#endif
