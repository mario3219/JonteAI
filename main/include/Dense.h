#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"

class Dense : public Layer {
  public:
    Dense(std::size_t input_dim, std::size_t hidden_layer_size);

    std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& input
    ) override;

    std::vector<std::vector<double>> backward(
        const std::vector<std::vector<double>>& gradient,
        double eta
    ) override;

    std::vector<std::vector<double>> W;
    std::vector<double> b;
};

#endif
