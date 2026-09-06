#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.h"

class Softmax : public Layer {
public:
    std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& input
    ) override;

    std::vector<std::vector<double>> backward(
        const std::vector<std::vector<double>>& gradient,
        double eta
    ) override;
};

#endif
