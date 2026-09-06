#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer {
public:
    virtual ~Layer() = default;

    virtual std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& input
    ) = 0;

    virtual std::vector<std::vector<double>> backward(
        const std::vector<std::vector<double>>& gradient,
        double eta
    ) = 0;
};

#endif
