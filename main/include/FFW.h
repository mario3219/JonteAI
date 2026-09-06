#ifndef FFW_H
#define FFW_H

#include <memory>
#include <vector>

#include "Layer.h"

class FFW {
public:
    FFW() = default;

    template <typename LayerType>
    void add(LayerType layer) {
        layers.push_back(std::make_unique<LayerType>(layer));
    }
    void train(const int& iterations, const double& eta);
    std::vector<std::vector<double>> operator()(
        const std::vector<std::vector<double>>& input
    );

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<std::vector<double>> Qs;
    std::vector<double> loss;
};

#endif
