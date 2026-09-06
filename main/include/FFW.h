#ifndef FFW_H
#define FFW_H

#include <memory>
#include <vector>

#include "Layer.h"
#include "Loss.h"

class FFW {
public:
    FFW() = default;

    template <typename LayerType>
    void add(LayerType layer) {
        layers.push_back(std::make_unique<LayerType>(layer));
    }

    template<typename LossType>
    void loss(LossType loss) {
      loss_function = std::make_unique<LossType>(
          std::move(loss)
      );
    }

    void fit(
        const std::vector<std::vector<double>>& x,
        const std::vector<std::vector<double>>& y,
        const int& iterations,
        const double& eta
    );

    std::vector<std::vector<double>> operator()(
        const std::vector<std::vector<double>>& input
    );

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<double> loss_arr;
    std::unique_ptr<Loss> loss_function;
};

#endif
