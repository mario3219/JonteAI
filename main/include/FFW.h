#ifndef FFW_H
#define FFW_H

#include <memory>
#include <vector>

#include "Layer.h"

class FFW {
public:
    FFW() = default;

    void add(std::unique_ptr<Layer> layer);

    void train(const int& iterations, const double& eta);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<std::vector<double>> Qs;
    std::vector<double> loss;
};

#endif
