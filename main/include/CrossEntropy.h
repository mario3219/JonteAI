#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include "Loss.h"

#include <cmath>

class CrossEntropy : public Loss {
public:
    double operator()(
        const std::vector<std::vector<double>>& y_hat,
        const std::vector<std::vector<double>>& y
    ) override {
        size_t num_examples = y[0].size();
        double loss = 0.0;
        for (std::size_t i = 0; i < num_examples; ++i) {
          std::size_t label = static_cast<std::size_t>(y[0][i]);
          double p = y_hat[i][label];
          loss += -std::log(p);
        }
        return loss/num_examples;
    }
};

#endif
