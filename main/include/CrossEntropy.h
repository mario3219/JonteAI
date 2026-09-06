#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include "Loss.h"

class CrossEntropy : public Loss {
public:
    double operator()(
        const std::vector<std::vector<double>>& y_hat,
        const std::vector<std::vector<double>>& y
    ) override {
        double loss = 0.0;

        return loss;
    }
};

#endif
