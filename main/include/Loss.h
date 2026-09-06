#ifndef LOSS_H
#define LOSS_H

#include <vector>

class Loss {
public:
    virtual ~Loss() = default;

    virtual double operator()(
        const std::vector<std::vector<double>>& y_hat,
        const std::vector<std::vector<double>>& y
    ) = 0;
};

#endif
