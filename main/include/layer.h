#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer {
  public:
    Layer(int input_dim, int hidden_layer_size);
    std::vector<double> operator[](size_t row);
  private:
    std::vector<std::vector<double>> W;
    std::vector<double> b;
};

#endif
