#ifndef LAYER
#define LAYER

#include <vector>
#include <utility>
#include "matrix.h"

class Layer {
  public:
    Layer(int input_dim, int hidden_layer_size);
    std::pair<int,int> shape();
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& input);
    
  private:
    Matrix W;
    std::vector<double> b;
};

#endif
