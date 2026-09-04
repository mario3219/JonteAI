#include <iostream>
#include <filesystem> // paths
#include <unordered_set>  // to get unique values
#include <random> // weight initializer
#include <vector>
#include <numeric>
#include <tuple>

// relu
#include <algorithm>

#include <cnpy.h> // read from csv
#include "utils.h"
#include "math.h"

namespace fs = std::filesystem;

template <typename T>
std::vector<T> relu(const std::vector<T>& x)
{
    std::vector<T> out(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::max(T{0}, x[i]);
    }
    return out;
}

double cross_entropy(
    const std::vector<double>& probs,
    const std::vector<double>& y,
    std::size_t num_examples,
    std::size_t num_classes
) {
    double loss = 0.0;

    for (std::size_t i = 0; i < num_examples; ++i) {

        std::size_t label =
            static_cast<std::size_t>(y[i]);

        double p =
            probs[i * num_classes + label];

        loss += -std::log(p);
    }

    return loss / num_examples;
}

int main() { 

  fs::path src_dir = fs::current_path().parent_path();
  fs::path input_x = src_dir / "data" / "x.npy";
  fs::path input_y = src_dir / "data" / "y.npy";

  auto [x, x_shape] = read<double,double>(input_x);
  auto [y, y_shape] = read<int64_t,double>(input_y);

  int input_dim = x_shape[1];
  int num_classes = std::unordered_set<double>(y.begin(),y.end()).size();

  int hidden_layer_size = 64;

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 0.01);

  std::vector<double> W1(input_dim * hidden_layer_size);
  std::vector<double> b1(hidden_layer_size, 0.0);
  std::vector<double> W2(hidden_layer_size * num_classes);
  std::vector<double> b2(num_classes, 0.0);

  for (double& w : W1) {
      w = dist(gen);
  }
  for (double& w : W2) {
      w = dist(gen);
  }
  
  double eta = 0.1;
  int iterations = 25000;
  int num_examples = x_shape[0];
  std::vector<double> loss(num_examples, 0.0);

  std::vector<double> L0 = x;
  //for (int i = 0; i <= iterations; i++) {
    
    std::vector<double> L1_in = add(
      matmul(L0, W1, x_shape[0], x_shape[1], hidden_layer_size),
      b1,
      x_shape[0], hidden_layer_size
    ); // shape x_shape[0], hidden_layer_size
 
    std::vector<double> L1_out = relu(L1_in);

    std::vector<double> L2_in = add(
      matmul(L1_out, W2, x_shape[0], hidden_layer_size, num_classes),
      b2,
      x_shape[0], num_classes
    ); // shape x_shape[0], num_classes

    std::vector<double> L2_out = softmax(
      L2_in, x_shape[0], num_classes
    );
    //loss[i] = cross_entropy(L2_out, y, num_examples, num_classes);
    loss[0] = cross_entropy(L2_out, y, num_examples, num_classes);
  //}

}
