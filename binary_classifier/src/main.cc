#include <iostream>
#include <filesystem> // paths
#include <unordered_set>  // to get unique values
#include <random> // weight initializer
#include <vector>
#include <numeric>
#include <tuple>

// relu
#include <algorithm>

#include <cnpy.h> // read from npy
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

std::vector<double> softmax(
    const std::vector<double>& A,
    std::size_t rows,
    std::size_t cols
) {
    std::vector<double> C(A.size());

    for (std::size_t i = 0; i < rows; ++i) {

        double sum = 0.0;

        // exponentiate and calculate row sum
        for (std::size_t j = 0; j < cols; ++j) {
            C[i * cols + j] = std::exp(A[i * cols + j]);
            sum += C[i * cols + j];
        }

        // divide every element by row sum
        for (std::size_t j = 0; j < cols; ++j) {
            C[i * cols + j] /= sum;
        }
    }

    return C;
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
  fs::path out_loss = src_dir / "data" / "loss.npy";

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

  std::vector<double> dL2out;
  std::vector<double> dL1;
  std::vector<double> dW1;
  std::vector<double> dW2;
  double db1;
  double db2;


  for (double& w : W1) {
      w = dist(gen);
  }
  for (double& w : W2) {
      w = dist(gen);
  }
  
  double eta = 0.1;
  int iterations = 25000;
  int num_examples = x_shape[0];
  std::vector<double> loss(iterations, 0.0);

  std::vector<double> L0 = x;
  for (int i = 0; i < iterations; i++) {

    std::cout << i << "/" << iterations << std::endl;
    
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
    ); // shape x_shape[0], num_classes
    loss[i] = cross_entropy(L2_out, y, num_examples, num_classes);

    dL2out = L2_out; //shape num_examples, num_classes
    for (std::size_t i = 0; i < num_examples; ++i) {
        dL2out[i * num_classes + y[i]] -= 1.0;
    }
    for (double& val: dL2out) {
      val = val/num_examples;
    }
    dW2 = matmul(
      transpose(L1_out, x_shape[0], hidden_layer_size),
      dL2out,
      hidden_layer_size,
      x_shape[0],
      num_classes
    );
    db2 = accumulate(dL2out.begin(), dL2out.end(), 0.0, std::plus<double>());
    dL1 = matmul(
      dL2out,
      transpose(W2, hidden_layer_size, num_classes),
      num_examples,
      num_classes,
      hidden_layer_size
    ); //shape num_classes, hidden_layer_size
    for (double& val: dL1) {
      if (val <= 0) {
        val = 0;
      }
    }
    dW1 = matmul(
      transpose(x, x_shape[0], x_shape[1]),
      dL1,
      x_shape[1],
      x_shape[0],
      hidden_layer_size
    ); //shape x_shape[1], hidden_layer_size
    db1 = accumulate(dL1.begin(), dL1.end(), 0.0, std::plus<double>());

    for (double& val: b1) {
      val = val-eta*db1;
    }
    for (double& val: b2) {
      val = val-eta*db2;
    }
    for (std::size_t k = 0; k < W1.size(); ++k) {
      W1[k] -= eta*dW1[k];
    }
    for (std::size_t k = 0; k < W2.size(); ++k) {
      W2[k] -= eta*dW2[k];
    } 
  }
  cnpy::npy_save(
      out_loss,
      loss.data(),
      {loss.size()},
      "w"
  );
}
