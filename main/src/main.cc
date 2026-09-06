#include <iostream>
#include <filesystem>

#include <unordered_set>

#include "FFW.h"
#include "Utils.h"
#include "Dense.h"
#include "ReLu.h"
#include "Softmax.h"

namespace fs = std::filesystem;

int main() {

  fs::path data_dir = fs::current_path().parent_path() / "data";
  fs::path input_x = data_dir / "x.npy";
  fs::path input_y = data_dir / "y.npy";

  auto x = read<double,double>(input_x);
  auto y = read<int64_t,double>(input_y);

  int input_dim = x[0].size();
  int hidden_layer_size = 64;
  int num_classes = 2;

  FFW model;
  model.add(Dense(input_dim, 64));
  model.add(ReLu());
  model.add(Dense(64, num_classes));
  model.add(Softmax());

  auto y_hat = model(x);

  return 0;
}
