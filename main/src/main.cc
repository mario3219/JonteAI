#include <iostream>
#include <filesystem>

#include <unordered_set>

#include "FFW.h"
#include "Utils.h"
#include "Dense.h"

namespace fs = std::filesystem;

int main() {

  fs::path data_dir = fs::current_path().parent_path() / "data";
  fs::path input_x = data_dir / "x.npy";
  fs::path input_y = data_dir / "y.npy";

  auto x = read<double,double>(input_x);
  auto y = read<int64_t,double>(input_y);

  int input_dim = x.size();
  int hidden_layer_size = 64;
  int num_classes = 2;

  FFW model;
  model.add(std::make_unique<Dense>(input_dim, hidden_layer_size));
  model.add(std::make_unique<Dense>(input_dim, num_classes));
  return 0;
}
