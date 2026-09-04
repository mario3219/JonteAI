#include "matrix.h"

Matrix::Matrix(int rows, int cols): rows(rows), cols(cols), matrix(rows*cols) {}
Matrix::Matrix(std::vector<double> matrix, int rows, int cols): matrix(matrix), rows(rows), cols(cols) {}

double& Matrix::operator[](size_t pos) {
  return matrix[pos];
}
