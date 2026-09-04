#ifndef MATRIX_H
#define MATRIX_H

class Matrix {
  public:
    Matrix(int rows, int cols);
    Matrix(std::vector<double> matrix, int rows, int cols);
    double& operator[](size_t pos);
    std::vector<double> matrix;
    int rows;
    int cols;
}
