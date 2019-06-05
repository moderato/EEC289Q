// Standard Library includes
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "cnpy.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

#include "kernel.cuh"

int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[4] = { 56, 56, 128, 128 };

  for (int i = 1; i < argc && i < 5; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 5; i < argc && i < 7; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  bool isColumnMajor = false;

  cudaError_t result = TestCutlassGemm(
    problem[0],     // H
    problem[1],     // W
    problem[2],     // IC (K)
    problem[3],     // OC (N)
    scalars[0],     // alpha
    scalars[1],     // beta
    isColumnMajor   // is column major
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}