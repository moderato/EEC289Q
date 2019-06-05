// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/gemm.h"

// Defines cutlass::gemm::SgemmTraits, the structural components for single-precision GEMM
#include "cutlass/gemm/sgemm_traits.h"

#pragma warning( disable : 4503)

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  bool isColumnMajor) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size.
  //
  // Note, GemmTraits<> is a generic template defined for various general matrix product
  // computations within CUTLASS. It is intended to be maximally flexible, and consequently
  // it contains numerous template arguments.
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
  //

  typedef cutlass::gemm::SgemmTraits<
    cutlass::MatrixLayout::kRowMajor,   // layout of A matrix
    cutlass::MatrixLayout::kRowMajor,   // layout of B matrix
    cutlass::Shape<8, 128, 128>         // threadblock tile size
  > GemmTraits;

  // Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

  // Construct and initialize CUTLASS GEMM parameters object.
  //
  // One of CUTLASS's design patterns is to define parameters objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  typename Gemm::Params params;

  int result = params.initialize(
    M,     // GEMM M dimension
    N,     // GEMM N dimension
    K,     // GEMM K dimension
    alpha, // scalar alpha
    A,     // matrix A operand
    lda,
    B,     // matrix B operand
    ldb,
    beta,  // scalar beta
    C,     // source matrix C
    ldc,
    C,     // destination matrix C (may be different memory than source C matrix)
    ldc
  );

  if (result) {
    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
    return cudaErrorInvalidValue;
  }

  // Launch the CUTLASS GEMM kernel.
  Gemm::launch(params);

  // Return any errors associated with the launch or cudaSuccess if no error.
  return cudaGetLastError();
}

__global__ void RowMajorToColumnMajor(float* out, float* in, int num_rows, int num_cols) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  // transpose with boundary test
  if (ix < num_cols && iy < num_rows) {
    out[ix*num_rows+iy] = in[iy*num_cols+ix];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int H, int W, int N, int K, float alpha, float beta, bool isColumnMajor) {
  cudaError_t result;

  // Define several matrices to be used as operands to GEMM kernels.

  // Compute leading dimensions for each matrix.
  int M = H * W;
  int lda = isColumnMajor ? M : K;
  int ldb = isColumnMajor ? K : N;
  int ldc = M;
  int numElement = M * N;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * numElement;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *B_tmp;
  float *C_cutlass;
  float *C_reference;
  float *host_cutlass, *host_reference;

  // Allocate matrices in GPU device memory with arbitrary seeds.
  int input_shape = M * K;
  int filter_shape = K * N;
  int output_shape = M * N;

  // Allocate GPU arrays
  cudaMalloc((void**)&A, input_shape * sizeof(float));
  cudaMalloc((void**)&B, filter_shape * sizeof(float));
  cudaMalloc((void**)&B_tmp, filter_shape * sizeof(float));
  cudaMalloc((void**)&C_cutlass, output_shape * sizeof(float));
  cudaMalloc((void**)&C_reference, output_shape * sizeof(float));
  host_reference = (float *)malloc(output_shape * sizeof(float));
  host_cutlass = (float *)malloc(output_shape * sizeof(float));

  // Filenames
  std::string input_name = "../npy/conv_input_1_" + std::to_string(H) + "_" + std::to_string(W) + "_" + std::to_string(K) + ".npy";
  std::string filter_name = "../npy/conv_weight_1_1_" + std::to_string(K) + "_" + std::to_string(N) + ".npy";
  std::string output_name = "../npy/conv_output_1_" + std::to_string(H) + "_" + std::to_string(W) + "_" + std::to_string(N) + ".npy";

  // Load data and copy to GPU arrays
  float *tmp;

  // A
  cnpy::NpyArray input_npy = cnpy::npy_load(input_name);
  tmp = input_npy.data<float>();
  cudaMemcpy(A, tmp, input_shape * sizeof(float), cudaMemcpyHostToDevice);
  // printf("****\n");
  // for (int i = 0; i < input_shape / 10; i++) {
  //   printf("%d, %f\n", i, tmp[i]);
  // }

  // B
  cnpy::NpyArray filter_npy = cnpy::npy_load(filter_name);
  tmp = filter_npy.data<float>();
  cudaMemcpy(B_tmp, tmp, filter_shape * sizeof(float), cudaMemcpyHostToDevice);
  // Try transpose it, don't know why
  dim3 blockB(16, 16);
  dim3 gridB(
    (N + blockB.x - 1) / blockB.x,
    (K + blockB.y - 1) / blockB.y
  );
  RowMajorToColumnMajor<<<gridB, blockB>>>(B, B_tmp, K, N);
  cudaDeviceSynchronize();
  // for (int i = 0; i < filter_shape / 10; i++) {
  //   printf("%d, %f\n", i, tmp[i]);
  // }

  // C
  cnpy::NpyArray output_npy = cnpy::npy_load(output_name);
  tmp = output_npy.data<float>();
  cudaMemcpy(C_cutlass, tmp, output_shape * sizeof(float), cudaMemcpyHostToDevice);
  // printf("****\n");
  // for (int i = 0; i < output_shape / 10; i++) {
  //   printf("%d, %f\n", i, tmp[i]);
  // }
  // Column major result
  dim3 blockC(16, 16);
  dim3 gridC(
    (N + blockC.x - 1) / blockC.x,
    (M + blockC.y - 1) / blockC.y
  );
  // printf("%d, %d\n", M, N);
  RowMajorToColumnMajor<<<gridC, blockC>>>(C_reference, C_cutlass, M, N);
  cudaDeviceSynchronize();
  result = cudaMemcpy(host_reference, C_reference, output_shape * sizeof(float), cudaMemcpyDeviceToHost);

  // Row major result
  // cudaMemcpy(C_reference, tmp, output_shape * sizeof(float), cudaMemcpyHostToDevice);


////////////////////////////////////////////////////////////////
  // Launch CUTLASS GEMM.
  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, isColumnMajor);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;
    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B_tmp)
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_cutlass, C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B_tmp)
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  int count = 0;
  for(int i = 0; i < output_shape; i++) {
    // printf("%d, %f, %lf\n", i, host_cutlass[i], host_reference[i]);
    if (abs(host_cutlass[i] - host_reference[i]) > 2e-4) {
      // printf("%d, %f, %lf\n", i, host_cutlass[i], host_reference[i]);
      count++;
    }
  }
  printf("Wrong count: %d\n", count);

  // Free device memory allocations.
  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B_tmp)
  cudaFree(B);
  cudaFree(A);

  if (count > 0) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}