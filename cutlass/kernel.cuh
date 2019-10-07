// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/gemm.h"

// Defines cutlass::gemm::SgemmTraits, the structural components for single-precision GEMM
#include "cutlass/gemm/sgemm_traits.h"

// Block swizzle
#include "cutlass/gemm/threadblock_swizzle.h"

#pragma warning( disable : 4503)

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmTT(
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


  // typedef cutlass::gemm::SgemmTraits<
  //   cutlass::MatrixLayout::kRowMajor,   // layout of A matrix
  //   cutlass::MatrixLayout::kRowMajor,   // layout of B matrix
  //   cutlass::Shape<8, 128, 128>         // threadblock tile size
  // > GemmTraits;

  // typedef cutlass::gemm::SgemmTraits<
  //   /// The layout for A.
  //   cutlass::MatrixLayout::kRowMajor,
  //   /// The layout for B.
  //   cutlass::MatrixLayout::kRowMajor,
  //   /// The output tile.
  //   cutlass::Shape<8, 128, 128>,
  //   /// The functor to use in the epilogue.
  //   cutlass::gemm::LinearScaling<float>,
  //   /// Tile size for thread-level GEMM (K-by-N-by-M)
  //   cutlass::Shape<8, 8, 8>,
  //   /// The number of floats loaded in one LDG for A.
  //   1,
  //   /// The number of floats loaded in one LDG for B.
  //   1,
  //   /// The index.
  //   int,
  //   /// The SGEMM config.
  //   cutlass::gemm::SgemmConfig<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, 1, 1, false>,
  //   /// The traits class for the epilogue.
  //   cutlass::gemm::SimplifiedGemmEpilogueTraits<
  //     cutlass::gemm::SgemmConfig<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, 1, 1, false>, 
  //     cutlass::gemm::LinearScaling<float>, 
  //     int> > GemmTraits;




template <
    ////////////////////////////
    /// The layout for A.
    cutlass::MatrixLayout::Kind kLayoutA_,

    ////////////////////////////
    /// The layout for B.
    cutlass::MatrixLayout::Kind kLayoutB_,

    ////////////////////////////
    /// Epilogue functor
    typename EpilogueFunctor_ = cutlass::gemm::LinearScaling<float>,

    ////////////////////////////
    /// The index
    typename Index_ = int,

    // OutputTile: KxNxM
    // ThreadGemmShape: KxNxM for thread level GEMM
    // global load/store, shared load/store
    // -> GemmGemmTileTraitsHelperA/B
    // -> GemmConfig
    // -> OutputTile, kScalarPerLdgA/B, ThreadGemmShape, 

    // ~~~~~~~~~~~~
    /// The config for the GEMM.
    typename GemmConfig_,
    // The configuration for the A matrix.
    typename GemmTileTraitsHelperA_ = GemmTileTraitsHelperA<kLayoutA_, GemmConfig_>,
    // The configuration for the B matrix.
    typename GemmTileTraitsHelperB_ = GemmTileTraitsHelperB<kLayoutB_, GemmConfig_>,
    // The helper class to create the streams and iterators.
    typename Helper_ =
        SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA_, GemmTileTraitsHelperB_, Index_>,
    /// The traits class for the epilogue.
    typename GemmEpilogueTraits_ =
        SimplifiedGemmEpilogueTraits<GemmConfig_, EpilogueFunctor_, Index_>,
    // Epilogue function.
    typename Epilogue_ = GemmEpilogue<GemmEpilogueTraits_> >
struct SpecialGemmTraits : public SimplifiedGemmTraits<
                         // The layout for A.
                         kLayoutA_,
                         // The layout for B.
                         kLayoutB_,
                         // The config.
                         GemmConfig_,
                         // The epilogue.
                         GemmEpilogue<GemmEpilogueTraits_>,
                         // The index.
                         Index_> {};


// public GemmTraits<
//   // The config.
//   GemmConfig_,
//   // The stream to load A from global memory to shared memory.
//   typename Helper_::GlobalLoadStreamA,
//   // The stream to load B from global memory to shared memory.
//   typename Helper_::GlobalLoadStreamB,
//   // The stream to load A from shared memory.
//   typename Helper_::SharedLoadStreamA,
//   // The stream to load B from shared memory.
//   typename Helper_::SharedLoadStreamB,
//   // The epilogue.
//   Epilogue_,
//   // The block swizzle to reorganize the grid.
//   cutlass::gemm::IdentityBlockSwizzle,
//   // The index.
//   Index_,
//   // The tool used to clear accumulators.
//   ClearAccumulators<typename GemmConfig_::Accumulators::Element> > {
// };

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

  // C
  cnpy::NpyArray output_npy = cnpy::npy_load(output_name);
  tmp = output_npy.data<float>();
  cudaMemcpy(C_cutlass, tmp, output_shape * sizeof(float), cudaMemcpyHostToDevice);
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
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms = 0;
  int repeatition = 1000;

  for (int i = 0; i < repeatition; i++) {
    float tmp_t = 0.0;
    cudaEventRecord(start);
    result = CutlassSgemmTT(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, isColumnMajor);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B_tmp);
      cudaFree(B);
      cudaFree(A);

      return result;
    }
    cudaEventElapsedTime(&tmp_t, start, stop);
    ms += tmp_t / repeatition;
  }
    
  printf("GEMM running time is %f us.\n", ms * 1000);

  result = cudaMemcpy(host_cutlass, C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B_tmp);
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
  cudaFree(B_tmp);
  cudaFree(B);
  cudaFree(A);

  if (count > 0) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}