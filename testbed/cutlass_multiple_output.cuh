#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/sgemm_traits.h"

// CUTLASS includes
#include "cutlass/tile_iterator.h"
#include "cutlass/tile_traits_standard.h"
#include "cutlass/gemm/thread_multiply_add.h"
#include "cutlass/gemm/gemm_shared_tile.h"

//
// CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "tools/util/tensor_view_io.h"

// Defines cutlass::HostMatrix<>
#include "tools/util/host_matrix.h"

// Defines cutlass::reference::device::TensorInitialize()
#include "tools/util/reference/device/tensor_elementwise.h"

// Defines cutlass::reference::host::TensorEquals()
#include "tools/util/reference/host/tensor_elementwise.h"

#pragma warning( disable : 4503)

extern "C" __global__ void DepthConvFused_2_kernel0( const float* Input, const float* DepthwiseFilter_1, const float* Conv2dFilter_1,  float* Conv2dOutput_0, int* d_data) {
  // int res;
  // clock_t start = clock();

   float Conv2dOutput_0_local[4];
   float DepthwiseConv2dOutput_0_local[1];
  
  __shared__ float intermediate[128];
  __shared__ float Conv2dFilter_1_shared[1024];
   float DepthwiseConv2dOutput_0_local1[4];
   float Conv2dOutput_0_local_rf[1];

  (( float4*)(Conv2dOutput_0_local))[0] = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);

  for (int rc_outer_v = 0; rc_outer_v < 4; ++rc_outer_v) {
    __syncthreads();

  ((__shared__ float4*)(Conv2dFilter_1_shared + (((((int)threadIdx.y) * 128) + ((((int)threadIdx.x) / 8) * 32)) + ((((int)threadIdx.x) % 8) * 4))))[0] = ((((((1 - ((int)threadIdx.y)) <= ((((int)blockIdx.x) / 28) * 2)) && (((((int)blockIdx.x) / 28) * 2) < (57 - ((int)threadIdx.y)))) && ((1 - (((int)threadIdx.x) / 8)) <= ((((int)blockIdx.x) % 28) * 2))) && (((((int)blockIdx.x) % 28) * 2) < (57 - (((int)threadIdx.x) / 8)))) ? 
    (( float4*)(Input + ((((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 256)) + (((int)threadIdx.y) * 7168)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 32)) - 7296)))[0] : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
    DepthwiseConv2dOutput_0_local[0] = 0.000000e+00f;


    if (threadIdx.y < 3) {
      for(int iter = 0; iter < 3; ++iter) {
        Conv2dFilter_1_shared[512 + (((((int)threadIdx.x)) + (iter * 96)) + (((int)threadIdx.y) * 32))] = DepthwiseFilter_1[(((((int)threadIdx.x) + (rc_outer_v * 32)) + (iter * 384)) + (((int)threadIdx.y) * 128))];
      }
    }

    __syncthreads();
    
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        DepthwiseConv2dOutput_0_local[0] = (DepthwiseConv2dOutput_0_local[0] + (Conv2dFilter_1_shared[((((((((int)threadIdx.y) / 2) * 128) + ((((int)threadIdx.y) % 2) * 32)) + ((int)threadIdx.x)) + (ry * 128)) + (rx * 32))] * DepthwiseFilter_1[(((((int)threadIdx.x) + (rc_outer_v * 32)) + (ry * 384)) + (rx * 128))]));
      }
    }
    intermediate[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = DepthwiseConv2dOutput_0_local[0];
    __syncthreads();

    // A: intermediate
    // B: Conv2dFilter_1_shared

    //////////////////////////////////////////////////////////////////////////////////////////////////
    ////  Load Conv2dFilter_1 (global) to B (shared) (To be refactored to GemmGlobalIteratorAb)
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // Load a (K = 32, M = 32) tile with 128 threads
    typedef cutlass::Shape<1, 32, 32> Tile_32x32;
    typedef cutlass::TileTraitsStandard<Tile_32x32, 128> TileTraits;

    typedef cutlass::TileLoadIterator<
        TileTraits,                     // the Traits type, defines shape/distribution of accesses
        float,                          // elements are of type float
        // kW = row major, kH = column major
        cutlass::IteratorAdvance::kW,   // post-increment accesses advance in strided (as opposed to
                                        //     contiguous dimension 
        cutlass::MemorySpace::kGlobal   // iterator loads from global memory 
        > TileLoadIterator;

    // Defines a tile store iterator
    typedef cutlass::TileStoreIterator<
        TileTraits,                     // the Traits type, defines shape/distribution of accesses
        float,                          // elements are of type float
        // kW = row major, kH = column major
        cutlass::IteratorAdvance::kW,   // post-increment accesses advance in strided (as opposed to
                                        //     contiguous) dimension
        cutlass::MemorySpace::kShared   // iterator stores into shared memory
        > TileStoreIterator;

    // Defines a predicate vector for managing statically sized vector of boolean predicates
    typedef typename TileLoadIterator::PredicateVector PredicateVector;

    // The parameters specified to the iterators. These include the pointer to the source of
    // addressable memory, and the strides and increments for each of the tile's dimensions  
    typename TileLoadIterator::Params load_params;
    typename TileStoreIterator::Params store_params;

    // Initializing the parameters for both of the iterators. The TileLoadIterator accesses the
    // input matrix and TileStoreIterator accesses the output matrix. The strides are set
    // identically since the data is being stored in the same way as it is loaded (column-major
    // mapping).
    //////
    // load_params.initialize(input, M*K, M, 1);
    // store_params.initialize(output, M*K, M, 1);
    //////
    load_params.initialize(Conv2dFilter_1 + 4096 * rc_outer_v, 1024, 32, 1);
    store_params.initialize(Conv2dFilter_1_shared, 1024, 32, 1);
   
    // Constructing the tile load and store iterators, and the predicates vector
    TileLoadIterator load_iterator(load_params);
    TileStoreIterator store_iterator(store_params);
    PredicateVector predicates;

    // Initializing the predicates with bounds set to <1, K, M>. This protects out-of-bounds loads.
    load_iterator.initialize_predicates(predicates.begin(), cutlass::make_Coord(1, 32, 32));

    // The fragment in which the elements are loaded into and stored from.
    typename TileLoadIterator::Fragment fragment_conv2d_filter;

    // Loading a tile into a fragment and advancing to the next tile's position
    load_iterator.load_post_increment(fragment_conv2d_filter, predicates.begin());
    // Storing a tile from fragment and advancing to the next tile's position
    store_iterator.store_post_increment(fragment_conv2d_filter);
    //////////////////////////////////////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////////////////////////////////////////////
    ////  Do GEMM with A and B in thread level
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // A
    typedef cutlass::Shape<1, 4, 8> WarpTile;
    typedef cutlass::Shape<1, 4> Warps;
    typedef cutlass::Shape<1, 4, 8> ThreadsPerWarp;
    typedef cutlass::Shape<1, 1, 1> InstShape;
    typedef GemmSharedLoadTileATraits<
      float,
      WarpTile,       // output warp size: a warp calculates 1x4x8
      Warps,          // warps: 1x4 warps in total
      ThreadsPerWarp, // 
      InstShape       // doubted!!
    > IntermediateDataIterator; // Load intermediate data from shared mem

    // B
    typedef cutlass::Shape<1, 4, 8> WarpTile;
    typedef cutlass::Shape<1, 4> Warps;
    typedef cutlass::Shape<1, 4, 8> ThreadsPerWarp;
    typedef cutlass::Shape<1, 1, 1> InstShape;
    typedef GemmSharedLoadTileBTraits<
      Conv2dFilter_1_shared, // ptr
      WarpTile,              // output warp size: a warp calculates 1x4x8
      Warps,                 // warps: 1x4 warps in total
      ThreadsPerWarp,        // 
      InstShape              // doubted!!
    > Conv2dFilterDataIterator; // Load intermediate data from shared mem
    

    typedef cutlass::Shape<4, 8, 8> ThreadTile;

    typedef ThreadMultiplyAdd<
      ThreadTile, 
      32,
      float, 
      float, 
      float
    > GMA;

  }

#pragma unroll
  for (int iter = 0; iter < 4; iter++)
    Conv2dOutput_0[(((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 256)) + ((((int)threadIdx.y) / 2) * 7168)) + ((((int)threadIdx.y) % 2) * 128)) + (iter * 32)) + ((int)threadIdx.x))] = Conv2dOutput_0_local[iter];

  // res = (int)(clock() - start);
  // printf("Takes %d cycles\n", res);

  // if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
  //   *d_data += 567;
}