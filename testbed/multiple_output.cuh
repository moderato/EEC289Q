#define FULL_MASK 0xffffffff
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

    // clock_t start;
    // int res;
    for (int iter = 0; iter < 4; iter++) {
      __syncthreads();
      Conv2dOutput_0_local_rf[0] = 0.000000e+00f;
#pragma unroll
      for (int ax2_outer_outer = 0; ax2_outer_outer < 2; ++ax2_outer_outer) {
        ((__shared__ float4*)(Conv2dFilter_1_shared + ((((((int)threadIdx.y) * 128) + ((((int)threadIdx.x) / 8) * 32)) + ((((int)threadIdx.x) % 8) * 4)) + (ax2_outer_outer * 512))))[0] = (( float4*)(Conv2dFilter_1 + ((((((((int)threadIdx.y) * 512) + (iter * 32)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 4096)) + (ax2_outer_outer * 2048))))[0];
      }
      __syncthreads();
      for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
        (( float4*)DepthwiseConv2dOutput_0_local1)[0] = (( float4*)(intermediate + (((((int)threadIdx.y) * 32) + (rc_inner_outer * 4)))))[0];

        Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[0] * Conv2dFilter_1_shared[(((int)threadIdx.x) + (rc_inner_outer * 128))]));
        Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[1] * Conv2dFilter_1_shared[((32 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
        Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[2] * Conv2dFilter_1_shared[((64 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
        Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[3] * Conv2dFilter_1_shared[((96 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
      }
      Conv2dOutput_0_local[iter] = (Conv2dOutput_0_local[iter] + Conv2dOutput_0_local_rf[0]);
    }
  }

#pragma unroll
  for (int iter = 0; iter < 4; iter++)
    Conv2dOutput_0[(((((((((int)blockIdx.x) / 28) * 14336) + ((((int)blockIdx.x) % 28) * 256)) + ((((int)threadIdx.y) / 2) * 7168)) + ((((int)threadIdx.y) % 2) * 128)) + (iter * 32)) + ((int)threadIdx.x))] = Conv2dOutput_0_local[iter];

  // res = (int)(clock() - start);
  // printf("Takes %d cycles\n", res);

  // if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
  //   *d_data += 567;
}

