#define FULL_MASK 0xffffffff
extern "C" __global__ void DepthConvFused_2_kernel0( const float* Input, const float* DepthwiseFilter_1, const float* Conv2dFilter_1,  float* Conv2dOutput_0, int* d_data) {
   float Conv2dOutput_0_local[1];
   float DepthwiseConv2dOutput_0_local[1];

  // __shared__ float red_buf0[128];
  __shared__ float Conv2dFilter_1_shared[1152];
   float red_buf[32];

  Conv2dOutput_0_local[0] = 0.000000e+00f;
  for (int rc_outer_v = 0; rc_outer_v < 4; ++rc_outer_v) {
    __syncthreads();
    ((__shared__ float4*)(Conv2dFilter_1_shared + (((((int)threadIdx.y) * 128) + ((((int)threadIdx.x) / 8) * 32)) + ((((int)threadIdx.x) % 8) * 4))))[0] =

     ((((((1 - ((int)threadIdx.y)) <= ((((int)blockIdx.x) / 112) * 2)) && (((((int)blockIdx.x) / 112) * 2) < (57 - ((int)threadIdx.y)))) && ((1 - (((int)threadIdx.x) / 8)) <= (((((int)blockIdx.x) / 4) % 28) * 2))) && ((((((int)blockIdx.x) / 4) % 28) * 2) < (57 - (((int)threadIdx.x) / 8)))) 

      ? (( float4*)(Input + ((((((((((int)blockIdx.x) / 112) * 14336) + (((((int)blockIdx.x) / 4) % 28) * 256)) + (((int)threadIdx.y) * 7168)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 32)) - 7296)))[0] 
      
      : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
    DepthwiseConv2dOutput_0_local[0] = 0.000000e+00f;
    __syncthreads();
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        DepthwiseConv2dOutput_0_local[0] = (DepthwiseConv2dOutput_0_local[0] + (Conv2dFilter_1_shared[((((((((int)threadIdx.y) / 2) * 128) + ((((int)threadIdx.y) % 2) * 32)) + ((int)threadIdx.x)) + (ry * 128)) + (rx * 32))] * DepthwiseFilter_1[(((((int)threadIdx.x) + (rc_outer_v * 32)) + (ry * 384)) + (rx * 128))]));
      }
    }

    __syncthreads();
    // red_buf0[((int)threadIdx.y) * 32 + ((int)threadIdx.x)] = 0.000000e+00f;
#pragma unroll
    for (int ax2_outer_outer = 0; ax2_outer_outer < 2; ++ax2_outer_outer) {
      ((__shared__ float4*)(Conv2dFilter_1_shared + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) / 8) * 36)) + ((((int)threadIdx.x) % 8) * 4)) + (ax2_outer_outer * 576))))[0] = ((float4*)(Conv2dFilter_1 + (((((((((int)blockIdx.x) % 4) * 32) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 4096)) + (ax2_outer_outer * 2048))))[0];
    }
    __syncthreads();

    //////////// 1783
    // for(char iter = 0; iter < 8; ++iter) {
    //   ((float4*)(red_buf))[0] = ((float4*)(Conv2dFilter_1_shared + ((int)threadIdx.x) * 36 + iter * 4))[0];
    //   red_buf[0] *= DepthwiseConv2dOutput_0_local[0];
    //   red_buf[1] *= DepthwiseConv2dOutput_0_local[0];
    //   red_buf[2] *= DepthwiseConv2dOutput_0_local[0];
    //   red_buf[3] *= DepthwiseConv2dOutput_0_local[0];

    //   for (unsigned int offset = 16; offset > 0; offset /= 2) {
    //     red_buf[0] += __shfl_xor_sync(FULL_MASK, red_buf[0], offset);
    //     red_buf[1] += __shfl_xor_sync(FULL_MASK, red_buf[1], offset);
    //     red_buf[2] += __shfl_xor_sync(FULL_MASK, red_buf[2], offset);
    //     red_buf[3] += __shfl_xor_sync(FULL_MASK, red_buf[3], offset);
    //   }

    //   if (iter == ((int)threadIdx.x) / 4) // Correct but slow
    //     Conv2dOutput_0_local[0] += red_buf[((int)threadIdx.x) % 4];

    //   // testing
    //   // Conv2dOutput_0_local[0] += red_buf[((int)threadIdx.x) % 4];
    // }

    ////////// 1150
    clock_t start;
    int res;
    for(char iter = 0; iter < 32; ++iter) {
      start = clock();
      red_buf[0] = Conv2dFilter_1_shared[((int)threadIdx.x) * 36 + iter] * DepthwiseConv2dOutput_0_local[0];
      res = (int)(clock() - start);
      if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0 && rc_outer_v == 0 && iter < 4)
        printf("Iter No.%d, load time: %d\n", iter, res);

      start = clock();
#pragma unroll
      for (unsigned int offset = 16; offset > 0; offset /= 2) {
        red_buf[0] += __shfl_xor_sync(FULL_MASK, red_buf[0], offset);
      }
      res = (int)(clock() - start);
      if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0 && rc_outer_v == 0 && iter < 4)
        printf("Iter No.%d, reduction time: %d\n", iter, res);

      start = clock();
      if (iter == ((char)threadIdx.x & 0x1f))
        Conv2dOutput_0_local[0] = Conv2dOutput_0_local[0] + red_buf[0];
      res = (int)(clock() - start);
      if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0 && rc_outer_v == 0 && iter < 4)
        printf("Iter No.%d, write time: %d\n", iter, res);
    }

    // /////////// 3220
    // #pragma unroll
    // for (int i = 0; i < 32; i++) {
    //   red_buf[i] = Conv2dFilter_1_shared[((int)threadIdx.x) * 36 + i] * DepthwiseConv2dOutput_0_local[0];
    // }
    // for (unsigned int offset = 16; offset > 0; offset /= 2) {
    //   #pragma unroll
    //   for (int i = 0; i < 32; i++) {
    //     red_buf[i] += __shfl_xor_sync(FULL_MASK, red_buf[i], offset);
    //   }
    // }
    // Conv2dOutput_0_local[0] += red_buf[(int)threadIdx.x]; // Correct but slow
  }

  Conv2dOutput_0[(((((((((int)blockIdx.x) / 112) * 14336) + ((((int)blockIdx.x) % 4) * 32)) + (((((int)blockIdx.x) / 4) % 28) * 256)) + ((((int)threadIdx.y) / 2) * 7168)) + ((((int)threadIdx.y) % 2) * 128)) + ((int)threadIdx.x))] = Conv2dOutput_0_local[0];

  // if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
  //   *d_data += 567;
}

