#define FULL_MASK 0xffffffff
extern "C" __global__ void DepthConvFused_2_kernel0( float* __restrict__ Conv2dOutput_0,  float* __restrict__ Input,  float* __restrict__ DepthwiseFilter_1,  float* __restrict__ Conv2dFilter_1) {
  __shared__ float PaddedInput_0_shared[1024];
   float DepthwiseConv2dOutput_0[1];
  __shared__ float red_buf0[128];
   float red_buf;
  Conv2dOutput_0[(((((((((int)blockIdx.x) / 112) * 14336) + ((((int)blockIdx.x) % 4) * 32)) + (((((int)blockIdx.x) / 4) % 28) * 256)) + ((((int)threadIdx.y) / 2) * 7168)) + ((((int)threadIdx.y) % 2) * 128)) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int rc_outer_v = 0; rc_outer_v < 4; ++rc_outer_v) {
    __syncthreads();
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      PaddedInput_0_shared[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + (ax1 * 128))] = ((((((1 - ax1) <= ((((int)blockIdx.x) / 112) * 2)) && (((((int)blockIdx.x) / 112) * 2) < (57 - ax1))) && ((1 - ((int)threadIdx.y)) <= (((((int)blockIdx.x) / 4) % 28) * 2))) && ((((((int)blockIdx.x) / 4) % 28) * 2) < (57 - ((int)threadIdx.y)))) ? Input[((((((((((int)blockIdx.x) / 112) * 14336) + (((((int)blockIdx.x) / 4) % 28) * 256)) + (((int)threadIdx.y) * 128)) + ((int)threadIdx.x)) + (rc_outer_v * 32)) + (ax1 * 7168)) - 7296)] : 0.000000e+00f);
    }
    DepthwiseConv2dOutput_0[0] = 0.000000e+00f;
    __syncthreads();
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        DepthwiseConv2dOutput_0[0] = (DepthwiseConv2dOutput_0[0] + (PaddedInput_0_shared[((((((((int)threadIdx.y) / 2) * 128) + ((((int)threadIdx.y) % 2) * 32)) + ((int)threadIdx.x)) + (ry * 128)) + (rx * 32))] * DepthwiseFilter_1[(((((int)threadIdx.x) + (rc_outer_v * 32)) + (ry * 384)) + (rx * 128))]));
      }
    }
    __syncthreads();
    for (int ax2_outer = 0; ax2_outer < 8; ++ax2_outer) {
      PaddedInput_0_shared[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + (ax2_outer * 128))] = Conv2dFilter_1[((((((((int)blockIdx.x) % 4) * 32) + (((int)threadIdx.y) * 128)) + ((int)threadIdx.x)) + (rc_outer_v * 4096)) + (ax2_outer * 512))];
    }
    __syncthreads();
    // ((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = (DepthwiseConv2dOutput_0[0] * PaddedInput_0_shared[(((int)threadIdx.x) * 33)]);
    // __syncthreads();
    // if (((int)threadIdx.x) < 16) {
    //   ((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = (((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] + ((volatile __shared__ float*)red_buf0)[((16 + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))]);
    //   ((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = (((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] + ((volatile __shared__ float*)red_buf0)[((8 + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))]);
    //   ((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = (((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] + ((volatile __shared__ float*)red_buf0)[((4 + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))]);
    //   ((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = (((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] + ((volatile __shared__ float*)red_buf0)[((2 + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))]);
    //   ((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = (((volatile __shared__ float*)red_buf0)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] + ((volatile __shared__ float*)red_buf0)[((1 + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))]);
    // }
    // __syncthreads();

    for(int iter = 0; iter < 32; ++iter) {
      float tmp = PaddedInput_0_shared[((int)threadIdx.x) + iter * 32];
      red_buf = DepthwiseConv2dOutput_0[0] * tmp;
      for (int offset = 16; offset > 0; offset /= 2)
        red_buf += __shfl_down_sync(FULL_MASK, red_buf, offset);
      ((volatile __shared__ float*)red_buf0)[iter + ((int)threadIdx.y) * 32] = red_buf;
    }
    __syncthreads();

    Conv2dOutput_0[(((((((((int)blockIdx.x) / 112) * 14336) + ((((int)blockIdx.x) % 4) * 32)) + (((((int)blockIdx.x) / 4) % 28) * 256)) + ((((int)threadIdx.y) / 2) * 7168)) + ((((int)threadIdx.y) % 2) * 128)) + ((int)threadIdx.x))] = (Conv2dOutput_0[(((((((((int)blockIdx.x) / 112) * 14336) + ((((int)blockIdx.x) % 4) * 32)) + (((((int)blockIdx.x) / 4) % 28) * 256)) + ((((int)threadIdx.y) / 2) * 7168)) + ((((int)threadIdx.y) % 2) * 128)) + ((int)threadIdx.x))] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.y) * 32) + ((int)threadIdx.x)]);
  }
}

