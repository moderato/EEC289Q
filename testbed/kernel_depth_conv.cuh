#define FULL_MASK 0xffffffff
extern "C" __global__ void DepthConvFused_2_kernel0( float* __restrict__ Input,  float* __restrict__ DepthwiseFilter_1,  float* __restrict__ Conv2dFilter_1,  float* __restrict__ Conv2dOutput_0, int* d_data) {
   float Conv2dOutput_0_local[1];
   float DepthwiseConv2dOutput_0_local[1];
  
  // 000
  // __shared__ float DepthwiseFilter_1_shared[288];

  // 001
  __shared__ float intermediate[128];
  __shared__ float Conv2dFilter_1_shared[1024];

  // // 002
  // // ****
  // __shared__ float intermediate[128];
  // __shared__ float Conv2dFilter_1_shared[512];
  //  float tmp[4];
  // // ****

  // // 003
  // __shared__ float Conv2dFilter_1_shared[1024];
  //  float tmp[4];
  //  float red_buf[4];
  // __shared__ float red_buf0[128];
   

   float Conv2dOutput_0_local_rf[1];
   float DepthwiseConv2dOutput_0_local1[4];
  Conv2dOutput_0_local[0] = 0.000000e+00f;
  for (int rc_outer_v = 0; rc_outer_v < 4; ++rc_outer_v) {
    __syncthreads();
    // ((__shared__ float4*)(Conv2dFilter_1_shared + (((((int)threadIdx.y) * 128) + ((((int)threadIdx.x) / 8) * 32)) + ((((int)threadIdx.x) % 8) * 4))))[0] =

    //  ((((((1 - ((int)threadIdx.y)) <= ((((int)blockIdx.x) / 112) * 2)) && (((((int)blockIdx.x) / 112) * 2) < (57 - ((int)threadIdx.y)))) && ((1 - (((int)threadIdx.x) / 8)) <= (((((int)blockIdx.x) / 4) % 28) * 2))) && ((((((int)blockIdx.x) / 4) % 28) * 2) < (57 - (((int)threadIdx.x) / 8)))) 

    //   ? (( float4*)(Input + ((((((((((int)blockIdx.x) / 112) * 14336) + (((((int)blockIdx.x) / 4) % 28) * 256)) + (((int)threadIdx.y) * 7168)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 32)) - 7296)))[0] 
      
    //   : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
    // DepthwiseConv2dOutput_0_local[0] = 0.000000e+00f;
    // __syncthreads();

    // // 000
    // if (threadIdx.y < 3) {
    //   for(int iter = 0; iter < 3; ++iter) {
    //     DepthwiseFilter_1_shared[(((((int)threadIdx.x)) + (iter * 96)) + (((int)threadIdx.y) * 32))] = DepthwiseFilter_1[(((((int)threadIdx.x) + (rc_outer_v * 32)) + (iter * 384)) + (((int)threadIdx.y) * 128))];
    //   }
    // }
    // __syncthreads();
    // for (int ry = 0; ry < 3; ++ry) {
    //   for (int rx = 0; rx < 3; ++rx) {

    //     // 000
    //     // DepthwiseConv2dOutput_0_local[0] = (DepthwiseConv2dOutput_0_local[0] + (Conv2dFilter_1_shared[((((((((int)threadIdx.y) / 2) * 128) + ((((int)threadIdx.y) % 2) * 32)) + ((int)threadIdx.x)) + (ry * 128)) + (rx * 32))] * DepthwiseFilter_1_shared[(((((int)threadIdx.x)) + (ry * 96)) + (rx * 32))]));
    //     DepthwiseConv2dOutput_0_local[0] = (DepthwiseConv2dOutput_0_local[0] + (Conv2dFilter_1_shared[((((((((int)threadIdx.y) / 2) * 128) + ((((int)threadIdx.y) % 2) * 32)) + ((int)threadIdx.x)) + (ry * 128)) + (rx * 32))] * DepthwiseFilter_1[(((((int)threadIdx.x) + (rc_outer_v * 32)) + (ry * 384)) + (rx * 128))]));

    //   }
    // }

    // 001
    Conv2dOutput_0_local_rf[0] = 0.000000e+00f;
    // __syncthreads();
    intermediate[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = DepthwiseConv2dOutput_0_local[0];
    for (int ax2_outer_outer = 0; ax2_outer_outer < 2; ++ax2_outer_outer) {
      ((__shared__ float4*)(Conv2dFilter_1_shared + ((((((int)threadIdx.y) * 128) + ((((int)threadIdx.x) / 8) * 32)) + ((((int)threadIdx.x) % 8) * 4)) + (ax2_outer_outer * 512))))[0] = (( float4*)(Conv2dFilter_1 + (((((((((int)blockIdx.x) % 4) * 32) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 4096)) + (ax2_outer_outer * 2048))))[0];
    }
    __syncthreads();

    // for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
    //   for (int ax3 = 0; ax3 < 4; ++ax3) {
    //     DepthwiseConv2dOutput_0_local1[ax3] = intermediate[(((((int)threadIdx.y) * 32) + (rc_inner_outer * 4)) + ax3)];
    //   }
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[0] * Conv2dFilter_1_shared[(((int)threadIdx.x) + (rc_inner_outer * 128))]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[1] * Conv2dFilter_1_shared[((32 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[2] * Conv2dFilter_1_shared[((64 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[3] * Conv2dFilter_1_shared[((96 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
    // }
    // Conv2dOutput_0_local[0] = (Conv2dOutput_0_local[0] + Conv2dOutput_0_local_rf[0]);


    // // 002
    // Conv2dOutput_0_local_rf[0] = 0.000000e+00f;
    // __syncthreads();
    // intermediate[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = DepthwiseConv2dOutput_0_local[0];
    // for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
    //   for (int ax3 = 0; ax3 < 4; ++ax3) {
    //     DepthwiseConv2dOutput_0_local1[ax3] = intermediate[(((((int)threadIdx.y) * 32) + (rc_inner_outer * 4)) + ax3)];
    //     tmp[ax3] = Conv2dFilter_1[(((int)blockIdx.x) % 4) * 32 + ax3 * 128 + (rc_inner_outer * 512) + ((int)threadIdx.x) + (rc_outer_v * 4096)];
    //   }
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[0] * tmp[0]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[1] * tmp[1]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[2] * tmp[2]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[3] * tmp[3]));
    // }
    // Conv2dOutput_0_local[0] = (Conv2dOutput_0_local[0] + Conv2dOutput_0_local_rf[0]);


    // // 002
    // __syncthreads();
    // intermediate[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = DepthwiseConv2dOutput_0_local[0];
    // Conv2dOutput_0_local_rf[0] = 0.000000e+00f;
    // for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
    //   Conv2dFilter_1_shared[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + ((rc_inner_outer % 4) * 128))] = Conv2dFilter_1[((((((((int)blockIdx.x) % 4) * 32) + (((int)threadIdx.y) * 128)) + ((int)threadIdx.x)) + (rc_outer_v * 4096)) + (rc_inner_outer * 512))];
    //   __syncthreads();
    //   for (int ax3 = 0; ax3 < 4; ++ax3) {
    //     DepthwiseConv2dOutput_0_local1[ax3] = intermediate[(((((int)threadIdx.y) * 32) + (rc_inner_outer * 4)) + ax3)];
    //   }
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[0] * Conv2dFilter_1_shared[(((int)threadIdx.x) + ((rc_inner_outer % 4) * 128))]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[1] * Conv2dFilter_1_shared[((32 + ((int)threadIdx.x)) + ((rc_inner_outer % 4) * 128))]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[2] * Conv2dFilter_1_shared[((64 + ((int)threadIdx.x)) + ((rc_inner_outer % 4) * 128))]));
    //   Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[3] * Conv2dFilter_1_shared[((96 + ((int)threadIdx.x)) + ((rc_inner_outer % 4) * 128))]));
    // }
    // Conv2dOutput_0_local[0] = (Conv2dOutput_0_local[0] + Conv2dOutput_0_local_rf[0]);


    // // 003
    // red_buf0[((int)threadIdx.y) * 32 + ((int)threadIdx.x)] = 0.000000e+00f;
    // for (int ax2_outer_outer = 0; ax2_outer_outer < 2; ++ax2_outer_outer) {
    //   ((__shared__ float4*)(Conv2dFilter_1_shared + ((((((int)threadIdx.y) * 128) + ((((int)threadIdx.x) / 8) * 32)) + ((((int)threadIdx.x) % 8) * 4)) + (ax2_outer_outer * 512))))[0] = ((float4*)(Conv2dFilter_1 + (((((((((int)blockIdx.x) % 4) * 32) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 4096)) + (ax2_outer_outer * 2048))))[0];
    // }
    // __syncthreads();
    // for(int iter = 0; iter < 8; ++iter) {
    //   ((float4*)(tmp))[0] = ((float4*)(Conv2dFilter_1_shared + ((int)threadIdx.x) * 32 + iter * 4))[0];
    //   ((float4*)(red_buf))[0] = make_float4(tmp[0] * DepthwiseConv2dOutput_0_local[0], tmp[1] * DepthwiseConv2dOutput_0_local[0], tmp[2] * DepthwiseConv2dOutput_0_local[0], tmp[3] * DepthwiseConv2dOutput_0_local[0]);
    //   __syncthreads();
    //   for (int offset = 16; offset > 0; offset /= 2) {
    //     red_buf[0] += __shfl_down_sync(FULL_MASK, red_buf[0], offset);
    //     red_buf[1] += __shfl_down_sync(FULL_MASK, red_buf[1], offset);
    //     red_buf[2] += __shfl_down_sync(FULL_MASK, red_buf[2], offset);
    //     red_buf[3] += __shfl_down_sync(FULL_MASK, red_buf[3], offset);
    //   }
    //   __syncthreads();
    //   if (((int)threadIdx.x) == 0)
    //     ((__shared__ float4*)(red_buf0 + iter * 4 + ((int)threadIdx.y) * 32))[0] = ((float4*)(red_buf))[0];
    //   __syncthreads();
    // }

    // Conv2dOutput_0_local[0] = (Conv2dOutput_0_local[0] + red_buf0[((int)threadIdx.y) * 32 + ((int)threadIdx.x)]);
  }
  Conv2dOutput_0[(((((((((int)blockIdx.x) / 112) * 14336) + ((((int)blockIdx.x) % 4) * 32)) + (((((int)blockIdx.x) / 4) % 28) * 256)) + ((((int)threadIdx.y) / 2) * 7168)) + ((((int)threadIdx.y) % 2) * 128)) + ((int)threadIdx.x))] = Conv2dOutput_0_local[0];

  // if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
  //   *d_data += 567;
}

