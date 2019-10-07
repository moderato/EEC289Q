#define FULL_MASK 0xffffffff
extern "C" __global__ void DepthConvFused_2_kernel0( const float* Input, const float* DepthwiseFilter_1, const float* Conv2dFilter_1,  float* Conv2dOutput_0) {
  // int res;
  // clock_t start = clock();

   float Conv2dOutput_0_local[4];
   float DepthwiseConv2dOutput_0_local[1];
  
  __shared__ float intermediate[128];
  __shared__ float Conv2dFilter_1_shared[1024];

   //
   float filter[9];
   float buffer[8];
   int thx = threadIdx.x, thy = threadIdx.y, blx = blockIdx.x;
   //

  ((float4*)(Conv2dOutput_0_local))[0] = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);

  for (int rc_outer_v = 0; rc_outer_v < 4; ++rc_outer_v) {
    __syncthreads();

    ((float4*)(Conv2dFilter_1_shared + ((((thy) * 128) + (((thx) / 8) * 32)) + (((thx) % 8) * 4))))[0] = ((((((1 - (thy)) <= (((blx) / 28) * 2)) && ((((blx) / 28) * 2) < (57 - (thy)))) && ((1 - ((thx) / 8)) <= (((blx) % 28) * 2))) && ((((blx) % 28) * 2) < (57 - ((thx) / 8)))) ? ((float4*)(Input + (((((((((blx) / 28) * 14336) + (((blx) % 28) * 256)) + ((thy) * 7168)) + (((thx) / 8) * 128)) + (((thx) % 8) * 4)) + (rc_outer_v * 32)) - 7296)))[0] : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
    DepthwiseConv2dOutput_0_local[0] = 0.000000e+00f;

    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        filter[ry * 3 + rx] = DepthwiseFilter_1[((((thx) + (rc_outer_v * 32)) + (ry * 384)) + (rx * 128))];
      }
    }

    __syncthreads();
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        DepthwiseConv2dOutput_0_local[0] += (Conv2dFilter_1_shared[(((((((thy) / 2) * 128) + (((thy) % 2) * 32)) + (thx)) + (ry * 128)) + (rx * 32))] * filter[ry * 3 + rx]);
      }
    }
    intermediate[(((thy) * 32) + (thx))] = DepthwiseConv2dOutput_0_local[0];

    if (blockIdx.x == 29 && rc_outer_v == 0) {
      printf("%d, %f\n", threadIdx.y * 32 + threadIdx.x, intermediate[threadIdx.y * 32 + threadIdx.x]);
      // printf("%d, %f\n", 128 + thy * 32 + thx, intermediate[128 + thy * 32 + thx]);
      // printf("%d, %f\n", 256 + thy * 32 + thx, intermediate[256 + thy * 32 + thx]);
      // printf("%d, %f\n", 384 + thy * 32 + thx, intermediate[384 + thy * 32 + thx]);
    }

    // gmem to rmem
    ((float4*)(buffer))[0] = ((float4*)(Conv2dFilter_1 + (((((((thy) * 512)) + (((thx) / 8) * 128)) + (((thx) % 8) * 4)) + (rc_outer_v * 4096)) + (0))))[0];
    ((float4*)(buffer + 4))[0] = ((float4*)(Conv2dFilter_1 + (((((((thy) * 512)) + (((thx) / 8) * 128)) + (((thx) % 8) * 4)) + (rc_outer_v * 4096)) + (2048))))[0];

    for (int iter = 0; iter < 4; iter++) {
      __syncthreads();
      // rmem to smem
      ((float4*)(Conv2dFilter_1_shared + (((((thy) * 128) + (((thx) / 8) * 32)) + (((thx) % 8) * 4)) + (0))))[0] = ((float4*)(buffer))[0];
      ((float4*)(Conv2dFilter_1_shared + (((((thy) * 128) + (((thx) / 8) * 32)) + (((thx) % 8) * 4)) + (512))))[0] = ((float4*)(buffer+4))[0];

      __syncthreads();

      // compute on smem
      for (int i = 0; i < 32; i++) {
        Conv2dOutput_0_local[iter] += intermediate[((thy) / 2) * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[(thx % 16) + 32 * i + ((thy) % 2) * 16];
      }

      // gmem to rmem
      if (iter < 3) {
        ((float4*)(buffer))[0] = ((float4*)(Conv2dFilter_1 + (((((((thy) * 512) + ((iter+1) * 32)) + (((thx) / 8) * 128)) + (((thx) % 8) * 4)) + (rc_outer_v * 4096)) + (0))))[0];
        ((float4*)(buffer + 4))[0] = ((float4*)(Conv2dFilter_1 + (((((((thy) * 512) + ((iter+1) * 32)) + (((thx) / 8) * 128)) + (((thx) % 8) * 4)) + (rc_outer_v * 4096)) + (2048))))[0];
      }
    }


  }

#pragma unroll
  for (int iter = 0; iter < 4; iter++)
    Conv2dOutput_0[(((blx) / 28) * 14336) + (((blx) % 28) * 256) + (((thy) / 2) * 7168) + (( ((thx) / 16) % 2) * 128) + (iter * 32) + ((thx) % 16) + (((thy) % 2) * 16) ] = Conv2dOutput_0_local[iter];
}

