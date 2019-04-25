extern "C" __global__ void DepthConvFused_2_kernel0( float* __restrict__ Input,  float* __restrict__ DepthwiseFilter_1,  float* __restrict__ Conv2dFilter_1,  float* __restrict__ Conv2dOutput_0, int* d_data) {
   float Conv2dOutput_0_local[1];
  __shared__ float PaddedInput_0_shared[512];
   float DepthwiseConv2dOutput_0_local[1];
  __shared__ float Conv2dFilter_1_shared[1024];
   float Conv2dOutput_0_local_rf[1];
   float DepthwiseConv2dOutput_0_local1[4];
  Conv2dOutput_0_local[0] = 0.000000e+00f;
  for (int rc_outer_v = 0; rc_outer_v < 4; ++rc_outer_v) {
    __syncthreads();
    ((__shared__ float4*)(PaddedInput_0_shared + (((((int)threadIdx.y) * 128) + ((((int)threadIdx.x) / 8) * 32)) + ((((int)threadIdx.x) % 8) * 4))))[0] = ((((((1 - ((int)threadIdx.y)) <= ((((int)blockIdx.x) / 112) * 2)) && (((((int)blockIdx.x) / 112) * 2) < (57 - ((int)threadIdx.y)))) && ((1 - (((int)threadIdx.x) / 8)) <= (((((int)blockIdx.x) / 4) % 28) * 2))) && ((((((int)blockIdx.x) / 4) % 28) * 2) < (57 - (((int)threadIdx.x) / 8)))) ? (( float4*)(Input + ((((((((((int)blockIdx.x) / 112) * 14336) + (((((int)blockIdx.x) / 4) % 28) * 256)) + (((int)threadIdx.y) * 7168)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 32)) - 7296)))[0] : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
    DepthwiseConv2dOutput_0_local[0] = 0.000000e+00f;
    __syncthreads();
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        DepthwiseConv2dOutput_0_local[0] = (DepthwiseConv2dOutput_0_local[0] + (PaddedInput_0_shared[((((((((int)threadIdx.y) / 2) * 128) + ((((int)threadIdx.y) % 2) * 32)) + ((int)threadIdx.x)) + (ry * 128)) + (rx * 32))] * DepthwiseFilter_1[(((((int)threadIdx.x) + (rc_outer_v * 32)) + (ry * 384)) + (rx * 128))]));
      }
    }
    __syncthreads();
    PaddedInput_0_shared[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))] = DepthwiseConv2dOutput_0_local[0];
    for (int ax2_outer_outer = 0; ax2_outer_outer < 2; ++ax2_outer_outer) {
      ((__shared__ float4*)(Conv2dFilter_1_shared + ((((((int)threadIdx.y) * 128) + ((((int)threadIdx.x) / 8) * 32)) + ((((int)threadIdx.x) % 8) * 4)) + (ax2_outer_outer * 512))))[0] = (( float4*)(Conv2dFilter_1 + (((((((((int)blockIdx.x) % 4) * 32) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) / 8) * 128)) + ((((int)threadIdx.x) % 8) * 4)) + (rc_outer_v * 4096)) + (ax2_outer_outer * 2048))))[0];
    }
    Conv2dOutput_0_local_rf[0] = 0.000000e+00f;
    __syncthreads();
    for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
      for (int ax3 = 0; ax3 < 4; ++ax3) {
        DepthwiseConv2dOutput_0_local1[ax3] = PaddedInput_0_shared[(((((int)threadIdx.y) * 32) + (rc_inner_outer * 4)) + ax3)];
      }
      Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[0] * Conv2dFilter_1_shared[(((int)threadIdx.x) + (rc_inner_outer * 128))]));
      Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[1] * Conv2dFilter_1_shared[((32 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
      Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[2] * Conv2dFilter_1_shared[((64 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
      Conv2dOutput_0_local_rf[0] = (Conv2dOutput_0_local_rf[0] + (DepthwiseConv2dOutput_0_local1[3] * Conv2dFilter_1_shared[((96 + ((int)threadIdx.x)) + (rc_inner_outer * 128))]));
    }
    Conv2dOutput_0_local[0] = (Conv2dOutput_0_local[0] + Conv2dOutput_0_local_rf[0]);
  }
  Conv2dOutput_0[(((((((((int)blockIdx.x) / 112) * 14336) + ((((int)blockIdx.x) % 4) * 32)) + (((((int)blockIdx.x) / 4) % 28) * 256)) + ((((int)threadIdx.y) / 2) * 7168)) + ((((int)threadIdx.y) % 2) * 128)) + ((int)threadIdx.x))] = Conv2dOutput_0_local[0];
}

