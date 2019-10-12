#include "helper.cuh"

// OC_stride = IC_stride for now
template <int H, int W, int IC, int OC, 
          int IC_stride, int OC_stride>
__global__ void DepthConvFused_2_kernel0(const float* Input, 
                                          const float* DepthwiseFilter_1, 
                                          const float* Conv2dFilter_1, 
                                          float* Conv2dOutput_0) {

  static_assert((OC_stride <= OC && OC_stride > 0), "!");
  static_assert((IC_stride <= IC && IC_stride > 0), "!");

  // Params
  int thx = threadIdx.x, thy = threadIdx.y;
  int blx = blockIdx.x, bly = blockIdx.y;
  int _g_h_blk = bly * OUTPUT_TILE_H;
  int _g_w_blk = blx * OUTPUT_TILE_W;
  int _g_h, _g_w, _s_h, _s_w;
  int num_thx_per_seg = OC_stride / 4;
  int _g_coord, _s_coord;

  // Shared memory
  extern __shared__ float s[];
  float *intermediate = s;
  float *Conv2dFilter_1_shared = &s[OUTPUT_TILE_H * OUTPUT_TILE_W * OC_stride];

  // Registers
  float Conv2dOutput_0_local[OUTPUT_TILE_H * OUTPUT_TILE_W] = { 0.0f };
  float DepthwiseConv2dOutput_0_local[1] = { 0.0f };
  float filter[FILTER_H * FILTER_W];
  float buffer[8];

  for (int _g_oc_step = 0; _g_oc_step < IC / IC_stride; _g_oc_step++) {
    ///////////// Preprocessing /////////////
    // Load filter to RMem
    loadDepthwiseFilterGlobalToRegister(DepthwiseFilter_1, filter, IC, _g_oc_step * IC_stride);

    // Load tile ((2,0), (3,3)) to SMem, ((0,0), (1,3)) to RMem
    /*********************
    |+++|+++|+++|+++|+++|+++|
    |   |   |   |   |   |   | 5
    |+++|+++|+++|+++|+++|+++|
    |   |   |   |   |   |   | 4
    |+++|+++|+++|+++|+++|+++|
    |   |   | s | s | r | r | 3
    |+++|+++|+++|+++|+++|+++|
    |   |   | s | s | r | r | 2
    |+++|+++|+++|+++|+++|+++|
    |   |   | s | s | r | r | 1
    |+++|+++|+++|+++|+++|+++|
    |   |   | s | s | r | r | 0
    |+++|+++|+++|+++|+++|+++|
      5   4   3   2   1   0
    *********************/
    prefetchInputData(Input,                 /* Input src    */
                      Conv2dFilter_1_shared, /* Shared dst   */
                      buffer,                /* Register dst */
                      _g_coord, _s_coord,
                      W, IC, _g_oc_step * IC_stride);

    int _s_h_coord = 0;
    int _s_w_coord = BUFFER_STRIDE;

    //////////////////////////// Loop ////////////////////////////
    // Load from RMem to SMem
      /*********************
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  
      |   |   |   |   |   |   |  |   |   |   |   |   |   |  |   |   |   |   |   |   |  | r | r | r | r |   |   |  
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  
      |   |   |   |   |   |   |  |   |   |   |   |   |   |  |   |   |   |   |   |   |  | r | r | r | r |   |   |  
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  
      |   |   | r | r | s | s |  | r | r | s | s | s | s |  | s | s | s | s |   |   |  | s | s | s | s |   |   |  
      |+++|+++|+++|+++|+++|+++|->|+++|+++|+++|+++|+++|+++|->|+++|+++|+++|+++|+++|+++|->|+++|+++|+++|+++|+++|+++|->
      |   |   | r | r | s | s |  | r | r | s | s | s | s |  | s | s | s | s |   |   |  | s | s | s | s |   |   |  
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  
      |   |   | r | r | s | s |  | r | r | s | s | s | s |  | s | s | s | s |   |   |  | s | s | s | s |   |   |  
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  
      |   |   | r | r | s | s |  | r | r | s | s | s | s |  | s | s | s | s |   |   |  | s | s | s | s |   |   |  
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  

      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   |  | s | s | s | s | r | r |  |   |   | s | s | s | s |
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   |  | s | s | s | s | r | r |  |   |   | s | s | s | s |
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   |  | s | s | s | s | r | r |  |   |   | s | s | s | s |
      |+++|+++|+++|+++|+++|+++|->|+++|+++|+++|+++|+++|+++|->|+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   |  | s | s | s | s | r | r |  |   |   | s | s | s | s |
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   |  |   |   |   |   |   |   |  |   |   |   |   |   |   |
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   |  |   |   |   |   |   |   |  |   |   |   |   |   |   |
      |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|  |+++|+++|+++|+++|+++|+++|
    *********************/

    int _s_orig_h, _s_orig_w, shared_idx;
    bool isTall;
    for (int loop = 0; loop < STEP_H * STEP_W; loop++) {
      DepthwiseConv2dOutput_0_local[0] = 0.0e+00f;

      // Load from register to shared
      ((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = ((float2*)(buffer))[0];
      __syncthreads();

      // Calculate the coordinates for space filling looping
      spaceFillingCalculation(loop, isTall,
                              _s_orig_h, _s_orig_w,
                              _s_h_coord, _s_w_coord);

      // Depthwise and store the result and store result to RMem
      shared_idx = thx + ((thy % STEP_OUTPUT_TILE_W) + thy / STEP_OUTPUT_TILE_H * 4 + _s_orig_w) * IC_stride + _s_orig_h * IC;
      depthwiseConvSingleNum(Conv2dFilter_1_shared, filter, DepthwiseConv2dOutput_0_local, 
                            _s_orig_h, _s_orig_w);
      intermediate[shared_idx] = DepthwiseConv2dOutput_0_local[0];

      if (loop != 3) {
        // Get HWs
        getGlobalSharedHW(isTall, _g_h_blk, _g_w_blk, _s_h_coord, _s_w_coord, _g_h, _g_w, _s_h, _s_w);

        // Load from GMem to RMem
        _g_coord = getGlobalCoordFloat2(_g_h, _g_w, W, IC, _g_oc_step * IC_stride);
        _s_coord = getSharedCoordFloat2(_s_h, _s_w);
        loadGlobalWithBoundCheck(Input, buffer, _g_coord - 7296, _g_h, _g_w);
      }
      __syncthreads();
    }

    // gmem to rmem
    int _g_input_origin = _g_oc_step * OC * OC_stride;
    int _g_input_offset = (thx % num_thx_per_seg) * 4 + (thx + blockDim.x * thy) / num_thx_per_seg * OC;

    load1x1FilterGlobalToRegister(Conv2dFilter_1, 
                                  buffer,
                                  _g_input_origin, 
                                  _g_input_offset);
    load1x1FilterGlobalToRegister(Conv2dFilter_1, 
                                  buffer + 4, 
                                  _g_input_origin + blockDim.x / num_thx_per_seg * blockDim.y * OC, // Row offset
                                  _g_input_offset);
    // if (bly == 0 && blx == 0 && thy == 0 && thx == 0) {
    //   printf("buffer 0: %f\n", buffer[0]);
    // }
    // __syncthreads();

    // 4 = OC / OC_stride, will result in slow speed
    for (int iter = 0; iter < OC / OC_stride; iter++) {
      // rmem to smem
      int _s_offset = (thx % num_thx_per_seg) * 4 + (thx + blockDim.x * thy) / num_thx_per_seg * OC_stride;
      load1x1FilterRegisterToShared(buffer, 
                                    Conv2dFilter_1_shared, 
                                    0, 
                                    _s_offset);
      load1x1FilterRegisterToShared(buffer + 4, 
                                    Conv2dFilter_1_shared, 
                                    blockDim.x / num_thx_per_seg * blockDim.y * OC_stride, 
                                    _s_offset);
      __syncthreads();

      // compute on smem
      for (int i = 0; i < IC_stride; i++) {
        Conv2dOutput_0_local[iter * 2 + 0] += intermediate[thy * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[(thx % 16) + 32 * i];
        Conv2dOutput_0_local[iter * 2 + 1] += intermediate[thy * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[16 + (thx % 16) + 32 * i];

        Conv2dOutput_0_local[iter * 2 + 8] += intermediate[256 + thy * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[(thx % 16) + 32 * i];
        Conv2dOutput_0_local[iter * 2 + 9] += intermediate[256 + thy * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[16 + (thx % 16) + 32 * i];
      }

      // gmem to rmem
      if (iter < 3) {
        _g_input_origin += OC_stride;
        load1x1FilterGlobalToRegister(Conv2dFilter_1,
                                      buffer,
                                      _g_input_origin,
                                      _g_input_offset);
        load1x1FilterGlobalToRegister(Conv2dFilter_1,
                                      buffer + 4,
                                      _g_input_origin + blockDim.x / num_thx_per_seg * blockDim.y * OC, // Row offset
                                      _g_input_offset);
      }
      __syncthreads();
    }
  }

    // if (bly == 0 && blx == 0 && thy == 0 && thx == 0) {
    //   printf("idx: %d, result: %d\n", 0, Conv2dOutput_0_local[0]);
    // }

  for (int i = 0; i < 4; i++) {
    int idx = (_g_h_blk + (thy / 2)) * W * OC + 
              (_g_w_blk + (thy % 2) * 2 + thx / 16) * OC + 
              i * OC_stride + 
              thx % 16;

    Conv2dOutput_0[idx] = Conv2dOutput_0_local[i * 2];
    Conv2dOutput_0[idx + 16] = Conv2dOutput_0_local[i * 2 + 1];
    Conv2dOutput_0[idx + 2 * W * OC] = Conv2dOutput_0_local[i * 2 + 8];
    Conv2dOutput_0[idx + 2 * W * OC + 16] = Conv2dOutput_0_local[i * 2 + 9];
    // if (bly == 0 && blx == 0 && thy == 0 && thx == 0) {
    //   printf("idx: %d, result: %d\n", idx, Conv2dOutput_0_local[i * 2]);
    // }
  }
}

// template __global__ void DepthConvFused_2_kernel0<56, 56, 128, 32>(const float* A, const float* B, const float* C, float* D);