#include "helper.cuh"

// OC_stride = IC_stride for now
template <int H, int W, int IC, int OC, 
          int IC_stride, int OC_stride,
          int REG_BUFFER_SIZE, int OC_STEP>
__global__ void DepthConvFused_2_kernel0(const float* Input, 
                                         const float* DepthwiseFilter_1, 
                                         const float* Conv2dFilter_1, 
                                         float* Conv2dOutput_0) {

  static_assert((OC_stride <= OC && OC_stride > 0), "!");
  static_assert((IC_stride <= IC && IC_stride > 0), "!");
  static_assert((IC_stride & (IC_stride - 1)) == 0, "!"); // IC_stride is power of 2
  static_assert((OC_stride & (OC_stride - 1)) == 0, "!"); // OC_stride is Power of 2

  // Params
  int thx = threadIdx.x, thy = threadIdx.y, blx = blockIdx.x, bly = blockIdx.y;
  int num_thx_per_seg = OC_stride / 4;
  int _g_coord, _s_coord, _s_h_coord, _s_w_coord;
  int _s_orig_h, _s_orig_w, shared_idx;
  bool isTall;

  // Shared memory
  extern __shared__ float s[];
  float *intermediate = s;
  float *Conv2dFilter_1_shared = &s[OUTPUT_TILE_H * OUTPUT_TILE_W * OC_stride];

  // Registers
  float Conv2dOutput_0_local[OUTPUT_TILE_H * OUTPUT_TILE_W] = { 0.0f };
  float DepthwiseConv2dOutput_0_local[1] = { 0.0f };
  float filter[FILTER_H * FILTER_W];
  // 8 = OC_stride * OC_stride / (blockDim.y * blockDim.x)
  float buffer[REG_BUFFER_SIZE];

  for (int _g_oc_step = 0; _g_oc_step < IC / IC_stride; _g_oc_step++) {
    ///////////// Preprocessing /////////////
    // Load filter to RMem
    loadDepthwiseFilterGlobalToRegister<IC, IC_stride>(DepthwiseFilter_1, filter, _g_oc_step);

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
    prefetchInputData<H, W, IC, IC_stride>(Input,                 /* Input src    */
                                          Conv2dFilter_1_shared, /* Shared dst   */
                                          buffer,                /* Register dst */
                                          _g_coord, _s_coord,
                                          _g_oc_step);

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
      shared_idx = thx + 
      				(_s_orig_w + (thy % STEP_OUTPUT_TILE_W)) * IC_stride + 
      				(_s_orig_h + (thy / STEP_OUTPUT_TILE_H)) * IC_stride * OUTPUT_TILE_W;
      depthwiseConvSingleNum<IC_stride>(Conv2dFilter_1_shared,
                                        filter,
                                        DepthwiseConv2dOutput_0_local,
                                        _s_orig_h, _s_orig_w);
      intermediate[shared_idx] = DepthwiseConv2dOutput_0_local[0];

      // if (bly == 0 && blx == 0 && thy == 2 && thx == 0) {
      //     printf("loop: %d, _s_orig_h: %d, _s_orig_w: %d, _s_h_coord: %d, _s_w_coord: %d, shared_idx: %d\n", 
      //     			loop, _s_orig_h, _s_orig_w, _s_h_coord, _s_w_coord, shared_idx);
      // }

      if (loop + 1 != STEP_H * STEP_W) {
        // Load from global to register
        loadWrapper<H, W, IC, IC_stride>(Input, buffer,
                                          isTall,
                                          false, /* Load to register: false = not to SMem */
                                          _g_coord, _s_coord,
                                          _s_h_coord, _s_w_coord,
                                          _g_oc_step);
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
      // OC_STRIDE_SPLIT usually 16
      for (int i = 0; i < IC_stride; i++) {
        int inter_offset = thy * 64 + (thx / OC_STRIDE_SPLIT) * IC_stride + i;
        int filter_offset = (thx % OC_STRIDE_SPLIT) + OC_stride * i;

        // if (bly == 0 && blx == 0 && thy == 2 && thx == 0 && iter == 0) {
    		//   printf("Conv2dOutput_0_local[16]: %f\n", Conv2dOutput_0_local[16]);
    		// }

        // // 8 = BLOCK_Y_SIZE * 32 / OC_STRIDE_SPLIT, which is basically fixed
        // #pragma unroll
        //   for (int j = 0, a = iter * 2, b = inter_offset; 
        //         j < OUTPUT_TILE_H * OUTPUT_TILE_W / 8; 
        //         j++, a += OC_STEP, b += 8 * OC_stride) {
        //     Conv2dOutput_0_local[a]     += intermediate[b] * Conv2dFilter_1_shared[filter_offset];
        //     Conv2dOutput_0_local[a+1]   += intermediate[b] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
        //   }

        // Replace the above with the below to get a ~15us speedup for W=8,H=4
        {
          Conv2dOutput_0_local[iter * 2 + 0] 	+= intermediate[inter_offset] 		  * Conv2dFilter_1_shared[filter_offset];
          Conv2dOutput_0_local[iter * 2 + 1] 	+= intermediate[inter_offset] 		  * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
          Conv2dOutput_0_local[iter * 2 + 8] 	+= intermediate[256 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
          Conv2dOutput_0_local[iter * 2 + 9] 	+= intermediate[256 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];

          Conv2dOutput_0_local[iter * 2 + 16] += intermediate[512 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
          Conv2dOutput_0_local[iter * 2 + 17] += intermediate[512 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
          Conv2dOutput_0_local[iter * 2 + 24] += intermediate[768 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
          Conv2dOutput_0_local[iter * 2 + 25] += intermediate[768 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
        }
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

  for (int _g_oc_step = 0; _g_oc_step < (OC / OC_stride); _g_oc_step++) {
    int idx = getOutputBaseCoord<W, OC, OC_stride>(_g_oc_step);

#pragma unroll
  	for (int i = 0, a = 0, b = _g_oc_step * 2; 
          i < OUTPUT_TILE_H * OUTPUT_TILE_W / 8;
          i++, a += BLOCK_Y_SIZE * 2 / OUTPUT_TILE_W * W * OC, b += 8) {
  	    Conv2dOutput_0[idx + a] 		= 	Conv2dOutput_0_local[b];
  	    Conv2dOutput_0[idx + a + 16] 	= 	Conv2dOutput_0_local[b + 1];
  	}
  }
}