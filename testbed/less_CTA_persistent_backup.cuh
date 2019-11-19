#include "helper.cuh"

// OC_stride = IC_stride for now
template <int H, int W, int IC, int OC, 
          int IC_stride, int OC_stride,
          int REG_BUFFER_SIZE, int OC_STEP,
          int NUM_THX_PER_SEG,
          int total_step_num_h, int total_step_num_w>
__global__ void DepthConvFused_2_kernel0(const float* Input, 
                                         const float* DepthwiseFilter_1, 
                                         const float* Conv2dFilter_1, 
                                         float* Conv2dOutput_0) {

  {
    static_assert((OC_stride <= OC && OC_stride > 0), "!");
    static_assert((IC_stride <= IC && IC_stride > 0), "!");
    static_assert((IC_stride & (IC_stride - 1)) == 0, "!"); // IC_stride is power of 2
    static_assert((OC_stride & (OC_stride - 1)) == 0, "!"); // OC_stride is Power of 2
    static_assert((NUM_THX_PER_SEG == OC_stride / 4), "!");
    // static_assert((STEP_PER_ROUND_CTA % 2) != 0, "!"); // STEP_PER_ROUND_CTA should be even to make sure the output works
  }

  // Params
  int thx = threadIdx.x, thy = threadIdx.y, blx = blockIdx.x; // 1D grid
  int global_start_step = blx * STEP_PER_CTA;
  int round = 0;
  int _g_coord, _s_coord, _s_h_coord, _s_w_coord, global_h, global_w;
  int shared_idx;
  bool isTall;

  // Shared memory
  extern __shared__ float s[];
  float *intermediate = s;
  float *Conv2dFilter_1_shared = &s[OUTPUT_SIZE_HW * IC_stride];

	do {
		__syncthreads();
		// Registers
		float Conv2dOutput_0_local[OUTPUT_SIZE_HW] = { 0.0f };
		float DepthwiseConv2dOutput_0_local[1] = { 0.0f };
		float filter[FILTER_H * FILTER_W] = { 0.0f };
		// 8 = OC_stride * OC_stride / (BLOCK_DIM_Y * BLOCK_DIM_X)
		float buffer[REG_BUFFER_SIZE] = { 0.0f };
		int global_h_array[STEP_PER_ROUND_CTA] = { -1 };
  		int global_w_array[STEP_PER_ROUND_CTA] = { -1 };

		// Load and compute
		for (int _g_ic_step = 0; _g_ic_step < (IC / IC_stride); _g_ic_step++) {
			///////////// Preprocessing /////////////
			// Load filter to RMem
			loadDepthwiseFilterGlobalToRegister<IC, IC_stride>(DepthwiseFilter_1, filter, _g_ic_step);

			// Calculate the current step HW: return global_h, global_w, isTall
			spaceFillingGlobal<total_step_num_h, total_step_num_w>(global_start_step, isTall,
																	global_h, global_w);

			global_h_array[0] = global_h;
			global_w_array[0] = global_w;

			// Prefetch the 4x4 tile that caculates the above 2x2
			prefetchInputData<H, W, IC, IC_stride>(Input,                /* Input src    */
			                                      Conv2dFilter_1_shared, /* Shared dst   */
			                                      buffer,                /* Register dst */
												  global_h, global_w,
			                                      _g_coord, _s_coord,
			                                      _g_ic_step, isTall);

			//////////////////////////// Loop ////////////////////////////
			for (int loop = 0; loop < STEP_PER_ROUND_CTA; loop++) {
				DepthwiseConv2dOutput_0_local[0] = 0.0e+00f;

				// Load from register to shared
				((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = ((float2*)(buffer))[0];
				__syncthreads();

				// if (blx == 13 && thy == 0 && thx == 0 && loop == 3 && _g_ic_step == 0)
				// 	for (int i = 0; i < 16; i++) {
				// 		printf("coord: %d, input: %f\n", i * IC_stride, Conv2dFilter_1_shared[i * IC_stride]);
				// 	}

				// Return _s_h_coord, _s_w_coord
				spaceFillingSharedPersistent<total_step_num_h, total_step_num_w>(
																	global_start_step + loop, isTall,
														            _s_h_coord, _s_w_coord);

				// Depthwise and store the result and store result to RMem
				shared_idx = thx + (loop * STEP_OUTPUT_TILE_W * STEP_OUTPUT_TILE_H + thy) * IC_stride;

				depthwiseConvSingleNum<IC_stride>(Conv2dFilter_1_shared,
					                                filter,
					                                DepthwiseConv2dOutput_0_local,
					                                global_h, global_w);
				intermediate[shared_idx] = DepthwiseConv2dOutput_0_local[0];

				// if (shared_idx == 256 && blx == 0 && _g_ic_step == 0) {
				// 	printf("loop: %d, thy: %d, thx: %d, _g_coord: %d, _s_coord: %d, _s_h_coord: %d, _s_w_coord: %d, intermediate: %d\n", 
				// 		loop, thy, thx, _g_coord, _s_coord, _s_h_coord, _s_w_coord, DepthwiseConv2dOutput_0_local[0]);
				// }

				// if (blx == 13 && thy == 0 && thx == 0 && _g_ic_step == 0)
				// 	printf("****After spaceFillingSharedPersistent\nround: %d, loop: %d, _s_h_coord: %d, _s_w_coord: %d, global_h: %d, global_w: %d\n", round, loop, _s_h_coord, _s_w_coord, global_h, global_w);

				if (!isLastStep(loop)) {
					// if (blockIdx.y == 0 && blockIdx.x == 13 && thy == 0 && thx == 0 && _g_ic_step == 0) {
					// 	printf("Calling loadWrapper independently.\n");
					// }

					// Load from global to register
					loadWrapper<H, W, IC, IC_stride>(Input, /* Source */
													buffer, /* Destination */
													isTall,
													false, 	/* Load to register: false = not to SMem */
													global_h, global_w,
													_g_coord, _s_coord,
													_s_h_coord, _s_w_coord,
													_g_ic_step);
					// if (blx == 13 && thy == 0 && thx == 0)
					// 	printf("****After independent loadWrapper\nround: %d, loop: %d, _s_h_coord: %d, _s_w_coord: %d, global_h: %d, global_w: %d\n", round, loop, _s_h_coord, _s_w_coord, global_h, global_w);

					// Update global_h, global_w and isTall for the next loop
					spaceFillingGlobal<total_step_num_h, total_step_num_w>(global_start_step + loop + 1,
																			isTall,
																			global_h, global_w);

					global_h_array[loop+1] = global_h;
					global_w_array[loop+1] = global_w;
				}
				__syncthreads();
			}

			// if (blx == 0 && thy == 0 && thx == 0 && _g_ic_step == 0) {
			// 		for (int i = 0; i < 16; i++) {
			// 			printf("coord: %d, intermediate: %f\n", i * IC_stride, intermediate[i * IC_stride]);
			// 		}
			// 	}

			// gmem to rmem
			int _g_input_offset = _g_ic_step * OC * OC_stride + /* origin */
			                      (thx % NUM_THX_PER_SEG) * 4 + 
			                      (thx + BLOCK_DIM_X * thy) / NUM_THX_PER_SEG * OC; /* offset */

			// Slow for ~25us don't know why
			loadBlockGlobalToRegister<IC_stride, OC, NUM_THX_PER_SEG>(Conv2dFilter_1, buffer, 
			                                                           _g_input_offset, 0);

			for (int iter = 0; iter < OC / OC_stride; iter++) {
			  // rmem to smem
			  int _s_offset = (thx % NUM_THX_PER_SEG) * 4 + 
			                  (thx + BLOCK_DIM_X * thy) / NUM_THX_PER_SEG * OC_stride;

			  loadBlockRegisterToShared<IC_stride, OC_stride, NUM_THX_PER_SEG>(buffer, 
			  																	Conv2dFilter_1_shared,
			                                                                   	0, _s_offset);

			  __syncthreads();

			  // compute on smem
			  // OC_STRIDE_SPLIT usually 16
			  for (int i = 0; i < IC_stride; i++) {
			    int inter_offset = thy * 64 + (thx / OC_STRIDE_SPLIT) * IC_stride + i;
			    int filter_offset = (thx % OC_STRIDE_SPLIT) + OC_stride * i;

			    // 8 = BLOCK_DIM_Y * 32 / OC_STRIDE_SPLIT, which is basically fixed
			    #pragma unroll
			    for (int j = 0, a = iter * 2, b = inter_offset;
		            j < OUTPUT_SIZE_HW / 8;
		            j++, a += OC_STEP, b += 8 * OC_stride)
			    {

			        Conv2dOutput_0_local[a]     += intermediate[b] * Conv2dFilter_1_shared[filter_offset];
			        Conv2dOutput_0_local[a+1]   += intermediate[b] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];

			   		// if (blx == 0 && thy == 0 && thx == 0 && iter == 0 && a == 0)
						// // printf("i: %d, j: %d, a: %d, b: %d, inter_offset: %d, filter_offset: %d, inter: %f, filter: %f, accumulator: %f\n", i, j, a, b, inter_offset, filter_offset, intermediate[b], Conv2dFilter_1_shared[filter_offset], Conv2dOutput_0_local[a]);
			    }

			    // // Replace the above with the below to get a ~15us speedup for W=8,H=4
			    // {
			    //   Conv2dOutput_0_local[iter * 2 + 0] += intermediate[inter_offset] * Conv2dFilter_1_shared[filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 1] += intermediate[inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 8] += intermediate[256 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 9] += intermediate[256 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];

			    //   Conv2dOutput_0_local[iter * 2 + 16] += intermediate[512 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 17] += intermediate[512 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 24] += intermediate[768 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 25] += intermediate[768 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];

			    //   Conv2dOutput_0_local[iter * 2 + 32] += intermediate[1024 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 33] += intermediate[1024 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 40] += intermediate[1280 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 41] += intermediate[1280 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];

			    //   Conv2dOutput_0_local[iter * 2 + 48] += intermediate[1536 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 49] += intermediate[1536 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 56] += intermediate[1792 + inter_offset] * Conv2dFilter_1_shared[filter_offset];
			    //   Conv2dOutput_0_local[iter * 2 + 57] += intermediate[1792 + inter_offset] * Conv2dFilter_1_shared[OC_STRIDE_SPLIT + filter_offset];
			    // }
			  }

			  __syncthreads();

			  // gmem to rmem
			  if (iter + 1 != OC / OC_stride) {
			    _g_input_offset += OC_stride;
			    loadBlockGlobalToRegister<IC_stride, OC, NUM_THX_PER_SEG>(Conv2dFilter_1, buffer, 
			                                                              _g_input_offset, 0);
			  }
			  __syncthreads();
			}
		}

		// Write
		for (int _g_oc_step = 0; _g_oc_step < (OC / OC_stride); _g_oc_step++) {

			#pragma unroll
		    for (int i = 0, b = _g_oc_step * 2;
		          i < OUTPUT_SIZE_HW / 8;
		          i++, b += 8)
		    {
		    	int idx = getOutputCoord<W, OC, OC_stride>(_g_oc_step, global_h_array, global_w_array, (thy / 2 + 2 * i));

		    	// if (idx == 1536) {
		    	// 	printf("round: %d, global_start_step: %d, idx: %d, blx: %d, thy: %d, thx: %d, _g_oc_step: %d, i: %d, b: %d, result: %f, global_h: %d, global_w: %d\n", round, global_start_step, idx, blx, thy, thx, _g_oc_step, i, b, Conv2dOutput_0_local[b], global_h_array[thy / 2 + 2 * i], global_w_array[thy / 2 + 2 * i]);
		    	// 	// for (int k = 0; k < STEP_PER_ROUND_CTA; k++) {
		    	// 	// 	printf("k: %d, global_h: %d, global_w: %d\n", k, global_h_array[k], global_w_array[k]);
		    	// 	// }
		    	// }
		        Conv2dOutput_0[idx]     				=   Conv2dOutput_0_local[b];
		        Conv2dOutput_0[idx + OC_STRIDE_SPLIT]  	=   Conv2dOutput_0_local[b + 1];
		    }
		}

		round++;
		if (round >= ROUND_PER_CTA)
			break;
		global_start_step += STEP_PER_ROUND_CTA;

	} while (true);

}