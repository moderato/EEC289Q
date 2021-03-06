#define FILTER_H 3
#define FILTER_W 3
#define BUFFER_STRIDE 2 // The stride the buffer moves each time
#define STEP_OUTPUT_TILE_H 2
#define STEP_OUTPUT_TILE_W 2 // e.g. EACH BLOCK EACH STEP reads a 4x4xC_stride chunk and computes a 2x2xC_stride chunk in stage 1
#define OUTPUT_TILE_H 4
#define OUTPUT_TILE_W 4
#define READ_TILE_H (OUTPUT_TILE_H + FILTER_H - 1)
#define READ_TILE_W (OUTPUT_TILE_W + FILTER_W - 1) // The tile size of input data to be read, e.g. read 6x6 to compute 4x4

#define STEP_READ_TILE_H (STEP_OUTPUT_TILE_H + FILTER_H - 1)
#define STEP_READ_TILE_W (STEP_OUTPUT_TILE_W + FILTER_W - 1) // The tile size of input data needed in one step, e.g. read 4x4 to compute 2x2

#define STEP_H ((READ_TILE_H - STEP_READ_TILE_H) / STEP_OUTPUT_TILE_H + 1)
#define STEP_W ((READ_TILE_W - STEP_READ_TILE_W) / STEP_OUTPUT_TILE_W + 1) // The step (number of stride moving) needed for a row/col, e.g. reading 4x4 in a 6x6 tile takes 2 steps in a row and 2 steps in a col

#define BLOCK_Y_SIZE 4

__device__ void getSharedHW(bool isTall, int& h, int& w) {
  // 2x2 warps over H and W, 16 threads over C dimension

  if (isTall) {
    // (H, W) = (thy, thx / 16)) for loading 2 cols and 4 rows (isTall is true)
    h = threadIdx.y;
    w = threadIdx.x / 16;
  } else {
    // (H, W) = ((thy / 2), ((thy % 2) * 2 + (thx / 16))) for loading 2 rows 4 cols tile
    h = threadIdx.y / 2;
    w = (threadIdx.y) % 2 * 2 + (threadIdx.x / 16);
  }
}

__device__ bool inGlobalRange(int _g_h, int _g_w) {
  return 1 <= _g_h && _g_h < 57 && 1 <= _g_w && _g_w < 57;
}

// GMem coord
__device__ int getGlobalCoordFloat2(int _g_h, int _g_w, int _g_c_step, int W, int C, int C_stride) {
  return _g_h * C * W +
          _g_w * C +
          _g_c_step * C_stride +
          (threadIdx.x % 16) * 2;
}

// SMem coord in a form of circular buffer
__device__ int getSharedCoordFloat2(int _s_h, int _s_w) {
  return ((_s_h % 4) * 4 + (_s_w % 4)) * 32 + (threadIdx.x % 16) * 2;
}

__device__ void depthwiseConvSingleNum(float* Conv2dFilter_1_shared, 
                                      float* filter,
                                      float* DepthwiseConv2dOutput_0_local, 
                                      int _s_orig_h, int _s_orig_w) {
#pragma unroll
  for (int ry = 0; ry < FILTER_H; ++ry) {
    for (int rx = 0; rx < FILTER_W; ++rx) {
      int w = _s_orig_w + rx + threadIdx.y % STEP_OUTPUT_TILE_W;
      int h = _s_orig_h + ry + ((int)threadIdx.y / STEP_OUTPUT_TILE_W);
      int input_idx = threadIdx.x + 32 * ((w % 4) + (h % 4) * 4);

      DepthwiseConv2dOutput_0_local[0] += (
          Conv2dFilter_1_shared[input_idx] * filter[ry * FILTER_W + rx]);
    }
  }
}

__device__ void loadGlobalToShared(const float* Input, int _g_coord, 
                                  int _g_h, int _g_w,
                                  float* Conv2dFilter_1_shared, int _s_coord) {
  ((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = 
    inGlobalRange(_g_h, _g_w) ? 
      ((float2*)(Input + _g_coord - 7296))[0] : 
      make_float2(0.0e+00f, 0.0e+00f);
}

__device__ void loadGlobalToRegister(const float* Input, int _g_coord, 
                                    int _g_h, int _g_w,
                                    float* buffer) {
  ((float2*)(buffer))[0] = 
      inGlobalRange(_g_h, _g_w) ? 
        ((float2*)(Input + _g_coord - 7296))[0] : 
        make_float2(0.0e+00f, 0.0e+00f);
}

__device__ void getGlobalSharedHW(bool isTall,
                                  int _g_h_blk, int _g_w_blk,
                                  int _s_h_coord, int _s_w_coord,
                                  int& _g_h, int& _g_w,
                                  int& _s_h, int& _s_w) {
  int h = 0, w = 0;
  getSharedHW(isTall, h, w);
  _g_h = _g_h_blk + _s_h_coord + h;
  _g_w = _g_w_blk + _s_w_coord + w;
  _s_h =          + _s_h_coord + h;
  _s_w =          + _s_w_coord + w;
}

extern "C" __global__ void DepthConvFused_2_kernel0(const float* Input, 
                                                    const float* DepthwiseFilter_1, 
                                                    const float* Conv2dFilter_1, 
                                                    float* Conv2dOutput_0, 
                                                    int H, int W, int C, int C_stride) {

  float Conv2dOutput_0_local[OUTPUT_TILE_H * OUTPUT_TILE_W] = { 0.0f };
  float DepthwiseConv2dOutput_0_local[1] = { 0.0f };
  
  extern __shared__ float s[];
  float *intermediate = s;
  float *Conv2dFilter_1_shared = &s[OUTPUT_TILE_H * OUTPUT_TILE_W * C_stride];

  //
  float filter[FILTER_H * FILTER_W];
  float buffer[8];
  int thx = threadIdx.x, thy = threadIdx.y, blx = blockIdx.x, bly = blockIdx.y;
  int _g_h_blk = bly * OUTPUT_TILE_H;
  int _g_w_blk = blx * OUTPUT_TILE_W;
  int _g_h, _g_w, _s_h, _s_w;
  //

  for (int rc_outer_v = 0; rc_outer_v < (int)(C / C_stride); ++rc_outer_v) {
    int _g_c_step = rc_outer_v;

    ///////////// Preprocessing /////////////
    // Load filter to RMem
    #pragma unroll
    for (int ry = 0; ry < FILTER_H; ++ry) {
      for (int rx = 0; rx < FILTER_W; ++rx) {
        filter[ry * FILTER_W + rx] = DepthwiseFilter_1[thx + (rc_outer_v * C_stride) + (ry * C * FILTER_W) + (rx * C)];
      }
    }

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
    // G->S
    int _s_h_coord = 0;
    int _s_w_coord = 0;
    // Get HWs
    getGlobalSharedHW(true, _g_h_blk, _g_w_blk, _s_h_coord, _s_w_coord, _g_h, _g_w, _s_h, _s_w);
    // Get global and shared coords
    int _s_coord = getSharedCoordFloat2(_s_h, _s_w);
    int _g_coord = getGlobalCoordFloat2(_g_h, _g_w, _g_c_step, W, C, C_stride);
    // Load from GMem to SMem
    loadGlobalToShared(Input, _g_coord, _g_h, _g_w, Conv2dFilter_1_shared, _s_coord);

    // G->R
    _s_h_coord = 0;
    _s_w_coord = BUFFER_STRIDE;
    // Get HWs
    getGlobalSharedHW(true, _g_h_blk, _g_w_blk, _s_h_coord, _s_w_coord, _g_h, _g_w, _s_h, _s_w);
    // Load from GMem to RMem
    _g_coord = getGlobalCoordFloat2(_g_h, _g_w, _g_c_step, W, C, C_stride);
    loadGlobalToRegister(Input, _g_coord, _g_h, _g_w, buffer);

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
      int step_h = loop / STEP_W, step_w = loop % STEP_W;
      DepthwiseConv2dOutput_0_local[0] = 0.0e+00f;

      ((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = ((float2*)(buffer))[0];
      __syncthreads();

      // If the input data tile to be read is tall or long
      isTall = (step_w != STEP_W - 1);

      // The origin point (ref (0,0)) of the intermediate data tile to be written
      _s_orig_h = step_h * BUFFER_STRIDE;
      _s_orig_w = (step_h % 2) ? (BUFFER_STRIDE * (STEP_W - 1 - step_w)) : (BUFFER_STRIDE * step_w); // Even rows in increase order, odd rows in decrease order.

      // The origin point (ref (0,0)) of the input data tile to be read
      _s_h_coord = _s_orig_h + (!isTall) * 2 * BUFFER_STRIDE;
      _s_w_coord = _s_orig_w + isTall * (2 - 3 * (step_h % 2)) * BUFFER_STRIDE;

      // if (bly == 0 && blx == 0 && thy == 0 && thx == 0 && rc_outer_v == 0) {
      //   printf("num of loops: %d, current loop: %d\n", STEP_H * STEP_W, loop);
      //   // printf("_s_orig_w: %d, (_s_orig_h % 2): %d, (BUFFER_STRIDE * (STEP_W - step_w)): %d, (BUFFER_STRIDE * step_w): %d\n", _s_orig_w, (_s_orig_h % 2), (BUFFER_STRIDE * (STEP_W - step_w)), (BUFFER_STRIDE * step_w));
      //   // printf("_s_w_coord: %d, _s_orig_w: %d, isTall: %d, (step_h % 2): %d\n", _s_w_coord, _s_orig_w, isTall, (step_h % 2));

      //   printf("_s_orig_h: %d, _s_orig_w: %d\n", _s_orig_h, _s_orig_w);
      //   printf("_s_h_coord: %d, _s_w_coord: %d\n", _s_h_coord, _s_w_coord);
      //   printf("*********\n");
      // }

      shared_idx = thx + ((thy % 2) + thy / 2 * 4 + _s_orig_w) * C_stride + _s_orig_h * C;

      // Depthwise
      depthwiseConvSingleNum(Conv2dFilter_1_shared, filter, DepthwiseConv2dOutput_0_local, _s_orig_h, _s_orig_w);
      // Write from RMem to SMem
      intermediate[shared_idx] = DepthwiseConv2dOutput_0_local[0];

      if (loop != 3) {
        // Get HWs
        getGlobalSharedHW(isTall, _g_h_blk, _g_w_blk, _s_h_coord, _s_w_coord, _g_h, _g_w, _s_h, _s_w);

        // Load from GMem to RMem
        _g_coord = getGlobalCoordFloat2(_g_h, _g_w, _g_c_step, W, C, C_stride);
        loadGlobalToRegister(Input, _g_coord, _g_h, _g_w, buffer);
      }
      __syncthreads();
    }

    // gmem to rmem
    ((float4*)(buffer))[0] = ((float4*)(Conv2dFilter_1 + (thy * 512) + (thx / 8) * 128 + (thx % 8) * 4 + rc_outer_v * 4096 + 0))[0];
    ((float4*)(buffer + 4))[0] = ((float4*)(Conv2dFilter_1 + ((thy * 512) + (thx / 8) * 128 + (thx % 8) * 4 + rc_outer_v * 4096) + 2048))[0];

    for (int iter = 0; iter < 4; iter++) {
      // rmem to smem
      ((float4*)(Conv2dFilter_1_shared + (thy * 128) + (thx / 8) * C_stride + (thx % 8) * 4 + 0))[0] = ((float4*)(buffer))[0];
      ((float4*)(Conv2dFilter_1_shared + (thy * 128) + (thx / 8) * C_stride + (thx % 8) * 4 + 512))[0] = ((float4*)(buffer + 4))[0];

      __syncthreads();

      // compute on smem
      for (int i = 0; i < 32; i++) {
        Conv2dOutput_0_local[iter * 2 + 0] += intermediate[thy * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[(thx % 16) + 32 * i];
        Conv2dOutput_0_local[iter * 2 + 1] += intermediate[thy * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[16 + (thx % 16) + 32 * i];

        Conv2dOutput_0_local[iter * 2 + 8] += intermediate[256 + thy * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[(thx % 16) + 32 * i];
        Conv2dOutput_0_local[iter * 2 + 9] += intermediate[256 + thy * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[16 + (thx % 16) + 32 * i];
      }

      // gmem to rmem
      if (iter < 3) {
        ((float4*)(buffer))[0] = ((float4*)(Conv2dFilter_1 + thy * 512 + (iter+1) * C_stride + (thx / 8) * 128 + (thx % 8) * 4 + rc_outer_v * 4096 + 0))[0];
        ((float4*)(buffer + 4))[0] = ((float4*)(Conv2dFilter_1 + thy * 512 + (iter+1) * C_stride + (thx / 8) * 128 + (thx % 8) * 4 + rc_outer_v * 4096 + 2048))[0];
      }
      __syncthreads();
    }
  }

  for (int i = 0; i < 4; i++) {
    int idx = (_g_h_blk + (thy / 2)) * W * C + 
              (_g_w_blk + (thy % 2) * 2 + thx / 16) * C + 
              i * C_stride + 
              thx % 16;

    Conv2dOutput_0[idx] = Conv2dOutput_0_local[i * 2];
    Conv2dOutput_0[idx + 16] = Conv2dOutput_0_local[i * 2 + 1];
    Conv2dOutput_0[idx + 2 * W * C] = Conv2dOutput_0_local[i * 2 + 8];
    Conv2dOutput_0[idx + 2 * W * C + 16] = Conv2dOutput_0_local[i * 2 + 9];
  }
}

