// Fix this value in a foreseeable future.
#define BLOCK_Y_SIZE 4
#define FILTER_H 3
#define FILTER_W 3
#define BUFFER_STRIDE 2 // The stride the buffer moves each time
#define STEP_OUTPUT_TILE_H 2
#define STEP_OUTPUT_TILE_W 2 // e.g. EACH BLOCK EACH STEP reads a 4x4xIC_stride chunk and computes a 2x2xIC_stride chunk in stage 1
#define STEP_READ_TILE_H (STEP_OUTPUT_TILE_H + FILTER_H - 1)
#define STEP_READ_TILE_W (STEP_OUTPUT_TILE_W + FILTER_W - 1) // The tile size of input data needed in one step, e.g. read 4x4 to compute 2x2

/********************* Can be changed *********************/
#define OUTPUT_TILE_H 4
#define OUTPUT_TILE_W 8

#define READ_TILE_H (OUTPUT_TILE_H + FILTER_H - 1)
#define READ_TILE_W (OUTPUT_TILE_W + FILTER_W - 1) // The tile size of input data to be read, e.g. read 6x6 to compute 4x4

#define STEP_H ((READ_TILE_H - STEP_READ_TILE_H) / STEP_OUTPUT_TILE_H + 1)
#define STEP_W ((READ_TILE_W - STEP_READ_TILE_W) / STEP_OUTPUT_TILE_W + 1) // The step (number of stride moving) needed for a row/col, e.g. reading 4x4 in a 6x6 tile takes 2 steps in a row and 2 steps in a col

#define OC_STRIDE_SPLIT 16 // Split OC_stride of 1x1 filter
/**********************************************************/

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

__device__ void load1x1FilterGlobalToRegister(const float* src, float* dst, int origin, int offset) {
  ((float4*)(dst))[0] = ((float4*)(src + (
      origin + offset
    )
  ))[0];
}

__device__ void load1x1FilterRegisterToShared(float* src, float* dst, int origin, int offset) {
  ((float4*)(dst + (
      origin + offset
    )
  ))[0] = ((float4*)(src))[0];
}

template<int H, int W>
__device__ bool inGlobalRange(int _g_h, int _g_w) {
  return 1 <= _g_h && (_g_h < H + 1) && 1 <= _g_w && (_g_w < W + 1);
}

// SMem coord in a form of circular buffer
template<int IC_stride>
__device__ int getSharedCoordFloat2(int _s_h, int _s_w) {
  return ((_s_h % STEP_READ_TILE_H) * STEP_READ_TILE_W + 
            (_s_w % STEP_READ_TILE_W)) * IC_stride + (threadIdx.x % 16) * 2;
}

// GMem coord
template<int W, int IC, int IC_stride>
__device__ int getGlobalCoordFloat2(int _g_h, int _g_w, int IC_step) {
  return _g_h * IC * W +
          _g_w * IC +
          IC_stride * IC_step +
          (threadIdx.x % 16) * 2;
}

template<int W, int OC, int OC_stride>
__device__ int getOutputBaseCoord(int _g_oc_step) {
  int _g_h_blk = blockIdx.y * OUTPUT_TILE_H;
  int _g_w_blk = blockIdx.x * OUTPUT_TILE_W;

  return (_g_h_blk + (threadIdx.y * 2 / OUTPUT_TILE_W)) * W * OC + 
      (_g_w_blk + (threadIdx.y * 2 % OUTPUT_TILE_W) + threadIdx.x / OC_STRIDE_SPLIT) * OC + 
      _g_oc_step * OC_stride + 
      threadIdx.x % OC_STRIDE_SPLIT;
}

template<int IC, int IC_stride>
__device__ void loadDepthwiseFilterGlobalToRegister(const float* src, float* dst, int IC_step) {
#pragma unroll
  for (int ry = 0; ry < FILTER_H; ++ry) {
    for (int rx = 0; rx < FILTER_W; ++rx) {
      dst[ry * FILTER_W + rx] = src[threadIdx.x + IC_stride * IC_step + (ry * IC * FILTER_W) + (rx * IC)];
    }
  }
}

template<int IC_stride>
__device__ void depthwiseConvSingleNum(float* Conv2dFilter_1_shared, 
                                      float* filter,
                                      float* DepthwiseConv2dOutput_0_local, 
                                      int _s_orig_h, int _s_orig_w) {
#pragma unroll
  for (int ry = 0; ry < FILTER_H; ++ry) {
    for (int rx = 0; rx < FILTER_W; ++rx) {
      int w = _s_orig_w + rx + (threadIdx.y % STEP_OUTPUT_TILE_W);
      int h = _s_orig_h + ry + (threadIdx.y / STEP_OUTPUT_TILE_W);
      int input_idx = threadIdx.x + IC_stride * ((w % STEP_READ_TILE_W) + 
                                                  (h % STEP_READ_TILE_H) * STEP_READ_TILE_W);

      DepthwiseConv2dOutput_0_local[0] += (
          Conv2dFilter_1_shared[input_idx] * filter[ry * FILTER_W + rx]);
    }
  }
}

template<int H, int W>
__device__ void loadGlobalWithBoundCheck(const float* src, float* dst,
                                          int offset,
                                          int _g_h, int _g_w) {
  ((float2*)(dst))[0] = 
      inGlobalRange<H, W>(_g_h, _g_w) ? 
        ((float2*)(src + offset))[0] : 
        make_float2(0.0e+00f, 0.0e+00f);
}

template<int H, int W, int IC, int IC_stride>
__device__ void loadWrapper(const float* src, float* dst,
                            bool isTall, 
                            bool toShared, /* Whether load global to SMem or RMem*/
                            int& _g_coord, int& _s_coord,
                            int _s_h_coord, int _s_w_coord,
                            int IC_step) {
  int _g_h_blk = blockIdx.y * OUTPUT_TILE_H;
  int _g_w_blk = blockIdx.x * OUTPUT_TILE_W;
  int _g_h, _g_w, _s_h, _s_w;

  // Get global and shared HW coord
  getGlobalSharedHW(isTall, _g_h_blk, _g_w_blk, _s_h_coord, _s_w_coord, _g_h, _g_w, _s_h, _s_w);
  // Get global and shared coords
  _g_coord = getGlobalCoordFloat2<W, IC, IC_stride>(_g_h, _g_w, IC_step);
  _s_coord = getSharedCoordFloat2<IC_stride>(_s_h, _s_w);
  int offset = toShared ? _s_coord : 0;
  // Load from GMem to SMem
  loadGlobalWithBoundCheck<H, W>(src, dst + offset, 
                                _g_coord - (W + 1) * IC, /* pad = 1 in all sides, might be changed later */
                                _g_h, _g_w);
}

template<int H, int W, int IC, int IC_stride>
__device__ void prefetchInputData(const float* src, float* _s_dst, float* _r_dst,
                                  int& _g_coord, int& _s_coord,
                                  int IC_step) {
  // G->S
  int _s_h_coord = 0;
  int _s_w_coord = 0;
  loadWrapper<H, W, IC, IC_stride>(src, _s_dst, true, true, _g_coord, _s_coord, _s_h_coord, _s_w_coord, IC_step);

  // G->R
  _s_h_coord = 0;
  _s_w_coord = BUFFER_STRIDE;
  loadWrapper<H, W, IC, IC_stride>(src, _r_dst, true, false, _g_coord, _s_coord, _s_h_coord, _s_w_coord, IC_step);
}

/**************
0-->-->-->-->|
             |
|<--<--<--<--v
|
v-->-->-->-->|
             |
x<--<--<--<--v
***************/
__device__ void spaceFillingCalculation(int loop, bool& isTall,
                                        int& _s_orig_h, int& _s_orig_w,
                                        int& _s_h_coord, int& _s_w_coord) {
  int step_h = loop / STEP_W, step_w = loop % STEP_W;

  // The origin point (ref (0,0)) of the intermediate data tile to be written
  _s_orig_h = step_h * BUFFER_STRIDE;
  _s_orig_w = (step_h % 2) ? (BUFFER_STRIDE * (STEP_W - 1 - step_w)) : (BUFFER_STRIDE * step_w); // Even rows in increase order, odd rows in decrease order.

  // If the input data tile to be read is tall or long
  isTall = (step_w != STEP_W - 1);

  // The origin point (ref (0,0)) of the input data tile to be read
  _s_h_coord = _s_orig_h + (!isTall) * 2 * BUFFER_STRIDE;
  _s_w_coord = _s_orig_w + isTall * (2 - 3 * (step_h % 2)) * BUFFER_STRIDE;
}