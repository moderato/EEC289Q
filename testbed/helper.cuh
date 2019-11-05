// Fix this value in a foreseeable future.
#define FILTER_H 3
#define FILTER_W 3
#define BUFFER_STRIDE 2 // The stride the buffer moves each time
#define STEP_OUTPUT_TILE_H 2
#define STEP_OUTPUT_TILE_W 2 // e.g. EACH BLOCK EACH STEP reads a 4x4xIC_stride chunk and computes a 2x2xIC_stride chunk in stage 1
#define STEP_READ_TILE_H (STEP_OUTPUT_TILE_H + FILTER_H - 1)
#define STEP_READ_TILE_W (STEP_OUTPUT_TILE_W + FILTER_W - 1) // The tile size of input data needed in one step, e.g. read 4x4 to compute 2x2
#define BLOCK_DIM_Y (STEP_OUTPUT_TILE_H * STEP_OUTPUT_TILE_W)

/********************* Can be changed *********************/
#define BLOCK_DIM_X 32
#define OC_STRIDE_SPLIT 16 // Split OC_stride of 1x1 filter

#define OUTPUT_TILE_H 4
#define OUTPUT_TILE_W 8

#define READ_TILE_H (OUTPUT_TILE_H + FILTER_H - 1)
#define READ_TILE_W (OUTPUT_TILE_W + FILTER_W - 1) // The tile size of input data to be read, e.g. read 6x6 to compute 4x4

#define STEP_H ((READ_TILE_H - STEP_READ_TILE_H) / STEP_OUTPUT_TILE_H + 1)
#define STEP_W ((READ_TILE_W - STEP_READ_TILE_W) / STEP_OUTPUT_TILE_W + 1) // The step (number of stride moving) needed for a row/col, e.g. reading 4x4 in a 6x6 tile takes 2 steps in a row and 2 steps in a col

// #define OUTPUT_SIZE (OUTPUT_TILE_H * OUTPUT_TILE_W)

#define ROUND_PER_CTA 1
#define STEP_PER_ROUND_CTA 7
#define STEP_PER_CTA (ROUND_PER_CTA * STEP_PER_ROUND_CTA) // Total number of 2x2 HW block to be output per CTA
#define OUTPUT_SIZE_HW (STEP_PER_ROUND_CTA * STEP_OUTPUT_TILE_H * STEP_OUTPUT_TILE_W) // Total HW output per round, e.g. 4 * 2 * 2 = 16 
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
                                  int& _s_h, int& _s_w)
{
  int h = 0, w = 0;
  getSharedHW(isTall, h, w);
  _g_h = _g_h_blk + _s_h_coord + h;
  _g_w = _g_w_blk + _s_w_coord + w; // _s_h/w_coord: HW origin of long/tall tile (2x4 or 4x2 by default) in SMem
  _s_h = (_g_h) % STEP_READ_TILE_H;
  _s_w = (_g_w) % STEP_READ_TILE_W; // h/w: offset for each thread
}

__device__ void loadFloat4(const float* src, float* dst, int src_offset, int dst_offset) {
  reinterpret_cast<float4*>(dst + dst_offset)[0] = ((float4*)(src + src_offset))[0];
}

__device__ void loadFloat4(float* src, float* dst, int src_offset, int dst_offset) {
  reinterpret_cast<float4*>(dst + dst_offset)[0] = reinterpret_cast<float4*>(src + src_offset)[0];
}

// *******************************************************************/

__device__ bool isLastLoop(int step) {
  return ((step + 1) == STEP_PER_ROUND_CTA);
}

__device__ bool isLastLoop(int round, int step) {
  return ((round * STEP_PER_ROUND_CTA + step + 1) == STEP_PER_CTA);
}

template<int H, int W>
__device__ bool inGlobalRange(int _g_h, int _g_w) {
  // 1: padding
  return 1 <= _g_h && (_g_h < H + 1) && 1 <= _g_w && (_g_w < W + 1);
}

// SMem coord in a form of circular buffer
template<int IC_stride>
__device__ int getSharedCoordFloat2(int _s_h, int _s_w) {
  return (_s_h * STEP_READ_TILE_W + _s_w) * IC_stride + 
            (threadIdx.x % 16) * 2;
}

// GMem coord
template<int W, int IC, int IC_stride>
__device__ int getGlobalCoordFloat2(int _g_h, int _g_w, int IC_step) {
  return _g_h * IC * W +
          _g_w * IC +
          IC_stride * IC_step +
          (threadIdx.x % 16) * 2;
}

// For less_CTA_together.cuh
template<int W, int OC, int OC_stride>
__device__ int getOutputBaseCoord(int _g_oc_step, int _g_h_blk, int _g_w_blk) {
  return (_g_h_blk + (threadIdx.y * 2 / OUTPUT_TILE_W)) * W * OC + 
      (_g_w_blk + (threadIdx.y * 2 % OUTPUT_TILE_W) + threadIdx.x / OC_STRIDE_SPLIT) * OC + 
      _g_oc_step * OC_stride + 
      threadIdx.x % OC_STRIDE_SPLIT;
}

// For less_CTA_persistent.cuh
template<int W, int OC, int OC_stride>
__device__ int getOutputCoord(int _g_oc_step, int _g_h_blk, int _g_w_blk) {
  return (_g_h_blk + (threadIdx.y % 2)) * W * OC + 
      (_g_w_blk + threadIdx.x / OC_STRIDE_SPLIT) * OC + 
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
                                      int orig_h, int orig_w) {
#pragma unroll
  for (int ry = 0; ry < FILTER_H; ++ry) {
    for (int rx = 0; rx < FILTER_W; ++rx) {
      int w = orig_w + rx + (threadIdx.y % STEP_OUTPUT_TILE_W);
      int h = orig_h + ry + (threadIdx.y / STEP_OUTPUT_TILE_W);
      int input_idx = threadIdx.x + IC_stride * ((w % STEP_READ_TILE_W) + 
                                                  (h % STEP_READ_TILE_H) * STEP_READ_TILE_W);

      DepthwiseConv2dOutput_0_local[0] += (
          Conv2dFilter_1_shared[input_idx] * filter[ry * FILTER_W + rx]);

      // if (blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
      //    printf("intermediate: %f, w: %d, h: %d, input_idx: %d, _s_orig_h: %d, _s_orig_w: %d, input: %f, filter: %f\n", DepthwiseConv2dOutput_0_local[0], w, h, input_idx, _s_orig_h, _s_orig_w, Conv2dFilter_1_shared[input_idx], filter[ry * FILTER_W + rx]);
    }
  }
  // if (blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
  //     printf("******\n");
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
                            int _g_h_blk, int _g_w_blk,
                            int& _g_coord, int& _s_coord,
                            int _s_h_coord, int _s_w_coord,
                            int IC_step)
{
  int _g_h, _g_w, _s_h, _s_w;

  // Get global and shared HW coord
  getGlobalSharedHW(isTall, _g_h_blk, _g_w_blk, _s_h_coord, _s_w_coord, _g_h, _g_w, _s_h, _s_w);
  // Get global and shared coords
  _g_coord = getGlobalCoordFloat2<W, IC, IC_stride>(_g_h, _g_w, IC_step);
  _s_coord = getSharedCoordFloat2<IC_stride>(_s_h, _s_w);
  // if (blockIdx.y == 0 && blockIdx.x == 0 && _s_coord == 128)
  //     printf("_g_coord: %d, _s_coord: %d, _g_h: %d, _g_w: %d, _s_h: %d, _s_w: %d, global_h: %d, global_w: %d, _s_h_coord: %d, _s_w_coord: %d, thy: %d, thx: %d\n", _g_coord, _s_coord, _g_h, _g_w, _s_h, _s_w, _g_h_blk, _g_w_blk, _s_h_coord, _s_w_coord, threadIdx.y, threadIdx.x);

  // if (blockIdx.y == 0 && blockIdx.x == 13 && threadIdx.y == 0 && threadIdx.x == 0 && IC_step == 0)
  //   printf("isTall: %d, _g_coord: %d, _s_coord: %d, _g_h: %d, _g_w: %d, _s_h: %d, _s_w: %d, global_h: %d, global_w: %d, _s_h_coord: %d, _s_w_coord: %d\n", isTall, _g_coord, _s_coord, _g_h, _g_w, _s_h, _s_w, _g_h_blk, _g_w_blk, _s_h_coord, _s_w_coord);

  int offset = toShared ? _s_coord : 0;
  // Load from GMem to SMem
  loadGlobalWithBoundCheck<H, W>(src, dst + offset, 
                                _g_coord - (W + 1) * IC, /* pad = 1 in all sides, might be changed later */
                                _g_h, _g_w);
}

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

template<int H, int W, int IC, int IC_stride>
__device__ void prefetchInputData(const float* src, float* _s_dst, float* _r_dst,
                                  int _g_h_blk, int _g_w_blk,
                                  int& _g_coord, int& _s_coord,
                                  int IC_step, bool isTall)
{
  // if (blockIdx.y == 0 && blockIdx.x == 13 && threadIdx.y == 0 && threadIdx.x == 0 && IC_step == 0) {
  //   printf("Calling loadWrapper from prefetch.\n");
  // }

  // G->S
  int _s_h_coord = 0;
  int _s_w_coord = 0;
  loadWrapper<H, W, IC, IC_stride>(src, _s_dst,
                                  isTall, true,
                                  _g_h_blk, _g_w_blk,
                                  _g_coord, _s_coord, _s_h_coord, _s_w_coord, IC_step);

  // if (blockIdx.y == 0 && blockIdx.x == 0 && _s_coord == 128)
  //     printf("*******************\n");

  // G->R
  _s_h_coord = 0;
  _s_w_coord = BUFFER_STRIDE;
  loadWrapper<H, W, IC, IC_stride>(src, _r_dst,
                                  isTall, false,
                                  _g_h_blk, _g_w_blk,
                                  _g_coord, _s_coord, _s_h_coord, _s_w_coord, IC_step);

  // if (blockIdx.y == 0 && blockIdx.x == 0 && _s_coord == 128)
  //     printf("^^^^^^^^^^^^^^^^^^\n");
}

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

0-->-->-->-->|
             |
|<--<--<--<--v
|
v-->-->-->-->|
             |
x<--<--<--<--v
*********************/

// Return the shared origin and tile HW coordinates given a loop step in shared space filling, e.g. (OUTPUT_TILE_H*OUTPUT_TILE_W)
__device__ void spaceFillingShared(int loop, bool& isTall,
                                  int& _s_orig_h, int& _s_orig_w,
                                  int& _s_h_coord, int& _s_w_coord) {
  int step_h = loop / STEP_W, step_w = loop % STEP_W;

  // The origin point (ref (0,0)) of the intermediate data tile to be written
  _s_orig_h = step_h * BUFFER_STRIDE;
  _s_orig_w = (step_h % 2) ? (BUFFER_STRIDE * (STEP_W - 1 - step_w)) : (BUFFER_STRIDE * step_w); // Even rows in increase order, odd rows in decrease order.

  // If the input data tile to be read is tall or long
  isTall = (step_w != STEP_W - 1);

  // The origin point (ref to **(_s_orig_h, _s_orig_w)**) of the input data tile to be read followed right after the 
  _s_h_coord = _s_orig_h + (!isTall) * 2 * BUFFER_STRIDE;
  _s_w_coord = _s_orig_w + isTall * (2 - 3 * (step_h % 2)) * BUFFER_STRIDE;
}

// Return the global start HW coordinates given a loop step in global space filling, e.g. (H*W)
template<int total_step_num_h, int total_step_num_w>
__device__ void spaceFillingGlobal(int loop, bool& isTall, int& start_h, int& start_w) {
  int step_h = loop / total_step_num_w, step_w = loop % total_step_num_w;

  // The origin point (ref (0,0)) of the intermediate data tile to be written
  start_h = step_h * BUFFER_STRIDE;
  start_w = (step_h % 2) ? (BUFFER_STRIDE * (total_step_num_w - 1 - step_w)) : (BUFFER_STRIDE * step_w); // Even rows in increase order, odd rows in decrease order.

  // If the input data tile to be read is tall or long
  // isTall is determined by GLOBAL SPACE FILLING
  isTall = (step_w != total_step_num_w - 1);
}

template<int total_step_num_h, int total_step_num_w>
__device__ void spaceFillingSharedPersistent(int loop, bool isTall,
                                            int& _s_h_coord, int& _s_w_coord) {
  int global_row = loop / total_step_num_w;

  // The origin point (ref to **(0,0)**) of the NEXT input data tile to be READ right after this compute
  _s_h_coord = (!isTall) * 2 * BUFFER_STRIDE;
  _s_w_coord = isTall * (2 - 3 * (global_row % 2)) * BUFFER_STRIDE;
}

template<int IC_stride, int OC, int NUM_THX_PER_SEG>
__device__ void loadBlockGlobalToRegister(const float* src, float* dst, 
                                          int src_offset, int dst_offset) {
  int stride = BLOCK_DIM_X / NUM_THX_PER_SEG * BLOCK_DIM_Y * OC;
  int steps = IC_stride / (BLOCK_DIM_X / NUM_THX_PER_SEG * BLOCK_DIM_Y);

#pragma unroll
  for (int num_steps = 0, offset = src_offset; num_steps < steps; num_steps++, offset += stride) {
    loadFloat4(src, dst, offset, 4 * num_steps);
  }
}

template<int IC_stride, int OC_stride, int NUM_THX_PER_SEG>
__device__ void loadBlockRegisterToShared(float* src, float* dst, 
                                          int src_offset, int dst_offset) {
  int stride = BLOCK_DIM_X / NUM_THX_PER_SEG * BLOCK_DIM_Y * OC_stride;
  int steps = IC_stride / (BLOCK_DIM_X / NUM_THX_PER_SEG * BLOCK_DIM_Y);

#pragma unroll
  for (int num_steps = 0, offset = dst_offset; num_steps < steps; num_steps++, offset += stride) {
    loadFloat4(src, dst, 4 * num_steps, offset);
  }
}