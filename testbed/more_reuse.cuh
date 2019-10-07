#define TALL true

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

__device__ void depthwiseConvSingleNum(float* Conv2dFilter_1_shared, float* filter, float* DepthwiseConv2dOutput_0_local, int start_w, int start_h) {
#pragma unroll
  for (int ry = 0; ry < 3; ++ry) {
    for (int rx = 0; rx < 3; ++rx) {
      int w = start_w + rx + threadIdx.y % 2;
      int h = start_h + ry + threadIdx.y / 2;
      int input_idx = threadIdx.x + 32 * ((w % 4) + (h % 4) * 4);

      // if (blockIdx.x == 0 && thy == 0 && rc_outer_v == 0 && ry == 0 && rx == 0)
      //   printf("depthwise: %f\n", DepthwiseConv2dOutput_0_local[0]);

      DepthwiseConv2dOutput_0_local[0] += (
          Conv2dFilter_1_shared[input_idx] * filter[ry * 3 + rx]);

      // if (blockIdx.x == 0 && thy == 1 && thx == 0 && rc_outer_v == 0) {
      //   printf("rx: %d, ry: %d, w: %d, h: %d, input_idx: %d, tmp result: %f\n", rx, ry, w, h, input_idx, DepthwiseConv2dOutput_0_local[0]);
      //   printf("input: %f, filter: %f\n", Conv2dFilter_1_shared[input_idx], filter[ry * 3 + rx]);
      //   printf("ref filter: %f\n", DepthwiseFilter_1[(ry * 3 + rx) * 128]);
      // }
    }
  }
}

extern "C" __global__ void DepthConvFused_2_kernel0(const float* Input, const float* DepthwiseFilter_1, const float* Conv2dFilter_1, float* Conv2dOutput_0) {

  float Conv2dOutput_0_local[16] = { 0.0f };
  float DepthwiseConv2dOutput_0_local[1];
  
  __shared__ float intermediate[512];
  __shared__ float Conv2dFilter_1_shared[1024];

  //
  float filter[9];
  float buffer[8];
  int thx = threadIdx.x, thy = threadIdx.y, blx = blockIdx.x;
  int W = 56, C = 128, C_stride = 32;
  int _g_h_blk = (blx / 14) * 4;// 14: 14 stride-of-4 across H and W's 56
  int _g_w_blk = (blx % 14) * 4;
  int h = 0, w = 0;
  //

  // ((float4*)(Conv2dOutput_0_local))[0] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);
  // ((float4*)(Conv2dOutput_0_local))[4] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);
  // ((float4*)(Conv2dOutput_0_local))[8] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);
  // ((float4*)(Conv2dOutput_0_local))[12] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);

  for (int rc_outer_v = 0; rc_outer_v < 4; ++rc_outer_v) {
    int _g_c_step = rc_outer_v;

    ///////////// Preprocessing /////////////
    // Load filter to RMem
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        filter[ry * 3 + rx] = DepthwiseFilter_1[thx + 
                                                (rc_outer_v * 32) + 
                                                (ry * 384) + 
                                                (rx * 128)];
      }
    }

    // Clear SMem
    // ((float4*)(intermediate + 4 * (thx + 32 * thy)))[0] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);
    // ((float4*)(Conv2dFilter_1_shared + 4 * (thx + 32 * thy)))[0] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);
    // ((float4*)(Conv2dFilter_1_shared + 4 * (thx + 32 * thy) + 512))[0] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);
    // __syncthreads();

    // Load tile ((2,0), (3,3)) to SMem
    /*********************
    |+++|+++|+++|+++|+++|+++|
    |   |   |   |   |   |   | 5
    |+++|+++|+++|+++|+++|+++|
    |   |   |   |   |   |   | 4
    |+++|+++|+++|+++|+++|+++|
    |   |   | s | s |   |   | 3
    |+++|+++|+++|+++|+++|+++|
    |   |   | s | s |   |   | 2
    |+++|+++|+++|+++|+++|+++|
    |   |   | s | s |   |   | 1
    |+++|+++|+++|+++|+++|+++|
    |   |   | s | s |   |   | 0
    |+++|+++|+++|+++|+++|+++|
      5   4   3   2   1   0
    *********************/
    int _s_h_coord = 0;
    int _s_w_coord = 2;

    // Get HWs
    getSharedHW(true, h, w);
    int _g_h = _g_h_blk + _s_h_coord + h;
    int _g_w = _g_w_blk + _s_w_coord + w;
    int _s_h =          + _s_h_coord + h;
    int _s_w =          + _s_w_coord + w;

    // Get global and shared coords
    int _s_coord = getSharedCoordFloat2(_s_h, _s_w);
    int _g_coord = getGlobalCoordFloat2(_g_h, _g_w, _g_c_step, W, C, C_stride);

    // if (blockIdx.x == 100 && threadIdx.y == 3 && rc_outer_v == 0) {
    //   printf("thx: %d, _g_h: %d, _g_w: %d, _s_h: %d, _s_w: %d, _s_coord: %d, _g_coord: %d\n", thx, _g_h, _g_w, _s_h, _s_w, _s_coord, _g_coord);
    // }

    // Load from GMem to SMem
    ((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = 
      inGlobalRange(_g_h, _g_w) ? 
        ((float2*)(Input + _g_coord - 7296))[0] : 
        make_float2(0.0e+00f, 0.0e+00f);
    __syncthreads();

    ///////////////////////////////////////////
    // Load tile ((0,0), (1,3)) to RMem
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

    _s_h_coord = 0;
    _s_w_coord = 0;

    // Get HWs
    getSharedHW(true, h, w);
    _g_h = _g_h_blk + _s_h_coord + h;
    _g_w = _g_w_blk + _s_w_coord + w;
    _s_h =          + _s_h_coord + h;
    _s_w =          + _s_w_coord + w;

    _g_coord = getGlobalCoordFloat2(_g_h, _g_w, _g_c_step, W, C, C_stride);

    // if (blockIdx.x == 100 && threadIdx.y == 2 && rc_outer_v == 0) {
    //   printf("thx: %d, _g_h: %d, _g_w: %d, _s_h: %d, _s_w: %d, _s_coord: %d, _g_coord: %d\n", thx, _g_h, _g_w, _s_h, _s_w, _s_coord, _g_coord);
    // }

    ((float2*)(buffer))[0] = 
      inGlobalRange(_g_h, _g_w) ? 
        ((float2*)(Input + _g_coord - 7296))[0] : 
        make_float2(0.0e+00f, 0.0e+00f);


    //////////////////////////// Loop 1 ////////////////////////////
    {
      DepthwiseConv2dOutput_0_local[0] = 0.0e+00f;

      // Load from RMem to SMem
      /*********************
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 5
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 4
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 3
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 2
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 1
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 0
      |+++|+++|+++|+++|+++|+++|
        5   4   3   2   1   0
      *********************/
      _s_coord = getSharedCoordFloat2(_s_h, _s_w);
      ((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = ((float2*)(buffer))[0];
      __syncthreads();

      // if (blockIdx.x == 0 && rc_outer_v == 0 && thy == 0 && thx < 16) {
      //   printf("first slice shared, index: %d, input: %f\n", thx * 32, Conv2dFilter_1_shared[thx * 32]);
      //   printf("first slice global, index: %d, input: %f\n", thx * 128, Input[thx * 128]);
      // }

      // if (blockIdx.x == 0 && thy == 0 && rc_outer_v == 0) {
      //   printf("thx: %d, _g_h: %d, _g_w: %d, _s_h: %d, _s_w: %d, _s_coord: %d, _g_coord: %d\n", thx, _g_h, _g_w, _s_h, _s_w, _s_coord, _g_coord);
      // }

      // Depthwise
      depthwiseConvSingleNum(Conv2dFilter_1_shared, filter, DepthwiseConv2dOutput_0_local, 0, 0);
      // Write from RMem to SMem
      intermediate[thx + ((thy % 2) + thy / 2 * 4) * 32] = DepthwiseConv2dOutput_0_local[0];

      // Load tile ((0,4), (3,5)) to RMem
      /*********************
      |+++|+++|+++|+++|+++|+++|
      |   |   | r | r | r | r | 5
      |+++|+++|+++|+++|+++|+++|
      |   |   | r | r | r | r | 4
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 3
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 2
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 1
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 0
      |+++|+++|+++|+++|+++|+++|
        5   4   3   2   1   0
      *********************/
      _s_h_coord = 4;
      _s_w_coord = 0;

      // Get HWs
      getSharedHW(false, h, w);
      _g_h = _g_h_blk + _s_h_coord + h;
      _g_w = _g_w_blk + _s_w_coord + w;
      _s_h =          + _s_h_coord + h;
      _s_w =          + _s_w_coord + w;

      _g_coord = getGlobalCoordFloat2(_g_h, _g_w, _g_c_step, W, C, C_stride);

      // if (blockIdx.x == 100 && threadIdx.y == 1 && rc_outer_v == 0) {
      //   printf("thx: %d, _g_h: %d, _g_w: %d, _s_h: %d, _s_w: %d, _s_coord: %d, _g_coord: %d\n", thx, _g_h, _g_w, _s_h, _s_w, _s_coord, _g_coord);
      // }

      ((float2*)(buffer))[0] = 
        inGlobalRange(_g_h, _g_w) ? 
          ((float2*)(Input + _g_coord - 7296))[0] : 
          make_float2(0.0e+00f, 0.0e+00f);
    }

    //////////////////////////// Loop 2 ////////////////////////////
    {
      DepthwiseConv2dOutput_0_local[0] = 0.0e+00f;

      // Load tile ((0,4), (3,5)) to SMem
      /*********************
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 5
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 4
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 3
      |+++|+++|+++|+++|+++|+++|
      |   |   | s | s | s | s | 2
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 1
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 0
      |+++|+++|+++|+++|+++|+++|
        5   4   3   2   1   0
      *********************/
      _s_coord = getSharedCoordFloat2(_s_h, _s_w);
      ((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = ((float2*)(buffer))[0];
      __syncthreads();

      // Depthwise
      depthwiseConvSingleNum(Conv2dFilter_1_shared, filter, DepthwiseConv2dOutput_0_local, 0, 2);
      // Write from RMem to SMem
      intermediate[thx + ((thy % 2) + thy / 2 * 4) * 32 + 256] = DepthwiseConv2dOutput_0_local[0];
      // if (blockIdx.x == 100 && thx == 0 && thx == 0 && rc_outer_v == 0) {
      //   printf("shared_idx: %d\n", thx + ((thy % 2) + thy / 2 * 4) * 32 + 256);
      // }

      // Load tile ((4,2), (5,5)) to RMem
      /*********************
      |+++|+++|+++|+++|+++|+++|
      | r | r | s | s | s | s | 5
      |+++|+++|+++|+++|+++|+++|
      | r | r | s | s | s | s | 4
      |+++|+++|+++|+++|+++|+++|
      | r | r | s | s | s | s | 3
      |+++|+++|+++|+++|+++|+++|
      | r | r | s | s | s | s | 2
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 1
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 0
      |+++|+++|+++|+++|+++|+++|
        5   4   3   2   1   0
      *********************/
      _s_h_coord = 2;
      _s_w_coord = 4;

      // Get HWs
      getSharedHW(true, h, w);
      _g_h = _g_h_blk + _s_h_coord + h;
      _g_w = _g_w_blk + _s_w_coord + w;
      _s_h =          + _s_h_coord + h;
      _s_w =          + _s_w_coord + w;

      _g_coord = getGlobalCoordFloat2(_g_h, _g_w, _g_c_step, W, C, C_stride);

      ((float2*)(buffer))[0] = 
        inGlobalRange(_g_h, _g_w) ? 
          ((float2*)(Input + _g_coord - 7296))[0] : 
          make_float2(0.0e+00f, 0.0e+00f);
    }

    //////////////////////////// Loop 3 ////////////////////////////
    {
      DepthwiseConv2dOutput_0_local[0] = 0.0e+00f;

      // Load tile ((4,2), (5,5)) to SMem
      /*********************
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 5
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 4
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 3
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 2
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 1
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 0
      |+++|+++|+++|+++|+++|+++|
        5   4   3   2   1   0
      *********************/

      _s_coord = getSharedCoordFloat2(_s_h, _s_w);
      // if (blockIdx.x == 0 && threadIdx.y == 0 && rc_outer_v == 0) {
      //   printf("thx: %d, _g_h: %d, _g_w: %d, _s_h: %d, _s_w: %d, _s_coord: %d, _g_coord: %d\n", thx, _g_h, _g_w, _s_h, _s_w, _s_coord, _g_coord);
      // }

      ((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = ((float2*)(buffer))[0];
      __syncthreads();

      // // ///////////////////////// Correctness of reading global input
      // if (blockIdx.x == 0 && rc_outer_v == 0 && thy == 0 && thx < 16) {
      //   printf("first slice shared, thx: %d, index: %d, input: %f\n", thx, thx * 32, Conv2dFilter_1_shared[thx * 32]);
      //   printf("first slice global, thx: %d, index: %d, input: %f\n", thx, 14336 - 7296 + thx / 4 * 7168 + 256 + (thx % 4) * 128, Input[14336 - 7296 + thx / 4 * 7168 + 256 + (thx % 4) * 128]);
      // }
      // // /////////////////////////

      // Depthwise
      depthwiseConvSingleNum(Conv2dFilter_1_shared, filter, DepthwiseConv2dOutput_0_local, 2, 2);
      // Write from RMem to SMem
      intermediate[thx + ((thy % 2) + (thy / 2) * 4 + 2) * 32 + 256] = DepthwiseConv2dOutput_0_local[0];
      // if (blockIdx.x == 0 && thy == 0 && thx == 0 && rc_outer_v == 0) {
      //   float ref1 = Input[7296] * DepthwiseFilter_1[0] + Input[7424] * DepthwiseFilter_1[128] + Input[7552] * DepthwiseFilter_1[256] +
      //               Input[14464] * DepthwiseFilter_1[384] + Input[14592] * DepthwiseFilter_1[512] + Input[14720] * DepthwiseFilter_1[640] + 
      //               Input[21632] * DepthwiseFilter_1[768] + Input[21760] * DepthwiseFilter_1[896] + 
      //               Input[21888] * DepthwiseFilter_1[1024];

      //   // printf("ref1: %f\n", ref1);

      //   // printf("^^^^^^^^global filter\n");

      //   // float ref1 = Input[7296] * DepthwiseFilter_1[0 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[7296], DepthwiseFilter_1[0 * 128], ref1);
      //   // ref1 += Input[7424] * DepthwiseFilter_1[1 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[7424], DepthwiseFilter_1[1 * 128], ref1);
      //   // ref1 += Input[7552] * DepthwiseFilter_1[2 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[7552], DepthwiseFilter_1[2 * 128], ref1);
      //   // ref1 += Input[14976] * DepthwiseFilter_1[3 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[14976], DepthwiseFilter_1[3 * 128], ref1);
      //   // ref1 += Input[15104] * DepthwiseFilter_1[4 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[15104], DepthwiseFilter_1[4 * 128], ref1);
      //   // ref1 += Input[15232] * DepthwiseFilter_1[5 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[15232], DepthwiseFilter_1[5 * 128], ref1);
      //   // ref1 += Input[22656] * DepthwiseFilter_1[6 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[22656], DepthwiseFilter_1[6 * 128], ref1);
      //   // ref1 += Input[22784] * DepthwiseFilter_1[7 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[22784], DepthwiseFilter_1[7 * 128], ref1);
      //   // ref1 += Input[22912] * DepthwiseFilter_1[8 * 128];
      //   // printf("input: %f, filter: %f, ref1: %f\n", Input[22912], DepthwiseFilter_1[8 * 128], ref1);

      //   printf("^^^^^^^^local filter\n");

      //   float ref2 = Input[7296] * filter[0];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[7296], filter[0], ref2);
      //   ref2 += Input[7424] * filter[1];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[7424], filter[1], ref2);
      //   ref2 += Input[7552] * filter[2];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[7552], filter[2], ref2);
      //   ref2 += Input[14464] * filter[3];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[14464], filter[3], ref2);
      //   ref2 += Input[14592] * filter[4];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[14592], filter[4], ref2);
      //   ref2 += Input[14720] * filter[5];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[14720], filter[5], ref2);
      //   ref2 += Input[21632] * filter[6];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[21632], filter[6], ref2);
      //   ref2 += Input[21760] * filter[7];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[21760], filter[7], ref2);
      //   ref2 += Input[21888] * filter[8];
      //   printf("input: %f, filter: %f, ref2: %f\n", Input[21888], filter[8], ref2);

      //   printf("ref result1: %f, result2: %f\n", ref1, ref2);
      //   printf("tmp, result: %f, inter_idx: %d\n", DepthwiseConv2dOutput_0_local[0], thx + ((thy % 2) + (thy / 2) * 4 + 2) * 32 + 256);
      // }

      // Load tile ((2,0), (5,1)) to SMem
      /*********************
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 5
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 4
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 3
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 2
      |+++|+++|+++|+++|+++|+++|
      | r | r | r | r |   |   | 1
      |+++|+++|+++|+++|+++|+++|
      | r | r | r | r |   |   | 0
      |+++|+++|+++|+++|+++|+++|
        5   4   3   2   1   0
      *********************/
      _s_h_coord = 0;
      _s_w_coord = 2;

      // Get HWs
      getSharedHW(false, h, w);
      _g_h = _g_h_blk + _s_h_coord + h;
      _g_w = _g_w_blk + _s_w_coord + w;
      _s_h =          + _s_h_coord + h;
      _s_w =          + _s_w_coord + w;

      _g_coord = getGlobalCoordFloat2(_g_h, _g_w, _g_c_step, W, C, C_stride);

      ((float2*)(buffer))[0] = 
        inGlobalRange(_g_h, _g_w) ? 
          ((float2*)(Input + _g_coord - 7296))[0] : 
          make_float2(0.0e+00f, 0.0e+00f);
    }

    //////////////////////////// Loop 4 ////////////////////////////
    {
      DepthwiseConv2dOutput_0_local[0] = 0.0e+00f;

      // Load tile ((2,0), (5,1)) to SMem
      /*********************
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 5
      |+++|+++|+++|+++|+++|+++|
      |   |   |   |   |   |   | 4
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 3
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 2
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 1
      |+++|+++|+++|+++|+++|+++|
      | s | s | s | s |   |   | 0
      |+++|+++|+++|+++|+++|+++|
        5   4   3   2   1   0
      *********************/

      _s_coord = getSharedCoordFloat2(_s_h, _s_w);
      ((float2*)(Conv2dFilter_1_shared + _s_coord))[0] = ((float2*)(buffer))[0];
      __syncthreads();
      // if (blockIdx.x == 0 && threadIdx.y == 0 && rc_outer_v == 0) {
      //   printf("thx: %d, _g_h: %d, _g_w: %d, _s_h: %d, _s_w: %d, _s_coord: %d, _g_coord: %d\n", thx, _g_h, _g_w, _s_h, _s_w, _s_coord, _g_coord);
      // }

      // // ///////////////////////// Correctness of reading global input
      // if (blockIdx.x == 0 && rc_outer_v == 0 && thy == 0 && thx < 16) {
      //   printf("first slice shared, thx: %d, index: %d, input: %f\n", thx, thx * 32, Conv2dFilter_1_shared[thx * 32]);
      //   printf("first slice global, thx: %d, index: %d, input: %f\n", thx, -7296 + thx / 4 * 7168 + 256 + (thx % 4) * 128, (-7296 + thx / 4 * 7168 + 256 + (thx % 4) * 128 >= 0) ? Input[-7296 + thx / 4 * 7168 + 256 + (thx % 4) * 128] : 0.0f);
      // }
      // // /////////////////////////

      // Depthwise
      depthwiseConvSingleNum(Conv2dFilter_1_shared, filter, DepthwiseConv2dOutput_0_local, 2, 0);
      // Write from RMem to SMem
      intermediate[thx + ((thy % 2) + (thy / 2) * 4 + 2) * 32] = DepthwiseConv2dOutput_0_local[0];
      // if (blockIdx.x == 100 && thx == 0 && thx == 0 && rc_outer_v == 0) {
      //   printf("shared_idx: %d\n", thx + ((thy % 2) + thy / 2 * 4 + 2) * 32);
      // }
    }

    // if (blockIdx.x == 30 && rc_outer_v == 0) {
    //   // printf("%d, %f\n", thy * 32 + thx, intermediate[thy * 32 + thx]);
    //   // printf("%d, %f\n", 128 + thy * 32 + thx, intermediate[128 + thy * 32 + thx]);
    //   // printf("%d, %f\n", 256 + thy * 32 + thx, intermediate[256 + thy * 32 + thx]);
    //   printf("%d, %f\n", 384 + thy * 32 + thx, intermediate[384 + thy * 32 + thx]);
    // }
    // ((float4*)(buffer))[0] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);
    // ((float4*)(buffer))[4] = make_float4(0.0e+00f, 0.0e+00f, 0.0e+00f, 0.0e+00f);
    __syncthreads();

    // gmem to rmem
    ((float4*)(buffer))[0] = ((float4*)(Conv2dFilter_1 + (thy * 512) + (thx / 8) * 128 + (thx % 8) * 4 + rc_outer_v * 4096 + 0))[0];
    ((float4*)(buffer + 4))[0] = ((float4*)(Conv2dFilter_1 + ((thy * 512) + (thx / 8) * 128 + (thx % 8) * 4 + rc_outer_v * 4096) + 2048))[0];

    // if (rc_outer_v == 0) {
    //   // for (int i = 0; i < 16; i++) {
    //   //   Conv2dOutput_0_local[i] = 0.0f;
    //   // }


    //   ((float4*)(Conv2dOutput_0_local))[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    //   ((float4*)(Conv2dOutput_0_local))[4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    //   ((float4*)(Conv2dOutput_0_local))[8] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    //   ((float4*)(Conv2dOutput_0_local))[12] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // }

    for (int iter = 0; iter < 4; iter++) {
      // rmem to smem
      ((float4*)(Conv2dFilter_1_shared + (thy * 128) + (thx / 8) * 32 + (thx % 8) * 4 + 0))[0] = ((float4*)(buffer))[0];
      ((float4*)(Conv2dFilter_1_shared + (thy * 128) + (thx / 8) * 32 + (thx % 8) * 4 + 512))[0] = ((float4*)(buffer + 4))[0];

      __syncthreads();

      // compute on smem
      for (int j = 0; j < 4; j++) {
        // if (blockIdx.x == 30 && thy == 0 && thx == 0 && rc_outer_v == 0 && iter == 0) {
        //   printf("inter slice: %d, inter idx: %d, filter idx: %d, local output: %f\n", j, 128 * j + (thy / 2) * 64 + (thx / 16) * 32, (thx % 16) + (thy % 2) * 16, Conv2dOutput_0_local[4 * j + iter]);
        // }

      #pragma unroll
        for (int i = 0; i < 32; i++) {
          Conv2dOutput_0_local[4 * j + iter] += intermediate[128 * j + (thy / 2) * 64 + (thx / 16) * 32 + i] * Conv2dFilter_1_shared[(thx % 16) + 32 * i + (thy % 2) * 16];
        }
        // if (blockIdx.x == 0 && thx == 0 && thy == 0 && (4 * j + iter == 4 || 4 * j + iter == 0)) {
        //   printf("idx: %d, tmp: %f\n", 4 * j + iter, Conv2dOutput_0_local[4 * j + iter]);
        // }
      }

      // gmem to rmem
      if (iter < 3) {
        ((float4*)(buffer))[0] = ((float4*)(Conv2dFilter_1 + thy * 512 + (iter+1) * 32 + (thx / 8) * 128 + (thx % 8) * 4 + rc_outer_v * 4096 + 0))[0];
        ((float4*)(buffer + 4))[0] = ((float4*)(Conv2dFilter_1 + thy * 512 + (iter+1) * 32 + (thx / 8) * 128 + (thx % 8) * 4 + rc_outer_v * 4096 + 2048))[0];
      }
    }

    __syncthreads();
  }

  // Epilogue

// #pragma unroll
  for (int iter = 0; iter < 16; iter++) {
    int i = iter / 4, j = iter % 4;
    int idx = (_g_h_blk + i) * W * C + 
              (_g_w_blk + thy / 2 * 2 + thx / 16) * C + 
              j * C_stride + 
              (thy % 2) * 16 + 
              thx % 16;
    Conv2dOutput_0[idx] = Conv2dOutput_0_local[iter];

    // if (idx == 65536) {
    //   printf("thx: %d, thy: %d, blockIdx: %d, iter: %d\n", thx, thy, blockIdx.x, iter);
    // }

    // if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    //   printf("_g_h_blk: %d, _g_w_blk: %d, idx: %d, output: %f\n", _g_h_blk, _g_w_blk, idx, Conv2dOutput_0_local[iter]);
    // }
  }
}

