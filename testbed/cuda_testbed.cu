#include <iostream>
#include <string>
#include "cnpy.h"
// #include "more_reuse.cuh"
// #include "half_param_165.cuh"
// #include "general_more_reuse.cuh"
#include "less_CTA.cuh"

#define IC_stride 32
#define OC_stride 32

// #define H 112
// #define W 112
// #define IC 32
// #define OC 32
// #define C 32

#define H 56
#define W 56
#define IC 128
#define OC 128
#define C 128

// #define H 28
// #define W 28
// #define IC 256
// #define OC 256
// #define C 256

// #define H 14
// #define W 14
// #define IC 512
// #define OC 512
// #define C 512

using namespace std;

int main(int argc, char const *argv[])
{
	int N = 1, output_tile_H = 4, output_tile_W = 4;

	// Block and grid size
	int threadx_num = 32;
	dim3 block(threadx_num, 4, 1);

	// 1D grid
	// int block_x = (int)(H / output_tile_H) * (int)(W / output_tile_W), block_y = 1;
	// 2D grid
	int block_x = (int)(H / output_tile_H), block_y = (int)(W / output_tile_W);

	dim3 grid(block_x, block_y, 1);
	printf("block x: %d, block_y: %d\n", block_x, block_y);

	// Shared memory size
	size_t inter_size = output_tile_H * output_tile_W * OC_stride * sizeof(float);
	size_t filter_1_size = IC_stride * OC_stride * sizeof(float);
	size_t shared_size = inter_size + filter_1_size;

	// Sizes
	size_t input_shape = N * H * W * IC;
	size_t filter_d_shape = 3 * 3 * IC * 1;
	size_t filter_1_shape = 1 * 1 * IC * OC;
	size_t output_shape = N * H * W * OC;

	// Filenames
	string folder_name = "../npy/depth_conv_" + to_string(N) + "_" + to_string(H) + "_" + to_string(W) + "_" + to_string(IC) + "_" + to_string(OC) + "_" + to_string(3) + "/";
	string input_name = folder_name + "input.npy";
	string filter_d_name = folder_name + "filter_d.npy";
	string filter_1_name = folder_name + "filter_1.npy";
	string output_name = folder_name + "output.npy";

	// Definitions of GPU arrays
	float *input, *filter_d, *filter_1, *output;
	cudaMalloc((void**)&input, input_shape * sizeof(float));
	cudaMalloc((void**)&filter_d, filter_d_shape * sizeof(float));
	cudaMalloc((void**)&filter_1, filter_1_shape * sizeof(float));
	cudaMalloc((void**)&output, output_shape * sizeof(float));

	// Load data and copy to GPU arrays
	float *tmp;

    cnpy::NpyArray input_npy = cnpy::npy_load(input_name);
    tmp = input_npy.data<float>();
    cudaMemcpy(input, tmp, input_shape * sizeof(float), cudaMemcpyHostToDevice);

    cnpy::NpyArray filter_d_npy = cnpy::npy_load(filter_d_name);
    tmp = filter_d_npy.data<float>();
    cudaMemcpy(filter_d, tmp, filter_d_shape * sizeof(float), cudaMemcpyHostToDevice);

    cnpy::NpyArray filter_1_npy = cnpy::npy_load(filter_1_name);
    tmp = filter_1_npy.data<float>();
    cudaMemcpy(filter_1, tmp, filter_1_shape * sizeof(float), cudaMemcpyHostToDevice);

    // Execute kernel
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;
	int repeatition = 1000;

    for (int i = 0; i < repeatition; i++) {
    	cudaMemset(output, 0, output_shape * sizeof(float));
    	float tmp_t = 0.0;
    	cudaEventRecord(start);

    	// more_reuse.cuh and previous
	    // DepthConvFused_2_kernel0<<<grid, block>>>(input, filter_d, filter_1, output);

	    // // half_param_165.cuh
	    // DepthConvFused_2_kernel0<<<grid, block, shared_size>>>(
	    // 	input,
	    // 	filter_d, filter_1,
	    // 	output,
	    // 	H, W, C, C_stride,
	    // 	output_tile_H, output_tile_W
	    // );

	    // general_more_reuse.cuh
	    // DepthConvFused_2_kernel0<<<grid, block, shared_size>>>(
	    // 	input, 
	    // 	filter_d, filter_1,
	    // 	output,
	    // 	H, W, C, C_stride
	    // );

	    // less_CTA.cuh
	    DepthConvFused_2_kernel0 <H, W, IC, OC, 
	    							IC_stride, OC_stride> <<<grid, block, shared_size>>> (
	    	input,
	    	filter_d, filter_1,
	    	output
	    );


	    cudaEventRecord(stop);

	    cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmp_t, start, stop);
		ms += tmp_t / repeatition;
    }
    
	printf("Fusion running time is %f us.\n", ms * 1000);

    // Verification
    // Something wrong with float, use double and convert back
    cnpy::NpyArray output_npy = cnpy::npy_load(output_name);
    double *tmp2 = output_npy.data<double>();

    // Copy result back to CPU for comparison
    float *result;
    result = (float*)malloc(output_shape * sizeof(float));
    cudaMemcpy(result, output, output_shape * sizeof(float), cudaMemcpyDeviceToHost);
    int count = 0;
    for(int i = 0; i < output_shape; i++) {
    	// printf("%d, %f, %lf\n", i, result[i], tmp2[i]);
    	// assert(abs(result[i] - (float)tmp2[i]) < 1e-4);
    	if (abs(result[i] - (float)tmp2[i]) > 1e-3) {
    		printf("%d, %f, %lf\n", i, result[i], tmp2[i]);
    		count++;
    	}
    }
    printf("Wrong count: %d\n", count);

    cudaFree(input);
    cudaFree(filter_d);
    cudaFree(filter_1);
    cudaFree(output);

	return 0;
}