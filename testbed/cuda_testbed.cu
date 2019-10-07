#include <iostream>
#include <string>
#include "cnpy.h"
#include "general_more_reuse.cuh"

using namespace std;

#define BLOCK_Y_SIZE 4 // Fix this value in a foreseeable future

int main(int argc, char const *argv[])
{
	// int N = 1, output_tile_H = 2, output_tile_W = 2;
	int N = 1, output_tile_H = 4, output_tile_W = 4;

	// int H = 112, W = 112, C = 32;
	int H = 56, W = 56, C = 128;
	// int H = 28, W = 28, C = 256;
	// int H = 14, W = 14, C = 512;

	// Block and grid size
	int threadx_num = 32, C_stride = 32;
	dim3 block(threadx_num, BLOCK_Y_SIZE, 1);

	// 1D grid
	// int block_x = (int)(H / output_tile_H) * (int)(W / output_tile_W), block_y = 1;
	// 2D grid
	int block_x = (int)(H / output_tile_H), block_y = (int)(W / output_tile_W);

	dim3 grid(block_x, block_y, 1);
	printf("block x: %d, block_y: %d\n", block_x, block_y);

	// Shared memory size
	size_t inter_size = output_tile_H * output_tile_W * C_stride * sizeof(float);
	size_t filter_1_size = C_stride * C_stride * sizeof(float);
	size_t shared_size = inter_size + filter_1_size;

	// Sizes
	size_t input_shape = N * H * W * C;
	size_t filter_d_shape = 3 * 3 * C * 1;
	size_t filter_1_shape = 1 * 1 * C * C;
	size_t output_shape = N * H * W * C;

	// Filenames
	string input_name = "../npy/depth_conv_input_" + to_string(N) + "_" + to_string(H) + "_" + to_string(W) + "_" + to_string(C) + ".npy";
	string filter_d_name = "../npy/depth_conv_filter_d_3_3_" + to_string(C) + "_1.npy";
	string filter_1_name = "../npy/depth_conv_filter_1_1_1_" + to_string(C) + "_" + to_string(C) + ".npy";
	string output_name = "../npy/depth_conv_output_" + to_string(N) + "_" + to_string(H) + "_" + to_string(W) + "_" + to_string(C) + ".npy";

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

	    // DepthConvFused_2_kernel0<<<grid, block>>>(input, filter_d, filter_1, output);
	    DepthConvFused_2_kernel0<<<grid, block, shared_size>>>(
	    	input, 
	    	filter_d, filter_1, 
	    	output, 
	    	H, W, C, C_stride,
	    	output_tile_H, output_tile_W
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