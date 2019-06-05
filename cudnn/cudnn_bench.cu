// Originally from: https://gist.github.com/goldsborough/865e6717e64fbae75cdaf6c9914a130d

#include <iostream>
#include <string>
#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include "cnpy.h"

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void cudnnCall(cudnnHandle_t cudnn_handle,
               cudnnConvolutionDescriptor_t convolution_descriptor,
               cudnnTensorDescriptor_t input_descriptor,
               cudnnFilterDescriptor_t kernel_descriptor,
               cudnnTensorDescriptor_t output_descriptor,
               float* d_input, float* d_kernel, float* d_output,
               int input_height, int input_width, int input_channel,
               int kernel_height, int kernel_width, 
               int kernel_in_channel, int kernel_out_channel_or_multiplier,
               int output_height, int output_width, int output_channel,
               bool depthwise) {
    int group_count = depthwise ? input_channel : 1;
    // std::cout << "Depthwise? " << depthwise << std::endl;

    // std::cout << input_height << ", " << input_width << ", " << input_channel << std::endl;
    // std::cout << kernel_height << ", " << kernel_width << ", " << kernel_in_channel << ", " << kernel_out_channel_or_multiplier << std::endl;
    // std::cout << output_height << ", " << output_width << ", " << output_channel << std::endl;

    // conv
    checkCUDNN(cudnnSetConvolutionGroupCount(/*conv_descriptor*/convolution_descriptor,  
                                             /*group_count*/group_count));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                               /*pad_height=*/kernel_height == 1 ? 0 : 1,
                                               /*pad_width=*/kernel_width == 1 ? 0 : 1,
                                               /*vertical_stride=*/1,
                                               /*horizontal_stride=*/1,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT));

    // input
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/input_channel,
                                          /*image_height=*/input_height,
                                          /*image_width=*/input_width));

    // filter
    // the filter npy has to be restored as OIHW (as it is in NCHW computation)
    // setting format to NHWC results in 0-byte workspace
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/output_channel,
                                          /*in_channels=*/(int)(input_channel / group_count),
                                          /*kernel_d_height=*/kernel_height,
                                          /*kernel_d_width=*/kernel_width));

    // get output dim
    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     &batch_size,
                                                     &channels,
                                                     &height,
                                                     &width));
    // std::cout << batch_size << ", " << channels << ", " << height << ", " << width << std::endl;
    // assert(batch_size == 1 && channels == output_channel && height == output_height && width == output_width);

    // output
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/output_channel,
                                          /*image_height=*/output_height,
                                          /*image_width=*/output_width));

    // find algorithm
    cudnnConvolutionFwdAlgo_t convolution_algorithm = (cudnnConvolutionFwdAlgo_t)0; // Use default
    // checkCUDNN(
    //     cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
    //                                         input_descriptor,
    //                                         kernel_descriptor,
    //                                         convolution_descriptor,
    //                                         output_descriptor,
    //                                         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //                                         /*memoryLimitInBytes=*/0,
    //                                         &convolution_algorithm));

    // Get workspace
    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));
    // // sometimes 0 but can run normally
    // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
    //           << std::endl;
    // assert(workspace_bytes > 0);

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    // do the convolution
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnn_handle,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       d_output));

    cudaFree(d_workspace);
}

int main(int argc, const char* argv[]) {
  bool first = true, second = true;
  if (argc > 2) {
    first = (1 == std::atoi(argv[1])), second = (1 == std::stoi(argv[2]));
    std::cout << first << ", " << second << std::endl;
  }
  assert(first || second); // should run at least one

  // gpu_id
  int gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;
  cudaSetDevice(gpu_id);

  int input_height = 56, input_width = 56, input_channel = 128;
  int kernel_d_height = 3, kernel_d_width = 3, kernel_d_in_channel = 128, kernel_d_out_multiplier = 1;
  int inter_height = 56, inter_width = 56, inter_channel = 128;
  int kernel_1_height = 1, kernel_1_width = 1, kernel_1_in_channel = 128, kernel_1_out_channel = 128;
  int output_height = 56, output_width = 56, output_channel = 128;

  // filenames
  std::string input_name = "../npy/depth_input_1_" + std::to_string(input_height) + "_" + std::to_string(input_width) + "_" + std::to_string(input_channel) + ".npy";
  std::string kernel_d_name = "../npy/depth_weight_" + std::to_string(kernel_d_height) + "_" + std::to_string(kernel_d_width) + "_" + std::to_string(kernel_d_in_channel) + "_" + std::to_string(kernel_d_out_multiplier) + "_NCHW.npy";
  std::string inter_name;
  if (first) {
    inter_name = "../npy/depth_output_1_" + std::to_string(inter_height) + "_" + std::to_string(inter_width) + "_" + std::to_string(inter_channel) + ".npy";
  } else {
    inter_name = "../npy/conv_input_1_" + std::to_string(inter_height) + "_" + std::to_string(inter_width) + "_" + std::to_string(inter_channel) + ".npy";
  }
  std::string kernel_1_name = "../npy/conv_weight_" + std::to_string(kernel_1_height) + "_" + std::to_string(kernel_1_width) + "_" + std::to_string(kernel_1_in_channel) + "_" + std::to_string(kernel_1_out_channel) + "_NCHW.npy";
  std::string output_name = "../npy/conv_output_1_" + std::to_string(output_height) + "_" + std::to_string(output_width) + "_" + std::to_string(output_channel) + ".npy";

  std::cout << input_name << std::endl << kernel_d_name << std::endl << inter_name << std::endl << kernel_1_name << std::endl << output_name << std::endl;

  // tensor sizes
  size_t input_shape = 1 * input_height * input_width * input_channel;
  size_t kernel_d_shape = kernel_d_height * kernel_d_width * kernel_d_in_channel * kernel_d_out_multiplier;
  size_t inter_shape = 1 * inter_height * inter_width * inter_channel;
  size_t kernel_1_shape = kernel_1_height * kernel_1_width * kernel_1_in_channel * kernel_1_out_channel;
  size_t output_shape = 1 * output_height * output_width * output_channel;

  // gpu pointers
  float* d_input{nullptr};
  float* d_kernel_d{nullptr};
  float* d_inter{nullptr};
  float* d_kernel_1{nullptr};
  float* d_output{nullptr};
  cudaMalloc(&d_input, input_shape * sizeof(float));
  cudaMalloc(&d_kernel_d, kernel_d_shape * sizeof(float));
  cudaMalloc(&d_inter, inter_shape * sizeof(float));
  cudaMalloc(&d_kernel_1, kernel_1_shape * sizeof(float));
  cudaMalloc(&d_output, output_shape * sizeof(float));

  // Load data and copy to GPU arrays
  float *tmp;

  cnpy::NpyArray input_npy = cnpy::npy_load(input_name);
  tmp = input_npy.data<float>();
  cudaMemcpy(d_input, tmp, input_shape * sizeof(float), cudaMemcpyHostToDevice);

  cnpy::NpyArray kernel_d_npy = cnpy::npy_load(kernel_d_name);
  tmp = kernel_d_npy.data<float>();
  cudaMemcpy(d_kernel_d, tmp, kernel_d_shape * sizeof(float), cudaMemcpyHostToDevice);

  cnpy::NpyArray inter_npy = cnpy::npy_load(inter_name);
  if (!first) {
    tmp = inter_npy.data<float>();
    cudaMemcpy(d_inter, tmp, inter_shape * sizeof(float), cudaMemcpyHostToDevice);
    // for(int i = 0; i < 100; i++) {
    //   printf("%d, %f\n", i, tmp[i]);
    // }
  }

  if (second) {
    cnpy::NpyArray kernel_1_npy = cnpy::npy_load(kernel_1_name);
    tmp = kernel_1_npy.data<float>();
    cudaMemcpy(d_kernel_1, tmp, kernel_1_shape * sizeof(float), cudaMemcpyHostToDevice);
    // for(int i = 0; i < 100; i++) {
    //   printf("%d, %f\n", i, tmp[i]);
    // }
  }

  for (int i = 0; i < 5; i++) {
    // create handles
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    // create descriptors
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));

    if (first)
      cudnnCall(cudnn_handle,
                convolution_descriptor,
                input_descriptor, kernel_descriptor, output_descriptor,
                d_input, d_kernel_d, d_inter,
                input_height, input_width, input_channel, 
                kernel_d_height, kernel_d_width, kernel_d_in_channel, kernel_d_out_multiplier,
                inter_height, inter_width, inter_channel,
                true);

    if (second)
      cudnnCall(cudnn_handle,
                convolution_descriptor,
                input_descriptor, kernel_descriptor, output_descriptor,
                d_inter, d_kernel_1, d_output,
                inter_height, inter_width, inter_channel,
                kernel_1_height, kernel_1_width, kernel_1_in_channel, kernel_1_out_channel,
                output_height, output_width, output_channel,
                false);

    // destroy descriptors
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn_handle);
  }

  // Verification
  int count;
  float *inter_result, *output_result;
  inter_result = (float*)malloc(inter_shape * sizeof(float));
  output_result = (float*)malloc(output_shape * sizeof(float));

  // inter
  if (first) {
    float *tmp2 = inter_npy.data<float>();
    cudaMemcpy(inter_result, d_inter, inter_shape * sizeof(float), cudaMemcpyDeviceToHost);

    count = 0;
    for(int i = 0; i < inter_shape; i++) {
      // printf("%d, %f, %lf\n", i, inter_result[i], tmp2[i]);
      // assert(abs(inter_result[i] - (float)tmp2[i]) < 1e-4);
      if (abs(inter_result[i] - tmp2[i]) > 1e-4)
        count++;
    }
    printf("Inter wrong count: %d\n", count);
  }

  // output
  if (second) {
    cnpy::NpyArray output_npy = cnpy::npy_load(output_name);
    float *tmp3 = output_npy.data<float>();
    cudaMemcpy(output_result, d_output, output_shape * sizeof(float), cudaMemcpyDeviceToHost);

    count = 0;
    for(int i = 0; i < output_shape; i++) {
      // printf("%d, %f, %lf\n", i, output_result[i], tmp3[i]);
      // assert(abs(output_result[i] - (float)tmp3[i]) < 1e-4);
      if (abs(output_result[i] - tmp3[i]) > 2e-4) // A few nums have bigger errors
        count++;
    }
    printf("Output wrong count: %d\n", count);
  }

  free(inter_result);
  free(output_result);
  cudaFree(d_input);
  cudaFree(d_kernel_d);
  cudaFree(d_input);
  cudaFree(d_kernel_1);
  cudaFree(d_output);
}