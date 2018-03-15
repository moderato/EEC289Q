#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <algorithm>


using namespace std;

#define CAFFE_CUDA_NUM_THREADS 512
inline int CAFFE_GET_BLOCKS(const int N) {
   return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
 }


template <typename Dtype>
__global__ void ConvForward(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
		const int c = index % channels; 
		const int pw = (index / channels) % conved_width;
		const int ph = (index / channels / conved_width) % conved_height;
		const int n = index / conved_width / conved_height / channels;
		int hstart = ph * stride_h - pad_h; // input pointer starting point
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h); // boundary
		int wend = min(wstart + kernel_w, width + pad_w); 
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height); // height=output hight
		wend = min(wend, width);

		Dtype aveval = 0;
		int khstart=hend<kernel_h?kernel_h-hend:0;
		int kwstart=wend<kernel_w?kernel_w-wend:0;
		#pragma unroll
		for (int h = hstart; h < hend; ++h) {
		#pragma unroll
			for (int w = wstart; w < wend; ++w) {

				aveval += bottom_data[(h * width + w)*channels+c]*weight[((khstart+h-hstart) * kernel_w + (kwstart+w-wstart))*channels+c]; // (h-hstart)=>0~kernel_h
			}
		}

		top_data[index] = aveval;
	}
}


template <typename Dtype>
__global__ void ConvForwardTile4(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < nthreads){ 

		Dtype out[4]={0}; //local output
		const int c = index % channels; 
		const int pw = (index / channels) * 2 % conved_width;
		const int ph = (index / channels  * 2 / conved_width ) * 2  % conved_height;
		//const int n = index / conved_width / conved_height / channels;


		#pragma unroll
		for(int j=0; j<2; j++)
			#pragma unroll
			for(int i=0; i<2; i++)
		{
			/*int hstart = ph * stride_h - pad_h; // input pointer starting point
			int wstart = pw * stride_w - pad_w;
			int hend = min(hstart + kernel_h, height + pad_h); // boundary
			int wend = min(wstart + kernel_w, width + pad_w); 
			hstart = max(hstart, 0);
			wstart = max(wstart, 0);
			hend = min(hend, height); // height=output hight
			wend = min(wend, width);

			Dtype aveval = 0;
			int khstart=hend<kernel_h?kernel_h-hend:0;
			int kwstart=wend<kernel_w?kernel_w-wend:0;*/
			Dtype aveval = 0;

			const int hstart = (ph + j )* stride_h - pad_h >0? (ph + j )* stride_h - pad_h :0;
			const int wstart = (pw + i) * stride_w - pad_w >0? (pw + i) * stride_w - pad_w :0;
			const int hend = (ph + j )* stride_h - pad_h + kernel_h< height? (ph + j )* stride_h - pad_h + kernel_h : height; 
			const int wend = (pw + i) * stride_w - pad_w + kernel_w< width? (pw + i) * stride_w - pad_w + kernel_w : width; 
			const int khstart=hend<kernel_h?kernel_h-hend:0;
			const int kwstart=wend<kernel_w?kernel_w-wend:0;
			#pragma unroll
			for (int h = hstart; h < hend; ++h) {
			#pragma unroll
				for (int w = wstart; w < wend; ++w) {

					//aveval += bottom_data[(h * width + w)*channels+c]*weight[((khstart+h-hstart) * kernel_w + (kwstart+w-wstart))*channels+c]; // (h-hstart)=>0~kernel_h
					out[j*2+i]+= bottom_data[(h * width + w)*channels+c]*weight[((khstart+h-hstart) * kernel_w + (kwstart+w-wstart))*channels+c];
				}
			}

			//top_data[(((ph+j)*width)+(pw+i))*channels+c] = aveval;
		}
		#pragma unroll
		for(int j=0; j<2; j++)
			#pragma unroll
			for(int i=0; i<2; i++)
			top_data[(((ph+j)*width)+(pw+i))*channels+c] = out[j*2+i];//hard code numbers here will increase speed
	
	}
}


template <typename Dtype>
__global__ void ConvForwardAnd1x1ConvTile4(const int nthreads,
		const Dtype* __restrict__ const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* __restrict__ const top_data,const Dtype* __restrict__ const weight,const Dtype* const weight1x1,const Dtype* __restrict__ const bias,const bool bias_term_) {

	volatile __shared__ float tmp_out[4*512];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < nthreads){ 

		Dtype out[4]={0}; //local output
		const int c = index % channels; 
		const int pw = (index / channels) * 2 % conved_width;
		const int ph = (index / channels  * 2 / conved_width ) * 2  % conved_height;
		//const int n = index / conved_width / conved_height / channels;


		#pragma unroll
		for(int j=0; j<2; j++)
			#pragma unroll
			for(int i=0; i<2; i++)
		{
	
			Dtype aveval = 0;

			const int hstart = (ph + j )* stride_h - pad_h >0? (ph + j )* stride_h - pad_h :0;
			const int wstart = (pw + i) * stride_w - pad_w >0? (pw + i) * stride_w - pad_w :0;
			const int hend = (ph + j )* stride_h - pad_h + kernel_h< height? (ph + j )* stride_h - pad_h + kernel_h : height; 
			const int wend = (pw + i) * stride_w - pad_w + kernel_w< width? (pw + i) * stride_w - pad_w + kernel_w : width; 
			const int khstart=hend<kernel_h?kernel_h-hend:0;
			const int kwstart=wend<kernel_w?kernel_w-wend:0;
			#pragma unroll
			for (int h = hstart; h < hend; ++h) {
			#pragma unroll
				for (int w = wstart; w < wend; ++w) {
					//aveval += bottom_data[(h * width + w)*channels+c]*weight[((khstart+h-hstart) * kernel_w + (kwstart+w-wstart))*channels+c]; // (h-hstart)=>0~kernel_h
					out[(j*2+i)]+= bottom_data[(h * width + w)*channels+c]*weight[((khstart+h-hstart) * kernel_w + (kwstart+w-wstart))*channels+c];
				}
			}

		}
		#pragma unroll
		for(int j=0; j<2; j++)
			#pragma unroll
			for(int i=0; i<2; i++)
			//top_data[(((ph+j)*width)+(pw+i))*channels+c] = out[j*2+i];
			tmp_out[(j*2+i)*channels+c] = out[j*2+i];

		__syncthreads();

		//start 1x1 weight1x1
		float output_local[4]; output_local[0]=0.0;output_local[1]=0.0;output_local[2]=0.0;output_local[3]=0.0;
		for (int rc = 0; rc < 512; rc++) {
		//#pragma unroll
		for(int j=0; j<2;j++)
			//#pragma unroll
			for(int i=0; i<2;i++)
	    		output_local[j*2+i] += tmp_out[(j*2+i)*channels+rc] * weight1x1[c+ rc * channels ];
	  
	 	 }
		//#pragma unroll
		for(int j=0; j<2;j++)
			//#pragma unroll
			for(int i=0; i<2;i++)
			top_data[((ph+j) * width + (pw+i))*channels+c] = output_local[j*2+i];

	
	}
}




template <typename Dtype>
__global__ void GPU1x1Conv(const Dtype* const in, const Dtype* const weight, Dtype* const out, int const height, int const width, int const channels, int const out_channels)
{
	const int blockSize = 64;
	volatile __shared__ Dtype s_in[blockSize*4];// channel/2
	unsigned int tid = threadIdx.x;
	unsigned int stride = blockSize;

	//w map to block.x; h map to block.y
	int pos = blockIdx.y*width + blockIdx.x;
	#pragma unroll
	for(int oc=0; oc< out_channels; oc++ )
	{

		s_in[tid] = in[pos*channels+tid]*weight[oc*channels+tid] + in[pos*channels+tid+stride]*weight[oc*channels+tid+stride] + in[pos*channels+tid+stride*2]*weight[oc*channels+tid+stride*2] + in[pos*channels+tid+stride*3]*weight[oc*channels+tid+stride*3]+in[pos*channels+tid+stride*4]*weight[oc*channels+tid+stride*4] + in[pos*channels+tid+stride*5]*weight[oc*channels+tid+stride*5] + in[pos*channels+tid+stride*6]*weight[oc*channels+tid+stride*6] + in[pos*channels+tid+stride*7]*weight[oc*channels+tid+stride*7];
		
		/*s_in[tid]=0;		
		#pragma unroll
		for(int i=0; i<8; i++)	
			s_in[tid] +=in[pos*channels+tid+stride*i]*weight[oc*channels+tid+stride*i];*/

		__syncthreads();

		
		 //if (tid < 128) { s_in[tid] += s_in[tid + 128]; } 
		 //__syncthreads(); 
		 //if (tid < 64) { s_in[tid] += s_in[tid + 64]; } 
		 //__syncthreads(); 
			if (tid < 32) {
			s_in[tid] += s_in[tid + 32]; // __syncthreads(); 
			s_in[tid] += s_in[tid + 16]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 8]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 4]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 2]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 1]; //__syncthreads(); 
			}
			

		if (tid == 0) out[pos*out_channels+oc] = s_in[0];
		
		

	}
}

template <typename Dtype>
__global__ void GPU1x1ConvTile4(const Dtype* const in, const Dtype* const weight, Dtype* const out, int const height, int const width, int const channels, int const out_channels)
{
	const int blockSize = 64;
	volatile __shared__ Dtype s_in[blockSize*4];// channel/2
	unsigned int tid = threadIdx.x;
	unsigned int stride = blockSize;

	//w map to block.x; h map to block.y
	int o_pos = blockIdx.y*2*width + blockIdx.x*2;
	#pragma unroll
	for(int oc=0; oc< out_channels; oc++ )
	{
		int pos=o_pos;
		s_in[tid] = in[pos*channels+tid]*weight[oc*channels+tid] + in[pos*channels+tid+stride]*weight[oc*channels+tid+stride] + in[pos*channels+tid+stride*2]*weight[oc*channels+tid+stride*2] + in[pos*channels+tid+stride*3]*weight[oc*channels+tid+stride*3]+in[pos*channels+tid+stride*4]*weight[oc*channels+tid+stride*4] + in[pos*channels+tid+stride*5]*weight[oc*channels+tid+stride*5] + in[pos*channels+tid+stride*6]*weight[oc*channels+tid+stride*6] + in[pos*channels+tid+stride*7]*weight[oc*channels+tid+stride*7];
		pos=o_pos+1;
		s_in[tid+64] = in[pos*channels+tid]*weight[oc*channels+tid] + in[pos*channels+tid+stride]*weight[oc*channels+tid+stride] + in[pos*channels+tid+stride*2]*weight[oc*channels+tid+stride*2] + in[pos*channels+tid+stride*3]*weight[oc*channels+tid+stride*3]+in[pos*channels+tid+stride*4]*weight[oc*channels+tid+stride*4] + in[pos*channels+tid+stride*5]*weight[oc*channels+tid+stride*5] + in[pos*channels+tid+stride*6]*weight[oc*channels+tid+stride*6] + in[pos*channels+tid+stride*7]*weight[oc*channels+tid+stride*7];
		pos=o_pos+width;
		s_in[tid+128] = in[pos*channels+tid]*weight[oc*channels+tid] + in[pos*channels+tid+stride]*weight[oc*channels+tid+stride] + in[pos*channels+tid+stride*2]*weight[oc*channels+tid+stride*2] + in[pos*channels+tid+stride*3]*weight[oc*channels+tid+stride*3]+in[pos*channels+tid+stride*4]*weight[oc*channels+tid+stride*4] + in[pos*channels+tid+stride*5]*weight[oc*channels+tid+stride*5] + in[pos*channels+tid+stride*6]*weight[oc*channels+tid+stride*6] + in[pos*channels+tid+stride*7]*weight[oc*channels+tid+stride*7];
		pos=o_pos+width+1;
		s_in[tid+192] = in[pos*channels+tid]*weight[oc*channels+tid] + in[pos*channels+tid+stride]*weight[oc*channels+tid+stride] + in[pos*channels+tid+stride*2]*weight[oc*channels+tid+stride*2] + in[pos*channels+tid+stride*3]*weight[oc*channels+tid+stride*3]+in[pos*channels+tid+stride*4]*weight[oc*channels+tid+stride*4] + in[pos*channels+tid+stride*5]*weight[oc*channels+tid+stride*5] + in[pos*channels+tid+stride*6]*weight[oc*channels+tid+stride*6] + in[pos*channels+tid+stride*7]*weight[oc*channels+tid+stride*7];
		/*s_in[tid]=0;		
		#pragma unroll
		for(int i=0; i<8; i++)	
			s_in[tid] +=in[pos*channels+tid+stride*i]*weight[oc*channels+tid+stride*i];*/

		__syncthreads();

		
		 //if (tid < 128) { s_in[tid] += s_in[tid + 128]; } 
		 //__syncthreads(); 
		 //if (tid < 64) { s_in[tid] += s_in[tid + 64]; } 
		 //__syncthreads(); 
			if (tid < 32) {
			s_in[tid] += s_in[tid + 32]; // __syncthreads(); 
			s_in[tid] += s_in[tid + 16]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 8]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 4]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 2]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 1]; //__syncthreads(); 
			}
			if (tid < 96&&tid>=64) {
			s_in[tid] += s_in[tid + 32]; // __syncthreads(); 
			s_in[tid] += s_in[tid + 16]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 8]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 4]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 2]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 1]; //__syncthreads(); 
			}
			if (tid < 160&&tid>=128) {
			s_in[tid] += s_in[tid + 32]; // __syncthreads(); 
			s_in[tid] += s_in[tid + 16]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 8]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 4]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 2]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 1]; //__syncthreads(); 
			}
			if (tid < 224&&tid>=192) {
			s_in[tid] += s_in[tid + 32]; // __syncthreads(); 
			s_in[tid] += s_in[tid + 16]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 8]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 4]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 2]; //__syncthreads(); 
			s_in[tid] += s_in[tid + 1]; //__syncthreads(); 
			}



		//if (tid == 0) out[pos*out_channels+oc] = s_in[0];
		pos=o_pos;
		if (tid == 0)out[pos*out_channels+oc] = s_in[tid];
		pos=o_pos+1;
		if (tid == 1)out[pos*out_channels+oc] = s_in[tid+63];
		pos=o_pos+width;
		if (tid == 2)out[pos*out_channels+oc] = s_in[tid+126];
		pos=o_pos+width+1;
		if (tid == 3)out[pos*out_channels+oc] = s_in[tid+189];


	}
}

template <typename Dtype>
__global__ void GPU1x1Conv2(const Dtype* const in, const Dtype* const weight, Dtype* const out, int const height, int const width, int const channels, int const out_channels)
{       //each thread responsable for one output, bsz=196 gsz=512 
	const int blockSize = 196;
	const int tid = threadIdx.x;
	const int oc = blockIdx.x;
	float sum=0;
	for(int i=0; i<channels; i++)
		{sum += in[tid*channels+i]*weight[oc*channels+i];}

	out[tid*channels+oc]=sum;
	

}

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

template <typename Dtype>
__global__ void GPU1x1Conv3(const Dtype* const in, const Dtype* const weight, Dtype* const out, int const height, int const width, int const channels, int const out_channels)
{
	const int blockSize = 256;
	static __shared__ Dtype shared[8];
	unsigned int tid = threadIdx.x;
	unsigned int stride = blockSize;
	unsigned int lane = threadIdx.x % warpSize;
	unsigned int wid = threadIdx.x / warpSize;

	//w map to block.x; h map to block.y
	const int pos = blockIdx.y*width + blockIdx.x;
	#pragma unroll
	for(int oc=0; oc< out_channels; oc++ )
	{
		float sum = in[pos*channels+tid]*weight[oc*channels+tid];
		#pragma unroll
		for (int offset = warpSize/2; offset > 0; offset /= 2) {  
        		sum += __shfl_down(sum, offset);  
   		 }  
		if (lane==0) shared[wid]=sum;
	
		__syncthreads();
		sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
		#pragma unroll
		for (int offset = 4; offset > 0; offset /= 2) {  
        		sum += __shfl_down(sum, offset);  
   		 }  	

		if (tid == 0) out[pos*out_channels+oc] = sum;

	}
}


__global__ void GPU1x1Conv4( float* __restrict__ A,  float* __restrict__ W,  float* __restrict__ Conv2dOutput) {
   float Conv2dOutput_local[4];
  Conv2dOutput_local[0] = 0.000000e+00f;
  Conv2dOutput_local[2] = 0.000000e+00f;
  Conv2dOutput_local[1] = 0.000000e+00f;
  Conv2dOutput_local[3] = 0.000000e+00f;
  for (int rc = 0; rc < 512; ++rc) {
    Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (A[(((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + rc)] * W[(((int)threadIdx.x) + (rc )* 512)]));
    Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (A[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + rc) + 7168)] * W[(((int)threadIdx.x) + (rc)* 512)]));
    Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (A[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + rc) + 512)] * W[(((int)threadIdx.x) + (rc)* 512)]));
    Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (A[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + rc) + 7680)] * W[(((int)threadIdx.x) + (rc)* 512)]));
  }
  Conv2dOutput[(((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x))] = Conv2dOutput_local[0];
  Conv2dOutput[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x)) + 7168)] = Conv2dOutput_local[2];
  Conv2dOutput[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x)) + 512)] = Conv2dOutput_local[1];
  Conv2dOutput[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x)) + 7680)] = Conv2dOutput_local[3];
}

__global__ void GPU1x1Conv5( float* __restrict__ A,  float* __restrict__ W,  float* __restrict__ Conv2dOutput, int const width, int const channels) {
   float Conv2dOutput_local[4]={0};

  for (int rc = 0; rc < 512; rc++) {
	#pragma unroll
	for(int j=0; j<2;j++)
		#pragma unroll
		for(int i=0; i<2;i++)
    		Conv2dOutput_local[j*2+i] += A[(((blockIdx.x / 7) * width) + (blockIdx.x % 7)) * 2 * channels + rc + i*width + j*width*channels] * W[threadIdx.x + rc * channels ];
  
  }
#pragma unroll
for(int j=0; j<2;j++)
	#pragma unroll
	for(int i=0; i<2;i++)
	Conv2dOutput[((blockIdx.x / 7) * width + (blockIdx.x) % 7) * 2 * channels + threadIdx.x + i*width + j*width*channels] = Conv2dOutput_local[j*2+i];

}

__global__ void GPU1x1Conv6( float* __restrict__ A,  float* __restrict__ W,  float* __restrict__ Conv2dOutput) {
  __shared__ float PaddedInput_shared[2048];
   float Conv2dOutput_local[4];
  for (int ax1 = 0; ax1 < 2; ++ax1) {
    for (int ax2 = 0; ax2 < 2; ++ax2) {
      PaddedInput_shared[((((int)threadIdx.x) + (ax1 * 1024)) + (ax2 * 512))] = A[(((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x)) + (ax1 * 7168)) + (ax2 * 512))];
    }
  }
  Conv2dOutput_local[0] = 0.000000e+00f;
  Conv2dOutput_local[2] = 0.000000e+00f;
  Conv2dOutput_local[1] = 0.000000e+00f;
  Conv2dOutput_local[3] = 0.000000e+00f;
  __syncthreads();
  for (int rc = 0; rc < 512; ++rc) {
    Conv2dOutput_local[0] = (Conv2dOutput_local[0] + (PaddedInput_shared[rc] * W[(((int)threadIdx.x) + (rc * 512))]));
    Conv2dOutput_local[2] = (Conv2dOutput_local[2] + (PaddedInput_shared[(rc + 1024)] * W[(((int)threadIdx.x) + (rc * 512))]));
    Conv2dOutput_local[1] = (Conv2dOutput_local[1] + (PaddedInput_shared[(rc + 512)] * W[(((int)threadIdx.x) + (rc * 512))]));
    Conv2dOutput_local[3] = (Conv2dOutput_local[3] + (PaddedInput_shared[(rc + 1536)] * W[(((int)threadIdx.x) + (rc * 512))]));
  }
  Conv2dOutput[(((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x))] = Conv2dOutput_local[0];
  Conv2dOutput[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x)) + 7168)] = Conv2dOutput_local[2];
  Conv2dOutput[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x)) + 512)] = Conv2dOutput_local[1];
  Conv2dOutput[((((((((int)blockIdx.x) / 7) * 14) + (((int)blockIdx.x) % 7)) * 1024) + ((int)threadIdx.x)) + 7680)] = Conv2dOutput_local[3];
}


void CPU1x1Conv(float *in, float *weight, double *out, int const height, int const width, int const channels, int const out_channels)
{

for(int h=0; h< height; h++)
	for(int w=0; w< width; w++)
		for(int oc=0; oc< out_channels; oc++)
				out[(h*width + w)*out_channels+oc] = 0;


for(int h=0; h< height; h++)
	for(int w=0; w< width; w++)
		for(int oc=0; oc< out_channels; oc++)
			for(int c=0; c< channels; c++)
			{
				out[(h*width + w)*out_channels+oc] += in[(h*width + w)*channels+c]*weight[oc*channels + c];

			}


}





int main(int argc, char* argv[]) 
{

	
	const int channels = 512;
	const int height = 14;
	const int width = 14;

	const int kernel_h = 3;
	const int kernel_w = 3;
	const int stride_h = 1;
	const int stride_w = 1;
	const int pad_h = 1;
	const int pad_w = 1;

	const int conved_height = height;
	const int conved_weight = width;

	const bool bias_term = false;
	const int n=channels * height * width;
	const int m=channels * kernel_h * kernel_w;
	const int wn=channels * channels;

	float *d_weight, *d_bottom, *d_bottom_padded, *d_top1, *d_top2, *d_weight1x1, *d_saparable_out;
	float *d_top2nhwc, *d_bottom_nchw, *d_weight_nchw;
	
	cudaMallocManaged(&d_weight, m*sizeof(float));
	cudaMallocManaged(&d_weight1x1, wn*sizeof(float));
	cudaMallocManaged(&d_bottom, n*sizeof(float));
	cudaMallocManaged(&d_top1, n*sizeof(float));
	cudaMallocManaged(&d_top2, n*sizeof(float));
	cudaMallocManaged(&d_bottom_nchw, n*sizeof(float));
	cudaMallocManaged(&d_weight_nchw, m*sizeof(float));
	for(int i=0;i<m;i++)
	d_weight[i]=((double) rand() / (RAND_MAX)/10);
	for(int i=0;i<n;i++)
	d_bottom[i]=((double) rand() / (RAND_MAX)/10);
	for(int i=0;i<wn;i++)
	d_weight1x1[i]=((double) rand() / (RAND_MAX)/10);


	printf("numblocks=%d", CAFFE_GET_BLOCKS(n));
	ConvForward<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
					n, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_top1,d_weight,0,bias_term);
	/*ConvForward<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
					n, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_top1,d_weight,0,bias_term);*/

	int nb=CAFFE_GET_BLOCKS(n);
	int bs=CAFFE_CUDA_NUM_THREADS/4;
	int nt=n/4;
	/*ConvForwardTile4<float><<<nb, bs>>>(
					nt, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_top2,d_weight,0,bias_term);*/
	
	
	float *out1 = new float[n];
	//float *out2 = new float[n];
	cudaMemcpy(out1, d_top1, n*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(out2, d_top2, n*sizeof(float), cudaMemcpyDeviceToHost);
	int c=0;
	/*for(int i=0;i<n;i++)
		if(out1[i]!=out2[i]&&c<20)
			{printf("top1[%d]=%f, top2[%d]=%f", i, out1[i], i, out2[i]);
			c++;}*/


		//saparable convolution

		cudaMallocManaged(&d_saparable_out, n*sizeof(float));
		float *weight1x1 = new float[wn];
		double *saparable_out = new double[n];
		cudaMemcpy(weight1x1, d_weight1x1, wn*sizeof(float), cudaMemcpyDeviceToHost);
		for(int i=0; i<n; i++) saparable_out[i]=0;

		CPU1x1Conv(out1, weight1x1, saparable_out, height, width, channels, channels);
		
		dim3 numBlocks(14,14,1);
		
		//GPU1x1Conv<float><<<numBlocks,64>>>(d_top1, d_weight1x1, d_saparable_out, height, width, channels, channels);
		

		
		float *d_weight1x1hwio;
		cudaMallocManaged(&d_weight1x1hwio, wn*sizeof(float));
		for(int i=0; i<512; i++)
			for(int j=0; j<512; j++)
				{

				d_weight1x1hwio[j*512+i]=weight1x1[i*512+j];
				}
		/*ConvForwardAnd1x1ConvTile4<float><<<49, 512>>>(
					nt, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_saparable_out,d_weight,d_weight1x1hwio,0,bias_term);
		ConvForwardAnd1x1ConvTile4<float><<<49, 512>>>(
					nt, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_saparable_out,d_weight,d_weight1x1hwio,0,bias_term);*/
		
		//GPU1x1Conv4<<<49,512>>>( d_top1,  d_weight1x1hwio,  d_saparable_out);
		//GPU1x1Conv5<<<196,512/4>>>( d_top1,  d_weight1x1hwio,  d_saparable_out, width, channels);
		//dim3 numBlocks1(14,14,1);
		GPU1x1Conv6<<<49,512>>>( d_top1,  d_weight1x1hwio,  d_saparable_out);
		GPU1x1Conv6<<<49,512>>>( d_top1,  d_weight1x1hwio,  d_saparable_out);
		//GPU1x1Conv<float><<<numBlocks1,64>>>(d_top1, d_weight1x1, d_saparable_out, height, width, channels, channels);

		float *outc = new float[n];
		cudaMemcpy(outc, d_saparable_out, n*sizeof(float), cudaMemcpyDeviceToHost);
		c=0;
		for(int i=n-50;i<n;i++)
			if(abs(outc[i]-saparable_out[i])>0.0001&&c<20)
				//if(c<20)
				{printf("outc[%d]=%f, saparable_out[%d]=%f", i, outc[i], i, saparable_out[i]);
				c++;}

		//for(int i=0;i<n;i++)if(outc[i]==0) printf("here[%d]",i);
		


return 0;
}







