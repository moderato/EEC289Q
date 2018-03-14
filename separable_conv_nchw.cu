#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

#define CAFFE_CUDA_NUM_THREADS 196
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
		const int pw = index % conved_width; // width position of output
		const int ph = (index / conved_width) % conved_height;
		const int c = (index / conved_width / conved_height) % channels;
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
		const Dtype* const bottom_slice =
		bottom_data + (n * channels + c) * height * width;
		const Dtype* const weight_slice =
		weight + c * kernel_h * kernel_w;
		int khstart=hend<kernel_h?kernel_h-hend:0;
		int kwstart=wend<kernel_w?kernel_w-wend:0;

		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {

				aveval += bottom_slice[h * width + w]*weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)]; // (h-hstart)=>0~kernel_h
			}
		}
		if(bias_term_) {
			aveval+=bias[c];
		}
		top_data[index] = aveval;
	}
}



template <typename Dtype>
__global__ void ConvForwardShared(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {

__shared__ Dtype s_bottom[CAFFE_CUDA_NUM_THREADS], s_weight[CAFFE_CUDA_NUM_THREADS];


	/*for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads * 4; index += blockDim.x * gridDim.x){
		s_bottom[index%CAFFE_CUDA_NUM_THREADS]= bottom_data[index];
		s_weight[index%CAFFE_CUDA_NUM_THREADS]= weight[index%CAFFE_CUDA_NUM_THREADS + blockIdx.x * kernel_h * kernel_w * 1];

}*/
	//Dtype l_weight [9];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index % blockDim.x < kernel_h * kernel_w){
		s_weight[index % (kernel_h * kernel_w)]= weight[index % (kernel_h * kernel_w) + blockIdx.x * kernel_h * kernel_w * 1];

}
	#pragma unroll
	for (int i=index % blockDim.x  ; i < blockDim.x * 4 ; i += blockDim.x){
		s_bottom[i]= bottom_data[blockIdx.x * blockDim.x * 4 +i]; 

}
__syncthreads();
	//for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads ; index += blockDim.x * gridDim.x){

	/*#pragma unroll	
	for(int i=0; i<9;i++)
		l_weight[i]=s_weight[i];*/

	if (index < nthreads){
		Dtype out[4]={0}; //local output
		const int pw = (index * 2) % conved_width; // width position of output
		const int ph = (index * 2 / conved_width) * 2 % conved_height;
		const int c = (index * 4 / conved_width / conved_height) % channels;
		const int n = index / conved_width / conved_height / channels;// =0
	

		/*const Dtype* const bottom_slice =
		bottom_data + (n * channels + c) * height * width;*/
		/*const Dtype* const weight_slice =
		weight + c * kernel_h * kernel_w;*/
		
#pragma unroll
		for(int j=0; j<2; j++)
#pragma unroll
			for(int i=0; i<2; i++)
		{		
				//Dtype aveval=0;
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

						//aveval += s_bottom[h * width + w ] * s_weight[(khstart+ h -hstart) * kernel_w + (kwstart + w -wstart)]; 
						out[j*2+i]+= s_bottom[h * width + w ] * s_weight[(khstart+ h -hstart) * kernel_w + (kwstart + w -wstart)];

					}
				}
	
			//if(bias_term_) aveval+=bias[c];

			//top_data[(c * conved_height + ph + j) * conved_width + pw + i] = aveval;

		}
		#pragma unroll
		for(int j=0; j<2; j++)
			#pragma unroll
			for(int i=0; i<2; i++)
			top_data[(c * conved_height + ph + j) * conved_width + pw + i] = out[j*2+i];//hard code numbers here will increase speed
	}
}




template <typename Dtype>
__global__ void ConvForwardCheat(const Dtype* const bottom_data, Dtype* const top_data,const Dtype* const weight) {

__shared__ Dtype s_bottom[196], s_weight[196];

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index % blockDim.x < 9){
		s_weight[index % 9]= weight[index % 9 + blockIdx.x *9];

}
	#pragma unroll
	for (int i=index % blockDim.x  ; i < blockDim.x * 4 ; i += blockDim.x){
		s_bottom[i]= bottom_data[blockIdx.x * blockDim.x * 4 +i]; 

}
__syncthreads();

	if (index < 512*14*14){


		Dtype out[4]={0};
		const int pw = (index * 2) % 14; // width position of output
		const int ph = (index * 2 / 14)*2 % 14;
		const int c = (index /49) % 512;
		//const int n = index / conved_width / conved_height / channels;// =0
		
#pragma unroll
		for(int j=0; j<2; j++)
#pragma unroll
			for(int i=0; i<2; i++)
		{		
				Dtype aveval = 0;
				const int hstart = (ph + j ) - 1 >0? (ph + j ) - 1 :0;
				const int wstart = (pw + i)  - 1 >0? (pw + i)  - 1 :0;
				const int hend = (ph + j )< 12? (ph + j) +2 : 14; 
				const int wend = (pw + i) < 12? (pw + i) +2 : 14; 
				const int khstart=hend<3?3-hend:0;
				const int kwstart=wend<3?3-wend:0;
				#pragma unroll
				for (int h = hstart; h < hend; ++h) {
					#pragma unroll
					for (int w = wstart; w < wend; ++w) {

						//aveval += s_bottom[h * 14 + w ] * s_weight[(khstart+ h -hstart) * 3 + (kwstart + w -wstart)]; 
						out[j*2+i]+= s_bottom[h * 14 + w ] * s_weight[(khstart+ h -hstart) * 3 + (kwstart + w -wstart)];

					}
				}

			//top_data[(c * 14 + ph + j) * 14 + pw + i] = aveval;

		}

		#pragma unroll
		for(int j=0; j<2; j++)
			#pragma unroll
			for(int i=0; i<2; i++)
			top_data[(c * 14 + ph + j) * 14 + pw + i] = out[j*2+i];
		
	}
}

template <typename Dtype>
__global__ void GPU1x1Conv(const Dtype* const in, const Dtype* const weight, Dtype* const out, int const height, int const width, int const channels, int const out_channels)
{
	const int blockSize = 256;
	__shared__ Dtype s_in[blockSize];// channel/2
	unsigned int tid = threadIdx.x;
	unsigned int startId = tid*width*height; // 1~256 slice
	unsigned int stride = blockSize*width*height;

	//w map to block.x; h map to block.y
	const int pos = blockIdx.y*width + blockIdx.x;

	for(int oc=0; oc< out_channels; oc++ )
	{
		s_in[tid] = in[startId+pos]*weight[oc*channels+tid] + in[startId+pos+stride]*weight[oc*channels+tid+blockSize];
		__syncthreads();

		 if (tid < 128) { s_in[tid] += s_in[tid + 128]; } 
		 __syncthreads(); 
		 if (tid < 64) { s_in[tid] += s_in[tid + 64]; } 
		 __syncthreads(); 
			if (tid < 32) {
			s_in[tid] += s_in[tid + 32]; __syncthreads(); 
			s_in[tid] += s_in[tid + 16]; __syncthreads(); 
			s_in[tid] += s_in[tid + 8]; __syncthreads(); 
			s_in[tid] += s_in[tid + 4]; __syncthreads(); 
			s_in[tid] += s_in[tid + 2]; __syncthreads(); 
			s_in[tid] += s_in[tid + 1]; __syncthreads(); 
			}
		if (tid == 0) out[oc*width*height+pos] = s_in[0];

	}
}





void CPU1x1Conv(float *in, float *weight, double *out, int const height, int const width, int const channels, int const out_channels)
{


for(int oc=0; oc< out_channels; oc++)
for(int h=0; h< height; h++)
	for(int w=0; w< width; w++)
		for(int c=0; c< channels; c++)
			{
				out[oc*height*width+ h*width + w] += in[c*height*width + h*width + w]*weight[oc*channels + c];

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
	const int wn=channels * channels;

	float *d_weight, *d_bottom, *d_bottom_padded, *d_top1, *d_top2, *d_weight1x1, *d_saparable_out;
	cudaMallocManaged(&d_weight, n*sizeof(float));
	cudaMallocManaged(&d_weight1x1, wn*sizeof(float));
	cudaMallocManaged(&d_bottom, n*sizeof(float));
	cudaMallocManaged(&d_top1, n*sizeof(float));
	cudaMallocManaged(&d_top2, n*sizeof(float));
	for(int i=0;i<n;i++)
	d_weight[i]=((double) rand() / (RAND_MAX)/10);
	for(int i=0;i<n;i++)
	d_bottom[i]=((double) rand() / (RAND_MAX)/10);
	for(int i=0;i<wn;i++)
	d_weight1x1[i]=((double) rand() / (RAND_MAX)/10);

	int pcount = (height+pad_h*2)*(width+pad_w*2)*channels;


	printf("numblocks=%d", CAFFE_GET_BLOCKS(n));
	ConvForward<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
					n, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_top1,d_weight,0,bias_term);
	int nb=CAFFE_GET_BLOCKS(n);
	int bs=CAFFE_CUDA_NUM_THREADS/4;
	int nt=n/4;


	ConvForwardShared<float><<<nb, bs>>>(
					nt, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_top2,d_weight,0,bias_term);

	//ConvForwardCheat<float><<<nb, bs>>>(d_bottom, d_top2,d_weight);



		float *out1 = new float[n];
		float *out2 = new float[n];
		cudaMemcpy(out1, d_top1, n*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(out2, d_top2, n*sizeof(float), cudaMemcpyDeviceToHost);
		int c=0;
		for(int i=0;i<n;i++)
			if(out1[i]!=out2[i]&&c<20)
				{printf("top1[%d]=%f, top2[%d]=%f", i, out1[i], i, out2[i]);
				c++;}
		cudaFree(d_top2);
		//saparable convolution
		cudaMallocManaged(&d_saparable_out, n*sizeof(float));
		float *weight1x1 = new float[wn];
		double *saparable_out = new double[n];
		cudaMemcpy(weight1x1, d_weight1x1, wn*sizeof(float), cudaMemcpyDeviceToHost);
		for(int i=0; i<n; i++) saparable_out[i]=0;

		CPU1x1Conv(out1, weight1x1, saparable_out, height, width, channels, channels);
		
		dim3 numBlocks(14,14,1);
		
		GPU1x1Conv<float><<<numBlocks,channels/2>>>(d_top1, d_weight1x1, d_saparable_out, height, width, channels, channels);

		float *outc = new float[n];
		cudaMemcpy(outc, d_saparable_out, n*sizeof(float), cudaMemcpyDeviceToHost);
		c=0;
		for(int i=0;i<n;i++)
			if(abs(outc[i]-saparable_out[i])>0.1&&c<20)
				//if(c<20)
				{printf("outc[%d]=%f, saparable_out[%d]=%f", i, outc[i], i, saparable_out[i]);
				c++;}
		


return 0;
}







