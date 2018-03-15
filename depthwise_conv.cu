#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

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

__shared__ float s_bottom[CAFFE_CUDA_NUM_THREADS], s_weight[CAFFE_CUDA_NUM_THREADS];


	/*for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads * 4; index += blockDim.x * gridDim.x){
		s_bottom[index%CAFFE_CUDA_NUM_THREADS]= bottom_data[index];
		s_weight[index%CAFFE_CUDA_NUM_THREADS]= weight[index%CAFFE_CUDA_NUM_THREADS + blockIdx.x * kernel_h * kernel_w * 1];

}*/
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index % (kernel_h * kernel_w) < kernel_h * kernel_w){
		s_weight[index % (kernel_h * kernel_w)]= weight[index % (kernel_h * kernel_w) + blockIdx.x * kernel_h * kernel_w * 1];

}
	/*#pragma unroll
	for (int i=index % blockDim.x  ; i < blockDim.x * 4 ; i += blockDim.x){
		s_bottom[i]= bottom_data[blockIdx.x * blockDim.x * 4 +i]; 

}*/
__syncthreads();
	//for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads ; index += blockDim.x * gridDim.x){
	if (index < nthreads){

		const int pw = (index * 2) % conved_width; // width position of output
		const int ph = (index * 2 / conved_width) * 2 % conved_height;
		const int c = (index * 4 / conved_width / conved_height) % channels;
		const int n = index / conved_width / conved_height / channels;// =0


		//int hend = min(hstart + kernel_h, height); 
		//int wend = min(wstart + kernel_w, width); 
		
		
		
		const Dtype* const bottom_slice =
		bottom_data + (n * channels + c) * height * width;
		/*const Dtype* const weight_slice =
		weight + c * kernel_h * kernel_w;*/
		

		for(int j=0; j<2; j++)
			for(int i=0; i<2; i++)
		{		
				Dtype aveval = 0;
				//int hstart = (ph + j )* stride_h - pad_h; // input pointer starting point
				//int wstart = (pw + i) * stride_w - pad_w;
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

						aveval += bottom_slice[h * width + w ] * s_weight[(khstart+ h -hstart) * kernel_w + (kwstart + w -wstart)]; // (h-hstart)=>0~kernel_h

					}
				}
			
			if(bias_term_) aveval+=bias[c];

			top_data[(c * conved_height + ph + j) * conved_width + pw + i] = aveval;

		}
	}
}

template <typename Dtype>
__global__ void Padding(const int nthreads, const Dtype* const bottom_in, Dtype* bottom_out, const int height, const int width, int pad_h, const int pad_w)
{

//extern __shared__ Dtype s_bottom[];

//int index = blockIdx.x * blockDim.x + threadIdx.x;
int index = threadIdx.x;


if (index>=width+pad_w*3 && index < (width+pad_w*2)*(height+pad_h*2)-width-pad_w*3 && 
index %(width+pad_w*2) != 0 && index %(width+pad_w*2) != width+pad_w*2-1){
bottom_out[index+blockIdx.x * blockDim.x ]=bottom_in[(index/(width+pad_w*2)-pad_h)*width + index%(width+pad_w*2)-pad_w + blockIdx.x * height*width];
//printf("bid=%d, bdim=%d", blockIdx.x, blockDim.x);	
}
else
bottom_out[index+blockIdx.x * blockDim.x ]=0;
}



template <typename Dtype>
__global__ void ConvForwardPadded(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {

__shared__ float s_bottom[CAFFE_CUDA_NUM_THREADS], s_weight[CAFFE_CUDA_NUM_THREADS];


	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index % (kernel_h * kernel_w) < kernel_h * kernel_w){
		s_weight[index % (kernel_h * kernel_w)]= weight[index % (kernel_h * kernel_w) + blockIdx.x * kernel_h * kernel_w * 1];

}
	/*#pragma unroll
	for (int i=index % blockDim.x  ; i < blockDim.x * 4 ; i += blockDim.x){
		s_bottom[i]= bottom_data[blockIdx.x * blockDim.x * 4 +i]; 

}*/
__syncthreads();
	
	if (index < nthreads){

		const int pw = (index * 2) % conved_width; // width position of output
		const int ph = (index * 2 / conved_width) * 2 % conved_height;
		const int c = (index * 4 / conved_width / conved_height) % channels;
		const int n = index / conved_width / conved_height / channels;// =0

		const Dtype* const bottom_slice =
		bottom_data + (n * channels + c) * (height+pad_h*2) * (width+pad_w*2);
		

		for(int j=0; j<2; j++)
			for(int i=0; i<2; i++)
		{		
				Dtype aveval = 0;
				const int hstart = (ph + j )* stride_h ;
				const int wstart = (pw + i) * stride_w ;
				//const int hend = hstart + kernel_h; 
				//const int wend = (pw + i) * stride_w - pad_w + kernel_w; 
				const int khstart=0;
				const int kwstart=0;
				#pragma unroll
				for (int h = hstart; h < hstart + kernel_h; ++h) {
					#pragma unroll
					for (int w = wstart; w < wstart + kernel_w; ++w) {

						aveval += bottom_slice[h * (width+pad_w*2) + w ] * s_weight[(h -hstart) * kernel_w + (w -wstart)]; // (h-hstart)=>0~kernel_h

					}
				}
			
			if(bias_term_) aveval+=bias[c];

			top_data[(c * conved_height + ph + j) * conved_width + pw + i] = aveval;

		}
	}
}





int main(int argc, char* argv[]) 
{

//filter 3 × 3 × 512 dw
//input 14 × 14 × 512
	//float* weight = new float[1024];
	//float* bottom = new float[1024];

/*printf("here\n");
float *a,*b,*c;
cudaMallocManaged(&a, 256*sizeof(float));
cudaMallocManaged(&b, 256*sizeof(float));
cudaMallocManaged(&c, 256*sizeof(float));
float *out = new float[256];
for(int i=0;i<256;i++)
{a[i]=1;b[i]=2;}
test<<<16, 256>>>(a,b,c);
cudaMemcpy(out, c, 256*sizeof(float), cudaMemcpyDeviceToHost);
printf("c[3]=%f",out[3]);
*/


	
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
	const int count = n;
	float *d_weight, *d_bottom, *d_bottom_padded, *d_top1, *d_top2;
	cudaMallocManaged(&d_weight, n*sizeof(float));
	cudaMallocManaged(&d_bottom, n*sizeof(float));
	cudaMallocManaged(&d_top1, n*sizeof(float));
	cudaMallocManaged(&d_top2, n*sizeof(float));
	for(int i=0;i<n;i++)
	d_weight[i]=((double) rand() / (RAND_MAX));
	for(int i=0;i<n;i++)
	d_bottom[i]=((double) rand() / (RAND_MAX));

	int pcount = (height+pad_h*2)*(width+pad_w*2)*channels;


	

	printf("numblocks=%d", CAFFE_GET_BLOCKS(n));
	ConvForward<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
					n, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_top1,d_weight,0,bias_term);
	int nb=CAFFE_GET_BLOCKS(n);
	int bs=CAFFE_CUDA_NUM_THREADS/4;
	int nt=n/4;


	/*ConvForwardShared<float><<<nb, bs>>>(
					nt, d_bottom, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_top2,d_weight,0,bias_term);*/




		int numPadThreads=(height+pad_h*2)*(width+pad_w*2);
		cudaMallocManaged(&d_bottom_padded, pcount*sizeof(float));
		Padding<float><<<(pcount + numPadThreads - 1) / numPadThreads, numPadThreads>>>(pcount, d_bottom, d_bottom_padded, height, width, pad_h, pad_w );
		float *bottom_padded= new float[pcount];
		cudaMemcpy(bottom_padded, d_bottom_padded, pcount*sizeof(float), cudaMemcpyDeviceToHost);

		for(int j=0;j< (height+pad_h*2)+10; j++){
			for(int i=0; i< (width+pad_w*2); i++)
				printf("%.1f  ", bottom_padded[i+j*(width+pad_w*2)]);
			printf("\n");
		}

		ConvForwardPadded<float><<<nb, bs>>>(
					nt, d_bottom_padded, n, channels,
					height, width,conved_height,conved_weight,kernel_h,
					kernel_w, stride_h, stride_w, pad_h, pad_w, d_top2,d_weight,0,bias_term);


		float *out1 = new float[n];
		float *out2 = new float[n];
		cudaMemcpy(out1, d_top1, n*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(out2, d_top2, n*sizeof(float), cudaMemcpyDeviceToHost);
		int c=0;
		for(int i=0;i<n;i++)
			if(out1[i]!=out2[i]&&c<20)
				{printf("top1[%d]=%f, top2[%d]=%f", i, out1[i], i, out2[i]);
				c++;}


return 0;
}







