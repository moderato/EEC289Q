###Separable Convolution by CUDA and TVM
CUDA and TVM code for separable convolution (depthwise convolution + 1x1 convolution).

####Mar 2018
TVM scheduler for normal convolution with the tensor layout in the form of NHWC, currently missing in the TVM repository.    
Improved runtime of TVM’s depthwise convolution scheduler by replacing filter-reuse with input-reuse.     
####Nov 2018
Adding simple benchmarks for analyzing bottlenecks of fused and non-fused computation