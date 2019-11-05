### Separable Convolution by CUDA and TVM
CUDA and TVM code for separable convolution (depthwise convolution + 1x1 convolution).

#### Mar 2018
TVM scheduler for normal convolution with the tensor layout in the form of NHWC, currently missing in the TVM repository.    
Improved runtime of TVM’s depthwise convolution scheduler by replacing filter-reuse with input-reuse.     
#### Nov 2018
Adding simple benchmarks for analyzing bottlenecks of fused and non-fused computation
#### Oct 2019
Reaching 147us over 56x56x112. Expanding to all size.

#### Files:
*depth\_1by1\_fused\_compute.py:* Fused compute of depthwise & 1by1.
*depth\_1by1\_test.py:* Measure cuDNN's running time; generate .npy files (especially intermediate output) for cudnn\_bench.
*depth\_conv\_fused\_schedule\_test.py:* Generate .npy files for fused kernel.
