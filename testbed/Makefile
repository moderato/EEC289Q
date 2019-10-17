NVCC_OPTIONS := -keep -O3 -gencode arch=compute_61,code=sm_61 -Xptxas --warn-on-spills --resource-usage -lineinfo --maxrregcount=72
INCLUDE_PATH := -I/home/zhongyilin/Documents/cutlass

all:
	nvcc $(INCLUDE_PATH) $(NVCC_OPTIONS) cuda_testbed.cu -o test -lcnpy -lz --std=c++11
dryrun:
	nvcc $(INCLUDE_PATH) -dryrun -arch=sm_61 -o test cuda_testbed.cu --keep -lcnpy -lz --std=c++11 2>dryrun.out
	# sed -i 's/#\$\ //' dryrun.out
clean:
	mv cuda_testbed.cu ..
	rm -rf cuda_testbed*
	rm -rf test*
	mv ../cuda_testbed.cu .
