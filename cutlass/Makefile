NVCC_OPTIONS := -keep -O3 -gencode arch=compute_61,code=sm_61 -Xptxas --warn-on-spills --resource-usage -lineinfo
INCLUDE_PATH := -I/home/zhongyilin/Documents/cutlass

all:
	nvcc $(INCLUDE_PATH) $(NVCC_OPTIONS) pipeline.cu -o pipeline -lcnpy -lz --std=c++11
dryrun:
	nvcc $(INCLUDE_PATH) -dryrun -arch=sm_61 -o pipeline pipeline.cu --keep -lcnpy -lz --std=c++11 2>dryrun.out
	# sed -i 's/#\$\ //' dryrun.out
clean:
	mv pipeline.cu ..
	rm -rf pipeline*
	mv ../pipeline.cu .
