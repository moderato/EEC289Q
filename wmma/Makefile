all: simpleTensorCoreGEMM.cu
	nvcc -o TCGemm -arch=sm_70 -lcublas -lcurand simpleTensorCoreGEMM.cu

clean:
	rm -f TCGemm
