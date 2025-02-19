all:
	nvcc -O3 -arch=sm_86 -lm tensor.c tensor.cu main.cu -o main.out
