all:
	nvcc -arch=sm_86 -lm tensor.c tensor.cu main.cu -o main.out
