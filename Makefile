all:
	nvcc -arch=sm_86 -lm tensor.c tensor.cu nn.cu -o nn.out
