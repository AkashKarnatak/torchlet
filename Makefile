all:
	nvcc -arch=sm_86 -lm tensor.c tensor.cu nn.c -o nn.out
