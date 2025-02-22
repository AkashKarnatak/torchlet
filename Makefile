all:
	nvcc -O3 -arch=sm_86 -lm tensor.c tensor.cu mnist_gpu.cu -o mnist_gpu.out
	gcc -g -O3 -fsanitize=address -lm tensor.c mnist_cpu.c  -o mnist_cpu.out
