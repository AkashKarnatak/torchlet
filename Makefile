all:
	nvcc -O3 -Xcompiler -Wall -Xcompiler -Werror -arch=sm_86 -lm tensor.c tensor.cu mnist_utils.c mnist_gpu.cu -Xcompiler -fopenmp -o mnist_gpu.out
	gcc -Wall -Werror -g -O3 -fsanitize=address -lm -fopenmp tensor.c mnist_utils.c mnist_cpu.c -o mnist_cpu.out
	gcc -Wall -Werror -g -O3 -fsanitize=address -lm -lraylib tensor.c mnist_gui.c -fopenmp -o mnist_gui.out
