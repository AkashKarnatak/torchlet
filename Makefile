all:
	gcc -fsanitize=address -lm tensor.c nn.c -o nn.out
