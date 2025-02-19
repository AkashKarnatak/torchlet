all:
	gcc -fsanitize=address tensor.c -o tensor.out
