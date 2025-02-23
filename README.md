# torchlet

A lightweight torch like tensor computation library implemented from scratch in C and CUDA.

I created this to better understand the inner workings of tensor libraries like PyTorch,
making it an excellent resource for learning how these libraries work under the hood.

## Features

- Basic tensor operations implemented in C and CUDA
- Simple and readable implementation focused on learning
- GPU acceleration support through CUDA
- Zero external dependencies
- Clear code structure for educational purposes

## Prerequisites

- C compiler (gcc/clang)
- CUDA Toolkit
- Make

## Getting Started

### Basic Usage

Here's a simple example demonstrating matrix multiplication with torchlet:

```c
// main.cu
#include "tensor.h"

int main() {
    // Create two random tensors
    Tensor *a = tensor_randn((size_t[]){4, 5}, 2);  // 4x5 matrix
    Tensor *b = tensor_randn((size_t[]){5, 3}, 2);  // 5x3 matrix
    
    // Perform matrix multiplication
    Tensor *c = tensor_matmul(a, b);  // Results in a 4x3 matrix
    
    // Display results
    tensor_print(c);   // Print tensor values
    tensor_debug(c);   // Print tensor metadata
    
    // Clean up
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    
    return 0;
}
```

### Compiling Your Code

```bash
nvcc -O3 -Xcompiler -Wall -Xcompiler -Werror -arch=sm_86 -lm tensor.c tensor.cu main.cu -Xcompiler -fopenmp -o main.out
```

### Running the Example

```bash
./main
```

## Contributing

Contributions are welcome! If you find a bug, have an idea for an enhancement, or want to contribute in any way, feel free to open an issue or submit a pull request.

## License

This project is licensed under the AGPL3 License. For details, see the [LICENSE](LICENSE) file.
