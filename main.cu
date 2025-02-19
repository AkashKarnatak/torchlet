#include "tensor.h"

int main() {
  cudaDeviceSynchronize();

  Tensor *a, *b, *c, *c_cpu;
  Tensor *a_gpu, *b_gpu, *c_gpu;

  a = tensor_randn((size_t[]){4, 5}, 2);
  b = tensor_randn((size_t[]){5, 3}, 2);

  a_gpu = tensor_to_gpu(a);
  b_gpu = tensor_to_gpu(b);

  c = tensor_matmul(a, b);
  c_gpu = tensor_matmul_gpu(a_gpu, b_gpu);

  c_cpu = tensor_to_cpu(c_gpu);

  tensor_print(c);
  tensor_debug(c);

  tensor_print(c_cpu);
  tensor_debug(c_cpu);

  tensor_free(a);
  tensor_free(b);
  tensor_free(c);

  return 0;
}
