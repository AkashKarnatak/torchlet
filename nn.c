#include "tensor.h"

Tensor *linear(Tensor *in, size_t in_dims, size_t out_dims, float gain) {
  va_list args;
  size_t weight_shape[] = {in_dims, out_dims};
  size_t bias_shape[] = {in->shape[1], out_dims};

  Tensor *weight = tensor_zeros(weight_shape, 2);
  kaiming_init(weight, gain);
  Tensor *bias = tensor_zeros(bias_shape, 2);

  Tensor *z = tensor_matmul(in, weight);
  Tensor *out = tensor_add(z, bias);

  tensor_free(weight);
  tensor_free(bias);
  tensor_free(z);

  return out;
}

int main() {
  Tensor *data;
  Tensor *out1;

  data = tensor_randn((size_t[]){32, 784}, 2);

  out1 = linear(data, 784, 512, sqrt(2));

  tensor_debug(out1);

  tensor_free(data);
  tensor_free(out1);
}
