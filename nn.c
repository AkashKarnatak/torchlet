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
  Tensor *out1, *out2, *out3, *out4, *out5;

  data = tensor_randn((size_t[]){32, 784}, 2);

  out1 = linear(data, 784, 512, sqrt(2));
  out2 = tensor_relu(out1);
  out3 = linear(out2, 512, 10, 1);
  out4 = tensor_softmax(out3, -1);
  out5 = tensor_sum_at(out4, -1);

  printf("mean: %f, std: %f\n", tensor_mean(data), tensor_std(data));

  printf("mean: %f, std: %f\n", tensor_mean(out1), tensor_std(out1));

  printf("mean: %f, std: %f\n", tensor_mean(out2), tensor_std(out2));

  printf("mean: %f, std: %f\n", tensor_mean(out3), tensor_std(out3));

  tensor_debug(out4);
  tensor_print(out4);
  tensor_print(out5);

  tensor_free(data);
  tensor_free(out1);
  tensor_free(out2);
  tensor_free(out3);
  tensor_free(out4);
  tensor_free(out5);

  return 0;
}
