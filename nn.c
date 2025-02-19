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
  Tensor *data, *out1, *out2, *out3, *out4, *out5, *y;
  Tensor *d_out5;
  float loss;

  data = tensor_randn((size_t[]){32, 784}, 2);
  y = tensor_randint((size_t[]){32}, 1, 0, 10);

  out1 = linear(data, 784, 512, sqrt(2));
  out2 = tensor_relu(out1);
  out3 = linear(out2, 512, 10, 1);
  out4 = tensor_softmax(out3, -1);
  out5 = tensor_cross_entropy(out3, y);
  loss = tensor_mean(out5);

  d_out5 = tensor_copy(out4);
  for (size_t i = 0; i < 32; ++i) {
    d_out5->data[i * d_out5->stride[1] + (int32_t)y->data[i]] -= 1.0f;
  }

  tensor_print(y);
  tensor_print(d_out5);

  tensor_free(data);
  tensor_free(out1);
  tensor_free(out2);
  tensor_free(out3);
  tensor_free(out4);
  tensor_free(out5);
  tensor_free(d_out5);
  tensor_free(y);

  return 0;
}
