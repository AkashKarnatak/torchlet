#include "tensor.h"

int main() {
  Tensor *data, *y;
  Tensor *w1, *w2, *b1, *b2;
  Tensor *out1, *out2, *out3, *out4, *out5, *out6;
  Tensor *d_out1, *d_out2, *d_out3, *d_out4, *d_out5;
  Tensor *d_w1, *d_w2, *d_b1, *d_b2;
  float loss;

  data = tensor_randn((size_t[]){32, 784}, 2);
  y = tensor_randint((size_t[]){32}, 1, 0, 10);

  w1 = tensor_zeros((size_t[]){784, 512}, 2);
  kaiming_init(w1, sqrt(2));
  b1 = tensor_zeros((size_t[]){32, 512}, 2);

  w2 = tensor_zeros((size_t[]){512, 10}, 2);
  kaiming_init(w2, 1);
  b2 = tensor_zeros((size_t[]){32, 10}, 2);

  out1 = tensor_matmul(data, w1);
  out2 = tensor_matadd(out1, b1);
  out3 = tensor_relu(out2);
  out4 = tensor_matmul(out3, w2);
  out5 = tensor_matadd(out4, b2);
  out6 = tensor_cross_entropy(out5, y);
  loss = tensor_mean(out6);

  d_out5 = tensor_softmax(out5, -1);
  for (size_t i = 0; i < 32; ++i) {
    d_out5->data[i * d_out5->stride[1] + (int32_t)y->data[i]] -= 1.0f;
  }
  tensor_div_scaler(d_out5, y->shape[0]);

  d_out4 = tensor_copy(d_out5);

  d_b2 = tensor_copy(d_out5);

  tensor_transpose(out3);
  tensor_debug(out3);
  d_w2 = tensor_matmul(out3, d_out4);

  tensor_transpose(w2);
  d_out3 = tensor_matmul(d_out4, w2);

  tensor_print(d_out3);

  tensor_free(data);
  tensor_free(y);
  tensor_free(w1);
  tensor_free(b1);
  tensor_free(w2);
  tensor_free(b2);
  tensor_free(out1);
  tensor_free(out2);
  tensor_free(out3);
  tensor_free(out4);
  tensor_free(out5);
  tensor_free(out6);
  tensor_free(d_out5);
  tensor_free(d_out4);
  tensor_free(d_out3);
  /* tensor_free(d_out2); */
  /* tensor_free(d_out1); */
  tensor_free(d_w2);
  tensor_free(d_b2);
  /* tensor_free(d_w1); */
  /* tensor_free(d_b1); */

  return 0;
}
