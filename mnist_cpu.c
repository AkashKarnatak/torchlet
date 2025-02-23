#include "mnist_utils.h"
#include "tensor.h"
#include <unistd.h>

typedef struct {
  float loss;
  float acc;
} Metrics;

void sgd_step(Tensor *tensor, Tensor *grad, float lr) {
  size_t numel;

  numel = tensor_numel(tensor);

  for (size_t i = 0; i < numel; ++i) {
    tensor->data[i] -= lr * grad->data[i];
  }
}

Metrics eval(Dataset eval_ds, Tensor **params, size_t n) {
  Tensor *w1, *b1, *w2, *b2;
  Tensor *out1, *out2, *out3, *out4, *out5, *out6, *out7;
  float loss, acc;
  size_t cnt;

  w1 = params[0], b1 = params[1], w2 = params[2], b2 = params[3];

  // test
  loss = 0.0f, cnt = 0, acc = 0;
  while (dataset_next(&eval_ds)) {
    out1 = tensor_matmul(eval_ds.data_t, w1);
    out2 = tensor_matadd(out1, b1);
    out3 = tensor_relu(out2);
    out4 = tensor_matmul(out3, w2);
    out5 = tensor_matadd(out4, b2);
    out6 = tensor_argmax_at(out5, -1);
    out7 = tensor_cross_entropy(out5, eval_ds.label_t);
    loss += tensor_mean(out7);
    for (size_t i = 0; i < eval_ds.label_t->shape[0]; ++i) {
      if (eval_ds.label_t->data[i] == out6->data[i]) {
        ++acc;
      }
    }

    tensor_free(out1);
    tensor_free(out2);
    tensor_free(out3);
    tensor_free(out4);
    tensor_free(out5);
    tensor_free(out6);
    tensor_free(out7);
    ++cnt;
  }

  return (Metrics){loss / cnt, acc / eval_ds.n};
}

void train(Dataset train_ds, Dataset val_ds, Tensor **params, size_t n_params) {
  Tensor *w1, *b1, *w2, *b2;
  Tensor *out1, *out2, *out3, *out4, *out5, *out6;
  Tensor *d_out1, *d_out2, *d_out3, *d_out4, *d_out5;
  Tensor *d_w1, *d_w2, *d_b1, *d_b2;
  float loss;
  size_t numel, steps;

  w1 = params[0], b1 = params[1], w2 = params[2], b2 = params[3];

  steps = 0;
  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    while (dataset_next(&train_ds)) {
      // forward pass
      out1 = tensor_matmul(train_ds.data_t, w1);
      out2 = tensor_matadd(out1, b1);
      out3 = tensor_relu(out2);
      out4 = tensor_matmul(out3, w2);
      out5 = tensor_matadd(out4, b2);
      out6 = tensor_cross_entropy(out5, train_ds.label_t);
      loss = tensor_mean(out6);

      // backward pass
      d_out5 = tensor_softmax(out5, -1);
      for (size_t row = 0; row < train_ds.batch_len; ++row) {
        d_out5->data[row * d_out5->stride[1] +
                     (int32_t)train_ds.label_t->data[row]] -= 1.0f;
      }
      tensor_div_scaler(d_out5, train_ds.label_t->shape[0]);

      d_out4 = tensor_copy(d_out5);

      d_b2 = tensor_sum_at(d_out5, 0);

      tensor_transpose(out3);
      d_w2 = tensor_matmul(out3, d_out4);

      tensor_transpose(w2);
      d_out3 = tensor_matmul(d_out4, w2);
      tensor_transpose(w2);

      d_out2 = tensor_copy(d_out3);
      numel = tensor_numel(out3);
      for (size_t i = 0; i < numel; ++i) {
        if (out3->data[i] <= 0) {
          d_out2->data[i] = 0;
        }
      }

      d_out1 = tensor_copy(d_out2);

      d_b1 = tensor_sum_at(d_out2, 0);

      tensor_transpose(train_ds.data_t);
      d_w1 = tensor_matmul(train_ds.data_t, d_out1);
      tensor_transpose(train_ds.data_t);

      // update
      sgd_step(w1, d_w1, 1e-2);
      sgd_step(w2, d_w2, 1e-2);
      sgd_step(b1, d_b1, 1e-2);
      sgd_step(b2, d_b2, 1e-2);

      ++steps;

      printf("Epoch: %lu, Step: %lu, Loss: %f\n", epoch, steps, loss);

      tensor_free(out1);
      tensor_free(out2);
      tensor_free(out3);
      tensor_free(out4);
      tensor_free(out5);
      tensor_free(out6);
      tensor_free(d_out5);
      tensor_free(d_out4);
      tensor_free(d_out3);
      tensor_free(d_out2);
      tensor_free(d_out1);
      tensor_free(d_w2);
      tensor_free(d_b2);
      tensor_free(d_w1);
      tensor_free(d_b1);

      if (steps % 100 == 0) {
        // validate
        Metrics m = eval(val_ds, params, n_params);
        printf("-----------------------------------\n");
        printf("Epoch: %lu, Validation Loss: %f, Validation Accuracy: %f\n",
               epoch, m.loss, m.acc);
        printf("-----------------------------------\n");
      }
      dataset_reset(&val_ds);
    }
    dataset_reset(&train_ds);
  }
}

int main() {
  size_t batch_size;
  Dataset train_ds, val_ds, test_ds;

  Tensor *w1, *w2, *b1, *b2;
  const size_t n_params = 4;
  Tensor *params[n_params];

  batch_size = 128;
  train_ds = dataset_load("../../Downloads/MNIST_CSV/mnist_train_data.bin",
                          "../../Downloads/MNIST_CSV/mnist_train_label.bin",
                          60000, 784, batch_size);
  val_ds = dataset_load("../../Downloads/MNIST_CSV/mnist_val_data.bin",
                        "../../Downloads/MNIST_CSV/mnist_val_label.bin", 5000,
                        784, batch_size);
  test_ds = dataset_load("../../Downloads/MNIST_CSV/mnist_test_data.bin",
                         "../../Downloads/MNIST_CSV/mnist_test_label.bin", 5000,
                         784, batch_size);

  if (access("model.bin", F_OK) == 0) {
    load_model("model.bin", params, n_params);
  } else {
    w1 = tensor_zeros((size_t[]){784, 512}, 2);
    kaiming_init(w1, sqrt(2));
    b1 = tensor_zeros((size_t[]){512}, 1);

    w2 = tensor_zeros((size_t[]){512, 10}, 2);
    kaiming_init(w2, 1);
    b2 = tensor_zeros((size_t[]){10}, 1);

    params[0] = w1, params[1] = b1, params[2] = w2, params[3] = b2;

    train(train_ds, val_ds, params, n_params);
    save_model("model.bin", params, n_params);
  }

  Metrics m = eval(test_ds, params, n_params);
  printf("-----------------------------------\n");
  printf("Test Loss: %f Test Accuracy: %f\n", m.loss, m.acc);
  printf("-----------------------------------\n");

  for (size_t i = 0; i < n_params; ++i) {
    tensor_free(params[i]);
  }
  dataset_free(&train_ds);
  dataset_free(&val_ds);
  dataset_free(&test_ds);
}
