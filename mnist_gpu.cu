#include "mnist_utils.h"
#include "tensor.h"
#include <unistd.h>

typedef struct {
  float loss;
  float acc;
} Metrics;

__global__ void sgd_step_kernel(float *t_data, float *grad_data, float lr,
                                size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  t_data[i] -= lr * grad_data[i];
}

void sgd_step(Tensor *tensor, Tensor *grad, float lr) {
  size_t numel;

  numel = tensor_numel(tensor);

  size_t numThreads = 1024;
  size_t numBlocks = (numel + numThreads - 1) / numThreads;

  sgd_step_kernel<<<numBlocks, numThreads>>>(tensor->data, grad->data, lr,
                                             numel);
}

Metrics eval(Dataset eval_ds, Tensor **params, size_t n) {
  Tensor *w1, *b1, *w2, *b2;
  Tensor *data_gpu, *label_gpu;
  Tensor *out1, *out2, *out3, *out4, *out5, *out6, *out6_cpu, *out7, *out7_cpu;
  float loss, acc;
  size_t cnt;

  w1 = params[0], b1 = params[1], w2 = params[2], b2 = params[3];

  // test
  loss = 0.0f, cnt = 0, acc = 0;
  while (dataset_next(&eval_ds)) {
    data_gpu = tensor_to_gpu(eval_ds.data_t);
    label_gpu = tensor_to_gpu(eval_ds.label_t);
    out1 = tensor_matmul_gpu(data_gpu, w1);
    out2 = tensor_matadd_gpu(out1, b1);
    out3 = tensor_relu_gpu(out2);
    out4 = tensor_matmul_gpu(out3, w2);
    out5 = tensor_matadd_gpu(out4, b2);
    out6 = tensor_argmax_at_gpu(out5, -1);
    out6_cpu = tensor_to_cpu(out6);
    out7 = tensor_cross_entropy_gpu(out5, label_gpu);
    out7_cpu = tensor_to_cpu(out7);
    loss += tensor_mean(out7_cpu);
    for (size_t i = 0; i < eval_ds.label_t->shape[0]; ++i) {
      if (eval_ds.label_t->data[i] == out6_cpu->data[i]) {
        ++acc;
      }
    }

    tensor_free_gpu(out1);
    tensor_free_gpu(out2);
    tensor_free_gpu(out3);
    tensor_free_gpu(out4);
    tensor_free_gpu(out5);
    tensor_free_gpu(out6);
    tensor_free_gpu(out6_cpu);
    tensor_free_gpu(out7);
    tensor_free_gpu(out7_cpu);
    ++cnt;
  }

  return (Metrics){loss / cnt, acc / eval_ds.n};
}

void train_gpu(Dataset train_ds, Dataset val_ds, Tensor **params,
               size_t n_params) {
  Tensor *w1, *b1, *w2, *b2;
  Tensor *data_gpu, *label_gpu;
  Tensor *out1, *out2, *out3, *out4, *out5, *out6, *out6_cpu;
  Tensor *d_out1, *d_out2, *d_out3, *d_out4, *d_out5;
  Tensor *d_w1, *d_w2, *d_b1, *d_b2;
  float loss;
  size_t steps;

  w1 = params[0], b1 = params[1], w2 = params[2], b2 = params[3];

  steps = 0;
  for (size_t epoch = 1; epoch <= 20; ++epoch) {
    while (dataset_next(&train_ds)) {
      // forward pass
      data_gpu = tensor_to_gpu(train_ds.data_t);
      label_gpu = tensor_to_gpu(train_ds.label_t);
      out1 = tensor_matmul_gpu(data_gpu, w1);
      out2 = tensor_matadd_gpu(out1, b1);
      out3 = tensor_relu_gpu(out2);
      out4 = tensor_matmul_gpu(out3, w2);
      out5 = tensor_matadd_gpu(out4, b2);
      out6 = tensor_cross_entropy_gpu(out5, label_gpu);
      out6_cpu = tensor_to_cpu(out6);
      loss = tensor_mean(out6_cpu);

      // backward pass
      d_out5 = tensor_softmax_gpu(out5, -1);
      tensor_update_at_gpu(d_out5, label_gpu, -1);
      tensor_div_scaler_gpu(d_out5, train_ds.label_t->shape[0]);

      d_out4 = tensor_copy_gpu(d_out5);

      d_b2 = tensor_sum_at_gpu(d_out5, 0);

      tensor_transpose(out3);
      d_w2 = tensor_matmul_gpu(out3, d_out4);

      tensor_transpose(w2);
      d_out3 = tensor_matmul_gpu(d_out4, w2);
      tensor_transpose(w2);

      d_out2 = tensor_copy_gpu(d_out3);
      tensor_relu_backward_gpu(d_out2, out3);

      d_out1 = tensor_copy_gpu(d_out2);

      d_b1 = tensor_sum_at_gpu(d_out2, 0);

      tensor_transpose(data_gpu);
      d_w1 = tensor_matmul_gpu(data_gpu, d_out1);
      tensor_transpose(data_gpu);

      // update
      sgd_step(w1, d_w1, 1e-2);
      sgd_step(w2, d_w2, 1e-2);
      sgd_step(b1, d_b1, 1e-2);
      sgd_step(b2, d_b2, 1e-2);
      cudaDeviceSynchronize();

      ++steps;

      printf("Epoch: %lu, Step: %lu, Loss: %f\n", epoch, steps, loss);

      tensor_free_gpu(out1);
      tensor_free_gpu(out2);
      tensor_free_gpu(out3);
      tensor_free_gpu(out4);
      tensor_free_gpu(out5);
      tensor_free_gpu(out6);
      tensor_free(out6_cpu);
      tensor_free_gpu(d_out5);
      tensor_free_gpu(d_out4);
      tensor_free_gpu(d_out3);
      tensor_free_gpu(d_out2);
      tensor_free_gpu(d_out1);
      tensor_free_gpu(d_w2);
      tensor_free_gpu(d_b2);
      tensor_free_gpu(d_w1);
      tensor_free_gpu(d_b1);

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
  Tensor *w1_gpu, *w2_gpu, *b1_gpu, *b2_gpu;
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

  if (access("model_gpu.bin", F_OK) == 0) {
    load_model("model_gpu.bin", params, n_params);
  } else {
    w1 = tensor_zeros((size_t[]){784, 512}, 2);
    kaiming_init(w1, sqrt(2));
    b1 = tensor_zeros((size_t[]){512}, 1);

    w2 = tensor_zeros((size_t[]){512, 10}, 2);
    kaiming_init(w2, 1);
    b2 = tensor_zeros((size_t[]){10}, 1);

    w1_gpu = tensor_to_gpu(w1);
    b1_gpu = tensor_to_gpu(b1);
    w2_gpu = tensor_to_gpu(w2);
    b2_gpu = tensor_to_gpu(b2);
    params[0] = w1_gpu, params[1] = b1_gpu, params[2] = w2_gpu,
    params[3] = b2_gpu;

    tensor_free(w1);
    tensor_free(b1);
    tensor_free(w2);
    tensor_free(b2);

    // train data on GPU
    train_gpu(train_ds, val_ds, params, n_params);

    w1 = tensor_to_cpu(w1_gpu);
    b1 = tensor_to_cpu(b1_gpu);
    w2 = tensor_to_cpu(w2_gpu);
    b2 = tensor_to_cpu(b2_gpu);
    params[0] = w1, params[1] = b1, params[2] = w2, params[3] = b2;

    tensor_free_gpu(w1_gpu);
    tensor_free_gpu(b1_gpu);
    tensor_free_gpu(w2_gpu);
    tensor_free_gpu(b2_gpu);

    save_model("model_gpu.bin", params, n_params);
  }

  for (size_t i = 0; i < n_params; ++i) {
    Tensor *t = params[i];
    params[i] = tensor_to_gpu(params[i]);
    tensor_free(t);
  }
  Metrics m = eval(test_ds, params, n_params);
  printf("-----------------------------------\n");
  printf("Test Loss: %f Test Accuracy: %f\n", m.loss, m.acc);
  printf("-----------------------------------\n");

  for (size_t i = 0; i < n_params; ++i) {
    tensor_free_gpu(params[i]);
  }
  dataset_free(&train_ds);
  dataset_free(&val_ds);
  dataset_free(&test_ds);
}
