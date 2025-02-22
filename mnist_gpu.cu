#include "tensor.h"
#include <unistd.h>

typedef struct {
  float *data;
  float *label;
  size_t n;
  size_t row_size;
  size_t batch_len;
  size_t _batch_size;
  size_t _offset;
  Tensor *data_t;
  Tensor *label_t;
} Dataset;

Dataset dataset_init(float *data, float *label, size_t n, size_t row_size,
                     size_t batch_size) {
  Tensor *data_t, *label_t;
  Dataset ds;

  data_t = (Tensor *)malloc(sizeof(Tensor));
  assert(data_t != NULL);
  label_t = (Tensor *)malloc(sizeof(Tensor));
  assert(label_t != NULL);

  data_t->ndims = 2, label_t->ndims = 1;
  data_t->on_gpu = false, label_t->on_gpu = false;

  data_t->shape = (size_t *)malloc(data_t->ndims * sizeof(size_t));
  assert(data_t->shape != NULL);
  label_t->shape = (size_t *)malloc(label_t->ndims * sizeof(size_t));
  assert(label_t->shape != NULL);

  data_t->stride = (size_t *)malloc(data_t->ndims * sizeof(size_t));
  assert(data_t->stride != NULL);
  label_t->stride = (size_t *)malloc(label_t->ndims * sizeof(size_t));
  assert(label_t->stride != NULL);

  data_t->shape[0] = row_size, data_t->shape[1] = batch_size;
  data_t->stride[0] = 1, data_t->stride[1] = row_size;
  label_t->shape[0] = batch_size;
  label_t->stride[0] = 1;

  data_t->data = data, label_t->data = label;

  ds.data = data, ds.label = label, ds.n = n, ds.row_size = row_size,
  ds.batch_len = batch_size, ds._batch_size = batch_size, ds._offset = 0;
  ds.data_t = data_t, ds.label_t = label_t;

  return ds;
}

void dataset_free(Dataset *ds) {
  free(ds->data_t->shape);
  free(ds->data_t->stride);
  free(ds->data_t);
  free(ds->label_t->shape);
  free(ds->label_t->stride);
  free(ds->label_t);
  free(ds->data);
  free(ds->label);
}

Dataset dataset_load(const char *data_path, const char *label_path, size_t n,
                     size_t row_size, size_t batch_size) {
  FILE *f;
  float *data, *label;
  Dataset ds;

  f = fopen(data_path, "r");
  assert(f != NULL);
  data = (float *)malloc(n * row_size * sizeof(float));
  assert(data != NULL);
  assert(fread(data, sizeof(float), n * row_size, f) == n * row_size);
  assert(fclose(f) == 0);

  f = fopen(label_path, "r");
  assert(f != NULL);
  label = (float *)malloc(n * sizeof(float));
  assert(label != NULL);
  assert(fread(label, sizeof(float), n, f) == n);
  assert(fclose(f) == 0);

  ds = dataset_init(data, label, n, row_size, batch_size);

  return ds;
}

bool dataset_next(Dataset *ds) {
  if (ds->_offset >= ds->n) {
    return false;
  }

  ds->data_t->data = ds->data + ds->_offset * ds->row_size;
  ds->label_t->data = ds->label + ds->_offset;
  ds->batch_len = ds->_batch_size;
  if (ds->n - ds->_offset < ds->_batch_size) {
    ds->data_t->shape[1] = ds->n - ds->_offset;
    ds->label_t->shape[0] = ds->n - ds->_offset;
    ds->batch_len = ds->n - ds->_offset;
  }
  ds->_offset += ds->_batch_size;

  return true;
}

void dataset_reset(Dataset *ds) {
  ds->data_t->data = ds->data;
  ds->label_t->data = ds->label;
  ds->data_t->shape[1] = ds->_batch_size;
  ds->label_t->shape[0] = ds->_batch_size;
  ds->_offset = 0;
  ds->batch_len = ds->_batch_size;
}

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

void save_tensor(FILE *f, Tensor *t) {
  size_t numel = tensor_numel(t);
  fwrite(&t->ndims, sizeof(size_t), 1, f);
  fwrite(&t->on_gpu, sizeof(bool), 1, f);
  fwrite(t->shape, sizeof(size_t), t->ndims, f);
  fwrite(t->stride, sizeof(size_t), t->ndims, f);
  fwrite(t->data, sizeof(float), numel, f);
}

Tensor *load_tensor(FILE *f) {
  Tensor *t;
  size_t numel;

  t = (Tensor *)malloc(sizeof(Tensor));
  assert(t != NULL);
  assert(fread(&t->ndims, sizeof(size_t), 1, f) == 1);
  assert(fread(&t->on_gpu, sizeof(bool), 1, f) == 1);

  t->shape = (size_t *)malloc(t->ndims * sizeof(size_t));
  assert(t->shape != NULL);
  assert(fread(t->shape, sizeof(size_t), t->ndims, f) == t->ndims);

  t->stride = (size_t *)malloc(t->ndims * sizeof(size_t));
  assert(t->stride != NULL);
  assert(fread(t->stride, sizeof(size_t), t->ndims, f) == t->ndims);

  numel = tensor_numel(t);
  t->data = (float *)malloc(numel * sizeof(float));
  assert(t->data != NULL);
  assert(fread(t->data, sizeof(float), numel, f) == numel);

  return t;
}

void save_model(const char *path, Tensor **tensors, size_t n) {
  FILE *f;

  f = fopen(path, "w");
  assert(f != NULL);
  for (size_t i = 0; i < n; ++i) {
    save_tensor(f, tensors[i]);
  }
  assert(fclose(f) == 0);
}

void load_model(const char *path, Tensor **tensors, size_t n) {
  FILE *f;

  f = fopen(path, "r");
  assert(f != NULL);
  for (size_t i = 0; i < n; ++i) {
    tensors[i] = load_tensor(f);
  }
  assert(fclose(f) == 0);
}

float eval(Dataset eval_ds, Tensor **params, size_t n) {
  Tensor *w1, *b1, *w2, *b2;
  Tensor *data_gpu, *label_gpu;
  Tensor *out1, *out2, *out3, *out4, *out5, *out6, *out6_cpu;
  float loss;
  size_t cnt;

  w1 = params[0], b1 = params[1], w2 = params[2], b2 = params[3];

  // test
  loss = 0.0f;
  cnt = 0;
  while (dataset_next(&eval_ds)) {
    data_gpu = tensor_to_gpu(eval_ds.data_t);
    label_gpu = tensor_to_gpu(eval_ds.label_t);
    out1 = tensor_matmul_gpu(data_gpu, w1);
    out2 = tensor_matadd_gpu(out1, b1);
    out3 = tensor_relu_gpu(out2);
    out4 = tensor_matmul_gpu(out3, w2);
    out5 = tensor_matadd_gpu(out4, b2);
    out6 = tensor_cross_entropy_gpu(out5, label_gpu);
    out6_cpu = tensor_to_cpu(out6);
    loss += tensor_mean(out6_cpu);

    tensor_free_gpu(out1);
    tensor_free_gpu(out2);
    tensor_free_gpu(out3);
    tensor_free_gpu(out4);
    tensor_free_gpu(out5);
    tensor_free_gpu(out6);
    ++cnt;
  }

  return loss / cnt;
}

void train_gpu(Dataset train_ds, Dataset val_ds, Tensor **params, size_t n) {
  Tensor *w1, *b1, *w2, *b2;
  Tensor *data_gpu, *label_gpu;
  Tensor *out1, *out2, *out3, *out4, *out5, *out6, *out6_cpu;
  Tensor *d_out1, *d_out2, *d_out3, *d_out4, *d_out5;
  Tensor *d_w1, *d_w2, *d_b1, *d_b2;
  float loss, val_loss;
  size_t steps;

  w1 = params[0], b1 = params[1], w2 = params[2], b2 = params[3];

  steps = 0;
  for (size_t epoch = 1; epoch <= 1; ++epoch) {
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
      sgd_step(w1, d_w1, 8e-4);
      sgd_step(w2, d_w2, 8e-4);
      sgd_step(b1, d_b1, 8e-4);
      sgd_step(b2, d_b2, 8e-4);
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
        val_loss = eval(val_ds, params, n);
        printf("-----------------------------------\n");
        printf("Epoch: %lu, Validation Loss: %f\n", epoch, val_loss);
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
  float test_loss;
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
  test_loss = eval(test_ds, params, 4);
  printf("-----------------------------------\n");
  printf("Test Loss: %f\n", test_loss);
  printf("-----------------------------------\n");

  for (size_t i = 0; i < n_params; ++i) {
    tensor_free_gpu(params[i]);
  }
  dataset_free(&train_ds);
  dataset_free(&val_ds);
  dataset_free(&test_ds);
}
