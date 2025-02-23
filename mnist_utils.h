#ifndef MNIST_UTILS_H
#define MNIST_UTILS_H

#include "tensor.h"

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

#ifdef __cplusplus
extern "C" {
#endif

Dataset dataset_init(float *data, float *label, size_t n, size_t row_size,
                     size_t batch_size);

void dataset_free(Dataset *ds);

Dataset dataset_load(const char *data_path, const char *label_path, size_t n,
                     size_t row_size, size_t batch_size);

bool dataset_next(Dataset *ds);

void dataset_reset(Dataset *ds);

#ifdef __cplusplus
}
#endif

#endif
