#ifndef TENSOR_H
#define TENSOR_H

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  size_t ndims;
  size_t *shape;
  size_t *stride;
  float *data;
  bool on_gpu;
} Tensor;

#ifdef __cplusplus
extern "C" {
#endif

Tensor *tensor_zeros(size_t *shape, size_t ndims);

Tensor *tensor_randn(size_t *shape, size_t ndims);

Tensor *tensor_randint(size_t *shape, size_t ndims, int32_t low, int32_t high);

Tensor *tensor_empty(size_t *shape, size_t ndims);

Tensor *tensor_empty_like(Tensor *t);

Tensor *tensor_copy(Tensor *t);

void kaiming_init(Tensor *t, float gain);

float tensor_sum(Tensor *t);

Tensor *tensor_sum_at(Tensor *t, int32_t dim);

Tensor *tensor_argmax_at(Tensor *t, int32_t dim);

float tensor_mean(Tensor *t);

Tensor *tensor_mean_at(Tensor *t, int32_t dim);

float tensor_std(Tensor *t);

Tensor *tensor_relu(Tensor *t);

Tensor *tensor_softmax(Tensor *t, int32_t dim);

Tensor *tensor_cross_entropy(Tensor *pred, Tensor *target);

void tensor_free(Tensor *t);

void tensor_reshape(Tensor *t, size_t *shape, size_t ndims);

void tensor_transpose(Tensor *t);

void tensor_print(Tensor *t);

void tensor_debug(Tensor *t);

size_t tensor_numel(Tensor *t);

void tensor_add_scaler(Tensor *t, float x);

void tensor_sub_scaler(Tensor *t, float x);

void tensor_mul_scaler(Tensor *t, float x);

void tensor_div_scaler(Tensor *t, float x);

Tensor *tensor_matadd(Tensor *a, Tensor *b);

Tensor *tensor_matmul(Tensor *a, Tensor *b);

bool tensor_allclose(Tensor *a, Tensor *b, float eps);

void save_tensor(FILE *f, Tensor *t);

Tensor *load_tensor(FILE *f);

void save_model(const char *path, Tensor **tensors, size_t n);

void load_model(const char *path, Tensor **tensors, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif

Tensor *tensor_to_gpu(Tensor *in);

Tensor *tensor_to_cpu(Tensor *in);

Tensor *tensor_copy_gpu(Tensor *in);

void tensor_free_gpu(Tensor *t);

Tensor *tensor_relu_gpu(Tensor *in);

void tensor_relu_backward_gpu(Tensor *a, Tensor *b);

Tensor *tensor_sum_at_gpu(Tensor *t, int32_t dim);

Tensor *tensor_argmax_at_gpu(Tensor *t, int32_t dim);

Tensor *tensor_matadd_gpu(Tensor *a, Tensor *b);

Tensor *tensor_matmul_gpu(Tensor *a, Tensor *b);

void tensor_add_scaler_gpu(Tensor *t, float x);

void tensor_sub_scaler_gpu(Tensor *t, float x);

void tensor_mul_scaler_gpu(Tensor *t, float x);

void tensor_div_scaler_gpu(Tensor *t, float x);

void tensor_update_at_gpu(Tensor *in, Tensor *idx, float x);

Tensor *tensor_cross_entropy_gpu(Tensor *pred, Tensor *target);

Tensor *tensor_softmax_gpu(Tensor *in, int32_t dim);

#endif
