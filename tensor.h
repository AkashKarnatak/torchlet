#ifndef TENSOR_H
#define TENSOR_H

#include <assert.h>
#include <math.h>
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
} Tensor;

Tensor *tensor_zeros(size_t *shape, size_t ndims);

Tensor *tensor_randn(size_t *shape, size_t ndims);

Tensor *tensor_like(Tensor *t);

void kaiming_init(Tensor *t, float gain);

float tensor_sum(Tensor *t);

Tensor *tensor_sum_at(Tensor *t, int32_t dim);

float tensor_mean(Tensor *t);

Tensor *tensor_mean_at(Tensor *t, int32_t dim);

float tensor_std(Tensor *t);

Tensor *tensor_relu(Tensor *t);

void tensor_free(Tensor *t);

void tensor_reshape(Tensor *t, size_t *shape, size_t ndims);

void tensor_print(Tensor *t);

void tensor_debug(Tensor *t);

size_t tensor_numel(Tensor *t);

Tensor *tensor_add(Tensor *a, Tensor *b);

Tensor *tensor_matmul(Tensor *a, Tensor *b);

#endif
