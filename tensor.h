#ifndef TENSOR_H
#define TENSOR_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
  size_t ndims;
  size_t *shape;
  size_t *stride;
  float *data;
} Tensor;

Tensor *tensor_init(size_t *shape, size_t ndims);

Tensor *tensor_like(Tensor *t);

void tensor_free(Tensor *t);

void tensor_reshape(Tensor *t, size_t *shape, size_t ndims);

void tensor_print(Tensor *t);

void tensor_debug(Tensor *t);

size_t tensor_numel(Tensor *t);

Tensor *tensor_matmul(Tensor *a, Tensor *b);

#endif
