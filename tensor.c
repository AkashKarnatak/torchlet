#include "tensor.h"

// NOTE: contrary to the usual convention lower index of tensor shape
// corresponds to the lower dimension
Tensor *_tensor_init(size_t *shape, size_t ndims) {
  Tensor *t;
  size_t numel;

  t = (Tensor *)malloc(sizeof(Tensor));
  assert(t != NULL);

  t->ndims = ndims;

  t->shape = (size_t *)malloc(ndims * sizeof(size_t));
  assert(t->shape != NULL);

  t->stride = (size_t *)malloc(ndims * sizeof(size_t));
  assert(t->stride != NULL);

  numel = 1;
  for (size_t i = 0; i < ndims; ++i) {
    t->shape[i] = shape[i];
    t->stride[i] = numel;
    numel *= shape[i];
  }

  t->data = (float *)malloc(numel * sizeof(float));
  assert(t->data != NULL);

  return t;
}

Tensor *tensor_init(size_t *shape, size_t ndims) {
  size_t shape_r[ndims];
  for (int32_t i = 0; i < (ndims + 1) / 2; ++i) {
    shape_r[i] = shape[ndims - 1 - i];
    shape_r[ndims - 1 - i] = shape[i];
  }
  return _tensor_init(shape_r, ndims);
}

Tensor *tensor_like(Tensor *t) { return _tensor_init(t->shape, t->ndims); }

void tensor_free(Tensor *t) {
  free(t->shape);
  free(t->stride);
  free(t->data);
  free(t);
}

void _tensor_reshape(Tensor *t, size_t *shape, size_t ndims) {
  size_t numel;

  numel = 1;
  for (size_t i = 0; i < ndims; ++i) {
    numel *= shape[i];
  }
  assert(tensor_numel(t) == numel);

  t->ndims = ndims;

  t->shape = (size_t *)realloc(t->shape, ndims * sizeof(size_t));
  assert(t->shape != NULL);

  t->stride = (size_t *)realloc(t->stride, ndims * sizeof(size_t));
  assert(t->stride != NULL);

  numel = 1;
  for (size_t i = 0; i < ndims; ++i) {
    t->shape[i] = shape[i];
    t->stride[i] = numel;
    numel *= shape[i];
  }
}

void tensor_reshape(Tensor *t, size_t *shape, size_t ndims) {
  size_t shape_r[ndims];
  for (int32_t i = 0; i < (ndims + 1) / 2; ++i) {
    shape_r[i] = shape[ndims - 1 - i];
    shape_r[ndims - 1 - i] = shape[i];
  }
  return _tensor_reshape(t, shape_r, ndims);
}

void tensor_print(Tensor *t) {
  size_t batch_size;

  batch_size = 1;
  for (size_t i = 2; i < t->ndims; ++i) {
    batch_size *= t->shape[i];
  }

  size_t t_stride_2 = t->ndims > 2 ? t->stride[2] : 1;

  printf("[\n");
  for (size_t batch = 0; batch < batch_size; ++batch) {
    for (size_t row = 0; row < t->shape[1]; ++row) {
      printf(" [\n ");
      for (size_t col = 0; col < t->shape[0]; ++col) {
        printf(" %.5f,",
               t->data[batch * t_stride_2 + row * t->stride[1] + col]);
      }
      printf("\n ],\n");
    }
  }
  printf("]\n");
}

void tensor_debug(Tensor *t) {
  size_t batch_size;

  batch_size = 1;
  for (size_t i = 2; i < t->ndims; ++i) {
    batch_size *= t->shape[i];
  }

  printf("Tensor(shape=[");
  printf("%lu", t->shape[t->ndims - 1]);
  for (size_t i = 1; i < t->ndims; ++i) {
    printf(", %lu", t->shape[t->ndims - 1 - i]);
  }

  printf("], stride=[");
  printf("%lu", t->stride[t->ndims - 1]);
  for (size_t i = 1; i < t->ndims; ++i) {
    printf(", %lu", t->stride[t->ndims - 1 - i]);
  }
  printf("])\n");
}

size_t tensor_numel(Tensor *t) {
  size_t numel = 1;
  for (size_t i = 0; i < t->ndims; ++i) {
    numel *= t->shape[i];
  }
  return numel;
}

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
  assert(a->ndims > 1 && b->ndims > 1);

  // NOTE: does not support broadcasting as of now
  assert(a->ndims == b->ndims);

  size_t batch_size = 1;
  size_t ndims = a->ndims;

  // check whether a and b are compatible
  assert(a->shape[0] == b->shape[1]);
  for (size_t i = 2; i < ndims; ++i) {
    assert(a->shape[i] == b->shape[i]);
    batch_size *= a->shape[i] != 1 ? a->shape[i] : b->shape[i];
  }

  size_t c_shape[ndims];
  c_shape[0] = b->shape[0];
  c_shape[1] = a->shape[1];
  for (size_t i = 2; i < ndims; ++i) {
    c_shape[i] = a->shape[i];
  }

  Tensor *c = _tensor_init(c_shape, ndims);

  size_t a_stride_2 = ndims > 2 ? a->stride[2] : 1;
  size_t b_stride_2 = ndims > 2 ? b->stride[2] : 1;
  size_t c_stride_2 = ndims > 2 ? c->stride[2] : 1;

  for (size_t batch = 0; batch < batch_size; ++batch) {
    for (size_t row = 0; row < c->shape[1]; ++row) {
      for (size_t col = 0; col < c->shape[0]; ++col) {
        float sum = 0.0f;
        for (size_t k = 0; k < a->shape[0]; ++k) {
          sum += a->data[batch * a_stride_2 + row * a->stride[1] + k] *
                 b->data[batch * b_stride_2 + k * b->stride[1] + col];
        }
        c->data[batch * c_stride_2 + row * c->stride[1] + col] = sum;
      }
    }
  }

  return c;
}

int main() {
  size_t a_shape[] = {5, 7};
  size_t b_shape[] = {7, 3};

  Tensor *a = tensor_init(a_shape, sizeof(a_shape) / sizeof(size_t));
  Tensor *b = tensor_init(b_shape, sizeof(b_shape) / sizeof(size_t));

  size_t a_numel = tensor_numel(a);
  size_t b_numel = tensor_numel(b);

  for (size_t i = 0; i < a_numel; ++i) {
    a->data[i] = (float)rand() / RAND_MAX;
  }

  for (size_t i = 0; i < b_numel; ++i) {
    b->data[i] = (float)rand() / RAND_MAX;
  }

  Tensor *c = tensor_matmul(a, b);

  tensor_print(a);
  printf("--------\n");
  tensor_print(b);
  printf("--------\n");
  tensor_print(c);
  printf("--------\n");

  tensor_debug(a);
  tensor_debug(b);
  tensor_debug(c);

  tensor_free(a);
  tensor_free(b);
  tensor_free(c);
}
