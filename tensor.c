#include "tensor.h"
#include <omp.h>

float randn() {
  float u1, u2;

  u1 = (float)rand() / RAND_MAX;
  u2 = (float)rand() / RAND_MAX;

  // box-muller transformation
  float z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

  return z;
}

int32_t unif(int32_t low, int32_t high) {
  int32_t diff = high - low;
  return low + rand() % diff;
}

// NOTE: contrary to the usual convention lower index of tensor shape
// corresponds to the lower dimension
Tensor *_tensor_empty(size_t *shape, size_t ndims) {
  Tensor *t;
  size_t numel;

  t = (Tensor *)malloc(sizeof(Tensor));
  assert(t != NULL);

  t->ndims = ndims;
  t->on_gpu = false;

  t->shape = (size_t *)malloc(ndims * sizeof(size_t));
  assert(t->shape != NULL);

  t->stride = (size_t *)malloc(ndims * sizeof(size_t));
  assert(t->stride != NULL);

  numel = 1;
  for (size_t i = 0; i < ndims; ++i) {
    assert(shape[i] != 0);
    t->shape[i] = shape[i];
    t->stride[i] = numel;
    numel *= shape[i];
  }

  t->data = (float *)malloc(numel * sizeof(float));
  assert(t->data != NULL);

  return t;
}

Tensor *tensor_empty(size_t *shape, size_t ndims) {
  size_t shape_r[ndims];
  for (int32_t i = 0; i < (ndims + 1) / 2; ++i) {
    shape_r[i] = shape[ndims - 1 - i];
    shape_r[ndims - 1 - i] = shape[i];
  }
  return _tensor_empty(shape_r, ndims);
}

Tensor *tensor_empty_like(Tensor *t) {
  return _tensor_empty(t->shape, t->ndims);
}

Tensor *tensor_copy(Tensor *in) {
  Tensor *out;
  size_t numel;

  out = tensor_empty_like(in);
  for (size_t i = 0; i < in->ndims; ++i) {
    out->shape[i] = in->shape[i];
    out->stride[i] = in->stride[i];
  }
  numel = tensor_numel(in);
  memcpy(out->data, in->data, numel * sizeof(float));

  return out;
}

Tensor *tensor_zeros(size_t *shape, size_t ndims) {
  Tensor *t;

  t = tensor_empty(shape, ndims);
  memset(t->data, 0, tensor_numel(t) * sizeof(float));

  return t;
}

void kaiming_init(Tensor *t, float gain) {
  size_t numel;

  assert(t->ndims == 2);

  numel = tensor_numel(t);
  for (size_t i = 0; i < numel; ++i) {
    t->data[i] = randn() * gain / sqrt(t->shape[1]);
  }
}

Tensor *tensor_randn(size_t *shape, size_t ndims) {
  size_t numel;
  Tensor *t;

  t = tensor_empty(shape, ndims);

  numel = tensor_numel(t);
  for (size_t i = 0; i < numel; ++i) {
    t->data[i] = randn();
  }

  return t;
}

Tensor *tensor_randint(size_t *shape, size_t ndims, int32_t low, int32_t high) {
  size_t numel;
  Tensor *t;

  t = tensor_empty(shape, ndims);

  numel = tensor_numel(t);
  for (size_t i = 0; i < numel; ++i) {
    t->data[i] = unif(low, high);
  }

  return t;
}

float tensor_sum(Tensor *t) {
  size_t numel;
  float sum;

  numel = tensor_numel(t);
  sum = 0.0f;
  for (size_t i = 0; i < numel; ++i) {
    sum += t->data[i];
  }

  return sum;
}

Tensor *tensor_sum_at(Tensor *t, int32_t dim) {
  size_t outer, inner, outer_stride, inner_stride;
  size_t out_dims;
  Tensor *out;

  assert(dim == -1 || (dim >= 0 && dim < (int32_t)t->ndims));

  if (t->ndims == 1) {
    out = _tensor_empty((size_t[]){1}, 1);
    out->data[0] = tensor_sum(t);
    return out;
  }

  if (dim == -1) {
    dim = 0;
  } else {
    dim = t->ndims - 1 - dim;
  }

  out_dims = t->ndims - 1;
  size_t out_shape[out_dims];
  for (size_t i = 0; i < dim; ++i) {
    out_shape[i] = t->shape[i];
  }
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    out_shape[i - 1] = t->shape[i];
  }
  out = _tensor_empty(out_shape, out_dims);

  inner = 1;
  for (size_t i = 0; i < dim; ++i) {
    inner *= t->shape[i];
  }

  outer = 1;
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    outer *= t->shape[i];
  }

  inner_stride = dim - 1 >= 0 ? 1 : 0;
  outer_stride = dim + 1 < t->ndims ? t->stride[dim + 1] : 0;

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < outer; ++i) {
    for (size_t j = 0; j < inner; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < t->shape[dim]; ++k) {
        sum +=
            t->data[i * outer_stride + k * t->stride[dim] + j * inner_stride];
      }
      out->data[i * inner + j] = sum;
    }
  }

  return out;
}

Tensor *tensor_argmax_at(Tensor *t, int32_t dim) {
  size_t outer, inner, outer_stride, inner_stride;
  size_t out_dims;
  Tensor *out;

  assert(dim == -1 || (dim >= 0 && dim < (int32_t)t->ndims));

  if (t->ndims == 1) {
    out = _tensor_empty((size_t[]){1}, 1);
    out->data[0] = tensor_sum(t);
    return out;
  }

  if (dim == -1) {
    dim = 0;
  } else {
    dim = t->ndims - 1 - dim;
  }

  out_dims = t->ndims - 1;
  size_t out_shape[out_dims];
  for (size_t i = 0; i < dim; ++i) {
    out_shape[i] = t->shape[i];
  }
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    out_shape[i - 1] = t->shape[i];
  }
  out = _tensor_empty(out_shape, out_dims);

  inner = 1;
  for (size_t i = 0; i < dim; ++i) {
    inner *= t->shape[i];
  }

  outer = 1;
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    outer *= t->shape[i];
  }

  inner_stride = dim - 1 >= 0 ? 1 : 0;
  outer_stride = dim + 1 < t->ndims ? t->stride[dim + 1] : 0;

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < outer; ++i) {
    for (size_t j = 0; j < inner; ++j) {
      float maximum = -INFINITY;
      size_t argmax = -1;
      for (size_t k = 0; k < t->shape[dim]; ++k) {
        if (t->data[i * outer_stride + k * t->stride[dim] + j * inner_stride] >
            maximum) {
          maximum =
              t->data[i * outer_stride + k * t->stride[dim] + j * inner_stride];
          argmax = k;
        }
      }
      out->data[i * inner + j] = argmax;
    }
  }

  return out;
}

float tensor_mean(Tensor *t) {
  size_t numel;
  float sum;

  numel = tensor_numel(t);
  sum = 0.0f;
  for (size_t i = 0; i < numel; ++i) {
    sum += t->data[i];
  }

  return sum / numel;
}

Tensor *tensor_mean_at(Tensor *t, int32_t dim) {
  size_t outer, inner, outer_stride, inner_stride;
  size_t out_dims;
  Tensor *out;

  assert(dim == -1 || (dim >= 0 && dim < (int32_t)t->ndims));

  if (t->ndims == 1) {
    out = _tensor_empty((size_t[]){1}, 1);
    out->data[0] = tensor_mean(t);
    return out;
  }

  if (dim == -1) {
    dim = 0;
  } else {
    dim = t->ndims - 1 - dim;
  }

  out_dims = t->ndims - 1;
  size_t out_shape[out_dims];
  for (size_t i = 0; i < dim; ++i) {
    out_shape[i] = t->shape[i];
  }
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    out_shape[i - 1] = t->shape[i];
  }
  out = _tensor_empty(out_shape, out_dims);

  inner = 1;
  for (size_t i = 0; i < dim; ++i) {
    inner *= t->shape[i];
  }

  outer = 1;
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    outer *= t->shape[i];
  }

  inner_stride = dim - 1 >= 0 ? 1 : 0;
  outer_stride = dim + 1 < t->ndims ? t->stride[dim + 1] : 0;

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < outer; ++i) {
    for (size_t j = 0; j < inner; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < t->shape[dim]; ++k) {
        sum +=
            t->data[i * outer_stride + k * t->stride[dim] + j * inner_stride];
      }
      out->data[i * inner + j] = sum / t->shape[dim];
    }
  }

  return out;
}

float tensor_std(Tensor *t) {
  size_t numel;
  float sum, sum2;

  assert(t->ndims == 2);

  numel = tensor_numel(t);
  sum = 0.0f, sum2 = 0.0f;
  for (size_t i = 0; i < numel; ++i) {
    sum += t->data[i];
    sum2 += t->data[i] * t->data[i];
  }

  return sqrt((sum2 / numel) - (sum / numel) * (sum / numel));
}

Tensor *tensor_relu(Tensor *t) {
  size_t numel;
  Tensor *out;

  numel = tensor_numel(t);
  out = tensor_empty_like(t);

#pragma omp parallel for
  for (size_t i = 0; i < numel; ++i) {
    out->data[i] = t->data[i] > 0 ? t->data[i] : 0;
  }

  return out;
}

Tensor *tensor_softmax(Tensor *t, int32_t dim) {
  size_t outer, inner, outer_stride, inner_stride;
  Tensor *out;

  if (dim == -1) {
    dim = 0;
  } else {
    dim = t->ndims - 1 - dim;
  }

  inner = 1;
  for (size_t i = 0; i < dim; ++i) {
    inner *= t->shape[i];
  }

  outer = 1;
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    outer *= t->shape[i];
  }

  inner_stride = dim - 1 >= 0 ? 1 : 0;
  outer_stride = dim + 1 < t->ndims ? t->stride[dim + 1] : 0;

  out = tensor_empty_like(t);

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < outer; ++i) {
    for (size_t j = 0; j < inner; ++j) {
      float sum, maximum;

      maximum = -INFINITY;
      for (size_t k = 0; k < t->shape[dim]; ++k) {
        if (t->data[i * outer_stride + k * t->stride[dim] + j * inner_stride] >
            maximum)
          maximum =
              t->data[i * outer_stride + k * t->stride[dim] + j * inner_stride];
      }

      sum = 0.0f;
      for (size_t k = 0; k < t->shape[dim]; ++k) {
        sum += expf(
            t->data[i * outer_stride + k * t->stride[dim] + j * inner_stride] -
            maximum);
      }

      for (size_t k = 0; k < t->shape[dim]; ++k) {
        out->data[i * outer_stride + k * t->stride[dim] + j * inner_stride] =
            expf(t->data[i * outer_stride + k * t->stride[dim] +
                         j * inner_stride] -
                 maximum) /
            sum;
      }
    }
  }

  return out;
}

Tensor *tensor_cross_entropy(Tensor *pred, Tensor *target) {
  Tensor *out;

  assert(pred->ndims == 2 && target->ndims == 1);
  assert(pred->shape[1] == target->shape[0]);

  for (size_t i = 0; i < target->shape[0]; ++i) {
    assert((int32_t)target->data[i] >= 0 &&
           (int32_t)target->data[i] < pred->shape[0]);
  }

  out = tensor_empty_like(target);

#pragma omp parallel for
  for (size_t row = 0; row < pred->shape[1]; ++row) {
    float sum, maximum;

    maximum = -INFINITY;
    for (size_t col = 0; col < pred->shape[0]; ++col) {
      if (pred->data[row * pred->stride[1] + col * pred->stride[0]] > maximum) {
        maximum = pred->data[row * pred->stride[1] + col * pred->stride[0]];
      }
    }

    sum = 0.0f;
    for (size_t col = 0; col < pred->shape[0]; ++col) {
      sum += expf(pred->data[row * pred->stride[1] + col * pred->stride[0]] -
                  maximum);
    }

    out->data[row] =
        -log(expf(pred->data[row * pred->stride[1] +
                             (int32_t)target->data[row] * pred->stride[0]] -
                  maximum) /
             sum);
  }

  return out;
}

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

void tensor_transpose(Tensor *t) {
  for (int32_t i = 0; i < (t->ndims + 1) / 2; ++i) {
    size_t tmp;

    tmp = t->shape[i];
    t->shape[i] = t->shape[t->ndims - 1 - i];
    t->shape[t->ndims - 1 - i] = tmp;

    tmp = t->stride[i];
    t->stride[i] = t->stride[t->ndims - 1 - i];
    t->stride[t->ndims - 1 - i] = tmp;
  }
}

void tensor_print(Tensor *t) {
  size_t batch_size;

  if (t->ndims == 1) {
    printf("[\n");
    for (size_t i = 0; i < t->shape[0]; ++i) {
      printf(" %f, ", t->data[i]);
    }
    printf("\n]\n");
    return;
  }

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
        printf(" %f,", t->data[batch * t_stride_2 + row * t->stride[1] +
                               col * t->stride[0]]);
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

void tensor_add_scaler(Tensor *t, float x) {
  size_t numel;
  numel = tensor_numel(t);
#pragma omp parallel for
  for (size_t i = 0; i < numel; ++i) {
    t->data[i] += x;
  }
}

void tensor_sub_scaler(Tensor *t, float x) {
  size_t numel;
  numel = tensor_numel(t);
#pragma omp parallel for
  for (size_t i = 0; i < numel; ++i) {
    t->data[i] -= x;
  }
}

void tensor_mul_scaler(Tensor *t, float x) {
  size_t numel;
  numel = tensor_numel(t);
#pragma omp parallel for
  for (size_t i = 0; i < numel; ++i) {
    t->data[i] *= x;
  }
}

void tensor_div_scaler(Tensor *t, float x) {
  size_t numel;
  numel = tensor_numel(t);
#pragma omp parallel for
  for (size_t i = 0; i < numel; ++i) {
    t->data[i] /= x;
  }
}

Tensor *tensor_matadd(Tensor *a, Tensor *b) {
  size_t ndims, numel;
  Tensor *c;

  ndims = a->ndims;
  numel = tensor_numel(a);

  if (b->ndims == 1) {
    size_t batch = numel / a->shape[0];

    assert(a->shape[0] == b->shape[0]);

    c = tensor_empty_like(a);

#pragma omp parallel for
    for (size_t i = 0; i < batch; ++i) {
      for (size_t j = 0; j < a->shape[0]; ++j) {
        c->data[i * c->stride[1] + j * c->stride[0]] =
            a->data[i * a->stride[1] + j * a->stride[0]] + b->data[j];
      }
    }

    return c;
  }

  assert(a->ndims == b->ndims);

  for (size_t i = 0; i < ndims; ++i) {
    assert(a->shape[i] == b->shape[i]);
  }

  c = tensor_empty_like(a);

#pragma omp parallel for
  for (size_t i = 0; i < numel; ++i) {
    c->data[i] = a->data[i] + b->data[i];
  }

  return c;
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

  Tensor *c = _tensor_empty(c_shape, ndims);

  size_t a_stride_2 = ndims > 2 ? a->stride[2] : 1;
  size_t b_stride_2 = ndims > 2 ? b->stride[2] : 1;
  size_t c_stride_2 = ndims > 2 ? c->stride[2] : 1;

#pragma omp parallel for collapse(3)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    for (size_t row = 0; row < c->shape[1]; ++row) {
      for (size_t col = 0; col < c->shape[0]; ++col) {
        float sum = 0.0f;
        for (size_t k = 0; k < a->shape[0]; ++k) {
          sum += a->data[batch * a_stride_2 + row * a->stride[1] +
                         k * a->stride[0]] *
                 b->data[batch * b_stride_2 + k * b->stride[1] +
                         col * b->stride[0]];
        }
        c->data[batch * c_stride_2 + row * c->stride[1] + col * c->stride[0]] =
            sum;
      }
    }
  }

  return c;
}

bool tensor_allclose(Tensor *a, Tensor *b, float eps) {
  size_t ndims, numel;

  assert(a->ndims == b->ndims);

  ndims = a->ndims;
  numel = tensor_numel(a);

  for (size_t i = 0; i < ndims; ++i) {
    assert(a->shape[i] == b->shape[i]);
  }

  for (size_t i = 0; i < numel; ++i) {
    if (fabsf(a->data[i] - b->data[i]) > eps)
      return false;
  }

  return true;
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
