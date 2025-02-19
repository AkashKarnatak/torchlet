#include "tensor.h"

__global__ void relu_kernel(float *in_data, float *out_data, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  out_data[i] = in_data[i] > 0 ? in_data[i] : 0.0f;
}

__global__ void matadd_kernel(float *a_data, float *b_data, float *c_data,
                              size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  c_data[i] = a_data[i] + b_data[i];
}

__global__ void matmul_kernel(float *a_data, float *b_data, float *c_data,
                              size_t N, size_t K, size_t M, size_t a_stride_2,
                              size_t a_stride_1, size_t a_stride_0,
                              size_t b_stride_2, size_t b_stride_1,
                              size_t b_stride_0) {
  size_t batch = blockDim.z * blockIdx.z + threadIdx.z;
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= N || col >= M)
    return;

  float sum = 0.0f;

  for (size_t k = 0; k < K; ++k) {
    sum += a_data[batch * a_stride_2 + row * a_stride_1 + k * a_stride_0] *
           b_data[batch * b_stride_2 + k * b_stride_1 + col * b_stride_0];
  }
  c_data[batch * N * M + row * M + col] = sum;
}

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

Tensor *_tensor_empty_gpu(size_t *shape, size_t ndims) {
  Tensor *out;
  size_t numel;

  out = (Tensor *)malloc(sizeof(Tensor));
  assert(out != NULL);

  out->ndims = ndims;
  out->on_gpu = true;

  out->shape = (size_t *)malloc(ndims * sizeof(size_t));
  assert(out->shape != NULL);

  out->stride = (size_t *)malloc(ndims * sizeof(size_t));
  assert(out->stride != NULL);

  numel = 1;
  for (size_t i = 0; i < ndims; ++i) {
    assert(shape[i] != 0);
    out->shape[i] = shape[i];
    out->stride[i] = numel;
    numel *= shape[i];
  }

  cudaMalloc(&out->data, numel * sizeof(float));

  return out;
}

Tensor *tensor_empty_gpu(size_t *shape, size_t ndims) {
  size_t shape_r[ndims];
  for (int32_t i = 0; i < (ndims + 1) / 2; ++i) {
    shape_r[i] = shape[ndims - 1 - i];
    shape_r[ndims - 1 - i] = shape[i];
  }
  return _tensor_empty_gpu(shape_r, ndims);
}

Tensor *tensor_empty_like_gpu(Tensor *in) {
  return _tensor_empty_gpu(in->shape, in->ndims);
}

Tensor *tensor_to_gpu(Tensor *in) {
  Tensor *out;
  size_t numel;

  out = tensor_empty_like_gpu(in);
  numel = tensor_numel(in);

  cudaMemcpy(out->data, in->data, numel * sizeof(float),
             cudaMemcpyHostToDevice);

  return out;
}

Tensor *tensor_relu_gpu(Tensor *in) {
  size_t numel;
  Tensor *out;

  assert(in->on_gpu);

  numel = tensor_numel(in);
  out = tensor_empty_like_gpu(in);

  size_t numThreads = 1024;
  size_t numBlocks = cdiv(numel, numThreads);

  relu_kernel<<<numBlocks, numThreads>>>(in->data, out->data, numel);

  return out;
}

Tensor *tensor_matadd_gpu(Tensor *a, Tensor *b) {
  size_t ndims, numel;
  Tensor *c;

  assert(a->ndims == b->ndims);
  assert(a->on_gpu && b->on_gpu);

  ndims = a->ndims;
  numel = tensor_numel(a);

  for (size_t i = 0; i < ndims; ++i) {
    assert(a->shape[i] == b->shape[i]);
  }

  c = tensor_empty_like_gpu(a);

  size_t numThreads = 1024;
  size_t numBlocks = cdiv(numel, numThreads);

  matadd_kernel<<<numBlocks, numThreads>>>(a->data, b->data, c->data, numel);

  return c;
}

Tensor *tensor_matmul_gpu(Tensor *a, Tensor *b) {
  assert(a->ndims > 1 && b->ndims > 1);

  assert(a->ndims == b->ndims);
  assert(a->on_gpu && b->on_gpu);

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

  Tensor *c = tensor_empty_gpu(c_shape, ndims);

  size_t a_stride_2 = ndims > 2 ? a->stride[2] : 1;
  size_t b_stride_2 = ndims > 2 ? b->stride[2] : 1;
  size_t c_stride_2 = ndims > 2 ? c->stride[2] : 1;

  dim3 numThreads(32, 32, 1);
  dim3 numBlocks(cdiv(c_shape[0], numThreads.x), cdiv(c_shape[1], numThreads.y),
                 batch_size);

  matmul_kernel<<<numBlocks, numThreads>>>(
      a->data, b->data, c->data, a->shape[1], a->shape[0], b->shape[0],
      a_stride_2, a->stride[1], a->stride[0], b_stride_2, b->stride[1],
      b->stride[0]);

  return c;
}
