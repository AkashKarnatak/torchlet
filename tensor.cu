#include "tensor.h"

__global__ void matadd_kernel(float *a_data, float *b_data, float *c_data,
                              size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  c_data[i] = a_data[i] + b_data[i];
}

__global__ void relu_kernel(float *in_data, float *out_data, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  out_data[i] = in_data[i] > 0 ? in_data[i] : 0.0f;
}

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

Tensor *tensor_empty_like_gpu(Tensor *in) {
  Tensor *out;
  size_t numel;

  out = (Tensor *)malloc(sizeof(Tensor));
  assert(out != NULL);

  out->ndims = in->ndims;
  out->on_gpu = true;

  out->shape = (size_t *)malloc(out->ndims * sizeof(size_t));
  assert(out->shape != NULL);

  out->stride = (size_t *)malloc(out->ndims * sizeof(size_t));
  assert(out->stride != NULL);

  for (size_t i = 0; i < out->ndims; ++i) {
    out->shape[i] = in->shape[i];
    out->stride[i] = in->stride[i];
  }

  numel = tensor_numel(in);

  cudaMalloc(&out->data, numel * sizeof(float));

  return out;
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
