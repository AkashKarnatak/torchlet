#include "tensor.h"

#define BLOCK_SIZE 1024

__host__ __device__ size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void relu_kernel(float *in_data, float *out_data, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  out_data[i] = in_data[i] > 0 ? in_data[i] : 0.0f;
}

__global__ void relu_backward_kernel(float *a_data, float *b_data, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  if (b_data[i] <= 0)
    a_data[i] = 0;
}

__global__ void matadd_kernel(float *a_data, float *b_data, float *c_data,
                              size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  c_data[i] = a_data[i] + b_data[i];
}

__global__ void matadd_2d_1d_kernel(float *a_data, float *b_data, float *c_data,
                                    size_t N, size_t M, size_t a_stride_1,
                                    size_t a_stride_0) {
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= N || col >= M)
    return;
  c_data[row * M + col] =
      a_data[row * a_stride_1 + col * a_stride_0] + b_data[col];
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

__global__ void add_scaler_kernel(float *data, float x, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  data[i] += x;
}

__global__ void sub_scaler_kernel(float *data, float x, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  data[i] -= x;
}

__global__ void mul_scaler_kernel(float *data, float x, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  data[i] *= x;
}

__global__ void div_scaler_kernel(float *data, float x, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  data[i] /= x;
}

__global__ void update_at_kernel(float *in_data, float *idx_data, float x,
                                 size_t N, size_t in_stride_1,
                                 size_t in_stride_0) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;

  in_data[i * in_stride_1 + (size_t)idx_data[i] * in_stride_0] += x;
}

__global__ void block_reduction_sum_at_kernel(float *in_data, float *out_data,
                                              size_t M, size_t inner,
                                              size_t dim_stride,
                                              size_t outer_stride,
                                              size_t inner_stride) {
  size_t i = blockIdx.x / inner;
  size_t j = blockIdx.x % inner;

  float local_sum;
  __shared__ float mem_s[BLOCK_SIZE];

  local_sum = 0.0f;
  for (size_t block = 0; block < cdiv(M, BLOCK_SIZE); ++block) {
    if (block * BLOCK_SIZE + threadIdx.x >= M)
      continue;
    local_sum += in_data[i * outer_stride +
                         (block * BLOCK_SIZE + threadIdx.x) * dim_stride +
                         j * inner_stride];
  }

  mem_s[threadIdx.x] = local_sum;

  __syncthreads();

  for (size_t numThreads = BLOCK_SIZE / 2; numThreads > 0; numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      mem_s[threadIdx.x] += mem_s[threadIdx.x + numThreads];
    }
    __syncthreads();
  }

  out_data[blockIdx.x] = mem_s[0];
}

__global__ void
block_reduction_argmax_at_kernel(float *in_data, float *out_data, size_t M,
                                 size_t inner, size_t dim_stride,
                                 size_t outer_stride, size_t inner_stride) {
  size_t i = blockIdx.x / inner;
  size_t j = blockIdx.x % inner;

  float local_max, local_argmax;
  __shared__ float max_s[BLOCK_SIZE];
  __shared__ float argmax_s[BLOCK_SIZE];

  local_max = -INFINITY, local_argmax = threadIdx.x;
  for (size_t block = 0; block < cdiv(M, BLOCK_SIZE); ++block) {
    if (block * BLOCK_SIZE + threadIdx.x >= M)
      continue;
    float curr = in_data[i * outer_stride +
                         (block * BLOCK_SIZE + threadIdx.x) * dim_stride +
                         j * inner_stride];
    if (curr > local_max) {
      local_max = curr;
      local_argmax = block * BLOCK_SIZE + threadIdx.x;
    }
  }

  max_s[threadIdx.x] = local_max;
  argmax_s[threadIdx.x] = local_argmax;

  __syncthreads();

  for (size_t numThreads = BLOCK_SIZE / 2; numThreads > 0; numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      if (max_s[threadIdx.x + numThreads] > max_s[threadIdx.x]) {
        max_s[threadIdx.x] = max_s[threadIdx.x + numThreads];
        argmax_s[threadIdx.x] = argmax_s[threadIdx.x + numThreads];
      }
    }
    __syncthreads();
  }

  out_data[blockIdx.x] = argmax_s[0];
}

__global__ void cross_entropy_kernel(float *pred_data, float *target_data,
                                     float *out_data, size_t N, size_t M,
                                     size_t pred_stride_1,
                                     size_t pred_stride_0) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;

  float sum, maximum;

  maximum = -INFINITY;
  for (size_t k = 0; k < M; ++k) {
    if (pred_data[i * pred_stride_1 + k * pred_stride_0] > maximum)
      maximum = pred_data[i * pred_stride_1 + k * pred_stride_0];
  }

  sum = 0.0f;
  for (size_t k = 0; k < M; ++k) {
    sum += expf(pred_data[i * pred_stride_1 + k * pred_stride_0] - maximum);
  }

  out_data[i] = -log(expf(pred_data[i * pred_stride_1 +
                                    (int32_t)target_data[i] * pred_stride_0] -
                          maximum) /
                     sum);
}

__global__ void block_reduction_softmax_kernel(float *in_data, float *out_data,
                                               size_t M, size_t inner,
                                               size_t outer_stride,
                                               size_t inner_stride) {
  size_t i = blockIdx.x / inner;
  size_t j = blockIdx.x % inner;

  float local_sum, local_max, global_sum, global_max;
  __shared__ float mem_s[BLOCK_SIZE];

  local_sum = 0.0f, local_max = -INFINITY;
  for (size_t block = 0; block < cdiv(M, BLOCK_SIZE); ++block) {
    if (block * BLOCK_SIZE + threadIdx.x >= M)
      continue;
    float curr = in_data[i * outer_stride + block * BLOCK_SIZE + threadIdx.x +
                         j * inner_stride];
    if (curr > local_max) {
      local_sum *= expf(local_max - curr);
      local_max = curr;
    }
    local_sum += expf(curr - local_max);
  }

  mem_s[threadIdx.x] = local_max;

  __syncthreads();

  for (size_t numThreads = BLOCK_SIZE / 2; numThreads > 0; numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      mem_s[threadIdx.x] =
          max(mem_s[threadIdx.x], mem_s[threadIdx.x + numThreads]);
    }
    __syncthreads();
  }

  global_max = mem_s[0];

  mem_s[threadIdx.x] = local_sum * expf(local_max - global_max);

  __syncthreads();

  for (size_t numThreads = BLOCK_SIZE / 2; numThreads > 0; numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      mem_s[threadIdx.x] += mem_s[threadIdx.x + numThreads];
    }
    __syncthreads();
  }

  global_sum = mem_s[0];

  __syncthreads();

  for (size_t block = 0; block < cdiv(M, BLOCK_SIZE); ++block) {
    if (block * BLOCK_SIZE + threadIdx.x < M) {
      out_data[i * outer_stride + block * BLOCK_SIZE + threadIdx.x +
               j * inner_stride] =
          expf(in_data[i * outer_stride + block * BLOCK_SIZE + threadIdx.x +
                       j * inner_stride] -
               global_max) /
          global_sum;
    }
  }
}

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
  for (size_t i = 0; i < (ndims + 1) / 2; ++i) {
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
  for (size_t i = 0; i < in->ndims; ++i) {
    out->shape[i] = in->shape[i];
    out->stride[i] = in->stride[i];
  }
  numel = tensor_numel(in);

  cudaMemcpy(out->data, in->data, numel * sizeof(float),
             cudaMemcpyHostToDevice);

  return out;
}

Tensor *tensor_copy_gpu(Tensor *in) {
  Tensor *out;
  size_t numel;

  assert(in->on_gpu);

  out = tensor_empty_like_gpu(in);
  for (size_t i = 0; i < in->ndims; ++i) {
    out->shape[i] = in->shape[i];
    out->stride[i] = in->stride[i];
  }
  numel = tensor_numel(in);
  cudaMemcpy(out->data, in->data, numel * sizeof(float),
             cudaMemcpyDeviceToDevice);

  return out;
}

Tensor *tensor_to_cpu(Tensor *in) {
  Tensor *out;
  size_t numel;

  assert(in->on_gpu);

  out = tensor_empty_like(in);
  for (size_t i = 0; i < in->ndims; ++i) {
    out->shape[i] = in->shape[i];
    out->stride[i] = in->stride[i];
  }
  numel = tensor_numel(in);

  cudaMemcpy(out->data, in->data, numel * sizeof(float),
             cudaMemcpyDeviceToHost);

  return out;
}

void tensor_free_gpu(Tensor *t) {
  free(t->shape);
  free(t->stride);
  cudaFree(t->data);
  free(t);
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
  cudaDeviceSynchronize();

  return out;
}

void tensor_relu_backward_gpu(Tensor *a, Tensor *b) {
  size_t numel;

  assert(a->on_gpu && b->on_gpu);

  numel = tensor_numel(a);

  size_t numThreads = 1024;
  size_t numBlocks = cdiv(numel, numThreads);

  relu_backward_kernel<<<numBlocks, numThreads>>>(a->data, b->data, numel);
  cudaDeviceSynchronize();
}

Tensor *tensor_sum_at_gpu(Tensor *t, int32_t dim) {
  size_t outer, inner, outer_stride, inner_stride;
  size_t out_dims;
  Tensor *out;

  assert(dim == -1 || (dim >= 0 && dim < (int32_t)t->ndims));

  assert(t->ndims > 1);

  if (dim == -1) {
    dim = 0;
  } else {
    dim = t->ndims - 1 - dim;
  }

  out_dims = t->ndims - 1;
  size_t out_shape[out_dims];
  for (size_t i = 0; i < (size_t)dim; ++i) {
    out_shape[i] = t->shape[i];
  }
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    out_shape[i - 1] = t->shape[i];
  }
  out = _tensor_empty_gpu(out_shape, out_dims);

  inner = 1;
  for (size_t i = 0; i < (size_t)dim; ++i) {
    inner *= t->shape[i];
  }

  outer = 1;
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    outer *= t->shape[i];
  }

  inner_stride = dim - 1 >= 0 ? 1 : 0;
  outer_stride = (size_t)dim + 1 < t->ndims ? t->stride[dim + 1] : 0;

  size_t numThreads = 1024;
  size_t numBlocks = inner * outer;
  block_reduction_sum_at_kernel<<<numBlocks, numThreads>>>(
      t->data, out->data, t->shape[dim], inner, t->stride[dim], outer_stride,
      inner_stride);
  cudaDeviceSynchronize();

  return out;
}

Tensor *tensor_argmax_at_gpu(Tensor *t, int32_t dim) {
  size_t outer, inner, outer_stride, inner_stride;
  size_t out_dims;
  Tensor *out;

  assert(dim == -1 || (dim >= 0 && dim < (int32_t)t->ndims));

  assert(t->ndims > 1);

  if (dim == -1) {
    dim = 0;
  } else {
    dim = t->ndims - 1 - dim;
  }

  out_dims = t->ndims - 1;
  size_t out_shape[out_dims];
  for (size_t i = 0; i < (size_t)dim; ++i) {
    out_shape[i] = t->shape[i];
  }
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    out_shape[i - 1] = t->shape[i];
  }
  out = _tensor_empty_gpu(out_shape, out_dims);

  inner = 1;
  for (size_t i = 0; i < (size_t)dim; ++i) {
    inner *= t->shape[i];
  }

  outer = 1;
  for (size_t i = dim + 1; i < t->ndims; ++i) {
    outer *= t->shape[i];
  }

  inner_stride = dim - 1 >= 0 ? 1 : 0;
  outer_stride = (size_t)dim + 1 < t->ndims ? t->stride[dim + 1] : 0;

  size_t numThreads = 1024;
  size_t numBlocks = inner * outer;
  block_reduction_argmax_at_kernel<<<numBlocks, numThreads>>>(
      t->data, out->data, t->shape[dim], inner, t->stride[dim], outer_stride,
      inner_stride);
  cudaDeviceSynchronize();

  return out;
}

Tensor *tensor_matadd_gpu(Tensor *a, Tensor *b) {
  size_t ndims, numel;
  Tensor *c;

  ndims = a->ndims;
  numel = tensor_numel(a);

  if (b->ndims == 1) {
    size_t batch = numel / a->shape[0];

    assert(a->shape[0] == b->shape[0]);

    c = tensor_empty_like_gpu(a);

    dim3 numThreads(32, 32);
    dim3 numBlocks(cdiv(a->shape[0], numThreads.x), cdiv(batch, numThreads.y));

    matadd_2d_1d_kernel<<<numBlocks, numThreads>>>(a->data, b->data, c->data,
                                                   batch, a->shape[0],
                                                   a->stride[1], a->stride[0]);
    cudaDeviceSynchronize();

    return c;
  }

  assert(a->ndims == b->ndims);
  assert(a->on_gpu && b->on_gpu);

  for (size_t i = 0; i < ndims; ++i) {
    assert(a->shape[i] == b->shape[i]);
  }

  c = tensor_empty_like_gpu(a);

  size_t numThreads = 1024;
  size_t numBlocks = cdiv(numel, numThreads);

  matadd_kernel<<<numBlocks, numThreads>>>(a->data, b->data, c->data, numel);
  cudaDeviceSynchronize();

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

  Tensor *c = _tensor_empty_gpu(c_shape, ndims);

  size_t a_stride_2 = ndims > 2 ? a->stride[2] : 1;
  size_t b_stride_2 = ndims > 2 ? b->stride[2] : 1;

  dim3 numThreads(32, 32, 1);
  dim3 numBlocks(cdiv(c_shape[0], numThreads.x), cdiv(c_shape[1], numThreads.y),
                 batch_size);

  matmul_kernel<<<numBlocks, numThreads>>>(
      a->data, b->data, c->data, a->shape[1], a->shape[0], b->shape[0],
      a_stride_2, a->stride[1], a->stride[0], b_stride_2, b->stride[1],
      b->stride[0]);
  cudaDeviceSynchronize();

  return c;
}

void tensor_add_scaler_gpu(Tensor *t, float x) {
  size_t numel;
  numel = tensor_numel(t);
  size_t numThreads = 1024;
  size_t numBlocks = cdiv(numel, numThreads);
  add_scaler_kernel<<<numBlocks, numThreads>>>(t->data, x, numel);
  cudaDeviceSynchronize();
}

void tensor_sub_scaler_gpu(Tensor *t, float x) {
  size_t numel;
  numel = tensor_numel(t);
  size_t numThreads = 1024;
  size_t numBlocks = cdiv(numel, numThreads);
  sub_scaler_kernel<<<numBlocks, numThreads>>>(t->data, x, numel);
  cudaDeviceSynchronize();
}

void tensor_mul_scaler_gpu(Tensor *t, float x) {
  size_t numel;
  numel = tensor_numel(t);
  size_t numThreads = 1024;
  size_t numBlocks = cdiv(numel, numThreads);
  mul_scaler_kernel<<<numBlocks, numThreads>>>(t->data, x, numel);
  cudaDeviceSynchronize();
}

void tensor_div_scaler_gpu(Tensor *t, float x) {
  size_t numel;
  numel = tensor_numel(t);
  size_t numThreads = 1024;
  size_t numBlocks = cdiv(numel, numThreads);
  div_scaler_kernel<<<numBlocks, numThreads>>>(t->data, x, numel);
  cudaDeviceSynchronize();
}

void tensor_update_at_gpu(Tensor *in, Tensor *idx, float x) {
  assert(in->on_gpu && idx->on_gpu);

  assert(in->ndims == 2 && idx->ndims == 1);

  size_t numThreads = 1024;
  size_t numBlocks = cdiv(idx->shape[0], numThreads);

  update_at_kernel<<<numBlocks, numThreads>>>(
      in->data, idx->data, x, in->shape[1], in->stride[1], in->stride[0]);
  cudaDeviceSynchronize();
}

Tensor *tensor_cross_entropy_gpu(Tensor *pred, Tensor *target) {
  Tensor *out;

  assert(pred->ndims == 2 && target->ndims == 1);
  assert(pred->shape[1] == target->shape[0]);

  assert(pred->on_gpu && target->on_gpu);

  out = tensor_empty_like_gpu(target);

  size_t numThreads = 1024;
  size_t numBlocks = cdiv(out->shape[0], numThreads);

  cross_entropy_kernel<<<numBlocks, numThreads>>>(
      pred->data, target->data, out->data, pred->shape[1], pred->shape[0],
      pred->stride[1], pred->stride[0]);
  cudaDeviceSynchronize();

  return out;
}

Tensor *tensor_softmax_gpu(Tensor *in, int32_t dim) {
  size_t outer, inner, outer_stride, inner_stride;
  Tensor *out;

  assert(in->on_gpu);

  if (dim == -1) {
    dim = 0;
  } else {
    dim = in->ndims - 1 - dim;
  }

  inner = 1;
  for (size_t i = 0; i < (size_t)dim; ++i) {
    inner *= in->shape[i];
  }

  outer = 1;
  for (size_t i = (size_t)dim + 1; i < in->ndims; ++i) {
    outer *= in->shape[i];
  }

  inner_stride = dim - 1 >= 0 ? 1 : 0;
  outer_stride = (size_t)dim + 1 < in->ndims ? in->stride[dim + 1] : 0;

  out = tensor_empty_like_gpu(in);

  size_t numThreads(BLOCK_SIZE);
  size_t numBlocks(inner * outer);

  block_reduction_softmax_kernel<<<numBlocks, numThreads>>>(
      in->data, out->data, in->shape[dim], inner, outer_stride, inner_stride);
  cudaDeviceSynchronize();

  return out;
}
