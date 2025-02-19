#include "tensor.h"
#include <time.h>

struct timer {
  struct timespec start_time, end_time;
};

void start_timer(struct timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->start_time);
}

void stop_timer(struct timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->end_time);
}

double time_diff(struct timer *t) {
  double diff = (t->end_time.tv_sec - t->start_time.tv_sec) +
                (t->end_time.tv_nsec - t->start_time.tv_nsec) / 1000000000.0;
  return diff;
}

struct timer t;
int main() {
  cudaDeviceSynchronize();

  Tensor *a, *b, *c, *c_cpu;
  Tensor *a_gpu, *b_gpu, *c_gpu;

  a = tensor_randn((size_t[]){1027, 1529}, 2);
  b = tensor_randn((size_t[]){1529, 2747}, 2);

  a_gpu = tensor_to_gpu(a);
  b_gpu = tensor_to_gpu(b);

  start_timer(&t);
  c = tensor_matmul(a, b);
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));

  start_timer(&t);
  c_gpu = tensor_matmul_gpu(a_gpu, b_gpu);
  c_cpu = tensor_to_cpu(c_gpu);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  printf("Match: %s", tensor_allclose(c, c_cpu, 1e-4) ? "true" : "false");

  // tensor_print(c);
  // tensor_debug(c);

  // tensor_print(c_cpu);
  // tensor_debug(c_cpu);

  tensor_free(a);
  tensor_free(b);
  tensor_free(c);
  tensor_free(c_cpu);
  tensor_free_gpu(a_gpu);
  tensor_free_gpu(b_gpu);
  tensor_free_gpu(c_gpu);

  return 0;
}
