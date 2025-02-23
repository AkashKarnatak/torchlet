#include "tensor.h"
#include "mnist_utils.h"

Dataset dataset_init(float *data, float *label, size_t n, size_t row_size,
                     size_t batch_size) {
  Tensor *data_t, *label_t;
  Dataset ds;

  data_t = (Tensor *)malloc(sizeof(Tensor));
  assert(data_t != NULL);
  label_t = (Tensor *)malloc(sizeof(Tensor));
  assert(label_t != NULL);

  data_t->ndims = 2, label_t->ndims = 1;
  data_t->on_gpu = false, label_t->on_gpu = false;

  data_t->shape = (size_t *)malloc(data_t->ndims * sizeof(size_t));
  assert(data_t->shape != NULL);
  label_t->shape = (size_t *)malloc(label_t->ndims * sizeof(size_t));
  assert(label_t->shape != NULL);

  data_t->stride = (size_t *)malloc(data_t->ndims * sizeof(size_t));
  assert(data_t->stride != NULL);
  label_t->stride = (size_t *)malloc(label_t->ndims * sizeof(size_t));
  assert(label_t->stride != NULL);

  data_t->shape[0] = row_size, data_t->shape[1] = batch_size;
  data_t->stride[0] = 1, data_t->stride[1] = row_size;
  label_t->shape[0] = batch_size;
  label_t->stride[0] = 1;

  data_t->data = data, label_t->data = label;

  ds.data = data, ds.label = label, ds.n = n, ds.row_size = row_size,
  ds.batch_len = batch_size, ds._batch_size = batch_size, ds._offset = 0;
  ds.data_t = data_t, ds.label_t = label_t;

  return ds;
}

void dataset_free(Dataset *ds) {
  free(ds->data_t->shape);
  free(ds->data_t->stride);
  free(ds->data_t);
  free(ds->label_t->shape);
  free(ds->label_t->stride);
  free(ds->label_t);
  free(ds->data);
  free(ds->label);
}

Dataset dataset_load(const char *data_path, const char *label_path, size_t n,
                     size_t row_size, size_t batch_size) {
  FILE *f;
  float *data, *label;
  Dataset ds;

  f = fopen(data_path, "r");
  assert(f != NULL);
  data = (float *)malloc(n * row_size * sizeof(float));
  assert(data != NULL);
  assert(fread(data, sizeof(float), n * row_size, f) == n * row_size);
  assert(fclose(f) == 0);

  f = fopen(label_path, "r");
  assert(f != NULL);
  label = (float *)malloc(n * sizeof(float));
  assert(label != NULL);
  assert(fread(label, sizeof(float), n, f) == n);
  assert(fclose(f) == 0);

  ds = dataset_init(data, label, n, row_size, batch_size);

  return ds;
}

bool dataset_next(Dataset *ds) {
  if (ds->_offset >= ds->n) {
    return false;
  }

  ds->data_t->data = ds->data + ds->_offset * ds->row_size;
  ds->label_t->data = ds->label + ds->_offset;
  ds->batch_len = ds->_batch_size;
  if (ds->n - ds->_offset < ds->_batch_size) {
    ds->data_t->shape[1] = ds->n - ds->_offset;
    ds->label_t->shape[0] = ds->n - ds->_offset;
    ds->batch_len = ds->n - ds->_offset;
  }
  ds->_offset += ds->_batch_size;

  return true;
}

void dataset_reset(Dataset *ds) {
  ds->data_t->data = ds->data;
  ds->label_t->data = ds->label;
  ds->data_t->shape[1] = ds->_batch_size;
  ds->label_t->shape[0] = ds->_batch_size;
  ds->_offset = 0;
  ds->batch_len = ds->_batch_size;
}

