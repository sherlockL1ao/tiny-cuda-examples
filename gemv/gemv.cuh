#include <cuda_runtime.h>

#define TILE_SIZE 32
#define SMEM_B_SIZE 128

/*
A: [m, k], B:[1, k], C: [m, 1]
*/
__global__ void naive_sgemv_kernel(const float *A, const float *B, float *C, int m, int k) {
  int rows = blockIdx.x * blockDim.x + threadIdx.x;
  if (rows < m) {
    int a_idx = rows * k;
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
      sum += A[a_idx + i] * B[i];
    }
    C[rows] = sum;
  }
}

__global__ void smemGEMVKernel(const float *A, const float* B, float *C, int m, int k) {
  __shared__ float smemB[SMEM_B_SIZE];
  int rows = blockIdx.x * blockDim.x + threadIdx.x;
  if (rows < m) {

  }
}
