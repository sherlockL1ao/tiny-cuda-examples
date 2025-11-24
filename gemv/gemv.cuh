#include <cuda_runtime.h>

#define TILE_SIZE 32

template <unsigned int warpSize> __device__ __forceinline__ float warpReduceSum(float sum) {
  if (warpSize >= 32) {
    sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, ...
  }
  if (warpSize >= 16) {
    sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, ...
  }
  if (warpSize >= 8) {
    sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, ...
  }
  if (warpSize >= 4) {
    sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, ...
  }
  if (warpSize >= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1
  }
  return sum;
}

/*
A: [m, k], B:[1, k], C: [m, 1]
*/
__global__ void SgemvNaiveKernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
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

__global__ void SgemvK32(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
  int rows = blockIdx.x * blockDim.y + threadIdx.y;
  if (rows < m) {
    int   a_idx = rows * k;
    float sum = A[a_idx + threadIdx.x] * B[threadIdx.x];
    sum = warpReduceSum<32>(sum);
    if (threadIdx.x == 0) {
      C[rows] = sum;
    }
  }
}
