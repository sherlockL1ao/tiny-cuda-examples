#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

template <unsigned int wrap_size> __device__ __forceinline__ float warpReduceSum(float sum) {
  if (wrap_size >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, ...
  if (wrap_size >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, ...
  if (wrap_size >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, ...
  if (wrap_size >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, ...
  if (wrap_size >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1
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
  int lane = threadIdx.x; // 0..31
  if (rows >= m) return;
  int   a_idx = rows * k;
  float sum = 0;
#pragma unroll
  for (int i = lane; i < k; i += WARP_SIZE) {
    sum += A[a_idx + i] * B[i];
  }
  sum = warpReduceSum<32>(sum);
  if (lane == 0) C[rows] = sum;
}

__global__ void SgemvK128(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
  int rows = blockIdx.x * blockDim.y + threadIdx.y;
  int lane = threadIdx.x; // 0..31
  if (rows >= m) return;
  int a_idx = rows*k;
  float sum = 0;
  int kIteration = k / (WARP_SIZE*4);
#pragma unroll
  for (int i = 0; i < kIteration; ++i) {
    int col_idx = (i * WARP_SIZE + lane) * 4;
    const float4* a_vals = reinterpret_cast<const float4*>(A + a_idx + col_idx);
    const float4* b_vals = reinterpret_cast<const float4*>(B + col_idx);
    sum += a_vals->x * b_vals->x;
    sum += a_vals->y * b_vals->y;
    sum += a_vals->z * b_vals->z;
    sum += a_vals->w * b_vals->w;
  }
  sum = warpReduceSum<32>(sum);
  if (lane == 0) C[rows] = sum;
}
