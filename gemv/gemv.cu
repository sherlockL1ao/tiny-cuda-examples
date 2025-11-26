#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define SMALL_K 8
#define WARP_SIZE 32

template <unsigned int warp_size> __device__ __forceinline__ float warpReduceSum(float sum) {
  if (warp_size >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, ...
  if (warp_size >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, ...
  if (warp_size >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, ...
  if (warp_size >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, ...
  if (warp_size >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1
  return sum;
}

/*
A: [m, k], B:[1, k], C: [m, 1]
*/
__global__ void SgemvSmallK(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
  __shared__ float shared_B[SMALL_K];

  int rows = blockIdx.x * blockDim.x + threadIdx.x;
  if (rows >= m) return;

  int tid = threadIdx.x;
  if (tid < k) shared_B[tid] = B[tid]; // load B to shared memory
  __syncthreads();

  int   a_idx = rows * k;
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < k; ++i) {
    sum += A[a_idx + i] * shared_B[i];
  }
  C[rows] = sum;
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
  int   a_idx = rows * k;
  float sum = 0;
  int   kIteration = k / (WARP_SIZE * 4);
#pragma unroll
  for (int i = 0; i < kIteration; ++i) {
    int           col_idx = (i * WARP_SIZE + lane) * 4;
    const float4* a_vals = reinterpret_cast<const float4*>(A + a_idx + col_idx); // load 4 floats
    const float4* b_vals = reinterpret_cast<const float4*>(B + col_idx);
    sum += a_vals->x * b_vals->x;
    sum += a_vals->y * b_vals->y;
    sum += a_vals->z * b_vals->z;
    sum += a_vals->w * b_vals->w;
  }
  sum = warpReduceSum<32>(sum);
  if (lane == 0) C[rows] = sum;
}

__global__ void SgemvK16(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
  int rows = (blockIdx.x * blockDim.y + threadIdx.y) * 2;
  int lane = threadIdx.x; // 0..31
  if (rows >= m) return;
  int a_idx = rows * k;
  float sum = A[a_idx + lane] * B[lane % 16];
  sum = warpReduceSum<16>(sum);
  if (lane == 0) C[rows] = sum;
  if (lane == 16) C[rows + 1] = sum;
}

// Simple wrapper that launches gemv on two float32 CUDA tensors.
// Only float32 is supported to keep the example compact.
/*
A: [m, k], B:[1, k], C: [m, 1]
*/
torch::Tensor gemv_launcher(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D matrices");
  TORCH_CHECK(A.size(1) == B.size(1), "shape mismatch: A.cols must equal B.cols");
  TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");
  TORCH_CHECK(B.size(0) == 1, "B must be a row vector");

  // Set CUDA device for the duration of this call.
  c10::cuda::CUDAGuard device_guard(A.device());

  // Respect PyTorch's current stream (per-thread default stream semantics).
  auto stream = c10::cuda::getCurrentCUDAStream(A.device().index());

  // Make sure data is contiguous for raw pointer access.
  auto lhs = A.contiguous();
  auto rhs = B.contiguous();
  torch::Tensor C = torch::empty({lhs.size(0), 1}, lhs.options());

  const int64_t m = lhs.size(0);
  const int64_t k = lhs.size(1);

  if (k <= SMALL_K) {
    dim3 block(128);
    dim3 grid((m + block.x - 1) / block.x);
    SgemvSmallK<<<grid, block, 0, stream>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);
  } else {
    dim3 block(32, 4); // 128 threads, 4 warps
    if (k == 16) {
      dim3 grid((m + block.y * 2 - 1) / (block.y * 2));
      SgemvK16<<<grid, block, 0, stream>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);
    } else {
      dim3 grid((m + block.y - 1) / block.y);
      if (k == 128) {
        SgemvK128<<<grid, block, 0, stream>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);
      } else {
        SgemvK32<<<grid, block, 0, stream>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);
      }
    }
  }

  // Surface CUDA errors (helps debugging when called from Python).
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  // Catch runtime errors from the kernel (useful during development).
  err = cudaStreamSynchronize(stream);
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return C;
}
