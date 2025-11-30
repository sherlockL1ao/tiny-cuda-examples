#include <cuda.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define SMALL_K 8
#define WARP_SIZE 32
#define HALF_WARP_SIZE 16

template <unsigned int warp_size>
__device__ __forceinline__ float warpReduceSum(float sum) {
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

__global__ void SgemvK64(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
  int rows = blockIdx.x * blockDim.y + threadIdx.y;
  int lane = threadIdx.x; // 0..31
  if (rows >= m) return;
  int   a_idx = rows * k;
  float sum = 0.f;
  int   kIteration = k / (WARP_SIZE * 2);
#pragma unroll
  for (int i = 0; i < kIteration; ++i) {
    int           col_idx = (i * WARP_SIZE + lane) * 2;
    const float2* a_vals = reinterpret_cast<const float2*>(A + a_idx + col_idx); // load 2 floats
    const float2* b_vals = reinterpret_cast<const float2*>(B + col_idx);
    sum += a_vals->x * b_vals->x;
    sum += a_vals->y * b_vals->y;
  }
  int remaining = k % (WARP_SIZE * 2);
  if (remaining > 0) {
    int offs = kIteration * (WARP_SIZE * 2);
#pragma unroll
    for (int i = lane; i + offs < k; i += WARP_SIZE) {
      sum += A[a_idx + offs + i] * B[offs + i];
    }
  }
  sum = warpReduceSum<32>(sum);
  if (lane == 0) C[rows] = sum;
}

__global__ void SgemvSubwarp8(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
  constexpr int ROWS_PER_WARP = 4;
  constexpr int THREADS_PER_ROW = WARP_SIZE / ROWS_PER_WARP; // 8

  extern __shared__ float shared_B[];

  int num_threads = blockDim.x * blockDim.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // Load B into shared memory
  for (int i = tid; i < k; i += num_threads) {
    shared_B[i] = B[i];
  }
  __syncthreads();

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;

  int row_in_warp = lane_id / THREADS_PER_ROW; // 0..3
  int lane_in_row = lane_id % THREADS_PER_ROW; // 0..7

  int rows_per_block = blockDim.y * ROWS_PER_WARP;
  int current_row = blockIdx.x * rows_per_block + warp_id * ROWS_PER_WARP + row_in_warp;

  if (current_row >= m) return;

  const float* row_A = A + current_row * k;
  float        sum = 0.f;

#pragma unroll
  for (int i = lane_in_row; i < k; i += THREADS_PER_ROW) {
    sum += row_A[i] * shared_B[i];
  }
  sum = warpReduceSum<THREADS_PER_ROW>(sum);
  if (lane_in_row == 0) C[current_row] = sum;
}

// threads (x, y) = (32, 8)
__global__ void SgemvVec4Subwarp8(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
  constexpr int ROWS_PER_WARP = 4;
  constexpr int THREADS_PER_ROW = WARP_SIZE / ROWS_PER_WARP; // 8

  extern __shared__ float shared_B[];

  int num_float4 = k / 4;
  int num_threads = blockDim.x * blockDim.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // Load B in chunks of float4
  for (int i = tid; i < num_float4; i += num_threads) {
    reinterpret_cast<float4*>(shared_B)[i] = reinterpret_cast<const float4*>(B)[i];
  }
  __syncthreads();

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;

  int row_in_warp = lane_id / THREADS_PER_ROW; // 0..3
  int lane_in_row = lane_id % THREADS_PER_ROW; // 0..7

  int rows_per_block = blockDim.y * ROWS_PER_WARP;
  int current_row = blockIdx.x * rows_per_block + warp_id * ROWS_PER_WARP + row_in_warp;

  if (current_row >= m) return;

  const float* row_A = A + current_row * k;
  float        sum = 0.f;

#pragma unroll
  for (int i = lane_in_row; i < num_float4; i += THREADS_PER_ROW) {
    float4 a_val = reinterpret_cast<const float4*>(row_A)[i];
    float4 b_val = reinterpret_cast<const float4*>(shared_B)[i];

    sum += a_val.x * b_val.x;
    sum += a_val.y * b_val.y;
    sum += a_val.z * b_val.z;
    sum += a_val.w * b_val.w;
  }

  sum = warpReduceSum<THREADS_PER_ROW>(sum);
  if (lane_in_row == 0) C[current_row] = sum;
}

// block(32, 4)
__global__ void SgemvK16(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int k) {
  constexpr int ROWS_PER_WARP = 2;
  constexpr int THREADS_PER_ROW = WARP_SIZE / ROWS_PER_WARP; // 16
  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x; // 0..31

  int row_in_warp = lane_id / THREADS_PER_ROW; // 0..1
  int lane_in_row = lane_id % THREADS_PER_ROW; // 0..15
  int current_row = blockIdx.x * (blockDim.y * ROWS_PER_WARP) + warp_id * ROWS_PER_WARP + row_in_warp;
  if (current_row >= m) return;

  float sum=0.f;
  if (lane_in_row < k) {
    sum += A[current_row * k + lane_in_row] * B[lane_in_row];
  }
  sum = warpReduceSum<THREADS_PER_ROW>(sum);
  if (lane_in_row == 0) C[current_row] = sum;
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

  // Make sure data is contiguous for raw pointer access.
  auto lhs = A.contiguous();
  auto rhs = B.contiguous();
  torch::Tensor C = torch::empty({lhs.size(0), 1}, lhs.options());

  const int64_t m = lhs.size(0);
  const int64_t k = lhs.size(1);

  if (k <= SMALL_K) {
    dim3 block(128);
    dim3 grid((m + block.x - 1) / block.x);
    SgemvSmallK<<<grid, block, 0>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);
  } else if (k <= 16) {
    dim3 block(32, 4); // 128 threads, 4 warps
    dim3 grid((m + block.y * 2 - 1) / (block.y * 2));
    SgemvK16<<<grid, block, 0>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);
  } else {
    dim3 block(32, 8); // 256threads, 8 warps
    dim3 grid((m + block.y * 4 - 1) / block.y * 4);
    int  shared_mem = k * sizeof(float);
    if (k % 4 != 0) {
      SgemvSubwarp8<<<grid, block, shared_mem>>>(
          lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);
    } else {
      SgemvVec4Subwarp8<<<grid, block, shared_mem>>>(
          lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);
    }
  }

  // Surface CUDA errors (helps debugging when called from Python).
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  // Catch runtime errors from the kernel (useful during development).
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return C;
}

torch::Tensor gemv_cublas(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D matrices");
  TORCH_CHECK(A.size(1) == B.size(1), "shape mismatch: A.cols must equal B.cols");
  TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");
  TORCH_CHECK(B.size(0) == 1, "B must be a row vector");

  // Make sure data is contiguous for raw pointer access.
  auto lhs = A.contiguous();
  auto rhs = B.contiguous();
  torch::Tensor C = torch::empty({lhs.size(0), 1}, lhs.options());

  // Dimensions
  // m: rows of A (output height)
  // k: cols of A (and length of vector B)
  const int m = static_cast<int>(lhs.size(0));
  const int k = static_cast<int>(lhs.size(1));

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  float alpha = 1.0f;
  float beta = 0.0f;

  // KEY FIX:
  // PyTorch is Row-Major. cuBLAS is Col-Major.
  // An (m x k) Row-Major matrix looks like a (k x m) Col-Major matrix in memory.
  // We use CUBLAS_OP_T to effectively transpose the (k x m) view back to (m x k).
  cublasSgemv(
      handle,
      CUBLAS_OP_T,          // Transpose the Col-Major view
      k,                    // Rows of the matrix in memory (k)
      m,                    // Cols of the matrix in memory (m)
      &alpha,
      lhs.data_ptr<float>(),// Matrix A
      k,                    // Leading dimension (stride between cols in memory)
      rhs.data_ptr<float>(),// Vector x
      1,                    // incx
      &beta,
      C.data_ptr<float>(),  // Result y
      1                     // incy
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return C;
}
