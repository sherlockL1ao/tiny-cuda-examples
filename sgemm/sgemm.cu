#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int WARP_SIZE = 32;

// Naive SGEMM kernel: C = A * B, each thread compute a C[m,n]
// A: [M, K], B: [K, N], C: [M, N]
template <int NUM_WARPS>
__global__ void
sgemm_naive_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int M, int K, int N) {
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  int g_idx = TB_SIZE * blockIdx.x + threadIdx.x;
  // out-of-bounds check
  if (g_idx >= M * N) {
    return;
  }
  int m_idx = g_idx / N;
  int n_idx = g_idx % N;

  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < K; ++i) {
    float a_val = A[m_idx * K + i];
    float b_val = B[i * N + n_idx];
    acc += a_val * b_val;
  }
  C[m_idx * N + n_idx] = acc;
}

// shared memory tiling
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS>
__global__ void
sgemm_smem_tile_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int M, int K, int N) {
  __shared__ float As[BLOCK_M][BLOCK_K];
  __shared__ float Bs[BLOCK_K][BLOCK_N];

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int AsLoadIter = (BLOCK_M * BLOCK_K) / TB_SIZE;
  constexpr int BsLoadIter = (BLOCK_K * BLOCK_N) / TB_SIZE;

  int tid = threadIdx.x;
  int tid_m = tid / BLOCK_N;
  int tid_n = tid % BLOCK_N;

  // pointers move of A and B, save registers
  {
    int bid = blockIdx.x;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    int A_offs = bid / grid_n * BLOCK_M * K;
    int B_offs = bid % grid_n * BLOCK_N;
    A += A_offs;
    B += B_offs;

    int C_offs = bid / grid_n * BLOCK_M * N + bid % grid_n * BLOCK_N;
    C += C_offs;
  }

  auto gmem_to_smem = [&]() {
    // load A global memory to shared memory
    for (int j = 0; j < AsLoadIter; ++j) {
      int index = tid + j * TB_SIZE;
      int As_m_idx = index / BLOCK_K;
      int As_k_idx = index % BLOCK_K;
      As[As_m_idx][As_k_idx] = A[As_m_idx * K + As_k_idx];
    }
    // load B global memory to shared memory
    for (int j = 0; j < BsLoadIter; ++j) {
      int index = tid + j * TB_SIZE;
      int Bs_k_idx = index / BLOCK_N;
      int Bs_n_idx = index % BLOCK_N;
      Bs[Bs_k_idx][Bs_n_idx] = B[Bs_k_idx * N + Bs_n_idx];
    }
  };

  float acc = 0.0f;
  auto  compute = [&]() {
    // compute
    for (int ki = 0; ki < BLOCK_K; ++ki) {
      float a_val = As[tid_m][ki];
      float b_val = Bs[ki][tid_n];
      acc += a_val * b_val;
    }
  };

  for (int i = 0; i < K / BLOCK_K; ++i) {
    gmem_to_smem();
    __syncthreads();
    compute();
    __syncthreads();
    // next iter
    A += BLOCK_K;
    B += BLOCK_K * N;
  }
  // write back to global memory
  C[tid_m * N + tid_n] = acc;
}

// Launcher function
torch::Tensor sgemm_launcher(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D matrices");
  TORCH_CHECK(A.size(1) == B.size(0), "shape mismatch: A.cols must equal B.rows");
  TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

  // Make sure data is contiguous
  auto lhs = A.contiguous();
  auto rhs = B.contiguous();

  const int M = lhs.size(0);
  const int K = lhs.size(1);
  const int N = rhs.size(1);

  constexpr int BLOCK_M = 16;
  constexpr int BLOCK_N = 8;
  constexpr int BLOCK_K = 64;

  torch::Tensor C = torch::empty({M, N}, lhs.options());

  constexpr int NUM_WARPS = 4;
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  static_assert(BLOCK_M * BLOCK_N == TB_SIZE, "BLOCK_M * BLOCK_N must equal TB_SIZE");

  dim3 block(NUM_WARPS * WARP_SIZE);
  dim3 grid((M * N + block.x - 1) / block.x);
  // sgemm_naive_kernel<NUM_WARPS>
  //     <<<grid, block>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
  sgemm_smem_tile_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS>
      <<<grid, block>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return C;
}
