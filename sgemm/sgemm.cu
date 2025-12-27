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
  for (int i = 0; i < K; ++i) {
    float a_val = A[m_idx * K + i];
    float b_val = B[i * N + n_idx];
    acc += a_val * b_val;
  }
  C[m_idx * N + n_idx] = acc;
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

  torch::Tensor C = torch::empty({M, N}, lhs.options());

  constexpr int NUM_WARPS = 4;

  dim3 block(NUM_WARPS * WARP_SIZE);
  dim3 grid((M * N + block.x - 1) / block.x);
  sgemm_naive_kernel<NUM_WARPS>
      <<<grid, block>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return C;
}
