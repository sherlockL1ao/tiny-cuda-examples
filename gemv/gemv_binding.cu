#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "gemv.cuh"

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

  dim3 block(32, 4); // 128 threads, 4 warps
  dim3 grid((m + block.y - 1) / block.y);
  SgemvK32<<<grid, block, 0, stream>>>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), C.data_ptr<float>(), m, k);

  // Surface CUDA errors (helps debugging when called from Python).
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  // Catch runtime errors from the kernel (useful during development).
  err = cudaStreamSynchronize(stream);
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return C;
}

PYBIND11_MODULE(gemv_ext, m) {
  m.def("gemv", &gemv_launcher, "Tiled GEMM (CUDA)");
}
