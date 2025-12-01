#include <torch/extension.h>
#include <cuda_runtime.h>


// __global__ void

torch::Tensor nvfp4_gemv_launcher(
    const torch::Tensor& a, // mat
    const torch::Tensor& b, // vec
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    torch::Tensor        out) {
  return out;
}
