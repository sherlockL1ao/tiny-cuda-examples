#include <torch/extension.h>

torch::Tensor nvfp4_gemv_launcher(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    torch::Tensor        out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_gemv", &nvfp4_gemv_launcher, "Nvfp4 GEMV (CUDA)");
}
