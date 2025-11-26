#include <torch/extension.h>

torch::Tensor gemv_launcher(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &gemv_launcher, "Tiled GEMV (CUDA)");
}
