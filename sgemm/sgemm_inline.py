from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load_inline

cuda_path = Path(__file__).parent / "sgemm.cu"

# C++ binding code as a string
cpp_source = """
#include <torch/extension.h>

torch::Tensor sgemm_launcher(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgemm", &sgemm_launcher, "SGEMM (CUDA)");
}
"""


def build_extension(cuda_path: Path, module_name: str, verbose: bool = True) -> Any:
    """Compile the CUDA source with inline C++ binding into a PyTorch extension."""
    if not cuda_path.exists(): raise FileNotFoundError(f"Missing CUDA source file: {cuda_path}")

    return load_inline(
        name=module_name,
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_path.read_text()],
        extra_cuda_cflags=["-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr"],# "--ptxas-options=-v"],
        verbose=verbose,
    )


sgemm_module = build_extension(cuda_path, module_name="sgemm_ext")
sgemm_cuda = sgemm_module.sgemm
