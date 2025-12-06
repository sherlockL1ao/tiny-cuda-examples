from pathlib import Path
from typing import TypeAlias

import torch
from jit_utils import InlineSource, build_extension

input_t: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
output_t: TypeAlias = torch.Tensor


_BASE_DIR = Path(__file__).resolve().parent
cuda_path = _BASE_DIR / "nvfp4_gemv.cu"
cpp_code = """
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
"""

try: cuda_code = cuda_path.read_text()
except FileNotFoundError:
    # Leave None so we can surface a helpful error at call time if the caller
    # does not provide inline CUDA source.
    cuda_code = None


def nvfp4_gemv_cuda(
    data: input_t,
    *,
    cpp_source: str | None = None,
    cuda_source: str | None = None,
) -> output_t:
    """Run nvfp4 GEMV using an inline-compiled C++/CUDA extension.

    Args:
      data: Tuple of tensors consumed by the kernel.
      cpp_source: Optional override for the C++ binding code. Defaults to the
        built-in `cpp_code` string.
      cuda_source: Optional override for the CUDA kernel code. If not provided,
        the contents of `nvfp4_gemv.cu` are used when available.
    """

    cpp_inline = InlineSource(code=cpp_source or cpp_code, ext=".cpp", name="nvfp4_gemv_interface")
    cuda_src = cuda_source or cuda_code
    if cuda_src is None:
        raise FileNotFoundError("CUDA source unavailable. Provide `cuda_source` or place nvfp4_gemv.cu next to nvfp4_gemv_inline.py.")
    cuda_inline = InlineSource(code=cuda_src, ext=".cu", name="nvfp4_gemv_kernel")

    # torch.utils.cpp_extension.load handles caching and will only compile if needed
    nvfp4_gemv_module = build_extension(module_name="nvfp4_gemv_ext", sources=[cpp_inline, cuda_inline])
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    # [128, k, l] -> [1, k, l]
    # [128, k/vec_size, l] -> [1, k/vec_size, l]
    nvfp4_gemv_module.nvfp4_gemv(a_ref, b_ref, sfa_ref_cpu.cuda(), sfb_ref_cpu.cuda(), c_ref)
    return c_ref
