from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

cuda_path = Path(__file__).resolve().with_name("nvfp4_quant.cu")

cpp_code = """
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> nvfp4_quantize_launcher(
    const torch::Tensor& x,
    int group_size);

torch::Tensor nvfp4_dequantize_launcher(
    const torch::Tensor& packed_w,
    const torch::Tensor& s_group,
    int group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_quantize", &nvfp4_quantize_launcher, "NVFP4 Quantize (CUDA)");
  m.def("nvfp4_dequantize", &nvfp4_dequantize_launcher, "NVFP4 Dequantize (CUDA)");
}
"""


def build_extension(cpp_path: str, cuda_path: Path, module_name: str, verbose: bool = True) -> Any:
    """Compile the external C++/CUDA sources into a PyTorch extension."""
    return load_inline(
        name=module_name,
        cpp_sources=[cpp_path],
        cuda_sources=[cuda_path.read_text()],
        extra_cuda_cflags=["-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr"],
        verbose=verbose,
    )


nvfp4_quant_module = build_extension(cpp_code, cuda_path, module_name="nvfp4_quant_ext")


def nvfp4_quantize(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 tensor to NVFP4 format with group scaling.

    Args:
        x: Input tensor [M, N] in bfloat16
        group_size: Size of quantization groups (default: 16)

    Returns:
        Tuple of (packed_w_u8, s_group_f8)
    """
    return nvfp4_quant_module.nvfp4_quantize(x, group_size)


def nvfp4_dequantize(packed_w: torch.Tensor, s_group: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Dequantize NVFP4 packed tensor back to float16.

    Args:
        packed_w: Packed uint8 tensor [M, N/2]
        s_group: Group scale factors [M, N/group_size] in float8_e4m3fn
        group_size: Size of quantization groups (default: 16)

    Returns:
        Dequantized tensor [M, N] in float16
    """
    return nvfp4_quant_module.nvfp4_dequantize(packed_w, s_group, group_size)
