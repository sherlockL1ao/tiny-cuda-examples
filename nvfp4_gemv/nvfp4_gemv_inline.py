from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch.utils.cpp_extension import load_inline

cuda_path = Path("nvfp4_gemv.cu")

cpp_code = """
#include <torch/extension.h>

torch::Tensor nvfp4_gemv_asm_launcher(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    torch::Tensor        out);

torch::Tensor nvfp4_gemv_asmv2_launcher(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    torch::Tensor        out);

torch::Tensor nvfp4_gemv_naive_launcher(
    const torch::Tensor& a, // mat
    const torch::Tensor& b, // vec
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    torch::Tensor out);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_gemv_asm", &nvfp4_gemv_asm_launcher, "Nvfp4 GEMV ASM (CUDA)");
  m.def("nvfp4_gemv_asmv2", &nvfp4_gemv_asmv2_launcher, "Nvfp4 GEMV ASMV2 (CUDA)");
  m.def("nvfp4_gemv_naive", &nvfp4_gemv_naive_launcher, "Nvfp4 GEMV Naive (CUDA)");
}
"""


def build_extension(cpp_path: str, cuda_path: Path, module_name: str, verbose: bool = True) -> Any:
    """Compile the external C++/CUDA sources into a PyTorch extension."""
    #   if not cpp_path.exists(): raise FileNotFoundError(f"Missing C++ source file: {cpp_path}")
    #   if not cuda_path.exists(): raise FileNotFoundError(f"Missing CUDA source file: {cuda_path}")

    return load_inline(
        name=module_name,
        cpp_sources=[cpp_path],
        cuda_sources=[cuda_path.read_text()],
        # Flags for the C++ wrapper
        # extra_cflags=["-g", "-O0"],
        # Flags for the NVCC compiler
        # extra_cuda_cflags=["-g", "-G", "-O0"],
        extra_cuda_cflags=["-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr"],
        verbose=verbose,
    )

nvfp4_gemv_module = build_extension(cpp_code, cuda_path, module_name="nvfp4_gemv_ext")

input_t: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
output_t: TypeAlias = torch.Tensor


def nvfp4_gemv_naive(data: input_t) -> output_t:
    """Run nvfp4 GEMV using an inline-compiled C++/CUDA extension.

    Args:
      data: Tuple of tensors consumed by the kernel.
      cpp_source: Optional override for the C++ binding code. Defaults to the
        built-in `cpp_code` string.
      cuda_source: Optional override for the CUDA kernel code. If not provided,
        the contents of `nvfp4_gemv.cu` are used when available.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    # [128, k, l] -> [1, k, l]
    # [128, k/vec_size, l] -> [1, k/vec_size, l]
    nvfp4_gemv_module.nvfp4_gemv_naive(a_ref, b_ref, sfa_ref_cpu.cuda(), sfb_ref_cpu.cuda(), c_ref)
    return c_ref


def nvfp4_gemv_asm(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    nvfp4_gemv_module.nvfp4_gemv_asm(a_ref, b_ref, sfa_ref_cpu.cuda(), sfb_ref_cpu.cuda(), c_ref)
    return c_ref


def nvfp4_gemv_asmv2(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    nvfp4_gemv_module.nvfp4_gemv_asmv2(a_ref, b_ref, sfa_ref_cpu.cuda(), sfb_ref_cpu.cuda(), c_ref)
    return c_ref
