from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch.utils.cpp_extension import load

input_t: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
output_t: TypeAlias = torch.Tensor


cuda_path = Path("nvfp4_gemv.cu")
cpp_path = Path("nvfp4_gemv_interface.cpp")


def build_extension(cpp_path: Path, cuda_path: Path, module_name: str, verbose: bool = True) -> Any:
  """Compile the external C++/CUDA sources into a PyTorch extension."""
  if not cpp_path.exists(): raise FileNotFoundError(f"Missing C++ source file: {cpp_path}")
  if not cuda_path.exists(): raise FileNotFoundError(f"Missing CUDA source file: {cuda_path}")

  return load(
    name=module_name,
    sources=[str(cpp_path), str(cuda_path)],
    # Flags for the C++ wrapper
    # extra_cflags=["-g", "-O0"],
    # Flags for the NVCC compiler
    # extra_cuda_cflags=["-g", "-G", "-O0"],
    verbose=verbose,
  )


nvfp4_gemv_module = build_extension(cuda_path, cpp_path, module_name="nvfp4_gemv_ext")


def nvfp4_gemv_cuda(data: input_t) -> output_t:
  a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
  _, _, l = c_ref.shape
  scale_a = sfa_ref_cpu
  scale_b = sfb_ref_cpu
  nvfp4_gemv_module.nvfp4_gemv(a_ref, b_ref, scale_a.cuda(), scale_b.cuda(), c_ref)
  return c_ref
