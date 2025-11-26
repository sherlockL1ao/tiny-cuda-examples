from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load

cuda_path = Path("gemv.cu")
cpp_path = Path("gemv_interface.cpp")


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

gemv_module = build_extension(cuda_path, cpp_path, module_name="gemv_ext")
gemv_cuda = gemv_module.gemv
