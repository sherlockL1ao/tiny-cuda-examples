from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from torch.utils.cpp_extension import load


@dataclass(frozen=True)
class InlineSource:
  """Inline code blob that will be materialized to a temp file for compilation."""

  code: str
  ext: str  # ".cpp" or ".cu"
  name: str | None = None

  def _filename(self, index: int) -> str:
    suffix = self.ext if self.ext.startswith(".") else f".{self.ext}"
    digest = hashlib.sha1(self.code.encode("utf-8")).hexdigest()[:8]
    stem = self.name or "inline"
    return f"{stem}_{digest}_{index}{suffix}"


SourceLike = str | Path | InlineSource


def _resolve_sources(module_name: str, sources: Sequence[SourceLike], base_dir: Path | None) -> list[Path]:
  resolved: list[Path] = []
  inline_root: Path | None = None

  for idx, src in enumerate(sources):
    if isinstance(src, InlineSource):
      if inline_root is None:
        # Always place inline temp files under the system temp dir to avoid
        # cluttering the project tree even when a base_dir is provided.
        inline_root = Path(tempfile.gettempdir()) / f".{module_name}_inline"
        inline_root.mkdir(parents=True, exist_ok=True)
      path = inline_root / src._filename(idx)
      # Refresh content if it changed to keep torch's build cache in sync.
      if not path.exists() or path.read_text() != src.code:
        path.write_text(src.code)
      resolved.append(path)
      continue

    path = Path(src)
    if not path.is_absolute() and base_dir is not None:
      path = base_dir / path
    path = path.resolve()
    if not path.exists():
      raise FileNotFoundError(f"Missing source file: {path}")
    resolved.append(path)
  return resolved


def build_extension(
  module_name: str,
  sources: Sequence[SourceLike],
  *,
  base_dir: str | Path | None = None,
  verbose: bool = True,
  extra_cflags: Sequence[str] | None = None,
  extra_cuda_cflags: Sequence[str] | None = None,
) -> Any:
  """Compile given sources into a PyTorch extension via torch.utils.cpp_extension.load.

  Args:
    module_name: Unique name for the compiled extension.
    sources: Collection of C++/CUDA source specifications. Entries can be
      filesystem paths or InlineSource instances containing in-memory code.
    base_dir: Optional directory to resolve relative source paths against.
    verbose: Forwarded to torch.utils.cpp_extension.load.
    extra_cflags: Optional list of extra flags for the host compiler.
    extra_cuda_cflags: Optional list of extra flags for NVCC.
  """

  base_dir_path = Path(base_dir).resolve() if base_dir is not None else None
  resolved_sources = _resolve_sources(module_name, list(sources), base_dir_path)

  return load(
    name=module_name, sources=[str(p) for p in resolved_sources], extra_cflags=extra_cflags, extra_cuda_cflags=extra_cuda_cflags, verbose=verbose
  )
