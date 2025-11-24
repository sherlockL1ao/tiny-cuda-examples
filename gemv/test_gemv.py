import torch
from gemv_ext import gemv # pyright: ignore[reportMissingImports]

a = torch.randn(32, 128, device='cuda')
b = torch.randn(1, 128, device='cuda')

torch_c = torch.matmul(a, b.t())

c = gemv(a, b)
print((torch_c - c).abs().max())
