import torch
from gemv_inline import gemv_cublas, gemv_cuda

torch.set_printoptions(precision=6, sci_mode=False)


def run_test(a, b, test_name):
  # print("=" * 80)
  # print(f"{test_name}")
  # print("-" * 80)
  # print(f"a.shape       = {a.shape}, a.dtype = {a.dtype}, a.device = {a.device}")
  # print(f"b.shape       = {b.shape}, b.dtype = {b.dtype}, b.device = {b.device}")

  # Torch reference
  # torch_c = torch.matmul(a, b.t())
  torch_c = gemv_cublas(a, b)
  c = gemv_cuda(a, b)

  diff = torch_c - c
  max_abs_diff = diff.abs().max().item()
  max_rel_diff = (diff.abs() / (torch_c.abs() + 1e-12)).max().item()

  # print(f"max abs diff  = {max_abs_diff:.6e}")
  # print(f"max rel diff  = {max_rel_diff:.6e}")
  # print(f"allclose?     = {torch.allclose(torch_c, c, rtol=1e-4, atol=1e-4)}")
  if not torch.allclose(torch_c, c, rtol=1e-4, atol=1e-4):
    print(f"Test FAILED: {test_name}")

  # Optionally show a few entries
  # print("\nFirst 5 entries (torch_c vs c):")
  # torch_flat = torch_c.view(-1)
  # c_flat = c.view(-1)
  # num_print = min(5, torch_flat.numel())
  # for i in range(num_print):
  #     print(f"  [{i:2d}] torch={torch_flat[i].item(): .6e}  gemv={c_flat[i].item(): .6e}  "
  #           f"diff={(torch_flat[i]-c_flat[i]).item(): .6e}")
  # print()


def time_pytorch_function(func, a, b):
  """
  Measure the execution time of a PyTorch function.

  Args:
      func (callable): The PyTorch function to be timed.
      input: The input to the function.

  Returns:
      float: The execution time in milliseconds.
  """
  # Since CUDA is asynchronous, we can't use Python's time module to measure time.
  # Instead, we use PyTorch's CUDA events to measure the time.
  start = torch.cuda.Event(enable_timing=True)  # Create a start event
  end = torch.cuda.Event(enable_timing=True)  # Create an end event

  # Perform a warmup to ensure the GPU is ready
  for _ in range(5):
    func(a, b)  # Run the function 5 times to warm up the GPU
  torch.cuda.synchronize()  # Wait for the kernel to finish

  # Start the timer
  start.record()
  for _ in range(1000):
    func(a, b)  # Run the function to be timed
  end.record()  # Stop the timer
  torch.cuda.synchronize()  # Wait for the kernel to finish
  return start.elapsed_time(end) # Return the elapsed time in milliseconds


for n in [3, 5, 8, 10, 16, 25, 32, 40, 64, 100, 128, 159, 256, 280, 512, 1024, 2048, 4096, 8192]:
  a = torch.rand(15384, n, device="cuda")
  b = torch.rand(1, n, device="cuda")
  cublas_time = time_pytorch_function(torch.matmul, a, b.t().contiguous())
  kernel_time = time_pytorch_function(gemv_cuda, a, b)
  print(f"n={n:4d} | GEMV Kernel Time: {kernel_time:8.3f} ms | torch.matmul Time: {cublas_time:8.3f} ms | Speedup: {cublas_time / kernel_time:6.2f}x")
  run_test(a, b, f"Variable n Test: (15384, {n}) @ ({n},)")
