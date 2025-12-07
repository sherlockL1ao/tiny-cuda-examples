import torch
from nvfp4_gemv_inline import nvfp4_gemv_asm, nvfp4_gemv_asmv2, nvfp4_gemv_naive, nvfp4_gemv_asm_warp
from nvfp4_gemv_reference import generate_input, ref_kernel
from utils import make_match_reference


def time_pytorch_function(func, data):
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
    func(data)  # Run the function 5 times to warm up the GPU
  torch.cuda.synchronize()  # Wait for the kernel to finish

  # Start the timer
  start.record()
  for _ in range(100):
    func(data)  # Run the function to be timed
  end.record()  # Stop the timer
  torch.cuda.synchronize()  # Wait for the kernel to finish
  return start.elapsed_time(end) # Return the elapsed time in milliseconds


m, k, l = 128, 512, 4

data_ref = generate_input(m=m, k=k, l=l, seed=42)
# ref_out = ref_kernel(data_ref)
ref_out = nvfp4_gemv_asmv2(data_ref)

data_cuda = generate_input(m=m, k=k, l=l, seed=42)
check_impl = make_match_reference(nvfp4_gemv_asm_warp, rtol=1e-03, atol=1e-03)
matched, msg = check_impl(data_cuda, ref_out)

if not matched:
    print(msg)
    exit()


for m, k in ((2048, 512), (512, 2048), (8192, 8192)):
    data = generate_input(m=m, k=k, l=32, seed=42)
    torch_time = time_pytorch_function(ref_kernel, data)
    naive_time = time_pytorch_function(nvfp4_gemv_naive, data)
    asm_time = time_pytorch_function(nvfp4_gemv_asm, data)
    asmv2_time = time_pytorch_function(nvfp4_gemv_asmv2, data)
    asm_warp_time = time_pytorch_function(nvfp4_gemv_asm_warp, data)
    print(
        f"m={m}, k={k} | "
        f"PyTorch: {torch_time:.2f} ms | "
        f"Naive: {naive_time:.2f} ms | "
        f"ASM: {asm_time:.2f} ms | "
        f"ASMV2: {asmv2_time:.2f} ms | "
        f"ASM warp: {asm_warp_time:.2f} ms"
    )
