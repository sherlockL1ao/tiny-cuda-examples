import torch
from sgemm_inline import sgemm_cuda

torch.set_printoptions(precision=6, sci_mode=False)


def run_test(a, b, test_name):
    """Test custom SGEMM against PyTorch reference."""
    # PyTorch reference
    torch_c = torch.matmul(a, b)
    c = sgemm_cuda(a, b)

    diff = torch_c - c
    max_abs_diff = diff.abs().max().item()
    max_rel_diff = (diff.abs() / (torch_c.abs() + 1e-12)).max().item()

    if not torch.allclose(torch_c, c, rtol=1e-4, atol=1e-4):
        print(f"Test FAILED: {test_name}")
        print(f"  max abs diff = {max_abs_diff:.6e}")
        print(f"  max rel diff = {max_rel_diff:.6e}")
    else:
        print(f"Test PASSED: {test_name}")


def time_pytorch_function(func, a, b):
    """
    Measure the execution time of a PyTorch function.

    Args:
        func (callable): The PyTorch function to be timed.
        a, b: Input tensors

    Returns:
        float: The execution time in milliseconds.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(a, b)
    torch.cuda.synchronize()

    # Measure
    start.record()
    for _ in range(100):
        func(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


if __name__ == "__main__":
    print("=" * 80)
    print("SGEMM Correctness Tests")
    print("=" * 80)

    # Small matrices
    a = torch.rand(16, 16, device="cuda")
    b = torch.rand(16, 16, device="cuda")
    run_test(a, b, "Small square (16x16) @ (16x16)")

    a = torch.rand(32, 64, device="cuda")
    b = torch.rand(64, 32, device="cuda")
    run_test(a, b, "Small rectangular (32x64) @ (64x32)")

    # Medium matrices
    a = torch.rand(128, 128, device="cuda")
    b = torch.rand(128, 128, device="cuda")
    run_test(a, b, "Medium square (128x128) @ (128x128)")

    a = torch.rand(256, 512, device="cuda")
    b = torch.rand(512, 256, device="cuda")
    run_test(a, b, "Medium rectangular (256x512) @ (512x256)")

    # Large matrices
    a = torch.rand(512, 512, device="cuda")
    b = torch.rand(512, 512, device="cuda")
    run_test(a, b, "Large square (512x512) @ (512x512)")

    a = torch.rand(1024, 1024, device="cuda")
    b = torch.rand(1024, 1024, device="cuda")
    run_test(a, b, "Large square (1024x1024) @ (1024x1024)")

    print("\n" + "=" * 80)
    print("SGEMM Performance Benchmarks")
    print("=" * 80)

    # Benchmark different sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for size in sizes:
        a = torch.rand(size, size, device="cuda")
        b = torch.rand(size, size, device="cuda")

        torch_time = time_pytorch_function(torch.matmul, a, b)
        kernel_time = time_pytorch_function(sgemm_cuda, a, b)

        print(f"Size: {size:4d}x{size:4d} | Custom: {kernel_time:8.3f} ms | PyTorch: {torch_time:8.3f} ms | Ratio: {kernel_time / torch_time:6.2f}x")

    print("\n" + "=" * 80)
    print("Non-square matrix benchmarks")
    print("=" * 80)

    test_shapes = [
        (128, 256, 128),  # (M, K, N)
        (256, 128, 256),
        (512, 256, 128),
        (1024, 512, 256),
    ]

    for M, K, N in test_shapes:
        a = torch.rand(M, K, device="cuda")
        b = torch.rand(K, N, device="cuda")

        torch_time = time_pytorch_function(torch.matmul, a, b)
        kernel_time = time_pytorch_function(sgemm_cuda, a, b)

        print(
            f"Shape: ({M:4d},{K:4d}) @ ({K:4d},{N:4d}) | "
            f"Custom: {kernel_time:8.3f} ms | "
            f"PyTorch: {torch_time:8.3f} ms | "
            f"Ratio: {kernel_time / torch_time:6.2f}x"
        )
