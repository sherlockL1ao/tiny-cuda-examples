"""Sanity test for the NVFP4 quantize/dequantize scaffold."""
"""Sanity test for the NVFP4 quantize/dequantize scaffold."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from nvfp4_quant_inline import nvfp4_dequantize, nvfp4_quantize


def float_to_fp8_e4m3fn_ceil(x: torch.Tensor) -> torch.Tensor:
    """Mirrors CUDA float_to_fp8_ceil: round-toward-positive-infinity for FP8 E4M3."""
    x = x.clamp_min(0.0)
    y = x.to(torch.float8_e4m3fn)
    y_bits = y.view(torch.uint8)
    yf = y.float()
    # If the converted value is less than the original and we haven't hit max finite (0x7E), bump up.
    needs_ceil = (yf < x) & (y_bits < 0x7E)
    y_bits_ceil = torch.where(needs_ceil, y_bits + 1, y_bits)
    return y_bits_ceil.view(torch.float8_e4m3fn)


def nvfp4_quantize_reference(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation of NVFP4 quantization.

    Matches the CUDA kernel semantics:
    - group_size = 16 contiguous elements share one scale
    - absmax computed in float32
    - scale = absmax / 6.0 (or 1.0 if absmax == 0), stored as FP8 with ceil rounding
    - scale stored as float8_e4m3fn
    - quantize x * scale_used to FP4 E2M1 with round-to-nearest-even
    - pack 2 FP4 nibbles per byte: lower nibble = even index, upper nibble = odd index
    """
    M, N = x.shape
    assert N % group_size == 0, "N must be divisible by group_size"
    assert N % 2 == 0, "N must be even for packing"

    # E2M1 positive values: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    e2m1_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=x.device)

    # Reshape to (M, N//group_size, group_size)
    x_grouped = x.view(M, N // group_size, group_size)

    # Compute absmax per group in float32
    x_f32 = x_grouped.float()
    absmax = x_f32.abs().amax(dim=2, keepdim=True)  # (M, N//group_size, 1)

    # Compute scale: absmax / 6.0 (handle zero case)
    scale_f32 = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 6.0)

    # Convert to float8_e4m3fn with ceil rounding
    s_group = float_to_fp8_e4m3fn_ceil(scale_f32.squeeze(2))  # (M, N//group_size)

    # Use the quantized scale for quantization: scale_used = 1 / s_group
    scale_used = (1.0 / s_group.float()).unsqueeze(2)  # (M, N//group_size, 1)

    # Apply scaling
    x_scaled = x_f32 * scale_used  # (M, N//group_size, group_size)

    # Extract sign and magnitude
    sign = (x_scaled < 0).to(torch.uint8)  # 1 if negative, 0 if positive
    mag = x_scaled.abs()

    # Quantize magnitude using round-to-nearest-even
    # Compute distance to each E2M1 value
    mag_expanded = mag.unsqueeze(-1)  # (..., 1)
    dists = (mag_expanded - e2m1_values).abs()  # (..., 8)

    # Find minimum distance
    min_dist, indices = dists.min(dim=-1)

    # Handle ties: when distance is equal, choose the even index (round-to-nearest-even)
    # Check if there's a tie with the next higher value
    next_indices = (indices + 1).clamp(max=7)
    next_dists = dists.gather(-1, next_indices.unsqueeze(-1)).squeeze(-1)
    is_tie = (next_dists - min_dist).abs() < 1e-7  # tolerance for floating point comparison
    # If tie and next index is even, use next index
    indices = torch.where(is_tie & (next_indices % 2 == 0), next_indices, indices)
    indices = indices.to(torch.uint8)

    # Combine sign and magnitude: bit 3 is sign, bits 0-2 are magnitude
    fp4_values = indices | (sign << 3)  # 0..15

    # Pack pairs: (even, odd) -> byte
    fp4_flat = fp4_values.view(M, N)
    even_nibbles = fp4_flat[:, ::2]   # shape (M, N//2)
    odd_nibbles = fp4_flat[:, 1::2]   # shape (M, N//2)

    # Pack: lower 4 bits = even index, upper 4 bits = odd index
    packed_w = even_nibbles | (odd_nibbles << 4)

    return packed_w, s_group

def test_corner_case():
    """Test a small known input to verify packing order."""
    print("\n=== Testing Corner Case ===")
    # Create a simple input with known values
    group_size = 16

    # Simple pattern: [0, 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6, 0.5, 1.5, 3]
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0,
                       -2.0, -3.0, -4.0, -5.0, -6.0, 0.5, 1.5, 3.0]],
                     dtype=torch.bfloat16, device="cuda")

    print(f"Input: {x}")

    # Run kernel
    packed_k, scale_k = nvfp4_quantize(x, group_size)

    # Run reference
    packed_r, scale_r = nvfp4_quantize_reference(x, group_size)

    print(f"Kernel   packed: {packed_k[0, :8].cpu()}")
    print(f"Reference packed: {packed_r[0, :8].cpu()}")
    print(f"Kernel   scale: {scale_k[0, 0]}")
    print(f"Reference scale: {scale_r[0, 0]}")

    # Compare
    if not torch.equal(packed_k, packed_r):
        print("❌ Packed values don't match!")
        print(f"First mismatch at: {(packed_k != packed_r).nonzero()[0]}")
    else:
        print("✓ Packed values match!")

    if not torch.equal(scale_k.view(torch.uint8), scale_r.view(torch.uint8)):
        print("❌ Scales don't match!")
    else:
        print("✓ Scales match!")


def test_scaffold():
    print("Testing NVFP4 quantize/dequantize scaffold...")

    M, N = 64, 128
    group_size = 16

    print(f"\nCreating input tensor: [{M}, {N}] bfloat16")
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    print("Running nvfp4_quantize (CUDA kernel)...")
    packed_w, s_group = nvfp4_quantize(x, group_size)

    print(f"  packed_w shape: {packed_w.shape}, dtype: {packed_w.dtype}")
    assert packed_w.shape == (M, N // 2), f"Expected shape {(M, N // 2)}, got {packed_w.shape}"
    assert packed_w.dtype == torch.uint8, f"Expected uint8, got {packed_w.dtype}"

    print(f"  s_group shape: {s_group.shape}, dtype: {s_group.dtype}")
    assert s_group.shape == (M, N // group_size), f"Expected shape {(M, N // group_size)}, got {s_group.shape}"
    assert s_group.dtype == torch.float8_e4m3fn, f"Expected float8_e4m3fn, got {s_group.dtype}"

    print("\nRunning nvfp4_quantize_reference (PyTorch)...")
    packed_ref, s_group_ref = nvfp4_quantize_reference(x, group_size)

    print(f"  packed_ref shape: {packed_ref.shape}, dtype: {packed_ref.dtype}")
    print(f"  s_group_ref shape: {s_group_ref.shape}, dtype: {s_group_ref.dtype}")

    print("\nComparing kernel vs reference outputs...")

    # Compare scales (bitwise)
    scales_match = torch.equal(s_group.view(torch.uint8), s_group_ref.view(torch.uint8))
    if scales_match:
        print("  ✓ Scales match (bitwise)!")
    else:
        print("  ❌ Scales don't match!")
        mismatch_idx = (s_group.view(torch.uint8) != s_group_ref.view(torch.uint8)).nonzero()
        print(f"    First 5 mismatches at: {mismatch_idx[:5]}")
        for idx in mismatch_idx[:3]:
            i, j = idx[0].item(), idx[1].item()
            print(f"    [{i}, {j}] kernel={s_group.view(torch.uint8)[i, j].item()}, ref={s_group_ref.view(torch.uint8)[i, j].item()}")

    # Compare packed values (bitwise)
    packed_match = torch.equal(packed_w, packed_ref)
    if packed_match:
        print("  ✓ Packed values match (bitwise)!")
    else:
        print("  ❌ Packed values don't match!")
        mismatch_idx = (packed_w != packed_ref).nonzero()
        print(f"    Total mismatches: {mismatch_idx.shape[0]} / {packed_w.numel()}")
        print(f"    First 5 mismatches at: {mismatch_idx[:5]}")
        for idx in mismatch_idx[:3]:
            i, j = idx[0].item(), idx[1].item()
            k_val = packed_w[i, j].item()
            r_val = packed_ref[i, j].item()
            print(f"    [{i}, {j}] kernel={k_val:08b} ({k_val}), ref={r_val:08b} ({r_val})")

    if not (scales_match and packed_match):
        print("\n❌ Validation FAILED! Outputs don't match.")
        sys.exit(1)

    print("\n✅ Validation PASSED! Kernel matches reference implementation.")
    return

    print("Running nvfp4_dequantize...")
    out = nvfp4_dequantize(packed_w, s_group, group_size)

    print(f"  out shape: {out.shape}, dtype: {out.dtype}")
    assert out.shape == (M, N), f"Expected shape {(M, N)}, got {out.shape}"
    assert out.dtype == torch.float16, f"Expected float16, got {out.dtype}"

    print("\n✓ All shape and dtype checks passed!")
    print(f"  Stub outputs: packed_w[0,0]={packed_w[0, 0].item()}, s_group[0,0]={s_group[0, 0]}, out[0,0]={out[0, 0].item()}")
    print("  (As expected from stub: packed=0, group_scale=0, out=0.0)")

    print("\nTesting input validation (N not divisible by 2)...")
    try:
        bad_x = torch.randn(M, 127, dtype=torch.bfloat16, device="cuda")
        nvfp4_quantize(bad_x, group_size)
        print("  ERROR: Should have raised exception!")
        sys.exit(1)
    except RuntimeError as e:
        print(f"  ✓ Caught expected error: {e}")

    print("\nTesting input validation (N not divisible by group_size)...")
    try:
        bad_x = torch.randn(M, 126, dtype=torch.bfloat16, device="cuda")
        nvfp4_quantize(bad_x, group_size)
        print("  ERROR: Should have raised exception!")
        sys.exit(1)
    except RuntimeError as e:
        print(f"  ✓ Caught expected error: {e}")

    print("\n✅ All tests passed! Scaffold is working correctly.")


if __name__ == "__main__":
    test_corner_case()
    test_scaffold()
