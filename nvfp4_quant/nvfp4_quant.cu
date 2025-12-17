#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>

constexpr int   kWarpSize = 32;
constexpr int   kBlockSize = 16;
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;


__device__ void ldcs_i32x4(uint32_t* dst, const void* src) {
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
}

__device__ __forceinline__ void stg_i32x2(void* dst, uint32_t a, uint32_t b) {
  asm volatile("st.global.v2.b32 [%0], {%1, %2};" : : "l"(dst), "r"(a), "r"(b) : "memory");
}

__device__ __forceinline__ uint8_t quant_pack2_nvfp4_e2m1(__nv_bfloat162 a, float scale) {
  // 1) apply scale in float
  float2 f = __bfloat1622float2(a);
  f.x *= scale;
  f.y *= scale;

  // 2) convert + saturate + round, returning packed fp4x2 storage (8-bit)
  __nv_fp4x2_storage_t p = __nv_cvt_float2_to_fp4x2(f, __NV_E2M1, cudaRoundNearest);

  // 3) treat that storage as a byte
  return static_cast<uint8_t>(p);
}

//=============================================================================
// Quantize Kernel (Stub)
//=============================================================================
template <int NUM_WARPS>
__global__ void Nvfp4QuantizeKernelV1(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ nvfp4_x,
    __nv_fp8_e4m3* __restrict__ block_sf,
    const int M,
    const int N) {
  int tid = threadIdx.x; // 0..256
  int global_idx = (blockIdx.x * blockDim.x + tid) * kBlockSize;

  constexpr int TB_SIZE = NUM_WARPS * kWarpSize;

  {
    int x_off = global_idx;
    x += x_off;

    int nvfp4_x_off = global_idx / 2;
    nvfp4_x += nvfp4_x_off;

    int sf_off = global_idx / kBlockSize;
    block_sf += sf_off;
  }

  int grid_stride = gridDim.x * blockDim.x * kBlockSize;
  int stride_x_elems = grid_stride;
  int stride_nvfp4_elems = grid_stride / 2;
  int stride_sf_elems = grid_stride / kBlockSize;
  int num_iters = (M * N + grid_stride - 1) / grid_stride;

  uint32_t frag_x[2][4];

#pragma unroll
  for (int i = 0; i < num_iters; ++i) {
    if (global_idx >= M * N) break;

    ldcs_i32x4(frag_x[0], x);
    ldcs_i32x4(frag_x[1], x + 8);
    // compute absmax
    __nv_bfloat162 absmax_acc = __floats2bfloat162_rn(0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      __nv_bfloat162 val = *reinterpret_cast<__nv_bfloat162*>(&frag_x[j / 4][j % 4]);
      val = __habs2(val);
      absmax_acc = __hmax2(absmax_acc, val);
    }
    float2 temp = __bfloat1622float2(absmax_acc);
    float  absmax = fmaxf(temp.x, temp.y);

    float scale_f = (absmax == 0.0f) ? 1.0f : (kE2M1Max / absmax);
    // TODO(xingyu): consider to use round-up
    __nv_fp8_storage_t scale_fp8 = __nv_cvt_float_to_fp8(scale_f, __NV_SATFINITE, __NV_E4M3);

    // write scale to global memory
    // TODO(xingyu): vector store
    block_sf[0].__x = scale_fp8;

    // Quantize & packing
    __nv_fp8_e4m3 sf_v;
    sf_v.__x = scale_fp8;
    float sf_used = (float)sf_v;

    uint32_t packed[2];
    for (int k = 0; k < 2; ++k) {
      uint32_t b0 = quant_pack2_nvfp4_e2m1(*reinterpret_cast<__nv_bfloat162*>(&frag_x[k][0]), sf_used);
      uint32_t b1 = quant_pack2_nvfp4_e2m1(*reinterpret_cast<__nv_bfloat162*>(&frag_x[k][1]), sf_used);
      uint32_t b2 = quant_pack2_nvfp4_e2m1(*reinterpret_cast<__nv_bfloat162*>(&frag_x[k][2]), sf_used);
      uint32_t b3 = quant_pack2_nvfp4_e2m1(*reinterpret_cast<__nv_bfloat162*>(&frag_x[k][3]), sf_used);

      packed[k] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    stg_i32x2(nvfp4_x, packed[0], packed[1]);

    x += stride_x_elems;
    nvfp4_x += stride_nvfp4_elems;
    block_sf += stride_sf_elems;
    global_idx += stride_x_elems;
  }
}

//=============================================================================
// Dequantize Kernel (Stub)
//=============================================================================
__global__ void Nvfp4DequantizeKernel(
    const uint8_t* __restrict__ packed_w,
    const __nv_fp8_e4m3* __restrict__ s_group,
    __half* __restrict__ out,
    const int M,
    const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = M * N;

  if (idx < total_elements) {
    out[idx] = __float2half(0.0f);
  }
}

//=============================================================================
// Launchers
//=============================================================================
std::tuple<torch::Tensor, torch::Tensor> nvfp4_quantize_launcher(const torch::Tensor& x, int group_size) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(x.dim() == 2, "Input must be a 2D tensor");
  TORCH_CHECK(x.dtype() == torch::kBFloat16, "Input must be bfloat16");

  const int M = x.size(0), N = x.size(1);

  TORCH_CHECK(N % 2 == 0, "N must be divisible by 2 for packing");
  TORCH_CHECK(N % group_size == 0, "N must be divisible by group_size");
  TORCH_CHECK(group_size == 16, "group_size must be 16 for nvfp4 quantization");

  auto nvfp4_x = torch::empty({M, N / 2}, torch::dtype(torch::kUInt8).device(x.device()));
  auto block_sf = torch::empty({M, N / group_size}, torch::dtype(torch::kFloat8_e4m3fn).device(x.device()));

  constexpr int num_warps = 4;
  constexpr int TB_SIZE = num_warps * kWarpSize;

  int workloads = (M * N) / kBlockSize;
  int num_blocks = std::min(1024, (workloads + TB_SIZE - 1) / TB_SIZE);

  dim3 block(TB_SIZE);
  dim3 grid(num_blocks);
  Nvfp4QuantizeKernelV1<num_warps><<<grid, block>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      nvfp4_x.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_fp8_e4m3*>(block_sf.data_ptr()),
      M,
      N);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return std::make_tuple(nvfp4_x, block_sf);
}

torch::Tensor nvfp4_dequantize_launcher(const torch::Tensor& packed_w, const torch::Tensor& s_group, int group_size) {
  TORCH_CHECK(packed_w.is_cuda(), "packed_w must be on CUDA device");
  TORCH_CHECK(s_group.is_cuda(), "s_group must be on CUDA device");
  TORCH_CHECK(packed_w.is_contiguous(), "packed_w must be contiguous");
  TORCH_CHECK(s_group.is_contiguous(), "s_group must be contiguous");

  TORCH_CHECK(packed_w.dim() == 2, "packed_w must be a 2D tensor");
  TORCH_CHECK(s_group.dim() == 2, "s_group must be a 2D tensor");

  TORCH_CHECK(packed_w.dtype() == torch::kUInt8, "packed_w must be uint8");
  TORCH_CHECK(s_group.dtype() == torch::kFloat8_e4m3fn, "s_group must be float8_e4m3fn");

  const int M = packed_w.size(0), N = packed_w.size(1) * 2;

  TORCH_CHECK(s_group.size(0) == M, "s_group M dimension mismatch");
  TORCH_CHECK(s_group.size(1) == N / group_size, "s_group N dimension mismatch");

  auto out = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(packed_w.device()));

  dim3 block(256);
  dim3 grid((M * N + block.x - 1) / block.x);
  Nvfp4DequantizeKernel<<<grid, block>>>(
      packed_w.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_fp8_e4m3*>(s_group.data_ptr()),
      reinterpret_cast<__half*>(out.data_ptr()),
      M,
      N);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return out;
}
