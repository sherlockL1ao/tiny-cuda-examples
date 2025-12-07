#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define SF_VEC_SIZE 16
#define PACK_SIZE 2
#define VEC_SIZE 2
#define SF_PACK_SIZE (SF_VEC_SIZE / PACK_SIZE)

__device__ __forceinline__ void fp4x8_to_fp16x2x4(uint32_t* out, uint32_t in) {
  asm volatile(
    "{\n\t"
    ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n\t"
    "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4; // unpack 32-bit register to 4x fp4x2\n\t"
    "cvt.rn.f16x2.e2m1x2 %0, tmp0;\n\t"
    "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n\t"
    "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n\t"
    "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n\t"
    "}"
    : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
    : "r"(in)
  );
}

__device__ __forceinline__ void fp8x2_to_fp16x2(half2* out, uint16_t in) {
  uint32_t* out_i32 = reinterpret_cast<uint32_t *>(out);
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(out_i32[0]) : "h"(in));
}

// Load Cache All
__device__ __forceinline__ void ldcs_i16(uint16_t* dst, const void* src) {
  asm volatile("ld.global.L1::no_allocate.b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
}

// Load Cache Streaming
__device__ __forceinline__ void ldca_i16(uint16_t* dst, const void* src) {
  asm volatile("ld.global.L1::evict_last.b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
}

__device__ __forceinline__ void ldcs_i32x4(uint32_t* dst, const void* src) {
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
}

__device__ __forceinline__ void ldca_i32x4(uint32_t* dst, const void* src) {
  asm volatile("ld.global.L1::evict_last.v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
}

__global__ void Nvfp4gemvNaive(
    const void* __restrict__ a,
    const void* __restrict__ b,
    const void* __restrict__ scale_a,
    const void* __restrict__ scale_b,
    void* __restrict__ out,
    const int m,
    const int k,
    const int l) {
  int current_row = blockIdx.x * blockDim.x + threadIdx.x;
  int current_batch = blockIdx.y;
  if (current_row >= m) return;

  const int k_packed = k / PACK_SIZE;
  const int stride_l = m * k_packed;
  const int sf_k = k / SF_VEC_SIZE;
  float     sum = 0.f;
  float     acc = 0.f;
  for (int i = 0; i < k_packed; ++i) {
    int  a_off = current_batch * stride_l + current_row * k_packed + i;
    int  b_off = current_batch * 128 * k_packed + i;

    // convert packed fp4x2 → half2
    auto        a_packed = *(static_cast<const __nv_fp4x2_e2m1*>(a) + a_off);
    __half2_raw a_h2_raw = __nv_cvt_fp4x2_to_halfraw2(a_packed.__x, __NV_E2M1);
    __half2     a_h2 = *reinterpret_cast<__half2*>(&a_h2_raw);

    auto        b_packed = *(static_cast<const __nv_fp4x2_e2m1*>(b) + b_off);
    __half2_raw b_h2_raw = __nv_cvt_fp4x2_to_halfraw2(b_packed.__x, __NV_E2M1);
    __half2     b_h2 = *reinterpret_cast<__half2*>(&b_h2_raw);

    // 1. Multiply in half (Result is __half, potential precision loss/overflow)
    __half2 prod_half2 = __hmul2(a_h2, b_h2);
    // 2. Convert to float and Add
    float2 prod_f2 = __half22float2(prod_half2);
    acc += (prod_f2.x + prod_f2.y);
    if ((i + 1) % SF_PACK_SIZE == 0) {
      // finish one group
      int        group_idx = i / SF_PACK_SIZE;
      int        sfa_off = current_batch * sf_k * m + current_row * sf_k + group_idx;
      int        sfb_off = current_batch * sf_k * 128 + group_idx;
      auto       scale_a_val = *(static_cast<const __nv_fp8_e4m3*>(scale_a) + sfa_off);
      auto       scale_b_val = *(static_cast<const __nv_fp8_e4m3*>(scale_b) + sfb_off);
      __half_raw sfa_half_raw = __nv_cvt_fp8_to_halfraw(scale_a_val.__x, __NV_E4M3);
      float      sfa_f = __half2float(*reinterpret_cast<__half*>(&sfa_half_raw));
      __half_raw sfb_half_raw = __nv_cvt_fp8_to_halfraw(scale_b_val.__x, __NV_E4M3);
      float      sfb_f = __half2float(*reinterpret_cast<__half*>(&sfb_half_raw));
      sum += (acc * sfa_f * sfb_f);
      acc = 0.f;
    }
  }
  *(static_cast<__half*>(out) + current_batch * m + current_row) = __float2half(sum);
}

__global__ void Nvfp4GemvAsmLoad(
    const void* __restrict__ a,
    const void* __restrict__ b,
    const void* __restrict__ scale_a,
    const void* __restrict__ scale_b,
    void* __restrict__ out,
    const int m,
    const int k,
    const int l) {
  int current_row = blockIdx.x * blockDim.x + threadIdx.x;
  int current_batch = blockIdx.y;
  if (current_row >= m) return;

  auto          a_ptr = static_cast<const __nv_fp4x2_e2m1*>(a);
  auto          b_ptr = static_cast<const __nv_fp4x2_e2m1*>(b);
  auto          sfa_ptr = static_cast<const __nv_fp8_e4m3*>(scale_a);
  auto          sfb_ptr = static_cast<const __nv_fp8_e4m3*>(scale_b);
  uint16_t       a_rmem[1];
  uint16_t       b_rmem[1];
  __nv_fp8_e4m3 sfa_rmem[1];
  __nv_fp8_e4m3 sfb_rmem[1];

  const int k_packed = k / PACK_SIZE;
  const int stride_l = m * k_packed;
  const int sf_k = k / SF_VEC_SIZE;
  const int sfa_stride_l = m * sf_k;
  float     sum = 0.f;
  float     acc = 0.f;
  {
    int a_off = current_batch * stride_l + current_row * k_packed;
    int b_off = current_batch * 128 * k_packed;
    a_ptr += a_off;
    b_ptr += b_off;

    int sfa_off = current_batch * sfa_stride_l + current_row * sf_k;
    int sfb_off = current_batch * 128 * sf_k;
    sfa_ptr += sfa_off;
    sfb_ptr += sfb_off;
  }
#pragma unroll
  for (int i = 0; i < (k_packed / VEC_SIZE); ++i) {
    ldca_i16(a_rmem, a_ptr + i * VEC_SIZE);
    ldcs_i16(b_rmem, b_ptr + i * VEC_SIZE);

    // unpack
    auto a_fp4 = reinterpret_cast<const __nv_fp4x2_e2m1*>(a_rmem);
    auto b_fp4 = reinterpret_cast<const __nv_fp4x2_e2m1*>(b_rmem);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      __half2_raw a_h2_raw = __nv_cvt_fp4x2_to_halfraw2(a_fp4[j].__x, __NV_E2M1);
      __half2     a_h2 = *reinterpret_cast<__half2*>(&a_h2_raw);

      __half2_raw b_h2_raw = __nv_cvt_fp4x2_to_halfraw2(b_fp4[j].__x, __NV_E2M1);
      __half2     b_h2 = *reinterpret_cast<__half2*>(&b_h2_raw);

      // 1. Multiply in half (Result is __half, potential precision loss/overflow)
      __half2 prod_half2 = __hmul2(a_h2, b_h2);
      // 2. Convert to float and Add
      float2 prod_f2 = __half22float2(prod_half2);
      acc += (prod_f2.x + prod_f2.y);
    }

    if ((i + 1) % (SF_PACK_SIZE / VEC_SIZE) == 0) {
      // finish one block
      int group_idx = i / (SF_PACK_SIZE / VEC_SIZE);
      sfa_rmem[0] = *(sfa_ptr + group_idx);
      sfb_rmem[0] = *(sfb_ptr + group_idx);
      __half_raw sfa_half_raw = __nv_cvt_fp8_to_halfraw(sfa_rmem->__x, __NV_E4M3);
      float      sfa_f = __half2float(*reinterpret_cast<__half*>(&sfa_half_raw));
      __half_raw sfb_half_raw = __nv_cvt_fp8_to_halfraw(sfb_rmem->__x, __NV_E4M3);
      float      sfb_f = __half2float(*reinterpret_cast<__half*>(&sfb_half_raw));
      sum += acc * sfa_f * sfb_f;
      acc = 0.f;
    }
  }
  *(static_cast<__half*>(out) + current_batch * m + current_row) = __float2half(sum);
}

__global__ void Nvfp4GemvAsmLoadv2(
    const void* __restrict__ a,
    const void* __restrict__ b,
    const void* __restrict__ scale_a,
    const void* __restrict__ scale_b,
    void* __restrict__ out,
    const int m,
    const int k,
    const int l) {
  int current_row = blockIdx.x * blockDim.x + threadIdx.x;
  int current_batch = blockIdx.y;
  if (current_row >= m) return;

  const int FP4_VEC_SIZE = 32;
  const int DATA_REG_COUNT = 4;
  uint32_t  a_rmem_fp4[DATA_REG_COUNT];
  uint32_t  b_rmem_fp4[DATA_REG_COUNT];
  uint32_t  a_rmem_half[4];
  uint32_t  b_rmem_half[4];

  uint16_t sfa_rmem_fp8;
  uint16_t sfb_rmem_fp8;
  half2    sfa_rmem_half;
  half2    sfb_rmem_half;

  auto a_ptr = static_cast<const __nv_fp4x2_e2m1*>(a);
  auto b_ptr = static_cast<const __nv_fp4x2_e2m1*>(b);
  auto sfa_ptr = static_cast<const __nv_fp8_e4m3*>(scale_a);
  auto sfb_ptr = static_cast<const __nv_fp8_e4m3*>(scale_b);

  const int k_packed = k / PACK_SIZE;
  const int stride_i = FP4_VEC_SIZE / PACK_SIZE;
  const int stride_l = m * k_packed;
  const int sf_k = k / SF_VEC_SIZE;
  const int sfa_stride_l = m * sf_k;

  {
    int a_off = current_batch * stride_l + current_row * k_packed;
    int b_off = current_batch * 128 * k_packed;
    a_ptr += a_off;
    b_ptr += b_off;

    int sfa_off = current_batch * sfa_stride_l + current_row * sf_k;
    int sfb_off = current_batch * 128 * sf_k;
    sfa_ptr += sfa_off;
    sfb_ptr += sfb_off;
  }

  float sum = 0.f;
  half2 acc[2] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
#pragma unroll
  for (int i = 0; i < (k / FP4_VEC_SIZE); ++i) {
    ldcs_i32x4(a_rmem_fp4, a_ptr + i * stride_i);
    ldca_i32x4(b_rmem_fp4, b_ptr + i * stride_i);
    ldcs_i16(&sfa_rmem_fp8, sfa_ptr + i * 2);
    ldca_i16(&sfb_rmem_fp8, sfb_ptr + i * 2);
    fp8x2_to_fp16x2(&sfa_rmem_half, sfa_rmem_fp8);
    fp8x2_to_fp16x2(&sfb_rmem_half, sfb_rmem_fp8);

#pragma unroll
    for (int m = 0; m < DATA_REG_COUNT; ++m) {
      // unpack fp4x8 -> fp16x2x4
      fp4x8_to_fp16x2x4(a_rmem_half, a_rmem_fp4[m]);
      fp4x8_to_fp16x2x4(b_rmem_half, b_rmem_fp4[m]);

      // compute
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2_raw a_h2_raw = *reinterpret_cast<__half2_raw*>(&a_rmem_half[j]);
        half2       a_h2 = *reinterpret_cast<half2*>(&a_h2_raw);

        __half2_raw b_h2_raw = *reinterpret_cast<__half2_raw*>(&b_rmem_half[j]);
        half2       b_h2 = *reinterpret_cast<half2*>(&b_h2_raw);

        __half2 prod_half2 = __hmul2(a_h2, b_h2);
        acc[m / 2] = __hadd2(acc[m / 2], prod_half2);
      }
    }

    half2 acc_h2 = __halves2half2(__hadd(acc[0].x, acc[0].y), __hadd(acc[1].x, acc[1].y));
    half2 tmp = __hmul2(acc_h2, sfa_rmem_half);
    tmp = __hmul2(tmp, sfb_rmem_half);

    sum += __half2float(tmp.x) + __half2float(tmp.y);

    acc[0] = __float2half2_rn(0.0f);
    acc[1] = __float2half2_rn(0.0f);
  }
  *(static_cast<__half*>(out) + current_batch * m + current_row) = __float2half(sum);
}

torch::Tensor nvfp4_gemv_launcher(
    const torch::Tensor& a, // mat
    const torch::Tensor& b, // vec
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    torch::Tensor        out) {

  const int m = a.size(0), k = a.size(1) * PACK_SIZE, l = a.size(2);

  dim3 block(256);
  dim3 grid((m + block.x - 1) / block.x, l);
  // Nvfp4gemvNaive<<<grid, block>>>(
  //     a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(), out.data_ptr(), m, k, l);
  Nvfp4GemvAsmLoadv2<<<grid, block>>>(
      a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(), out.data_ptr(), m, k, l);

  return out;
}
