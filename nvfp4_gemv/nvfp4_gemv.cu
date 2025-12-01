#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define SF_VEC_SIZE 16
#define PACK_SIZE 2
#define SF_PACK_SIZE (SF_VEC_SIZE / PACK_SIZE)

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
  if (current_row >= m || current_batch >= l) return;

  const int k_packed = k / PACK_SIZE;
  const int stride_l = m * k_packed;
  const int sf_k = k / SF_VEC_SIZE;
  const int sfa_stride_l = m * sf_k;
  float     sum = 0.f;
  float     acc = 0.f;
  for (int i = 0; i < k_packed; ++i) {
    int  a_off = current_batch * stride_l + current_row * k_packed + i;
    int  b_off = current_batch * k_packed + i;
    auto a_packed = *(static_cast<const __nv_fp4x2_e2m1*>(a) + a_off);
    // convert packed fp4x2 → half2
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
    if (i % SF_PACK_SIZE == 0) {
      // finish one group
      int        group_idx = i / SF_PACK_SIZE;
      int        sfa_off = current_batch * sfa_stride_l + current_row * sf_k + group_idx;
      int        sfb_off = current_batch * sf_k + group_idx;
      auto       scale_a_val = *(static_cast<const __nv_fp8_e4m3*>(scale_a) + sfa_off);
      auto       scale_b_val = *(static_cast<const __nv_fp8_e4m3*>(scale_b) + sfb_off);
      __half_raw sfa_half_raw = __nv_cvt_fp8_to_halfraw(scale_a_val.__x, __NV_E4M3);
      float      sfa_f = __half2float(*reinterpret_cast<__half*>(&sfa_half_raw));
      __half_raw sfb_half_raw = __nv_cvt_fp8_to_halfraw(scale_b_val.__x, __NV_E4M3);
      float      sfb_f = __half2float(*reinterpret_cast<__half*>(&sfb_half_raw));
      sum += acc * sfa_f * sfb_f;
      acc = 0.f;
    }
  }
  *(static_cast<__half*>(out) + current_batch * m + current_row) = __float2half(sum);
}

torch::Tensor nvfp4_gemv_launcher(
    const torch::Tensor& a, // mat
    const torch::Tensor& b, // vec
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    torch::Tensor        out) {

  const int m = a.size(0), k = a.size(1), l = a.size(2);

  dim3 block(256);
  dim3 grid((m + block.x - 1) / block.x, l);
  Nvfp4gemvNaive<<<grid, block>>>(
      a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(), out.data_ptr(), m, k, l);

  return out;
}
