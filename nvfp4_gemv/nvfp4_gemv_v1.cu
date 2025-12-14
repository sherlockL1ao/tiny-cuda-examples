#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int WARPSIZE = 32;

template <unsigned int warp_size>
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  if (warp_size >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
  if (warp_size >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
  if (warp_size >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
  if (warp_size >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
  if (warp_size >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}


__device__ __forceinline__ void fp4x8_to_fp16x2x4(uint32_t* out, uint32_t in) {
  asm volatile("{\n\t"
               ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n\t"
               "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4; // unpack 32-bit register to 4x fp4x2\n\t"
               "cvt.rn.f16x2.e2m1x2 %0, tmp0;\n\t"
               "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n\t"
               "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n\t"
               "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n\t"
               "}"
               : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
               : "r"(in));
}

__device__ __forceinline__ void fp8x2_to_fp16x2(half2* out, uint16_t in) {
  uint32_t* out_i32 = reinterpret_cast<uint32_t*>(out);
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(out_i32[0]) : "h"(in));
}

// Load Cache streaming
__device__ __forceinline__ void ldcs_i16(uint16_t* dst, const void* src) {
  asm volatile("ld.global.L1::no_allocate.b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
}

// Load Cache All
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

template <int BLOCK_M, int BLOCK_K, int THREADS_K, int NUM_WARPS>
__global__ void Nvfp4GemvRegTile(
    const void* __restrict__ a,
    const void* __restrict__ b,
    const void* __restrict__ scale_a,
    const void* __restrict__ scale_b,
    void* __restrict__ out,
    const int M,
    const int K,
    const int L,
    const int SF_K) {
  int tid = threadIdx.x;
  int tid_k = tid % THREADS_K; // 0..31
  int tid_m = tid / THREADS_K;

  constexpr int FP4X2_PER_LOAD = 16; // 4 x int32 = 128 bit = 16 bytes
  constexpr int TB_SIZE = NUM_WARPS * WARPSIZE;
  constexpr int THREADS_M = TB_SIZE / THREADS_K;
  constexpr int ROWS_PER_THREAD = BLOCK_M / THREADS_M; // each thread process strided rows

  constexpr int K_CHUNKS_PER_THREAD = BLOCK_K / FP4X2_PER_LOAD / THREADS_K;

  int current_row = blockIdx.x * BLOCK_M + tid_m * ROWS_PER_THREAD;
  int current_batch = blockIdx.y;
  if (current_row >= M) return;

  auto a_ptr = static_cast<const __nv_fp4x2_e2m1*>(a);
  auto b_ptr = static_cast<const __nv_fp4x2_e2m1*>(b);
  auto sfa_ptr = static_cast<const __nv_fp8_e4m3*>(scale_a);
  auto sfb_ptr = static_cast<const __nv_fp8_e4m3*>(scale_b);

  {
    int a_off = current_batch * M * K + current_row * K;
    int b_off = current_batch * 128 * K;
    a_ptr += a_off;
    b_ptr += b_off;

    int sfa_off = current_batch * M * SF_K+ current_row * SF_K;
    int sfb_off = current_batch * 128 * SF_K;
    sfa_ptr += sfa_off;
    sfb_ptr += sfb_off;
  }

  // data registers
  uint32_t frag_A[ROWS_PER_THREAD][K_CHUNKS_PER_THREAD][4];

  int num_iters = (K + BLOCK_K - 1) / BLOCK_K;
  for (int i = 0; i < num_iters; ++i) {
#pragma unroll
    for (int m = 0; m < ROWS_PER_THREAD; ++m) {
      for (int k = 0; k < K_CHUNKS_PER_THREAD; ++k) {
        const int idx_m = m * THREADS_M + tid_m;
        const int idx_k = k * THREADS_K + tid_k;
        ldcs_i32x4(frag_A[m][k], a_ptr + idx_m * K + idx_k * FP4X2_PER_LOAD);
      }
    }

    a_ptr += i * BLOCK_K;
    b_ptr += i * BLOCK_K;
  }
}

torch::Tensor nvfp4_gemv_reg_tile_launcher(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    torch::Tensor        out) {

  // to make our calculations simple, let's treat fp4x2 as a unit.
  // hence, K = number of fp4x2 elements
  const int m = a.size(0), k = a.size(1), l = a.size(2), sf_k = scale_a.size(1);

  constexpr int NUM_WARPS = 4;
  constexpr int TB_SIZE = NUM_WARPS * WARPSIZE; // 4 warps per block, 128 threads

  constexpr int BLOCK_M = 8;
  constexpr int BLOCK_K = 512; // one block, number of bytes to process along k
  constexpr int THREADS_K = 32;

  dim3 block(TB_SIZE);
  dim3 grid((m + BLOCK_M - 1) / BLOCK_M, l);
  Nvfp4GemvRegTile<BLOCK_M, BLOCK_K, THREADS_K, NUM_WARPS><<<grid, block>>>(
      a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(), out.data_ptr(), m, k, l, sf_k);

  return out;
}
