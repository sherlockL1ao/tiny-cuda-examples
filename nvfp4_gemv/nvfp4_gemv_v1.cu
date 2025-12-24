#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int WARPSIZE = 32;

#ifndef NVFP4_GEMV_COMMON_HELPERS_DEFINED
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

#endif

template <int BLOCK_M, int BLOCK_K, int THREADS_K, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS* WARPSIZE) Nvfp4GemvRegTile(
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
  int tid_m = tid / THREADS_K; // 0..3 for NUM_WARPS=4, THREADS_K=32

  constexpr int BYTES_PER_LOAD = 16; // 4 x int32 = 128 bit = 16 bytes
  constexpr int TB_SIZE = NUM_WARPS * WARPSIZE;
  constexpr int THREADS_M = TB_SIZE / THREADS_K;
  constexpr int ROWS_PER_THREAD = BLOCK_M / THREADS_M; // each thread process strided rows

  constexpr int K_CHUNKS_PER_THREAD = BLOCK_K / BYTES_PER_LOAD / THREADS_K;

  int current_row = blockIdx.x * BLOCK_M;
  int current_batch = blockIdx.y;

  auto a_ptr = static_cast<const __nv_fp4x2_e2m1*>(a);
  auto b_ptr = static_cast<const __nv_fp4x2_e2m1*>(b);
  auto sfa_ptr = static_cast<const __nv_fp8_e4m3*>(scale_a);
  auto sfb_ptr = static_cast<const __nv_fp8_e4m3*>(scale_b);

  {
    int a_off = current_batch * M * K + current_row * K;
    int b_off = current_batch * 128 * K;
    a_ptr += a_off;
    b_ptr += b_off;

    int sfa_off = current_batch * M * SF_K + current_row * SF_K;
    int sfb_off = current_batch * 128 * SF_K;
    sfa_ptr += sfa_off;
    sfb_ptr += sfb_off;
  }

  // data registers
  uint32_t frag_A[ROWS_PER_THREAD][K_CHUNKS_PER_THREAD][4];
  uint32_t frag_A_h2[ROWS_PER_THREAD][K_CHUNKS_PER_THREAD][4][/*8xfp4->4xh2 */ 4];
  uint32_t frag_B[K_CHUNKS_PER_THREAD][4];
  uint32_t frag_B_h2[K_CHUNKS_PER_THREAD][4][4];
  uint16_t frag_sfa[ROWS_PER_THREAD][K_CHUNKS_PER_THREAD][1];
  uint16_t frag_sfb[K_CHUNKS_PER_THREAD][1];
  __half2  frag_sfb_h2[K_CHUNKS_PER_THREAD][1];
  __half2  frag_sfa_h2[ROWS_PER_THREAD][K_CHUNKS_PER_THREAD][1];

  float master_acc[ROWS_PER_THREAD] = {};

  auto gmem_to_rmem = [&]() {
    for (int k = 0; k < K_CHUNKS_PER_THREAD; ++k) {
      const int idx_k = k * THREADS_K + tid_k;
      const int global_k_byte = idx_k * BYTES_PER_LOAD;
      // Skip if this chunk is out of bounds (for K < BLOCK_K cases)
      if (global_k_byte >= K) {
        // Zero-fill the registers to avoid garbage accumulation
        frag_B[k][0] = frag_B[k][1] = frag_B[k][2] = frag_B[k][3] = 0;
        frag_sfb[k][0] = 0;
        for (int m = 0; m < ROWS_PER_THREAD; ++m) {
          frag_A[m][k][0] = frag_A[m][k][1] = frag_A[m][k][2] = frag_A[m][k][3] = 0;
          frag_sfa[m][k][0] = 0;
        }
        continue;
      }
      ldca_i32x4(frag_B[k], b_ptr + idx_k * BYTES_PER_LOAD);
      ldca_i16(frag_sfb[k], sfb_ptr + idx_k * /*2 fp8 scale*/ 2); // 32 fp4 with block_size=16

      for (int m = 0; m < ROWS_PER_THREAD; ++m) {
        const int idx_m = m * THREADS_M + tid_m;
        ldcs_i32x4(frag_A[m][k], a_ptr + idx_m * K + idx_k * BYTES_PER_LOAD);
        ldcs_i16(frag_sfa[m][k], sfa_ptr + idx_m * SF_K + idx_k * 2);
      }
    }
  };

  auto unpack_and_convert = [&]() {
    for (int k = 0; k < K_CHUNKS_PER_THREAD; ++k) {
      fp8x2_to_fp16x2(frag_sfb_h2[k], frag_sfb[k][0]);
      for (int n = 0; n < 4; ++n) {
        fp4x8_to_fp16x2x4(frag_B_h2[k][n], frag_B[k][n]);
      }

      for (int m = 0; m < ROWS_PER_THREAD; ++m) {
        fp8x2_to_fp16x2(frag_sfa_h2[m][k], frag_sfa[m][k][0]);
        for (int n = 0; n < 4; ++n) {
          fp4x8_to_fp16x2x4(frag_A_h2[m][k][n], frag_A[m][k][n]);
        }
      }
    }
  };

  auto compute = [&]() {
    __half2_raw sf_prod[ROWS_PER_THREAD][K_CHUNKS_PER_THREAD];
    // pre-compute scale factors multiplication
    for (int k = 0; k < K_CHUNKS_PER_THREAD; ++k) {
      for (int m = 0; m < ROWS_PER_THREAD; ++m) {
        sf_prod[m][k] = __hmul2(frag_sfa_h2[m][k][0], frag_sfb_h2[k][0]);
      }
    }

    __half2 acc[ROWS_PER_THREAD][2] = {};
    for (int k = 0; k < K_CHUNKS_PER_THREAD; ++k) {
      for (int m = 0; m < ROWS_PER_THREAD; ++m) {
        for (int e = 0; e < 4; ++e) {
          for (int i = 0; i < 2; ++i) {
            __half2 a_h2 = *reinterpret_cast<const __half2*>(&frag_A_h2[m][k][i][e]);
            __half2 b_h2 = *reinterpret_cast<const __half2*>(&frag_B_h2[k][i][e]);
            acc[m][0] = __hfma2(a_h2, b_h2, acc[m][0]);
          }

          for (int i = 2; i < 4; ++i) {
            __half2 a_h2 = *reinterpret_cast<const __half2*>(&frag_A_h2[m][k][i][e]);
            __half2 b_h2 = *reinterpret_cast<const __half2*>(&frag_B_h2[k][i][e]);
            acc[m][1] = __hfma2(a_h2, b_h2, acc[m][1]);
          }
        }

        __half_raw group0 = __hadd(acc[m][0].x, acc[m][0].y);
        __half_raw group1 = __hadd(acc[m][1].x, acc[m][1].y);

        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group0.x), "h"(sf_prod[m][k].x));
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group1.x), "h"(sf_prod[m][k].y));

        for (int i = 0; i < 2; ++i) {
          acc[m][i] = __float2half2_rn(0.0f);
        }
      }
    }
  };

  int num_iters = (K + BLOCK_K - 1) / BLOCK_K;
#pragma unroll
  for (int i = 0; i < num_iters; ++i) {
    gmem_to_rmem();
    unpack_and_convert();
    compute();

    a_ptr += BLOCK_K;
    b_ptr += BLOCK_K;
    sfa_ptr += BLOCK_K / 8;
    sfb_ptr += BLOCK_K / 8;
  }

  if constexpr (THREADS_K > WARPSIZE) { // cross-warp reduction
    // constexpr int    NUM_WARPS_K = THREADS_K / WARPSIZE;
    // __shared__ float smem[ROWS_PER_THREAD][THREADS_M][NUM_WARPS_K];
    // for (int i = 0; i < ROWS_PER_THREAD; ++i) {
    //   master_acc[i] = warp_reduce_sum<WARPSIZE>(master_acc[i]);
    // }
    // int lane_id = tid_k % WARPSIZE;
    // int warp_id = tid_k / WARPSIZE;
    // for (int i = 0; i < ROWS_PER_THREAD; ++i) {
    //   if (lane_id == 0) {
    //     smem[i][tid_m][warp_id] = master_acc[i];
    //   }
    // }
    // __syncthreads();

    // for (int i = 0; i < ROWS_PER_THREAD; ++i) {
    //   master_acc[i] = 0.0f;
    // }
    // for (int i = 0; i < ROWS_PER_THREAD; ++i) {
    //   for (int j = 0; j < NUM_WARPS_K; ++j) {
    //     master_acc[i] += smem[i][tid_m][j];
    //   }
    // }

    __shared__ float smem[ROWS_PER_THREAD][THREADS_M][THREADS_K];
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
      smem[i][tid_m][tid_k] = master_acc[i];
    }
    __syncthreads();

    for (int stride = THREADS_K / 2; stride >= WARPSIZE; stride /= 2) {
      if (tid_k < stride) {
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
          smem[i][tid_m][tid_k] += smem[i][tid_m][tid_k + stride];
        }
      }
      __syncthreads();
    }
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
      if (tid_k < WARPSIZE) {
        master_acc[i] = smem[i][tid_m][tid_k];
      }
    }

    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
      master_acc[i] = warp_reduce_sum<WARPSIZE>(master_acc[i]);
    }
  } else {

    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
      master_acc[i] = warp_reduce_sum<THREADS_K>(master_acc[i]);
    }
  }
  if (tid_k == 0) {
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
      int row = current_row + i * THREADS_M + tid_m;
      if (row < M) {
        __half* out_ptr = static_cast<__half*>(out) + current_batch * M + row;
        out_ptr[0] = __float2half(master_acc[i]);
      }
    }
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
  constexpr int BLOCK_K = 1024; // one block, number of bytes to process along k
  constexpr int THREADS_K = BLOCK_K / 16;

  dim3 block(TB_SIZE);
  dim3 grid((m + BLOCK_M - 1) / BLOCK_M, l);
  Nvfp4GemvRegTile<BLOCK_M, BLOCK_K, THREADS_K, NUM_WARPS><<<grid, block>>>(
      a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(), out.data_ptr(), m, k, l, sf_k);

  return out;
}
