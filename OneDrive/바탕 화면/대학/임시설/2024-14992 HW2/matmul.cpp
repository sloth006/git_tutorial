#include <stdlib.h>
#include <iostream>
#include <arm_neon.h>
#include <pthread.h>
#include <thread>
#include <algorithm>
#include "../support/matmul.h"

//inspired by https://github.com/siboehm/SGEMM_CUDA
//goal optimization on 4096*4096*4096 mma
//only made for matrix sizes that are multiples of 32, otherwise it will have out of bounds access

//jetson orio nano
//6 cores, 4xA57 + 2xA72 1.43GHz
//L1: 64KB per core, L2: 256kb, L3 4MB

//need to calculate again what was i doing...
//consideration for L1 cache
// A 64*64 B (64*64)T C 64*64 48kb

//consideration for L2 cache
// A 128*128 B (128*128)T C 128*128 196kb

//consideration for L3 cache
// 6 cores share L3 cache A 128*128 change B (128*128)T * 6 given to each core C 128*128 *6 change << 3MB
//broadcast system not bad?

// 4096/768 = 5.33333
// for the last 0.33333 which will be B 4096*256 we can padde it to be 4096*288 
// which will make the calculation to A 128*128 B (48*128)T C 128*48 for L2 cache
// and A 64*64 B (48*64)T C 64*48 48kb for L1 cache

//<arm_neon.h> support 128 bit wide SIMD registers
//int32x4_t
//vld1q_f32()
//vst1q_f32()
//__builtin_prefetch()
//vmlaq_n_s32()

//fastest
#define L2_M   128
#define L2_N   128
#define L2_K   128
#define L1_M   64
#define L1_N   64
#define L1_K   64
// #define L2_M   96
// #define L2_N   96
// #define L2_K   96
// #define L1_M   48
// #define L1_N   48
// #define L1_K   48
#define SIMD_WIDTH  4

static inline void pack_B(const int *inputB, int *packedB, int N, int j_start, int j_block, int bk, int block_k) {
    int idx = 0;
    for (int j = 0; j + SIMD_WIDTH <= j_block; j += SIMD_WIDTH) {
        for (int k = 0; k < block_k; k++) {
            packedB[idx++] = inputB[(bk + k) * N + (j_start + j + 0)];
            packedB[idx++] = inputB[(bk + k) * N + (j_start + j + 1)];
            packedB[idx++] = inputB[(bk + k) * N + (j_start + j + 2)];
            packedB[idx++] = inputB[(bk + k) * N + (j_start + j + 3)];
        }
    }
}

static inline void micro_kernel_4x4(const int *inputA, const int *packedB_panel, int *output, int i, int j, int bk, int block_k, int K, int N) {
    int32x4_t c0, c1, c2, c3;
    if (bk == 0) {
        c0 = vdupq_n_s32(0);
        c1 = vdupq_n_s32(0);
        c2 = vdupq_n_s32(0);
        c3 = vdupq_n_s32(0);
    } else {
        c0 = vld1q_s32(&output[(i + 0) * N + j]);
        c1 = vld1q_s32(&output[(i + 1) * N + j]);
        c2 = vld1q_s32(&output[(i + 2) * N + j]);
        c3 = vld1q_s32(&output[(i + 3) * N + j]);
    }

    const int* b_ptr = packedB_panel;

    for (int k = 0; k < block_k; k++) {
        int32x4_t b_vec = vld1q_s32(b_ptr);
        b_ptr += SIMD_WIDTH;

        int32_t a0 = inputA[(i + 0) * K + (bk + k)];
        int32_t a1 = inputA[(i + 1) * K + (bk + k)];
        int32_t a2 = inputA[(i + 2) * K + (bk + k)];
        int32_t a3 = inputA[(i + 3) * K + (bk + k)];

        c0 = vmlaq_n_s32(c0, b_vec, a0);
        c1 = vmlaq_n_s32(c1, b_vec, a1);
        c2 = vmlaq_n_s32(c2, b_vec, a2);
        c3 = vmlaq_n_s32(c3, b_vec, a3);

        if (bk + k + 8 < K)
            __builtin_prefetch(&inputA[(i + 0) * K + (bk + k + 8)], 0, 1);
    }

    vst1q_s32(&output[(i + 0) * N + j], c0);
    vst1q_s32(&output[(i + 1) * N + j], c1);
    vst1q_s32(&output[(i + 2) * N + j], c2);
    vst1q_s32(&output[(i + 3) * N + j], c3);
}

static inline void do_l1_block(const int *inputA, const int *inputB, int *output,
                               int i0, int i1, int j0, int j1, int k0, int k1, int N, int K) {
    for (int i = i0; i < i1; i++) {
        for (int j = j0; j < j1; j += SIMD_WIDTH) {
            if (j + SIMD_WIDTH <= j1) {
                int32x4_t acc = (k0 == 0) ? vdupq_n_s32(0) : vld1q_s32(output + i * N + j);
                for (int k = k0; k < k1; k++) {
                    int32_t a_val = inputA[i * K + k];
                    int32x4_t b_vec = vld1q_s32(inputB + k * N + j);
                    acc = vmlaq_n_s32(acc, b_vec, a_val);
                }
                vst1q_s32(output + i * N + j, acc);
            } else {
                for (int jj = j; jj < j1; jj++) {
                    int sum = (k0 == 0) ? 0 : output[i * N + jj];
                    for (int k = k0; k < k1; k++)
                        sum += inputA[i * K + k] * inputB[k * N + jj];
                    output[i * N + jj] = sum;
                }
                break;
            }
        }
    }
}

static void core_matmul_block(const int *inputA, const int *inputB, int *output, int M, int N, int K,
                              int bi, int i_block, int n_start, int n_len) {
    int packedB[L2_N * L2_K];
    const int prefetch_step = 16;

    for (int bj = 0; bj < n_len; bj += L2_N) {
        int j_block = std::min(L2_N, n_len - bj);
        int j_start = n_start + bj;

        for (int bk = 0; bk < K; bk += L2_K) {
            int block_k = std::min(L2_K, K - bk);

            if (bk + L2_K < K) {
                int next_bk = bk + L2_K;
                int next_k_len = std::min(L2_K, K - next_bk);
                for (int r = 0; r < i_block; r++)
                    for (int off = 0; off < next_k_len; off += prefetch_step)
                        __builtin_prefetch(inputA + (bi + r) * K + next_bk + off, 0, 3);
                for (int r = 0; r < next_k_len; r++)
                    for (int off = 0; off < j_block; off += prefetch_step)
                        __builtin_prefetch(inputB + (next_bk + r) * N + j_start + off, 0, 3);
            }

            pack_B(inputB, packedB, N, j_start, j_block, bk, block_k);

            for (int kk = 0; kk < block_k; kk += L1_K) {
                int k_len = std::min(L1_K, block_k - kk);
                for (int ii = 0; ii < i_block; ii += L1_M) {
                    int i_tile = std::min(L1_M, i_block - ii);
                    for (int jj = 0; jj < j_block; jj += L1_N) {
                        int j_tile = std::min(L1_N, j_block - jj);
                        for (int ii_inner = 0; ii_inner + 4 <= i_tile; ii_inner += 4) {
                            for (int jj_inner = 0; jj_inner + 4 <= j_tile; jj_inner += 4) {
                                const int* packedB_panel = &packedB[((jj + jj_inner) / 4) * block_k * 4 + kk * 4];
                                micro_kernel_4x4(inputA, packedB_panel, output,
                                                bi + ii + ii_inner, j_start + jj + jj_inner,
                                                bk + kk, k_len, K, N);
                            }
                        }
                    }
                }
            }

            int i_done = (i_block / 4) * 4;
            int j_done = (j_block / 4) * 4;
            if (bi + i_done < bi + i_block)
                do_l1_block(inputA, inputB, output, bi + i_done, bi + i_block, j_start, j_start + j_block, bk, bk + block_k, N, K);
            if (j_start + j_done < j_start + j_block)
                do_l1_block(inputA, inputB, output, bi, bi + i_done, j_start + j_done, j_start + j_block, bk, bk + block_k, N, K);
        }
    }
}

static void core_matmul(const int *inputA, const int *inputB, int *output,
                        const int M, const int N, const int K, int n_start, int n_len) {
    for (int bi = 0; bi < M; bi += L2_M) {
        int i_block = std::min(L2_M, M - bi);
        core_matmul_block(inputA, inputB, output, M, N, K, bi, i_block, n_start, n_len);
    }
}

#define num_cores     6
#define Align         16
#define Thread_strip  (num_cores * L2_M)

void matmul(const int *inputA, const int *inputB, int *output, const int M, const int N, const int K) {
    std::thread threads[num_cores];
    const int n_full_strips = N / Thread_strip;
    for (int phase = 0; phase < n_full_strips; phase++) {
        const int bj_super = phase * Thread_strip;
        for (int t = 0; t < num_cores; t++) {
            int start = bj_super + t * L2_N;
            int len = L2_N;
            threads[t] = std::thread(core_matmul, inputA, inputB, output, M, N, K, start, len);
        }
        for (int t = 0; t < num_cores; t++)
            threads[t].join();
    }

    const int remainder = N - n_full_strips * Thread_strip;
    if (remainder > 0) {
        const int units = remainder / Align;
        const int r = remainder % Align;
        const int base = units / num_cores;
        const int rem_units = units % num_cores;
        int len[6];
        for (int t = 0; t < num_cores; t++) {
            len[t] = (base + (t < rem_units ? 1 : 0)) * Align;
        }
        len[num_cores - 1] += r;

        int n_start = n_full_strips * Thread_strip;
        for (int t = 0; t < num_cores; t++) {
            if (len[t] > 0) {
                int start = n_start;
                threads[t] = std::thread(core_matmul, inputA, inputB, output, M, N, K, start, len[t]);
                n_start += len[t];
            }
        }
        for (int t = 0; t < num_cores; t++)
            if (threads[t].joinable())
                threads[t].join();
    }
}

// My tries which didn't make things faster

// #include <stdlib.h>
// #include <iostream>
// #include <arm_neon.h>
// #include <pthread.h>
// #include <omp.h>
// #include "../support/matmul.h"

// /* Toggle for benchmarking: set to 1 to use software prefetch, 0 to rely on hardware prefetcher. */
// #ifndef ENABLE_SOFTWARE_PREFETCH
// #define ENABLE_SOFTWARE_PREFETCH 0
// #endif
// #if ENABLE_SOFTWARE_PREFETCH
// #define PREFETCH_A(ptr)  __builtin_prefetch((ptr), 0, 3)
// #define PREFETCH_B(ptr)  __builtin_prefetch((ptr), 0, 3)
// #else
// #define PREFETCH_A(ptr)  ((void)0)
// #define PREFETCH_B(ptr)  ((void)0)
// #endif

// //inspired by https://github.com/siboehm/SGEMM_CUDA
// //and my hw #1

// //jetson orio nano
// //6 cores, 4xA57 + 2xA72 1.43GHz
// //L1: 64KB per core, L2: 256kb, L3 4MB

// //consideration for L1 cache
// // A 64*64 B (64*64)T C 64*64 48kb

// //consideration for L2 cache
// // A 128*128 B (128*128)T C 128*128 196kb

// //consideration for L3 cache
// // 6 cores share L3 cache A 128*128 change B (128*128)T * 6 given to each core C 128*128 *6 change << 3MB
// //broadcast system not bad?
// //changing the block size to be smaller for small matrix (not experiemental)

// // Deport block size (tried to make changes for smaller on small matrix see matmul function)
// #define L2_M_MAX  128
// #define L2_N_MAX  128
// #define L2_K_MAX  128
// #define L1_M_MAX  64
// #define L1_N_MAX  64
// #define L1_K_MAX  64

// #define SIMD_WIDTH  4
// #define MR          8   /* micro-kernel output rows (8x4 tile) */
// #define NR          4   /* micro-kernel output cols (SIMD width) */

// static inline void pack_A(const int *inputA, int *packedA, int K, int i_start, int i_rows, int bk, int block_k) {
//     int idx = 0;
//     for (int r = 0; r < i_rows; r++) {
//         for (int k = 0; k < block_k; k++) {
//             packedA[idx++] = inputA[(i_start + r) * K + (bk + k)];
//         }
//     }
// }

// static inline void pack_A_strided(const int *inputA, int *packedA, int K, int i_start, int i_rows, int bk, int block_k, int row_stride) {
//     for (int r = 0; r < i_rows; r++) {
//         for (int k = 0; k < block_k; k++) {
//             packedA[r * row_stride + k] = inputA[(i_start + r) * K + (bk + k)];
//         }
//     }
// }

// static inline void pack_B(const int *inputB, int *packedB, int N, int j_start, int j_block, int bk, int block_k) {
//     int idx = 0;
//     for (int j = 0; j + SIMD_WIDTH <= j_block; j += SIMD_WIDTH) {
//         for (int k = 0; k < block_k; k++) {
//             packedB[idx++] = inputB[(bk + k) * N + (j_start + j + 0)];
//             packedB[idx++] = inputB[(bk + k) * N + (j_start + j + 1)];
//             packedB[idx++] = inputB[(bk + k) * N + (j_start + j + 2)];
//             packedB[idx++] = inputB[(bk + k) * N + (j_start + j + 3)];
//         }
//     }
// }

// static inline void micro_kernel_4x4(const int *packedA_panel, int row_stride, const int *packedB_panel, int *output,
//                                     int i, int j, int block_k, int N, int initial_zero) {
//     packedA_panel = (const int *)__builtin_assume_aligned(packedA_panel, 64);
//     packedB_panel = (const int *)__builtin_assume_aligned(packedB_panel, 64);
//     output        = (int *)__builtin_assume_aligned(output, 64);

//     int32x4_t c0, c1, c2, c3;
//     if (initial_zero) {
//         c0 = vdupq_n_s32(0);
//         c1 = vdupq_n_s32(0);
//         c2 = vdupq_n_s32(0);
//         c3 = vdupq_n_s32(0);
//     } else {
//         c0 = vld1q_s32(&output[(i + 0) * N + j]);
//         c1 = vld1q_s32(&output[(i + 1) * N + j]);
//         c2 = vld1q_s32(&output[(i + 2) * N + j]);
//         c3 = vld1q_s32(&output[(i + 3) * N + j]);
//     }

//     const int *a0_ptr = packedA_panel;
//     const int *a1_ptr = packedA_panel + row_stride;
//     const int *a2_ptr = packedA_panel + 2 * row_stride;
//     const int *a3_ptr = packedA_panel + 3 * row_stride;
//     const int *b_ptr = packedB_panel;

//     int k = 0;
//     for (; k + 4 <= block_k; k += 4) {
//         int32x4_t b0 = vld1q_s32(b_ptr); b_ptr += SIMD_WIDTH;
//         int32x4_t b1 = vld1q_s32(b_ptr); b_ptr += SIMD_WIDTH;
//         int32x4_t b2 = vld1q_s32(b_ptr); b_ptr += SIMD_WIDTH;
//         int32x4_t b3 = vld1q_s32(b_ptr); b_ptr += SIMD_WIDTH;

//         int32x4_t a0_vec = vld1q_s32(a0_ptr); a0_ptr += SIMD_WIDTH;
//         int32x4_t a1_vec = vld1q_s32(a1_ptr); a1_ptr += SIMD_WIDTH;
//         int32x4_t a2_vec = vld1q_s32(a2_ptr); a2_ptr += SIMD_WIDTH;
//         int32x4_t a3_vec = vld1q_s32(a3_ptr); a3_ptr += SIMD_WIDTH;

//         c0 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c0, b0, a0_vec, 0), b1, a0_vec, 1), b2, a0_vec, 2), b3, a0_vec, 3);
//         c1 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c1, b0, a1_vec, 0), b1, a1_vec, 1), b2, a1_vec, 2), b3, a1_vec, 3);
//         c2 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c2, b0, a2_vec, 0), b1, a2_vec, 1), b2, a2_vec, 2), b3, a2_vec, 3);
//         c3 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c3, b0, a3_vec, 0), b1, a3_vec, 1), b2, a3_vec, 2), b3, a3_vec, 3);
//     }
//     for (; k < block_k; k++) {
//         int32x4_t b_vec = vld1q_s32(b_ptr);
//         b_ptr += SIMD_WIDTH;
//         c0 = vmlaq_n_s32(c0, b_vec, *a0_ptr++);
//         c1 = vmlaq_n_s32(c1, b_vec, *a1_ptr++);
//         c2 = vmlaq_n_s32(c2, b_vec, *a2_ptr++);
//         c3 = vmlaq_n_s32(c3, b_vec, *a3_ptr++);
//     }

//     vst1q_s32(&output[(i + 0) * N + j], c0);
//     vst1q_s32(&output[(i + 1) * N + j], c1);
//     vst1q_s32(&output[(i + 2) * N + j], c2);
//     vst1q_s32(&output[(i + 3) * N + j], c3);
// }

// static inline void micro_kernel_8x4(const int *packedA_panel, int row_stride, const int *packedB_panel, int *output,
//                                     int i, int j, int block_k, int N, int initial_zero) {
//     packedA_panel = (const int *)__builtin_assume_aligned(packedA_panel, 64);
//     packedB_panel = (const int *)__builtin_assume_aligned(packedB_panel, 64);
//     output        = (int *)__builtin_assume_aligned(output, 64);

//     int32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
//     if (initial_zero) {
//         c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = vdupq_n_s32(0);
//     } else {
//         c0 = vld1q_s32(&output[(i + 0) * N + j]);
//         c1 = vld1q_s32(&output[(i + 1) * N + j]);
//         c2 = vld1q_s32(&output[(i + 2) * N + j]);
//         c3 = vld1q_s32(&output[(i + 3) * N + j]);
//         c4 = vld1q_s32(&output[(i + 4) * N + j]);
//         c5 = vld1q_s32(&output[(i + 5) * N + j]);
//         c6 = vld1q_s32(&output[(i + 6) * N + j]);
//         c7 = vld1q_s32(&output[(i + 7) * N + j]);
//     }

//     const int *a0_ptr = packedA_panel;
//     const int *a1_ptr = packedA_panel + row_stride;
//     const int *a2_ptr = packedA_panel + 2 * row_stride;
//     const int *a3_ptr = packedA_panel + 3 * row_stride;
//     const int *a4_ptr = packedA_panel + 4 * row_stride;
//     const int *a5_ptr = packedA_panel + 5 * row_stride;
//     const int *a6_ptr = packedA_panel + 6 * row_stride;
//     const int *a7_ptr = packedA_panel + 7 * row_stride;
//     const int *b_ptr  = packedB_panel;

//     int k = 0;
//     for (; k + 4 <= block_k; k += 4) {
//         int32x4_t b0 = vld1q_s32(b_ptr); b_ptr += SIMD_WIDTH;
//         int32x4_t b1 = vld1q_s32(b_ptr); b_ptr += SIMD_WIDTH;
//         int32x4_t b2 = vld1q_s32(b_ptr); b_ptr += SIMD_WIDTH;
//         int32x4_t b3 = vld1q_s32(b_ptr); b_ptr += SIMD_WIDTH;

//         int32x4_t a0_vec = vld1q_s32(a0_ptr); a0_ptr += SIMD_WIDTH;
//         int32x4_t a1_vec = vld1q_s32(a1_ptr); a1_ptr += SIMD_WIDTH;
//         int32x4_t a2_vec = vld1q_s32(a2_ptr); a2_ptr += SIMD_WIDTH;
//         int32x4_t a3_vec = vld1q_s32(a3_ptr); a3_ptr += SIMD_WIDTH;
//         int32x4_t a4_vec = vld1q_s32(a4_ptr); a4_ptr += SIMD_WIDTH;
//         int32x4_t a5_vec = vld1q_s32(a5_ptr); a5_ptr += SIMD_WIDTH;
//         int32x4_t a6_vec = vld1q_s32(a6_ptr); a6_ptr += SIMD_WIDTH;
//         int32x4_t a7_vec = vld1q_s32(a7_ptr); a7_ptr += SIMD_WIDTH;

//         c0 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c0, b0, a0_vec, 0), b1, a0_vec, 1), b2, a0_vec, 2), b3, a0_vec, 3);
//         c1 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c1, b0, a1_vec, 0), b1, a1_vec, 1), b2, a1_vec, 2), b3, a1_vec, 3);
//         c2 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c2, b0, a2_vec, 0), b1, a2_vec, 1), b2, a2_vec, 2), b3, a2_vec, 3);
//         c3 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c3, b0, a3_vec, 0), b1, a3_vec, 1), b2, a3_vec, 2), b3, a3_vec, 3);
//         c4 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c4, b0, a4_vec, 0), b1, a4_vec, 1), b2, a4_vec, 2), b3, a4_vec, 3);
//         c5 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c5, b0, a5_vec, 0), b1, a5_vec, 1), b2, a5_vec, 2), b3, a5_vec, 3);
//         c6 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c6, b0, a6_vec, 0), b1, a6_vec, 1), b2, a6_vec, 2), b3, a6_vec, 3);
//         c7 = vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(vmlaq_laneq_s32(c7, b0, a7_vec, 0), b1, a7_vec, 1), b2, a7_vec, 2), b3, a7_vec, 3);
//     }
//     for (; k < block_k; k++) {
//         int32x4_t b_vec = vld1q_s32(b_ptr);
//         b_ptr += SIMD_WIDTH;
//         c0 = vmlaq_n_s32(c0, b_vec, *a0_ptr++);
//         c1 = vmlaq_n_s32(c1, b_vec, *a1_ptr++);
//         c2 = vmlaq_n_s32(c2, b_vec, *a2_ptr++);
//         c3 = vmlaq_n_s32(c3, b_vec, *a3_ptr++);
//         c4 = vmlaq_n_s32(c4, b_vec, *a4_ptr++);
//         c5 = vmlaq_n_s32(c5, b_vec, *a5_ptr++);
//         c6 = vmlaq_n_s32(c6, b_vec, *a6_ptr++);
//         c7 = vmlaq_n_s32(c7, b_vec, *a7_ptr++);
//     }

//     vst1q_s32(&output[(i + 0) * N + j], c0);
//     vst1q_s32(&output[(i + 1) * N + j], c1);
//     vst1q_s32(&output[(i + 2) * N + j], c2);
//     vst1q_s32(&output[(i + 3) * N + j], c3);
//     vst1q_s32(&output[(i + 4) * N + j], c4);
//     vst1q_s32(&output[(i + 5) * N + j], c5);
//     vst1q_s32(&output[(i + 6) * N + j], c6);
//     vst1q_s32(&output[(i + 7) * N + j], c7);
// }

// static inline void do_l1_block(const int *inputA, const int *inputB, int *output,
//                                int i0, int i1, int j0, int j1, int k0, int k1, int N, int K) {
//     for (int i = i0; i < i1; i++) {
//         for (int j = j0; j < j1; j += SIMD_WIDTH) {
//             if (j + SIMD_WIDTH <= j1) {
//                 int32x4_t acc = (k0 == 0) ? vdupq_n_s32(0) : vld1q_s32(output + i * N + j);
//                 for (int k = k0; k < k1; k++) {
//                     int32_t a_val = inputA[i * K + k];
//                     int32x4_t b_vec = vld1q_s32(inputB + k * N + j);
//                     acc = vmlaq_n_s32(acc, b_vec, a_val);
//                 }
//                 vst1q_s32(output + i * N + j, acc);
//             } else {
//                 for (int jj = j; jj < j1; jj++) {
//                     int sum = (k0 == 0) ? 0 : output[i * N + jj];
//                     for (int k = k0; k < k1; k++)
//                         sum += inputA[i * K + k] * inputB[k * N + jj];
//                     output[i * N + jj] = sum;
//                 }
//                 break;
//             }
//         }
//     }
// }

// static void core_matmul_block(const int *inputA, const int *inputB, int *output, int M, int N, int K,
//                               int bi, int i_block, int n_start, int n_len,
//                               const int *global_packedA, const int *global_packedB) {
//     const int num_k_blocks = (K + L2_K_MAX - 1) / L2_K_MAX;
//     const int num_n_blocks = (N + L2_N_MAX - 1) / L2_N_MAX;
//     const int prefetch_step = 16;

//     alignas(64) int packedB[L2_N_MAX * L2_K_MAX];
//     alignas(64) int packedA[L2_M_MAX * L2_K_MAX];

//     for (int bj = 0; bj < n_len; bj += L2_N_MAX) {
//         int j_block = std::min(L2_N_MAX, n_len - bj);
//         int j_start = n_start + bj;

//         for (int bk = 0; bk < K; bk += L2_K_MAX) {
//             int block_k = std::min(L2_K_MAX, K - bk);
//             const int bk_block = bk / L2_K_MAX;
//             const int *packedB_block = global_packedB ? (global_packedB + (bk_block * num_n_blocks + (n_start + bj) / L2_N_MAX) * (L2_N_MAX * L2_K_MAX)) : packedB;

//             if (bk + L2_K_MAX < K) {
//                 int next_bk = bk + L2_K_MAX;
//                 int next_k_len = std::min(L2_K_MAX, K - next_bk);
//                 if (global_packedA) {
//                     for (int r = 0; r < i_block && r < L2_M_MAX; r++)
//                         PREFETCH_A(global_packedA + ((bi / L2_M_MAX) * num_k_blocks + (next_bk / L2_K_MAX)) * (L2_M_MAX * L2_K_MAX) + r * L2_K_MAX);
//                 } else {
//                     for (int r = 0; r < i_block; r++)
//                         for (int off = 0; off < next_k_len; off += prefetch_step)
//                             PREFETCH_A(inputA + (bi + r) * K + next_bk + off);
//                 }
//                 if (global_packedB)
//                     PREFETCH_B(global_packedB + ((next_bk / L2_K_MAX) * num_n_blocks + (n_start + bj) / L2_N_MAX) * (L2_N_MAX * L2_K_MAX));
//                 else {
//                     for (int r = 0; r < next_k_len; r++)
//                         for (int off = 0; off < j_block; off += prefetch_step)
//                             PREFETCH_B(inputB + (next_bk + r) * N + j_start + off);
//                 }
//             }

//             if (!global_packedB)
//                 pack_B(inputB, packedB, N, j_start, j_block, bk, block_k);

//             for (int kk = 0; kk < block_k; kk += L1_K_MAX) {
//                 int k_len = std::min(L1_K_MAX, block_k - kk);
//                 const int *packedA_block = global_packedA ? (global_packedA + ((bi / L2_M_MAX) * num_k_blocks + bk_block) * (L2_M_MAX * L2_K_MAX)) : nullptr;

//                 for (int ii = 0; ii < i_block; ii += L1_M_MAX) {
//                     int i_tile = std::min(L1_M_MAX, i_block - ii);
//                     if (!global_packedA)
//                         pack_A(inputA, packedA, K, bi + ii, i_tile, bk + kk, k_len);

//                     for (int jj = 0; jj < j_block; jj += L1_N_MAX) {
//                         int j_tile = std::min(L1_N_MAX, j_block - jj);
//                         const int *cur_A = global_packedA ? (packedA_block + ii * L2_K_MAX + kk) : packedA;
//                         int a_stride = global_packedA ? L2_K_MAX : k_len;

//                         for (int ii_inner = 0; ii_inner + MR <= i_tile; ii_inner += MR) {
//                             for (int jj_inner = 0; jj_inner + NR <= j_tile; jj_inner += NR) {
//                                 const int *a_panel = global_packedA ? (cur_A + ii_inner * L2_K_MAX) : &packedA[ii_inner * k_len];
//                                 const int *b_panel = packedB_block + ((jj + jj_inner) / NR) * block_k * NR + kk * NR;
//                                 int initial_zero = (bk + kk == 0) ? 1 : 0;
//                                 micro_kernel_8x4(a_panel, a_stride, b_panel, output,
//                                                 bi + ii + ii_inner, j_start + jj + jj_inner,
//                                                 k_len, N, initial_zero);
//                             }
//                         }
//                         for (int ii_inner = (i_tile / MR) * MR; ii_inner + 4 <= i_tile; ii_inner += 4) {
//                             for (int jj_inner = 0; jj_inner + NR <= j_tile; jj_inner += NR) {
//                                 const int *a_panel = global_packedA ? (cur_A + ii_inner * L2_K_MAX) : &packedA[ii_inner * k_len];
//                                 const int *b_panel = packedB_block + ((jj + jj_inner) / NR) * block_k * NR + kk * NR;
//                                 int initial_zero = (bk + kk == 0) ? 1 : 0;
//                                 micro_kernel_4x4(a_panel, a_stride, b_panel, output,
//                                                 bi + ii + ii_inner, j_start + jj + jj_inner,
//                                                 k_len, N, initial_zero);
//                             }
//                         }
//                     }
//                 }
//             }

//             int i_done = (i_block / MR) * MR;
//             int j_done = (j_block / NR) * NR;
//             if (bi + i_done < bi + i_block)
//                 do_l1_block(inputA, inputB, output, bi + i_done, bi + i_block, j_start, j_start + j_block, bk, bk + block_k, N, K);
//             if (j_start + j_done < j_start + j_block)
//                 do_l1_block(inputA, inputB, output, bi, bi + i_done, j_start + j_done, j_start + j_block, bk, bk + block_k, N, K);
//         }
//     }
// }

// static void core_matmul(const int *global_packedA, const int *global_packedB,
//                         const int *inputA, const int *inputB, int *output,
//                         const int M, const int N, const int K, int n_start, int n_len) {
//     for (int bi = 0; bi < M; bi += L2_M_MAX) {
//         int i_block = std::min(L2_M_MAX, M - bi);
//         core_matmul_block(inputA, inputB, output, M, N, K, bi, i_block, n_start, n_len,
//                           global_packedA, global_packedB);
//     }
// }


// #define NUM_THREADS   6
// #define Align         16

// /* inputA, inputB, and output must be allocated on a 64-byte boundary (e.g. posix_memalign,
//    aligned_alloc, or alignas(64)) for optimal NEON load/store and prefetch.

//    Global pre-pack B only (pack_B gives large speedup; A is packed locally per block to avoid
//    cache overload from a full packed A). Tune L2_*_MAX / L1_*_MAX if cache pressure is high. */
// void matmul(const int *inputA, const int *inputB, int *output, const int M, const int N, const int K) {
//     const int num_k_blocks = (K + L2_K_MAX - 1) / L2_K_MAX;
//     const int num_n_blocks = (N + L2_N_MAX - 1) / L2_N_MAX;
//     const size_t sizeB = (size_t)num_k_blocks * num_n_blocks * L2_N_MAX * L2_K_MAX * sizeof(int);

//     void *pB = nullptr;
//     int *packedB = nullptr;
//     if (posix_memalign(&pB, 64, sizeB) == 0)
//         packedB = (int *)pB;

//     if (packedB) {
//         for (int bk = 0; bk < K; bk += L2_K_MAX) {
//             int block_k = std::min(L2_K_MAX, K - bk);
//             for (int bj = 0; bj < N; bj += L2_N_MAX) {
//                 int j_block = std::min(L2_N_MAX, N - bj);
//                 int *dst = packedB + ((bk / L2_K_MAX) * num_n_blocks + bj / L2_N_MAX) * (L2_N_MAX * L2_K_MAX);
//                 pack_B(inputB, dst, N, bj, j_block, bk, block_k);
//             }
//         }
//     }

// #pragma omp parallel for schedule(dynamic)
//     for (int bj = 0; bj < N; bj += L2_N_MAX) {
//         int j_block = std::min(L2_N_MAX, N - bj);
//         core_matmul(nullptr, packedB, inputA, inputB, output, M, N, K, bj, j_block);
//     }

//     free(packedB);
// }
