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
