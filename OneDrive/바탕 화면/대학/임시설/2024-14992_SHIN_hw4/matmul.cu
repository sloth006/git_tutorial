#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <math.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <algorithm>

#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>

struct VersionConfig {
    long int M;
    long int K;
    long int N;
};

#define n_ver 3
#define SEED 1234


// Architecture: NVIDIA Ampere
// CUDA Cores: 1024
// Tensor Cores: do not use
// SM Count: 8
// L1 Cache: 128 KB per SM
// L2 Cache: 2 MB

// 3-level blocked SIMT GEMM (no tensor cores), shaped for Ampere-class cache hierarchy (Orin: ~128 KiB L1/SM, ~2 MiB L2):
//  L2 / CTA level — BM×BN output per block (64×64) keeps panels of A/B hot in L2 while K advances.
//  Shared / "fast memory" level — BK=32 K-strip in SMEM, ping-pong shA/shB (~33 KiB per stage).
//  Register level — each thread accumulates TM×TN = 4×4 outputs (L1 traffic only via SMEM/ops, not C spilling).

constexpr int BM = 64;   // CTA tile M
constexpr int BN = 64;   // CTA tile N
constexpr int BK = 32;   // K depth per shared stage (fits <48 KiB SMEM with two buffers + padding)
constexpr int TM = 4;    // register tile rows per thread
constexpr int TN = 4;    // register tile cols per thread
constexpr int THR_X = BN / TN;
constexpr int THR_Y = BM / TM;
constexpr int THREADS = THR_X * THR_Y;
static_assert(THR_X * TM == BM && THR_Y * TN == BN, "CTA tile must divide thread tile");
static_assert((BM * BK) % THREADS == 0 && (BK * BN) % THREADS == 0, "even load distribution");

constexpr int STRIDE_A = BK + 1;
constexpr int STRIDE_B = BN + 1;

static_assert(BK % 4 == 0 && BN % 4 == 0, "vector loads need BK/BN divisible by 4");

// Swizzle k-index in A SMEM to reduce 32-bank conflicts when warps walk rows at fixed k.
__device__ __forceinline__ int swizzle_a_col(int row, int k) {
    return k ^ ((row >> 1) & 7);
}

// 128-bit global loads (float4); scalar scatter into SMEM (swizzle/padding prevent vector SMEM stores).
__device__ __forceinline__ void load_cta_tile(
    float (*sA)[STRIDE_A], float (*sB)[STRIDE_B],
    const float* __restrict__ A, const float* __restrict__ B,
    int M, int N, int K, int kk, int bx, int by) {
    const int tid = threadIdx.y * THR_X + threadIdx.x;

    const int A_vec_tiles = (BM * BK) / 4;
    for (int p = tid; p < A_vec_tiles; p += THREADS) {
        const int row_l = p / (BK / 4);
        const int k_base = (p % (BK / 4)) * 4;
        const int g_row = by * BM + row_l;
        const int g_k0 = kk + k_base;
        if (g_row < M && g_k0 + 3 < K) {
            const float* src = &A[(size_t)g_row * K + g_k0];
            if (reinterpret_cast<uintptr_t>(src) % 16 == 0) {
                const float4 v = *reinterpret_cast<const float4*>(src);
                sA[row_l][swizzle_a_col(row_l, k_base + 0)] = v.x;
                sA[row_l][swizzle_a_col(row_l, k_base + 1)] = v.y;
                sA[row_l][swizzle_a_col(row_l, k_base + 2)] = v.z;
                sA[row_l][swizzle_a_col(row_l, k_base + 3)] = v.w;
            } else {
                for (int t = 0; t < 4; ++t) {
                    const int g_k = g_k0 + t;
                    float* d = &sA[row_l][swizzle_a_col(row_l, k_base + t)];
                    *d = (g_row < M && g_k < K) ? A[(size_t)g_row * K + g_k] : 0.0f;
                }
            }
        } else {
            for (int t = 0; t < 4; ++t) {
                const int g_k = g_k0 + t;
                float* d = &sA[row_l][swizzle_a_col(row_l, k_base + t)];
                *d = (g_row < M && g_k < K) ? A[(size_t)g_row * K + g_k] : 0.0f;
            }
        }
    }

    const int B_vec_tiles = BK * (BN / 4);
    for (int p = tid; p < B_vec_tiles; p += THREADS) {
        const int k_l = p / (BN / 4);
        const int col_base = (p % (BN / 4)) * 4;
        const int g_k = kk + k_l;
        const int g_col0 = bx * BN + col_base;
        if (g_k < K && g_col0 + 3 < N) {
            const float* src = &B[(size_t)g_k * N + g_col0];
            if (reinterpret_cast<uintptr_t>(src) % 16 == 0) {
                const float4 v = *reinterpret_cast<const float4*>(src);
                sB[k_l][col_base + 0] = v.x;
                sB[k_l][col_base + 1] = v.y;
                sB[k_l][col_base + 2] = v.z;
                sB[k_l][col_base + 3] = v.w;
            } else {
                for (int t = 0; t < 4; ++t) {
                    const int g_col = g_col0 + t;
                    sB[k_l][col_base + t] =
                        (g_k < K && g_col < N) ? B[(size_t)g_k * N + g_col] : 0.0f;
                }
            }
        } else {
            for (int t = 0; t < 4; ++t) {
                const int g_col = g_col0 + t;
                sB[k_l][col_base + t] =
                    (g_k < K && g_col < N) ? B[(size_t)g_k * N + g_col] : 0.0f;
            }
        }
    }
}

__global__ void __launch_bounds__(THREADS, 2)
matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
       int M, int N, int K) {
    __shared__ __align__(16) float shA[2][BM][STRIDE_A];
    __shared__ __align__(16) float shB[2][BK][STRIDE_B];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int num_tiles = (K + BK - 1) / BK;

    float accum[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    int stage = 0;
    for (int ti = 0; ti < num_tiles; ++ti) {
        const int kk = ti * BK;

        load_cta_tile(shA[stage], shB[stage], A, B, M, N, K, kk, bx, by);
        __syncthreads();

        // Level-1: register tile — outer k over BK is short; inner TM×TN uses fmaf for throughput.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int ii = 0; ii < TM; ++ii) {
                const int row_sm = ty * TM + ii;
                const float a_val = shA[stage][row_sm][swizzle_a_col(row_sm, k)];
                #pragma unroll
                for (int jj = 0; jj < TN; ++jj) {
                    const float b_val = shB[stage][k][tx * TN + jj];
                    accum[ii][jj] = fmaf(a_val, b_val, accum[ii][jj]);
                }
            }
        }

        __syncthreads();
        stage = 1 - stage;
    }

    #pragma unroll
    for (int ii = 0; ii < TM; ++ii) {
        const int r = by * BM + ty * TM + ii;
        const int c = bx * BN + tx * TN;  // TN == 4
        float* row = &C[(size_t)r * N + c];
        if (r < M && c + 3 < N &&
            (reinterpret_cast<uintptr_t>(row) % sizeof(float4) == 0)) {
            const float4 res = make_float4(accum[ii][0], accum[ii][1], accum[ii][2], accum[ii][3]);
            *reinterpret_cast<float4*>(row) = res;
        } else {
            #pragma unroll
            for (int jj = 0; jj < TN; ++jj) {
                if (r < M && c + jj < N) {
                    C[(size_t)r * N + c + jj] = accum[ii][jj];
                }
            }
        }
    }
}

void fill_random(float* arr, size_t size) {
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        arr[i] = dist(gen);
    }
}

bool validate(const float* A, const float* B, size_t size, float rtol = 1e-3f, float atol = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        float diff = fabs(A[i] - B[i]);
        float tol = atol + rtol * fabs(A[i]);
        if (diff > tol) return false;
    }
    return true;
}

void print_first_10(const float* arr, size_t size) {
    size_t limit = std::min(size, (size_t)10);
    for (size_t i = 0; i < limit; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

float* cpu(const float* A, const float* B, int A_height, int A_width, int B_width) {
    float* C = new float[(size_t)A_height * B_width]();

    omp_set_num_threads(6);
    // Only parallelize i: parallel k would race on updates to the same row of C.
    #pragma omp parallel for
    for (int i = 0; i < A_height; ++i) {
        for (int k = 0; k < A_width; ++k) {
            float Aik = A[i * A_width + k];
            for (int j = 0; j < B_width; ++j) {
                C[i * B_width + j] += Aik * B[k * B_width + j];
            }
        }
    }

    return C;
}

int main(int argc, char* argv[]) {

    VersionConfig configs[n_ver] = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096}
    };

    bool use_custom_config = false;
    VersionConfig custom_config;

    if (argc == 4) {
        custom_config.M = std::atol(argv[1]);
        custom_config.K  = std::atol(argv[2]);
        custom_config.N  = std::atol(argv[3]);
        use_custom_config = true;
    } else if (argc != 1) {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << "\n";
        std::cerr << "  " << argv[0] << " M K N\n";
        return 1;
    }

    // Run kernel
    std::cout << "----------------Mat Mul----------------\n";

    int num_tests = use_custom_config ? 1 : n_ver;

    // Run all version
    for(int ver = 0; ver < num_tests; ver++) {

        // Random input matrix
        VersionConfig config = use_custom_config ? custom_config : configs[ver];

        int A_height = config.M;
        int A_width = config.K;
        int B_height = config.K;
        int B_width = config.N;

        size_t A_size = (size_t)A_height * A_width;
        size_t B_size = (size_t)B_height * B_width;
        size_t C_size = (size_t)A_height * B_width;

        float* A_host = new float[A_size];
        float* B_host = new float[B_size];
        float* C_host = new float[C_size];

        fill_random(A_host, A_size);
        fill_random(B_host, B_size);

        for (size_t i = 0; i < C_size; ++i) C_host[i] = 0.0f;

        // TODO: Allocate device memory and copy A_host, B_host to device memory
        float *A_dev, *B_dev, *C_dev;
        cudaMalloc(&A_dev, A_size * sizeof(float));
        cudaMalloc(&B_dev, B_size * sizeof(float));
        cudaMalloc(&C_dev, C_size * sizeof(float));
        cudaStream_t stream{};
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(A_dev, A_host, A_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(B_dev, B_host, B_size * sizeof(float), cudaMemcpyHostToDevice, stream);

        cudaEvent_t start, stop;
        float execution_time = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
        // TODO: launch kernel -> hint! matmul<<<dimGrid, dimBlock>>>( device memory );
        const dim3 dimBlock(THR_X, THR_Y);
        const dim3 dimGrid((B_width + BN - 1) / BN, (A_height + BM - 1) / BM);
        matmul<<<dimGrid, dimBlock, 0, stream>>>(A_dev, B_dev, C_dev, A_height, B_width, A_width);

        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);

        cudaEventElapsedTime(&execution_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(cudaErr) << std::endl;
        } else {
            std::cout << ">>> Test" << ver + 1 << " Execution time: " << execution_time << "ms" << std::endl;
        }

        // TODO: copy back kernel result to C_host
        cudaMemcpyAsync(C_host, C_dev, C_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaStreamDestroy(stream);

        float* C_answer = cpu(A_host, B_host, A_height, A_width, B_width);

        if (validate(C_answer, C_host, C_size, 1e-3f, 1e-2f)) {
            std::cout << ">>> Test Pass!" << std::endl;
        } else {
            std::cout << ">>> Test Fail!" << std::endl;
            std::cout << ">>> >>First 10 elements of C_answer: \n";
            print_first_10(C_answer, C_size);
            std::cout << ">>> >>First 10 elements of C_host: \n";
            print_first_10(C_host, C_size);
        }

        cudaFree(A_dev);
        cudaFree(B_dev);
        cudaFree(C_dev);

        delete[] A_host;
        delete[] B_host;
        delete[] C_host;
        delete[] C_answer;
    }

    return 0;
}
