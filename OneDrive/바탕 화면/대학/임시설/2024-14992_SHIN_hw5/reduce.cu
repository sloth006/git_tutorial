#include <cuda_runtime.h>
#include <iostream>

// Ampere+ (sm_80): explicit L2 prefetch.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void prefetch_global_l2(const void* addr) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(addr) : "memory");
}
#else
__device__ __forceinline__ void prefetch_global_l2(const void*) {}
#endif

// --- Error Checking Macro ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =====================================================================
// KERNEL 1: SMALL WORKLOADS (N <= 1024)
// =====================================================================
__global__ void reduce_kernel_small(const float* __restrict__ d_input, int N, 
                                    float* __restrict__ d_final_result) {
    __shared__ float warpLevelSums[32]; 
    float sum = 0.0f;

    if (threadIdx.x < N) {
        sum = d_input[threadIdx.x];
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane_id == 0) {
        warpLevelSums[warp_id] = sum;
    }
    __syncthreads();

    float val = 0.0f;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        val = (lane_id < num_warps) ? warpLevelSums[lane_id] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
    }

    if (threadIdx.x == 0) {
        *d_final_result = val; 
    }
}

// =====================================================================
// KERNEL 2: LARGE WORKLOADS (N > 1024)
// =====================================================================
__global__ void reduce_kernel_large(const float* __restrict__ d_input, int N,
                                    float* __restrict__ d_final_result) {
    __shared__ float warpLevelSums[8];

    // float4 loads must be 16-byte aligned. cudaMalloc guarantees 256-byte alignment.
    const float4* p4 = reinterpret_cast<const float4*>(d_input);
    float sum = 0.0f;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int M = N / 4; 

    int j = tid;
    while (j < M) {
        const float4 v = p4[j];
        sum += v.x + v.y + v.z + v.w;
        if (j + stride < M) {
            prefetch_global_l2(p4 + j + stride);
        }
        j += stride;
    }

    for (int i = 4 * M + tid; i < N; i += stride) {
        sum += d_input[i];
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane_id == 0) {
        warpLevelSums[warp_id] = sum;
    }
    __syncthreads();

    float val = 0.0f;
    if (warp_id == 0) {
        val = (lane_id < 8) ? warpLevelSums[lane_id] : 0.0f;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
    }

    if (threadIdx.x == 0) {
        // atomicAdd for floats is natively supported on Ampere
        atomicAdd(d_final_result, val);
    }
}

// =====================================================================
// WRAPPER EXPECTED BY MAIN.CPP
// =====================================================================
void reduce(float* h_idata, float* h_odata, float* g_idata, float* g_odata, int N) {
    
    // Correctly map the device pointers based on main.cpp's arguments
    float* d_input = g_idata; 
    float* d_final_result = g_odata; 

    // Zero out the result variable before we run the kernel so atomicAdd starts from 0
    CUDA_CHECK(cudaMemset(d_final_result, 0, sizeof(float)));

    if (N <= 1024) {
        int block_size = ((N + 31) / 32) * 32;
        if (block_size == 0) block_size = 32;
        reduce_kernel_small<<<1, block_size>>>(d_input, N, d_final_result);
    } else {
        int max_blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, reduce_kernel_large, 256, 0));

        const int needed = (N + 255) / 256;
        int max_grid = max_blocks_per_sm * 8; // Orin Nano has 8 SMs
        if (max_grid <= 0) max_grid = needed;
        const int grid = (needed < max_grid) ? needed : max_grid;

        reduce_kernel_large<<<grid, 256>>>(d_input, N, d_final_result);
    }
    
    CUDA_CHECK(cudaGetLastError());
    // main.cpp calls cudaDeviceSynchronize() right after this, so we are good to go.
}