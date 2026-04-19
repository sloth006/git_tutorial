#include "kernel.h"

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {

inline void sync_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

constexpr int kThreads = 256;

__global__ void relu_inplace_fp32(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v < 0.f ? 0.f : v;
    }
}

__global__ void gelu_fp32(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

__global__ void sigmoid_fp32(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 1.f / (1.f + expf(-v));
    }
}

__global__ void add_same_fp32(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        o[i] = a[i] + b[i];
    }
}

__global__ void add_bc24_fp32(const float* bc, const float* bchw, float* o, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) {
        return;
    }
    int w = idx % W;
    int t = idx / W;
    int h = t % H;
    t /= H;
    int c = t % C;
    int b = t / C;
    float tbc = bc[b * C + c];
    o[idx] = bchw[idx] + tbc;
}

__global__ void add_bc11_fp32(const float* small_b, const float* big, float* o, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) {
        return;
    }
    int w = idx % W;
    int t = idx / W;
    int h = t % H;
    t /= H;
    int c = t % C;
    int b = t / C;
    float s = small_b[b];
    o[idx] = big[idx] + s;
}

__global__ void sub_same_fp32(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        o[i] = a[i] - b[i];
    }
}

__global__ void sub_bchw_bc11_fp32(const float* a, const float* b1111, float* o, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) {
        return;
    }
    int w = idx % W;
    int t = idx / W;
    int h = t % H;
    t /= H;
    int c = t % C;
    int b = t / C;
    o[idx] = a[idx] - b1111[b];
}

__global__ void mul_same_fp32(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        o[i] = a[i] * b[i];
    }
}

__global__ void mul_bc11_fp32(const float* small_b, const float* big, float* o, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) {
        return;
    }
    int w = idx % W;
    int t = idx / W;
    int h = t % H;
    t /= H;
    int c = t % C;
    int b = t / C;
    o[idx] = big[idx] * small_b[b];
}

__global__ void mul_bc1hw_fp32(const float* a1hw, const float* big, float* o, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) {
        return;
    }
    int w = idx % W;
    int t = idx / W;
    int h = t % H;
    t /= H;
    int c = t % C;
    int b = t / C;
    int aidx = (b * H + h) * W + w;
    o[idx] = big[idx] * a1hw[aidx];
}

__global__ void div_same_fp32(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        o[i] = a[i] / b[i];
    }
}

__global__ void div_bchw_bc11_fp32(const float* a, const float* b1111, float* o, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) {
        return;
    }
    int w = idx % W;
    int t = idx / W;
    int h = t % H;
    t /= H;
    int c = t % C;
    int b = t / C;
    o[idx] = a[idx] / b1111[b];
}

__global__ void scale_bias_fp32(const float* x, float* y, int n, float scale, float bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = scale * x[i] + bias;
    }
}

__global__ void conv2d_pad_fill_fp32(const float* x, float* out, int B, int C, int H, int W, int OH, int OW, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * OH * OW;
    if (idx >= total) {
        return;
    }
    out[idx] = 0.f;
    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int c = t % C;
    int b = t / C;
    int iw = ow - pad;
    int ih = oh - pad;
    if (iw >= 0 && iw < W && ih >= 0 && ih < H) {
        int xidx = ((b * C + c) * H + ih) * W + iw;
        out[idx] = x[xidx];
    }
}

__global__ void upsample_nearest2d_fp32(const float* x, float* y, int B, int C, int H, int W, int OH, int OW, int sf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * OH * OW;
    if (idx >= total) {
        return;
    }
    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int c = t % C;
    int b = t / C;
    int ih = oh / sf;
    int iw = ow / sf;
    int xidx = ((b * C + c) * H + ih) * W + iw;
    y[idx] = x[xidx];
}

__global__ void conv2d_forward_fp32(float* __restrict__ Out, const float* __restrict__ X, const float* __restrict__ W,
                                    const float* __restrict__ Bias, int B, int IC, int PH, int PW, int OC, int R, int S,
                                    int stride, int OH, int OW, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * OH * OW;
    if (idx >= total) {
        return;
    }
    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int oc = t % OC;
    int b = t / OC;

    int GOC = OC / groups;
    int GIC = IC / groups;
    int g = oc / GOC;

    float acc = Bias[oc];
    for (int ic = 0; ic < GIC; ++ic) {
        int gic = g * GIC + ic;
        for (int kh = 0; kh < R; ++kh) {
            for (int kw = 0; kw < S; ++kw) {
                int h_in = oh * stride + kh;
                int w_in = ow * stride + kw;
                int xi = ((b * IC + gic) * PH + h_in) * PW + w_in;
                int wi = (((oc * GIC + ic) * R + kh) * S + kw);
                acc += X[xi] * W[wi];
            }
        }
    }
    Out[idx] = acc;
}

__global__ void layer_norm_forward_fp32(const float* __restrict__ x, float* __restrict__ y,
                                        const float* __restrict__ wln, const float* __restrict__ bln, int C, int H, int W,
                                        float eps) {
    extern __shared__ float sh[];
    int b = blockIdx.x;
    int N = C * H * W;
    int tid = threadIdx.x;
    int bd = blockDim.x;

    float* sreduce = sh;

    float sum = 0.f;
    for (int i = tid; i < N; i += bd) {
        sum += x[b * N + i];
    }
    sreduce[tid] = sum;
    __syncthreads();
    for (int s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sreduce[tid] += sreduce[tid + s];
        }
        __syncthreads();
    }
    float mean = sreduce[0] / static_cast<float>(N);

    float vsum = 0.f;
    for (int i = tid; i < N; i += bd) {
        float d = x[b * N + i] - mean;
        vsum += d * d;
    }
    sreduce[tid] = vsum;
    __syncthreads();
    for (int s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sreduce[tid] += sreduce[tid + s];
        }
        __syncthreads();
    }
    float var = sreduce[0] / static_cast<float>(N);
    float invstd = rsqrtf(var + eps);

    for (int i = tid; i < N; i += bd) {
        int rem = i;
        int c = rem / (H * W);
        rem %= (H * W);
        int h = rem / W;
        int w = rem % W;
        float xv = x[b * N + i];
        float norm = (xv - mean) * invstd;
        y[b * N + i] = norm * wln[c] + bln[c];
    }
}

// WMMA FP16 m16n16k16 tiles, FP32 accumulators; shared < 3 KiB (<< 163 KiB per-block cap on Ampere).
__global__ void __launch_bounds__(32, 16) linear_wmma_fp16acc_kernel(const float* __restrict__ x,
                                                                   const float* __restrict__ W,
                                                                   const float* __restrict__ Bias,
                                                                   float* __restrict__ y, int I, int O,
                                                                   int use_bias) {
    using namespace nvcuda::wmma;

    const int o0 = blockIdx.x * 16;
    if (o0 >= O) {
        return;
    }

    __shared__ alignas(16) half sW[16 * 16];
    __shared__ alignas(16) half sB[16 * 16];
    __shared__ alignas(16) float sC[16 * 16];

    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.f);

    for (int k0 = 0; k0 < I; k0 += 16) {
        for (int idx = threadIdx.x; idx < 256; idx += blockDim.x) {
            const int ty = idx / 16;
            const int tx = idx % 16;
            const int o = o0 + ty;
            const int k = k0 + tx;
            float wf = 0.f;
            if (o < O && k < I) {
                wf = W[o * I + k];
            }
            sW[ty * 16 + tx] = __float2half_rn(wf);
        }
        for (int idx = threadIdx.x; idx < 256; idx += blockDim.x) {
            const int k = idx / 16;
            const int n = idx % 16;
            float xf = 0.f;
            if (n == 0 && k0 + k < I) {
                xf = x[k0 + k];
            }
            sB[n * 16 + k] = __float2half_rn(xf);
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
        load_matrix_sync(a_frag, sW, 16);
        load_matrix_sync(b_frag, sB, 16);
        mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();
    }

    store_matrix_sync(sC, acc, 16, mem_row_major);
    __syncthreads();

    if (threadIdx.x < 16) {
        const int t = threadIdx.x;
        const int o = o0 + t;
        if (o < O) {
            float v = sC[t * 16 + 0];
            if (use_bias && Bias != nullptr) {
                v += Bias[o];
            }
            y[o] = v;
        }
    }
}

__global__ void linear_fp32_kernel(const float* __restrict__ x, const float* __restrict__ W,
                                   const float* __restrict__ Bias, float* __restrict__ y, int I, int O, int use_bias) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) {
        return;
    }
    float acc = 0.f;
    if (use_bias && Bias != nullptr) {
        acc = Bias[o];
    }
    for (int i = 0; i < I; ++i) {
        acc += x[i] * W[o * I + i];
    }
    y[o] = acc;
}

__global__ void extract_kernel(const float* constants, const float* timestamps, float* out, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        int t_idx = static_cast<int>(timestamps[b]);
        out[b] = constants[t_idx];
    }
}

__global__ void cumsum_kernel(const float* x, float* y, int N) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    float s = 0.f;
    for (int i = 0; i < N; ++i) {
        s += x[i];
        y[i] = s;
    }
}

__global__ void concat_dim1_kernel(const float* a, const float* b, float* o, int B, int Ca, int Cb, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * (Ca + Cb) * HW;
    if (idx >= total) {
        return;
    }
    int hw = idx % HW;
    int t = idx / HW;
    int c = t % (Ca + Cb);
    int batch = t / (Ca + Cb);
    if (c < Ca) {
        int aidx = (batch * Ca + c) * HW + hw;
        o[idx] = a[aidx];
    } else {
        int c2 = c - Ca;
        int bidx = (batch * Cb + c2) * HW + hw;
        o[idx] = b[bidx];
    }
}

__global__ void concat_generic_kernel(const float* a, const float* b, float* o, int dim, int n_outer, int chunk_a,
                                      int chunk_b) {
    (void)dim;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_outer) {
        return;
    }
    const float* pa = a + row * chunk_a;
    const float* pb = b + row * chunk_b;
    float* po = o + row * (chunk_a + chunk_b);
    for (int i = 0; i < chunk_a; ++i) {
        po[i] = pa[i];
    }
    for (int j = 0; j < chunk_b; ++j) {
        po[chunk_a + j] = pb[j];
    }
}

// Flat cosine similarity helpers: narrow global loads to FP16 in registers, accumulate in FP32.
__global__ void cosine_fp16_load_atomic_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                                 float* __restrict__ sum_dot, float* __restrict__ sum_na,
                                                 float* __restrict__ sum_nb, int n) {
    float local_dot = 0.f;
    float local_na = 0.f;
    float local_nb = 0.f;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const __half ha = __float2half_rd(__ldg(a + i));
        const __half hb = __float2half_rd(__ldg(b + i));
        const float fa = __half2float(ha);
        const float fb = __half2float(hb);
        local_dot += fa * fb;
        local_na += fa * fa;
        local_nb += fb * fb;
    }
    atomicAdd(sum_dot, local_dot);
    atomicAdd(sum_na, local_na);
    atomicAdd(sum_nb, local_nb);
}

} // namespace

Tensor launch_relu_kernel_gpu(Tensor& x) {
    int n = x.size();
    float* p = x.fp32();
    int blocks = (n + kThreads - 1) / kThreads;
    relu_inplace_fp32<<<blocks, kThreads>>>(p, n);
    sync_check(cudaGetLastError(), "relu");
    sync_check(cudaDeviceSynchronize(), "relu sync");
    return x;
}

Tensor launch_gelu_gpu(Tensor& x) {
    int n = x.size();
    Tensor y = Tensor::like(x);
    int blocks = (n + kThreads - 1) / kThreads;
    gelu_fp32<<<blocks, kThreads>>>(x.fp32(), y.fp32(), n);
    sync_check(cudaGetLastError(), "gelu");
    sync_check(cudaDeviceSynchronize(), "gelu sync");
    return y;
}

Tensor launch_sigmoid_gpu(Tensor& x) {
    int n = x.size();
    Tensor y = Tensor::like(x);
    int blocks = (n + kThreads - 1) / kThreads;
    sigmoid_fp32<<<blocks, kThreads>>>(x.fp32(), y.fp32(), n);
    sync_check(cudaGetLastError(), "sigmoid");
    sync_check(cudaDeviceSynchronize(), "sigmoid sync");
    return y;
}

Tensor launch_add_same_gpu(Tensor& a, Tensor& b) {
    int n = a.size();
    Tensor o = Tensor::like(a);
    int blocks = (n + kThreads - 1) / kThreads;
    add_same_fp32<<<blocks, kThreads>>>(a.fp32(), b.fp32(), o.fp32(), n);
    sync_check(cudaGetLastError(), "add_same");
    sync_check(cudaDeviceSynchronize(), "add_same sync");
    return o;
}

Tensor launch_add_bc24_gpu(Tensor& bc, Tensor& bchw) {
    int B = bchw.shape_dim(0);
    int C = bchw.shape_dim(1);
    int H = bchw.shape_dim(2);
    int W = bchw.shape_dim(3);
    Tensor o = Tensor::like(bchw);
    int total = B * C * H * W;
    int blocks = (total + kThreads - 1) / kThreads;
    add_bc24_fp32<<<blocks, kThreads>>>(bc.fp32(), bchw.fp32(), o.fp32(), B, C, H, W);
    sync_check(cudaGetLastError(), "add_bc24");
    sync_check(cudaDeviceSynchronize(), "add_bc24 sync");
    return o;
}

Tensor launch_add_bc11_gpu(Tensor& b1111, Tensor& bchw) {
    int B = bchw.shape_dim(0);
    int C = bchw.shape_dim(1);
    int H = bchw.shape_dim(2);
    int W = bchw.shape_dim(3);
    Tensor o = Tensor::like(bchw);
    int total = B * C * H * W;
    int blocks = (total + kThreads - 1) / kThreads;
    add_bc11_fp32<<<blocks, kThreads>>>(b1111.fp32(), bchw.fp32(), o.fp32(), B, C, H, W);
    sync_check(cudaGetLastError(), "add_bc11");
    sync_check(cudaDeviceSynchronize(), "add_bc11 sync");
    return o;
}

Tensor launch_subtract_same_gpu(Tensor& a, Tensor& b) {
    int n = a.size();
    Tensor o = Tensor::like(a);
    int blocks = (n + kThreads - 1) / kThreads;
    sub_same_fp32<<<blocks, kThreads>>>(a.fp32(), b.fp32(), o.fp32(), n);
    sync_check(cudaGetLastError(), "sub_same");
    sync_check(cudaDeviceSynchronize(), "sub_same sync");
    return o;
}

Tensor launch_subtract_bchw_bc11_gpu(Tensor& a, Tensor& b) {
    int B = a.shape_dim(0);
    int C = a.shape_dim(1);
    int H = a.shape_dim(2);
    int W = a.shape_dim(3);
    Tensor o = Tensor::like(a);
    int total = B * C * H * W;
    int blocks = (total + kThreads - 1) / kThreads;
    sub_bchw_bc11_fp32<<<blocks, kThreads>>>(a.fp32(), b.fp32(), o.fp32(), B, C, H, W);
    sync_check(cudaGetLastError(), "sub_bc11");
    sync_check(cudaDeviceSynchronize(), "sub_bc11 sync");
    return o;
}

Tensor launch_multiply_same_gpu(Tensor& a, Tensor& b) {
    int n = a.size();
    Tensor o = Tensor::like(a);
    int blocks = (n + kThreads - 1) / kThreads;
    mul_same_fp32<<<blocks, kThreads>>>(a.fp32(), b.fp32(), o.fp32(), n);
    sync_check(cudaGetLastError(), "mul_same");
    sync_check(cudaDeviceSynchronize(), "mul_same sync");
    return o;
}

Tensor launch_multiply_bc11_gpu(Tensor& b1111, Tensor& bchw) {
    int B = bchw.shape_dim(0);
    int C = bchw.shape_dim(1);
    int H = bchw.shape_dim(2);
    int W = bchw.shape_dim(3);
    Tensor o = Tensor::like(bchw);
    int total = B * C * H * W;
    int blocks = (total + kThreads - 1) / kThreads;
    mul_bc11_fp32<<<blocks, kThreads>>>(b1111.fp32(), bchw.fp32(), o.fp32(), B, C, H, W);
    sync_check(cudaGetLastError(), "mul_bc11");
    sync_check(cudaDeviceSynchronize(), "mul_bc11 sync");
    return o;
}

Tensor launch_multiply_bc1hw_gpu(Tensor& b1hw, Tensor& bchw) {
    int B = bchw.shape_dim(0);
    int C = bchw.shape_dim(1);
    int H = bchw.shape_dim(2);
    int W = bchw.shape_dim(3);
    Tensor o = Tensor::like(bchw);
    int total = B * C * H * W;
    int blocks = (total + kThreads - 1) / kThreads;
    mul_bc1hw_fp32<<<blocks, kThreads>>>(b1hw.fp32(), bchw.fp32(), o.fp32(), B, C, H, W);
    sync_check(cudaGetLastError(), "mul_bc1hw");
    sync_check(cudaDeviceSynchronize(), "mul_bc1hw sync");
    return o;
}

Tensor launch_divide_same_gpu(Tensor& a, Tensor& b) {
    int n = a.size();
    Tensor o = Tensor::like(a);
    int blocks = (n + kThreads - 1) / kThreads;
    div_same_fp32<<<blocks, kThreads>>>(a.fp32(), b.fp32(), o.fp32(), n);
    sync_check(cudaGetLastError(), "div_same");
    sync_check(cudaDeviceSynchronize(), "div_same sync");
    return o;
}

Tensor launch_divide_bchw_bc11_gpu(Tensor& a, Tensor& b) {
    int B = a.shape_dim(0);
    int C = a.shape_dim(1);
    int H = a.shape_dim(2);
    int W = a.shape_dim(3);
    Tensor o = Tensor::like(a);
    int total = B * C * H * W;
    int blocks = (total + kThreads - 1) / kThreads;
    div_bchw_bc11_fp32<<<blocks, kThreads>>>(a.fp32(), b.fp32(), o.fp32(), B, C, H, W);
    sync_check(cudaGetLastError(), "div_bc11");
    sync_check(cudaDeviceSynchronize(), "div_bc11 sync");
    return o;
}

void launch_concatenate_gpu(Tensor& a, Tensor& b, Tensor& out, int dim) {
    if (dim == 1 && a.dim() == 4 && b.dim() == 4) {
        int B = a.shape_dim(0);
        int Ca = a.shape_dim(1);
        int Cb = b.shape_dim(1);
        int H = a.shape_dim(2);
        int W = a.shape_dim(3);
        int HW = H * W;
        int total = B * (Ca + Cb) * HW;
        int blocks = (total + kThreads - 1) / kThreads;
        concat_dim1_kernel<<<blocks, kThreads>>>(a.fp32(), b.fp32(), out.fp32(), B, Ca, Cb, HW);
    } else {
        int n_outer = 1;
        int n_inner = 1;
        for (int i = 0; i < a.dim(); i++) {
            if (i < dim) {
                n_outer *= a.shape_dim(i);
            } else if (i > dim) {
                n_inner *= a.shape_dim(i);
            }
        }
        int chunk_a = a.shape_dim(dim) * n_inner;
        int chunk_b = b.shape_dim(dim) * n_inner;
        int blocks = (n_outer + kThreads - 1) / kThreads;
        concat_generic_kernel<<<blocks, kThreads>>>(a.fp32(), b.fp32(), out.fp32(), dim, n_outer, chunk_a, chunk_b);
    }
    sync_check(cudaGetLastError(), "concat");
    sync_check(cudaDeviceSynchronize(), "concat sync");
}

Tensor launch_upsample_gpu(Tensor& x, int scale_factor) {
    int B = x.shape_dim(0);
    int C = x.shape_dim(1);
    int H = x.shape_dim(2);
    int W = x.shape_dim(3);
    int OH = H * scale_factor;
    int OW = W * scale_factor;
    Tensor out = Tensor::from_shape({B, C, OH, OW});
    int total = B * C * OH * OW;
    int blocks = (total + kThreads - 1) / kThreads;
    upsample_nearest2d_fp32<<<blocks, kThreads>>>(x.fp32(), out.fp32(), B, C, H, W, OH, OW, scale_factor);
    sync_check(cudaGetLastError(), "upsample");
    sync_check(cudaDeviceSynchronize(), "upsample sync");
    return out;
}

Tensor launch_conv2d_pad_gpu(Tensor& x, int padding, Tensor& out) {
    int B = x.shape_dim(0);
    int C = x.shape_dim(1);
    int H = x.shape_dim(2);
    int W = x.shape_dim(3);
    int OH = H + 2 * padding;
    int OW = W + 2 * padding;
    int total = B * C * OH * OW;
    int blocks = (total + kThreads - 1) / kThreads;
    conv2d_pad_fill_fp32<<<blocks, kThreads>>>(x.fp32(), out.fp32(), B, C, H, W, OH, OW, padding);
    sync_check(cudaGetLastError(), "conv2d_pad");
    sync_check(cudaDeviceSynchronize(), "conv2d_pad sync");
    return out;
}

Tensor launch_conv2d_gpu(Tensor& x_padded, Tensor& weight, Tensor& bias, int stride, int padding, int groups) {
    (void)padding;
    int B = x_padded.shape_dim(0);
    int IC = x_padded.shape_dim(1);
    int PH = x_padded.shape_dim(2);
    int PW = x_padded.shape_dim(3);
    int OC = weight.shape_dim(0);
    int R = weight.shape_dim(2);
    int S = weight.shape_dim(3);
    int OH = (PH - R) / stride + 1;
    int OW = (PW - S) / stride + 1;
    Tensor out = Tensor::from_shape({B, OC, OH, OW});
    int total = B * OC * OH * OW;
    int blocks = (total + kThreads - 1) / kThreads;
    conv2d_forward_fp32<<<blocks, kThreads>>>(out.fp32(), x_padded.fp32(), weight.fp32(), bias.fp32(), B, IC, PH, PW, OC,
                                              R, S, stride, OH, OW, groups);
    sync_check(cudaGetLastError(), "conv2d");
    sync_check(cudaDeviceSynchronize(), "conv2d sync");
    return out;
}

Tensor launch_layer_norm_gpu(Tensor& x, Tensor& weight, Tensor& bias, float eps) {
    int B = x.shape_dim(0);
    int C = x.shape_dim(1);
    int H = x.shape_dim(2);
    int W = x.shape_dim(3);
    Tensor y = Tensor::like(x);
    int N = C * H * W;
    int threads = 256;
    while (threads < N && threads < 1024) {
        threads *= 2;
    }
    if (threads > 1024) {
        threads = 1024;
    }
    size_t shmem = threads * sizeof(float);
    layer_norm_forward_fp32<<<B, threads, shmem>>>(x.fp32(), y.fp32(), weight.fp32(), bias.fp32(), C, H, W, eps);
    sync_check(cudaGetLastError(), "layer_norm");
    sync_check(cudaDeviceSynchronize(), "layer_norm sync");
    return y;
}

Tensor launch_linear_gpu(Tensor& x, Tensor& weight, Tensor& bias, bool has_bias) {
    int I = weight.shape_dim(1);
    int O = weight.shape_dim(0);
    Tensor out;
    if (x.shape_dim(0) == 1) {
        out = Tensor::from_shape({1, O});
    } else {
        out = Tensor::from_shape({O});
    }
    const float* xp = x.fp32();
    float* yp = out.fp32();
    const int use_bias = has_bias ? 1 : 0;
    const float* bp = has_bias ? bias.fp32() : nullptr;

    const bool use_wmma = (I > 0 && O > 0 && (I % 16) == 0 && (O % 16) == 0);
    if (use_wmma) {
        const int grid = (O + 15) / 16;
        linear_wmma_fp16acc_kernel<<<grid, 32>>>(xp, weight.fp32(), bp, yp, I, O, use_bias);
        sync_check(cudaGetLastError(), "linear_wmma");
        sync_check(cudaDeviceSynchronize(), "linear_wmma sync");
    } else {
        int blocks = (O + kThreads - 1) / kThreads;
        linear_fp32_kernel<<<blocks, kThreads>>>(xp, weight.fp32(), bp, yp, I, O, use_bias);
        sync_check(cudaGetLastError(), "linear_fp32");
        sync_check(cudaDeviceSynchronize(), "linear_fp32 sync");
    }
    return out;
}

Tensor launch_extract_gpu(Tensor& constants, Tensor& timestamps, Tensor& output) {
    // Match CPU extract: one scalar per batch plane of `output` (same indexing as timestamps[b]).
    const int batch = output.shape_dim(0);
    int blocks = (batch + kThreads - 1) / kThreads;
    extract_kernel<<<blocks, kThreads>>>(constants.fp32(), timestamps.fp32(), output.fp32(), batch);
    sync_check(cudaGetLastError(), "extract");
    sync_check(cudaDeviceSynchronize(), "extract sync");
    return output;
}

Tensor launch_cumsum_gpu(Tensor& x) {
    Tensor y = Tensor::like(x);
    int N = x.size();
    cumsum_kernel<<<1, 1>>>(x.fp32(), y.fp32(), N);
    sync_check(cudaGetLastError(), "cumsum");
    sync_check(cudaDeviceSynchronize(), "cumsum sync");
    return y;
}

Tensor launch_normalize_neg1_1_gpu(Tensor& x) {
    int n = x.size();
    Tensor y = Tensor::like(x);
    int blocks = (n + kThreads - 1) / kThreads;
    scale_bias_fp32<<<blocks, kThreads>>>(x.fp32(), y.fp32(), n, 2.f, -1.f);
    sync_check(cudaGetLastError(), "normalize");
    sync_check(cudaDeviceSynchronize(), "normalize sync");
    return y;
}

Tensor launch_unnormalize_01_gpu(Tensor& x) {
    int n = x.size();
    Tensor y = Tensor::like(x);
    int blocks = (n + kThreads - 1) / kThreads;
    scale_bias_fp32<<<blocks, kThreads>>>(x.fp32(), y.fp32(), n, 0.5f, 0.5f);
    sync_check(cudaGetLastError(), "unnormalize");
    sync_check(cudaDeviceSynchronize(), "unnormalize sync");
    return y;
}

Tensor launch_cosine_similarity_mixed_gpu(Tensor& a, Tensor& b) {
    if (a.size() != b.size() || a.size() == 0) {
        throw std::runtime_error("cosine_similarity: tensors must be same non-zero length");
    }
    const int n = a.size();
    Tensor sums = Tensor::from_shape({3});
    float* sp = sums.fp32();
    std::memset(sp, 0, 3 * sizeof(float));

    const int threads = kThreads;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, 65535);
    blocks = std::max(blocks, 1);

    cosine_fp16_load_atomic_kernel<<<blocks, threads>>>(a.fp32(), b.fp32(), sp, sp + 1, sp + 2, n);
    sync_check(cudaGetLastError(), "cosine_similarity");
    sync_check(cudaDeviceSynchronize(), "cosine_similarity sync");

    const float dot = sp[0];
    const float na = sp[1];
    const float nb = sp[2];
    const float denom = std::sqrt(na) * std::sqrt(nb) + 1e-8f;
    const float cos_fp32 = dot / denom;
    const __half hcos = __float2half_rn(cos_fp32);

    Tensor out = Tensor::from_shape({1});
    out.fp32()[0] = __half2float(hcos);
    return out;
}
