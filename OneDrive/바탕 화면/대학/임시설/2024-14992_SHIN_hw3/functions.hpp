#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <cmath>
#include <cstring>
#include <functional>
#include <cassert>
#include <omp.h>

#include "tensor.h"

// You are allowed to use arm_neon.h
// and other STL libraries if necessary
#include <arm_neon.h>


namespace func {

namespace detail {
// OpenMP team size (Jetson Orin Nano class hardware); declared here so lambdas above the main detail block can use it.
constexpr int kOmpCores = 6;


// Defined later with other NEON helpers (gelu lambda appears above that definition).
float32x4_t gelu_approx_vec_f32x4(float32x4_t x);
float32x4_t sigmoid_approx_vec_f32x4(float32x4_t x);
float32x4_t vmulq_recip_f32(float32x4_t a, float32x4_t b);
} // namespace detail

/********************************
 * BASIC ELEMENT-WISE FUNCTIONS *
 ********************************/

inline std::function<Tensor(Tensor&)> identity = [](Tensor &x) {
	return x;
};

inline std::function<Tensor(Tensor&)> relu = [](Tensor &x) {
	constexpr int kOmpMinElements = 65536;
	int N = x.size();
	Tensor out = Tensor::copy(x);
	float *dst = &out[0];
	const float32x4_t vzero = vdupq_n_f32(0.f);
	if (N >= kOmpMinElements) {
		const int nchunks = (N + 3) / 4;
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
		for (int t = 0; t < nchunks; t++) {
			const int i = t * 4;
			const int rem = N - i;
			if (rem >= 4) {
				float32x4_t v = vld1q_f32(dst + i);
				v = vmaxq_f32(v, vzero);
				vst1q_f32(dst + i, v);
			} else {
				for (int j = 0; j < rem; j++) {
					if (dst[i + j] < 0.f)
						dst[i + j] = 0.f;
				}
			}
		}
	} else {
		int i = 0;
		for (; i + 4 <= N; i += 4) {
			float32x4_t v = vld1q_f32(dst + i);
			v = vmaxq_f32(v, vzero);
			vst1q_f32(dst + i, v);
		}
		for (; i < N; i++) {
			if (dst[i] < 0.f)
				dst[i] = 0.f;
		}
	}
	return out;
};

// This implementation is approximate (vectorized Padé tanh + same GELU form as reference)
// If you need more accurate results, you can implement it with a more accurate method
inline std::function<Tensor(Tensor&)> gelu = [](Tensor &x) {
	int N = x.size();
	Tensor out = Tensor::like(x);
	const float *src = &x[0];
	float *dst = &out[0];
	int i = 0;
	for (; i + 4 <= N; i += 4) {
		float32x4_t vx = vld1q_f32(src + i);
		vst1q_f32(dst + i, detail::gelu_approx_vec_f32x4(vx));
	}
	for (; i < N; i++) {
		float val = src[i];
		dst[i] = 0.5f * val * (1.f + std::tanh(0.7978845608f * (val + 0.044715f * val * val * val)));
	}
	return out;
};

// σ(x) ≈ (1 + tanh(x/2)) / 2 using same Padé tanh as GELU (fast, approximate)
inline std::function<Tensor(Tensor&)> sigmoid = [](Tensor &x) {
	int N = x.size();
	Tensor out = Tensor::like(x);
	const float *src = &x[0];
	float *dst = &out[0];
	int i = 0;
	for (; i + 4 <= N; i += 4) {
		float32x4_t vx = vld1q_f32(src + i);
		vst1q_f32(dst + i, detail::sigmoid_approx_vec_f32x4(vx));
	}
	for (; i < N; i++) {
		float val = src[i];
		dst[i] = 1.f / (1.f + std::exp(-val));
	}
	return out;
};


/****************************
 * BASIC BINOMIAL FUNCTIONS *
 ****************************/

// elementwise addition with broadcasting support
// Supports: 1) same shape, 2) (B,1,1,1) + (B,C,H,W), 3) (B,C) + (B,C,H,W)
inline std::function<Tensor(Tensor&, Tensor&)> add \
		= [](Tensor &a, Tensor &b) {
	constexpr int kOmpMinElements = 65536;
	if (a.shape() == b.shape()) {
		int N = a.size();
		Tensor out = Tensor::like(a);
		const float *pa = &a[0];
		const float *pb = &b[0];
		float *po = &out[0];
		if (N >= kOmpMinElements) {
			const int nchunks = (N + 3) / 4;
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
			for (int t = 0; t < nchunks; t++) {
				const int i = t * 4;
				const int rem = N - i;
				if (rem >= 4) {
					float32x4_t va = vld1q_f32(pa + i);
					float32x4_t vb = vld1q_f32(pb + i);
					vst1q_f32(po + i, vaddq_f32(va, vb));
				} else {
					for (int j = 0; j < rem; j++)
						po[i + j] = pa[i + j] + pb[i + j];
				}
			}
		} else {
			int i = 0;
			for (; i + 4 <= N; i += 4) {
				float32x4_t va = vld1q_f32(pa + i);
				float32x4_t vb = vld1q_f32(pb + i);
				vst1q_f32(po + i, vaddq_f32(va, vb));
			}
			for (; i < N; i++)
				po[i] = pa[i] + pb[i];
		}
		return out;
	} else if (a.dim() == 2 && b.dim() == 4) {
		assert(a.shape_dim(0) == b.shape_dim(0) && a.shape_dim(1) == b.shape_dim(1));

		int B = b.shape_dim(0);
		int C = b.shape_dim(1);
		int H = b.shape_dim(2);
		int W = b.shape_dim(3);
		const int plane = H * W;
		const int total = B * C * plane;
		Tensor out = Tensor::like(b);
		const float *pa = &a[0];
		const float *pb = &b[0];
		float *po = &out[0];

		if (total >= kOmpMinElements) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
			for (int batch = 0; batch < B; batch++) {
				for (int c = 0; c < C; c++) {
					const float32x4_t vadd = vdupq_n_f32(pa[batch * C + c]);
					const int base = (batch * C + c) * plane;
					for (int h = 0; h < H; h++) {
						const int row = base + h * W;
						int w = 0;
						for (; w + 4 <= W; w += 4) {
							float32x4_t vb = vld1q_f32(pb + row + w);
							vst1q_f32(po + row + w, vaddq_f32(vb, vadd));
						}
						for (; w < W; w++)
							po[row + w] = pb[row + w] + pa[batch * C + c];
					}
				}
			}
		} else {
			for (int batch = 0; batch < B; batch++) {
				for (int c = 0; c < C; c++) {
					const float32x4_t vadd = vdupq_n_f32(pa[batch * C + c]);
					const int base = (batch * C + c) * plane;
					for (int h = 0; h < H; h++) {
						const int row = base + h * W;
						int w = 0;
						for (; w + 4 <= W; w += 4) {
							float32x4_t vb = vld1q_f32(pb + row + w);
							vst1q_f32(po + row + w, vaddq_f32(vb, vadd));
						}
						for (; w < W; w++)
							po[row + w] = pb[row + w] + pa[batch * C + c];
					}
				}
			}
		}
		return out;
	} else {
		Tensor *smaller, *larger;
		if (a.size() < b.size()) {
			smaller = &a;
			larger = &b;
		} else {
			smaller = &b;
			larger = &a;
		}

		assert(smaller->dim() == 4 && larger->dim() == 4);
		assert(smaller->shape()[0] == larger->shape()[0]);
		assert(smaller->shape()[1] == 1 && smaller->shape()[2] == 1 && smaller->shape()[3] == 1);

		int B = larger->shape()[0];
		int C = larger->shape()[1];
		int H = larger->shape()[2];
		int W = larger->shape()[3];
		const int plane = C * H * W;
		const int total = B * plane;
		Tensor out = Tensor::like(*larger);
		const float *pl = &(*larger)[0];
		float *po = &out[0];

		if (total >= kOmpMinElements) {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
			for (int batch = 0; batch < B; batch++) {
				const float32x4_t vs = vdupq_n_f32((*smaller)[batch]);
				const int base = batch * plane;
				int i = 0;
				for (; i + 4 <= plane; i += 4) {
					float32x4_t vl = vld1q_f32(pl + base + i);
					vst1q_f32(po + base + i, vaddq_f32(vl, vs));
				}
				for (; i < plane; i++)
					po[base + i] = pl[base + i] + (*smaller)[batch];
			}
		} else {
			for (int batch = 0; batch < B; batch++) {
				const float32x4_t vs = vdupq_n_f32((*smaller)[batch]);
				const int base = batch * plane;
				int i = 0;
				for (; i + 4 <= plane; i += 4) {
					float32x4_t vl = vld1q_f32(pl + base + i);
					vst1q_f32(po + base + i, vaddq_f32(vl, vs));
				}
				for (; i < plane; i++)
					po[base + i] = pl[base + i] + (*smaller)[batch];
			}
		}
		return out;
	}
};

// elementwise subtraction with broadcasting support
// Supports: 1) same shape, 2) (B,C,H,W) - (B,1,1,1)
inline std::function<Tensor(Tensor&, Tensor&)> subtract \
		= [](Tensor &a, Tensor &b) {
	constexpr int kOmpMinElements = 65536;
	if (a.size() == b.size()) {
		int N = a.size();
		Tensor out = Tensor::like(a);
		const float *pa = &a[0];
		const float *pb = &b[0];
		float *po = &out[0];
		if (N >= kOmpMinElements) {
			const int nchunks = (N + 3) / 4;
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
			for (int t = 0; t < nchunks; t++) {
				const int i = t * 4;
				const int rem = N - i;
				if (rem >= 4) {
					float32x4_t va = vld1q_f32(pa + i);
					float32x4_t vb = vld1q_f32(pb + i);
					vst1q_f32(po + i, vsubq_f32(va, vb));
				} else {
					for (int j = 0; j < rem; j++)
						po[i + j] = pa[i + j] - pb[i + j];
				}
			}
		} else {
			int i = 0;
			for (; i + 4 <= N; i += 4) {
				float32x4_t va = vld1q_f32(pa + i);
				float32x4_t vb = vld1q_f32(pb + i);
				vst1q_f32(po + i, vsubq_f32(va, vb));
			}
			for (; i < N; i++)
				po[i] = pa[i] - pb[i];
		}
		return out;
	} else {
		assert(a.dim() == 4 && b.dim() == 4);
		assert(a.shape()[0] == b.shape()[0]);
		assert(b.shape()[1] == 1 && b.shape()[2] == 1 && b.shape()[3] == 1);

		int B = a.shape()[0];
		int C = a.shape()[1];
		int H = a.shape()[2];
		int W = a.shape()[3];
		const int plane = H * W;
		const int total = B * C * plane;
		Tensor out = Tensor::like(a);
		const float *pa = &a[0];
		float *po = &out[0];

		if (total >= kOmpMinElements) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
			for (int batch = 0; batch < B; batch++) {
				for (int c = 0; c < C; c++) {
					const float32x4_t vsub = vdupq_n_f32(b[batch]);
					const int base = (batch * C + c) * plane;
					for (int h = 0; h < H; h++) {
						const int row = base + h * W;
						int w = 0;
						for (; w + 4 <= W; w += 4) {
							float32x4_t va = vld1q_f32(pa + row + w);
							vst1q_f32(po + row + w, vsubq_f32(va, vsub));
						}
						for (; w < W; w++)
							po[row + w] = pa[row + w] - b[batch];
					}
				}
			}
		} else {
			for (int batch = 0; batch < B; batch++) {
				const float32x4_t vsub = vdupq_n_f32(b[batch]);
				for (int c = 0; c < C; c++) {
					const int base = (batch * C + c) * plane;
					for (int h = 0; h < H; h++) {
						const int row = base + h * W;
						int w = 0;
						for (; w + 4 <= W; w += 4) {
							float32x4_t va = vld1q_f32(pa + row + w);
							vst1q_f32(po + row + w, vsubq_f32(va, vsub));
						}
						for (; w < W; w++)
							po[row + w] = pa[row + w] - b[batch];
					}
				}
			}
		}
		return out;
	}
};

// elementwise multiplication with broadcasting support
// Supports: 1) same shape, 2) (B,1,1,1) * (B,C,H,W), 3) (B,1,H,W) * (B,C,H,W)
inline std::function<Tensor(Tensor&, Tensor&)> multiply \
		= [](Tensor &a, Tensor &b) {
	constexpr int kOmpMinElements = 65536;
	if (a.size() == b.size()) {
		int N = a.size();
		Tensor out = Tensor::like(a);
		const float *pa = &a[0];
		const float *pb = &b[0];
		float *po = &out[0];
		if (N >= kOmpMinElements) {
			const int nchunks = (N + 3) / 4;
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
			for (int t = 0; t < nchunks; t++) {
				const int i = t * 4;
				const int rem = N - i;
				if (rem >= 4) {
					float32x4_t va = vld1q_f32(pa + i);
					float32x4_t vb = vld1q_f32(pb + i);
					vst1q_f32(po + i, vmulq_f32(va, vb));
				} else {
					for (int j = 0; j < rem; j++)
						po[i + j] = pa[i + j] * pb[i + j];
				}
			}
		} else {
			int i = 0;
			for (; i + 4 <= N; i += 4) {
				float32x4_t va = vld1q_f32(pa + i);
				float32x4_t vb = vld1q_f32(pb + i);
				vst1q_f32(po + i, vmulq_f32(va, vb));
			}
			for (; i < N; i++)
				po[i] = pa[i] * pb[i];
		}
		return out;
	} else if (a.dim() == 4 && b.dim() == 4 && a.shape_dim(1) == 1 &&
		   a.shape_dim(0) == b.shape_dim(0) &&
		   a.shape_dim(2) == b.shape_dim(2) && a.shape_dim(3) == b.shape_dim(3)) {
		int B = b.shape_dim(0);
		int C = b.shape_dim(1);
		int H = b.shape_dim(2);
		int W = b.shape_dim(3);
		const int plane = H * W;
		const int total = B * C * plane;
		Tensor out = Tensor::like(b);
		const float *pa = &a[0];
		const float *pb = &b[0];
		float *po = &out[0];

		if (total >= kOmpMinElements) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
			for (int batch = 0; batch < B; batch++) {
				for (int c = 0; c < C; c++) {
					const int a_base = batch * plane;
					const int b_base = (batch * C + c) * plane;
					for (int h = 0; h < H; h++) {
						const int a_row = a_base + h * W;
						const int b_row = b_base + h * W;
						int w = 0;
						for (; w + 4 <= W; w += 4) {
							float32x4_t va = vld1q_f32(pa + a_row + w);
							float32x4_t vb = vld1q_f32(pb + b_row + w);
							vst1q_f32(po + b_row + w, vmulq_f32(vb, va));
						}
						for (; w < W; w++)
							po[b_row + w] = pb[b_row + w] * pa[a_row + w];
					}
				}
			}
		} else {
			for (int batch = 0; batch < B; batch++) {
				const int a_base = batch * plane;
				for (int c = 0; c < C; c++) {
					const int b_base = (batch * C + c) * plane;
					for (int h = 0; h < H; h++) {
						const int a_row = a_base + h * W;
						const int b_row = b_base + h * W;
						int w = 0;
						for (; w + 4 <= W; w += 4) {
							float32x4_t va = vld1q_f32(pa + a_row + w);
							float32x4_t vb = vld1q_f32(pb + b_row + w);
							vst1q_f32(po + b_row + w, vmulq_f32(vb, va));
						}
						for (; w < W; w++)
							po[b_row + w] = pb[b_row + w] * pa[a_row + w];
					}
				}
			}
		}
		return out;
	} else {
		Tensor *smaller, *larger;
		if (a.size() < b.size()) {
			smaller = &a;
			larger = &b;
		} else {
			smaller = &b;
			larger = &a;
		}

		assert(smaller->dim() == 4 && larger->dim() == 4);
		assert(smaller->shape()[0] == larger->shape()[0]);
		assert(smaller->shape()[1] == 1 && smaller->shape()[2] == 1 && smaller->shape()[3] == 1);

		int B = larger->shape()[0];
		int C = larger->shape()[1];
		int H = larger->shape()[2];
		int W = larger->shape()[3];
		const int plane = C * H * W;
		const int total = B * plane;
		Tensor out = Tensor::like(*larger);
		const float *pl = &(*larger)[0];
		float *po = &out[0];

		if (total >= kOmpMinElements) {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
			for (int batch = 0; batch < B; batch++) {
				const float32x4_t vs = vdupq_n_f32((*smaller)[batch]);
				const int base = batch * plane;
				int i = 0;
				for (; i + 4 <= plane; i += 4) {
					float32x4_t vl = vld1q_f32(pl + base + i);
					vst1q_f32(po + base + i, vmulq_f32(vl, vs));
				}
				for (; i < plane; i++)
					po[base + i] = pl[base + i] * (*smaller)[batch];
			}
		} else {
			for (int batch = 0; batch < B; batch++) {
				const float32x4_t vs = vdupq_n_f32((*smaller)[batch]);
				const int base = batch * plane;
				int i = 0;
				for (; i + 4 <= plane; i += 4) {
					float32x4_t vl = vld1q_f32(pl + base + i);
					vst1q_f32(po + base + i, vmulq_f32(vl, vs));
				}
				for (; i < plane; i++)
					po[base + i] = pl[base + i] * (*smaller)[batch];
			}
		}
		return out;
	}
};

// elementwise division with broadcasting support
// Supports: 1) same shape, 2) (B,C,H,W) / (B,1,1,1)
inline std::function<Tensor(Tensor&, Tensor&)> divide \
		= [](Tensor &a, Tensor &b) {
	constexpr int kOmpMinElements = 65536;
	if (a.size() == b.size()) {
		int N = a.size();
		Tensor out = Tensor::like(a);
		const float *pa = &a[0];
		const float *pb = &b[0];
		float *po = &out[0];
		if (N >= kOmpMinElements) {
			const int nchunks = (N + 3) / 4;
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
			for (int t = 0; t < nchunks; t++) {
				const int i = t * 4;
				const int rem = N - i;
				if (rem >= 4) {
					float32x4_t va = vld1q_f32(pa + i);
					float32x4_t vb = vld1q_f32(pb + i);
					vst1q_f32(po + i, detail::vmulq_recip_f32(va, vb));
				} else {
					for (int j = 0; j < rem; j++)
						po[i + j] = pa[i + j] / pb[i + j];
				}
			}
		} else {
			int i = 0;
			for (; i + 4 <= N; i += 4) {
				float32x4_t va = vld1q_f32(pa + i);
				float32x4_t vb = vld1q_f32(pb + i);
				vst1q_f32(po + i, detail::vmulq_recip_f32(va, vb));
			}
			for (; i < N; i++)
				po[i] = pa[i] / pb[i];
		}
		return out;
	} else {
		assert(a.dim() == 4 && b.dim() == 4);
		assert(a.shape()[0] == b.shape()[0]);
		assert(b.shape()[1] == 1 && b.shape()[2] == 1 && b.shape()[3] == 1);

		int B = a.shape()[0];
		int C = a.shape()[1];
		int H = a.shape()[2];
		int W = a.shape()[3];
		const int plane = H * W;
		const int total = B * C * plane;
		Tensor out = Tensor::like(a);
		const float *pa = &a[0];
		float *po = &out[0];

		if (total >= kOmpMinElements) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
			for (int batch = 0; batch < B; batch++) {
				for (int c = 0; c < C; c++) {
					const float inv_s = 1.f / b[batch];
					const float32x4_t vmul = vdupq_n_f32(inv_s);
					const int base = (batch * C + c) * plane;
					for (int h = 0; h < H; h++) {
						const int row = base + h * W;
						int w = 0;
						for (; w + 4 <= W; w += 4) {
							float32x4_t va = vld1q_f32(pa + row + w);
							vst1q_f32(po + row + w, vmulq_f32(va, vmul));
						}
						for (; w < W; w++)
							po[row + w] = pa[row + w] * inv_s;
					}
				}
			}
		} else {
			for (int batch = 0; batch < B; batch++) {
				const float inv_s = 1.f / b[batch];
				const float32x4_t vmul = vdupq_n_f32(inv_s);
				for (int c = 0; c < C; c++) {
					const int base = (batch * C + c) * plane;
					for (int h = 0; h < H; h++) {
						const int row = base + h * W;
						int w = 0;
						for (; w + 4 <= W; w += 4) {
							float32x4_t va = vld1q_f32(pa + row + w);
							vst1q_f32(po + row + w, vmulq_f32(va, vmul));
						}
						for (; w < W; w++)
							po[row + w] = pa[row + w] * inv_s;
					}
				}
			}
		}
		return out;
	}
};

// concatenate
inline std::function<Tensor(Tensor&, Tensor&, int)> concatenate 
        = [](Tensor &a, Tensor &b, int dim) {
    assert(a.dim() == b.dim());

    std::vector<int> out_shape = a.shape();
    out_shape[dim] += b.shape_dim(dim);
    
    for(int i = 0; i < a.dim(); i++) {
        if (i != dim) assert(a.shape_dim(i) == b.shape_dim(i));
    }

    Tensor out = Tensor::from_shape(out_shape);

    int n_outer = 1;
    int n_inner = 1;
	for(int i=0; i<a.dim(); i++) {
		if(i < dim) 	 n_outer *= a.shape_dim(i);
		else if(i > dim) n_inner *= a.shape_dim(i);
	}

    int chunk_a = a.shape_dim(dim) * n_inner;
    int chunk_b = b.shape_dim(dim) * n_inner;

    float* ptr_a = &a[0];
    float* ptr_b = &b[0];
    float* ptr_out = &out[0];

    for (int i = 0; i < n_outer; i++) {
        std::memcpy(ptr_out, ptr_a, chunk_a * sizeof(float));
        ptr_out += chunk_a;
        ptr_a   += chunk_a;

        std::memcpy(ptr_out, ptr_b, chunk_b * sizeof(float));
        ptr_out += chunk_b;
        ptr_b   += chunk_b;
    }

    return out;
};

namespace detail {

static inline float horizontal_sum_f32x4(float32x4_t v) {
#if defined(__aarch64__)
	return vaddvq_f32(v);
#else
	float tmp[4];
	vst1q_f32(tmp, v);
	return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}

// Approximate a/b using reciprocal estimate + two Newton steps (NEON has no divide).
inline float32x4_t vmulq_recip_f32(float32x4_t a, float32x4_t b) {
	float32x4_t r = vrecpeq_f32(b);
	r = vmulq_f32(vrecpsq_f32(b, r), r);
	r = vmulq_f32(vrecpsq_f32(b, r), r);
	return vmulq_f32(a, r);
}

// Padé-style tanh on a bounded range; output clamped to [-1, 1].
static inline float32x4_t tanh_pade_f32x4(float32x4_t x) {
	const float32x4_t klim = vdupq_n_f32(5.5f);
	float32x4_t xc = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-5.5f)), klim);
	float32x4_t x2 = vmulq_f32(xc, xc);
	float32x4_t num = vaddq_f32(vdupq_n_f32(27.f), x2);
	float32x4_t den = vaddq_f32(vdupq_n_f32(27.f), vmulq_f32(vdupq_n_f32(9.f), x2));
	float32x4_t t = vmulq_f32(xc, vmulq_recip_f32(num, den));
	return vminq_f32(vmaxq_f32(t, vdupq_n_f32(-1.f)), vdupq_n_f32(1.f));
}

// GELU: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
inline float32x4_t gelu_approx_vec_f32x4(float32x4_t x) {
	const float32x4_t k_sqrt_2_pi = vdupq_n_f32(0.7978845608f);
	const float32x4_t k_c0 = vdupq_n_f32(0.044715f);
	const float32x4_t half = vdupq_n_f32(0.5f);
	const float32x4_t one = vdupq_n_f32(1.f);
	float32x4_t x2 = vmulq_f32(x, x);
	float32x4_t x3 = vmulq_f32(x2, x);
	float32x4_t inner = vmulq_f32(k_sqrt_2_pi, vaddq_f32(x, vmulq_f32(k_c0, x3)));
	float32x4_t t = tanh_pade_f32x4(inner);
	return vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, t)));
}

// Sigmoid: σ(x) = (1 + tanh(x/2)) / 2 (exact identity; tanh is Padé-approximated)
inline float32x4_t sigmoid_approx_vec_f32x4(float32x4_t x) {
	const float32x4_t half = vdupq_n_f32(0.5f);
	const float32x4_t one = vdupq_n_f32(1.f);
	float32x4_t t = tanh_pade_f32x4(vmulq_f32(x, half));
	return vmulq_f32(half, vaddq_f32(one, t));
}

static inline void linear_gemv_neon(const float *x, int I, const float *weight, const float *bias, float *out,
				  int O, bool has_bias) {
#pragma omp parallel for num_threads(kOmpCores) schedule(static) if (O > 16)
	for (int o = 0; o < O; o++) {
		float sum = has_bias ? bias[o] : 0.f;
		int i = 0;
		for (; i + 4 <= I; i += 4) {
			float32x4_t vx = vld1q_f32(x + i);
			float32x4_t vw = vld1q_f32(weight + o * I + i);
			float32x4_t pr = vmulq_f32(vx, vw);
			float tmp[4];
			vst1q_f32(tmp, pr);
			sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
		}
		for (; i < I; i++)
			sum += x[i] * weight[o * I + i];
		out[o] = sum;
	}
}

// Load 4 input values for output columns ow..ow+3 at fixed (ic, kh, kw). When st==1,
// columns are contiguous in memory; otherwise gather with stride `st` between columns.
static inline float32x4_t vld4_ow_stride_f32(const float *row_base, int ow, int st, int kw) {
	if (__builtin_expect(st == 1, 1)) {
		return vld1q_f32(row_base + ow + kw);
	}
	const float *p = row_base + ow * st + kw;
	return (float32x4_t){p[0], p[st], p[2 * st], p[3 * st]};
}

// One 4-wide OW block at (oc, oh, ow); ow must satisfy ow + 4 <= OW.
static inline void conv2d_g1_neon_generic_block(const float *x_padded, const float *weight,
						 const float *bias, float *out, int b, int IC, int OC,
						 int PH, int PW, int OH, int OW, int R, int S, int stride,
						 int oc, int oh, int ow) {
	const int w_icrs = R * S;
	const float *w_oc = weight + oc * IC * w_icrs;
	float32x4_t acc = vdupq_n_f32(bias[oc]);
	for (int ic = 0; ic < IC; ic++) {
		const float *x_plane_ptr = x_padded + (b * IC + ic) * PH * PW;
		const float *w_ic = w_oc + ic * w_icrs;
		for (int kh = 0; kh < R; kh++) {
			const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
			for (int kw = 0; kw < S; kw++) {
				float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
				float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
				acc = vmlaq_f32(acc, vx, vw);
			}
		}
	}
	int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
	vst1q_f32(out + out_idx, acc);
}

static inline void conv2d_g1_neon_generic_scalar_pixel(const float *x_padded, const float *weight,
						       const float *bias, float *out, int b, int IC,
						       int OC, int PH, int PW, int OH, int OW, int R,
						       int S, int stride, int oc, int oh, int ow) {
	const int w_icrs = R * S;
	const float *w_oc = weight + oc * IC * w_icrs;
	float sum = bias[oc];
	for (int ic = 0; ic < IC; ic++) {
		const float *w_ic = w_oc + ic * w_icrs;
		for (int kh = 0; kh < R; kh++) {
			for (int kw = 0; kw < S; kw++) {
				int x_idx = ((b * IC + ic) * PH + (oh * stride + kh)) * PW +
					    (ow * stride + kw);
				sum += x_padded[x_idx] * w_ic[kh * S + kw];
			}
		}
	}
	int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
	out[out_idx] = sum;
}

// groups==1, generic kernel: NEON across OW; scalar tail when OW % 4 != 0.
static inline void conv2d_g1_neon_generic_oc_range(const float *x_padded, const float *weight,
						   const float *bias, float *out, int b, int IC, int OC,
						   int PH, int PW, int OH, int OW, int R, int S, int stride,
						   int oc_begin, int oc_end) {
	const int w_icrs = R * S;
	for (int oc = oc_begin; oc < oc_end; oc++) {
		const float *w_oc = weight + oc * IC * w_icrs;
		float32x4_t vbias = vdupq_n_f32(bias[oc]);

		for (int oh = 0; oh < OH; oh++) {
			int ow = 0;
			for (; ow + 4 <= OW; ow += 4) {
				float32x4_t acc = vbias;
				for (int ic = 0; ic < IC; ic++) {
					const float *x_plane_ptr =
					    x_padded + (b * IC + ic) * PH * PW;
					const float *w_ic = w_oc + ic * w_icrs;
					for (int kh = 0; kh < R; kh++) {
						const float *row_base =
						    x_plane_ptr + (oh * stride + kh) * PW;
						for (int kw = 0; kw < S; kw++) {
							float32x4_t vx =
							    vld4_ow_stride_f32(row_base, ow, stride, kw);
							float32x4_t vw =
							    vdupq_n_f32(w_ic[kh * S + kw]);
							acc = vmlaq_f32(acc, vx, vw);
						}
					}
				}
				int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
				vst1q_f32(out + out_idx, acc);
			}
			for (; ow < OW; ow++) {
				conv2d_g1_neon_generic_scalar_pixel(x_padded, weight, bias, out, b, IC, OC,
								    PH, PW, OH, OW, R, S, stride, oc, oh,
								    ow);
			}
		}
	}
}

// 7x7: one 4-wide OW block — mirrors k7_oc_range tiling; stride==1 uses direct vld1q (no gather).
static inline void conv2d_g1_neon_k7_block(const float *x_padded, const float *weight,
					   const float *bias, float *out, int b, int IC, int OC,
					   int PH, int PW, int OH, int OW, int stride, int oc, int oh,
					   int ow) {
	constexpr int R = 7, S = 7;
	const int w_icrs = R * S;
	const float *w_oc = weight + oc * IC * w_icrs;
	float32x4_t acc = vdupq_n_f32(bias[oc]);
	if (__builtin_expect(stride == 1, 1)) {
		for (int ic = 0; ic < IC; ic++) {
			const float *x_plane_ptr = x_padded + (b * IC + ic) * PH * PW;
			const float *w_ic = w_oc + ic * w_icrs;
			for (int kh = 0; kh < 4; kh++) {
				const float *row_base = x_plane_ptr + (oh + kh) * PW;
				for (int kw = 0; kw < 4; kw++) {
					float32x4_t vx = vld1q_f32(row_base + ow + kw);
					float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
					acc = vmlaq_f32(acc, vx, vw);
				}
			}
			for (int kh = 0; kh < 4; kh++) {
				const float *row_base = x_plane_ptr + (oh + kh) * PW;
				for (int kw = 4; kw < 7; kw++) {
					float32x4_t vx = vld1q_f32(row_base + ow + kw);
					float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
					acc = vmlaq_f32(acc, vx, vw);
				}
			}
			for (int kh = 4; kh < 7; kh++) {
				const float *row_base = x_plane_ptr + (oh + kh) * PW;
				for (int kw = 0; kw < 4; kw++) {
					float32x4_t vx = vld1q_f32(row_base + ow + kw);
					float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
					acc = vmlaq_f32(acc, vx, vw);
				}
			}
			for (int kh = 4; kh < 7; kh++) {
				const float *row_base = x_plane_ptr + (oh + kh) * PW;
				for (int kw = 4; kw < 7; kw++) {
					float32x4_t vx = vld1q_f32(row_base + ow + kw);
					float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
					acc = vmlaq_f32(acc, vx, vw);
				}
			}
		}
	} else {
		for (int ic = 0; ic < IC; ic++) {
			const float *x_plane_ptr = x_padded + (b * IC + ic) * PH * PW;
			const float *w_ic = w_oc + ic * w_icrs;
			for (int kh = 0; kh < 4; kh++) {
				const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
				for (int kw = 0; kw < 4; kw++) {
					float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
					float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
					acc = vmlaq_f32(acc, vx, vw);
				}
			}
			for (int kh = 0; kh < 4; kh++) {
				const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
				for (int kw = 4; kw < 7; kw++) {
					float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
					float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
					acc = vmlaq_f32(acc, vx, vw);
				}
			}
			for (int kh = 4; kh < 7; kh++) {
				const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
				for (int kw = 0; kw < 4; kw++) {
					float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
					float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
					acc = vmlaq_f32(acc, vx, vw);
				}
			}
			for (int kh = 4; kh < 7; kh++) {
				const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
				for (int kw = 4; kw < 7; kw++) {
					float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
					float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
					acc = vmlaq_f32(acc, vx, vw);
				}
			}
		}
	}
	int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
	vst1q_f32(out + out_idx, acc);
}

static inline void conv2d_g1_neon_k7_scalar_pixel(const float *x_padded, const float *weight,
						  const float *bias, float *out, int b, int IC, int OC,
						  int PH, int PW, int OH, int OW, int stride, int oc,
						  int oh, int ow) {
	conv2d_g1_neon_generic_scalar_pixel(x_padded, weight, bias, out, b, IC, OC, PH, PW, OH, OW, 7,
					    7, stride, oc, oh, ow);
}

// 7x7 kernel, groups==1: tiled NEON; scalar tail for OW % 4.
static inline void conv2d_g1_neon_k7_oc_range(const float *x_padded, const float *weight,
					      const float *bias, float *out, int b, int IC, int OC,
					      int PH, int PW, int OH, int OW, int stride, int oc_begin,
					      int oc_end) {
	constexpr int R = 7, S = 7;
	const int w_icrs = R * S;
	for (int oc = oc_begin; oc < oc_end; oc++) {
		const float *w_oc = weight + oc * IC * w_icrs;
		float32x4_t vbias = vdupq_n_f32(bias[oc]);

		for (int oh = 0; oh < OH; oh++) {
			int ow = 0;
			for (; ow + 4 <= OW; ow += 4) {
				float32x4_t acc = vbias;
				for (int ic = 0; ic < IC; ic++) {
					const float *x_plane_ptr =
					    x_padded + (b * IC + ic) * PH * PW;
					const float *w_ic = w_oc + ic * w_icrs;
					for (int kh = 0; kh < 4; kh++) {
						const float *row_base =
						    x_plane_ptr + (oh * stride + kh) * PW;
						for (int kw = 0; kw < 4; kw++) {
							float32x4_t vx =
							    vld4_ow_stride_f32(row_base, ow, stride, kw);
							float32x4_t vw =
							    vdupq_n_f32(w_ic[kh * S + kw]);
							acc = vmlaq_f32(acc, vx, vw);
						}
					}
					for (int kh = 0; kh < 4; kh++) {
						const float *row_base =
						    x_plane_ptr + (oh * stride + kh) * PW;
						for (int kw = 4; kw < 7; kw++) {
							float32x4_t vx =
							    vld4_ow_stride_f32(row_base, ow, stride, kw);
							float32x4_t vw =
							    vdupq_n_f32(w_ic[kh * S + kw]);
							acc = vmlaq_f32(acc, vx, vw);
						}
					}
					for (int kh = 4; kh < 7; kh++) {
						const float *row_base =
						    x_plane_ptr + (oh * stride + kh) * PW;
						for (int kw = 0; kw < 4; kw++) {
							float32x4_t vx =
							    vld4_ow_stride_f32(row_base, ow, stride, kw);
							float32x4_t vw =
							    vdupq_n_f32(w_ic[kh * S + kw]);
							acc = vmlaq_f32(acc, vx, vw);
						}
					}
					for (int kh = 4; kh < 7; kh++) {
						const float *row_base =
						    x_plane_ptr + (oh * stride + kh) * PW;
						for (int kw = 4; kw < 7; kw++) {
							float32x4_t vx =
							    vld4_ow_stride_f32(row_base, ow, stride, kw);
							float32x4_t vw =
							    vdupq_n_f32(w_ic[kh * S + kw]);
							acc = vmlaq_f32(acc, vx, vw);
						}
					}
				}
				int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
				vst1q_f32(out + out_idx, acc);
			}
			for (; ow < OW; ow++) {
				conv2d_g1_neon_k7_scalar_pixel(x_padded, weight, bias, out, b, IC, OC, PH,
								 PW, OH, OW, stride, oc, oh, ow);
			}
		}
	}
}

// 3x3 groups==1: fully unrolled 3x3 spatial (common for pointwise/strided convs in blocks).
static inline void conv2d_g1_neon_k3_block(const float *x_padded, const float *weight,
					   const float *bias, float *out, int b, int IC, int OC,
					   int PH, int PW, int OH, int OW, int stride, int oc, int oh,
					   int ow) {
	constexpr int R = 3, S = 3;
	const int w_icrs = R * S;
	const float *w_oc = weight + oc * IC * w_icrs;
	float32x4_t acc = vdupq_n_f32(bias[oc]);
	for (int ic = 0; ic < IC; ic++) {
		const float *x_plane_ptr = x_padded + (b * IC + ic) * PH * PW;
		const float *w_ic = w_oc + ic * w_icrs;
		for (int kh = 0; kh < 3; kh++) {
			const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
			for (int kw = 0; kw < 3; kw++) {
				float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
				float32x4_t vw = vdupq_n_f32(w_ic[kh * S + kw]);
				acc = vmlaq_f32(acc, vx, vw);
			}
		}
	}
	int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
	vst1q_f32(out + out_idx, acc);
}

static inline void conv2d_g1_neon_k3_scalar_pixel(const float *x_padded, const float *weight,
						  const float *bias, float *out, int b, int IC, int OC,
						  int PH, int PW, int OH, int OW, int stride, int oc,
						  int oh, int ow) {
	conv2d_g1_neon_generic_scalar_pixel(x_padded, weight, bias, out, b, IC, OC, PH, PW, OH, OW, 3,
					    3, stride, oc, oh, ow);
}

static inline void conv2d_g1_neon_k3_oc_range(const float *x_padded, const float *weight,
					      const float *bias, float *out, int b, int IC, int OC,
					      int PH, int PW, int OH, int OW, int stride, int oc_begin,
					      int oc_end) {
	constexpr int R = 3, S = 3;
	const int w_icrs = R * S;
	for (int oc = oc_begin; oc < oc_end; oc++) {
		const float *w_oc = weight + oc * IC * w_icrs;
		float32x4_t vbias = vdupq_n_f32(bias[oc]);

		for (int oh = 0; oh < OH; oh++) {
			int ow = 0;
			for (; ow + 4 <= OW; ow += 4) {
				float32x4_t acc = vbias;
				for (int ic = 0; ic < IC; ic++) {
					const float *x_plane_ptr =
					    x_padded + (b * IC + ic) * PH * PW;
					const float *w_ic = w_oc + ic * w_icrs;
					for (int kh = 0; kh < 3; kh++) {
						const float *row_base =
						    x_plane_ptr + (oh * stride + kh) * PW;
						for (int kw = 0; kw < 3; kw++) {
							float32x4_t vx =
							    vld4_ow_stride_f32(row_base, ow, stride, kw);
							float32x4_t vw =
							    vdupq_n_f32(w_ic[kh * S + kw]);
							acc = vmlaq_f32(acc, vx, vw);
						}
					}
				}
				int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
				vst1q_f32(out + out_idx, acc);
			}
			for (; ow < OW; ow++) {
				conv2d_g1_neon_k3_scalar_pixel(x_padded, weight, bias, out, b, IC, OC, PH,
							       PW, OH, OW, stride, oc, oh, ow);
			}
		}
	}
}

// True depthwise 7x7 (GIC == 1): one input plane per output channel, weight [OC,1,7,7].
static inline void conv2d_depthwise_neon_k7_block(const float *x_padded, const float *weight,
						  const float *bias, float *out, int b, int IC, int OC,
						  int PH, int PW, int OH, int OW, int stride, int oc,
						  int oh, int ow) {
	constexpr int R = 7, S = 7;
	const int w_icrs = R * S;
	const float *w_oc = weight + oc * w_icrs;
	const float *x_plane_ptr = x_padded + (b * IC + oc) * PH * PW;
	float32x4_t acc = vdupq_n_f32(bias[oc]);
	for (int kh = 0; kh < 4; kh++) {
		const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
		for (int kw = 0; kw < 4; kw++) {
			float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
			float32x4_t vw = vdupq_n_f32(w_oc[kh * S + kw]);
			acc = vmlaq_f32(acc, vx, vw);
		}
	}
	for (int kh = 0; kh < 4; kh++) {
		const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
		for (int kw = 4; kw < 7; kw++) {
			float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
			float32x4_t vw = vdupq_n_f32(w_oc[kh * S + kw]);
			acc = vmlaq_f32(acc, vx, vw);
		}
	}
	for (int kh = 4; kh < 7; kh++) {
		const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
		for (int kw = 0; kw < 4; kw++) {
			float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
			float32x4_t vw = vdupq_n_f32(w_oc[kh * S + kw]);
			acc = vmlaq_f32(acc, vx, vw);
		}
	}
	for (int kh = 4; kh < 7; kh++) {
		const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
		for (int kw = 4; kw < 7; kw++) {
			float32x4_t vx = vld4_ow_stride_f32(row_base, ow, stride, kw);
			float32x4_t vw = vdupq_n_f32(w_oc[kh * S + kw]);
			acc = vmlaq_f32(acc, vx, vw);
		}
	}
	int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
	vst1q_f32(out + out_idx, acc);
}

static inline void conv2d_depthwise_neon_k7_scalar_pixel(const float *x_padded, const float *weight,
							 const float *bias, float *out, int b, int IC,
							 int OC, int PH, int PW, int OH, int OW, int stride,
							 int oc, int oh, int ow) {
	constexpr int R = 7, S = 7;
	const float *w_oc = weight + oc * (R * S);
	float sum = bias[oc];
	for (int kh = 0; kh < R; kh++) {
		for (int kw = 0; kw < S; kw++) {
			int x_idx = ((b * IC + oc) * PH + (oh * stride + kh)) * PW + (ow * stride + kw);
			sum += x_padded[x_idx] * w_oc[kh * S + kw];
		}
	}
	int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
	out[out_idx] = sum;
}

static inline void conv2d_depthwise_neon_k7_oc_range(const float *x_padded, const float *weight,
						     const float *bias, float *out, int b, int IC, int OC,
						     int PH, int PW, int OH, int OW, int stride,
						     int oc_begin, int oc_end) {
	constexpr int R = 7, S = 7;
	const int w_icrs = R * S;
	for (int oc = oc_begin; oc < oc_end; oc++) {
		const float *w_oc = weight + oc * w_icrs;
		float32x4_t vbias = vdupq_n_f32(bias[oc]);
		const float *x_plane_ptr = x_padded + (b * IC + oc) * PH * PW;

		for (int oh = 0; oh < OH; oh++) {
			int ow = 0;
			for (; ow + 4 <= OW; ow += 4) {
				float32x4_t acc = vbias;
				for (int kh = 0; kh < 4; kh++) {
					const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
					for (int kw = 0; kw < 4; kw++) {
						float32x4_t vx =
						    vld4_ow_stride_f32(row_base, ow, stride, kw);
						float32x4_t vw = vdupq_n_f32(w_oc[kh * S + kw]);
						acc = vmlaq_f32(acc, vx, vw);
					}
				}
				for (int kh = 0; kh < 4; kh++) {
					const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
					for (int kw = 4; kw < 7; kw++) {
						float32x4_t vx =
						    vld4_ow_stride_f32(row_base, ow, stride, kw);
						float32x4_t vw = vdupq_n_f32(w_oc[kh * S + kw]);
						acc = vmlaq_f32(acc, vx, vw);
					}
				}
				for (int kh = 4; kh < 7; kh++) {
					const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
					for (int kw = 0; kw < 4; kw++) {
						float32x4_t vx =
						    vld4_ow_stride_f32(row_base, ow, stride, kw);
						float32x4_t vw = vdupq_n_f32(w_oc[kh * S + kw]);
						acc = vmlaq_f32(acc, vx, vw);
					}
				}
				for (int kh = 4; kh < 7; kh++) {
					const float *row_base = x_plane_ptr + (oh * stride + kh) * PW;
					for (int kw = 4; kw < 7; kw++) {
						float32x4_t vx =
						    vld4_ow_stride_f32(row_base, ow, stride, kw);
						float32x4_t vw = vdupq_n_f32(w_oc[kh * S + kw]);
						acc = vmlaq_f32(acc, vx, vw);
					}
				}
				int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
				vst1q_f32(out + out_idx, acc);
			}
			for (; ow < OW; ow++) {
				conv2d_depthwise_neon_k7_scalar_pixel(x_padded, weight, bias, out, b, IC, OC,
									PH, PW, OH, OW, stride, oc, oh, ow);
			}
		}
	}
}

} // namespace detail

/******************************
 * MAIN OPERATIONAL FUNCTIONS *
 ******************************/

// Nearest-neighbor upsample: scale==2 uses NEON (zip duplicate) + 4-wide W; other scales scalar inner.
inline std::function<Tensor(Tensor&, int)> upsample = [](Tensor &x, int scale_factor) {
	assert(scale_factor > 1); // scale factor should be greater than 1
	assert(x.shape_dim(2) == x.shape_dim(3)); // check if input is square
	assert(x.dim() == 4); // check if input is 4d tensor

	int B = x.shape_dim(0);
	int C = x.shape_dim(1);
	int H = x.shape_dim(2);
	int W = x.shape_dim(3);
	int OH = H * scale_factor;
	int OW = W * scale_factor;
	Tensor out = Tensor::from_shape({B, C, OH, OW});
	const float *xp = &x[0];
	float *op = &out[0];

	if (scale_factor == 2) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
		for (int b = 0; b < B; b++) {
			for (int c = 0; c < C; c++) {
				const int plane_in = (b * C + c) * H * W;
				const int plane_out = (b * C + c) * OH * OW;
				for (int h = 0; h < H; h++) {
					int w = 0;
					for (; w + 4 <= W; w += 4) {
						float32x4_t v = vld1q_f32(xp + plane_in + h * W + w);
						float32x4_t z0 = vzip1q_f32(v, v);
						float32x4_t z1 = vzip2q_f32(v, v);
						int base = plane_out + (h * 2) * OW + w * 2;
						vst1q_f32(op + base, z0);
						vst1q_f32(op + base + 4, z1);
						vst1q_f32(op + base + OW, z0);
						vst1q_f32(op + base + OW + 4, z1);
					}
					for (; w < W; w++) {
						float v = xp[plane_in + h * W + w];
						int base = plane_out + (h * 2) * OW + w * 2;
						op[base] = v;
						op[base + 1] = v;
						op[base + OW] = v;
						op[base + OW + 1] = v;
					}
				}
			}
		}
		return out;
	}

#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
	for (int b = 0; b < B; b++) {
		for (int c = 0; c < C; c++) {
			const int plane_in = (b * C + c) * H * W;
			const int plane_out = (b * C + c) * OH * OW;
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {
					float v = xp[plane_in + h * W + w];
					int base = plane_out + (h * scale_factor) * OW + (w * scale_factor);
					for (int i = 0; i < scale_factor; i++) {
						for (int j = 0; j < scale_factor; j++)
							op[base + i * OW + j] = v;
					}
				}
			}
		}
	}

	return out;
};

// linear layer forward — GEMV with NEON (same math as blocked matmul on tiny M=1)
inline std::function<Tensor(Tensor&, Tensor&, Tensor&, bool)> linear \
		= [](Tensor &x, Tensor &weight, Tensor &bias, bool has_bias) {
	int I = weight.shape_dim(1);
	int O = weight.shape_dim(0);

	Tensor out;
	if (x.shape_dim(0) == 1) {
		out = Tensor::from_shape({1, O});
		detail::linear_gemv_neon(&x[0], I, &weight[0], has_bias ? &bias[0] : nullptr, &out[0], O, has_bias);
	} else {
		out = Tensor::from_shape({O});
		detail::linear_gemv_neon(&x[0], I, &weight[0], has_bias ? &bias[0] : nullptr, &out[0], O, has_bias);
	}

	return out;
};

// layer normalization — fused 2-pass (mean+sumsq, then normalize+affine), NEON 4-wide
inline std::function<Tensor(Tensor&, Tensor&, Tensor&, float)> layer_norm \
		= [](Tensor &x, Tensor &weight, Tensor &bias, float eps) {
	assert(x.dim() == 4);

	int B = x.shape_dim(0);
	int C = x.shape_dim(1);
	int H = x.shape_dim(2);
	int W = x.shape_dim(3);
	const int N = C * H * W;
	const float inv_n = 1.f / static_cast<float>(N);
	Tensor out = Tensor::like(x);
	const float *xptr = &x[0];
	float *optr = &out[0];
	const float *wp = &weight[0];
	const float *bp = &bias[0];

#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
	for (int b = 0; b < B; b++) {
		const int base = b * N;
		// Pass 1: sum and sum of squares (Var = E[X^2] - E[X]^2)
		float32x4_t v_sum = vdupq_n_f32(0.f);
		float32x4_t v_sum_sq = vdupq_n_f32(0.f);
		int i = 0;
		for (; i + 4 <= N; i += 4) {
			float32x4_t vx = vld1q_f32(xptr + base + i);
			v_sum = vaddq_f32(v_sum, vx);
			v_sum_sq = vmlaq_f32(v_sum_sq, vx, vx);
		}
		float sum = detail::horizontal_sum_f32x4(v_sum);
		float sum_sq = detail::horizontal_sum_f32x4(v_sum_sq);
		for (; i < N; i++) {
			float v = xptr[base + i];
			sum += v;
			sum_sq += v * v;
		}
		const float mean = sum * inv_n;
		const float mean_sq = mean * mean;
		const float ex2 = sum_sq * inv_n;
		const float var = ex2 - mean_sq;
		const float inv_std = 1.f / std::sqrt(var + eps);

		const float32x4_t v_mean = vdupq_n_f32(mean);
		const float32x4_t v_invstd = vdupq_n_f32(inv_std);

		for (int c = 0; c < C; c++) {
			const float32x4_t v_w = vdupq_n_f32(wp[c]);
			const float32x4_t v_b = vdupq_n_f32(bp[c]);
			for (int h = 0; h < H; h++) {
				const int row = base + ((c * H + h) * W);
				int w = 0;
				for (; w + 4 <= W; w += 4) {
					float32x4_t vx = vld1q_f32(xptr + row + w);
					float32x4_t centered = vsubq_f32(vx, v_mean);
					float32x4_t norm = vmulq_f32(centered, v_invstd);
					float32x4_t y = vmlaq_f32(v_b, norm, v_w);
					vst1q_f32(optr + row + w, y);
				}
				for (; w < W; w++) {
					float xi = xptr[row + w];
					optr[row + w] = (xi - mean) * inv_std * wp[c] + bp[c];
				}
			}
		}
	}

	return out;
};

inline std::function<Tensor(Tensor&, int)> conv2d_pad \
		= [](Tensor &x, int padding) {
	assert(padding != 0); // if padding is zero, then we don't need to pad

	int B = x.shape_dim(0);
	int C = x.shape_dim(1);
	int H = x.shape_dim(2);
	int W = x.shape_dim(3);
	// assumes square padding
	int OH = H + 2 * padding;
	int OW = W + 2 * padding;
	Tensor out({B, C, OH, OW}, 0.f);
	const float *xp = &x[0];
	float *op = &out[0];

#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
	for (int b = 0; b < B; b++) {
		for (int c = 0; c < C; c++) {
			const int in_plane = (b * C + c) * H * W;
			const int out_plane = (b * C + c) * OH * OW;
			for (int h = 0; h < H; h++) {
				const int out_row = out_plane + (h + padding) * OW + padding;
				int w = 0;
				for (; w + 4 <= W; w += 4) {
					float32x4_t v = vld1q_f32(xp + in_plane + h * W + w);
					vst1q_f32(op + out_row + w, v);
				}
				for (; w < W; w++) {
					op[out_row + w] = xp[in_plane + h * W + w];
				}
			}
		}
	}

	return out;
};

// Conv2D — groups==1: direct loops (same math as grouped path); depthwise + grouped: parallel as below
inline std::function<Tensor(Tensor&, Tensor&, Tensor&, int, int, int)> conv2d \
        = [](Tensor &x, Tensor &weight, Tensor &bias, int stride, int padding, int groups) {	
    assert(x.dim() == 4);
    assert(weight.dim() == 4);
    assert(stride >= 1);

    int B = x.shape_dim(0);
    int IC = x.shape_dim(1);
    int IH = x.shape_dim(2);
    int IW = x.shape_dim(3);
    int OC = weight.shape_dim(0);
    int R = weight.shape_dim(2);
    int S = weight.shape_dim(3);
    assert(R == S);

    assert(IC % groups == 0);
    assert(OC % groups == 0);
    int GIC = IC / groups;
    int GOC = OC / groups;
    assert(weight.shape_dim(1) == GIC);
    assert(bias.size() == OC);

    Tensor x_padded = (padding == 0) ? x : conv2d_pad(x, padding);
    int PH = x_padded.shape_dim(2);
    int PW = x_padded.shape_dim(3);

    int OH = (PH - R) / stride + 1;
    int OW = (PW - S) / stride + 1;
    assert(OH > 0 && OW > 0);
    Tensor out = Tensor::from_shape({B, OC, OH, OW});

    const int spatial = OH * OW;
    constexpr int kSpatialSmall = 256;
    constexpr int kSpatialLarge = 64 * 64;
    const bool spatial_small = spatial < kSpatialSmall;
    const bool spatial_large = spatial >= kSpatialLarge;

    // True depthwise 7x7 (groups == IC == OC, GIC == 1): dedicated NEON path
    const bool depthwise_k7 = (groups == IC && IC == OC && R == 7 && S == 7);
    if (depthwise_k7) {
	const float *xp = &x_padded[0];
	const float *wp = &weight[0];
	const float *bp = &bias[0];
	float *op = &out[0];

	if (spatial_small) {
	    if (B > 1) {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
		for (int b = 0; b < B; b++) {
		    detail::conv2d_depthwise_neon_k7_oc_range(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW,
							       stride, 0, OC);
		}
	    } else {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
		for (int oc = 0; oc < OC; oc++) {
		    detail::conv2d_depthwise_neon_k7_oc_range(xp, wp, bp, op, 0, IC, OC, PH, PW, OH, OW,
							      stride, oc, oc + 1);
		}
	    }
	} else if (spatial_large) {
	    const int OWB = OW / 4;
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
	    for (int oh = 0; oh < OH; oh++) {
		for (int owb = 0; owb < OWB; owb++) {
		    const int ow = owb * 4;
		    for (int b = 0; b < B; b++) {
			for (int oc = 0; oc < OC; oc++) {
			    detail::conv2d_depthwise_neon_k7_block(xp, wp, bp, op, b, IC, OC, PH, PW, OH,
								   OW, stride, oc, oh, ow);
			}
		    }
		}
	    }
	    if (OW % 4 != 0) {
		const int ow0 = (OW / 4) * 4;
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
		for (int oh = 0; oh < OH; oh++) {
		    for (int ow = ow0; ow < OW; ow++) {
			for (int b = 0; b < B; b++) {
			    for (int oc = 0; oc < OC; oc++) {
				detail::conv2d_depthwise_neon_k7_scalar_pixel(
				    xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW, stride, oc, oh, ow);
			    }
			}
		    }
		}
	    }
	} else {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
	    for (int oh = 0; oh < OH; oh++) {
		for (int b = 0; b < B; b++) {
		    for (int oc = 0; oc < OC; oc++) {
			int ow = 0;
			for (; ow + 4 <= OW; ow += 4) {
			    detail::conv2d_depthwise_neon_k7_block(xp, wp, bp, op, b, IC, OC, PH, PW, OH,
								 OW, stride, oc, oh, ow);
			}
			for (; ow < OW; ow++) {
			    detail::conv2d_depthwise_neon_k7_scalar_pixel(xp, wp, bp, op, b, IC, OC, PH,
									  PW, OH, OW, stride, oc, oh, ow);
			}
		    }
		}
	    }
	}
	return out;
    }

    if (groups == 1) {
	const float *xp = &x_padded[0];
	const float *wp = &weight[0];
	const float *bp = &bias[0];
	float *op = &out[0];
	const bool k7 = (R == 7 && S == 7);
	const bool k3 = (R == 3 && S == 3);

	if (spatial_small) {
	    if (B > 1) {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
		for (int b = 0; b < B; b++) {
		    if (k7) {
			detail::conv2d_g1_neon_k7_oc_range(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW, stride, 0,
							   OC);
		    } else if (k3) {
			detail::conv2d_g1_neon_k3_oc_range(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW, stride, 0,
							   OC);
		    } else {
			detail::conv2d_g1_neon_generic_oc_range(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW, R, S,
								stride, 0, OC);
		    }
		}
	    } else {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
		for (int oc = 0; oc < OC; oc++) {
		    if (k7) {
			detail::conv2d_g1_neon_k7_oc_range(xp, wp, bp, op, 0, IC, OC, PH, PW, OH, OW, stride, oc,
							   oc + 1);
		    } else if (k3) {
			detail::conv2d_g1_neon_k3_oc_range(xp, wp, bp, op, 0, IC, OC, PH, PW, OH, OW, stride, oc,
							   oc + 1);
		    } else {
			detail::conv2d_g1_neon_generic_oc_range(xp, wp, bp, op, 0, IC, OC, PH, PW, OH, OW, R, S,
								stride, oc, oc + 1);
		    }
		}
	    }
	} else if (spatial_large) {
	    const int OWB = OW / 4;
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
	    for (int oh = 0; oh < OH; oh++) {
		for (int owb = 0; owb < OWB; owb++) {
		    const int ow = owb * 4;
		    for (int b = 0; b < B; b++) {
			for (int oc = 0; oc < OC; oc++) {
			    if (k7) {
				detail::conv2d_g1_neon_k7_block(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW,
								 stride, oc, oh, ow);
			    } else if (k3) {
				detail::conv2d_g1_neon_k3_block(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW,
								stride, oc, oh, ow);
			    } else {
				detail::conv2d_g1_neon_generic_block(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW,
								     R, S, stride, oc, oh, ow);
			    }
			}
		    }
		}
	    }
	    if (OW % 4 != 0) {
		const int ow0 = (OW / 4) * 4;
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
		for (int oh = 0; oh < OH; oh++) {
		    for (int ow = ow0; ow < OW; ow++) {
			for (int b = 0; b < B; b++) {
			    for (int oc = 0; oc < OC; oc++) {
				if (k7) {
				    detail::conv2d_g1_neon_k7_scalar_pixel(xp, wp, bp, op, b, IC, OC, PH, PW,
									   OH, OW, stride, oc, oh, ow);
				} else if (k3) {
				    detail::conv2d_g1_neon_k3_scalar_pixel(xp, wp, bp, op, b, IC, OC, PH, PW,
									   OH, OW, stride, oc, oh, ow);
				} else {
				    detail::conv2d_g1_neon_generic_scalar_pixel(xp, wp, bp, op, b, IC, OC, PH,
									       PW, OH, OW, R, S, stride, oc, oh,
									       ow);
				}
			    }
			}
		    }
		}
	    }
	} else {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
	    for (int oh = 0; oh < OH; oh++) {
		for (int b = 0; b < B; b++) {
		    for (int oc = 0; oc < OC; oc++) {
			int ow = 0;
			for (; ow + 4 <= OW; ow += 4) {
			    if (k7) {
				detail::conv2d_g1_neon_k7_block(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW,
								stride, oc, oh, ow);
			    } else if (k3) {
				detail::conv2d_g1_neon_k3_block(xp, wp, bp, op, b, IC, OC, PH, PW, OH, OW,
								stride, oc, oh, ow);
			    } else {
				detail::conv2d_g1_neon_generic_block(xp, wp, bp, op, b, IC, OC, PH, PW, OH,
								     OW, R, S, stride, oc, oh, ow);
			    }
			}
			for (; ow < OW; ow++) {
			    if (k7) {
				detail::conv2d_g1_neon_k7_scalar_pixel(xp, wp, bp, op, b, IC, OC, PH, PW, OH,
								     OW, stride, oc, oh, ow);
			    } else if (k3) {
				detail::conv2d_g1_neon_k3_scalar_pixel(xp, wp, bp, op, b, IC, OC, PH, PW, OH,
								     OW, stride, oc, oh, ow);
			    } else {
				detail::conv2d_g1_neon_generic_scalar_pixel(xp, wp, bp, op, b, IC, OC, PH,
									     PW, OH, OW, R, S, stride, oc, oh,
									     ow);
			    }
			}
		    }
		}
	    }
	}
    } else if (groups > GOC) {
	for (int b = 0; b < B; b++) {
	    if (spatial_small) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
		for (int g = 0; g < groups; g++) {
		    for (int oc = 0; oc < GOC; oc++) {
			int global_oc = g * GOC + oc;
			for (int oh = 0; oh < OH; oh++) {
			    for (int ow = 0; ow < OW; ow++) {
				int out_idx = ((b * OC + global_oc) * OH + oh) * OW + ow;
				float sum = bias[global_oc];
				for (int ic = 0; ic < GIC; ic++) {
				    int global_ic = g * GIC + ic;
				    for (int kh = 0; kh < R; kh++) {
					for (int kw = 0; kw < S; kw++) {
					    int x_idx = ((b * IC + global_ic) * PH + (oh * stride + kh)) * PW +
							    (ow * stride + kw);
					    int w_idx = (((global_oc * GIC + ic) * R + kh) * S + kw);
					    sum += x_padded[x_idx] * weight[w_idx];
					}
				    }
				}
				out[out_idx] = sum;
			    }
			}
		    }
		}
	    } else if (spatial_large) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
		for (int oh = 0; oh < OH; oh++) {
		    for (int ow = 0; ow < OW; ow++) {
			for (int g = 0; g < groups; g++) {
			    for (int oc = 0; oc < GOC; oc++) {
				int global_oc = g * GOC + oc;
				int out_idx = ((b * OC + global_oc) * OH + oh) * OW + ow;
				float sum = bias[global_oc];
				for (int ic = 0; ic < GIC; ic++) {
				    int global_ic = g * GIC + ic;
				    for (int kh = 0; kh < R; kh++) {
					for (int kw = 0; kw < S; kw++) {
					    int x_idx = ((b * IC + global_ic) * PH + (oh * stride + kh)) * PW +
							    (ow * stride + kw);
					    int w_idx = (((global_oc * GIC + ic) * R + kh) * S + kw);
					    sum += x_padded[x_idx] * weight[w_idx];
					}
				    }
				}
				out[out_idx] = sum;
			    }
			}
		    }
		}
	    } else {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
		for (int oh = 0; oh < OH; oh++) {
		    for (int ow = 0; ow < OW; ow++) {
			for (int g = 0; g < groups; g++) {
			    for (int oc = 0; oc < GOC; oc++) {
				int global_oc = g * GOC + oc;
				int out_idx = ((b * OC + global_oc) * OH + oh) * OW + ow;
				float sum = bias[global_oc];
				for (int ic = 0; ic < GIC; ic++) {
				    int global_ic = g * GIC + ic;
				    for (int kh = 0; kh < R; kh++) {
					for (int kw = 0; kw < S; kw++) {
					    int x_idx = ((b * IC + global_ic) * PH + (oh * stride + kh)) * PW +
							    (ow * stride + kw);
					    int w_idx = (((global_oc * GIC + ic) * R + kh) * S + kw);
					    sum += x_padded[x_idx] * weight[w_idx];
					}
				    }
				}
				out[out_idx] = sum;
			    }
			}
		    }
		}
	    }
	}
    } else {
	for (int b = 0; b < B; b++) {
	    if (spatial_small) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
		for (int g = 0; g < groups; g++) {
		    for (int oc = 0; oc < GOC; oc++) {
			int global_oc = g * GOC + oc;
			for (int oh = 0; oh < OH; oh++) {
			    for (int ow = 0; ow < OW; ow++) {
				int out_idx = ((b * OC + global_oc) * OH + oh) * OW + ow;
				float sum = bias[global_oc];
				for (int ic = 0; ic < GIC; ic++) {
				    int global_ic = g * GIC + ic;
				    for (int kh = 0; kh < R; kh++) {
					for (int kw = 0; kw < S; kw++) {
					    int x_idx = ((b * IC + global_ic) * PH + (oh * stride + kh)) * PW +
							    (ow * stride + kw);
					    int w_idx = (((global_oc * GIC + ic) * R + kh) * S + kw);
					    sum += x_padded[x_idx] * weight[w_idx];
					}
				    }
				}
				out[out_idx] = sum;
			    }
			}
		    }
		}
	    } else if (spatial_large) {
#pragma omp parallel for collapse(2) num_threads(detail::kOmpCores) schedule(static)
		for (int oh = 0; oh < OH; oh++) {
		    for (int ow = 0; ow < OW; ow++) {
			for (int g = 0; g < groups; g++) {
			    for (int oc = 0; oc < GOC; oc++) {
				int global_oc = g * GOC + oc;
				int out_idx = ((b * OC + global_oc) * OH + oh) * OW + ow;
				float sum = bias[global_oc];
				for (int ic = 0; ic < GIC; ic++) {
				    int global_ic = g * GIC + ic;
				    for (int kh = 0; kh < R; kh++) {
					for (int kw = 0; kw < S; kw++) {
					    int x_idx = ((b * IC + global_ic) * PH + (oh * stride + kh)) * PW +
							    (ow * stride + kw);
					    int w_idx = (((global_oc * GIC + ic) * R + kh) * S + kw);
					    sum += x_padded[x_idx] * weight[w_idx];
					}
				    }
				}
				out[out_idx] = sum;
			    }
			}
		    }
		}
	    } else {
#pragma omp parallel for num_threads(detail::kOmpCores) schedule(static)
		for (int oh = 0; oh < OH; oh++) {
		    for (int ow = 0; ow < OW; ow++) {
			for (int g = 0; g < groups; g++) {
			    for (int oc = 0; oc < GOC; oc++) {
				int global_oc = g * GOC + oc;
				int out_idx = ((b * OC + global_oc) * OH + oh) * OW + ow;
				float sum = bias[global_oc];
				for (int ic = 0; ic < GIC; ic++) {
				    int global_ic = g * GIC + ic;
				    for (int kh = 0; kh < R; kh++) {
					for (int kw = 0; kw < S; kw++) {
					    int x_idx = ((b * IC + global_ic) * PH + (oh * stride + kh)) * PW +
							    (ow * stride + kw);
					    int w_idx = (((global_oc * GIC + ic) * R + kh) * S + kw);
					    sum += x_padded[x_idx] * weight[w_idx];
					}
				    }
				}
				out[out_idx] = sum;
			    }
			}
		    }
		}
	    }
	}
    }

    return out;

};

inline std::function<Tensor(Tensor&)> normalize_to_neg_one_to_one = [](Tensor &x) {
	/* Base Implementation */
	// equivalent to: 2 * x - 1
	int N = x.size();
	Tensor out = Tensor::like(x);

	for(int i = 0; i < N; i++) {
		out[i] = 2.0f * x[i] - 1.0f;
	}

	return out;
};

inline std::function<Tensor(Tensor&)> unnormalize_to_zero_to_one = [](Tensor &x) {
	/* Base Implementation */
	// equivalent to: (x + 1) * 0.5
	int N = x.size();
	Tensor out = Tensor::like(x);

	for(int i = 0; i < N; i++) {
		out[i] = (x[i] + 1.0f) * 0.5f;
	}

	return out;
};

// precalculated constant extraction
inline std::function<Tensor(Tensor&, Tensor&, int)> extract \
		= [](Tensor &constants, Tensor &timestamps, int dim) {
	/* Base Implementation */
	int batch_size = timestamps.shape()[0];
	assert(constants.dim() == 1); // check if constants is 1d tensor
	assert(dim == 4); // b c h w
	Tensor output({batch_size, 1, 1, 1}, 0.0f);
	
	for(int b = 0; b < batch_size; b++) {
		int t_idx = static_cast<int>(timestamps[b]);
		output[b] = constants[t_idx];
	}
	
	return output;
};

// cumulative sum along the first dimension (for 1D tensors)
inline std::function<Tensor(Tensor&)> cumsum = [](Tensor &x) {
	assert(x.dim() == 1);  // Only works for 1D tensors
	int N = x.size();
	Tensor out = Tensor::like(x);
	
	float sum = 0.0f;
	for(int i = 0; i < N; i++) {
		sum += x[i];
		out[i] = sum;
	}

	return out;
};

} // end namespace func

#endif
