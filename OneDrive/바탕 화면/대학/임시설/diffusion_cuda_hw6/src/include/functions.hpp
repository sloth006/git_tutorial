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
#include "kernel.h"

// Large temporaries (concat, padded conv) must use Tensor::from_shape (perm): putting
// them on scratch after the UNet watermark leaves little room and scratch is never
// freed on Tensor destruction, so huge scratch_from_shape here causes scratch OOM.
// Tiny sampling helpers (extract) stay on perm as in the original handout as well.

namespace func {

/********************************
 * BASIC ELEMENT-WISE FUNCTIONS *
 ********************************/

inline std::function<Tensor(Tensor&)> identity = [](Tensor &x) {
	return x;
};

inline std::function<Tensor(Tensor&)> relu = [](Tensor &x) {
    /* Base Implementation */
	// int N = x.size();
	// Tensor out = Tensor::copy(x);
	// for(int i = 0; i < N; i++) {
	// 	if(out[i] < 0) out[i] = 0;
	// }

	/* Base Implementation of Relu using CUDA*/
	Tensor out = launch_relu_kernel_gpu(x);

	return out;
};

// This implementation is approximate
// If you need more accurate results, you can implement it with a more accurate method
inline std::function<Tensor(Tensor&)> gelu = [](Tensor &x) {
	return launch_gelu_gpu(x);
};

inline std::function<Tensor(Tensor&)> sigmoid = [](Tensor &x) {
	return launch_sigmoid_gpu(x);
};


/****************************
 * BASIC BINOMIAL FUNCTIONS *
 ****************************/

// elementwise addition with broadcasting support
// Supports: 1) same shape, 2) (B,1,1,1) + (B,C,H,W), 3) (B,C) + (B,C,H,W)
inline std::function<Tensor(Tensor&, Tensor&)> add \
		= [](Tensor &a, Tensor &b) {
	if (a.shape() == b.shape()) {
		return launch_add_same_gpu(a, b);
	} else if (a.dim() == 2 && b.dim() == 4) {
		assert(a.shape_dim(0) == b.shape_dim(0) && a.shape_dim(1) == b.shape_dim(1));
		return launch_add_bc24_gpu(a, b);
	} else {
		// Broadcasting: (B,1,1,1) + (B,C,H,W)
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
		
		return launch_add_bc11_gpu(*smaller, *larger);
	}
};

// elementwise subtraction with broadcasting support
// Supports: 1) same shape, 2) (B,C,H,W) - (B,1,1,1)
inline std::function<Tensor(Tensor&, Tensor&)> subtract \
		= [](Tensor &a, Tensor &b) {
	if (a.size() == b.size()) {
		return launch_subtract_same_gpu(a, b);
	} else {
		assert(a.dim() == 4 && b.dim() == 4);
		assert(a.shape()[0] == b.shape()[0]);
		assert(b.shape()[1] == 1 && b.shape()[2] == 1 && b.shape()[3] == 1);
		return launch_subtract_bchw_bc11_gpu(a, b);
	}
};

// elementwise multiplication with broadcasting support
// Supports: 1) same shape, 2) (B,1,1,1) * (B,C,H,W), 3) (B,1,H,W) * (B,C,H,W)
inline std::function<Tensor(Tensor&, Tensor&)> multiply \
		= [](Tensor &a, Tensor &b) {
	if (a.size() == b.size()) {
		return launch_multiply_same_gpu(a, b);
	} else if (a.dim() == 4 && b.dim() == 4 && a.shape_dim(1) == 1 && 
	           a.shape_dim(0) == b.shape_dim(0) && 
	           a.shape_dim(2) == b.shape_dim(2) && a.shape_dim(3) == b.shape_dim(3)) {
		return launch_multiply_bc1hw_gpu(a, b);
	} else {
		// Broadcasting: (B,1,1,1) * (B,C,H,W)
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
		
		return launch_multiply_bc11_gpu(*smaller, *larger);
	}
};

// elementwise division with broadcasting support
// Supports: 1) same shape, 2) (B,C,H,W) / (B,1,1,1)
inline std::function<Tensor(Tensor&, Tensor&)> divide \
		= [](Tensor &a, Tensor &b) {
	if (a.size() == b.size()) {
		return launch_divide_same_gpu(a, b);
	} else {
		assert(a.dim() == 4 && b.dim() == 4);
		assert(a.shape()[0] == b.shape()[0]);
		assert(b.shape()[1] == 1 && b.shape()[2] == 1 && b.shape()[3] == 1);
		return launch_divide_bchw_bc11_gpu(a, b);
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
    launch_concatenate_gpu(a, b, out, dim);
    return out;
};


/******************************
 * MAIN OPERATIONAL FUNCTIONS *
 ******************************/

// upsample with scale factor
// [TODO] - accelerate!
inline std::function<Tensor(Tensor&, int)> upsample = [](Tensor &x, int scale_factor) {
	assert(scale_factor > 1); // scale factor should be greater than 1
	assert(x.shape_dim(2) == x.shape_dim(3)); // check if input is square
	assert(x.dim() == 4); // check if input is 4d tensor

	return launch_upsample_gpu(x, scale_factor);
};

// linear layer forward
// [TODO] - accelerate!
inline std::function<Tensor(Tensor&, Tensor&, Tensor&, bool)> linear \
		= [](Tensor &x, Tensor &weight, Tensor &bias, bool has_bias) {
	return launch_linear_gpu(x, weight, bias, has_bias);
};

// layer normalization
// [TODO] - accelerate!
inline std::function<Tensor(Tensor&, Tensor&, Tensor&, float)> layer_norm \
		= [](Tensor &x, Tensor &weight, Tensor &bias, float eps) {
    assert(x.dim() == 4);
	return launch_layer_norm_gpu(x, weight, bias, eps);
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
	Tensor out = Tensor::from_shape({B, C, OH, OW});
	launch_conv2d_pad_gpu(x, padding, out);
	return out;
};

// Conv2D
// [TODO] - accelerate!
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

    if (padding == 0) {
        return launch_conv2d_gpu(x, weight, bias, stride, padding, groups);
    }
    Tensor x_padded = Tensor::from_shape({B, IC, IH + 2 * padding, IW + 2 * padding});
    launch_conv2d_pad_gpu(x, padding, x_padded);
    return launch_conv2d_gpu(x_padded, weight, bias, stride, padding, groups);
};


/***********************************
 * MISCELLANEOUS UTILITY FUNCTIONS *
 ***********************************/

inline std::function<Tensor(Tensor&)> normalize_to_neg_one_to_one = [](Tensor &x) {
	return launch_normalize_neg1_1_gpu(x);
};

inline std::function<Tensor(Tensor&)> unnormalize_to_zero_to_one = [](Tensor &x) {
	return launch_unnormalize_01_gpu(x);
};

// precalculated constant extraction
inline std::function<Tensor(Tensor&, Tensor&, int)> extract \
		= [](Tensor &constants, Tensor &timestamps, int dim) {
	int batch_size = timestamps.shape()[0];
	assert(constants.dim() == 1);
	assert(dim == 4);
	Tensor output({batch_size, 1, 1, 1}, 0.0f);
	return launch_extract_gpu(constants, timestamps, output);
};

// cumulative sum along the first dimension (for 1D tensors)
inline std::function<Tensor(Tensor&)> cumsum = [](Tensor &x) {
	assert(x.dim() == 1);
	return launch_cumsum_gpu(x);
};

// Cosine similarity over flattened vectors (same element count). Returns length-1 tensor (float storage).
inline std::function<Tensor(Tensor&, Tensor&)> cosine_similarity = [](Tensor& a, Tensor& b) {
	return launch_cosine_similarity_mixed_gpu(a, b);
};

} // end namespace func

#endif
