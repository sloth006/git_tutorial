#ifndef FUNCTIONS_CUDA
#define FUNCTIONS_CUDA

#include <cuda_runtime_api.h>

#include "tensor.h"

// All CUDA entry points implemented in kernel.cu (Ampere sm_87).

Tensor launch_relu_kernel_gpu(Tensor& x);
Tensor launch_gelu_gpu(Tensor& x);
Tensor launch_sigmoid_gpu(Tensor& x);
Tensor launch_add_same_gpu(Tensor& a, Tensor& b);
Tensor launch_add_bc24_gpu(Tensor& bc, Tensor& bchw);
Tensor launch_add_bc11_gpu(Tensor& b1111, Tensor& bchw);
Tensor launch_subtract_same_gpu(Tensor& a, Tensor& b);
Tensor launch_subtract_bchw_bc11_gpu(Tensor& a, Tensor& b);
Tensor launch_multiply_same_gpu(Tensor& a, Tensor& b);
Tensor launch_multiply_bc11_gpu(Tensor& b1111, Tensor& bchw);
Tensor launch_multiply_bc1hw_gpu(Tensor& b1hw, Tensor& bchw);
Tensor launch_divide_same_gpu(Tensor& a, Tensor& b);
Tensor launch_divide_bchw_bc11_gpu(Tensor& a, Tensor& b);
void launch_concatenate_gpu(Tensor& a, Tensor& b, Tensor& out, int dim);
Tensor launch_upsample_gpu(Tensor& x, int scale_factor);
Tensor launch_conv2d_pad_gpu(Tensor& x, int padding, Tensor& out);
Tensor launch_conv2d_gpu(Tensor& x_padded, Tensor& weight, Tensor& bias, int stride, int padding, int groups);
Tensor launch_layer_norm_gpu(Tensor& x, Tensor& weight, Tensor& bias, float eps);
Tensor launch_linear_gpu(Tensor& x, Tensor& weight, Tensor& bias, bool has_bias);
Tensor launch_extract_gpu(Tensor& constants, Tensor& timestamps, Tensor& output);
Tensor launch_cumsum_gpu(Tensor& x);
Tensor launch_normalize_neg1_1_gpu(Tensor& x);
Tensor launch_unnormalize_01_gpu(Tensor& x);

// Mixed precision: FP16 loads, FP32 reductions, FP16-rounded scalar in a length-1 float tensor.
Tensor launch_cosine_similarity_mixed_gpu(Tensor& a, Tensor& b);

#endif
