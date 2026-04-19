#include <iostream>
#include <functions.hpp>
#include <module.h>
#include <model.h>

int main(int argc, char* argv[]) {
    
    std::cout << "===== Unit Tests for Diffusion Sampling (ESD2026Spring) =====" << std::endl;
    std::cout << "=> Warning: This test does not guarantee the correctness" << std::endl;
    std::cout << "            of your project, but provides some simple tests." << std::endl;
    std::cout << std::endl;

    std::cout << ">>>>> Simple Conv2D Test" << std::endl;
    Tensor input = Tensor::load_npy("./test/test_ckpt/conv2d_input.npy");
    Tensor expected_output = Tensor::load_npy("./test/test_ckpt/conv2d_output.npy");

    std::cout << ">>>>> >>> Input Shape: " << input.shape()[0] << " " << input.shape()[1] << " " << input.shape()[2] << " " << input.shape()[3] << std::endl;
    std::cout << ">>>>> >>> Expected Output Shape: " << expected_output.shape()[0] << " " << expected_output.shape()[1] << " " << expected_output.shape()[2] << " " << expected_output.shape()[3] << std::endl;
    // initialize layer and load weights and bias
    Conv2DLayer test_conv(3, 64, 7, 1, 3, 1, true);
    test_conv.load_checkpoint("./test/test_ckpt/", "init_conv");
    // now we can test forward
    Tensor output = test_conv.forward(input);

    // print output shape
    std::cout << ">>>>> >>> Output Shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;

    // check whether the output is correct
    if (output.allclose(expected_output, 1.0E-4, 1.0E-6)) {
        std::cout << ">>>>> >>> Simple Conv2D Test Passed" << std::endl;
    } else {
        std::cout << ">>>>> >>> Simple Conv2D Test Failed" << std::endl;
    }

    // print 10 elements of expected output
    for (int i = 0; i < 10; i++) {
        std::cout << expected_output[i] << " ";
    }
    std::cout << std::endl;
    // print 10 elements of actual output
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << ">>>>> Group Conv2D Test" << std::endl;
    input = Tensor::load_npy("./test/test_ckpt/group_conv2d_input.npy");
    expected_output = Tensor::load_npy("./test/test_ckpt/group_conv2d_output.npy");
    
    std::cout << ">>>>> >>> Input Shape: " << input.shape()[0] << " " << input.shape()[1] << " " << input.shape()[2] << " " << input.shape()[3] << std::endl;
    std::cout << ">>>>> >>> Expected Output Shape: " << expected_output.shape()[0] << " " << expected_output.shape()[1] << " " << expected_output.shape()[2] << " " << expected_output.shape()[3] << std::endl;
    // initialize layer and load weights and bias
    Conv2DLayer test_group_conv(64, 64, 7, 1, 3, 64, true);
    test_group_conv.load_checkpoint("./test/test_ckpt/", "downs.0.0.in_conv");
    // now we can test forward
    output = test_group_conv.forward(input);

    // print output shape
    std::cout << ">>>>> >>> Output Shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;

    // check whether the output is correct
    if (output.allclose(expected_output, 1.0E-4, 1.0E-6)) {
        std::cout << ">>>>> >>> Group Conv2D Test Passed" << std::endl;
    } else {
        std::cout << ">>>>> >>> Group Conv2D Test Failed" << std::endl;
    }

    // print 10 elements of expected output
    for (int i = 0; i < 10; i++) {
        std::cout << expected_output[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << ">>>>> Upsample Test" << std::endl;
    input = Tensor::load_npy("./test/test_ckpt/upsample_input.npy");
    expected_output = Tensor::load_npy("./test/test_ckpt/upsample_output.npy");

    std::cout << ">>>>> >>> Input Shape: " << input.shape()[0] << " " << input.shape()[1] << " " << input.shape()[2] << " " << input.shape()[3] << std::endl;
    std::cout << ">>>>> >>> Expected Output Shape: " << expected_output.shape()[0] << " " << expected_output.shape()[1] << " " << expected_output.shape()[2] << " " << expected_output.shape()[3] << std::endl;

    // run forward
    output = func::upsample(input, 2);
    
    // print output shape
    std::cout << ">>>>> >>> Output Shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;
    
    // check whether the output is correct
    if (output.allclose(expected_output, 1.0E-4, 1.0E-6)) {
        std::cout << ">>>>> >>> Upsample Test Passed" << std::endl;
    } else {
        std::cout << ">>>>> >>> Upsample Test Failed" << std::endl;
    }

    // print 10 elements of expected output
    for (int i = 0; i < 10; i++) {
        std::cout << expected_output[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << ">>>>> Downsample Test" << std::endl;
    DownSample downsample(64, 64); // 128 -> 64 (fixed channel)
    downsample.load_checkpoint("./test/test_ckpt/", "downs.0.2");
    input = Tensor::load_npy("./test/test_ckpt/downsample_input.npy");
    expected_output = Tensor::load_npy("./test/test_ckpt/downsample_output.npy");

    std::cout << ">>>>> >>> Input Shape: " << input.shape()[0] << " " << input.shape()[1] << " " << input.shape()[2] << " " << input.shape()[3] << std::endl;
    std::cout << ">>>>> >>> Expected Output Shape: " << expected_output.shape()[0] << " " << expected_output.shape()[1] << " " << expected_output.shape()[2] << " " << expected_output.shape()[3] << std::endl;

    // run forward
    output = downsample.forward(input);
    
    // print output shape
    std::cout << ">>>>> >>> Output Shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;
    
    // check whether the output is correct
    if (output.allclose(expected_output, 1.0E-2, 1.0E-2)) {
        std::cout << ">>>>> >>> Downsample Test Passed" << std::endl;
    } else {
        std::cout << ">>>>> >>> Downsample Test Failed" << std::endl;
    }

    // print 10 elements of expected output
    for (int i = 0; i < 10; i++) {
        std::cout << expected_output[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl << std::endl;


    std::cout << ">>>>> Layer Norm. Test" << std::endl;
    LayerNorm layer_norm(64);
    layer_norm.load_checkpoint("./test/test_ckpt/", "downs.0.0.block.0");
    input = Tensor::load_npy("./test/test_ckpt/layer_norm_input.npy");
    expected_output = Tensor::load_npy("./test/test_ckpt/layer_norm_output.npy");

    std::cout << ">>>>> >>> Input Shape: " << input.shape()[0] << " " << input.shape()[1] << " " << input.shape()[2] << " " << input.shape()[3] << std::endl;
    std::cout << ">>>>> >>> Expected Output Shape: " << expected_output.shape()[0] << " " << expected_output.shape()[1] << " " << expected_output.shape()[2] << " " << expected_output.shape()[3] << std::endl;

    // run forward
    output = layer_norm.forward(input);
    
    // print output shape
    std::cout << ">>>>> >>> Output Shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;
    
    // check whether the output is correct
    if (output.allclose(expected_output, 1.0E-3, 1.0E-4)) {
        std::cout << ">>>>> >>> Layer Norm. Test Passed" << std::endl;
    } else {
        std::cout << ">>>>> >>> Layer Norm. Test Failed" << std::endl;
    }

    // print 10 elements of expected output
    for (int i = 0; i < 10; i++) {
        std::cout << expected_output[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << ">>>>> Final Image Unnormalization Test" << std::endl;
    input = Tensor::load_npy("./test/test_ckpt/unnormalize_input.npy");
    expected_output = Tensor::load_npy("./test/test_ckpt/unnormalize_output.npy");

    std::cout << ">>>>> >>> Input Shape: " << input.shape()[0] << " " << input.shape()[1] << " " << input.shape()[2] << " " << input.shape()[3] << std::endl;
    std::cout << ">>>>> >>> Expected Output Shape: " << expected_output.shape()[0] << " " << expected_output.shape()[1] << " " << expected_output.shape()[2] << " " << expected_output.shape()[3] << std::endl;

    output = func::unnormalize_to_zero_to_one(input);

    std::cout << ">>>>> >>> Output Shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;

    // check whether the output is correct
    if (output.allclose(expected_output, 1.0E-4, 1.0E-6)) {
        std::cout << ">>>>> >>> Final Image Unnormalization Test Passed" << std::endl;
    } else {
        std::cout << ">>>>> >>> Final Image Unnormalization Test Failed" << std::endl;
    }

    // print 10 elements of expected output
    for (int i = 0; i < 10; i++) {
        std::cout << expected_output[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << ">>>>> UNet Test" << std::endl;
    AttentionUNet model(32, 3, {1, 2, 4, 8});
    DiffusionModel diffusion_model(&model, 64, "linear", 1000);
    std::cout << ">>>>> >>> Model Created" << std::endl;

    diffusion_model.load_checkpoint("./test/test_model_ckpt/");

    input = Tensor::load_npy("./test/test_ckpt/unet_input.npy");
    expected_output = Tensor::load_npy("./test/test_ckpt/unet_output.npy");

    std::cout << ">>>>> >>> Input Shape: " << input.shape()[0] << " " << input.shape()[1] << " " << input.shape()[2] << " " << input.shape()[3] << std::endl;
    std::cout << ">>>>> >>> Expected Output Shape: " << expected_output.shape()[0] << " " << expected_output.shape()[1] << " " << expected_output.shape()[2] << " " << expected_output.shape()[3] << std::endl;

    Tensor batched_timestamps({1, 1}, 999.0f);

    // run forward
    output = diffusion_model.get_model()->forward(input, batched_timestamps);

    // print output shape
    std::cout << ">>>>> >>> Output Shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;

    // check whether the output is correct
    if (output.allclose(expected_output, 1.0E-2, 1.0E-2)) {
        std::cout << ">>>>> >>> UNet Test Passed" << std::endl;
    } else {
        std::cout << ">>>>> >>> UNet Test Failed" << std::endl;
    }

    // print 10 elements of expected output
    for (int i = 0; i < 10; i++) {
        std::cout << expected_output[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << ">>>>> One Step Test" << std::endl;
    AttentionUNet ddim_model(32, 3, {1, 2, 4, 8});
    DiffusionModel ddim_diffusion_model(&model, 64, "linear", 1000, "ddim");
    std::cout << ">>>>> >>> Model Created" << std::endl;
    ddim_diffusion_model.load_checkpoint("./test/test_model_ckpt/");

    input = Tensor::load_npy("./test/test_ckpt/psample_input.npy");
    expected_output = Tensor::load_npy("./test/test_ckpt/psample_output.npy");    

    std::cout << ">>>>> >>> Input Shape: " << input.shape()[0] << " " << input.shape()[1] << " " << input.shape()[2] << " " << input.shape()[3] << std::endl;
    std::cout << ">>>>> >>> Expected Output Shape: " << expected_output.shape()[0] << " " << expected_output.shape()[1] << " " << expected_output.shape()[2] << " " << expected_output.shape()[3] << std::endl;
    
    output = ddim_diffusion_model.p_sample(input, 980);

    // print output shape
    std::cout << ">>>>> >>> Output Shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;

    // check whether the output is correct
    if (output.allclose(expected_output, 1.0E-2, 1.0E-2)) {
        std::cout << ">>>>> >>> One Step Test Passed" << std::endl;
    } else {
        std::cout << ">>>>> >>> One Step Test Failed" << std::endl;
    }

    // print 10 elements of expected output
    for (int i = 0; i < 10; i++) {
        std::cout << expected_output[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "========= Done! Please check the functionality :) =========" << std::endl;

    return 0;
}