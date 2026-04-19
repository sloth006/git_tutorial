#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <ctime>
#include <vector>
#include <cassert>
#include <cstdarg> // for variadic arguments

#include <module.h>
#include <schedules.hpp>
#include <tqdm.hpp>

// UNet with attention mechanism
class AttentionUNet : public Module {
private:
    int _dim;
    int _out_dim;
    std::vector<int> _dim_mults;
    int _channels;
    int _sinusoidal_pos_emb_theta=10000;
    int _convnext_block_groups=8;
    int _time_dim;
    int _num_resolutions;

    // layers
    Module *_init_conv;
    Module *_pos_emb;
    Sequential *_time_mlp;
    std::vector<ModuleList*> _downs;
    std::vector<ModuleList*> _ups;
    Module *_intermediate_block1;
    Module *_intermediate_block2;
    Module *_final_res_block;
    Module *_final_conv;

    // others
    std::vector<int> _dims;
    std::vector<std::tuple<int, int>> _in_out;

public:
    AttentionUNet(int dim, int channels, std::vector<int> dim_mults);
    ~AttentionUNet() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        _init_conv->load_checkpoint(path, "init_conv");
        _time_mlp->load_checkpoint(path, "time_mlp");
        for (int i = 0; i < _downs.size(); i++) {
            _downs[i]->load_checkpoint(path, "downs." + std::to_string(i));
        }
        for (int i = 0; i < _ups.size(); i++) {
            _ups[i]->load_checkpoint(path, "ups." + std::to_string(i));
        }
        _intermediate_block1->load_checkpoint(path, "mid_block1");
        _intermediate_block2->load_checkpoint(path, "mid_block2");
        _final_res_block->load_checkpoint(path, "final_res_block");
        _final_conv->load_checkpoint(path, "final_conv");

    }

	Tensor forward(Tensor &x, ...);

    // _channels getter
    int channels() { return _channels; }
};

class DiffusionModel : public Module {
private:
    AttentionUNet *_model;
    int _image_size;
    std::string _beta_scheduler = "linear";
    std::string _type = "ddpm";
    int _ddim_n = 10; // default 10x faster
    bool _fix_seed = false;
    int _seed = 7524;
    Tensor _betas;
    Tensor _alphas;
    int _timesteps = 1000;
    std::vector<int> _timestep_vec;
    std::vector<int> _timestep_vec_rev;

    bool _auto_normalize = true;

    int _channels;
    
    Tensor _alphas_cumprod;
    Tensor _alphas_cumprod_prev;
    Tensor _posterior_variance;
    Tensor _sqrt_recip_alphas;
    Tensor _sqrt_alphas_cumprod;
    Tensor _one_minus_alphas_cumprod;
    Tensor _sqrt_one_minius_alphas_cumprod;

    std::function<Tensor(Tensor&)> _normalize;
    std::function<Tensor(Tensor&)> _unnormalize;

    Tensor __p_sample(Tensor &x, int timestep);
    Tensor __p_sample_ddim(Tensor &x, int timestep);

    Tensor __p_sample_loop(std::vector<int> shape);

public:
    DiffusionModel(AttentionUNet *model, int image_size, std::string beta_scheduler, int timesteps,
                    std::string type="ddpm", int ddim_n=10,
                    bool fix_seed=false, int seed=7524);
    ~DiffusionModel() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        _model->load_checkpoint(path);
    }

    Tensor sample(int batch_size) {
        return __p_sample_loop(std::vector<int>{batch_size, _channels, _image_size, _image_size});
    }

    Tensor forward(Tensor &x, ...) {
        assert(false); // we do not need forward function because we will use sample function
    }

    AttentionUNet* get_model() { return _model; }

    // for testing purposes
    Tensor p_sample(Tensor &x, int timestep) {
        if (_type == "ddim") {
            return __p_sample_ddim(x, timestep);
        } else {
            return __p_sample(x, timestep);
        }
    }
};

#endif