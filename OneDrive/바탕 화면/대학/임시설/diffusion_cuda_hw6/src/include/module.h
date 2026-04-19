#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <cstdarg> // for variadic arguments
#include <functions.hpp>


// base class for model
class Module {
public:
    Module() {}
    virtual ~Module() {}

    virtual void load_checkpoint(std::string path, std::string prefix="") = 0;
	virtual Tensor forward(Tensor &x, ...) {
		std::cout << "WARNING: Unimplemented Tensor forward function called." << std::endl;
		return x;
	}
};

// sequential wrapper (for pytorch-like model)
class Sequential : public Module {
private:
    std::vector<Module*> _modules;
public:
    Sequential() {}
    Sequential(std::vector<Module*> modules) : _modules(modules) {}
    ~Sequential() {}

    void add_module(Module *module) {
        _modules.push_back(module);
    }

    void load_checkpoint(std::string path, std::string prefix="") {
        for (int i = 0; i < _modules.size(); i++) {
            _modules[i]->load_checkpoint(path, prefix + "." + std::to_string(i));
        }
    }

	Tensor forward(Tensor &x, ...) {
		Tensor out = x;
		for (auto module : _modules) {
			out = module->forward(out);
		}
		return out;
	}
};

// module list
class ModuleList : public Module {
private:
    std::vector<Module*> _modules;
public:
    ModuleList() {}
    ModuleList(std::vector<Module*> modules) : _modules(modules) {}
    ~ModuleList() {}

    void add_module(Module *module) {
        _modules.push_back(module);
    }

    std::vector<Module*>::iterator begin() {
        return _modules.begin();
    }

    std::vector<Module*>::iterator end() {
        return _modules.end();
    }
    
    Module* at(int idx) {
        return _modules[idx];
    }

    void load_checkpoint(std::string path, std::string prefix="") {
        for (int i = 0; i < _modules.size(); i++) {
            _modules[i]->load_checkpoint(path, prefix + "." + std::to_string(i));
        }
    }
};


// gelu
class GELU : public Module {
public:
    GELU() {}
    ~GELU() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // skip - no need to load checkpoint
    }

	Tensor forward(Tensor &x, ...);
};

// relu
class ReLU : public Module {
public:
    ReLU() {}
    ~ReLU() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // skip - no need to load checkpoint
    }

	Tensor forward(Tensor &x, ...);
};

// sigmoid
class Sigmoid : public Module {
public:
    Sigmoid() {}
    ~Sigmoid() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // skip - no need to load checkpoint
    }

	Tensor forward(Tensor &x, ...);
};

// identity
class Identity : public Module {
public:
    Identity() {}
    ~Identity() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // skip - no need to load checkpoint
    }

	Tensor forward(Tensor &x, ...);
};

// sinusoidal positional embedding
class SinusoidalPositionalEmbedding : public Module {
private:
    int _dim;
    int _theta;
public:
    SinusoidalPositionalEmbedding(int dim, int theta);
    ~SinusoidalPositionalEmbedding() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // skip - no need to load checkpoint
    }
	
	Tensor forward(Tensor &x, ...);
};

class Conv2DLayer : public Module {
private:
    int _in_channels;
    int _out_channels;
    int _kernel_size;
    int _stride=1;
    int _padding=0;
    int _groups=1;
    bool _has_bias=true;

    Tensor _weight;
    Tensor _bias;
public:
    // our default constructor
    Conv2DLayer(int in_channels, int out_channels,
                int kernel_size, int stride, int padding, int groups, bool has_bias);
    ~Conv2DLayer() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // endpoint: load weights and bias
        _weight = Tensor::load_npy(path + prefix + ".weight.npy");
        _bias = Tensor::load_npy(path + prefix + ".bias.npy");
    }

	Tensor forward(Tensor &x, ...);
};

class LinearLayer : public Module {
private:
    int _in_features;
    int _out_features;
    bool _has_bias;

    Tensor _weight;
    Tensor _bias;
public:
    LinearLayer(int in_features, int out_features, bool bias);
    ~LinearLayer() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // endpoint: load weights and bias
        _weight = Tensor::load_npy(path + prefix + ".weight.npy");
        _bias = Tensor::load_npy(path + prefix + ".bias.npy");
    }

	Tensor forward(Tensor &x, ...);
};


// upsample
class UpSample : public Module {
private:
    int _dim;
    int _dim_out=-1;
    Module *_conv_net;
public:
    UpSample(int dim);
    UpSample(int dim, int dim_out);
    ~UpSample() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // future todo - hard coded !! - if we have a chance to change this, we should change
        _conv_net->load_checkpoint(path, prefix + ".net.1");
        // We were intended to Sequential for _upsample and _conv_net
        // but we changed _upsample to function
    }

	Tensor forward(Tensor &x, ...);
};

// downsample
class DownSample : public Module {
private:
    int _dim;
    int _dim_out;
    Module *_conv_net;
public:
    DownSample(int dim, int dim_out);
    ~DownSample() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        _conv_net->load_checkpoint(path, prefix + ".conv");
    }

	Tensor forward(Tensor &x, ...);
};

// layernorm
class LayerNorm : public Module {
private:
    int _num_channels;
    float _eps=1e-5;
    bool _affine=true;

    Tensor _weight;
    Tensor _bias;
public:
    LayerNorm(int num_channels);
    ~LayerNorm() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        // endpoint: load weights and bias
        _weight = Tensor::load_npy(path + prefix + ".weight.npy");
        _bias = Tensor::load_npy(path + prefix + ".bias.npy");
    }
    
	Tensor forward(Tensor &x, ...);
};

// class for ConvNextBlock
class ConvNextBlock : public Module {
private:
    int _in_channels;
    int _out_channels;
    int _mult=2;
    int _time_embedding_dim;
    bool _norm=true;
    int _groups=8;
    Sequential *_mlp;
    Module *_in_conv;
    Sequential *_block;
    Module *_residual_conv;
public:
    ConvNextBlock(int in_channels, int out_channels,
                int time_embedding_dim, int groups);
    ~ConvNextBlock() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        _mlp->load_checkpoint(path, prefix + ".mlp");
        _in_conv->load_checkpoint(path, prefix + ".in_conv");
        _block->load_checkpoint(path, prefix + ".block");
        if (_in_channels != _out_channels) {
            _residual_conv->load_checkpoint(path, prefix + ".residual_conv");
        } else {
            // skip (identity)
        }
    }

	Tensor forward(Tensor &x, ...);
};

// class for BlockAttention
class BlockAttention : public Module {
private:
    int _gate_in_channel;
    int _residual_in_channel;
    int _scale_factor;

    Module *_gate_conv;
    Module *_residual_conv;
    Module *_in_conv;

public:
    BlockAttention(int gate_in_channel, int residual_in_channel, int scale_factor);
    ~BlockAttention() {}

    void load_checkpoint(std::string path, std::string prefix="") {
        _gate_conv->load_checkpoint(path, prefix + ".gate_conv");
        _residual_conv->load_checkpoint(path, prefix + ".residual_conv");
        _in_conv->load_checkpoint(path, prefix + ".in_conv");
    }

	Tensor forward(Tensor &x, ...);
};

#endif