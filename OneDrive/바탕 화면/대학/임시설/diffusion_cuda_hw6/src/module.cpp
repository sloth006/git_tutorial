// model.h implementation
#include <module.h>

// Tiny module implementations
Tensor GELU::forward(Tensor &x, ...) {
	return func::gelu(x);
}

Tensor ReLU::forward(Tensor &x, ...) {
	return func::relu(x);
}

Tensor Sigmoid::forward(Tensor &x, ...) {
	return func::sigmoid(x);
}

Tensor Identity::forward(Tensor &x, ...) {
	return x;
}


SinusoidalPositionalEmbedding::SinusoidalPositionalEmbedding(int dim, int theta) :
    _dim(dim), _theta(theta) {
}


Tensor SinusoidalPositionalEmbedding::forward(Tensor &x, ...) {
	int half_dim = _dim / 2;
	double emb = std::log(_theta) / (half_dim-1);
	Tensor _emb_vec = Tensor::from_shape({half_dim});
	float x_elem = x[0];
	for(int i=0; i<half_dim; i++) {
		_emb_vec[i] = std::exp(i * -emb) * x_elem;
	}

	Tensor out = Tensor::from_shape({1, _dim});
	for(int i=0; i<half_dim; i++) {
		out[i] = std::sin(_emb_vec[i]);
		out[i + half_dim] = std::cos(_emb_vec[i]);
	}
	return out;
}

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels,
            int kernel_size, int stride, int padding, int groups, bool has_bias) :
            _in_channels(in_channels), _out_channels(out_channels),
            _kernel_size(kernel_size), _stride(stride),
            _padding(padding), _groups(groups), _has_bias(has_bias) {
    this->_weight = Tensor::from_shape({out_channels, in_channels, kernel_size, kernel_size});
    if (has_bias) {
        this->_bias = Tensor::from_shape({out_channels});
    }
}

Tensor Conv2DLayer::forward(Tensor &x, ...) {
	return func::conv2d(x, _weight, _bias, _stride, _padding, _groups);
}


LinearLayer::LinearLayer(int in_features, int out_features, bool bias=true) :
    _in_features(in_features), _out_features(out_features), _has_bias(bias) {
    this->_weight = Tensor::from_shape({out_features, in_features});
    if (bias) {
        this->_bias = Tensor::from_shape({out_features});
    }
}

Tensor LinearLayer::forward(Tensor &x, ...) {
	return func::linear(x, _weight, _bias, _has_bias);
}


UpSample::UpSample(int dim) : _dim(dim) {
    _conv_net = new Conv2DLayer(_dim, _dim, 
                                /* kernel_size = */3,
                                /* stride = */1,
                                /* padding = */1,
                                /* groups (default) = */1,
                                /* bias (default) = */true);
}

UpSample::UpSample(int dim, int dim_out) : _dim(dim), _dim_out(dim_out) {
    _conv_net = new Conv2DLayer(_dim, _dim_out, 
                                /* kernel_size = */3,
                                /* stride = */1,
                                /* padding = */1,
                                /* groups (default) = */1,
                                /* bias (default) = */true);
}

Tensor UpSample::forward(Tensor &x, ...) {
	Tensor upsampled = func::upsample(x, /* scale_factor = */2);
	return _conv_net->forward(upsampled);
}


DownSample::DownSample(int dim, int dim_out) : _dim(dim), _dim_out(dim_out) {
    _conv_net = new Conv2DLayer(_dim, _dim_out, 
                                /* kernel_size = */3,
                                /* stride = */2,
                                /* padding = */1,
                                /* groups (default) = */1,
                                /* bias (default) = */true);
}

Tensor DownSample::forward(Tensor &x, ...) {
	return _conv_net->forward(x);
}


LayerNorm::LayerNorm(int num_channels) : _num_channels(num_channels) {
    if (_affine) {
        _weight = Tensor::from_shape({num_channels});
        _bias = Tensor::from_shape({num_channels});
    } else {
        assert(false);
        // only assume affine for our case
    }
}

Tensor LayerNorm::forward(Tensor &x, ...) {
	return func::layer_norm(x, _weight, _bias, _eps);
}


ConvNextBlock::ConvNextBlock(int in_channels, int out_channels, int time_embedding_dim, int groups) :
    _in_channels(in_channels), _out_channels(out_channels),
    _time_embedding_dim(time_embedding_dim), _groups(groups) {

    _mlp = new Sequential();
    _mlp->add_module(new GELU());
    _mlp->add_module(new LinearLayer(_time_embedding_dim, _in_channels, true));
    
    _in_conv = new Conv2DLayer(_in_channels, _in_channels,
                                /* kernel_size = */7,
                                /* stride (default) = */1,
                                /* padding = */3,
                                /* groups = */_in_channels,
                                /* bias (default) = */true);

    _block = new Sequential();
    if (_norm) {
        _block->add_module(new LayerNorm(/*num_channels=*/_in_channels));
    } else {
        _block->add_module(new Identity());
    }
    _block->add_module(new Conv2DLayer(_in_channels, _out_channels * _mult,
                                        /* kernel_size = */3,
                                        /* stride (default) = */1,
                                        /* padding = */1,
                                        /* groups (default) = */1,
                                        /* bias (default) = */true));
    _block->add_module(new GELU());
    //_block->add_module(new ReLU());
    _block->add_module(new LayerNorm(/*num_channels=*/_out_channels * _mult));
    _block->add_module(new Conv2DLayer(_out_channels * _mult, _out_channels,
                                        /* kernel_size = */3,
                                        /* stride (default) = */1,
                                        /* padding = */1,
                                        /* groups (default) = */1,
                                        /* bias (default) = */true));

    if (_in_channels != _out_channels) {
        _residual_conv = new Conv2DLayer(_in_channels, _out_channels,
                                        /* kernel_size = */1,
                                        /* stride (default) = */1,
                                        /* padding (default) = */0,
                                        /* groups (default) = */1,
                                        /* bias (default) = */true);
    } else {
        _residual_conv = new Identity();
    }
}

Tensor ConvNextBlock::forward(Tensor &x, ...) {
	std::va_list args;
	va_start(args, x);
	Tensor _time_embedding = va_arg(args, Tensor);
	va_end(args);

	Tensor h = _in_conv->forward(x);
	Tensor time_emb = _mlp->forward(_time_embedding);
	Tensor h_with_time = func::add(time_emb, h);
	Tensor h_processed = _block->forward(h_with_time);
	Tensor residual = _residual_conv->forward(x);
	
	return func::add(h_processed, residual);
}


BlockAttention::BlockAttention(int gate_in_channel, int residual_in_channel, int _scale_factor) :
    _gate_in_channel(gate_in_channel), _residual_in_channel(residual_in_channel), _scale_factor(_scale_factor) {
    _gate_conv = new Conv2DLayer(gate_in_channel, gate_in_channel, 
                                /* kernel_size = */1,
                                /* stride = */1,
                                /* padding (default) = */0,
                                /* groups (default) = */1,
                                /* bias (default) = */true);
    _residual_conv = new Conv2DLayer(residual_in_channel, gate_in_channel, 
                                /* kernel_size = */1,
                                /* stride = */1,
                                /* padding (default) = */0,
                                /* groups (default) = */1,
                                /* bias (default) = */true);
    _in_conv = new Conv2DLayer(gate_in_channel, 1, 
                                /* kernel_size = */1,
                                /* stride = */1,
                                /* padding (default) = */0,
                                /* groups (default) = */1,
                                /* bias (default) = */true);
}

Tensor BlockAttention::forward(Tensor &x, ...) {
	// gate is always provided
	std::va_list args;
	va_start(args, x);
	Tensor _gate = va_arg(args, Tensor);
	va_end(args);
	
	auto gate_conv = _gate_conv->forward(_gate);
	auto residual_conv = _residual_conv->forward(x);
	auto in_attention = func::add(gate_conv, residual_conv);
	in_attention = func::relu(in_attention);
	in_attention = _in_conv->forward(in_attention);
	in_attention = func::sigmoid(in_attention);
	return func::multiply(in_attention, x);
}