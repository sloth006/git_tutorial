#include <model.h>

// UNet structure with attention
AttentionUNet::AttentionUNet(int dim, int channels,
                            std::vector<int> dim_mults) :
                _dim(dim), _channels(channels),
                _dim_mults(dim_mults) {
    int input_channels = _channels;
    _init_conv = new Conv2DLayer(input_channels, _dim,
                                /* kernel_size= */7,
                                /* stride= */1,
                                /* padding=*/3,
                                /* groups=1*/1,
                                /* bias=*/true);
    // initialize dims
    _dims = std::vector<int>();
    _dims.push_back(_dim);
    for (int i = 0; i < dim_mults.size(); i++) {
        _dims.push_back(_dim * dim_mults[i]);
    }

    for (int i = 0; i < _dims.size()-1; i++) {
        _in_out.push_back(std::make_tuple(_dims[i], _dims[i+1]));
    }

    // initialize positional embedding
    _pos_emb = new SinusoidalPositionalEmbedding(_dim, _sinusoidal_pos_emb_theta);

    // time dim
    _time_dim = _dim * 4;

    // initialize time mlp
    _time_mlp = new Sequential();
    _time_mlp->add_module(_pos_emb);
    _time_mlp->add_module(new LinearLayer(_dim, _time_dim, true));
    _time_mlp->add_module(new GELU());
    _time_mlp->add_module(new LinearLayer(_time_dim, _time_dim, true));

    _num_resolutions = _in_out.size();

    // iterate over _in_out
    for (int i = 0; i < _in_out.size(); i++) {
        bool is_last = i >= (_num_resolutions - 1);

        
        int in_channels = std::get<0>(_in_out[i]);
        int out_channels = std::get<1>(_in_out[i]);
        
        // downsampling
        ModuleList *module_list = new ModuleList();
        module_list->add_module(
            new ConvNextBlock(in_channels, in_channels, _time_dim, _convnext_block_groups)
        );
        module_list->add_module(
            new ConvNextBlock(in_channels, in_channels, _time_dim, _convnext_block_groups)
        );

        if (is_last) {
            module_list->add_module(new Conv2DLayer(in_channels, out_channels,
                                                    /* kernel_size= */3,
                                                    /* stride= */1,
                                                    /* padding=*/1,
                                                    /* groups=*/1,
                                                    /* bias=*/true));
        } else {
            module_list->add_module(new DownSample(in_channels, out_channels));
        }
        _downs.push_back(module_list);
    }

    int intermediate_dim = _dims[_dims.size() - 1];
    _intermediate_block1 = new ConvNextBlock(intermediate_dim, intermediate_dim, _time_dim, 8);
    _intermediate_block2 = new ConvNextBlock(intermediate_dim, intermediate_dim, _time_dim, 8);

    // iterate over _in_out in reverse
    for (int i = _in_out.size() - 1; i >= 0; i--) {
        bool is_first = i == _in_out.size() - 1;
        bool is_last = i == 0;

        int in_channels = std::get<0>(_in_out[i]);
        int out_channels = std::get<1>(_in_out[i]);

        // upsampling
        ModuleList *module_list = new ModuleList();
        int __ba_scale_factor = 2;
        if (is_first) {
            __ba_scale_factor = 1;
        }
        module_list->add_module(
            new BlockAttention(out_channels, in_channels, __ba_scale_factor)
        );
        module_list->add_module(
            new ConvNextBlock(in_channels + out_channels,
                            out_channels, _time_dim, _convnext_block_groups)
        );
        module_list->add_module(
            new ConvNextBlock(in_channels + out_channels,
                            out_channels, _time_dim, _convnext_block_groups)
        );
        if (is_last) {
            module_list->add_module(new Conv2DLayer(out_channels, in_channels,
                                                    /* kernel_size = */3,
                                                    /* stride (default) = */1,
                                                    /* padding = */1,
                                                    /* groups (default) = */1,
                                                    /* bias (default) = */true));
        } else {
            module_list->add_module(new UpSample(out_channels, in_channels));
        }
        _ups.push_back(module_list);
    }  
    _out_dim = _channels;
    _final_res_block = new ConvNextBlock(_dim * 2, _dim, _time_dim, /* group (default) = */8);
    _final_conv = new Conv2DLayer(_dim, _out_dim,
                                /* kernel_size = */1,
                                /* stride (default) = */1,
                                /* padding (default) = */0,
                                /* groups (default) = */1,
                                /* bias (default) = */true);
}

Tensor AttentionUNet::forward(Tensor &x, ...) {
	std::va_list args;
	va_start(args, x);
	// get additional arguments (in this case... batched_timestamps)
	Tensor batched_timestamps = va_arg(args, Tensor);
	va_end(args);

	x = _init_conv->forward(x);

	Tensor res = Tensor::copy(x);
	Tensor time_emb = _time_mlp->forward(batched_timestamps);

	std::vector<Tensor> unet_stack;

	// unet downsample
	for (int i = 0; i < _downs.size(); i++) {
		ModuleList *down = _downs[i];
		Module *down1 = down->at(0);
		Tensor __downed1 = down1->forward(x, time_emb);
		unet_stack.push_back(__downed1);
		Module *down2 = down->at(1);
		Tensor __downed2 = down2->forward(__downed1, time_emb);
		unet_stack.push_back(__downed2);
		Module *downsample = down->at(2);
		x = downsample->forward(__downed2);
	}

	x = _intermediate_block1->forward(x, time_emb);
	x = _intermediate_block2->forward(x, time_emb);

	// unet upsample
	for (int i = 0; i < _ups.size(); i++) {
		ModuleList *up = _ups[i];
		Module *__attention = up->at(0);
		Module *up1 = up->at(1);
		Module *up2 = up->at(2);
		Module *upsample = up->at(3);

		Tensor __attention_out = __attention->forward(unet_stack.back(), x);
		unet_stack.pop_back();
		Tensor __up0_out = func::concatenate(x, __attention_out, 1);
		x = up1->forward(__up0_out, time_emb);
		__attention_out = __attention->forward(unet_stack.back(), x);
		unet_stack.pop_back();
		Tensor __up1_out = func::concatenate(x, __attention_out, 1);
		x = up2->forward(__up1_out, time_emb);
		x = upsample->forward(x);
	}

	Tensor __upsample_final = func::concatenate(x, res, 1);
	x = _final_res_block->forward(__upsample_final, time_emb);
	x = _final_conv->forward(x);

	return x;
}


// Diffusion model (kinda UNet wrapper)
DiffusionModel::DiffusionModel(AttentionUNet *model, int image_size,
                            std::string beta_scheduler, int timesteps,
                            std::string type, int ddim_n,
                            bool fix_seed, int seed) :
                            _model(model), _image_size(image_size),
                            _beta_scheduler(beta_scheduler), _timesteps(timesteps),
                            _type(type), _ddim_n(ddim_n),
                            _fix_seed(fix_seed), _seed(seed) {
    _channels = _model->channels();
    
    if (_beta_scheduler == "linear") {
        _betas = sch::linear_beta_schedule(_timesteps);
    } else {
        assert(false);
        // In our project, we only use linear beta scheduler
    }

    // Check the DDIM condition
    if (_type == "ddim") {
        assert(_timesteps % _ddim_n == 0); // should be dividable
    }

    // Check fixing seed condition
    if (_fix_seed) {
        assert(_type == "ddim" && "DDPM requires randomness while sampling... use DDIM for fixing seed.");
    }

    /*
    * Precompute some values for sampling
    */

    // Calculate alphas
    _alphas = Tensor::from_shape({_timesteps});
    for (int i = 0; i < _timesteps; i++) {
        _alphas[i] = 1.0f - _betas[i];
    }

    // alphas_cumprod using exp(cumsum(log(alphas)))
    Tensor log_alphas = _alphas.log();
    Tensor cumsum_log_alphas = func::cumsum(log_alphas);
    _alphas_cumprod = cumsum_log_alphas.exp();

    // alphas_cumprod_prev
    _alphas_cumprod_prev = Tensor::from_shape({_timesteps});
    _alphas_cumprod_prev[0] = 1.0f;
    for (int i = 1; i < _timesteps; i++) {
        _alphas_cumprod_prev[i] = _alphas_cumprod[i - 1];
    }

    // posterior_variance
    _posterior_variance = Tensor::from_shape({_timesteps});
    for (int i = 0; i < _timesteps; i++) {
        _posterior_variance[i] = _betas[i] * 
            (1.0f - _alphas_cumprod_prev[i]) / (1.0f - _alphas_cumprod[i]);
    }

    // sqrt_recip_alphas
    _sqrt_recip_alphas = Tensor::from_shape({_timesteps});
    for (int i = 0; i < _timesteps; i++) {
        _sqrt_recip_alphas[i] = std::sqrt(1.0f / _alphas[i]);
    }

    // sqrt_alphas_cumprod
    _sqrt_alphas_cumprod = Tensor::from_shape({_timesteps});
    for (int i = 0; i < _timesteps; i++) {
        _sqrt_alphas_cumprod[i] = std::sqrt(_alphas_cumprod[i]);
    }

    // one_minus_alphas_cumprod
    _one_minus_alphas_cumprod = Tensor::from_shape({_timesteps});
    for (int i = 0; i < _timesteps; i++) {
        _one_minus_alphas_cumprod[i] = 1.0f - _alphas_cumprod[i];
    }

    // sqrt_one_minus_alphas_cumprod
    _sqrt_one_minius_alphas_cumprod = Tensor::from_shape({_timesteps});
    for (int i = 0; i < _timesteps; i++) {
        _sqrt_one_minius_alphas_cumprod[i] = std::sqrt(1.0f - _alphas_cumprod[i]);
    }

    // Initialize normalize and unnormalize functions
    if (_auto_normalize) {
        _normalize = func::normalize_to_neg_one_to_one;
        _unnormalize = func::unnormalize_to_zero_to_one;
    } else {
        _normalize = [](Tensor& x) { return x; };
        _unnormalize = [](Tensor& x) { return x; };
    }

    if (_type == "ddpm") {
        for (int i = 0; i < _timesteps; i++) {
            _timestep_vec.push_back(i);
            _timestep_vec_rev.push_back(_timesteps - i - 1);
        }
    } else if (_type == "ddim") {
        for (int i = 0; i < _timesteps; i++) {
            if (i % _ddim_n == 0) {
                _timestep_vec.push_back(i);
            }
        }
        for (int i = _timesteps-1; i >= 0; i--) {
            if (i % _ddim_n == 0) {
                _timestep_vec_rev.push_back(i);
            }
        }
    }
}

Tensor DiffusionModel::__p_sample(Tensor &x, int timestep) {
    int num_batches = x.shape()[0];

    Tensor input_x = Tensor::copy(x); // copy operation

    Tensor batched_timestamps = Tensor::from_shape({1, num_batches});
    for (int i = 0; i < num_batches; i++) {
        batched_timestamps[i] = static_cast<float>(timestep);
    }

    Tensor preds = _model->forward(x, batched_timestamps);

    // Extract values
    Tensor betas_t = func::extract(_betas, batched_timestamps, input_x.dim());
    Tensor sqrt_recip_alphas_t = func::extract(_sqrt_recip_alphas, batched_timestamps, input_x.dim());
    Tensor sqrt_one_minus_alphas_cumprod_t = func::extract(_sqrt_one_minius_alphas_cumprod, batched_timestamps, input_x.dim());

    // Calculate: sqrt_recip_alphas_t * (input_x - betas_t * preds / sqrt_one_minus_alphas_cumprod_t)
    Tensor preds_scaled = func::divide(preds, sqrt_one_minus_alphas_cumprod_t);
    Tensor betas_times_preds = func::multiply(betas_t, preds_scaled);
    Tensor diff = func::subtract(input_x, betas_times_preds);
    Tensor predicted_mean = func::multiply(sqrt_recip_alphas_t, diff);

    if (timestep == 0) {
        return predicted_mean;
    } else {
        Tensor posterior_variance = func::extract(_posterior_variance, batched_timestamps, input_x.dim());
        Tensor noise = Tensor::randn(input_x.shape());
        Tensor sqrt_variance = posterior_variance.sqrt();
        Tensor noise_scaled = func::multiply(sqrt_variance, noise);
        return func::add(predicted_mean, noise_scaled);
    }
}

Tensor DiffusionModel::__p_sample_ddim(Tensor &x, int timestep) {
    if (timestep == 0) {
        return x;
    }
    int num_batches = x.shape()[0];

    int prev_timestep = timestep - _ddim_n;

    Tensor input_x = Tensor::copy(x); // copy operation

    Tensor batched_timestamps({1, num_batches}, static_cast<float>(timestep));
    Tensor batched_timestamps_prev({1, num_batches}, static_cast<float>(prev_timestep));

    Tensor preds = _model->forward(x, batched_timestamps);

    // Extract values
    Tensor sqrt_alphas_cumprod_t = func::extract(_sqrt_alphas_cumprod, batched_timestamps, input_x.dim());
    Tensor sqrt_alphas_cumprod_prev_t = func::extract(_sqrt_alphas_cumprod, batched_timestamps_prev, input_x.dim());
    Tensor sqrt_one_minus_alphas_cumprod_t = func::extract(_sqrt_one_minius_alphas_cumprod, batched_timestamps, input_x.dim());
    Tensor sqrt_one_minus_alphas_cumprod_prev_t = func::extract(_sqrt_one_minius_alphas_cumprod, batched_timestamps_prev, input_x.dim());

    // eq 1: prediction
    // predicted_mean = sqrt_alphas_cumprod_prev_t / sqrt_alphas_cumprod_t * (input_x - sqrt_one_minus_alphas_cumprod_t * preds)
    Tensor ratio = func::divide(sqrt_alphas_cumprod_prev_t, sqrt_alphas_cumprod_t);
    Tensor preds_scaled = func::multiply(sqrt_one_minus_alphas_cumprod_t, preds);
    Tensor diff = func::subtract(input_x, preds_scaled);
    Tensor predicted_mean = func::multiply(ratio, diff);

    // eq 2: direction noise
    Tensor direction_noise = func::multiply(sqrt_one_minus_alphas_cumprod_prev_t, preds);

    return func::add(predicted_mean, direction_noise);
}

Tensor DiffusionModel::__p_sample_loop(std::vector<int> shape) {
    int batch_size = shape[0];

    // random generation of images
    if (_fix_seed) {
        Tensor::set_random_seed(_seed);
    } else {
        Tensor::set_random_seed(time(NULL));
    }
    Tensor img = Tensor::randn(shape);

    std::cout << "Exporting random image to npy... (results/initial_images_bchw.npy)" << std::endl;
    img.dump_npy("results/initial_images_bchw.npy");

    // iterate over reverse_timestep_vec (backward process of diffusion)
    for (int t : tq::tqdm(_timestep_vec_rev)) {
        if (_type == "ddpm") {
            img = __p_sample(img, t);
        } else if (_type == "ddim") {
            img = __p_sample_ddim(img, t);
        }
    }

    Tensor final_img = _unnormalize(img);

    std::cout << std::endl << "Exporting final image... (results/sampled_images_bchw.npy)" << std::endl;

    return final_img;
}
