#include <argparse.hpp>
#include <model.h>

#include <ctime>
#include <cuda.h>
#include <cuda_runtime_api.h>

void print_duration(const char* operation, const timespec& start, const timespec& end) {
    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("%s Time: %f seconds\n", operation, duration);
}


int main(int argc, char *argv[]) {

  /* === Argument Parsing === */

  argparse::ArgumentParser general_args("Embedded Systems Design Spring 2026 (Accelerated Intelligent Systems Lab.)");

  // how many samples to generate (we currently support single batch)
  general_args.add_argument("-n", "--n-samples")
    .default_value(1)
    .required()
    .help("the number of samples to generate")
    .scan<'i', int>();

  // fixing seed
  general_args.add_argument("-s", "--seed")
    .default_value(7524)
    .help("seed")
    .scan<'i', int>();
  general_args.add_argument("-fs", "--fix-seed")
    .help("fixing seed")
    .default_value(false)
    .implicit_value(true);

  // path for checkpoint
  general_args.add_argument("-p", "--path")
  .default_value(std::string("./test/test_model_ckpt/"))
  .help("checkpoint path");

  // ddpm or ddim
  general_args.add_argument("-t", "--type")
    .default_value(std::string("ddim"))
    .help("sampling type");

  // ddim_n
  general_args.add_argument("-dn", "--ddim-n")
    .default_value(10)
    .help("ddim-n")
    .scan<'i', int>();

  // verbosity
  general_args.add_argument("-v", "--verbose")
    .help("verbose output")
    .default_value(false)
    .implicit_value(true);

  try {
    general_args.parse_args(argc, argv);
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << general_args;
    return 1;
  }

  /* ========================= */

  std::cout << "==================== Start Diffusion Sampling (ESD2026Spring) ====================" << std::endl;

  if (general_args["--verbose"] == true) {
    std::cout << "Generate " << general_args.get<int>("-n") << " samples... (Our lab only supports single batch)" << std::endl;
    std::cout << "Ckpt Path: " << general_args.get<std::string>("-p") << std::endl;
  }
  assert(general_args.get<int>("-n") == 1);

  if (general_args["--fix-seed"] == true) {
    assert(general_args.get<std::string>("-t") == "ddim" && "DDPM requires randomness while sampling... use DDIM for fixing seed.");
  }

  if (general_args["--verbose"] == true) {
    std::cout << "Constructing Model... (64x64 SVHN Diffusion)" << std::endl;
    std::cout << ">>> Model Type: " << general_args.get<std::string>("-t") << std::endl;
    if (general_args.get<std::string>("-t") == "ddim") {
      std::cout << ">>>>>> DDIM with " << general_args.get<int>("-dn") << "x acceleration (DDIM n)" << std::endl;
    }
  }

  timespec start, end;
  double duration = 0.0f;
  clock_gettime(CLOCK_MONOTONIC, &start);

  AttentionUNet model(32, 3, {1, 2, 4, 8});
  
  DiffusionModel diffusion_model(&model, 64, "linear", 1000,
                                general_args.get<std::string>("-t"), general_args.get<int>("-dn"),
                                general_args.get<bool>("--fix-seed"), general_args.get<int>("-s"));

  diffusion_model.load_checkpoint(general_args.get<std::string>("-p"));

  Tensor output = diffusion_model.sample(general_args.get<int>("-n"));

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &end);
  duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Total Execution Time: %f seconds\n", duration);

  output.dump_npy("results/sampled_images_bchw.npy");
  std::cout << "Done... please run \'/nfs/home/proj1_env/bin/python3 img_convert.py\' in the utils directory." << std::endl;
  std::cout << "================================================================================" << std::endl;

  return 0;
}